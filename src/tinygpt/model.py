"""A GPT (decoder-only transformer) in pure NumPy, with a hand-written backward pass.

Architecture (same block structure as GPT-2):

    tokens -> token embedding + position embedding
           -> L x [ x + Attention(LayerNorm(x)),  x + MLP(LayerNorm(x)) ]
           -> LayerNorm -> linear head -> logits over the vocabulary

Every forward operation stores what its backward pass needs in a ``cache``;
``backward`` walks the layers in reverse and applies the chain rule by hand.
``tests/test_gradients.py`` checks every gradient against finite differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

GELU_C = float(np.sqrt(2.0 / np.pi))  # a Python float, so float32 arrays stay float32


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # maximum context length T
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128          # model width d
    dropout: float = 0.0


# ----------------------------------------------------------------------------- building blocks
def layernorm_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, dy.shape[-1]).sum(0)
    dxhat = dy * g
    # d/dx of (x - mean) / std, folded into one expression
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu_forward(x):
    t = np.tanh(GELU_C * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (1.0 + t), t


def gelu_backward(dy, x, t):
    dt = (1.0 - t * t) * GELU_C * (1.0 + 3 * 0.044715 * x * x)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * dt)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------- the model
class GPT:
    def __init__(self, cfg: GPTConfig, seed: int = 0, dtype=np.float32):
        self.cfg, self.dtype = cfg, dtype
        rng = np.random.default_rng(seed)
        d, V, T, L = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
        n = lambda *shape, std=0.02: (rng.standard_normal(shape) * std).astype(dtype)  # noqa: E731
        z = lambda *shape: np.zeros(shape, dtype)  # noqa: E731
        o = lambda *shape: np.ones(shape, dtype)  # noqa: E731
        res_std = 0.02 / np.sqrt(2 * L)  # GPT-2: scale residual projections by 1/sqrt(#residual adds)
        p = {"wte": n(V, d), "wpe": n(T, d)}
        for l in range(L):
            p.update({f"h{l}.ln1.g": o(d), f"h{l}.ln1.b": z(d),
                      f"h{l}.attn.w_qkv": n(d, 3 * d), f"h{l}.attn.b_qkv": z(3 * d),
                      f"h{l}.attn.w_o": n(d, d, std=res_std), f"h{l}.attn.b_o": z(d),
                      f"h{l}.ln2.g": o(d), f"h{l}.ln2.b": z(d),
                      f"h{l}.mlp.w_fc": n(d, 4 * d), f"h{l}.mlp.b_fc": z(4 * d),
                      f"h{l}.mlp.w_proj": n(4 * d, d, std=res_std), f"h{l}.mlp.b_proj": z(d)})
        p.update({"lnf.g": o(d), "lnf.b": z(d), "w_head": n(d, V)})
        self.params = p
        self.mask = np.triu(np.full((T, T), -1e9, dtype=dtype), k=1)  # causal: no peeking ahead

    def num_params(self, include_embeddings: bool = True) -> int:
        return sum(v.size for k, v in self.params.items() if include_embeddings or k not in ("wte", "wpe"))

    # ------------------------------------------------------------------ forward
    def forward(self, idx, targets=None, train=False, rng=None, keep_attention=False):
        """idx: (B, T) int array. Returns logits (B, T, V), mean loss (nats) and a cache."""
        cfg, p = self.cfg, self.params
        B, T = idx.shape
        H, d = cfg.n_head, cfg.n_embd
        dh = d // H
        scale = float(1.0 / np.sqrt(dh))
        drop = cfg.dropout if train else 0.0

        x = p["wte"][idx] + p["wpe"][:T]
        cache = {"idx": idx, "layers": []}
        attn_maps = []
        for l in range(cfg.n_layer):
            c = {}
            h, c["ln1"] = layernorm_forward(x, p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"])
            qkv = h @ p[f"h{l}.attn.w_qkv"] + p[f"h{l}.attn.b_qkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q, k, v = (a.reshape(B, T, H, dh).transpose(0, 2, 1, 3) for a in (q, k, v))  # (B, H, T, dh)
            att = (q @ k.transpose(0, 1, 3, 2)) * scale + self.mask[:T, :T]
            P = softmax(att)                                         # attention weights
            y = (P @ v).transpose(0, 2, 1, 3).reshape(B, T, d)       # weighted sum of values, heads merged
            a = y @ p[f"h{l}.attn.w_o"] + p[f"h{l}.attn.b_o"]
            m1 = (rng.random(a.shape, dtype=np.float32) >= drop).astype(self.dtype) / self.dtype(1 - drop) if drop > 0 else None
            if m1 is not None:
                a = a * m1
            x = x + a
            h2, c["ln2"] = layernorm_forward(x, p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"])
            f = h2 @ p[f"h{l}.mlp.w_fc"] + p[f"h{l}.mlp.b_fc"]
            g, t = gelu_forward(f)
            m = g @ p[f"h{l}.mlp.w_proj"] + p[f"h{l}.mlp.b_proj"]
            m2 = (rng.random(m.shape, dtype=np.float32) >= drop).astype(self.dtype) / self.dtype(1 - drop) if drop > 0 else None
            if m2 is not None:
                m = m * m2
            x = x + m
            c.update(h=h, q=q, k=k, v=v, P=P, y=y, m1=m1, h2=h2, f=f, g=g, t=t, m2=m2)
            cache["layers"].append(c)
            if keep_attention:
                attn_maps.append(P)
        xf, cache["lnf"] = layernorm_forward(x, p["lnf.g"], p["lnf.b"])
        logits = xf @ p["w_head"]
        cache["xf"] = xf
        if keep_attention:
            cache["attention"] = attn_maps
        loss = None
        if targets is not None:
            probs = softmax(logits)
            cache["probs"], cache["targets"] = probs, targets
            lp = np.log(probs.reshape(-1, probs.shape[-1])[np.arange(B * T), targets.reshape(-1)] + 1e-12)
            loss = float(-lp.mean())
        return logits, loss, cache

    # ------------------------------------------------------------------ backward
    def backward(self, cache) -> dict:
        cfg, p = self.cfg, self.params
        idx, targets, probs = cache["idx"], cache["targets"], cache["probs"]
        B, T = idx.shape
        H, d, V = cfg.n_head, cfg.n_embd, cfg.vocab_size
        dh = d // H
        scale = float(1.0 / np.sqrt(dh))
        grads = {}

        # cross-entropy + softmax: dL/dlogits = (probs - onehot) / N
        dlogits = probs.reshape(-1, V).copy()
        dlogits[np.arange(B * T), targets.reshape(-1)] -= 1.0
        dlogits /= B * T
        grads["w_head"] = cache["xf"].reshape(-1, d).T @ dlogits
        dxf = (dlogits @ p["w_head"].T).reshape(B, T, d)
        dx, grads["lnf.g"], grads["lnf.b"] = layernorm_backward(dxf, cache["lnf"])

        for l in reversed(range(cfg.n_layer)):
            c = cache["layers"][l]
            # ---- MLP branch:  x_out = x_mid + dropout(W_proj gelu(W_fc LN(x_mid)))
            dm = dx * c["m2"] if c["m2"] is not None else dx
            grads[f"h{l}.mlp.w_proj"] = c["g"].reshape(-1, 4 * d).T @ dm.reshape(-1, d)
            grads[f"h{l}.mlp.b_proj"] = dm.reshape(-1, d).sum(0)
            dg = dm @ p[f"h{l}.mlp.w_proj"].T
            df = gelu_backward(dg, c["f"], c["t"])
            grads[f"h{l}.mlp.w_fc"] = c["h2"].reshape(-1, d).T @ df.reshape(-1, 4 * d)
            grads[f"h{l}.mlp.b_fc"] = df.reshape(-1, 4 * d).sum(0)
            dh2 = df @ p[f"h{l}.mlp.w_fc"].T
            dln2, grads[f"h{l}.ln2.g"], grads[f"h{l}.ln2.b"] = layernorm_backward(dh2, c["ln2"])
            dx = dx + dln2
            # ---- attention branch:  x_mid = x_in + dropout(W_o Attn(LN(x_in)))
            da = dx * c["m1"] if c["m1"] is not None else dx
            grads[f"h{l}.attn.w_o"] = c["y"].reshape(-1, d).T @ da.reshape(-1, d)
            grads[f"h{l}.attn.b_o"] = da.reshape(-1, d).sum(0)
            dy = (da @ p[f"h{l}.attn.w_o"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            P, q, k, v = c["P"], c["q"], c["k"], c["v"]
            dP = dy @ v.transpose(0, 1, 3, 2)
            dv = P.transpose(0, 1, 3, 2) @ dy
            # softmax backward: dS = P * (dP - sum(dP * P))
            datt = P * (dP - (dP * P).sum(-1, keepdims=True)) * scale
            dq = datt @ k
            dk = datt.transpose(0, 1, 3, 2) @ q
            dqkv = np.concatenate([a.transpose(0, 2, 1, 3).reshape(B, T, d) for a in (dq, dk, dv)], axis=-1)
            grads[f"h{l}.attn.w_qkv"] = c["h"].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
            grads[f"h{l}.attn.b_qkv"] = dqkv.reshape(-1, 3 * d).sum(0)
            dh_ = dqkv @ p[f"h{l}.attn.w_qkv"].T
            dln1, grads[f"h{l}.ln1.g"], grads[f"h{l}.ln1.b"] = layernorm_backward(dh_, c["ln1"])
            dx = dx + dln1

        grads["wpe"] = np.zeros_like(p["wpe"])
        grads["wpe"][:T] = dx.sum(0)
        grads["wte"] = np.zeros_like(p["wte"])
        np.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, d))
        return grads

    # ------------------------------------------------------------------ inference helpers
    def next_token_probs(self, idx):
        """Probabilities of the next token after each sequence in idx (B, t), using the last block_size tokens."""
        idx = idx[:, -self.cfg.block_size:]
        logits, _, _ = self.forward(idx)
        return softmax(logits[:, -1, :])

    def generate(self, idx, n_new, temperature=1.0, rng=None, top_k=None):
        rng = rng or np.random.default_rng(0)
        for _ in range(n_new):
            logits, _, _ = self.forward(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -1e9, logits)
            probs = softmax(logits)
            nxt = np.array([rng.choice(len(pr), p=pr / pr.sum()) for pr in probs])
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
        return idx

    # ------------------------------------------------------------------ persistence
    def save(self, path, extra: dict | None = None):
        meta = {"config": asdict(self.cfg), **(extra or {})}
        np.savez_compressed(path, __meta__=np.array(repr(meta)), **self.params)

    @classmethod
    def load(cls, path):
        import ast
        z = np.load(path, allow_pickle=False)
        meta = ast.literal_eval(str(z["__meta__"]))
        model = cls(GPTConfig(**meta["config"]))
        for k in model.params:
            model.params[k] = z[k]
        return model, meta


# ----------------------------------------------------------------------------- optimiser
class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter), applied to matrices only."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            if params[k].ndim == 2 and k not in ("wpe",):
                params[k] -= lr * self.wd * params[k]
            params[k] -= (lr * mhat / (np.sqrt(vhat) + self.eps)).astype(params[k].dtype)


def clip_grad_norm(grads, max_norm=1.0):
    total = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
    if total > max_norm:
        for k in grads:
            grads[k] *= max_norm / (total + 1e-6)
    return total
