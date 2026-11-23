"""Numerical gradient check: is the hand-written backward pass correct?

For a sample of entries in every parameter tensor we compare the analytic
gradient with the central finite difference

    (L(theta + eps) - L(theta - eps)) / (2 eps)

in float64 on a tiny model. Relative errors around 1e-7 or below mean the
backward pass is right; a bug typically shows up as 1e-2 or worse.

    python tests/test_gradients.py        (also runs under pytest)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt.model import GPT, GPTConfig  # noqa: E402


def check(dropout=0.0, n_entries=6, eps=1e-5, seed=0):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=dropout)
    model = GPT(cfg, seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for k in model.params:  # non-trivial LayerNorm gains / biases so their gradients are exercised
        model.params[k] += rng.standard_normal(model.params[k].shape) * 0.05
    idx = rng.integers(0, cfg.vocab_size, size=(3, cfg.block_size))
    tgt = rng.integers(0, cfg.vocab_size, size=(3, cfg.block_size))

    def loss_fn():  # identical dropout masks on every call
        return model.forward(idx, tgt, train=dropout > 0, rng=np.random.default_rng(123))[1]

    _, _, cache = model.forward(idx, tgt, train=dropout > 0, rng=np.random.default_rng(123))
    grads = model.backward(cache)
    worst = {}
    for name, param in model.params.items():
        flat = param.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_entries, flat.size), replace=False)
        errs = []
        for i in picks:
            old = flat[i]
            flat[i] = old + eps; lp = loss_fn()
            flat[i] = old - eps; lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        worst[name] = max(errs)
    return worst


def test_gradients():
    for dropout in (0.0, 0.2):
        worst = check(dropout)
        assert max(worst.values()) < 1e-5, worst


if __name__ == "__main__":
    out = {}
    for dropout in (0.0, 0.2):
        worst = check(dropout)
        out[f"dropout={dropout}"] = worst
        print(f"dropout={dropout}: max relative error {max(worst.values()):.2e} over {len(worst)} tensors")
        for k, v in worst.items():
            print(f"   {k:20s} {v:.2e}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "gradient_check.json"), "w"), indent=2)
