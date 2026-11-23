"""Train a character-level GPT on the aerospace abstracts.

    python experiments/train.py --name main --n_embd 128 --n_layer 4 --steps 3000
    python experiments/train.py --name tiny --n_embd 32 --n_layer 2 --steps 3000

Writes checkpoints/<name>.npz (weights at the best validation loss) and results/train_<name>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from tinygpt.data import CharData  # noqa: E402
from tinygpt.evaluate import bpc  # noqa: E402
from tinygpt.model import GPT, AdamW, GPTConfig, clip_grad_norm  # noqa: E402

LN2 = np.log(2.0)


def lr_at(step, total, peak, warmup=100, floor_frac=0.05):
    """Linear warm-up, then cosine decay to 5% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + np.cos(np.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = CharData()
    cfg = GPTConfig(vocab_size=data.vocab_size, block_size=a.block_size, n_layer=a.n_layer,
                    n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)
    model = GPT(cfg, seed=a.seed)
    opt = AdamW(model.params, lr=a.lr, weight_decay=a.weight_decay)
    rng = np.random.default_rng(a.seed)
    print(f"[{a.name}] params {model.num_params():,}  train chars {len(data.train):,}  val chars {len(data.val):,}")

    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    ckpt = os.path.join(ROOT, "checkpoints", f"{a.name}.npz")
    log = {"args": vars(a), "params": model.num_params(), "params_no_emb": model.num_params(False),
           "steps": [], "train_bpc": [], "val_bpc": [], "train_eval_bpc": [], "lr": [], "grad_norm": []}
    best, t0 = np.inf, time.time()
    val_quick = data.val[:40_001]          # fast estimate during training
    train_quick = data.train[:40_001]      # same size, to measure the generalisation gap
    for step in range(a.steps):
        lr = lr_at(step, a.steps, a.lr)
        x, y = data.batch(rng, a.batch_size, a.block_size)
        _, loss, cache = model.forward(x, y, train=True, rng=rng)
        grads = model.backward(cache)
        gn = clip_grad_norm(grads, 1.0)
        opt.step(model.params, grads, lr)
        log["train_bpc"].append(loss / LN2)
        log["lr"].append(lr)
        log["grad_norm"].append(gn)
        if step % a.eval_every == 0 or step == a.steps - 1:
            v = bpc(model, val_quick)
            tr = bpc(model, train_quick)
            log["steps"].append(step); log["val_bpc"].append(v); log["train_eval_bpc"].append(tr)
            if v < best:
                best = v
                model.save(ckpt, {"step": step, "val_bpc_quick": v, "vocab": "".join(data.chars)})
            print(f"[{a.name}] step {step:5d}  train {np.mean(log['train_bpc'][-50:]):.3f}  "
                  f"train(eval) {tr:.3f}  val {v:.3f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)
    best_model, meta = GPT.load(ckpt)
    log["best_step"] = int(meta["step"])
    log["final_val_bpc_full"] = bpc(best_model, data.val)
    log["final_train_bpc_sample"] = bpc(best_model, data.train[:200_001])
    log["train_seconds"] = time.time() - t0
    json.dump(log, open(os.path.join(ROOT, "results", f"train_{a.name}.json"), "w"))
    print(f"[{a.name}] best step {log['best_step']}  full val BPC {log['final_val_bpc_full']:.4f}  "
          f"({log['train_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
