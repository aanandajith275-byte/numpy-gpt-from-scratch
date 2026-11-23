"""Evaluation in bits per character (BPC): the average number of bits the model needs to encode each
character of unseen text. Lower is better; 0 would be perfect prediction, log2(vocab) ~ 5.6 is a
uniform guess. Cross-entropy in nats / ln(2) = bits.
"""
from __future__ import annotations

import numpy as np

from .data import sliding_windows
from .model import softmax


def per_char_logprobs(model, data: np.ndarray, block_size: int | None = None, stride: int | None = None,
                      batch: int = 64, max_chars: int | None = None):
    """log2 P(char_i | previous chars) for the characters of ``data``.

    Returns (log2 probs, position of each prediction inside its context window, index in ``data`` of the
    predicted character)."""
    T = block_size or model.cfg.block_size
    stride = stride or T // 2
    if max_chars is not None:
        data = data[: max_chars + 1]
    xs, ys, firsts, starts = [], [], [], []
    for x, y, f, s in sliding_windows(data, T, stride):
        xs.append(x); ys.append(y); firsts.append(f); starts.append(s)
    out_lp, out_pos, out_abs = [], [], []
    for i in range(0, len(xs), batch):
        X, Y = np.stack(xs[i:i + batch]), np.stack(ys[i:i + batch])
        logits, _, _ = model.forward(X)
        p = softmax(logits)
        lp = np.log2(np.take_along_axis(p, Y[..., None], axis=-1)[..., 0] + 1e-12)
        for j, (f, s) in enumerate(zip(firsts[i:i + batch], starts[i:i + batch])):
            out_lp.append(lp[j, f:]); out_pos.append(np.arange(f, T)); out_abs.append(s + 1 + np.arange(f, T))
    return np.concatenate(out_lp), np.concatenate(out_pos), np.concatenate(out_abs)


def bpc(model, data, **kw) -> float:
    lp, _, _ = per_char_logprobs(model, data, **kw)
    return float(-lp.mean())
