"""Character-level dataset: 1,398 aerospace abstracts, split by document into train / validation."""
from __future__ import annotations

import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "abstracts.txt")


class CharData:
    def __init__(self, path: str = DATA, val_fraction: float = 0.1, seed: int = 0):
        docs = [line for line in open(path, encoding="utf-8").read().lower().split("\n") if line]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(docs))
        n_val = int(round(val_fraction * len(docs)))
        # split by *document*: validation abstracts are never seen in training
        self.val_docs = [docs[i] for i in order[:n_val]]
        self.train_docs = [docs[i] for i in order[n_val:]]
        self.chars = sorted(set("".join(docs) + "\n"))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = np.array(self.chars)
        self.train = self.encode("\n".join(self.train_docs) + "\n")
        self.val = self.encode("\n".join(self.val_docs) + "\n")

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[np.asarray(ids, dtype=np.int64)])

    def batch(self, rng, batch_size: int, block_size: int, split: str = "train"):
        data = self.train if split == "train" else self.val
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[s:s + block_size] for s in starts])
        y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
        return x, y


def sliding_windows(data: np.ndarray, block_size: int, stride: int):
    """Windows over the whole text so that every scored character sees >= block_size - stride context.

    Yields (x, y, first_scored_position, start). Only positions >= first_scored_position are scored, except in the
    very first window where all positions are scored.
    """
    n = len(data) - 1
    start = 0
    first = True
    while start + block_size <= n:
        yield (data[start:start + block_size], data[start + 1:start + block_size + 1],
               0 if first else block_size - stride, start)
        first = False
        start += stride
