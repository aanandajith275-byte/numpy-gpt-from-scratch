"""Character n-gram language model with interpolated Kneser-Ney smoothing (Chen & Goodman, 1998).

This is the strongest classical baseline for character-level language modelling and the model a
neural network has to beat to justify itself.

An n-gram model predicts the next character from only the previous n-1 characters, using counts:

    P(w | h) = max(c(h w) - D, 0) / c(h)  +  D * N1+(h .) / c(h) * P_lower(w | h')

* D (discount) takes a little probability mass from every seen continuation ...
* ... and gives it to the shorter-context model P_lower, weighted by how many *different*
  characters were seen after h (N1+(h .)). Contexts with many possible continuations trust the
  lower order more.
* Kneser-Ney's key idea: the lower-order models use *continuation counts*, the number of distinct
  contexts a string appears after, not raw counts. A string that is frequent but only ever follows one
  context (e.g. "ynolds" after "re") should not get high probability in new contexts.

Implementation: every n-gram is packed into one int64 (base-V digits), so counting is np.unique and
lookup is np.searchsorted; no Python dictionaries.
"""
from __future__ import annotations

import numpy as np


def _codes(seq: np.ndarray, n: int, V: int) -> np.ndarray:
    """Integer code of every length-n window of seq (base V)."""
    L = len(seq) - n + 1
    code = np.zeros(L, dtype=np.int64)
    for j in range(n):
        code = code * V + seq[j:j + L]
    return code


def _lookup(keys: np.ndarray, vals: np.ndarray, query: np.ndarray) -> np.ndarray:
    """vals[key == query], or 0 where the query is not present (keys sorted)."""
    if len(keys) == 0:
        return np.zeros(query.shape, dtype=vals.dtype)
    pos = np.clip(np.searchsorted(keys, query), 0, len(keys) - 1)
    return np.where(keys[pos] == query, vals[pos], 0)


class KneserNey:
    def __init__(self, order: int, vocab_size: int):
        self.N, self.V = order, vocab_size

    def fit(self, seq: np.ndarray):
        N, V = self.N, self.V
        self.levels = []
        # unique n-grams and their raw counts, for n = 1 .. N + 1 (N+1 needed for continuation counts)
        uniq = {}
        for n in range(1, N + 2):
            u, c = np.unique(_codes(seq, n, V), return_counts=True)
            uniq[n] = (u, c)
        for n in range(1, N + 1):
            if n == N:
                grams, counts = uniq[n]                          # highest order: raw counts
            else:
                longer = uniq[n + 1][0]                          # continuation: distinct left-extensions
                grams, counts = np.unique(longer % (V ** n), return_counts=True)
            n1, n2 = (counts == 1).sum(), (counts == 2).sum()
            D = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.5
            ctx = grams // V                                      # the (n-1)-char history of each n-gram
            ctx_keys, inv = np.unique(ctx, return_inverse=True)
            ctx_total = np.bincount(inv, weights=counts).astype(np.float64)
            ctx_types = np.bincount(inv).astype(np.float64)
            self.levels.append({"n": n, "D": D, "grams": grams, "counts": counts.astype(np.float64),
                                "ctx": ctx_keys, "ctx_total": ctx_total, "ctx_types": ctx_types})
        return self

    def _interpolate(self, gram_codes_by_order, ctx_codes_by_order, valid_by_order):
        """Shared recursion: start from uniform, then mix in each order where its history exists."""
        p = np.full(gram_codes_by_order[0].shape, 1.0 / self.V)
        for lvl, g, h, ok in zip(self.levels, gram_codes_by_order, ctx_codes_by_order, valid_by_order):
            c = _lookup(lvl["grams"], lvl["counts"], g)
            if lvl["n"] == 1:
                total = lvl["ctx_total"][0]; types = lvl["ctx_types"][0]
                p_n = (np.maximum(c - lvl["D"], 0) + lvl["D"] * types * p) / total
            else:
                total = _lookup(lvl["ctx"], lvl["ctx_total"], h)
                types = _lookup(lvl["ctx"], lvl["ctx_types"], h)
                seen = (total > 0) & ok
                safe = np.where(seen, total, 1.0)
                p_n = np.where(seen, (np.maximum(c - lvl["D"], 0) + lvl["D"] * types * p) / safe, p)
            p = p_n
        return p

    def logprobs(self, seq: np.ndarray) -> np.ndarray:
        """log2 P(seq[i] | seq[:i]) for i = 1 .. len(seq) - 1 (history truncated to N - 1 characters)."""
        V, L = self.V, len(seq)
        idx = np.arange(1, L)
        grams, ctxs, oks = [], [], []
        for n in range(1, self.N + 1):
            g = np.zeros(len(idx), dtype=np.int64)
            ok = idx - (n - 1) >= 0
            for j in range(n - 1, -1, -1):
                pos = np.clip(idx - j, 0, None)
                g = g * V + seq[pos]
            grams.append(g); ctxs.append(g // V); oks.append(ok)
        return np.log2(self._interpolate(grams, ctxs, oks))

    def next_dist(self, history: np.ndarray) -> np.ndarray:
        """Full distribution over the next character given a history (array of ids)."""
        V = self.V
        w = np.arange(V, dtype=np.int64)
        grams, ctxs, oks = [], [], []
        for n in range(1, self.N + 1):
            h = history[len(history) - (n - 1):] if n > 1 else history[:0]
            ok = len(h) == n - 1
            hc = 0
            for t in h:
                hc = hc * V + int(t)
            g = hc * V + w
            grams.append(g); ctxs.append(np.full(V, hc, dtype=np.int64)); oks.append(np.full(V, ok))
        p = self._interpolate(grams, ctxs, oks)
        return p / p.sum()

    def longest_seen_context(self, seq: np.ndarray) -> np.ndarray:
        """For each i = 1 .. len(seq) - 1: the largest n <= N whose (n-1)-character history before
        seq[i] occurs in the training text. 1 means even the previous character was never seen
        in that role; N means the full history was seen. Measures how 'novel' each context is."""
        V, L = self.V, len(seq)
        idx = np.arange(1, L)
        best = np.ones(len(idx), dtype=np.int64)
        for lvl in self.levels[1:]:
            n = lvl["n"]
            h = np.zeros(len(idx), dtype=np.int64)
            for j in range(n - 1, 0, -1):
                h = h * V + seq[np.clip(idx - j, 0, None)]
            seen = (_lookup(lvl["ctx"], lvl["ctx_total"], h) > 0) & (idx - (n - 1) >= 0)
            best = np.where(seen, n, best)
        return best
