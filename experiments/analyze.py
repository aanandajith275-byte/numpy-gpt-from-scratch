"""Analyse the trained GPT against the Kneser-Ney n-gram baseline. Writes results/*.json|csv.

    python experiments/analyze.py            (~15 minutes; needs checkpoints/main_long.npz)
"""
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from tinygpt.data import CharData  # noqa: E402
from tinygpt.evaluate import per_char_logprobs  # noqa: E402
from tinygpt.model import GPT, softmax  # noqa: E402
from tinygpt.ngram import KneserNey  # noqa: E402

RES = os.environ.get("RES_DIR", os.path.join(ROOT, "results"))
os.makedirs(RES, exist_ok=True)
t0 = time.time()
data = CharData()
V = data.vocab_size
CKPT = os.environ.get("CKPT", "main_long")          # override for quick debugging runs
model, meta = GPT.load(os.path.join(ROOT, "checkpoints", f"{CKPT}.npz"))
T = model.cfg.block_size
SPACE, NL = data.stoi[" "], data.stoi["\n"]
print(f"model: {model.num_params():,} params, best step {meta['step']}")

# ---------------------------------------------------------------- 1. n-gram order sweep
kn_rows, kn_models = [], {}
for n in range(1, 11):
    kn = KneserNey(n, V).fit(data.train)
    kn_models[n] = kn
    kn_rows.append({"order": n, "val_bpc": float(-kn.logprobs(data.val).mean()),
                    "train_bpc": float(-kn.logprobs(data.train[:200_001]).mean())})
pd.DataFrame(kn_rows).to_csv(os.path.join(RES, "ngram_orders.csv"), index=False)
best_order = min(kn_rows, key=lambda r: r["val_bpc"])["order"]
kn = kn_models[best_order]
print("KN orders:", [(r["order"], round(r["val_bpc"], 3)) for r in kn_rows], "best", best_order)

# ---------------------------------------------------------------- 2. aligned per-character comparison
gpt_lp, gpt_pos, gpt_abs = per_char_logprobs(model, data.val, stride=T // 2)
kn_lp_all = kn.logprobs(data.val)                 # index i-1 holds the prediction of val[i]
kn_lp = kn_lp_all[gpt_abs - 1]
novelty = kn.longest_seen_context(data.val)[gpt_abs - 1]
target = data.val[gpt_abs]
prev = data.val[gpt_abs - 1]
is_letter = np.isin(target, [data.stoi[c] for c in "abcdefghijklmnopqrstuvwxyz"])
is_digit = np.isin(target, [data.stoi[c] for c in "0123456789"])
prev_boundary = np.isin(prev, [SPACE, NL, data.stoi["("], data.stoi["-"], data.stoi["/"]])
cls = np.where(is_letter & prev_boundary, "first letter of a word",
       np.where(is_letter, "rest of a word",
       np.where(is_digit, "digit", np.where(target == SPACE, "space", "punctuation / newline"))))
cmp_rows = []
for c in ["first letter of a word", "rest of a word", "space", "digit", "punctuation / newline"]:
    s = cls == c
    cmp_rows.append({"class": c, "share_of_chars": float(s.mean()),
                     "gpt_bpc": float(-gpt_lp[s].mean()), "kn_bpc": float(-kn_lp[s].mean()),
                     "gpt_share_of_bits": float(gpt_lp[s].sum() / gpt_lp.sum()),
                     "kn_share_of_bits": float(kn_lp[s].sum() / kn_lp.sum())})
pd.DataFrame(cmp_rows).to_csv(os.path.join(RES, "bpc_by_char_class.csv"), index=False)
nov_rows = []
for n in range(1, best_order + 1):
    s = novelty == n
    if s.sum() >= 200:
        nov_rows.append({"longest_seen_context": n - 1, "n_chars": int(s.sum()), "share": float(s.mean()),
                         "gpt_bpc": float(-gpt_lp[s].mean()), "kn_bpc": float(-kn_lp[s].mean())})
pd.DataFrame(nov_rows).to_csv(os.path.join(RES, "bpc_by_novelty.csv"), index=False)

summary = {"gpt_val_bpc_aligned": float(-gpt_lp.mean()), "kn_val_bpc_aligned": float(-kn_lp.mean()),
           "kn_best_order": int(best_order), "n_scored_chars": int(len(gpt_lp)),
           "gpt_better_fraction": float((gpt_lp > kn_lp).mean()),
           "params": model.num_params(), "best_step": int(meta["step"]), "vocab_size": V,
           "train_chars": int(len(data.train)), "val_chars": int(len(data.val)),
           "kn_unique_ngrams_top_order": int(len(kn.levels[-1]["grams"]))}
# ensemble: the two models make different mistakes, so mix their probabilities
#   p = lam * p_gpt + (1 - lam) * p_kn
# lam is chosen by 2-fold cross-fitting: tuned on one half of the validation text and scored on the
# other half, then the other way round, so the reported number is never tuned on what it is scored on.
p_g, p_k = 2.0 ** gpt_lp, 2.0 ** kn_lp
half = len(p_g) // 2
lams = np.round(np.arange(0, 1.0001, 0.05), 2)
halves = [np.arange(0, half), np.arange(half, len(p_g))]
ens_lp = np.empty_like(gpt_lp)
chosen = []
for tune_i, score_i in [(0, 1), (1, 0)]:
    t = halves[tune_i]
    curve = [float(-np.log2(l * p_g[t] + (1 - l) * p_k[t]).mean()) for l in lams]
    lam_i = float(lams[int(np.argmin(curve))])
    chosen.append(lam_i)
    sidx = halves[score_i]
    ens_lp[sidx] = np.log2(lam_i * p_g[sidx] + (1 - lam_i) * p_k[sidx])
lam = float(np.mean(chosen))
summary["ensemble_lambdas"] = chosen
summary["ensemble_lambda"] = lam
summary["ensemble_val_bpc"] = float(-ens_lp.mean())
pd.DataFrame({"lambda": lams, "bpc_full_val_in_sample": [float(-np.log2(l * p_g + (1 - l) * p_k).mean()) for l in lams]}
             ).to_csv(os.path.join(RES, "ensemble_lambda.csv"), index=False)

# is the gap real? Characters inside one abstract are not independent, so resample whole abstracts
# (a cluster bootstrap): 95% intervals for the BPC differences, and how many abstracts each model wins.
doc = np.cumsum(data.val == NL)[gpt_abs]
_, doc = np.unique(doc, return_inverse=True)
n_docs = int(doc.max()) + 1
cnt = np.bincount(doc, minlength=n_docs)
boot_rng = np.random.default_rng(0)
B = boot_rng.integers(0, n_docs, size=(2000, n_docs))
for key, a_lp, b_lp in [("kn_minus_gpt", kn_lp, gpt_lp), ("gpt_minus_ensemble", gpt_lp, ens_lp)]:
    per_doc = np.bincount(doc, weights=(b_lp - a_lp).astype(np.float64), minlength=n_docs)  # bits saved
    diffs = per_doc[B].sum(1) / cnt[B].sum(1)
    summary[f"{key}_bpc"] = float(per_doc.sum() / cnt.sum())
    summary[f"{key}_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    summary[f"{key}_docs_won"] = float((per_doc > 0).mean())
summary["n_val_abstracts"] = n_docs

# the same task for general-purpose compressors. "primed" = size(train + val) - size(train): the
# compressor gets to see the training text first, like the models did.
import bz2, lzma, zlib  # noqa: E401,E402
tr_b, va_b = data.decode(data.train).encode(), data.decode(data.val).encode()
comp_rows = []
for name, f in [("gzip -9", lambda b: zlib.compress(b, 9)), ("bzip2 -9", lambda b: bz2.compress(b, 9)),
                ("xz -9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))]:
    comp_rows.append({"method": name, "bpc_alone": len(f(va_b)) * 8 / len(va_b),
                      "bpc_primed_with_train": (len(f(tr_b + va_b)) - len(f(tr_b))) * 8 / len(va_b)})
pd.DataFrame(comp_rows).to_csv(os.path.join(RES, "compressors.csv"), index=False)
print(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- 3. how the loss depends on context position
# non-overlapping windows: every position 0..T-1 sees the same characters on average
lp_nov, pos_nov, abs_nov = per_char_logprobs(model, data.val, stride=T)
pos_rows = []
for p in range(T):
    s = pos_nov == p
    pos_rows.append({"position": p, "gpt_bpc": float(-lp_nov[s].mean()),
                     "kn_bpc_same_chars": float(-kn_lp_all[abs_nov[s] - 1].mean())})
pd.DataFrame(pos_rows).to_csv(os.path.join(RES, "bpc_by_position.csv"), index=False)

# ---------------------------------------------------------------- 4. attention heads
rng = np.random.default_rng(0)
starts = rng.integers(0, len(data.val) - T - 1, size=96)
X = np.stack([data.val[s:s + T] for s in starts])
_, _, cache = model.forward(X, keep_attention=True)
i_idx = np.arange(T)[:, None]
j_idx = np.arange(T)[None, :]
dist = np.where(j_idx <= i_idx, i_idx - j_idx, 0).astype(np.float64)
head_rows = []
for l, P in enumerate(cache["attention"]):          # P: (B, H, T, T)
    P = P[:, :, 32:, :]                              # queries with at least 32 characters of history
    d_ = dist[32:]
    for h in range(model.cfg.n_head):
        ph = P[:, h]
        ent = -(ph * np.log2(ph + 1e-12)).sum(-1)
        prev_mass = np.mean([ph[:, i - 32, i - 1].mean() for i in range(32, T)])
        self_mass = np.mean([ph[:, i - 32, i].mean() for i in range(32, T)])
        # mass on the most recent space (start of the current word)
        sp = []
        for b in range(ph.shape[0]):
            for i in range(32, T):
                spaces = np.where(X[b, :i] == SPACE)[0]
                if len(spaces):
                    sp.append(ph[b, i - 32, spaces[-1]])
        head_rows.append({"layer": l, "head": h, "mean_distance": float((ph * d_).sum(-1).mean()),
                          "entropy_bits": float(ent.mean()), "prev_char_mass": float(prev_mass),
                          "self_mass": float(self_mass), "last_space_mass": float(np.mean(sp))})
pd.DataFrame(head_rows).to_csv(os.path.join(RES, "attention_heads.csv"), index=False)
example = "the effect of the boundary layer on the pressure distribution over the wing was measured in the wind tunnel ."
ex_ids = data.encode(example)[None, :]
_, _, ex_cache = model.forward(ex_ids, keep_attention=True)
np.savez_compressed(os.path.join(RES, "attention_example.npz"), text=np.array(example),
                    **{f"layer{l}": P[0] for l, P in enumerate(ex_cache["attention"])})

# ---------------------------------------------------------------- 5. sampling: temperature trade-off
train_text = data.decode(data.train)
train_words = Counter(re.findall(r"[a-z]+", train_text))
tt = data.train
# a 64-bit polynomial hash of every 20-character window of the training text (overflow wraps, by design)
h = np.zeros(len(tt) - 19, dtype=np.uint64)
for j in range(20):
    h = h * np.uint64(1000003) + tt[j:j + len(tt) - 19].astype(np.uint64)
train_20grams = set(h.tolist())


def text_stats(texts):
    words = [w for t in texts for w in re.findall(r"[a-z]{3,}", t)]  # words of 3+ letters
    real = np.mean([w in train_words for w in words]) if words else 0.0
    distinct = len(set(words)) / max(1, len(words))
    copied = []
    for t in texts:
        ids = data.encode(t)
        if len(ids) >= 20:
            hh = np.zeros(len(ids) - 19, dtype=np.uint64)
            for j in range(20):
                hh = hh * np.uint64(1000003) + ids[j:j + len(ids) - 19].astype(np.uint64)
            copied.append(np.mean([int(x) in train_20grams for x in hh]))
    return float(real), float(distinct), float(np.mean(copied)), len(words)


prompt = "the boundary layer"
p_ids = data.encode(prompt)
samp_rows, samples = [], {}
N_SAMPLES, N_CHARS = 12, 400
for temp in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
    out = model.generate(np.tile(p_ids, (N_SAMPLES, 1)), N_CHARS, temperature=temp, rng=np.random.default_rng(1))
    texts = [data.decode(o[len(p_ids):]) for o in out]
    real, distinct, copied, nw = text_stats(texts)
    samp_rows.append({"model": "GPT", "temperature": temp, "real_word_rate": real, "distinct_word_rate": distinct,
                      "copied_20gram_rate": copied, "n_words": nw})
    samples[f"gpt_T{temp}"] = [prompt + t for t in texts[:3]]
    print(f"  T={temp}: real {real:.3f} distinct {distinct:.3f} copied {copied:.3f}")
kn_rng = np.random.default_rng(1)
kn_texts = []
for s in range(N_SAMPLES):
    ids = list(p_ids)
    for _ in range(N_CHARS):
        ids.append(int(kn_rng.choice(V, p=kn.next_dist(np.array(ids[-(best_order - 1):])))))
    kn_texts.append(data.decode(ids[len(p_ids):]))
real, distinct, copied, nw = text_stats(kn_texts)
samp_rows.append({"model": f"Kneser-Ney {best_order}-gram", "temperature": 1.0, "real_word_rate": real,
                  "distinct_word_rate": distinct, "copied_20gram_rate": copied, "n_words": nw})
samples["kneser_ney"] = [prompt + t for t in kn_texts[:3]]
real_val, distinct_val, copied_val, _ = text_stats([data.decode(data.val[i:i + N_CHARS]) for i in range(0, 12 * 5000, 5000)])
samp_rows.append({"model": "real validation text", "temperature": np.nan, "real_word_rate": real_val,
                  "distinct_word_rate": distinct_val, "copied_20gram_rate": copied_val, "n_words": 0})
pd.DataFrame(samp_rows).to_csv(os.path.join(RES, "sampling.csv"), index=False)
json.dump(samples, open(os.path.join(RES, "samples.json"), "w"), indent=2)

# ---------------------------------------------------------------- 6. autocomplete: keystroke savings
# Simulated typist: before each keystroke the system proposes the rest of the current word (greedy
# decoding up to the next space). If the proposal is exactly right the user presses Tab (1 keystroke)
# instead of typing the remaining characters. Keystroke savings rate = keystrokes saved / characters.
AC_CHARS = int(os.environ.get("AC_CHARS", 12_000))
ac_text = data.val[:AC_CHARS]
words = []  # (start index, end index exclusive) of each word = maximal run of non-space/newline chars
i = 0
while i < len(ac_text):
    if ac_text[i] in (SPACE, NL):
        i += 1
        continue
    j = i
    while j < len(ac_text) and ac_text[j] not in (SPACE, NL):
        j += 1
    words.append((i, j))
    i = j
words = [w for w in words if w[0] >= T]  # every word gets a full context window
print(f"autocomplete: {len(words)} words in {AC_CHARS} characters")


def gpt_complete_batch(contexts, targets, max_len=24, mix=None):
    """For each (context ids, target continuation ids incl. trailing space), does greedy decoding
    reproduce the target? Decoding stops early as soon as it diverges."""
    n = len(contexts)
    ok = np.ones(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    seqs = [list(c) for c in contexts]
    for step in range(max_len):
        active = np.where(~done)[0]
        if len(active) == 0:
            break
        for b0 in range(0, len(active), 256):
            act = active[b0:b0 + 256]
            win = np.stack([np.array(seqs[a][-T:]) for a in act])
            probs = softmax(model.forward(win)[0][:, -1, :])
            if mix is not None:  # ensemble: lam * GPT + (1 - lam) * Kneser-Ney
                knp = np.stack([kn.next_dist(np.array(seqs[a][-(best_order - 1):])) for a in act])
                probs = mix * probs + (1 - mix) * knp
            nxt = probs.argmax(-1)
            for a, c in zip(act, nxt):
                tgt = targets[a]
                if step >= len(tgt) or c != tgt[step]:
                    ok[a] = False; done[a] = True
                elif c in (SPACE, NL) or step == len(tgt) - 1:
                    done[a] = True
                else:
                    seqs[a].append(int(c))
    return ok


def kn_complete(context, target, max_len=24):
    seq = list(context)
    for step in range(min(max_len, len(target))):
        c = int(np.argmax(kn.next_dist(np.array(seq[-(best_order - 1):]))))
        if c != target[step]:
            return False
        if c in (SPACE, NL):
            return True
        seq.append(c)
    return False


word_freq = Counter(re.findall(r"[^\s]+", data.decode(data.train)))
by_prefix = {}
for w, c in word_freq.most_common():
    for p in range(len(w)):
        by_prefix.setdefault(w[:p], w)  # most frequent word with this prefix


def simulate(method):
    """Returns typed keystrokes per word and whether it was completed."""
    remaining = list(range(len(words)))
    accepted_at = {}
    max_p = max(e - s for s, e in words)
    for p in range(max_p):
        cand = [k for k in remaining if words[k][1] - words[k][0] - p >= 2]  # >=2 chars left to be worth a Tab
        if not cand:
            break
        if method in ("gpt", "mix"):
            ctx = [ac_text[max(0, words[k][0] + p - T):words[k][0] + p] for k in cand]
            tgt = [np.append(ac_text[words[k][0] + p:words[k][1]], SPACE) for k in cand]
            ok = gpt_complete_batch(ctx, tgt, mix=lam if method == "mix" else None)
        elif method == "kn":
            ok = [kn_complete(ac_text[max(0, words[k][0] + p - 16):words[k][0] + p],
                              np.append(ac_text[words[k][0] + p:words[k][1]], SPACE)) for k in cand]
        else:  # most frequent training word with the typed prefix
            ok = [by_prefix.get(data.decode(ac_text[words[k][0]:words[k][0] + p])) ==
                  data.decode(ac_text[words[k][0]:words[k][1]]) for k in cand]
        for k, good in zip(cand, ok):
            if good:
                accepted_at[k] = p
        remaining = [k for k in remaining if k not in accepted_at]
    saved = sum((words[k][1] - words[k][0] - p) - 1 for k, p in accepted_at.items())
    total = len(ac_text) - words[0][0]  # characters typed from the first evaluated word onwards
    word_len = np.array([e - s for s, e in words])
    return {"method": method, "keystroke_savings_rate": saved / total,
            "words_completed": len(accepted_at) / len(words),
            "avg_chars_typed_before_accept": float(np.mean(list(accepted_at.values()))) if accepted_at else np.nan,
            "max_possible_savings": float((word_len - 1).clip(0).sum() / total),
            "accepted_at": {int(k): int(v) for k, v in accepted_at.items()}}


ac_rows = []
for method in ["freq", "kn", "gpt", "mix"]:
    t1 = time.time()
    r = simulate(method)
    print(f"  {method}: KSR {r['keystroke_savings_rate']:.3f}, words completed {r['words_completed']:.3f} "
          f"({time.time() - t1:.0f}s)")
    ac_rows.append(r)
json.dump(ac_rows, open(os.path.join(RES, "autocomplete.json"), "w"))
# per-prefix-length acceptance curves
curve = []
for r in ac_rows:
    acc = np.array(list(r["accepted_at"].values()))
    for p in range(8):
        curve.append({"method": r["method"], "chars_typed": p, "share_of_words_completed_by_then":
                      float((acc <= p).sum() / len(words))})
pd.DataFrame(curve).to_csv(os.path.join(RES, "autocomplete_curve.csv"), index=False)

summary["runtime_seconds"] = time.time() - t0
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)
print(f"done in {(time.time() - t0) / 60:.1f} min")
