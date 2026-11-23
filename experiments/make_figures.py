"""Draw every figure in figures/ from results/.

    python experiments/make_figures.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import style
from style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, SEQ_BLUES, VIOLET, YELLOW, subtitle

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.environ.get("RES_DIR", os.path.join(ROOT, "results"))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
style.apply()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def load_log(name):
    p = os.path.join(RES, f"train_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


S = json.load(open(os.path.join(RES, "summary.json")))
kn_orders = pd.read_csv(os.path.join(RES, "ngram_orders.csv"))
KN_BEST = float(kn_orders.val_bpc.min())
KN_N = int(kn_orders.loc[kn_orders.val_bpc.idxmin(), "order"])
HEAD = "main_long" if load_log("main_long") else "main"   # the headline model
main = load_log(HEAD)
short = load_log("main") if HEAD != "main" else None
GPT_VAL = main["final_val_bpc_full"]
ENS = S.get("ensemble_val_bpc")

# ---------------------------------------------------------------- 0. headline: every method on one scale
cp = os.path.join(RES, "compressors.csv")
if os.path.exists(cp):
    comp = pd.read_csv(cp)
    rows = [(f"{r.method} (general-purpose compressor)", r.bpc_primed_with_train, MUTED) for r in comp.itertuples()]
    rows += [(f"Kneser-Ney {KN_N}-gram (counting)", KN_BEST, ORANGE)]
    if short:
        rows += [(f"GPT, {short['args']['steps']:,} training steps", short["final_val_bpc_full"], SEQ_BLUES[1])]
    rows += [(f"GPT, {main['args']['steps']:,} training steps", GPT_VAL, BLUE)]
    if ENS:
        rows += [("GPT + n-gram mixture", ENS, AQUA)]
    rows = sorted(rows, key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], 0.62, color=[r[2] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.03, yi, f"{r[1]:.2f}", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("bits per character on the held-out abstracts (lower = better prediction = better compression)")
    ax.set_xlim(0, max(r[1] for r in rows) + 0.35)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"A 0.8M-parameter NumPy GPT needs {GPT_VAL:.2f} bits per character; gzip needs "
                 f"{comp.loc[comp.method == 'gzip -9', 'bpc_primed_with_train'].iloc[0]:.2f}")
    subtitle(ax, "All methods saw the same 1.3M training characters first. Plain text costs 8 bits per character")
    save(fig, "00_headline.png")

# ---------------------------------------------------------------- 1. training curves
fig, ax = plt.subplots(figsize=(8.4, 4.5))
runs = [(main, BLUE, f"{main['args']['steps']:,}-step schedule")]
if short:
    runs.append((short, ORANGE, f"{short['args']['steps']:,}-step schedule"))
for log, col, lab in runs:
    ax.plot(log["steps"], log["val_bpc"], color=col, lw=2.2, label=f"{lab}: validation")
    ax.plot(log["steps"], log["train_eval_bpc"], color=col, lw=1.4, ls="--", label=f"{lab}: training text")
ax.axhline(KN_BEST, color=INK2, lw=1, ls=":")
ax.text(main["steps"][-1] * 0.98, KN_BEST + 0.03,  f"Kneser-Ney {KN_N}-gram on validation ({KN_BEST:.2f})",
        va="bottom", ha="right", fontsize=9, color=INK2)
ax.set_ylim(0.9, 3.0)
ax.set_xlabel("training step (32 sequences x 128 characters each)")
ax.set_ylabel("bits per character (lower is better)")
ax.legend(loc="upper right", ncol=1 if short is None else 2, fontsize=8.8)
beats = "beating" if GPT_VAL < KN_BEST else "short of"
ax.set_title(f"The GPT reaches {GPT_VAL:.2f} bits/char on unseen abstracts, {beats} the n-gram")
gap = main["final_val_bpc_full"] - main["final_train_bpc_sample"]
subtitle(ax, f"Dashed = the same model scored on its training text; the {gap:.2f}-bit gap is memorisation")
save(fig, "01_training_curves.png")

# ---------------------------------------------------------------- 2. n-gram order sweep
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(kn_orders.order, kn_orders.train_bpc, color=ORANGE, marker="o", ls="--", label="n-gram on training text")
ax.plot(kn_orders.order, kn_orders.val_bpc, color=ORANGE, marker="o", label="n-gram on validation text")
ax.axhline(GPT_VAL, color=BLUE, lw=2)
ax.text(1.1, GPT_VAL + 0.05, f"GPT, validation ({GPT_VAL:.2f})", color=INK, fontsize=9, va="bottom")
if ENS:
    ax.axhline(ENS, color=AQUA, lw=2)
    ax.text(1.1, ENS - 0.05, f"GPT + n-gram mixture ({ENS:.2f})", color=INK, fontsize=9, va="top")
ax.annotate("longer contexts keep improving\nthe fit to the training text ...", (9, kn_orders.train_bpc.iloc[8]),
            xytext=(5.2, 0.25), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.annotate("... but stop helping on new text", (9, kn_orders.val_bpc.iloc[8]), xytext=(6.6, 2.0),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.set_xticks(kn_orders.order)
ax.set_xlabel("n-gram order n (prediction uses the previous n-1 characters)")
ax.set_ylabel("bits per character")
ax.set_ylim(0, 4.5)
ax.legend(loc="upper right")
ax.set_title("Counting memorises: beyond ~7 characters of context, n-grams stop improving on new text")
subtitle(ax, "Kneser-Ney smoothed character n-grams; the widening gap is the curse of dimensionality (51^n possible contexts)")
save(fig, "02_ngram_orders.png")

# ---------------------------------------------------------------- 3. scaling
sizes = []
for name in ["size_32x2", "size_64x2", "size_96x3", "main"]:
    lg = load_log(name)
    if lg:
        sizes.append({"name": name, "params": lg["params"], "params_no_emb": lg["params_no_emb"],
                      "val": lg["final_val_bpc_full"], "train": lg["final_train_bpc_sample"],
                      "d": lg["args"]["n_embd"], "L": lg["args"]["n_layer"]})
sz = pd.DataFrame(sizes)
if len(sz):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(sz.params, sz.val, color=BLUE, marker="o", ms=8, label="validation")
    ax.plot(sz.params, sz.train, color=BLUE, marker="o", ms=6, ls="--", alpha=0.6, label="training text")
    for _, r in sz.iterrows():
        ax.annotate(f"d={r.d}, {r.L} layers\n{r.params / 1e3:.0f}k params", (r.params, r.val), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.axhline(KN_BEST, color=ORANGE, lw=1.5, ls=":")
    ax.text(sz.params.min(), KN_BEST - 0.03, f"Kneser-Ney {KN_N}-gram ({KN_BEST:.2f})", color=INK, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("bits per character")
    ax.legend(loc="upper right")
    lo, hi = min(sz.val.min(), sz.train.min(), KN_BEST), max(sz.val.max(), sz.train.max())
    ax.set_ylim(lo - 0.12, hi + 0.2)
    ax.set_xlim(sz.params.min() / 1.8, sz.params.max() * 1.8)
    ax.set_title("Bigger models predict better, with diminishing returns")
    subtitle(ax, "Four model sizes, identical data and 3,000 training steps each")
    save(fig, "03_scaling.png")
    sz.to_csv(os.path.join(RES, "scaling.csv"), index=False)

# ---------------------------------------------------------------- 4. loss vs position in the context window
pos = pd.read_csv(os.path.join(RES, "bpc_by_position.csv"))
fig, ax = plt.subplots(figsize=(8.4, 4.3))
roll = pos.gpt_bpc.rolling(5, center=True, min_periods=1).mean()
ax.plot(pos.position, pos.gpt_bpc, color=BLUE, lw=0.8, alpha=0.35)
ax.plot(pos.position, roll, color=BLUE, lw=2.2, label="GPT")
kroll = pos.kn_bpc_same_chars.rolling(5, center=True, min_periods=1).mean()
ax.plot(pos.position, kroll, color=ORANGE, lw=2, label=f"Kneser-Ney {KN_N}-gram, same characters")
ax.set_xlabel("position of the predicted character inside the 128-character window (= characters of context seen)")
ax.set_ylabel("bits per character")
ax.legend(loc="upper right")
ax.set_ylim(0, max(4, pos.gpt_bpc.max() + 0.2))
mid, late = pos.gpt_bpc.iloc[8:16].mean(), pos.gpt_bpc.iloc[64:].mean()
cross = int(np.argmax(roll.values < kroll.values))      # first position where the GPT is better
ax.axvline(KN_N - 1, color=INK2, lw=1, ls=":")
ax.text(KN_N - 1 + 1.5, ax.get_ylim()[1] * 0.93, f"the n-gram's limit:\n{KN_N - 1} characters", fontsize=8.5, color=INK2, va="top")
ax.set_title(f"After ~{cross} characters the GPT overtakes the n-gram, and keeps improving past it")
subtitle(ax, f"GPT: {mid:.2f} bits with 8-15 characters of context, {late:.2f} with 64+. The n-gram line is flat: "
             f"it always sees its {KN_N - 1} characters")
save(fig, "04_loss_by_position.png")

# ---------------------------------------------------------------- 5. where the bits go
cc = pd.read_csv(os.path.join(RES, "bpc_by_char_class.csv"))
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
y = np.arange(len(cc))[::-1]
ax.barh(y + 0.19, cc.gpt_bpc, 0.36, color=BLUE, label="GPT")
ax.barh(y - 0.19, cc.kn_bpc, 0.36, color=ORANGE, label=f"Kneser-Ney {KN_N}-gram")
for yi, g, k in zip(y, cc.gpt_bpc, cc.kn_bpc):
    ax.text(g + 0.04, yi + 0.19, f"{g:.2f}", va="center", fontsize=8.5, color=INK)
    ax.text(k + 0.04, yi - 0.19, f"{k:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, [f"{c}\n({s:.0%} of characters)" for c, s in zip(cc["class"], cc.share_of_chars)], fontsize=9)
ax.set_xlabel("bits per character")
ax.grid(axis="y", visible=False)
ax.legend(loc="center right")
ax.set_title("Bits per character, by what is being predicted", fontsize=11)
ax = axes[1]
ax.barh(y, cc.gpt_share_of_bits * 100, 0.6, color=BLUE)
for yi, v in zip(y, cc.gpt_share_of_bits):
    ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [""] * len(y))
ax.set_xlabel("% of the GPT's total uncertainty (bits)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(cc.gpt_share_of_bits * 100) + 12)
ax.set_title("Share of all bits spent", fontsize=11)
fw = cc[cc["class"] == "first letter of a word"].iloc[0]
fig.suptitle(f"Word beginnings are {fw.share_of_chars:.0%} of the text but {fw.gpt_share_of_bits:.0%} of the uncertainty: "
             "choosing the next word is the hard part", x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.08)
save(fig, "05_where_the_bits_go.png")

# ---------------------------------------------------------------- 6. novelty
nov = pd.read_csv(os.path.join(RES, "bpc_by_novelty.csv"))
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(len(nov))
ax.bar(x - 0.2, nov.gpt_bpc, 0.38, color=BLUE, label="GPT")
ax.bar(x + 0.2, nov.kn_bpc, 0.38, color=ORANGE, label=f"Kneser-Ney {KN_N}-gram")
ax.set_xticks(x, [f"{int(r.longest_seen_context)}\n({r.share:.1%})" for _, r in nov.iterrows()], fontsize=9)
ax.set_xlabel("longest preceding context (characters) that also occurs in the training text\n(share of validation characters)")
ax.set_ylabel("bits per character")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
gwin = nov[nov.gpt_bpc < nov.kn_bpc]
lose = nov[nov.gpt_bpc >= nov.kn_bpc]
if len(lose) and lose.longest_seen_context.max() < gwin.longest_seen_context.min():
    ax.set_title(f"The GPT wins once {int(gwin.longest_seen_context.min())}+ characters of context are familiar; "
                 "not on near-novel strings")
    subtitle(ax, f"Contexts with only {int(lose.longest_seen_context.min())}-{int(lose.longest_seen_context.max())} "
                 f"familiar characters ({lose.share.sum():.0%} of text: rare word pairings, numbers, codes) suit the n-gram's back-off")
else:
    ax.set_title("GPT vs n-gram, by how familiar the context is")
    subtitle(ax, "Bins with fewer than 200 characters are omitted")
save(fig, "06_novel_contexts.png")

# ---------------------------------------------------------------- 7. attention heads
heads = pd.read_csv(os.path.join(RES, "attention_heads.csv"))
L, H = heads["layer"].max() + 1, heads["head"].max() + 1
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
fig.subplots_adjust(wspace=0.38)
cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
for ax, col, ttl, fmt in [(axes[0], "mean_distance", "Mean look-back distance\n(characters)", "{:.1f}"),
                          (axes[1], "prev_char_mass", "Attention on the previous\ncharacter", "{:.0%}"),
                          (axes[2], "entropy_bits", "Attention entropy (bits)\nlow = focused, high = spread", "{:.1f}")]:
    M = heads.pivot(index="layer", columns="head", values=col).values
    ax.imshow(M, cmap=cmap, aspect="auto")
    for i in range(L):
        for j in range(H):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if v > (M.min() + M.max()) / 2 else INK)
    ax.set_xticks(range(H), [f"head {j}" for j in range(H)], fontsize=8.5)
    ax.set_yticks(range(L), [f"layer {i}" for i in range(L)], fontsize=8.5)
    ax.grid(False)
    ax.set_title(ttl, fontsize=10.5, loc="center", pad=8)
pc = heads.sort_values("prev_char_mass").iloc[-1]
fig.suptitle(f"Heads specialise: layer {int(pc['layer'])} head {int(pc['head'])} attends to the previous character "
             f"{pc['prev_char_mass']:.0%} of the time; deeper layers look further back",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.06)
save(fig, "07_attention_heads.png")

ex = np.load(os.path.join(RES, "attention_example.npz"))
text = str(ex["text"])
# pick the most local and the most long-range head
loc = heads.sort_values("mean_distance").iloc[0]
far = heads.sort_values("mean_distance").iloc[-1]
spc = heads.sort_values("last_space_mass").iloc[-1]
picks = [(int(loc["layer"]), int(loc["head"]), "most local head"), (int(spc["layer"]), int(spc["head"]), "most start-of-word head"),
         (int(far["layer"]), int(far["head"]), "longest-range head")]
seg = slice(0, 40)
chars = list(text[seg])
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, (l, h, lab) in zip(axes, picks):
    A = ex[f"layer{l}"][h][seg, seg]
    ax.imshow(A, cmap=cmap, vmin=0, vmax=min(1, A.max()))
    ax.set_xticks(range(len(chars)), ["␣" if c == " " else c for c in chars], fontsize=7)
    ax.set_yticks(range(len(chars)), ["␣" if c == " " else c for c in chars], fontsize=7)
    ax.tick_params(length=0, pad=1)
    ax.grid(False)
    ax.set_xlabel("attended-to character", fontsize=9)
    ax.set_title(f"layer {l}, head {h}: {lab}", fontsize=10.5, loc="center", pad=6)
axes[0].set_ylabel("predicting the character after ...", fontsize=9)
fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.12)
fig.suptitle("Attention patterns on one sentence (darker = more attention; each row sums to 1)",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.995)
save(fig, "08_attention_examples.png")

# ---------------------------------------------------------------- 8. temperature trade-off
sm = pd.read_csv(os.path.join(RES, "sampling.csv"))
g = sm[sm.model == "GPT"]
real = sm[sm.model == "real validation text"].iloc[0]
knr = sm[sm.model.str.startswith("Kneser")].iloc[0]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(g.temperature, g.real_word_rate * 100, color=BLUE, marker="o", label="real words (% of generated words that exist)")
ax.plot(g.temperature, g.distinct_word_rate * 100, color=ORANGE, marker="o", label="variety (% of generated words that are distinct)")
ax.plot(g.temperature, g.copied_20gram_rate * 100, color=AQUA, marker="o", label="copying (% of 20-character spans found in training text)")
for v, col in [(real.real_word_rate, BLUE), (real.distinct_word_rate, ORANGE), (real.copied_20gram_rate, AQUA)]:
    ax.scatter([1.62], [v * 100], marker="D", s=40, color=col, zorder=3)
ax.text(1.62, real.copied_20gram_rate * 100 - 4, "real\ntext", ha="center", va="top", fontsize=8.5, color=INK2)
ax.set_xlim(0.2, 1.75)
ax.set_ylim(0, 105)
ax.set_xlabel("sampling temperature (divides the logits before the softmax)")
ax.set_ylabel("%")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, ncol=1)
ax.set_title("Temperature trades correctness for variety")
subtitle(ax, "Low temperature: repetitive and copied; high temperature: invented words. Diamonds = real abstracts")
save(fig, "09_temperature.png")

# ---------------------------------------------------------------- 9. autocomplete
ac = json.load(open(os.path.join(RES, "autocomplete.json")))
names = {"freq": "word\nfrequency\nonly", "kn": f"Kneser-Ney\n{KN_N}-gram", "gpt": "GPT",
         "mix": "GPT + n-gram\nmixture"}
cols = {"freq": MUTED, "kn": ORANGE, "gpt": BLUE, "mix": AQUA}
curve = pd.read_csv(os.path.join(RES, "autocomplete_curve.csv"))
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
x = np.arange(len(ac))
vals = [r["keystroke_savings_rate"] * 100 for r in ac]
ax.bar(x, vals, 0.6, color=[cols[r["method"]] for r in ac])
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.axhline(ac[0]["max_possible_savings"] * 100, color=INK2, ls=":", lw=1)
ax.text(-0.4, ac[0]["max_possible_savings"] * 100 + 0.8, "upper bound (every word guessed before its first letter)",
        fontsize=8, color=INK2)
ax.set_xticks(x, [names[r["method"]] for r in ac], fontsize=9)
ax.set_ylabel("keystrokes saved (% of characters)")
ax.set_ylim(0, ac[0]["max_possible_savings"] * 100 + 8)
ax.grid(axis="x", visible=False)
ax.set_title("Keystroke savings", fontsize=11)
ax = axes[1]
for r in ac:
    d = curve[curve.method == r["method"]]
    ax.plot(d.chars_typed, d.share_of_words_completed_by_then * 100, color=cols[r["method"]], marker="o",
            label={"freq": "most frequent word with the typed prefix"}.get(r["method"], names[r["method"]].replace("\n", " ")))
ax.set_xlabel("letters typed before accepting the suggestion")
ax.set_ylabel("% of words completed by then")
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.set_title("How early the right word is suggested", fontsize=11)
gpt_ksr = next(r for r in ac if r["method"] == "gpt")["keystroke_savings_rate"]
fig.suptitle(f"As a technical-writing autocomplete, the GPT would save {gpt_ksr:.0%} of keystrokes",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.04)
save(fig, "10_autocomplete.png")
