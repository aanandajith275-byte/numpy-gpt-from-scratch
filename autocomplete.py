"""Technical-writing autocomplete powered by the NumPy GPT.

Type the beginning of a sentence; the model suggests how to finish the current word and the most
likely next words, with probabilities.

    python autocomplete.py "the pressure distribution over a delta w"
    python autocomplete.py                 # interactive
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tinygpt.data import CharData  # noqa: E402
from tinygpt.model import GPT, softmax  # noqa: E402


def word_candidates(model, data, text, n=5, beam=12, max_len=20):
    """Beam search over characters until each hypothesis ends its word with a space.

    Returns up to n (completion, probability) pairs for the rest of the current word."""
    space, nl = data.stoi[" "], data.stoi["\n"]
    T = model.cfg.block_size
    ctx = list(data.encode(text))
    beams = [([], 0.0)]                      # (characters so far, log probability)
    finished = []
    for _ in range(max_len):
        if not beams:
            break
        X = np.stack([np.array((ctx + b)[-T:]) for b, _ in beams])
        probs = softmax(model.forward(X)[0][:, -1, :])
        cand = []
        for (b, lp), p in zip(beams, probs):
            for c in np.argsort(-p)[:6]:
                nlp = lp + float(np.log(p[c] + 1e-12))
                if c in (space, nl):
                    finished.append((b, nlp))
                else:
                    cand.append((b + [int(c)], nlp))
        cand.sort(key=lambda x: -x[1])
        beams = cand[:beam]
        # stop once the n-th best finished word beats every unfinished one (log-probs only go down)
        if len(finished) >= n and beams and sorted(f[1] for f in finished)[-n] > beams[0][1]:
            break
    finished.sort(key=lambda x: -x[1])
    return [(data.decode(b), float(np.exp(lp))) for b, lp in finished[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*")
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "checkpoints", "main_long.npz"))
    a = ap.parse_args()
    data = CharData()
    model, _ = GPT.load(a.checkpoint)

    def show(text):
        text = text.lower()
        if set(text) - set(data.chars):
            print(f"  (unknown characters: {sorted(set(text) - set(data.chars))})")
            return
        typed = text.split(" ")[-1]
        print(f"\n  {text}|")
        for comp, p in word_candidates(model, data, text):
            shown = typed + comp if (typed or comp) else "(end of text)"
            if not comp and typed:
                shown += "  (word already complete)"
            print(f"    {shown:<34s} {p:6.1%}")
        print()

    if a.text:
        show(" ".join(a.text))
        return
    while True:
        try:
            t = input("text> ")
        except (EOFError, KeyboardInterrupt):
            break
        if t.strip():
            show(t)


if __name__ == "__main__":
    main()
