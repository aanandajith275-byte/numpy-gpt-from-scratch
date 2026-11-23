"""Generate aerospace-abstract-style text with the trained NumPy GPT.

    python generate.py "the boundary layer"
    python generate.py --temperature 0.5 --chars 400 "experimental investigation of"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tinygpt.data import CharData  # noqa: E402
from tinygpt.model import GPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="*", default=["the"])
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "checkpoints", "main_long.npz"))
    a = ap.parse_args()

    data = CharData()
    model, _ = GPT.load(a.checkpoint)
    prompt = " ".join(a.prompt).lower()
    unknown = sorted(set(prompt) - set(data.chars))
    if unknown:
        sys.exit(f"characters not in the vocabulary: {unknown}")
    ids = np.tile(data.encode(prompt), (a.samples, 1))
    out = model.generate(ids, a.chars, temperature=a.temperature, rng=np.random.default_rng(a.seed))
    for i, row in enumerate(out, 1):
        text = data.decode(row).split("\n")[0]  # stop at the end of the abstract
        print(f"--- sample {i} (temperature {a.temperature})\n{text}\n")


if __name__ == "__main__":
    main()
