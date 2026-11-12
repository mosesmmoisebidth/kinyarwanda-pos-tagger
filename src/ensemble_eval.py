"""Majority-vote ensemble over prediction files (word gold pred per line); ties go to the first file.

python src/ensemble_eval.py --split dev runs/stage2/*/seed*
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.data import load_split  # noqa: E402
from kinpos.metrics import evaluate  # noqa: E402


def read_preds(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if not p:
            if cur:
                sents.append(cur)
                cur = []
            continue
        cur.append(p[-1])
    if cur:
        sents.append(cur)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data_dir", default="/scratch/network/$USER/kinpos/masakhane-pos/data")
    args = ap.parse_args()
    dirs = [d for pat in args.run_dirs for d in sorted(glob.glob(pat))
            if os.path.isfile(os.path.join(d, f"{args.split}_pred.txt"))]
    gold = load_split(args.data_dir, "kin", args.split)
    vocab = {w.lower() for s in load_split(args.data_dir, "kin", "train") for w in s["words"]}
    preds = [read_preds(os.path.join(d, f"{args.split}_pred.txt")) for d in dirs]
    vote = [[Counter(col).most_common(1)[0][0] for col in zip(*[p[i] for p in preds])] for i in range(len(gold))]
    singles = [evaluate(gold, p, vocab)["acc"] for p in preds]
    m = evaluate(gold, vote, vocab)
    print(json.dumps({"n_models": len(dirs), "single_mean": round(sum(singles) / len(singles), 3),
                      "ensemble_acc": m["acc"], "ensemble_oov": m["oov_acc"], "errors": m["errors"],
                      "top_confusions": m["top_confusions"][:6]}))


if __name__ == "__main__":
    main()
