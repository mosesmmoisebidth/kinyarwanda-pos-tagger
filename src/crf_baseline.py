"""Feature CRF baseline (MasakhaPOS paper reports 95.3 test accuracy for kin)."""
import argparse
import itertools
import json
import os
import sys

import sklearn_crfsuite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.data import load_split  # noqa: E402
from kinpos.metrics import evaluate  # noqa: E402


def word_feats(words, i):
    w = words[i]
    lw = w.lower()
    f = {
        "bias": 1.0, "w": lw, "len": min(len(w), 15), "title": w[:1].isupper(), "upper": w.isupper(),
        "digit": any(c.isdigit() for c in w), "alldigit": w.replace(",", "").replace(".", "").isdigit(),
        "punct": not any(c.isalnum() for c in w),
    }
    for n in (1, 2, 3, 4, 5):
        f[f"p{n}"] = lw[:n]
        f[f"s{n}"] = lw[-n:]
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(words):
            o = words[j].lower()
            f[f"{off}:w"] = o
            f[f"{off}:p3"] = o[:3]
            f[f"{off}:s3"] = o[-3:]
            f[f"{off}:title"] = words[j][:1].isupper()
        else:
            f[f"{off}:edge"] = True
    return f


def featurize(sents):
    return [[word_feats(s["words"], i) for i in range(len(s["words"]))] for s in sents]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--lang", default="kin")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tr, dv, te = (load_split(args.data_dir, args.lang, s) for s in ("train", "dev", "test"))
    vocab = {w.lower() for s in tr for w in s["words"]}
    Xtr, Xdv, Xte = featurize(tr), featurize(dv), featurize(te)
    ytr = [s["tags"] for s in tr]
    best = None
    for c1, c2 in itertools.product((0.02, 0.05, 0.1, 0.25), (0.01, 0.05, 0.1)):
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=300,
                                   all_possible_transitions=True)
        crf.fit(Xtr, ytr)
        acc = evaluate(dv, crf.predict(Xdv), vocab)["acc"]
        print(f"c1={c1} c2={c2} dev={acc}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, c1, c2, crf)
    acc, c1, c2, crf = best
    res = {"c1": c1, "c2": c2, "dev": evaluate(dv, crf.predict(Xdv), vocab),
           "test": evaluate(te, crf.predict(Xte), vocab)}
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("FINAL", json.dumps({"c1": c1, "c2": c2, "dev_acc": res["dev"]["acc"], "test_acc": res["test"]["acc"],
                               "test_oov": res["test"]["oov_acc"], "test_iv": res["test"]["iv_acc"]}))


if __name__ == "__main__":
    main()
