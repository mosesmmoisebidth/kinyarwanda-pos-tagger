"""Compare model pairs (general multilingual vs Afro-centric) on the Kinyarwanda test set.

Reads runs/<recipe>/<model>/seed<k>/{metrics.json,test_pred.txt} and writes
  results/summary.json      every (recipe, model): mean +/- std over seeds, OOV, per-tag F1
  results/pairs.json        each pair: accuracy gap, 95% CI (paired bootstrap over sentences),
                            permutation p-value over seeds, token wins/losses, per-tag gaps
  results/disagreements.json test sentences where the two models' majority votes differ (for the UI)
  results/REPORT.md         the same as readable tables

python src/compare_pairs.py --runs results/runs --data_dir data_raw/masakhane-pos/data --out results
"""
import argparse
import glob
import itertools
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.data import load_split  # noqa: E402
from kinpos.metrics import evaluate  # noqa: E402

MODELS = {
    "afroxlmr-76L": {"label": "AfroXLMR-large-76L", "hf": "Davlan/afro-xlmr-large-76L", "family": "Afro-centric",
                     "params": "560M", "note": "XLM-R large further pretrained on 76 African languages (incl. Kinyarwanda)"},
    "xlmr-large": {"label": "XLM-RoBERTa-large", "hf": "FacebookAI/xlm-roberta-large", "family": "General",
                   "params": "560M", "note": "100 languages from CommonCrawl; Kinyarwanda not among them"},
    "afriberta-large": {"label": "AfriBERTa-large", "hf": "castorini/afriberta_large", "family": "Afro-centric",
                        "params": "126M", "note": "trained from scratch on 11 African languages (incl. Kinyarwanda)"},
    "mbert": {"label": "mBERT (cased)", "hf": "google-bert/bert-base-multilingual-cased", "family": "General",
              "params": "178M", "note": "104 Wikipedia languages; Kinyarwanda not among them"},
}
PAIRS = [("xlmr-large", "afroxlmr-76L", "Large pair (same XLM-R large architecture)"),
         ("mbert", "afriberta-large", "Base pair (BERT-base size)")]
RECIPES = {"plain": "Plain fine-tuning on Kinyarwanda (MasakhaPOS protocol)",
           "full": "Full recipe: multilingual Stage 1 + char/affix features + CRF"}


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


def ms(xs):
    return (round(statistics.mean(xs), 3), round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0)


def perm_p(a, b):
    """Exact two-sided permutation test on the difference of means (seeds are independent runs)."""
    pooled, na = a + b, len(a)
    obs = abs(statistics.mean(a) - statistics.mean(b))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), na):
        s = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in s]
        hits += abs(statistics.mean(x) - statistics.mean(y)) >= obs - 1e-12
        total += 1
    return round(hits / total, 4)


def vote(pred_sets):
    return [[Counter(col).most_common(1)[0][0] for col in zip(*[p[i] for p in pred_sets])]
            for i in range(len(pred_sets[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gold = load_split(args.data_dir, "kin", "test")
    vocab = {w.lower() for s in load_split(args.data_dir, "kin", "train") for w in s["words"]}
    n_tok = [len(s["words"]) for s in gold]
    oov_mask = [[w.lower() not in vocab for w in s["words"]] for s in gold]

    runs = defaultdict(dict)   # (recipe, model) -> {seed: {"m": metrics, "p": preds}}
    for mpath in sorted(glob.glob(os.path.join(args.runs, "*", "*", "seed*", "metrics.json"))):
        d = os.path.dirname(mpath)
        seed = int(os.path.basename(d)[4:])
        model = os.path.basename(os.path.dirname(d))
        recipe = os.path.basename(os.path.dirname(os.path.dirname(d)))
        if recipe not in RECIPES or model not in MODELS or not os.path.isfile(os.path.join(d, "test_pred.txt")):
            continue
        runs[(recipe, model)][seed] = {"m": json.load(open(mpath, encoding="utf-8")),
                                       "p": read_preds(os.path.join(d, "test_pred.txt"))}

    summary = {}
    for (recipe, model), seeds in sorted(runs.items()):
        ms_ = [seeds[k]["m"] for k in sorted(seeds)]
        tags = sorted({t for m in ms_ for t in m["test"]["per_tag"]})
        per_tag = {t: round(statistics.mean(m["test"]["per_tag"].get(t, {"f1": 0})["f1"] for m in ms_), 2) for t in tags}
        support = {t: ms_[0]["test"]["per_tag"].get(t, {}).get("support", 0) for t in tags}
        best_seed = max(sorted(seeds), key=lambda k: seeds[k]["m"]["dev"]["acc"])
        ens = evaluate(gold, vote([seeds[k]["p"] for k in sorted(seeds)]), vocab)
        summary[f"{recipe}/{model}"] = {
            "recipe": recipe, "model": model, **MODELS[model], "seeds": sorted(seeds),
            "dev_acc": ms([m["dev"]["acc"] for m in ms_]), "dev_oov": ms([m["dev"]["oov_acc"] for m in ms_]),
            "test_acc": ms([m["test"]["acc"] for m in ms_]), "test_oov": ms([m["test"]["oov_acc"] for m in ms_]),
            "test_iv": ms([m["test"]["iv_acc"] for m in ms_]),
            "test_per_seed": {k: seeds[k]["m"]["test"]["acc"] for k in sorted(seeds)},
            "best_dev_seed": best_seed, "best_dev_seed_test": seeds[best_seed]["m"]["test"]["acc"],
            "seed_vote_ensemble_test": ens["acc"], "per_tag_f1": per_tag, "support": support,
            "minutes_per_run": round(statistics.mean(m.get("minutes", 0) for m in ms_), 1),
        }

    rng = random.Random(13)
    pairs_out, disagreements = [], []
    for recipe in RECIPES:
        for a, b, title in PAIRS:
            if (recipe, a) not in runs or (recipe, b) not in runs:
                continue
            A, B = runs[(recipe, a)], runs[(recipe, b)]
            accA = [A[k]["m"]["test"]["acc"] for k in sorted(A)]
            accB = [B[k]["m"]["test"]["acc"] for k in sorted(B)]
            # per-sentence correct counts averaged over seeds -> paired bootstrap over sentences
            def correct(seeds, oov_only=False):
                out = []
                for i, s in enumerate(gold):
                    c = 0.0
                    for k in seeds:
                        c += sum(1 for j, (g, p) in enumerate(zip(s["tags"], seeds[k]["p"][i]))
                                 if g == p and (not oov_only or oov_mask[i][j]))
                    out.append(c / len(seeds))
                return out
            cA, cB = correct(A), correct(B)
            diffs = [y - x for x, y in zip(cA, cB)]   # B minus A (Afro-centric minus general)
            total = sum(n_tok)
            gap = 100 * sum(diffs) / total
            boots = []
            idx = list(range(len(gold)))
            for _ in range(args.boot):
                samp = [rng.choice(idx) for _ in idx]
                boots.append(100 * sum(diffs[i] for i in samp) / sum(n_tok[i] for i in samp))
            boots.sort()
            lo, hi = boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot) - 1]
            p_boot = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / args.boot
            # token-level wins / losses on majority votes
            vA, vB = vote([A[k]["p"] for k in sorted(A)]), vote([B[k]["p"] for k in sorted(B)])
            b_only = a_only = both_wrong = 0
            for i, s in enumerate(gold):
                for g, x, y in zip(s["tags"], vA[i], vB[i]):
                    b_only += (y == g and x != g)
                    a_only += (x == g and y != g)
                    both_wrong += (x != g and y != g)
                if vA[i] != vB[i]:
                    disagreements.append({
                        "recipe": recipe, "pair": f"{a}|{b}", "sid": i, "words": s["words"], "gold": s["tags"],
                        "a": vA[i], "b": vB[i],
                        "a_err": sum(g != x for g, x in zip(s["tags"], vA[i])),
                        "b_err": sum(g != y for g, y in zip(s["tags"], vB[i]))})
            sa, sb = summary[f"{recipe}/{a}"], summary[f"{recipe}/{b}"]
            tag_gaps = {t: round(sb["per_tag_f1"].get(t, 0) - sa["per_tag_f1"].get(t, 0), 2)
                        for t in sa["per_tag_f1"] if sa["support"].get(t, 0) >= 20}
            pairs_out.append({
                "recipe": recipe, "title": title, "general": a, "afro": b,
                "general_test": sa["test_acc"], "afro_test": sb["test_acc"],
                "general_oov": sa["test_oov"], "afro_oov": sb["test_oov"],
                "gap_afro_minus_general": round(gap, 3), "ci95": [round(lo, 3), round(hi, 3)],
                "p_bootstrap": round(p_boot, 4), "p_permutation_seeds": perm_p(accB, accA),
                "oov_gap": round(sb["test_oov"][0] - sa["test_oov"][0], 3),
                "tokens_only_afro_right": b_only, "tokens_only_general_right": a_only, "tokens_both_wrong": both_wrong,
                "per_tag_gap": dict(sorted(tag_gaps.items(), key=lambda kv: -abs(kv[1]))),
                "winner": b if gap > 0 else a, "significant": lo > 0 or hi < 0,
            })

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), indent=1)
    json.dump(pairs_out, open(os.path.join(args.out, "pairs.json"), "w", encoding="utf-8"), indent=1)
    json.dump(disagreements, open(os.path.join(args.out, "disagreements.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    # ---------- REPORT.md ----------
    L = ["# Results: general vs Afro-centric models for Kinyarwanda POS tagging", "",
         f"Test set: MasakhaPOS Kinyarwanda, {len(gold)} sentences, {sum(n_tok)} tokens "
         f"({sum(map(sum, oov_mask))} unseen in training). Mean +/- std over seeds; "
         "each seed keeps its best dev epoch. The checkpoint shipped for each model is its best-dev seed.", ""]
    for recipe, desc in RECIPES.items():
        rows = [v for v in summary.values() if v["recipe"] == recipe]
        if not rows:
            continue
        L += [f"## {desc}", "", "| Model | Type | Params | Seeds | Dev acc | Test acc | Test OOV acc | Seed-vote ensemble |",
              "|---|---|---|---|---|---|---|---|"]
        for v in sorted(rows, key=lambda v: -v["test_acc"][0]):
            L.append(f"| {v['label']} | {v['family']} | {v['params']} | {len(v['seeds'])} | "
                     f"{v['dev_acc'][0]:.2f} ± {v['dev_acc'][1]:.2f} | **{v['test_acc'][0]:.2f} ± {v['test_acc'][1]:.2f}** | "
                     f"{v['test_oov'][0]:.2f} ± {v['test_oov'][1]:.2f} | {v['seed_vote_ensemble_test']:.2f} |")
        L.append("")
    if pairs_out:
        L += ["## Pairwise comparison (Afro-centric minus general)", "",
              "| Recipe | Pair | General | Afro-centric | Gap (pts) | 95% CI | p (bootstrap) | p (seeds) | OOV gap | Tokens only Afro right / only general right |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for p in pairs_out:
            L.append(f"| {p['recipe']} | {MODELS[p['general']]['label']} vs {MODELS[p['afro']]['label']} | "
                     f"{p['general_test'][0]:.2f} | {p['afro_test'][0]:.2f} | {p['gap_afro_minus_general']:+.2f} | "
                     f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | {p['p_bootstrap']:.4f} | {p['p_permutation_seeds']:.4f} | "
                     f"{p['oov_gap']:+.2f} | {p['tokens_only_afro_right']} / {p['tokens_only_general_right']} |")
        L.append("")
    open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
