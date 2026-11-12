"""Token accuracy, accuracy on unseen words, per-tag scores and top confusions."""
from collections import Counter


def evaluate(sents, preds, train_vocab=None):
    total = correct = 0
    oov_total = oov_correct = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    confusions = Counter()
    for s, p in zip(sents, preds):
        for w, g, t in zip(s["words"], s["tags"], p):
            total += 1
            ok = g == t
            correct += ok
            if train_vocab is not None and w.lower() not in train_vocab:
                oov_total += 1
                oov_correct += ok
            if ok:
                tp[g] += 1
            else:
                fp[t] += 1
                fn[g] += 1
                confusions[(g, t)] += 1
    per_tag = {}
    for tag in sorted(set(tp) | set(fp) | set(fn)):
        prec = tp[tag] / (tp[tag] + fp[tag]) if tp[tag] + fp[tag] else 0.0
        rec = tp[tag] / (tp[tag] + fn[tag]) if tp[tag] + fn[tag] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_tag[tag] = {"p": round(100 * prec, 2), "r": round(100 * rec, 2),
                        "f1": round(100 * f1, 2), "support": tp[tag] + fn[tag]}
    out = {
        "acc": round(100 * correct / max(total, 1), 3),
        "tokens": total,
        "errors": total - correct,
        "per_tag": per_tag,
        "top_confusions": [f"{g}->{t}: {n}" for (g, t), n in confusions.most_common(15)],
    }
    if train_vocab is not None:
        out["oov_acc"] = round(100 * oov_correct / max(oov_total, 1), 3)
        out["oov_tokens"] = oov_total
        iv_total = total - oov_total
        out["iv_acc"] = round(100 * (correct - oov_correct) / max(iv_total, 1), 3)
    return out
