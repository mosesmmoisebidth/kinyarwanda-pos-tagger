"""Reading MasakhaPOS files ("word TAG" per line, blank line between sentences)."""
import os

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
TAG2ID = {t: i for i, t in enumerate(UPOS)}


def read_conll(path, lang=None):
    """Return a list of {"words", "tags", "lang"} dicts. Tags may be None for unlabeled text."""
    sents, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                if words:
                    sents.append({"words": words, "tags": tags, "lang": lang})
                    words, tags = [], []
                continue
            words.append(parts[0])
            tags.append(parts[-1] if len(parts) > 1 else None)
    if words:
        sents.append({"words": words, "tags": tags, "lang": lang})
    for s in sents:
        if all(t is None for t in s["tags"]):
            s["tags"] = None
        elif any(t not in TAG2ID for t in s["tags"]):
            bad = sorted({t for t in s["tags"] if t not in TAG2ID})
            raise ValueError(f"{path}: unknown tags {bad}")
    return sents


def load_split(data_dir, lang, split):
    return read_conll(os.path.join(data_dir, lang, f"{split}.txt"), lang=lang)


def parse_lang_weights(spec):
    """'kin' or 'kin:1,sna:2,lug:1.5' -> {lang: weight}."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            lang, w = item.split(":")
            out[lang] = float(w)
        else:
            out[item] = 1.0
    return out


def write_predictions(path, sents, preds):
    with open(path, "w", encoding="utf-8") as f:
        for s, p in zip(sents, preds):
            gold = s["tags"] or ["_"] * len(s["words"])
            for w, g, t in zip(s["words"], gold, p):
                f.write(f"{w} {g} {t}\n")
            f.write("\n")
