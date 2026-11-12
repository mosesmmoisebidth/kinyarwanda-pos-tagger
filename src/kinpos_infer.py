"""Inference helpers used by the Colab notebook (and handy in any Python session).

    import kinpos_infer as K
    K.show(K.tag("full_afroxlmr-76L", "Abana barakina mu rugo."))
    K.show_compare("full_xlmr-large", "full_afroxlmr-76L", "Ejo tuzajya i Kigali.")

Checkpoints listed in models.json are downloaded from Google Drive on first use into models/<id>/model.pt.
"""
import html
import json
import os
import re
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kinpos.batching import encode_sentences  # noqa: E402
from kinpos.data import load_split  # noqa: E402
from kinpos.metrics import evaluate  # noqa: E402
from kinpos.pretok import pretokenize  # noqa: E402
from tag_text import load_model  # noqa: E402
from train_tagger import run_eval  # noqa: E402

MANIFEST = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
BY_ID = {m["id"]: m for m in MANIFEST["models"]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“«(])")
_cache = {}

GROUP = {"NOUN": "n", "PROPN": "n", "VERB": "v", "AUX": "v", "ADJ": "m", "ADV": "m", "NUM": "m", "PRON": "p",
         "DET": "p", "ADP": "f", "CCONJ": "f", "SCONJ": "f", "PART": "f", "INTJ": "o", "PUNCT": "o", "SYM": "o", "X": "o"}
COLORS = {"n": ("#2563eb", "#e8efff"), "v": ("#c2410c", "#fff0e6"), "m": ("#15803d", "#e7f6ec"),
          "p": ("#7c3aed", "#f1eaff"), "f": ("#0e7490", "#e3f5f8"), "o": ("#57534e", "#efedea")}


def model_dir(mid):
    return os.path.join(ROOT, "models", mid)


def download(mid, quiet=False):
    """Fetch one checkpoint from Google Drive (skipped if already there)."""
    m = BY_ID[mid]
    path = os.path.join(model_dir(mid), "model.pt")
    if not (os.path.isfile(path) and os.path.getsize(path) > 1_000_000):
        os.makedirs(model_dir(mid), exist_ok=True)
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown
        print(f"Downloading {m['label']} ({m['recipe']} recipe, {m['size_mb']} MB)...", flush=True)
        out = gdown.download(id=m["drive_id"], output=path + ".part", quiet=quiet)
        if not out or not os.path.isfile(path + ".part"):
            raise RuntimeError(f"Download failed. Open {MANIFEST['drive_folder']} and download {mid}/model.pt by hand.")
        os.replace(path + ".part", path)
    with open(os.path.join(model_dir(mid), "metrics.json"), "w", encoding="utf-8") as f:  # read by the web UI
        json.dump({"dev": {"acc": m.get("dev")}, "test": {"acc": m.get("test")}}, f)
    return path


def get(mid, keep=2):
    """Load a model (downloading it if needed); keeps the `keep` most recent models in memory."""
    if mid in _cache:
        _cache[mid] = _cache.pop(mid)
        return _cache[mid]
    while len(_cache) >= keep:
        _cache.pop(next(iter(_cache)))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    _cache[mid] = load_model(download(mid), DEVICE, local_files_only=False)
    return _cache[mid]


def split_text(text):
    sents = []
    for para in text.splitlines():
        for chunk in SENT_END.split(para.strip()):
            words = pretokenize(chunk)
            if words:
                sents.append({"words": words, "tags": None, "lang": "kin"})
    return sents


def tag_sentences(mid, sents):
    model, tok, chars, max_len = get(mid)
    with torch.inference_mode():
        preds = run_eval(model, encode_sentences(sents, tok, chars, max_len), sents, tok.pad_token_id, DEVICE, 32, False)
    model.eval()
    return preds


def tag(mid, text):
    """Returns a list of sentences, each a list of (word, tag)."""
    sents = split_text(text)
    return [list(zip(s["words"], p)) for s, p in zip(sents, tag_sentences(mid, sents))] if sents else []


def as_text(result):
    return "\n".join(" ".join(f"{w}_{t}" for w, t in s) for s in result)


def _chip(word, tags, outline=False, wrong=()):
    rows = []
    for i, t in enumerate(tags):
        fg, bg = COLORS[GROUP.get(t, "o")]
        style = f"color:{fg};font-weight:700;font-size:11px;letter-spacing:.03em"
        if i in wrong:
            style = "color:#c62828;background:#fde8e8;text-decoration:line-through;font-weight:700;font-size:11px"
        rows.append(f"<span style='{style}'>{t}</span>")
    fg, bg = COLORS[GROUP.get(tags[0], "o")]
    box = "outline:2px solid #b45309;outline-offset:1px;" if outline else ""
    back = "#f0eee9" if len(tags) > 1 else bg
    return (f"<span style='display:inline-flex;flex-direction:column;align-items:center;margin:3px;padding:4px 8px;"
            f"border-radius:8px;background:{back};{box}'><span style='font-size:15px;color:#1d1d1f'>"
            f"{html.escape(word)}</span>{''.join(rows)}</span>")


def show(result, title=None):
    from IPython.display import HTML, display
    head = f"<div style='font:600 14px system-ui;margin:6px 0'>{html.escape(title)}</div>" if title else ""
    body = "".join(f"<div style='margin:6px 0;font-family:system-ui'>{''.join(_chip(w, [t]) for w, t in s)}</div>"
                   for s in result)
    display(HTML(head + body))


def show_compare(mid_a, mid_b, text):
    """Tags `text` with both models; words they tag differently are outlined."""
    from IPython.display import HTML, display
    a, b = tag(mid_a, text), tag(mid_b, text)
    la, lb = BY_ID[mid_a]["label"], BY_ID[mid_b]["label"]
    diffs = sum(ta != tb for sa, sb in zip(a, b) for (_, ta), (_, tb) in zip(sa, sb))
    head = (f"<div style='font:14px system-ui;margin:6px 0'>Top tag: <b>{html.escape(la)}</b> &nbsp; Bottom tag: "
            f"<b>{html.escape(lb)}</b> &nbsp; <b>{diffs}</b> word(s) tagged differently (outlined)</div>")
    body = "".join("<div style='margin:6px 0;font-family:system-ui'>" + "".join(
        _chip(w, [ta, tb], outline=ta != tb) for (w, ta), (_, tb) in zip(sa, sb)) + "</div>" for sa, sb in zip(a, b))
    display(HTML(head + body))
    return a, b


def show_models():
    """Table of every trained model with its accuracy (best first within each recipe)."""
    from IPython.display import HTML, display
    td = "<td style='padding:4px 10px'>"
    rows = []
    for m in sorted(MANIFEST["models"], key=lambda m: (m["recipe"] != "full", -m["mean_test"])):
        best = m["id"] == MANIFEST["best"]
        fg, bg = ("#b45309", "#fff3e0") if m["family"] == "Afro-centric" else ("#1d4ed8", "#e8efff")
        rows.append(
            f"<tr style='{'font-weight:700' if best else ''}'>{td}{html.escape(m['label'])}{' (best)' if best else ''}</td>"
            f"{td}<span style='color:{fg};background:{bg};border-radius:99px;padding:1px 8px'>{m['family']}</span></td>"
            f"{td}{m['params']}</td>{td}{'full recipe' if m['recipe'] == 'full' else 'plain fine-tune'}</td>"
            f"{td}{m['mean_test']:.2f} ± {m['std_test']:.2f}</td>{td}{m['mean_oov']:.2f}</td>{td}{m['size_mb']} MB</td></tr>")
    head = "".join(f"<th style='text-align:left;padding:4px 10px'>{h}</th>" for h in
                   ["Model", "Type", "Size", "Recipe", "Test accuracy % (5 seeds)", "Unseen words %", "Download"])
    display(HTML(f"<table style='border-collapse:collapse;font:14px system-ui'><tr>{head}</tr>{''.join(rows)}</table>"))


def masakhane_data(dest="/content/masakhane-pos"):
    if not os.path.isdir(os.path.join(dest, "data", "kin")):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/masakhane-io/masakhane-pos", dest],
                       check=True)
    return os.path.join(dest, "data")


def evaluate_test(mid, data_dir=None):
    """Re-scores one checkpoint on the MasakhaPOS Kinyarwanda test set (reproduces the reported number)."""
    data_dir = data_dir or masakhane_data()
    test = load_split(data_dir, "kin", "test")
    vocab = {w.lower() for s in load_split(data_dir, "kin", "train") for w in s["words"]}
    res = evaluate(test, tag_sentences(mid, test), vocab)
    print(f"{BY_ID[mid]['label']} ({BY_ID[mid]['recipe']}): test accuracy {res['acc']:.2f}%  |  unseen words "
          f"{res['oov_acc']:.2f}%  |  {res['errors']} errors on {res['tokens']} tokens")
    return res


def tag_file(mid, in_path, out_path, fmt="conll"):
    """Tags a UTF-8 text file (one sentence or paragraph per line)."""
    sents = split_text(open(in_path, encoding="utf-8").read())
    preds = tag_sentences(mid, sents)
    with open(out_path, "w", encoding="utf-8") as f:
        for s, p in zip(sents, preds):
            if fmt == "conll":
                f.write("".join(f"{w}\t{t}\n" for w, t in zip(s["words"], p)) + "\n")
            else:
                f.write(" ".join(f"{w}_{t}" for w, t in zip(s["words"], p)) + "\n")
    return len(sents)
