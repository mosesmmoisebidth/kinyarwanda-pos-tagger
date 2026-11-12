"""After training: build the shareable Drive folder with every final model + a zip of the best one.

python colab/make_release.py --out /content/drive/MyDrive/kinpos_pairs \
       --release /content/drive/MyDrive/Kinyarwanda_POS_models
Needs results/summary.json from src/compare_pairs.py (written to <out>/results).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/mosesmmoisebidth/kinyarwanda-pos-tagger"
NOTEBOOK = ("https://colab.research.google.com/github/mosesmmoisebidth/kinyarwanda-pos-tagger/blob/main/"
            "notebooks/Kinyarwanda_POS_Tagger.ipynb")


def card(mid, s, ck):
    recipe = "full recipe (multilingual Stage 1 + char/affix features + CRF)" if s["recipe"] == "full" \
        else "plain fine-tuning"
    return f"""# Kinyarwanda POS tagger: {s['label']} ({s['recipe']})

Part-of-speech tagger for Kinyarwanda (17 Universal Dependencies tags), trained on MasakhaPOS.
Backbone: {s['hf']} ({s['family']}, {s['params']} parameters), trained with the {recipe}.

## Accuracy (MasakhaPOS Kinyarwanda test set, 604 sentences)
- This checkpoint (seed {s['best_dev_seed']}, picked on dev): dev {ck['dev']['acc']:.2f}%, test {ck['test']['acc']:.2f}%,
  unseen words {ck['test']['oov_acc']:.2f}%
- Mean over {len(s['seeds'])} seeds: test {s['test_acc'][0]:.2f} ± {s['test_acc'][1]:.2f}%, unseen words {s['test_oov'][0]:.2f}%

## Use it
```bash
pip install -r requirements.txt
echo "Abana barakina mu rugo." > in.txt
python src/tag_text.py --models model.pt --input in.txt          # -> Abana_NOUN barakina_VERB mu_ADP rugo_NOUN ._PUNCT
```
The first run downloads the backbone's config and tokenizer from Hugging Face ({s['hf']}).
Easiest: the ready-to-run Colab notebook {NOTEBOOK}

Code, all results and the comparison with the other models: {REPO}
Data license: MasakhaPOS is CC BY-NC 4.0, so this model is for non-commercial use.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="training output folder (has best/ and results/)")
    ap.add_argument("--release", required=True, help="folder to create (shared with 'anyone with the link')")
    args = ap.parse_args()
    summary = json.load(open(os.path.join(args.out, "results", "summary.json"), encoding="utf-8"))
    os.makedirs(args.release, exist_ok=True)
    rows = []
    for key, s in summary.items():
        recipe, model = key.split("/")
        mid = f"{recipe}_{model}"
        src = os.path.join(args.out, "best", recipe, model)
        dst = os.path.join(args.release, mid)
        os.makedirs(dst, exist_ok=True)
        for f in ("model.pt", "metrics.json"):
            if not os.path.isfile(os.path.join(dst, f)) or \
                    os.path.getsize(os.path.join(dst, f)) != os.path.getsize(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        ck = json.load(open(os.path.join(src, "metrics.json"), encoding="utf-8"))
        open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(card(mid, s, ck))
        rows.append((s["dev_acc"][0], mid, s, ck))
        print("copied", mid, flush=True)
    # best = highest mean DEV accuracy over seeds (the test set is not used to choose)
    rows.sort(key=lambda r: -r[0])
    _, best, s, ck = rows[0]
    zip_path = os.path.join(args.release, f"kinpos-best-{best}.zip")
    card_path = os.path.join(args.release, best, "README.md")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "package_model.py"), "--model_dir",
                    os.path.join(args.release, best), "--out", zip_path, "--card", card_path], check=True)
    res_dst = os.path.join(args.release, "results")
    os.makedirs(res_dst, exist_ok=True)
    for f in ("summary.json", "pairs.json", "REPORT.md"):
        shutil.copy(os.path.join(args.out, "results", f), os.path.join(res_dst, f))
    lines = ["Kinyarwanda POS tagger: trained models", "",
             f"Best model (highest dev accuracy): {best}  ->  {os.path.basename(zip_path)} (model + code, ready to run)", "",
             "Every folder holds model.pt (fp16 weights), metrics.json and a README with usage.", ""]
    for dev, mid, s, ck in rows:
        lines.append(f"{mid:26s} {s['label']:20s} {s['family']:13s} test {s['test_acc'][0]:.2f} ± {s['test_acc'][1]:.2f}  "
                     f"(this checkpoint {ck['test']['acc']:.2f})")
    lines += ["", f"Colab notebook: {NOTEBOOK}", f"Code and full results: {REPO}"]
    open(os.path.join(args.release, "README.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
