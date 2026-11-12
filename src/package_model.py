"""Bundle one trained checkpoint with the code needed to run it, as a zip for sharing.

python src/package_model.py --model_dir models/full_afroxlmr-76L --out dist/kinpos-afroxlmr-76L-full.zip
"""
import argparse
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = ["src/tag_text.py", "src/train_tagger.py", "src/kinpos/__init__.py", "src/kinpos/batching.py",
        "src/kinpos/crf.py", "src/kinpos/data.py", "src/kinpos/features.py", "src/kinpos/metrics.py",
        "src/kinpos/model.py", "src/kinpos/pretok.py", "requirements.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default=None, help="markdown model card to include as README.md")
    args = ap.parse_args()
    m = json.load(open(os.path.join(args.model_dir, "metrics.json"), encoding="utf-8"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    name = os.path.splitext(os.path.basename(args.out))[0]
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.write(os.path.join(args.model_dir, "model.pt"), f"{name}/model.pt", compress_type=zipfile.ZIP_STORED)
        z.write(os.path.join(args.model_dir, "metrics.json"), f"{name}/metrics.json")
        for rel in CODE:
            z.write(os.path.join(ROOT, rel), f"{name}/{rel}")
        if args.card:
            z.write(args.card, f"{name}/README.md")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2**20:.0f} MB); dev {m['dev']['acc']} test {m.get('test', {}).get('acc')}")


if __name__ == "__main__":
    main()
