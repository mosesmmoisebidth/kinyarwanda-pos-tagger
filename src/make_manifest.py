"""Write models.json: every shipped checkpoint with its Google Drive file id and its accuracy.

python src/make_manifest.py --summary results/summary.json --drive_ids drive_ids.json \
       --folder https://drive.google.com/drive/folders/<id> --out models.json
drive_ids.json maps "<recipe>_<model>" -> Drive file id of models/<recipe>_<model>/model.pt.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pairs import MODELS, PAIRS, RECIPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--drive_ids", required=True)
    ap.add_argument("--best_metrics_dir", default=None, help="folder with <recipe>_<model>/metrics.json of shipped checkpoints")
    ap.add_argument("--sizes", default=None, help="json: id -> size in MB")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="models.json")
    args = ap.parse_args()
    summary = json.load(open(args.summary, encoding="utf-8"))
    ids = json.load(open(args.drive_ids, encoding="utf-8"))
    sizes = json.load(open(args.sizes, encoding="utf-8")) if args.sizes else {}
    models = []
    for recipe in RECIPES:
        for a, b, _ in PAIRS:
            for key in (a, b):
                mid = f"{recipe}_{key}"
                s = summary.get(f"{recipe}/{key}")
                if not s or mid not in ids:
                    continue
                ck = {}
                if args.best_metrics_dir:
                    p = os.path.join(args.best_metrics_dir, mid, "metrics.json")
                    if os.path.isfile(p):
                        ck = json.load(open(p, encoding="utf-8"))
                models.append({
                    "id": mid, "recipe": recipe, "model": key, **MODELS[key], "drive_id": ids[mid],
                    "size_mb": sizes.get(mid), "seed": s["best_dev_seed"],
                    "dev": ck.get("dev", {}).get("acc"), "test": ck.get("test", {}).get("acc", s["best_dev_seed_test"]),
                    "mean_test": s["test_acc"][0], "std_test": s["test_acc"][1], "mean_oov": s["test_oov"][0],
                })
    best = max(models, key=lambda m: m["mean_test"])["id"] if models else None
    out = {"drive_folder": args.folder, "best": best, "recipes": RECIPES,
           "pairs": [list(p) for p in PAIRS], "models": models}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.out}: {len(models)} models, best = {best}")


if __name__ == "__main__":
    main()
