"""Mean and std of metrics.json files under a runs folder, grouped by parent folder."""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

root = sys.argv[1]
groups = defaultdict(list)
for p in sorted(glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True)):
    groups[os.path.dirname(os.path.dirname(p))].append(json.load(open(p, encoding="utf-8")))


def ms(xs):
    return f"{statistics.mean(xs):.2f} +/- {statistics.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


for g, runs in groups.items():
    line = f"{os.path.relpath(g, root):40s} n={len(runs)}  dev {ms([r['dev']['acc'] for r in runs])}"
    line += f"  dev_oov {ms([r['dev']['oov_acc'] for r in runs])}"
    if all("test" in r for r in runs):
        line += f"  | TEST {ms([r['test']['acc'] for r in runs])}  test_oov {ms([r['test']['oov_acc'] for r in runs])}"
    print(line)
