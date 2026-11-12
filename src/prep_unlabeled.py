"""Build the unlabeled Kinyarwanda corpus for in-domain MLM training and self-training.

Every line that shares an 8-word window with any dev or test sentence is dropped, so the
test score stays honest (the IGIHE corpus contains the raw text of the annotated sentences).
"""
import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.data import load_split  # noqa: E402

N = 8
WORD = re.compile(r"\w+", re.UNICODE)


def norm_words(text):
    return [w.lower() for w in WORD.findall(text)]


def shingles(words):
    if len(words) < N:
        return {hashlib.md5(" ".join(words).encode()).hexdigest()} if words else set()
    return {hashlib.md5(" ".join(words[i:i + N]).encode()).hexdigest() for i in range(len(words) - N + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_words", type=int, default=5)
    args = ap.parse_args()

    banned = set()
    for split in ("dev", "test"):
        for s in load_split(args.data_dir, "kin", split):
            banned |= shingles(norm_words(" ".join(s["words"])))
    seen, kept, dropped_overlap, dropped_dup, dropped_short = set(), 0, 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in args.inputs:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = " ".join(line.split())
                    words = norm_words(line)
                    if len(words) < args.min_words:
                        dropped_short += 1
                        continue
                    key = hashlib.md5(" ".join(words).encode()).hexdigest()
                    if key in seen:
                        dropped_dup += 1
                        continue
                    seen.add(key)
                    if shingles(words) & banned:
                        dropped_overlap += 1
                        continue
                    out.write(line + "\n")
                    kept += 1
    print(f"kept {kept} lines | dropped: overlap-with-dev/test {dropped_overlap}, "
          f"duplicate {dropped_dup}, short {dropped_short}")


if __name__ == "__main__":
    main()
