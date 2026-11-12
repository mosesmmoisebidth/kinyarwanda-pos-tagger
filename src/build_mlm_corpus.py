"""Build the Kinyarwanda MLM corpus from FineWeb-2 kin_Latn + the IGIHE text.

Keeps every news-site document (igihe.com first: the POS data comes from IGIHE) and a random
sample of other documents up to --max_mb. Lines that share an 8-word window with any dev/test
sentence are dropped, so in-domain pre-training cannot leak the evaluation text.
Also writes in-domain sentences (one per line) for self-training.
"""
import argparse
import hashlib
import os
import random
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.data import load_split  # noqa: E402
from prep_unlabeled import norm_words, shingles  # noqa: E402

NEWS_HINTS = ("igihe", "kigalitoday", "umuseke", "kt.rw", "newtimes", "bwiza", "inyarwanda", "ukwezi",
              "intyoza", "umurunga", "rba.co.rw", "imvaho", "taarifa", "isimbi", "panorama", "bbc", "voa",
              "ireme", "rwandanews", "kinyamateka", "umubavu", "makuruki", "rwanda")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--extra_text", nargs="*", default=[])
    ap.add_argument("--out_mlm", required=True)
    ap.add_argument("--out_sents", required=True)
    ap.add_argument("--max_mb", type=float, default=150)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    banned = set()
    for split in ("dev", "test"):
        for s in load_split(args.data_dir, "kin", split):
            banned |= shingles(norm_words(" ".join(s["words"])))

    news, other = [], []
    domains = Counter()
    for path in args.parquet:
        table = pq.read_table(path, columns=["text", "url"])
        for text, url in zip(table.column("text").to_pylist(), table.column("url").to_pylist()):
            host = urlparse(url or "").netloc.lower()
            domains[host] += 1
            (news if any(h in host for h in NEWS_HINTS) else other).append((host, text))
    random.shuffle(other)
    print("top domains:", domains.most_common(25), flush=True)
    print(f"news docs {len(news)} | other docs {len(other)}", flush=True)

    seen = set()
    stats = Counter()
    budget = args.max_mb * 1e6
    written = 0
    with open(args.out_mlm, "w", encoding="utf-8") as fm, open(args.out_sents, "w", encoding="utf-8") as fs:
        def emit(text, in_domain):
            nonlocal written
            for para in text.split("\n"):
                para = " ".join(para.split())
                words = norm_words(para)
                if len(words) < 5:
                    stats["short"] += 1
                    continue
                key = hashlib.md5(" ".join(words).encode()).hexdigest()
                if key in seen:
                    stats["dup"] += 1
                    continue
                seen.add(key)
                if shingles(words) & banned:
                    stats["overlap_dev_test"] += 1
                    continue
                fm.write(para + "\n")
                written += len(para.encode("utf-8")) + 1
                stats["kept_lines"] += 1
                if in_domain:
                    for s in SENT_SPLIT.split(para):
                        if 5 <= len(s.split()) <= 60:
                            fs.write(s + "\n")
                            stats["self_train_sents"] += 1

        for path in args.extra_text:
            with open(path, encoding="utf-8") as f:
                emit(f.read(), True)
        for host, text in sorted(news, key=lambda x: "igihe" not in x[0]):
            emit(text, "igihe" in host)
        for _, text in other:
            if written >= budget:
                break
            emit(text, False)
    print(f"wrote {written / 1e6:.1f} MB | {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
