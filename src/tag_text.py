"""Tag raw Kinyarwanda text with one model or an ensemble (majority vote per word).

Inference:   python src/tag_text.py --models runs/final/*/model.pt --input in.txt --output out.txt
Silver data: python src/tag_text.py --models ... --input selftrain_kin.txt --format conll --agree_only \
                 --max_sents 60000 --output silver.conll
"""
import argparse
import glob
import os
import random
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.batching import encode_sentences  # noqa: E402
from kinpos.data import UPOS  # noqa: E402
from kinpos.model import Tagger  # noqa: E402
from kinpos.pretok import pretokenize  # noqa: E402
from train_tagger import run_eval  # noqa: E402


def load_model(path, device, local_files_only, base_model=None, encoder_dtype=None):
    """base_model overrides the backbone name stored in the checkpoint (config + tokenizer only).
    encoder_dtype=torch.bfloat16 keeps the encoder at half precision (about half the RAM, for CPU serving);
    the small head stays fp32."""
    # mmap: tensors are read from the file on demand instead of a second full copy in RAM
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=encoder_dtype is not None)
    a = ckpt["args"]
    name = base_model or a["model_name"]
    model = Tagger(name, len(UPOS), len(ckpt["char_vocab"]), use_char=a["use_char"],
                   use_affix=a["use_affix"], use_crf=a["use_crf"], pool=a["pool"], mix_layers=a["mix_layers"],
                   head_hidden=a["head_hidden"], dropout=a["dropout"], local_files_only=local_files_only,
                   pretrained=False, encoder_dtype=encoder_dtype)
    if encoder_dtype is None:
        model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    else:  # load_state_dict copies into the existing params, converting dtype per tensor
        model.load_state_dict(ckpt["state_dict"])
    char_vocab = ckpt["char_vocab"]
    del ckpt
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    return model, tok, char_vocab, a.get("max_len", 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--input", required=True, help="raw text, one sentence per line ('-' = stdin)")
    ap.add_argument("--output", default="-")
    ap.add_argument("--format", choices=["inline", "conll"], default="inline")
    ap.add_argument("--agree_only", action="store_true", help="keep sentences where every model agrees")
    ap.add_argument("--max_sents", type=int, default=0)
    ap.add_argument("--min_words", type=int, default=1)
    ap.add_argument("--max_words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    paths = [p for m in args.models for p in sorted(glob.glob(m))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    src = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    lines = [ln.strip() for ln in src if ln.strip()]
    if args.max_sents and len(lines) > args.max_sents:
        random.Random(args.seed).shuffle(lines)
        lines = lines[:args.max_sents]
    sents = []
    for ln in lines:
        words = pretokenize(ln)
        if args.min_words <= len(words) <= args.max_words:
            sents.append({"words": words, "tags": None, "lang": "kin"})
    print(f"tagging {len(sents)} sentences with {len(paths)} model(s)", file=sys.stderr, flush=True)

    all_preds = []
    for p in paths:
        model, tok, char_vocab, max_len = load_model(p, device, args.local_files_only)
        chunks = encode_sentences(sents, tok, char_vocab, max_len)
        all_preds.append(run_eval(model, chunks, sents, tok.pad_token_id, device, 64, use_bf16))
        print(f"done {p}", file=sys.stderr, flush=True)
        del model
        torch.cuda.empty_cache()

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    kept = 0
    for i, s in enumerate(sents):
        per_model = [pr[i] for pr in all_preds]
        if args.agree_only and any(pm != per_model[0] for pm in per_model[1:]):
            continue
        vote = [Counter(col).most_common(1)[0][0] for col in zip(*per_model)]
        kept += 1
        if args.format == "conll":
            out.write("".join(f"{w} {t}\n" for w, t in zip(s["words"], vote)) + "\n")
        else:
            out.write(" ".join(f"{w}_{t}" for w, t in zip(s["words"], vote)) + "\n")
    print(f"wrote {kept} sentences", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
