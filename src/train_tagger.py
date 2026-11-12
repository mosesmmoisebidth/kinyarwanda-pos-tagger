"""Train / evaluate a POS tagger on MasakhaPOS.

Phase 0 baseline:  --train_langs kin  (no --use_char/--use_affix/--use_crf)
Stage 1 (multilingual):  --train_langs sna:2,lug:2,...  --save_model
Stage 2 (Kinyarwanda):   --train_langs kin --init_from runs/stage1/model.pt --use_char --use_affix --use_crf
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.batching import collate, encode_sentences, to_device  # noqa: E402
from kinpos.data import UPOS, load_split, parse_lang_weights, read_conll, write_predictions  # noqa: E402
from kinpos.features import build_char_vocab  # noqa: E402
from kinpos.metrics import evaluate  # noqa: E402
from kinpos.model import Tagger  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-large-76L")
    ap.add_argument("--train_langs", default="kin", help="'kin' or 'sna:2,lug:2,kin:1'")
    ap.add_argument("--eval_lang", default="kin")
    ap.add_argument("--aux_splits", default="train",
                    help="splits used for languages other than eval_lang, e.g. 'train,dev,test'")
    ap.add_argument("--silver", default=None, help="extra auto-tagged file (word TAG per line)")
    ap.add_argument("--silver_weight", type=float, default=0.5)
    ap.add_argument("--train_on_dev", action="store_true", help="add eval-lang dev to training; keep last epoch")
    ap.add_argument("--eval_test", action="store_true", help="also score the test set (baselines / final only)")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--use_char", action="store_true")
    ap.add_argument("--use_affix", action="store_true")
    ap.add_argument("--use_crf", action="store_true")
    ap.add_argument("--pool", choices=["first", "mean"], default="first")
    ap.add_argument("--mix_layers", type=int, default=0)
    ap.add_argument("--head_hidden", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="multilingual: fixed sampled steps per epoch")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without dev gain (0 = off)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_train_sents", type=int, default=0, help="smoke tests only")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--no_bf16", action="store_true", help="train in full fp32 (no mixed precision)")
    ap.add_argument("--save_half", action="store_true", help="store model.pt weights in fp16 (half the size)")
    return ap.parse_args()


def amp_dtype(device, no_amp):
    """bf16 on Ampere+ (A100, L4, H100); fp16 + loss scaling on older GPUs such as the Colab T4."""
    if device.type != "cuda" or no_amp:
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_char_sents(data_dir):
    """Character vocabulary from every language's train file, so it is identical across stages."""
    out = []
    for lang in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, lang, "train.txt")
        if os.path.isfile(p):
            out.append(read_conll(p, lang))
    return out


def run_eval(model, chunks, sents, pad_id, device, bs, use_bf16, dtype=torch.bfloat16):
    model.eval()
    preds = [[None] * len(s["words"]) for s in sents]
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].input_ids))
    for i in range(0, len(order), bs):
        group = [chunks[j] for j in order[i:i + bs]]
        batch = to_device(collate(group, pad_id), device)
        with torch.autocast("cuda", dtype=dtype, enabled=bool(use_bf16)):
            paths, _ = model.predict(batch)
        for c, path in zip(group, paths):
            for k, t in enumerate(path):
                preds[c.sent_id][c.offset + k] = UPOS[t]
    model.train()
    return preds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(device, args.no_bf16)
    use_bf16 = dtype is not None          # "use mixed precision" (bf16 or fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    print(f"device {device} {torch.cuda.get_device_name(device) if device.type == 'cuda' else ''} amp {dtype}", flush=True)
    t0 = time.time()

    # ---------- data ----------
    weights = parse_lang_weights(args.train_langs)
    train_sents = []
    for lang, w in weights.items():
        if lang == args.eval_lang:
            sents = load_split(args.data_dir, lang, "train")
        else:
            sents = [s for sp in args.aux_splits.split(",") for s in load_split(args.data_dir, lang, sp)]
        if args.train_on_dev and lang == args.eval_lang:
            sents += load_split(args.data_dir, lang, "dev")
        if args.max_train_sents:
            sents = sents[:args.max_train_sents]
        for s in sents:
            s["weight"] = 1.0
            s["sample_w"] = w
        train_sents += sents
    if args.silver:
        silver = read_conll(args.silver, "silver")
        for s in silver:
            s["weight"] = args.silver_weight
            s["sample_w"] = 1.0
        train_sents += silver
        print(f"silver sentences: {len(silver)}", flush=True)
    dev_sents = load_split(args.data_dir, args.eval_lang, "dev")
    test_sents = load_split(args.data_dir, args.eval_lang, "test") if args.eval_test else []
    kin_vocab = {w.lower() for s in load_split(args.data_dir, args.eval_lang, "train") for w in s["words"]}
    if args.train_on_dev:
        kin_vocab |= {w.lower() for s in dev_sents for w in s["words"]}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    init_state = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        char_vocab = ckpt["char_vocab"]
        init_state = ckpt["state_dict"]
    else:
        char_vocab = build_char_vocab(all_char_sents(args.data_dir))

    train_chunks = encode_sentences(train_sents, tokenizer, char_vocab, args.max_len)
    dev_chunks = encode_sentences(dev_sents, tokenizer, char_vocab, args.max_len)
    test_chunks = encode_sentences(test_sents, tokenizer, char_vocab, args.max_len) if test_sents else []
    pad_id = tokenizer.pad_token_id
    print(f"train sents {len(train_sents)} chunks {len(train_chunks)} | dev {len(dev_sents)} | "
          f"test {len(test_sents)} | langs {weights}", flush=True)

    sample_w = [train_sents[c.sent_id]["sample_w"] for c in train_chunks]
    multi = len(set(sample_w)) > 1 or args.steps_per_epoch > 0
    if multi:
        n_per_epoch = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else len(train_chunks)
        sampler = WeightedRandomSampler(sample_w, num_samples=n_per_epoch, replacement=True)
        loader = DataLoader(train_chunks, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=lambda b: collate(b, pad_id))
    else:
        loader = DataLoader(train_chunks, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate(b, pad_id))

    # ---------- model ----------
    model = Tagger(args.model_name, len(UPOS), len(char_vocab), use_char=args.use_char,
                   use_affix=args.use_affix, use_crf=args.use_crf, pool=args.pool,
                   mix_layers=args.mix_layers, head_hidden=args.head_hidden, dropout=args.dropout,
                   local_files_only=args.local_files_only)
    if init_state is not None:
        own = model.state_dict()
        usable = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
        missing = sorted(set(own) - set(usable))
        model.load_state_dict(usable, strict=False)
        print(f"init_from: loaded {len(usable)} tensors, fresh {len(missing)}: {missing[:8]}", flush=True)
    model.to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    enc_named = list(model.encoder.named_parameters())
    head_ids = {id(p) for p in model.head_parameters()}
    groups = [
        {"params": [p for n, p in enc_named if not n.endswith(no_decay)], "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in enc_named if n.endswith(no_decay)], "lr": args.lr, "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": args.head_lr, "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    total_steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total_steps), total_steps)

    # ---------- train ----------
    best = {"acc": -1.0, "epoch": -1}
    best_state = None
    log = open(os.path.join(args.output_dir, "train_log.jsonl"), "w", encoding="utf-8")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=dtype or torch.bfloat16, enabled=use_bf16):
                loss = model.loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            n += 1
        dev_preds = run_eval(model, dev_chunks, dev_sents, pad_id, device, 64, use_bf16, dtype)
        dev_m = evaluate(dev_sents, dev_preds, kin_vocab)
        rec = {"epoch": epoch, "loss": round(running / max(n, 1), 5), "dev_acc": dev_m["acc"],
               "dev_oov_acc": dev_m["oov_acc"], "min": round((time.time() - t0) / 60, 2)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        improved = dev_m["acc"] > best["acc"]
        if improved:
            best = {"acc": dev_m["acc"], "epoch": epoch}
            bad_epochs = 0
            if not args.train_on_dev:
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience and not args.train_on_dev:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    log.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- final eval ----------
    result = {"args": vars(args), "best_dev_epoch": best["epoch"] if not args.train_on_dev else args.epochs}
    dev_preds = run_eval(model, dev_chunks, dev_sents, pad_id, device, 64, use_bf16, dtype)
    result["dev"] = evaluate(dev_sents, dev_preds, kin_vocab)
    write_predictions(os.path.join(args.output_dir, "dev_pred.txt"), dev_sents, dev_preds)
    if test_sents:
        test_preds = run_eval(model, test_chunks, test_sents, pad_id, device, 64, use_bf16, dtype)
        result["test"] = evaluate(test_sents, test_preds, kin_vocab)
        write_predictions(os.path.join(args.output_dir, "test_pred.txt"), test_sents, test_preds)
    result["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    summary = {"dev_acc": result["dev"]["acc"], "dev_oov": result["dev"]["oov_acc"]}
    if test_sents:
        summary.update({"test_acc": result["test"]["acc"], "test_oov": result["test"]["oov_acc"]})
    print("FINAL", json.dumps(summary), flush=True)

    if args.save_model:
        cast = (lambda v: v.half() if v.is_floating_point() else v) if args.save_half else (lambda v: v)
        torch.save({"state_dict": {k: cast(v.cpu()) for k, v in model.state_dict().items()},
                    "char_vocab": char_vocab, "args": vars(args), "tags": UPOS},
                   os.path.join(args.output_dir, "model.pt"))
        print("saved model.pt", flush=True)


if __name__ == "__main__":
    main()
