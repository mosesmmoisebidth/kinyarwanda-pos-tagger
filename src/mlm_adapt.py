"""Continue masked-language-model training of the backbone on Kinyarwanda text (domain adaptation)."""
import argparse
import os

import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


class BlockDataset(torch.utils.data.Dataset):
    """Concatenate tokenized lines and cut into fixed-length blocks."""

    def __init__(self, path, tokenizer, block=256):
        ids = []
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(0, len(lines), 2000):
            enc = tokenizer(lines[i:i + 2000], add_special_tokens=False)["input_ids"]
            for e in enc:
                ids.extend(e + [tokenizer.sep_token_id])
        body = block - 2
        self.blocks = [[tokenizer.cls_token_id] + ids[i:i + body] + [tokenizer.sep_token_id]
                       for i in range(0, len(ids) - body, body)]
        print(f"{path}: {len(ids):,} tokens -> {len(self.blocks):,} blocks of {block}", flush=True)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": self.blocks[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--save_hours", type=float, default=0.5)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    ds = BlockDataset(args.train_file, tok, args.block)
    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="linear", bf16=torch.cuda.is_available(),
        logging_steps=50, save_strategy="steps", save_steps=1000, save_total_limit=1,
        dataloader_num_workers=2, report_to=[], gradient_checkpointing=True, seed=42)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    last = None
    if os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            last = os.path.join(args.output_dir, max(ckpts, key=lambda d: int(d.split("-")[1])))
            print("resuming from", last, flush=True)
    trainer.train(resume_from_checkpoint=last)
    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("SAVED", final, flush=True)


if __name__ == "__main__":
    main()
