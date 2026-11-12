"""Run every training job for the model-pair study, one after another, on a single GPU (Colab T4).

Resumable: a job whose metrics.json already exists in Drive is skipped, so after a Colab
disconnect you rerun the setup cell and start this script again.

Layout under OUT (Google Drive):
  runs/<recipe>/<model>/seed<k>/   metrics.json, dev_pred.txt, test_pred.txt, train_log.jsonl
  stage1/<model>/                  multilingual Stage 1 checkpoint (fp16) used by the "full" recipe
  best/<recipe>/<model>/           checkpoint of the seed with the best DEV accuracy (fp16)
  logs/                            one log per job;  status.json  live progress

python colab/run_queue.py --code /content/kinpos --data /content/masakhane-pos/data --out /content/drive/MyDrive/kinpos_pairs
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# Pairs: (general multilingual, Afro-centric). The main pair shares the XLM-R large architecture,
# so the only difference is the extra pretraining on African languages.
MODELS = {
    "afroxlmr-76L": "Davlan/afro-xlmr-large-76L",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "afriberta-large": "castorini/afriberta_large",
    "mbert": "google-bert/bert-base-multilingual-cased",
}
ORDER = ["afroxlmr-76L", "xlmr-large", "afriberta-large", "mbert"]

# Bantu languages (closest to Kinyarwanda) weighted up; others contribute train+dev+test, Kinyarwanda train only.
LANGS = "kin:3,sna:2,lug:2,nya:2,swa:2,xho:2,zul:2,tsn:2,bam:1,bbj:1,ewe:1,fon:1,hau:1,ibo:1,luo:1,mos:1,pcm:1,twi:1,wol:1,yor:1"

PLAIN = "--train_langs kin --eval_test --epochs 20 --batch_size 32 --lr 5e-5"
STAGE1 = (f"--train_langs {LANGS} --aux_splits train,dev,test --use_char --use_affix --use_crf --head_hidden 512 "
          "--epochs 6 --steps_per_epoch 800 --batch_size 16 --max_len 256 --lr 5e-5 --patience 2 --seed 1")
# T4 (15 GB): multilingual batches hold long sentences, so Stage 1 uses batch 16 x 800 steps (same sentences per
# epoch as 32 x 400) and 256-subword chunks; Stage 2 also uses 256-subword chunks.
FULL = ("--train_langs kin --init_from {init} --use_char --use_affix --use_crf --head_hidden 512 "
        "--eval_test --epochs 10 --batch_size 32 --max_len 256 --lr 2e-5")


def job_list(out):
    """Ordered so that a partial run still gives a complete picture: 3 seeds of everything first."""
    jobs = []
    plain = lambda seeds: [("plain", m, s) for s in seeds for m in ORDER]
    full = lambda seeds: [("full", m, s) for s in seeds for m in ORDER]
    jobs += plain([1, 2, 3])
    jobs += [("stage1", m, 1) for m in ORDER]
    jobs += full([1, 2, 3])
    jobs += plain([4, 5])
    jobs += full([4, 5])
    return jobs


def run_dir(out, recipe, model, seed):
    if recipe == "stage1":
        return os.path.join(out, "stage1", model)
    return os.path.join(out, "runs", recipe, model, f"seed{seed}")


def write_status(out, status):
    tmp = os.path.join(out, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1)
    os.replace(tmp, os.path.join(out, "status.json"))


def keep_if_best(out, recipe, model, work_dir, metrics):
    """Keep only the checkpoint of the best-DEV seed per (recipe, model); test plays no part."""
    best_dir = os.path.join(out, "best", recipe, model)
    best_m = os.path.join(best_dir, "metrics.json")
    old = json.load(open(best_m, encoding="utf-8"))["dev"]["acc"] if os.path.isfile(best_m) else -1
    if metrics["dev"]["acc"] > old:
        os.makedirs(best_dir, exist_ok=True)
        shutil.copy(os.path.join(work_dir, "model.pt"), os.path.join(best_dir, "model.pt.part"))
        os.replace(os.path.join(best_dir, "model.pt.part"), os.path.join(best_dir, "model.pt"))
        shutil.copy(os.path.join(work_dir, "metrics.json"), best_m)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--only", default="", help="comma list of recipes to run (plain,stage1,full)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    jobs = job_list(args.out)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    status = {"total": len(jobs), "done": [], "failed": [], "current": None, "started": time.strftime("%H:%M:%S")}
    for recipe, model, seed in jobs:
        dest = run_dir(args.out, recipe, model, seed)
        name = f"{recipe}/{model}/seed{seed}"
        done_file = os.path.join(dest, "model.pt" if recipe == "stage1" else "metrics.json")
        if os.path.isfile(done_file):
            status["done"].append({"job": name, "skipped": True})
            continue
        if recipe == "full" and not os.path.isfile(os.path.join(args.out, "stage1", model, "model.pt")):
            status["failed"].append({"job": name, "why": "no stage1 checkpoint"})
            continue
        work = os.path.join(args.work, recipe, model, f"seed{seed}")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        if recipe == "plain":
            extra = PLAIN
        elif recipe == "stage1":
            extra = STAGE1
        else:
            extra = FULL.format(init=os.path.join(args.out, "stage1", model, "model.pt"))
        cmd = [sys.executable, os.path.join(args.code, "src", "train_tagger.py"), "--data_dir", args.data,
               "--model_name", MODELS[model], "--eval_lang", "kin", "--output_dir", work,
               "--save_model", "--save_half", "--seed", str(seed)] + extra.split()
        status["current"] = {"job": name, "since": time.strftime("%H:%M:%S")}
        write_status(args.out, status)
        print(f"[{time.strftime('%H:%M:%S')}] START {name}", flush=True)
        t0 = time.time()
        log_path = os.path.join(args.out, "logs", name.replace("/", "_") + ".log")
        with open(log_path, "w", encoding="utf-8") as log:
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        minutes = round((time.time() - t0) / 60, 1)
        mfile = os.path.join(work, "metrics.json")
        if rc != 0 or not os.path.isfile(mfile):
            tail = open(log_path, encoding="utf-8", errors="replace").read()[-600:]
            status["failed"].append({"job": name, "rc": rc, "min": minutes, "tail": tail})
            print(f"FAILED {name} rc={rc}\n{tail}", flush=True)
            write_status(args.out, status)
            continue
        m = json.load(open(mfile, encoding="utf-8"))
        os.makedirs(dest, exist_ok=True)
        rec = {"job": name, "min": minutes, "dev": m["dev"]["acc"], "dev_oov": m["dev"].get("oov_acc")}
        if recipe == "stage1":
            shutil.copy(os.path.join(work, "model.pt"), os.path.join(dest, "model.pt.part"))
            for f in ("metrics.json", "dev_pred.txt", "train_log.jsonl"):
                shutil.copy(os.path.join(work, f), os.path.join(dest, f))
            os.replace(os.path.join(dest, "model.pt.part"), os.path.join(dest, "model.pt"))
        else:
            rec["best_so_far"] = keep_if_best(args.out, recipe, model, work, m)
            rec.update({"test": m["test"]["acc"], "test_oov": m["test"].get("oov_acc")})
            for f in ("dev_pred.txt", "test_pred.txt", "train_log.jsonl", "metrics.json"):  # metrics last = done
                shutil.copy(os.path.join(work, f), os.path.join(dest, f))
        shutil.rmtree(work, ignore_errors=True)
        status["done"].append(rec)
        status["current"] = None
        write_status(args.out, status)
        print(f"[{time.strftime('%H:%M:%S')}] DONE {json.dumps(rec)}", flush=True)
    status["finished"] = time.strftime("%H:%M:%S")
    write_status(args.out, status)
    print("ALL JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
