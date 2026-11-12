#!/bin/bash
# Phase 0: plain fine-tuning baselines, 2 backbones x 5 seeds (paper reports the mean of 5 runs).
#SBATCH --job-name=kinpos-p0
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# 20 GB MIG A100 slices are usually free; large-model fine-tuning on 19k tokens fits in 20 GB
#SBATCH --nodelist=adroit-h11g2
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/network/%u/kinpos/logs/p0-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NETID@princeton.edu
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODELS=(Davlan/afro-xlmr-large Davlan/afro-xlmr-large-76L)
TAGS=(afroxlmr-large afroxlmr-large-76L)
IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))
OUT=$BASE/runs/phase0/${TAGS[$IDX]}/seed$SEED
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$BASE/env/bin/python" "$BASE/code/src/train_tagger.py" \
  --data_dir "$BASE/masakhane-pos/data" --model_name "${MODELS[$IDX]}" --local_files_only \
  --train_langs kin --eval_lang kin --eval_test \
  --epochs 20 --batch_size 32 --lr 5e-5 --seed "$SEED" \
  --output_dir "$OUT"
