#!/bin/bash
# Runs every line of $CFG (format: name|model_or_path|extra args) with 5 seeds each.
# Submit:  sbatch --array=0-$((5*NLINES-1)) --export=ALL,CFG=slurm/cfg_phase3.txt,PHASE=phase3 slurm/ablate.slurm
#SBATCH --job-name=kinpos-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=adroit-h11g2
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/network/%u/kinpos/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NETID@princeton.edu
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$BASE/code"

LINE_NO=$((SLURM_ARRAY_TASK_ID / 5 + 1))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))
LINE=$(grep -v '^#' "$CFG" | grep -v '^\s*$' | sed -n "${LINE_NO}p")
NAME=$(echo "$LINE" | cut -d'|' -f1)
MODEL=$(echo "$LINE" | cut -d'|' -f2)
EXTRA=$(echo "$LINE" | cut -d'|' -f3- | sed "s#\$BASE#$BASE#g; s#{SEED}#$SEED#g")
OUT=$BASE/runs/${PHASE:-ablate}/$NAME/seed$SEED
echo "config=$NAME seed=$SEED model=$MODEL extra=$EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086
"$BASE/env/bin/python" src/train_tagger.py \
  --data_dir "$BASE/masakhane-pos/data" --model_name "$MODEL" --local_files_only \
  --eval_lang kin --seed "$SEED" --output_dir "$OUT" $EXTRA
