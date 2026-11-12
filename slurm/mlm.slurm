#!/bin/bash
# In-domain MLM adaptation of the backbone on IGIHE-first Kinyarwanda news (dev/test overlap removed).
# Resumable: re-submitting continues from the last checkpoint.
#SBATCH --job-name=kinpos-mlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=adroit-h11g2
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/%u/kinpos/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NETID@princeton.edu
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$BASE/code"
[ -f "$BASE/corpus/mlm_kin_60mb.txt" ] || head -c 60000000 "$BASE/corpus/mlm_kin.txt" | sed '$d' > "$BASE/corpus/mlm_kin_60mb.txt"
wc -lc "$BASE/corpus/mlm_kin_60mb.txt"
"$BASE/env/bin/python" src/mlm_adapt.py --model_name Davlan/afro-xlmr-large-76L --local_files_only \
  --train_file "$BASE/corpus/mlm_kin_60mb.txt" --output_dir "$BASE/runs/mlm_76L" \
  --epochs 1 --lr 3e-5 --batch_size 4 --grad_accum 8 --block 256
