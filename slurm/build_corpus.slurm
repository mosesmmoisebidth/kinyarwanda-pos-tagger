#!/bin/bash
# CPU job: build the dev/test-filtered Kinyarwanda MLM corpus + self-training sentences.
#SBATCH --job-name=kinpos-corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/%u/kinpos/logs/%x-%j.out
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
cd "$BASE/code"
mkdir -p "$BASE/corpus"
"$BASE/env/bin/python" src/build_mlm_corpus.py \
  --data_dir "$BASE/masakhane-pos/data" \
  --parquet "$BASE/raw/fineweb2_kin/000_00000.parquet" \
  --extra_text data_raw/unlabeled/igihe_clean.txt \
  --out_mlm "$BASE/corpus/mlm_kin.txt" --out_sents "$BASE/corpus/selftrain_kin.txt" --max_mb 150
wc -l "$BASE/corpus/mlm_kin.txt" "$BASE/corpus/selftrain_kin.txt"
echo CORPUS_DONE
