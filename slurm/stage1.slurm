#!/bin/bash
# Stage 1: multilingual POS training on all 20 MasakhaPOS languages (Bantu weighted up).
# Other languages contribute train+dev+test; Kinyarwanda contributes train only, kin dev picks the epoch.
# MODEL can be overridden: sbatch --export=ALL,MODEL=/path/to/mlm/final,NAME=stage1_mlm slurm/stage1.slurm
#SBATCH --job-name=kinpos-s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=adroit-h11g2
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/%u/kinpos/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NETID@princeton.edu
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$BASE/code"
MODEL=${MODEL:-Davlan/afro-xlmr-large-76L}
NAME=${NAME:-stage1_base}
LANGS="kin:3,sna:2,lug:2,nya:2,swa:2,xho:2,zul:2,tsn:2,bam:1,bbj:1,ewe:1,fon:1,hau:1,ibo:1,luo:1,mos:1,pcm:1,twi:1,wol:1,yor:1"
"$BASE/env/bin/python" src/train_tagger.py \
  --data_dir "$BASE/masakhane-pos/data" --model_name "$MODEL" --local_files_only \
  --train_langs "$LANGS" --aux_splits train,dev,test --eval_lang kin \
  --use_char --use_affix --use_crf --head_hidden 512 \
  --epochs 8 --steps_per_epoch 400 --batch_size 32 --lr 5e-5 --patience 3 --seed 1 \
  --output_dir "$BASE/runs/$NAME" --save_model
