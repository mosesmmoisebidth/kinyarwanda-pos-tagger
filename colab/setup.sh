#!/bin/bash
# Colab setup: data, deps check, code backup to Drive. Safe to rerun after a disconnect.
set -e
OUT=/content/drive/MyDrive/kinpos_pairs
mkdir -p "$OUT"
[ -d /content/masakhane-pos ] || git clone -q --depth 1 https://github.com/masakhane-io/masakhane-pos /content/masakhane-pos
ls /content/masakhane-pos/data | tr '\n' ' '; echo
python -c "import torch, transformers, sentencepiece; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
[ -f /content/kinpos_colab.tar.gz ] && cp /content/kinpos_colab.tar.gz "$OUT/kinpos_colab.tar.gz"
df -h /content | tail -1
echo SETUP_OK
