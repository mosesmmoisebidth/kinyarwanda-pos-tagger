#!/bin/bash
# Run on the LOGIN node (compute nodes have no internet). Fetches the PyTorch weights + tokenizer only.
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
export HF_HOME=$BASE/hf
"$BASE/env/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["FacebookAI/xlm-roberta-large", "google-bert/bert-base-multilingual-cased", "castorini/afriberta_large"]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.model", "vocab.txt", "model.safetensors", "pytorch_model.bin"],
                          ignore_patterns=["onnx/*"])
    print("ok", repo, p, flush=True)
PY
# Smoke test: every backbone + tokenizer loads offline and splits a Kinyarwanda word.
HF_HUB_OFFLINE=1 "$BASE/env/bin/python" - <<'PY'
import torch, transformers
from transformers import AutoTokenizer, AutoModel
print("torch", torch.__version__, "transformers", transformers.__version__)
for repo in ["FacebookAI/xlm-roberta-large", "google-bert/bert-base-multilingual-cased", "castorini/afriberta_large"]:
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModel.from_pretrained(repo, add_pooling_layer=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print("LOADS", repo, type(tok).__name__, f"{n:.0f}M params", tok.tokenize("Abanyarwanda barishimye"), flush=True)
PY
