#!/bin/bash
# One-time setup on the Adroit LOGIN node (it has internet; compute nodes do not).
# Usage: bash /scratch/network/$USER/kinpos/code/slurm/setup_env.sh
set -euo pipefail
BASE=/scratch/network/$USER/kinpos
mkdir -p "$BASE"/{runs,logs,hf,cache/pip,tmp}
export PIP_CACHE_DIR=$BASE/cache/pip TMPDIR=$BASE/tmp HF_HOME=$BASE/hf

# Python: reuse the existing conda Python 3.12 (no module/conda needed), in a separate venv.
PY=/scratch/network/$USER/envs/vllm/bin/python
if [ ! -x "$BASE/env/bin/python" ]; then
  "$PY" -m venv "$BASE/env"
fi
"$BASE/env/bin/python" -m pip install -q --upgrade pip
"$BASE/env/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
"$BASE/env/bin/python" -m pip install -q "transformers>=4.46,<5" sentencepiece protobuf numpy sklearn-crfsuite huggingface_hub "accelerate>=0.26" pyarrow

# Data
if [ ! -d "$BASE/masakhane-pos" ]; then
  git clone --depth 1 https://github.com/masakhane-io/masakhane-pos.git "$BASE/masakhane-pos"
fi

# Model weights into the scratch HF cache (jobs run offline)
"$BASE/env/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Davlan/afro-xlmr-large", "Davlan/afro-xlmr-large-76L"]:
    p = snapshot_download(m, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"])
    print("ok", m, p)
EOF
"$BASE/env/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo SETUP_DONE
