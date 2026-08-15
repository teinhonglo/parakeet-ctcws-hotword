#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env_name="${CONDA_ENV_NAME:-parakeet_ctcws}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_exe="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  conda_exe="$(command -v conda)"
else
  echo "conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

conda_base="$("${conda_exe}" info --base)"
source "${conda_base}/etc/profile.d/conda.sh"

if conda env list | awk -v env="${conda_env_name}" '$1 == env {found=1} END {exit !found}'; then
  echo "Conda environment exists, reuse: ${conda_env_name}"
else
  conda create -y -n "${conda_env_name}" python=3.12 pip
fi

conda activate "${conda_env_name}"

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python >= 3.12 is required by current NeMo Speech")
print("Python:", sys.version.split()[0])
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install Cython packaging

if ! python -c 'import torch, torchaudio' >/dev/null 2>&1; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    python -m pip install "torch>=2.7" "torchaudio>=2.7" --index-url "${TORCH_INDEX_URL}"
  else
    python -m pip install "torch>=2.7" "torchaudio>=2.7"
  fi
fi

python -m pip install -r "${project_root}/requirements.txt"
python -m pip install -e "${project_root}"

python - <<'PY'
import torch
from nemo.collections.asr.parts import context_biasing
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NeMo CTC-WS import: OK")
PY

echo "Conda environment ready: ${conda_env_name}"
