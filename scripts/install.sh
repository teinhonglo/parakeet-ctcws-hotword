#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-${project_root}/.venv}"
python_bin="${PYTHON_BIN:-python3}"

"${python_bin}" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python >= 3.12 is required by current NeMo Speech")
print("Python:", sys.version.split()[0])
PY

if [[ ! -d "${venv_dir}" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi

source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

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

echo "Environment ready: ${venv_dir}"
