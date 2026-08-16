#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env_name="${FUNASR_CONDA_ENV:-funasr_hotword}"

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

if ! conda env list | awk -v env="${conda_env_name}" '$1 == env {found=1} END {exit !found}'; then
  conda create -y -n "${conda_env_name}" python=3.12 pip
fi
conda activate "${conda_env_name}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${project_root}/requirements-funasr.txt"
python -m pip install -e "${project_root}"

python - <<'PY'
from funasr import AutoModel
print("FunASR AutoModel import: OK")
PY
echo "Conda environment ready: ${conda_env_name}"
