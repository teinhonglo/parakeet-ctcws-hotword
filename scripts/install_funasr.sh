#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env_name="${FUNASR_CONDA_ENV:-funasr_hotword}"
conda_channel="${FUNASR_CONDA_CHANNEL:-conda-forge}"

torch_index_url="${FUNASR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

# Never read or install packages from ~/.local. A user-site FunASR can mask a
# broken target environment and make the final import check misleading.
export PYTHONNOUSERSITE=1
export PIP_USER=false

default_conda_exe="/share/homes/teinhonglo/anaconda3/bin/conda"
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_exe="${CONDA_EXE}"
elif [[ -x "${default_conda_exe}" ]]; then
  conda_exe="${default_conda_exe}"
elif command -v conda >/dev/null 2>&1; then
  conda_exe="$(command -v conda)"
else
  echo "conda was not found at ${default_conda_exe} or on PATH." >&2
  exit 1
fi
eval "$("${conda_exe}" shell.bash hook)"

# It is safe to invoke this script after `source path.sh`: conda create is not
# affected by packages in the currently active environment, and conda activate
# below switches this shell to the dedicated target before any pip install.
echo "Current Conda environment: ${CONDA_DEFAULT_ENV:-none}"
echo "FunASR target environment: ${conda_env_name}"

if ! conda env list | awk -v env="${conda_env_name}" '$1 == env {found=1} END {exit !found}'; then
  # Do not inherit site/user channels. In particular, an unrelated Intel
  # channel in ~/.condarc can make environment creation fail before FunASR is
  # considered (for example when its TLS certificate chain is unavailable).
  if ! conda create -y -n "${conda_env_name}" \
      --override-channels -c "${conda_channel}" python=3.12 pip; then
    cat >&2 <<EOF
Failed to create ${conda_env_name} from ${conda_channel}.
Inspect inherited Conda configuration with: conda config --show-sources
If every HTTPS channel fails, configure your organization's CA certificate:
  conda config --set ssl_verify /path/to/organization-ca-bundle.pem
Do not disable SSL verification globally.
EOF
    exit 1
  fi
fi
conda activate "${conda_env_name}"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ ! -d "${site_packages}" || ! -w "${site_packages}" ]]; then
  cat >&2 <<EOF
The target environment is not writable: ${site_packages}
Remove and recreate it as the current user:
  conda env remove -n ${conda_env_name}
  bash scripts/install_funasr.sh
EOF
  exit 1
fi
python -m pip install --upgrade pip setuptools wheel

# FunASR's AutoModel imports torch at runtime. Install a CUDA-enabled PyTorch
# build explicitly before the remaining requirements instead of relying on a
# transitive dependency or silently accepting a CPU-only wheel.
if ! python -c 'import torch, torchaudio' >/dev/null 2>&1; then
  python -m pip install torch torchaudio --index-url "${torch_index_url}"
fi

python -m pip install -r "${project_root}/requirements-funasr.txt"
python -m pip install -e "${project_root}"
python -m pip check

python - <<'PY'
import sys
from importlib.metadata import version
import librosa
import openpyxl
import soundfile
import torch
import torchaudio
import transformers
from opencc import OpenCC
from funasr import AutoModel
version("wetext")

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())

print("FunASR inference dependencies: OK")
print("Benchmark evaluation dependencies: OK")
print("FunASR AutoModel import: OK")
PY
python -m hotword_asr.funasr_benchmark --help >/dev/null
echo "Local FunASR benchmark entry point: OK"

echo "Conda environment ready: ${conda_env_name}"
