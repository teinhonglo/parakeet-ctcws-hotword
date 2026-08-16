#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env_name="${FUNASR_CONDA_ENV:-funasr_hotword}"
conda_channel="${FUNASR_CONDA_CHANNEL:-conda-forge}"

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
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${project_root}/requirements-funasr.txt"
python -m pip install -e "${project_root}"

python - <<'PY'
from funasr import AutoModel
print("FunASR AutoModel import: OK")
PY
echo "Conda environment ready: ${conda_env_name}"
