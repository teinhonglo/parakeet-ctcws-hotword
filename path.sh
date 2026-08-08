#!/usr/bin/env bash

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-parakeet_ctcws}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_exe="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  conda_exe="$(command -v conda)"
else
  echo "conda was not found. Install Miniconda/Anaconda and rerun stage 0." >&2
  return 1 2>/dev/null || exit 1
fi

conda_base="$("${conda_exe}" info --base)"
source "${conda_base}/etc/profile.d/conda.sh"
if ! conda activate "${CONDA_ENV_NAME}"; then
  echo "Failed to activate Conda environment: ${CONDA_ENV_NAME}" >&2
  echo "Run: bash scripts/install.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
