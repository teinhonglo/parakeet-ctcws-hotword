#!/usr/bin/env bash

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

backend="${BACKEND:-default}"
case "${backend}" in
  default|parakeet|nemotron)
    conda_env_name="${PARAKEET_CONDA_ENV:-parakeet_ctcws}"
    cuda_dir="${PARAKEET_CUDA_DIR:-}"
    ;;
  funasr)
    conda_env_name="${FUNASR_CONDA_ENV:-funasr_hotword}"
    cuda_dir="${FUNASR_CUDA_DIR:-}"
    ;;
  *)
    echo "Unsupported BACKEND: ${backend}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

default_conda_exe="/share/homes/teinhonglo/anaconda3/bin/conda"
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_exe="${CONDA_EXE}"
elif [[ -x "${default_conda_exe}" ]]; then
  conda_exe="${default_conda_exe}"
elif command -v conda >/dev/null 2>&1; then
  conda_exe="$(command -v conda)"
else
  echo "conda was not found at ${default_conda_exe} or on PATH." >&2
  return 1 2>/dev/null || exit 1
fi

eval "$("${conda_exe}" shell.bash hook)"
if ! conda activate "${conda_env_name}"; then
  echo "Failed to activate Conda environment for ${backend}: ${conda_env_name}" >&2
  if [[ "${backend}" == "funasr" ]]; then
    echo "Run: bash scripts/install_funasr.sh" >&2
  else
    echo "Run: bash scripts/install.sh" >&2
  fi
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${cuda_dir}" ]]; then
  if [[ ! -d "${cuda_dir}" ]]; then
    echo "Configured CUDA directory does not exist: ${cuda_dir}" >&2
    return 1 2>/dev/null || exit 1
  fi
  export PATH="${cuda_dir}/bin:${PATH}"
  export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
fi

export CONDA_ENV_NAME="${conda_env_name}"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
