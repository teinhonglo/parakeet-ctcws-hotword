#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage=1
stop_stage=3
gpuid=0
benchmark_dir="${project_root}/hotword_benchmark"
model="${project_root}/models"
exp_dir="${project_root}/exp/parakeet_ctcws"
limit=""

. ./local/parse_options.sh
. ./path.sh

if (( stage <= 1 && stop_stage >= 1 )); then
  if ! find "${project_root}/models" -type f -name '*.nemo' -print -quit 2>/dev/null | grep -q .; then
    bash "${project_root}/scripts/download_model.sh" "${project_root}/models"
  else
    echo "Stage 1: existing .nemo checkpoint found, skip download"
  fi
fi

if (( stage <= 2 && stop_stage >= 2 )); then
  infer_args=(
    --benchmark-dir "${benchmark_dir}"
    --model "${model}"
    --output-dir "${exp_dir}"
    --device cuda
  )
  if [[ -n "${limit}" ]]; then
    infer_args+=(--limit "${limit}")
  fi
  CUDA_VISIBLE_DEVICES="${gpuid}" python -m hotword_asr.benchmark "${infer_args[@]}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  bash "${project_root}/scripts/evaluate_benchmark.sh" "${benchmark_dir}" "${exp_dir}"
fi
