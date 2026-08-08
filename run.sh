#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage=0
stop_stage=3
gpuid=0
benchmark_dir="${project_root}/hotword_benchmark"
model="${project_root}/models"
exp_dir="${project_root}/exp/parakeet_ctcws"
limit=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop-stage) stop_stage="$2"; shift 2 ;;
    --gpuid) gpuid="$2"; shift 2 ;;
    --benchmark-dir) benchmark_dir="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --exp-dir) exp_dir="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if (( stage <= 0 && stop_stage >= 0 )); then
  "${project_root}/scripts/install.sh"
fi

source "${project_root}/path.sh"

if (( stage <= 1 && stop_stage >= 1 )); then
  if ! find "${project_root}/models" -type f -name '*.nemo' -print -quit 2>/dev/null | grep -q .; then
    "${project_root}/scripts/download_model.sh" "${project_root}/models"
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
  "${project_root}/scripts/evaluate_benchmark.sh" "${benchmark_dir}" "${exp_dir}"
fi
