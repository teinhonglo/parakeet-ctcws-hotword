#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage=1
stop_stage=3000
gpuid=0
benchmark_dir="${project_root}/hotword_benchmark"
model="${project_root}/models"
exp_dir="${project_root}/exp/parakeet_ctcws"
nemotron_model="nvidia/nemotron-3.5-asr-streaming-0.6b"
nemotron_exp_dir="${project_root}/exp/nemotron_gpu_pb"
nemotron_target_lang="zh-CN"
vocabulary_source="ground-truth-union"
nemotron_boosting_tree_alpha=1.0
nemotron_boosting_context_score=1.0
nemotron_boosting_depth_scaling=2.0
nemotron_boosting_bpe_mode="case_insensitive"
overwrite=false
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
    --vocabulary-source "${vocabulary_source}"
  )
  if [[ -n "${limit}" ]]; then
    infer_args+=(--limit "${limit}")
  fi
  if ${overwrite}; then
    infer_args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${gpuid}" python -m hotword_asr.benchmark "${infer_args[@]}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  bash "${project_root}/scripts/evaluate_benchmark.sh" "${benchmark_dir}" "${exp_dir}"
fi

if (( stage <= 4 && stop_stage >= 4 )); then
  nemotron_args=(
    --benchmark-dir "${benchmark_dir}"
    --model "${nemotron_model}"
    --output-dir "${nemotron_exp_dir}"
    --device cuda
    --target-lang "${nemotron_target_lang}"
    --vocabulary-source "${vocabulary_source}"
    --boosting-tree-alpha "${nemotron_boosting_tree_alpha}"
    --boosting-context-score "${nemotron_boosting_context_score}"
    --boosting-depth-scaling "${nemotron_boosting_depth_scaling}"
    --boosting-bpe-mode "${nemotron_boosting_bpe_mode}"
  )
  if [[ -n "${limit}" ]]; then
    nemotron_args+=(--limit "${limit}")
  fi
  if ${overwrite}; then
    nemotron_args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${gpuid}" \
    python -m hotword_asr.nemotron_benchmark "${nemotron_args[@]}"
fi

if (( stage <= 5 && stop_stage >= 5 )); then
  bash "${project_root}/scripts/evaluate_nemotron_benchmark.sh" \
    "${benchmark_dir}" "${nemotron_exp_dir}"
fi
