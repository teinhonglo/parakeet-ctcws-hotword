#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage=1
# The default range reaches Stage 7: one invocation runs all nine experiments.
stop_stage=10000
gpuid=0
benchmark_dir="${project_root}/hotword_benchmark"
model="${project_root}/models"
exp_dir="${project_root}/exp/parakeet_ctcws"
nemotron_model="nvidia/nemotron-3.5-asr-streaming-0.6b"
nemotron_exp_dir="${project_root}/exp/nemotron_gpu_pb"
nemotron_target_lang="zh-CN"
nemotron_boosting_tree_alpha=1.0
nemotron_boosting_context_score=1.0
nemotron_boosting_depth_scaling=2.0
nemotron_boosting_bpe_mode="case_insensitive"
funasr_model="FunAudioLLM/Fun-ASR-Nano-2512"
funasr_exp_dir="${project_root}/exp/funasr_nano"
funasr_language="中文"
funasr_itn=false
funasr_vad_model="fsmn-vad"
funasr_max_single_segment_time=30000
funasr_hub="hf"
funasr_conda_env="funasr_hotword"
overwrite=false
limit=""

# Kaldi's parse_options.sh normally requires an explicit value for booleans.
# Accept the conventional standalone --overwrite requested by this runner too.
normalized_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--overwrite" ]]; then
    normalized_args+=(--overwrite true)
  else
    normalized_args+=("${arg}")
  fi
done
set -- "${normalized_args[@]}"

. "${project_root}/local/parse_options.sh"

if ! [[ "${stage}" =~ ^[0-9]+$ && "${stop_stage}" =~ ^[0-9]+$ ]]; then
  echo "run.sh: --stage and --stop-stage must be non-negative integers" >&2
  exit 2
fi
if (( stage > stop_stage )); then
  echo "run.sh: --stage (${stage}) cannot exceed --stop-stage (${stop_stage})" >&2
  exit 2
fi
if [[ -n "${limit}" ]] && ! [[ "${limit}" =~ ^[1-9][0-9]*$ ]]; then
  echo "run.sh: --limit must be a positive integer" >&2
  exit 2
fi

# Fail before activating the heavyweight runtime when the requested benchmark
# cannot possibly run. This also gives a useful error for misspelled paths.
if (( stage <= 7 && stop_stage >= 2 )); then
  required_benchmark_files=(hotwords.json all_hotwords.json evaluate.py)
  for required_file in "${required_benchmark_files[@]}"; do
    if [[ ! -f "${benchmark_dir}/${required_file}" ]]; then
      echo "run.sh: missing benchmark file: ${benchmark_dir}/${required_file}" >&2
      exit 2
    fi
  done
  if [[ ! -d "${benchmark_dir}/audio" ]]; then
    echo "run.sh: missing benchmark audio directory: ${benchmark_dir}/audio" >&2
    exit 2
  fi
fi

if (( stage <= 5 && stop_stage >= 1 )); then
  . "${project_root}/path.sh"
fi

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
      --condition all
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
      --condition all
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

if (( stage <= 6 && stop_stage >= 6 )); then
  funasr_args=(
    --benchmark-dir "${benchmark_dir}"
    --model "${funasr_model}"
    --output-dir "${funasr_exp_dir}"
    --device cuda
    --condition all
    --language "${funasr_language}"
    --vad-model "${funasr_vad_model}"
    --max-single-segment-time "${funasr_max_single_segment_time}"
    --hub "${funasr_hub}"
  )
  if ${funasr_itn}; then
    funasr_args+=(--itn)
  fi
  if [[ -n "${limit}" ]]; then
    funasr_args+=(--limit "${limit}")
  fi
  if ${overwrite}; then
    funasr_args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${gpuid}" PYTHONPATH="${project_root}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${funasr_conda_env}" \
    python -m hotword_asr.funasr_benchmark "${funasr_args[@]}"
fi

if (( stage <= 7 && stop_stage >= 7 )); then
  conda run --no-capture-output -n "${funasr_conda_env}" \
    bash "${project_root}/scripts/evaluate_funasr_benchmark.sh" \
    "${benchmark_dir}" "${funasr_exp_dir}"
fi

if (( stage <= 2 && stop_stage >= 5 )); then
  cat <<'EOF'
Completed experiments:

Parakeet
  [OK] Vanilla
  [OK] CTC-WS + All Hotwords
  [OK] CTC-WS + Oracle Hotwords

Nemotron
  [OK] Vanilla
  [OK] GPU-PB + All Hotwords
  [OK] GPU-PB + Oracle Hotwords
EOF
fi

if (( stage <= 6 && stop_stage >= 7 )); then
  cat <<'EOF'

Fun-ASR-Nano
  [OK] Vanilla
  [OK] Hotword + All Hotwords
  [OK] Hotword + Oracle Hotwords
EOF
fi
