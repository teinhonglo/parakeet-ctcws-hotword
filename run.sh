#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage=0
# The default range validates the benchmark, runs all nine experiments, and
# prints a single target comparison table.
stop_stage=8
gpuid=0
benchmark_dir="${project_root}/hotword_benchmark"
model="${project_root}/models"
exp_dir="${project_root}/exp/parakeet_ctcws"
parakeet_conda_env="parakeet_ctcws"
parakeet_chunk_seconds=30
parakeet_auto_variants=false
nemotron_model="nvidia/nemotron-3.5-asr-streaming-0.6b"
nemotron_exp_dir="${project_root}/exp/nemotron_gpu_pb"
nemotron_target_lang="zh-CN"
nemotron_boosting_tree_alpha=1.0
nemotron_boosting_context_score=1.0
nemotron_boosting_depth_scaling=2.0
nemotron_boosting_bpe_mode="case_insensitive"
nemotron_chunk_seconds=30
funasr_model="FunAudioLLM/Fun-ASR-Nano-2512"
funasr_exp_dir="${project_root}/exp/funasr_nano"
funasr_language="中文"
funasr_itn=true
funasr_vad_model="funasr/fsmn-vad"
funasr_max_single_segment_time=15000
funasr_batch_size_s=30
funasr_max_length=512
funasr_truncate_repetition=true
funasr_hub="hf"
funasr_conda_env="funasr_hotword"
overwrite=false
limit=""
target_mer=0.15

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
if (( stage <= 7 && stop_stage >= 0 )); then
  required_benchmark_files=(hotwords.json all_hotwords.json pseudo_transcripts.json evaluate.py)
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

export PARAKEET_CONDA_ENV="${parakeet_conda_env}"
export FUNASR_CONDA_ENV="${funasr_conda_env}"

if (( stage <= 0 && stop_stage >= 0 )); then
  PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m hotword_asr.validate_benchmark --benchmark-dir "${benchmark_dir}"
fi

if (( stage <= 1 && stop_stage >= 1 )); then
  if ! find "${project_root}/models" -type f -name '*.nemo' -print -quit 2>/dev/null | grep -q .; then
    bash "${project_root}/run_parakeet.sh" \
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
      --chunk-seconds "${parakeet_chunk_seconds}"
    )
    if ${parakeet_auto_variants}; then
      infer_args+=(--auto-variants)
    fi
    if [[ -n "${limit}" ]]; then
      infer_args+=(--limit "${limit}")
    fi
    if ${overwrite}; then
      infer_args+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="${gpuid}" bash "${project_root}/run_parakeet.sh" \
      python -m hotword_asr.benchmark "${infer_args[@]}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  bash "${project_root}/run_parakeet.sh" \
    bash "${project_root}/scripts/evaluate_benchmark.sh" \
    "${benchmark_dir}" "${exp_dir}"
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
      --chunk-seconds "${nemotron_chunk_seconds}"
    )
    if [[ -n "${limit}" ]]; then
      nemotron_args+=(--limit "${limit}")
    fi
    if ${overwrite}; then
      nemotron_args+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="${gpuid}" \
      bash "${project_root}/run_nemotron.sh" \
      python -m hotword_asr.nemotron_benchmark \
      "${nemotron_args[@]}"
fi

if (( stage <= 5 && stop_stage >= 5 )); then
  bash "${project_root}/run_nemotron.sh" \
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
    --batch-size-s "${funasr_batch_size_s}"
    --max-length "${funasr_max_length}"
    --hub "${funasr_hub}"
  )
  if ${funasr_itn}; then
    funasr_args+=(--itn)
  else
    funasr_args+=(--no-itn)
  fi
  if ${funasr_truncate_repetition}; then
    funasr_args+=(--truncate-repetition)
  else
    funasr_args+=(--no-truncate-repetition)
  fi
  if [[ -n "${limit}" ]]; then
    funasr_args+=(--limit "${limit}")
  fi
  if ${overwrite}; then
    funasr_args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${gpuid}" bash "${project_root}/run_funasr.sh" \
    python -m hotword_asr.funasr_benchmark "${funasr_args[@]}"
fi

if (( stage <= 7 && stop_stage >= 7 )); then
  bash "${project_root}/run_funasr.sh" \
    bash "${project_root}/scripts/evaluate_funasr_benchmark.sh" \
    "${benchmark_dir}" "${funasr_exp_dir}"
fi

if (( stage <= 8 && stop_stage >= 8 )); then
  bash "${project_root}/run_parakeet.sh" \
    python -m hotword_asr.summarize_results \
      --parakeet-dir "${exp_dir}" \
      --nemotron-dir "${nemotron_exp_dir}" \
      --funasr-dir "${funasr_exp_dir}" \
      --target-mer "${target_mer}" \
      --output-json "${project_root}/exp/benchmark_summary.json"
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
