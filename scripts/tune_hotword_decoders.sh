#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: $0 <benchmark-dir> <parakeet-model> <dev-ids-file> <output-dir> [gpuid]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_dir="$(cd "$1" && pwd)"
parakeet_model="$2"
dev_ids_file="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
output_dir="$4"
gpuid="${5:-0}"
nemotron_model="${NEMOTRON_MODEL:-nvidia/nemotron-3.5-asr-streaming-0.6b}"

if [[ ! -f "${dev_ids_file}" ]]; then
  echo "Missing development ID file: ${dev_ids_file}" >&2
  exit 2
fi
mkdir -p "${output_dir}/parakeet" "${output_dir}/nemotron"

read -r -a beam_thresholds <<< "${PARAKEET_BEAM_THRESHOLDS:-7 8 9}"
read -r -a context_scores <<< "${PARAKEET_CONTEXT_SCORES:-3 4 5}"
read -r -a alignment_weights <<< "${PARAKEET_ALIGNMENT_WEIGHTS:-0.5 0.6 0.7}"
read -r -a nemotron_alphas <<< "${NEMOTRON_ALPHAS:-0.5 1 2 4}"

for beam in "${beam_thresholds[@]}"; do
  for context in "${context_scores[@]}"; do
    for alignment in "${alignment_weights[@]}"; do
      name="beam_${beam}_context_${context}_alignment_${alignment}"
      experiment="${output_dir}/parakeet/${name}"
      CUDA_VISIBLE_DEVICES="${gpuid}" bash "${project_root}/run_parakeet.sh" \
        python -m hotword_asr.benchmark \
          --benchmark-dir "${benchmark_dir}" \
          --model "${parakeet_model}" \
          --output-dir "${experiment}" \
          --condition all-hotwords \
          --audio-ids-file "${dev_ids_file}" \
          --beam-threshold "${beam}" \
          --context-score "${context}" \
          --ctc-ali-token-weight "${alignment}" \
          --overwrite
      bash "${project_root}/run_parakeet.sh" \
        python -m hotword_asr.subset_score \
          --benchmark-dir "${benchmark_dir}" \
          --candidate-dir "${experiment}/all_hotwords/asr" \
          --audio-ids-file "${dev_ids_file}" \
          --output "${experiment}/dev_score.json"
    done
  done
done

for alpha in "${nemotron_alphas[@]}"; do
  name="alpha_${alpha}"
  experiment="${output_dir}/nemotron/${name}"
  CUDA_VISIBLE_DEVICES="${gpuid}" bash "${project_root}/run_nemotron.sh" \
    python -m hotword_asr.nemotron_benchmark \
      --benchmark-dir "${benchmark_dir}" \
      --model "${nemotron_model}" \
      --output-dir "${experiment}" \
      --condition all-hotwords \
      --audio-ids-file "${dev_ids_file}" \
      --boosting-tree-alpha "${alpha}" \
      --overwrite
  bash "${project_root}/run_nemotron.sh" \
    python -m hotword_asr.subset_score \
      --benchmark-dir "${benchmark_dir}" \
      --candidate-dir "${experiment}/all_hotwords/asr" \
      --audio-ids-file "${dev_ids_file}" \
      --output "${experiment}/dev_score.json"
done

bash "${project_root}/run_parakeet.sh" \
  python -m hotword_asr.rank_tuning \
    --root "${output_dir}/parakeet" \
    --output "${output_dir}/parakeet/tuning_summary.json"
bash "${project_root}/run_parakeet.sh" \
  python -m hotword_asr.rank_tuning \
    --root "${output_dir}/nemotron" \
    --output "${output_dir}/nemotron/tuning_summary.json"
