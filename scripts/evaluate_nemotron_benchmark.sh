#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <benchmark_dir> <nemotron_exp_dir>" >&2
  exit 2
fi

benchmark_dir="$(cd "$1" && pwd)"
exp_dir="$(cd "$2" && pwd)"

runner="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_evaluation.sh"

bash "${runner}" "${benchmark_dir}/evaluate.py" \
  "${exp_dir}/vanilla/asr" \
  "${exp_dir}/report_vanilla_asr.xlsx"

bash "${runner}" "${benchmark_dir}/evaluate.py" \
  "${exp_dir}/all_hotwords/asr" \
  "${exp_dir}/report_gpu_pb_all_hotwords_asr.xlsx"

bash "${runner}" "${benchmark_dir}/evaluate.py" \
  "${exp_dir}/oracle_hotwords/asr" \
  "${exp_dir}/report_gpu_pb_oracle_hotwords_asr.xlsx"
