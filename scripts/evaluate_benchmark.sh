#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <benchmark_dir> <exp_dir>" >&2
  exit 2
fi

benchmark_dir="$(cd "$1" && pwd)"
exp_dir="$(cd "$2" && pwd)"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${exp_dir}/vanilla/asr" \
  --output "${exp_dir}/report_vanilla_asr.xlsx"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${exp_dir}/all_hotwords/asr" \
  --predicted-keywords "${exp_dir}/all_hotwords/predicted_keywords.json" \
  --output "${exp_dir}/report_ctcws_all_hotwords_asr.xlsx"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${exp_dir}/oracle_hotwords/asr" \
  --predicted-keywords "${exp_dir}/oracle_hotwords/predicted_keywords.json" \
  --output "${exp_dir}/report_ctcws_oracle_hotwords_asr.xlsx"
