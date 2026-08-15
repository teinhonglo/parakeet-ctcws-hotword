#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <benchmark_dir> <exp_dir>" >&2
  exit 2
fi

benchmark_dir="$(cd "$1" && pwd)"
exp_dir="$(cd "$2" && pwd)"

ground_truth_dir="${exp_dir}/ground_truth_union"
phrase_boosting_dir="${exp_dir}/phrase_boosting_vocabulary"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${ground_truth_dir}/raw_asr" \
  --output "${exp_dir}/report_raw_asr.xlsx"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${ground_truth_dir}/ctcws_asr" \
  --predicted-keywords "${ground_truth_dir}/predicted_keywords.json" \
  --output "${exp_dir}/report_ctcws_ground_truth_union_asr.xlsx"

python "${benchmark_dir}/evaluate.py" \
  --candidate "${phrase_boosting_dir}/ctcws_asr" \
  --predicted-keywords "${phrase_boosting_dir}/predicted_keywords.json" \
  --output "${exp_dir}/report_ctcws_phrase_boosting_vocabulary_asr.xlsx"
