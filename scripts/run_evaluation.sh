#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <evaluate.py> <candidate-dir> <canonical-report> [extra evaluate args...]" >&2
  exit 2
fi

evaluator="$1"
candidate="$2"
report="$3"
shift 3

report_dir="$(dirname "${report}")"
report_name="$(basename "${report}" .xlsx)"
mkdir -p "${report_dir}"
temporary_report="${report_dir}/.${report_name}.current.$$.xlsx"
temporary_json="${report_dir}/.${report_name}.current.$$.per_audio.json"

python "${evaluator}" \
  --candidate "${candidate}" \
  --output "${temporary_report}" \
  --per-audio-json "${temporary_json}" \
  "$@"

canonical_json="${report%.xlsx}.per_audio.json"
if [[ -f "${report}" ]]; then
  backup_stamp="$(date +%Y%m%d_%H%M%S)_$$"
  mv "${report}" "${report%.xlsx}.${backup_stamp}.xlsx"
fi
if [[ -f "${canonical_json}" ]]; then
  backup_stamp="${backup_stamp:-$(date +%Y%m%d_%H%M%S)_$$}"
  mv "${canonical_json}" "${report%.xlsx}.${backup_stamp}.per_audio.json"
fi
mv "${temporary_report}" "${report}"
mv "${temporary_json}" "${canonical_json}"
