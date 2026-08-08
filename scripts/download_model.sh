#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${1:-${project_root}/models}"
artifact="${PARAKEET_NGC_ARTIFACT:-nvidia/riva/parakeet-ctc-riva-0-6b-unified-zh-cn:trainable_v3.0}"

if ! command -v ngc >/dev/null 2>&1; then
  echo "ERROR: ngc CLI is not on PATH." >&2
  echo "Run scripts/install_ngc_cli.sh, add its ngc-cli directory to PATH, then run 'ngc config set'." >&2
  exit 2
fi

mkdir -p "${dest}"
echo "Downloading ${artifact}"
ngc registry model download-version "${artifact}" --dest "${dest}"

mapfile -t nemo_files < <(find "${dest}" -type f -name '*.nemo' -print | sort)
if [[ "${#nemo_files[@]}" -eq 0 ]]; then
  echo "ERROR: download completed but no .nemo checkpoint was found under ${dest}" >&2
  exit 3
fi

echo "Found NeMo checkpoint(s):"
for file in "${nemo_files[@]}"; do
  echo "  ${file}"
done

