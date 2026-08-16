#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command> [args ...]" >&2
  exit 2
fi

export BACKEND=funasr
. "${project_root}/path.sh"
if ! python -c 'from funasr import AutoModel' >/dev/null 2>&1; then
  cat >&2 <<EOF
FunASR is not importable in the selected environment.
  BACKEND: ${BACKEND}
  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unknown}
  Python: $(command -v python)
Install or repair it with:
  bash ${project_root}/scripts/install_funasr.sh
EOF
  exit 1
fi
exec "$@"
