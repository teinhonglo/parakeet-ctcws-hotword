#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command> [args ...]" >&2
  exit 2
fi

export BACKEND=funasr
. "${project_root}/path.sh"
exec "$@"
