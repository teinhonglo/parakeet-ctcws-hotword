#!/usr/bin/env bash

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"

