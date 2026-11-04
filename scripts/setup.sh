#!/usr/bin/env bash
# first-run setup: .env, default model download, doctor. args pass through to `brainscan setup`.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then echo "run scripts/install.sh first" >&2; exit 1; fi
export PYTHONUTF8=1
exec .venv/bin/python -m brainscan setup "$@"
