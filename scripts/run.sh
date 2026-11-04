#!/usr/bin/env bash
# run brainscan from the venv. no args = web UI.
#   scripts/run.sh            web UI on http://127.0.0.1:7860
#   scripts/run.sh tui        terminal UI
#   scripts/run.sh capture "prompt" --label refused
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then echo "run scripts/install.sh first" >&2; exit 1; fi
export PYTHONUTF8=1
if [ $# -eq 0 ]; then set -- web; fi
exec .venv/bin/python -m brainscan "$@"
