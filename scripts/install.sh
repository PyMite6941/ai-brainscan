#!/usr/bin/env bash
# install ai-brainscan into ./.venv  (macOS + Linux)
#   scripts/install.sh [--gpu] [--extras web,tui,whitebox]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=0
EXTRAS="web,tui"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=1 ;;
    --extras) EXTRAS="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "need Python 3.10+ on PATH: https://www.python.org/downloads/" >&2
  exit 1
fi

export PYTHONUTF8=1
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip

if [ "$(uname -s)" = "Darwin" ] || [ "$GPU" = 1 ]; then
  "$VPY" -m pip install torch            # macOS: one wheel (MPS). Linux --gpu: default index = CUDA
else
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$VPY" -m pip install -e ".[${EXTRAS}]"

"$VPY" -m brainscan doctor || true
echo
echo "next: scripts/setup.sh   (creates .env, downloads the default model)"
echo "then: scripts/run.sh     (web UI)   |  scripts/run.sh tui  |  scripts/run.sh --help"
