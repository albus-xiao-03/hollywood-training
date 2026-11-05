#!/usr/bin/env bash
# Run the test harness with every temporary and cache file inside .temp/ (CONSTRICTION.md).
#
#   tests/run_tests.sh                          # everything (benchmarks stay opt-in)
#   tests/run_tests.sh tests/spec               # only the SPEC harness
#   tests/run_tests.sh -m "not e2e and not slow"  # quick pass, no extra processes
#   HT_RUN_BENCHMARKS=1 tests/run_tests.sh -m benchmark
#   QT_QPA_PLATFORM=xcb xvfb-run -a tests/run_tests.sh   # real X11 windows on Xvfb
#
# Extra arguments are passed to pytest unchanged.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp="$root/.temp"
mkdir -p "$temp/tmp" "$temp/xdg-cache"

export TMPDIR="$temp/tmp"                 # tempfile, multiprocessing sockets, torch caches
export XDG_CACHE_HOME="$temp/xdg-cache"   # fontconfig / Qt caches
export PYTHONDONTWRITEBYTECODE=1          # no __pycache__ folders in the project
export ONEDNN_JIT_PROFILE=0               # no /tmp/perf-<pid>.map files from PyTorch
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
    echo "the project virtual environment (.venv) is missing" >&2
    exit 2
fi

cd "$root"
exec "$python" -m pytest \
    -o cache_dir="$temp/pytest-cache" \
    --basetemp="$temp/pytest-basetemp" \
    "$@"
