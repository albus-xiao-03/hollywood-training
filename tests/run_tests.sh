#!/usr/bin/env bash
# Run the SPEC harness of hollywood-training (specs/R1.md + specs/R2.md).
#
#   tests/run_tests.sh                                   # everything
#   tests/run_tests.sh -m "not x11"                      # without the real-X11 tests
#   tests/run_tests.sh -m "not x11 and not subprocess and not slow"   # the quick part
#   tests/run_tests.sh tests/test_r2_7_keyboard_reroute.py            # one requirement
#
# Temporary files go to .temp/ (CONSTRICTION.md). Qt renders offscreen; the x11 tests use a
# private Xvfb server the harness starts itself - the display this is started from is never
# used. Extra arguments are passed to pytest unchanged.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(dirname "$here")"
cd "$project"

if [ -f "$project/.temp/env_setup/devenv.sh" ]; then
    # the project's development environment: venv + every cache inside .temp/
    # shellcheck source=/dev/null
    source "$project/.temp/env_setup/devenv.sh"
elif [ -f "$project/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$project/.venv/bin/activate"
    mkdir -p "$project/.temp/tmp"
    export TMPDIR="$project/.temp/tmp"
    export PYTHONPYCACHEPREFIX="$project/.temp/pycache"
    export PYTEST_ADDOPTS="-o cache_dir=$project/.temp/pytest-cache ${PYTEST_ADDOPTS:-}"
fi

# with pytest-cov (--cov): coverage's data file, by default <project>/.coverage, goes to .temp/
export COVERAGE_FILE="${COVERAGE_FILE:-$project/.temp/coverage/.coverage}"
mkdir -p "$(dirname "$COVERAGE_FILE")"

exec python -m pytest "$@"
