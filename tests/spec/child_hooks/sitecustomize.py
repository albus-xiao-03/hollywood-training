"""Test-only instrumentation for hollywood_training's GUI window process.

Tests put this folder on ``PYTHONPATH`` (see ``helpers.install_child_probe``); Python then
imports this module at start-up in every child process. It does nothing unless the
process is the window process of a test run: ``HT_TEST_CHILD_LOG`` names the log file and
``HOLLYWOOD_TRAINING_AUTHKEY`` (set only for the window process) is present.
Nothing here is part of the package.
"""

import os

if os.environ.get("HT_TEST_CHILD_LOG") and os.environ.get("HOLLYWOOD_TRAINING_AUTHKEY"):
    try:
        import ht_child_probe

        ht_child_probe.install()
    except Exception:  # never stop the window from starting; the test reports it
        import traceback

        traceback.print_exc()
