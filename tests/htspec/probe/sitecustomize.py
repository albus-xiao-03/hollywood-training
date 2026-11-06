"""SPEC harness probe loader for the Hollywood Training **window process**.

Python imports ``sitecustomize`` at start-up from the first ``sys.path`` entry that has one.
The harness puts this folder on ``PYTHONPATH`` of the processes it starts; the variable is
inherited by the window process that ``ht.run`` spawns. This file then:

1. runs the ``sitecustomize`` it shadows (e.g. the system one), so nothing else changes;
2. if ``HT_SPEC_PROBE_LOG`` is set **and** this is the window process (it alone receives
   ``HOLLYWOOD_TRAINING_AUTHKEY`` from the training process), installs the read-only probe
   of :mod:`ht_spec_probe`.

In every other process it does nothing.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_shadowed_sitecustomize() -> None:
    entries = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", entries)
    if spec is None or spec.loader is None or not spec.origin:
        return
    if os.path.dirname(os.path.abspath(spec.origin)) == _HERE:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # the shadowed file must not break start-up either
        traceback.print_exc()


_chain_shadowed_sitecustomize()

if os.environ.get("HT_SPEC_PROBE_LOG") and os.environ.get("HOLLYWOOD_TRAINING_AUTHKEY"):
    try:
        import ht_spec_probe

        ht_spec_probe.install()
    except Exception:
        traceback.print_exc()
