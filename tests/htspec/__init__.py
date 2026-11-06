"""Support package of the SPEC harness (``tests/``) - helpers only, no tests.

The harness verifies the package against ``specs/R1.md`` and ``specs/R2.md``. Modules:

* :mod:`.paths`      project locations; every temporary file goes to ``<project>/.temp/``
* :mod:`.env`        environment of the test process and of subprocesses (no real display)
"""
