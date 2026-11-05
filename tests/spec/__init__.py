"""SPEC-derived test harness: one module per SPEC.md requirement (``test_rNN_*``) and GUI
layout item (``test_gN_*``), plus cross-cutting failure and end-to-end suites.

Every test carries ``@pytest.mark.spec(...)`` with the requirement IDs it verifies; see
:mod:`tests.spec.requirements` and ``tests/README.md``.
"""
