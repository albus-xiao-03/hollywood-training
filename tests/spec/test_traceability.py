"""Every SPEC requirement is verified by at least one test, and every test names real
requirements (see ``requirements.py``; the run summary prints the coverage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from .requirements import MANUAL_CHECKS, REQUIREMENTS

HERE = Path(__file__).resolve().parent


def _spec_items(session: pytest.Session) -> list[pytest.Item]:
    """Collected requirement tests (this meta module itself excluded)."""
    return [
        item
        for item in session.items
        if Path(str(item.path)).resolve().parent == HERE
        and Path(str(item.path)).name != Path(__file__).name
    ]


def _ids(item: pytest.Item) -> set[str]:
    return {str(arg) for marker in item.iter_markers(name="spec") for arg in marker.args}


def test_every_test_names_known_requirements(request):
    unknown = {
        item.nodeid: sorted(_ids(item) - set(REQUIREMENTS))
        for item in _spec_items(request.session)
        if _ids(item) - set(REQUIREMENTS)
    }
    assert unknown == {}


def test_every_test_is_traced_to_a_requirement(request):
    untraced = [item.nodeid for item in _spec_items(request.session) if not _ids(item)]
    assert untraced == []


def test_every_requirement_is_covered(request):
    if request.config.getoption("markexpr") or request.config.getoption("keyword"):
        pytest.skip("tests were selected with -m / -k")
    modules = {Path(str(item.path)).name for item in _spec_items(request.session)}
    expected = {path.name for path in HERE.glob("test_*.py")} - {Path(__file__).name}
    if modules != expected:
        pytest.skip("only part of the SPEC harness was collected")
    covered = set().union(*(_ids(item) for item in _spec_items(request.session)))
    assert sorted(set(REQUIREMENTS) - covered) == []


def test_manual_checks_refer_to_known_requirements():
    assert set(MANUAL_CHECKS) - {"ALL"} <= set(REQUIREMENTS)
