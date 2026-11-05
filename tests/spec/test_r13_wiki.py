"""R13: a ``wiki`` folder with separate markdown files explaining EVERY component AND
the overall architecture; compatibility with GitHub wikis is preferred."""

from __future__ import annotations

import re

import pytest

import hollywood_training as ht

from .helpers import PACKAGE_DIR, PROJECT_ROOT, WIKI_DIR, package_sources

pytestmark = pytest.mark.spec("R13")

SUBPACKAGES = ("tracking", "training", "sync", "commands", "monitor", "gui")


def pages() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(WIKI_DIR.glob("*.md"))}


def without_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", "", text)


def links(text: str) -> list[str]:
    return [target.split("#")[0] for target in re.findall(r"\]\(([^)\s]+)\)", without_code(text))]


def test_the_wiki_folder_has_a_home_and_an_architecture_page():
    assert WIKI_DIR.is_dir() and WIKI_DIR.parent == PROJECT_ROOT
    names = {name.lower() for name in pages()}
    assert "home" in names
    assert any("architecture" in name for name in names)


def test_the_architecture_page_covers_every_subpackage():
    text = next(
        content for name, content in pages().items() if name.lower() == "architecture"
    ).lower()
    for subpackage in SUBPACKAGES:
        assert subpackage in text


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_every_subpackage_has_its_own_page(subpackage):
    assert any(name.lower().startswith(subpackage) for name in pages()), subpackage


@pytest.mark.parametrize(
    "path",
    [p for p in package_sources() if p.name != "__init__.py"],
    ids=lambda p: str(p.relative_to(PACKAGE_DIR)),
)
def test_every_module_is_explained(path):
    relative = path.relative_to(PACKAGE_DIR).as_posix()
    dotted = "hollywood_training." + relative.removesuffix(".py").replace("/", ".")
    folder = relative.rpartition("/")[0]
    for text in pages().values():
        if relative in text or dotted in text:
            return
        if path.name in text and (not folder or f"{folder}/" in text or f".{folder}" in text):
            return
    pytest.fail(f"no wiki page explains {relative}")


def test_every_public_name_is_documented():
    text = "\n".join(pages().values())
    missing = [name for name in ht.__all__ if not name.startswith("_") and name not in text]
    assert missing == []


def test_pages_are_separate_and_not_empty():
    content = pages()
    assert len(content) >= len(SUBPACKAGES) + 2  # one per component, plus home & architecture
    for name, text in content.items():
        if name.startswith("_"):  # GitHub wiki sidebar / footer
            continue
        assert re.search(r"^# \S", text, re.MULTILINE), f"{name} has no title"
        assert len(without_code(text).split()) >= 30, f"{name} explains almost nothing"


# --------------------------------------------------- GitHub wiki compatibility
def test_internal_links_resolve_the_github_wiki_way():
    """Preferred by the SPEC: page links are page names without '.md' (GitHub wiki style)."""
    names = set(pages())
    problems = []
    for name, text in pages().items():
        for target in links(text):
            if not target or re.match(r"^[a-z]+:", target):
                continue
            if target.endswith(".md") or target.startswith(".."):
                problems.append(f"{name}: {target} (not a wiki page link)")
            elif "/" in target or "." in target:
                if not (WIKI_DIR / target).exists():
                    problems.append(f"{name}: {target} (missing file)")
            elif target not in names:
                problems.append(f"{name}: {target} (no such page)")
    assert problems == []


def test_the_wiki_has_github_sidebar_and_no_orphan_pages():
    content = pages()
    assert "_Sidebar" in content
    linked = {target for text in content.values() for target in links(text)}
    orphans = [name for name in content if not name.startswith("_") and name != "Home"
               and name not in linked]  # fmt: skip
    assert orphans == []
