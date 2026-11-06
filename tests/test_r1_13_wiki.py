"""R1 requirement 13 - "The package **MUST** have a `wiki` folder provides separated
`markdown` files to explain **EVERY** component of the package **AND** the overall
architecture. The compatibility of `GitHUB wiki` and the `wiki` folder is preferred."

* ``wiki/`` holds separate markdown pages, each with a title;
* GitHub-wiki layout: ``Home.md`` and ``_Sidebar.md``, page links written as ``[text](Page)``,
  every link and image resolving, every page reachable from the sidebar;
* an architecture page covering every subpackage;
* **every component** has its explanation: every subpackage and every module that defines
  something (classes / functions) is named by a page (its ``hollywood_training/...`` path or
  dotted name) - the keyboard-only additions of R2 included - and every public name of
  ``hollywood_training`` appears in the wiki.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from htspec import paths

import hollywood_training as ht

pytestmark = pytest.mark.req("R1.13")

SUBPACKAGES = ("tracking", "training", "sync", "commands", "monitor", "gui")
LINK = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)\)")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def links(text: str) -> list[tuple[str, str]]:
    """``(image mark, target)`` of every markdown link outside code."""
    return LINK.findall(CODE.sub("", text))


def pages() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(paths.WIKI_DIR.glob("*.md"))}


def everything() -> str:
    return "\n".join(pages().values())


def components() -> list[Path]:
    """Modules that define classes or functions (not ``__init__`` / data-only files)."""
    found = []
    for path in sorted(paths.PACKAGE_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body):
            found.append(path)
    return found


def test_the_wiki_folder_holds_separate_markdown_pages():
    assert paths.WIKI_DIR.is_dir()
    content = {name: text for name, text in pages().items() if not name.startswith("_")}
    assert len(content) >= 10
    untitled = [name for name, text in content.items() if not text.lstrip().startswith("# ")]
    assert untitled == []


def test_the_wiki_has_the_github_wiki_layout():
    names = pages()
    assert "Home" in names and "_Sidebar" in names
    assert all(" " not in name for name in names)  # GitHub wiki page names have no spaces


def test_every_wiki_link_and_image_resolves():
    names = set(pages())
    broken = []
    for page, text in pages().items():
        for image, target in links(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            if image or "/" in target or "." in Path(target).name:
                exists = (paths.WIKI_DIR / target).exists()
            else:
                exists = target in names  # a wiki page link: [text](Page-Name)
            if not exists:
                broken.append(f"{page}: {target}")
    assert broken == []


def test_every_page_is_reachable_from_home_or_the_sidebar():
    texts = pages()
    reached, queue = set(), ["Home", "_Sidebar"]
    while queue:
        page = queue.pop()
        if page in reached or page not in texts:
            continue
        reached.add(page)
        queue += [target.split("#", 1)[0].removesuffix(".md") for _i, target in links(texts[page])]
    missing = [name for name in texts if not name.startswith("_") and name not in reached]
    assert missing == []


def test_an_architecture_page_covers_every_subpackage():
    architecture = pages().get("Architecture")
    assert architecture, "no Architecture page"
    lowered = architecture.lower()
    for name in SUBPACKAGES:
        assert name in lowered, name
    assert "process" in lowered and ("epoch" in lowered or "synchron" in lowered)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_subpackage_is_explained(name):
    wiki = everything()
    assert f"hollywood_training/{name}" in wiki or f"hollywood_training.{name}" in wiki, name


@pytest.mark.parametrize(
    "path", components(), ids=lambda p: p.relative_to(paths.PACKAGE_DIR).as_posix()
)
def test_every_component_module_is_explained(path):
    relative = path.relative_to(paths.PROJECT_ROOT).as_posix()
    dotted = relative.removesuffix(".py").replace("/", ".")
    explained = [name for name, text in pages().items() if relative in text or dotted in text]
    assert explained, f"no wiki page explains {relative}"


@pytest.mark.parametrize("name", [n for n in ht.__all__ if n != "__version__"])
def test_every_public_name_appears_in_the_wiki(name):
    assert re.search(rf"\b{re.escape(name)}\b", everything()), name
