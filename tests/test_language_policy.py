"""Policy tests: campaign and process vocabulary stays out of durable surfaces.

Plan labels (stage numbers, block letters, experiment codes) belong only in ``.agents/`` files.
Source, tests, registry text, CLI help, and the reader-facing docs use timeless names.
"""

import re
from pathlib import Path
from typing import cast

import pytest

from nfl_sos_ratings import cli
from nfl_sos_ratings.metrics import get_registry

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DURABLE_DOCS = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "methodology.md",
    _REPO_ROOT / "docs" / "stats-catalog.md",
    _REPO_ROOT / "docs" / "qb-stats-catalog.md",
)
_BANNED_PATTERNS = (
    re.compile(r"\bStage\s+\d+[a-z]?\b"),
    re.compile(r"\bBlock\s+[A-Z0-9]{1,3}\b"),
    re.compile(r"\bT[124]Weighted\b"),
    re.compile(r"\bQ[12]\b"),
    re.compile(r"\bD[1-7]\b"),
    re.compile(r"\bpreregister(?:ed|ing|s)?\b", re.IGNORECASE),
    re.compile(r"\brefreez(?:e|ed|es|ing)\b|\brefroze(?:n)?\b", re.IGNORECASE),
    re.compile(r"\boverhaul\b", re.IGNORECASE),
    re.compile(r"\bcutover\b", re.IGNORECASE),
)


def _banned_terms(text: str) -> list[str]:
    """Return the sorted distinct banned-term matches in ``text``."""
    return sorted(
        {match.group(0) for pattern in _BANNED_PATTERNS for match in pattern.finditer(text)}
    )


def _strings(value: object) -> list[str]:
    """Return every string nested in a JSON-like payload."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        children: list[object] = list(cast("dict[object, object]", value).values())
    elif isinstance(value, list):
        children = list(cast("list[object]", value))
    else:
        return []
    return [text for child in children for text in _strings(child)]


def _hits(paths: list[Path]) -> dict[str, list[str]]:
    """Return banned terms per file, for files that have any."""
    return {
        path.relative_to(_REPO_ROOT).as_posix(): found
        for path in paths
        if (found := _banned_terms(path.read_text(encoding="utf-8")))
    }


def test_python_sources_avoid_campaign_terms() -> None:
    # Arrange
    sources = [
        path
        for root in ("nfl_sos_ratings", "tests")
        for path in sorted((_REPO_ROOT / root).rglob("*.py"))
        if path != Path(__file__).resolve()
    ]

    # Act
    hits = _hits(sources)

    # Assert
    assert hits == {}


def test_durable_docs_avoid_campaign_terms() -> None:
    # Act
    hits = _hits(list(_DURABLE_DOCS))

    # Assert
    assert hits == {}


def test_registry_text_avoids_campaign_terms() -> None:
    # Arrange
    text = "\n".join(_strings(get_registry().payload()))

    # Act
    hits = _banned_terms(text)

    # Assert
    assert hits == []


@pytest.mark.parametrize("command", [command.name for command in cli.COMMANDS])
def test_command_help_avoids_campaign_terms(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    help_text = capsys.readouterr().out

    # Act
    hits = _banned_terms(help_text)

    # Assert
    assert hits == []
