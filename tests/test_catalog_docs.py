"""Tests for the stats catalogs generated from the metric registry."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nfl_sos_ratings.metrics.catalog_docs import CATALOG_PATHS, render_catalog

if TYPE_CHECKING:
    from nfl_sos_ratings.metrics.schema import Entity

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_render_catalog_lists_the_headline_team_rating() -> None:
    # Act
    text = render_catalog("team")

    # Assert
    assert "| `team_rating` | Team Rating |" in text


def test_render_catalog_lists_qb_metrics_under_their_category() -> None:
    # Act
    text = render_catalog("qb")

    # Assert
    assert text.index("## Rushing") < text.index("| `qb_designed_carries` |")


def test_render_catalog_marks_itself_as_generated() -> None:
    # Act
    text = render_catalog("team")

    # Assert
    assert "nfl-sos-ratings catalog" in text.splitlines()[2]


@pytest.mark.parametrize(("entity", "relative_path"), CATALOG_PATHS.items())
def test_committed_catalog_matches_the_registry(entity: Entity, relative_path: str) -> None:
    # Arrange
    committed = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    # Act
    generated = render_catalog(entity)

    # Assert
    assert committed == generated, f"run `nfl-sos-ratings catalog` to refresh {relative_path}"
