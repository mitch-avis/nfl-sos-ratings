"""Tests for the all-time strength-of-schedule leaderboard."""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.schedules import main, rank_schedules

if TYPE_CHECKING:
    from pathlib import Path


def _write_ratings(data_dir: Path, season: int, sos: dict[str, float]) -> None:
    """Write a minimal ratings file with the given schedule strengths."""
    pl.DataFrame(
        {
            "team": list(sos),
            "sos": list(sos.values()),
            "team_rating": [0.0] * len(sos),
        }
    ).write_parquet(data_dir / f"{season}_ratings.parquet")


@pytest.fixture
def two_seasons(tmp_path: Path) -> Path:
    """Write two seasons of ratings into a temporary data directory."""
    _write_ratings(tmp_path, 2024, {"AAA": 1.5, "BBB": -2.0, "CCC": 0.5})
    _write_ratings(tmp_path, 2025, {"AAA": -3.0, "BBB": 2.5, "CCC": 0.0})
    return tmp_path


def test_rank_schedules_ranks_every_team_season_softest_first(two_seasons: Path) -> None:
    # Act
    ranked = rank_schedules(two_seasons)

    # Assert
    assert ranked.select("season", "team", "softest_rank").rows()[:2] == [
        (2025, "AAA", 1),
        (2024, "BBB", 2),
    ]


def test_rank_schedules_reports_the_hardest_rank_too(two_seasons: Path) -> None:
    # Act
    ranked = rank_schedules(two_seasons)

    # Assert
    hardest = ranked.filter(pl.col("hardest_rank") == 1).row(0, named=True)
    assert (hardest["season"], hardest["team"]) == (2025, "BBB")


def test_rank_schedules_ignores_files_without_schedule_strength(tmp_path: Path) -> None:
    # Arrange
    _write_ratings(tmp_path, 2025, {"AAA": 1.0, "BBB": -1.0})
    pl.DataFrame({"team": ["AAA"], "SaCR": [1.0]}).write_parquet(tmp_path / "2024_ratings.parquet")

    # Act
    ranked = rank_schedules(tmp_path)

    # Assert
    assert ranked.get_column("season").unique().to_list() == [2025]


def test_main_prints_the_rank_of_one_team_season(
    two_seasons: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    main(["--data-dir", str(two_seasons), "--team", "AAA", "--season", "2025"])

    # Assert
    assert "2025 AAA: sos -3.00, softest 1 of 6" in capsys.readouterr().out
