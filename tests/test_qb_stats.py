"""Tests for nfl_sos_ratings.qb_stats."""

import polars as pl

from nfl_sos_ratings import qb_stats


def test_compute_qb_season_stats_includes_volume_and_eligibility() -> None:
    """Verify QB season aggregation computes averages, volume, and eligibility flags."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B", "QB_C"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B", "QB C"],
            "qb_attempts": [30, 28, 20, 22, 18],
            "qb_passer_rating": [100.0, 102.0, 92.0, 95.0, 90.0],
            "qb_completion_percentage_above_expectation": [2.0, 3.0, 0.0, 0.5, -1.0],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "points_for": [24, 17, 21, 20, 14],
            "points_allowed": [17, 20, 14, 21, 14],
        }
    )

    result = qb_stats.compute_qb_season_stats(
        qb_df,
        weekly_df=weekly_df,
        min_games=2,
        min_attempts=55,
    )

    qb_a = result.filter(pl.col("qb_id") == "QB_A")
    qb_b = result.filter(pl.col("qb_id") == "QB_B")

    assert qb_a.select("qb_games_played").item() == 2
    assert qb_a.select("qb_attempts_total").item() == 58
    assert qb_a.select("qb_is_eligible").item() is True
    assert qb_a.select("qb_passer_rating").item() == 101.0
    assert qb_a.select("qb_win_pct").item() == 0.5

    assert qb_b.select("qb_games_played").item() == 2
    assert qb_b.select("qb_attempts_total").item() == 42
    assert qb_b.select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_handles_missing_attempts_column() -> None:
    """Verify missing qb_attempts defaults attempts total to zero and marks ineligible."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_passer_rating": [100.0, 98.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(
        qb_df,
        min_games=2,
        min_attempts=1,
    )

    assert result.select("qb_attempts_total").item() == 0
    assert result.select("qb_is_eligible").item() is False
    assert result.select("qb_games_played").item() == 2
