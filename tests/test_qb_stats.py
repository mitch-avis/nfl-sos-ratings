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


def test_compute_qb_season_stats_defaults_to_238_attempt_threshold() -> None:
    """Verify default QB eligibility requires a full-season 238-attempt threshold."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC"],
            "week": [1, 2, 1, 2],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B"],
            "qb_attempts": [119, 119, 119, 118],
            "qb_passer_rating": [100.0, 101.0, 99.0, 98.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df, min_games=2)

    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_is_eligible").item() is True
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_default_eligibility_is_attempt_based() -> None:
    """Verify default qualification does not add a separate games-played cutoff."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN"] * 7,
            "week": list(range(1, 8)),
            "qb_id": ["QB_A"] * 7,
            "qb_name": ["QB A"] * 7,
            "qb_attempts": [34] * 7,
            "qb_passer_rating": [100.0] * 7,
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_games_played").item() == 7
    assert result.select("qb_attempts_total").item() == 238
    assert result.select("qb_is_eligible").item() is True


def test_compute_qb_season_stats_derives_attempt_normalized_rates() -> None:
    """Verify season efficiency rates use total attempts instead of game-average volume."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [10, 30],
            "qb_pass_yards": [100.0, 150.0],
            "qb_pass_touchdowns": [2.0, 1.0],
            "qb_interceptions": [0.0, 2.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_yards_per_attempt").item() == 6.25
    assert result.select("qb_touchdown_rate").item() == 0.075
    assert result.select("qb_interception_rate").item() == 0.05
