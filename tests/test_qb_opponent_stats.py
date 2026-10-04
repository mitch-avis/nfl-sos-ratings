"""Tests for the head-to-head-excluded QB opponent profiles."""

import polars as pl
import pytest

from nfl_sos_ratings import qb_opponent_stats


def test_compute_qb_opponent_profiles_excludes_head_to_head() -> None:
    """Verify QB opponent profiles exclude games against the evaluated QB's team."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["KC", "KC", "LAC", "LAC", "DEN", "DEN", "BUF", "LV"],
            "opponent_team": ["DEN", "BUF", "DEN", "LV", "KC", "LAC", "KC", "LAC"],
            "week": [1, 2, 1, 2, 1, 1, 2, 2],
            "points_allowed": [24, 14, 20, 21, 17, 20, 14, 21],
            "def_sacks": [2, 4, 1, 3, 3, 2, 4, 3],
            "def_interceptions": [0, 2, 1, 1, 1, 1, 2, 1],
            "def_pass_defended": [4, 7, 3, 5, 6, 5, 7, 5],
            "def_tackles_for_loss": [5, 8, 4, 6, 7, 6, 8, 6],
            "def_qb_hits": [6, 9, 5, 7, 8, 7, 9, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "BUF", "LV"],
            "week": [1, 2, 2],
            "qb_passer_rating": [100.0, 85.0, 90.0],
            "qb_completion_percentage_above_expectation": [2.0, -1.0, 0.5],
            "qb_aggressiveness": [11.0, 8.5, 9.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "LAC"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
            "qb_passer_rating": [100.0],
        }
    )

    qb_df = qb_df.with_columns(pl.col("team_abbr").replace({"DEN": "QB_DEN"}).alias("qb_id"))
    qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_name"))

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.columns[0] == "qb_id"
    assert "qopp_points_allowed" in profiles.columns
    assert "qopp_qb_passer_rating" in profiles.columns
    assert profiles.select("qopp_points_allowed").item() == 17.5
    assert profiles.select("qopp_qb_passer_rating").item() == 87.5
    assert details["QB_DEN"][0]["games_included"] == 1


def test_compute_qb_opponent_profiles_counts_actual_games_faced() -> None:
    """Verify repeated opponents are deduplicated before QB opponent averaging."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "DEN", "MIA", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "KC", "BUF", "KC", "BUF", "MIA", "MIA"],
            "week": [1, 2, 3, 4, 5, 4, 5],
            "points_allowed": [20, 24, 21, 30, 10, 14, 21],
            "def_sacks": [2, 3, 1, 5, 1, 4, 2],
            "def_interceptions": [1, 1, 0, 3, 0, 2, 1],
            "def_pass_defended": [4, 5, 3, 8, 2, 7, 4],
            "def_tackles_for_loss": [5, 6, 4, 9, 3, 8, 5],
            "def_qb_hits": [6, 7, 5, 10, 4, 9, 6],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "DEN", "MIA", "MIA"],
            "week": [1, 2, 3, 4, 5],
            "qb_id": ["QB_DEN", "QB_DEN", "QB_DEN", "QB_MIA", "QB_MIA"],
            "qb_name": ["Denver QB", "Denver QB", "Denver QB", "Miami QB", "Miami QB"],
            "qb_passer_rating": [100.0, 99.0, 98.0, 120.0, 60.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["DEN", "DEN", "DEN"],
            "away_team": ["KC", "KC", "BUF"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("qopp_qb_passer_rating").item() == 90.0
    assert [row["opponent"] for row in details["QB_DEN"]] == ["KC", "BUF"]


def test_compute_qb_opponent_profiles_skips_qb_without_reconstructable_opponents() -> None:
    """Verify QB opponent profiles do not fabricate schedules from team schedules."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "DEN", "KC", "KC", "KC", "BUF", "BUF", "BUF"],
            "opponent_team": ["KC", "KC", "BUF", "DEN", "DEN", "BUF", "KC", "KC", "DEN"],
            "week": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "points_allowed": [20, 24, 21, 30, 10, 14, 21, 17, 28],
            "def_sacks": [2, 3, 1, 5, 1, 4, 2, 3, 2],
            "def_interceptions": [1, 1, 0, 3, 0, 2, 1, 1, 1],
            "def_pass_defended": [4, 5, 3, 8, 2, 7, 4, 5, 5],
            "def_tackles_for_loss": [5, 6, 4, 9, 3, 8, 5, 6, 6],
            "def_qb_hits": [6, 7, 5, 10, 4, 9, 6, 7, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["BUF", "BUF", "KC"],
            "week": [1, 2, 3],
            "qb_id": ["QB_BUF1", "QB_BUF2", "QB_KC"],
            "qb_name": ["Buffalo 1", "Buffalo 2", "KC QB"],
            "qb_passer_rating": [80.0, 90.0, 70.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["DEN", "KC", "DEN"],
            "away_team": ["KC", "DEN", "BUF"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
            "qb_passer_rating": [100.0],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is None
    assert details["QB_DEN"] == []


def test_compute_qb_opponent_profiles_uses_each_qbs_actual_opponents() -> None:
    """Verify same-team QBs are profiled from the opponents each QB actually faced."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "BUF", "LV", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "BUF", "DEN", "DEN", "KC", "BUF", "LV", "MIA"],
            "week": [1, 2, 1, 2, 3, 3, 3, 3],
            "points_allowed": [20, 24, 21, 17, 14, 28, 14, 28],
            "def_sacks": [2, 3, 1, 4, 5, 2, 5, 2],
            "def_interceptions": [1, 1, 0, 2, 3, 1, 3, 1],
            "def_pass_defended": [4, 5, 3, 7, 8, 5, 8, 5],
            "def_tackles_for_loss": [5, 6, 4, 8, 9, 6, 9, 6],
            "def_qb_hits": [6, 7, 5, 9, 10, 7, 10, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "LV", "MIA"],
            "week": [1, 2, 3, 3],
            "qb_id": ["QB_DEN_1", "QB_DEN_2", "QB_LV", "QB_MIA"],
            "qb_name": ["Denver QB 1", "Denver QB 2", "Raiders QB", "Miami QB"],
            "qb_passer_rating": [100.0, 95.0, 70.0, 110.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN_1", "QB_DEN_2"],
            "qb_name": ["Denver QB 1", "Denver QB 2"],
            "team": ["DEN", "DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    profile_values = profiles.sort("qb_id").select("qopp_qb_passer_rating").to_series().to_list()
    assert profile_values == [70.0, 110.0]
    assert [row["opponent"] for row in details["QB_DEN_1"]] == ["KC"]
    assert [row["opponent"] for row in details["QB_DEN_2"]] == ["BUF"]


def test_compute_qb_opponent_profiles_uses_majority_snap_games_only() -> None:
    """Verify QB opponent profiles use only the games each QB primarily played."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "BUF", "LV", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "BUF", "DEN", "DEN", "KC", "BUF", "LV", "MIA"],
            "week": [1, 2, 1, 2, 3, 3, 3, 3],
            "points_allowed": [20, 24, 21, 17, 14, 28, 14, 28],
            "def_sacks": [2, 3, 1, 4, 5, 2, 5, 2],
            "def_interceptions": [1, 1, 0, 2, 3, 1, 3, 1],
            "def_pass_defended": [4, 5, 3, 7, 8, 5, 8, 5],
            "def_tackles_for_loss": [5, 6, 4, 8, 9, 6, 9, 6],
            "def_qb_hits": [6, 7, 5, 9, 10, 7, 10, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "DEN", "BUF", "LV", "MIA"],
            "week": [1, 1, 2, 2, 3, 3],
            "qb_id": ["QB_A", "QB_B", "QB_B", "QB_BUF", "QB_LV", "QB_MIA"],
            "qb_name": ["QB A", "QB B", "QB B", "Buffalo QB", "Raiders QB", "Miami QB"],
            "qb_offense_snaps": [45, 10, 55, 60, 60, 60],
            "qb_dropbacks": [22, 5, 25, 30, 20, 18],
            "qb_passer_rating": [100.0, 80.0, 95.0, 110.0, 70.0, 105.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_A", "QB_B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["DEN", "DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    profile_values = profiles.sort("qb_id").select("qopp_qb_passer_rating").to_series().to_list()
    assert profile_values == [70.0, 105.0]
    assert [row["opponent"] for row in details["QB_A"]] == ["KC"]
    assert [row["opponent"] for row in details["QB_B"]] == ["BUF"]


def test_compute_qb_opponent_profiles_handles_rams_alias_mismatch() -> None:
    """Verify LA/LAR source mismatches still produce Rams QB opponent profiles."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["LA", "SEA", "ARI", "SEA", "SF"],
            "opponent_team": ["SEA", "LA", "SEA", "ARI", "ARI"],
            "week": [1, 1, 2, 2, 2],
            "points_allowed": [17, 24, 14, 21, 21],
            "def_sacks": [3, 1, 4, 2, 2],
            "def_interceptions": [1, 0, 2, 1, 1],
            "def_pass_defended": [6, 3, 7, 5, 5],
            "def_tackles_for_loss": [7, 4, 8, 6, 6],
            "def_qb_hits": [8, 5, 9, 7, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["LAR", "ARI"],
            "week": [1, 2],
            "qb_id": ["QB_LAR", "QB_ARI"],
            "qb_name": ["Rams QB", "Cardinals QB"],
            "qb_passer_rating": [105.0, 88.0],
        }
    )
    schedule_df = pl.DataFrame(
        {
            "home_team": ["LA", "ARI"],
            "away_team": ["SEA", "SF"],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_LAR"],
            "qb_name": ["Rams QB"],
            "team": ["LAR"],
            "qb_passer_rating": [105.0],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("team").item() == "LAR"
    assert profiles.select("qopp_qb_passer_rating").item() == 88.0
    assert details["QB_LAR"] == [{"opponent": "SEA", "division": True, "games_included": 1}]


def test_compute_qb_opponent_profiles_derives_allowed_efficiency_rates() -> None:
    """Verify opponent QB context includes attempt-normalized allowed production rates."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "LV", "MIA", "KC", "KC"],
            "opponent_team": ["KC", "KC", "KC", "LV", "MIA"],
            "week": [1, 2, 3, 2, 3],
            "points_allowed": [20, 24, 21, 24, 21],
            "def_sacks": [2, 3, 1, 3, 1],
            "def_interceptions": [1, 1, 0, 1, 0],
            "def_pass_defended": [4, 5, 3, 5, 3],
            "def_tackles_for_loss": [5, 6, 4, 6, 4],
            "def_qb_hits": [6, 7, 5, 7, 5],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "LV", "MIA"],
            "week": [1, 2, 3],
            "qb_id": ["QB_DEN", "QB_LV", "QB_MIA"],
            "qb_name": ["Denver QB", "Raiders QB", "Miami QB"],
            "qb_attempts": [25, 40, 20],
            "qb_dropbacks": [28, 45, 22],
            "qb_pass_yards": [200.0, 200.0, 160.0],
            "qb_pass_touchdowns": [1.0, 1.0, 2.0],
            "qb_interceptions": [0.0, 1.0, 0.0],
            "qb_sacks": [1.0, 2.0, 1.0],
            "qb_sack_yards_lost": [8.0, 12.0, 8.0],
            "qb_passing_epa": [3.0, 10.0, 4.4],
            "qb_epa_per_dropback": [3.0 / 28.0, 10.0 / 45.0, 4.4 / 22.0],
            "qb_pass_yards_per_dropback": [200.0 / 28.0, 200.0 / 45.0, 160.0 / 22.0],
            "qb_td_int_margin_rate": [1.0 / 28.0, 0.0, 2.0 / 22.0],
            "qb_sack_rate": [1.0 / 28.0, 2.0 / 45.0, 1.0 / 22.0],
            "qb_any_a": [
                (200.0 + 20.0 - 0.0 - 8.0) / 26.0,
                (200.0 + 20.0 - 45.0 - 12.0) / 42.0,
                (160.0 + 40.0 - 0.0 - 8.0) / 21.0,
            ],
        }
    )
    schedule_df = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})
    qb_season_df = pl.DataFrame({"qb_id": ["QB_DEN"], "qb_name": ["Denver QB"], "team": ["DEN"]})

    # Act
    profiles, _ = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("qopp_qb_yards_per_attempt").item() == 6.0
    assert profiles.select("qopp_qb_touchdown_rate").item() == 0.05
    assert profiles.select("qopp_qb_interception_rate").item() == 1 / 60
    assert profiles.select("qopp_qb_epa_per_dropback").item() == pytest.approx(14.4 / 67.0)
    assert profiles.select("qopp_qb_any_a").item() == pytest.approx(355.0 / 63.0)
    assert profiles.select("qopp_qb_sack_rate").item() == pytest.approx(3.0 / 67.0)
    assert profiles.select("qopp_qb_td_int_margin_rate").item() == pytest.approx(2.0 / 67.0)
