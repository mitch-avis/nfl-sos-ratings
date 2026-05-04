"""Tests for QB opponent-profile and rating helpers."""

import numpy as np
import polars as pl

from nfl_sos_ratings import qb_opponent_stats, qb_ratings


def test_compute_qb_opponent_profiles_excludes_head_to_head() -> None:
    """Verify QB opponent profiles exclude games against the evaluated QB's team."""
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

    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    assert profiles.columns[0] == "qb_id"
    assert "qopp_points_allowed" in profiles.columns
    assert "qopp_qb_passer_rating" in profiles.columns
    assert profiles.select("qopp_points_allowed").item() == 17.5
    assert profiles.select("qopp_qb_passer_rating").item() == 87.5
    assert details["DEN"][0]["games_included"] == 1


def test_compute_qb_opponent_profiles_counts_actual_games_faced() -> None:
    """Verify repeated opponents are weighted by QB games actually faced."""
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

    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    assert profiles.select("qopp_qb_passer_rating").item() == 100.0
    assert [row["opponent"] for row in details["DEN"]] == ["KC", "KC", "BUF"]


def test_compute_qb_opponent_profiles_uses_unique_schedule_fallback_opponents() -> None:
    """Verify fallback schedule context still avoids duplicate sparse schedule rows."""
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

    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    assert profiles.select("qopp_qb_passer_rating").item() == 77.5
    assert [row["opponent"] for row in details["DEN"]] == ["BUF", "KC"]


def test_compute_qb_opponent_profiles_uses_each_qbs_actual_opponents() -> None:
    """Verify same-team QBs are profiled from the opponents each QB actually faced."""
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

    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    profile_values = profiles.sort("qb_id").select("qopp_qb_passer_rating").to_series().to_list()
    assert profile_values == [70.0, 110.0]
    assert [row["opponent"] for row in details["DEN"]] == ["BUF"]


def test_compute_qb_opponent_profiles_handles_rams_alias_mismatch() -> None:
    """Verify LA/LAR source mismatches still produce Rams QB opponent profiles."""
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

    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    assert profiles.select("team").item() == "LAR"
    assert profiles.select("qopp_qb_passer_rating").item() == 88.0
    assert details["LAR"] == [{"opponent": "SEA", "division": True, "games_included": 1}]


def test_compute_qb_opponent_profiles_derives_allowed_efficiency_rates() -> None:
    """Verify opponent QB context includes attempt-normalized allowed production rates."""
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
            "qb_pass_yards": [200.0, 200.0, 160.0],
            "qb_pass_touchdowns": [1.0, 1.0, 2.0],
            "qb_interceptions": [0.0, 1.0, 0.0],
        }
    )
    schedule_df = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})
    qb_season_df = pl.DataFrame({"qb_id": ["QB_DEN"], "qb_name": ["Denver QB"], "team": ["DEN"]})

    profiles, _ = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        schedule_df,
        qb_season_df,
    )

    assert profiles is not None
    assert profiles.select("qopp_qb_yards_per_attempt").item() == 6.0
    assert profiles.select("qopp_qb_touchdown_rate").item() == 0.05
    assert profiles.select("qopp_qb_interception_rate").item() == 1 / 60


def test_derive_qb_weights_uses_correlation_and_fallback() -> None:
    """Verify QB stat weights are correlation-driven with equal-weight fallback."""
    df = pl.DataFrame(
        {
            "qb_passer_rating": [80.0, 90.0, 100.0, 110.0],
            "qb_completion_percentage_above_expectation": [-2.0, 0.0, 2.0, 4.0],
            "qb_aggressiveness": [11.0, 10.8, 10.9, 11.1],
            "win_pct": [0.2, 0.4, 0.7, 0.9],
        }
    )

    weights = qb_ratings._derive_qb_weights(df)
    assert {stat for stat, _, _ in weights} >= {
        "qb_passer_rating",
        "qb_completion_percentage_above_expectation",
    }
    assert np.isclose(sum(weight for _, weight, _ in weights), 1.0)

    fallback = qb_ratings._derive_qb_weights(
        pl.DataFrame(
            {
                "qb_passer_rating": [100.0, 100.0, 100.0],
                "qb_aggressiveness": [10.0, 10.0, 10.0],
            }
        )
    )
    assert fallback == [
        ("qb_passer_rating", 1.0, True),
    ]


def test_compute_qb_ratings_returns_rankable_outputs() -> None:
    """Verify QB ratings helper returns expected columns and rankable values."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["A", "B", "C"],
            "qb_name": ["A QB", "B QB", "C QB"],
            "team": ["DEN", "KC", "BUF"],
            "qb_win_pct": [0.8, 0.3, 0.5],
            "qb_passer_rating": [105.0, 95.0, 100.0],
            "qb_completion_percentage_above_expectation": [3.0, 0.0, 1.5],
            "qb_aggressiveness": [11.0, 9.5, 10.2],
            "qb_avg_time_to_throw": [2.6, 2.9, 2.75],
            "qopp_qb_passer_rating": [88.0, 98.0, 93.0],
            "qopp_qb_completion_percentage_above_expectation": [0.5, 2.0, 1.1],
            "qopp_points_allowed": [18.0, 24.0, 21.0],
            "qopp_def_sacks": [3.0, 2.0, 2.5],
            "qopp_def_interceptions": [1.2, 0.8, 1.0],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined)

    assert ratings.columns == [
        "qb_id",
        "qb_name",
        "team",
        "QRaw",
        "QSaOR",
        "QSoS",
        "QSaCR",
        "QRaw_pct",
        "QSaOR_pct",
        "QSoS_pct",
        "QSaCR_pct",
        "QOutcome",
        "QOutcome_pct",
    ]
    assert ratings.height == 3
    assert ratings.filter(pl.col("qb_id") == "A").select("QSaCR").item() > 0
    assert ratings.filter(pl.col("qb_id") == "A").select("QSaCR_pct").item() > 50.0


def test_compute_qb_ratings_filters_to_eligible_and_uses_differentials() -> None:
    """Verify final QB ratings rank qualified passers by QB-vs-opponent differentials."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["RAW", "DIFF", "BACKUP"],
            "qb_name": ["Raw Star", "Diff Star", "Backup Star"],
            "team": ["DEN", "KC", "BUF"],
            "qb_is_eligible": [True, True, False],
            "qb_attempts_total": [300, 320, 20],
            "qb_win_pct": [0.5, 0.5, 1.0],
            "qb_passer_rating": [120.0, 95.0, 150.0],
            "diff_qb_passer_rating": [-10.0, 15.0, 50.0],
            "diff_qb_pass_yards": [-20.0, 40.0, 100.0],
            "diff_qb_pass_touchdowns": [-0.5, 1.0, 3.0],
            "diff_qb_interceptions": [0.5, -0.3, -2.0],
            "qopp_points_allowed": [20.0, 20.0, 20.0],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined, sos_weight=0.0)

    assert ratings.select("qb_id").to_series().to_list() == ["RAW", "DIFF"]
    assert (
        ratings.filter(pl.col("qb_id") == "RAW").select("QRaw").item()
        > ratings.filter(pl.col("qb_id") == "DIFF").select("QRaw").item()
    )
    assert (
        ratings.filter(pl.col("qb_id") == "DIFF").select("QSaCR").item()
        > ratings.filter(pl.col("qb_id") == "RAW").select("QSaCR").item()
    )


def test_compute_qb_ratings_standardizes_paired_context_before_adjusting() -> None:
    """Verify raw-unit differentials do not wash out paired schedule context."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["EASY", "HARD", "AVG"],
            "qb_name": ["Easy Raw", "Hard Context", "Average"],
            "team": ["NE", "LAR", "DEN"],
            "qb_is_eligible": [True, True, True],
            "qb_passer_rating": [120.0, 117.0, 80.0],
            "qopp_qb_passer_rating": [96.0, 94.0, 95.0],
            "qb_completion_percentage_above_expectation": [8.0, 5.8, -5.0],
            "qopp_qb_completion_percentage_above_expectation": [1.0, -1.0, 0.0],
            "qb_pass_yards": [300.0, 297.0, 180.0],
            "qopp_qb_pass_yards": [221.0, 219.0, 220.0],
            "qb_pass_touchdowns": [3.0, 2.7, 0.5],
            "qopp_qb_pass_touchdowns": [1.6, 1.4, 1.5],
            "qb_interceptions": [0.2, 0.4, 1.5],
            "qopp_qb_interceptions": [0.9, 1.0, 0.95],
            "diff_qb_passer_rating": [24.0, 23.0, -15.0],
            "diff_qb_completion_percentage_above_expectation": [7.0, 6.8, -5.0],
            "diff_qb_pass_yards": [79.0, 78.0, -40.0],
            "diff_qb_pass_touchdowns": [1.4, 1.3, -1.0],
            "diff_qb_interceptions": [-0.7, -0.6, 0.55],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined, sos_weight=0.0)

    assert (
        ratings.filter(pl.col("qb_id") == "HARD").select("QSaCR").item()
        > ratings.filter(pl.col("qb_id") == "EASY").select("QSaCR").item()
    )


def test_compute_qb_ratings_raw_composite_prioritizes_production() -> None:
    """Verify QRaw does not reward style fields over conventional QB production."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["PRODUCT", "STYLE", "AVG"],
            "qb_name": ["Production", "Style", "Average"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "qb_passer_rating": [110.0, 95.0, 90.0],
            "qb_completion_percentage_above_expectation": [5.0, 0.0, -1.0],
            "qb_pass_yards": [280.0, 210.0, 220.0],
            "qb_pass_touchdowns": [2.4, 1.4, 1.2],
            "qb_interceptions": [0.4, 0.8, 1.0],
            "qb_aggressiveness": [8.0, 30.0, 10.0],
            "qb_avg_intended_air_yards": [7.0, 14.0, 8.0],
            "qb_avg_air_yards_to_sticks": [0.0, 8.0, 1.0],
            "qb_avg_time_to_throw": [2.7, 2.5, 2.8],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined, sos_weight=0.0)

    assert (
        ratings.filter(pl.col("qb_id") == "PRODUCT").select("QRaw").item()
        > ratings.filter(pl.col("qb_id") == "STYLE").select("QRaw").item()
    )


def test_compute_qb_ratings_qsos_ignores_style_only_context() -> None:
    """Verify QSoS is not moved by opponent QB style fields alone."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["LOW_STYLE", "HIGH_STYLE"],
            "qb_name": ["Low Style", "High Style"],
            "team": ["A", "B"],
            "qb_is_eligible": [True, True],
            "qb_passer_rating": [100.0, 100.0],
            "qopp_points_allowed": [22.0, 22.0],
            "qopp_def_sacks": [2.5, 2.5],
            "qopp_def_interceptions": [1.0, 1.0],
            "qopp_qb_pass_yards": [220.0, 220.0],
            "qopp_qb_pass_touchdowns": [1.5, 1.5],
            "qopp_qb_passer_rating": [92.0, 92.0],
            "qopp_qb_completion_percentage_above_expectation": [1.0, 1.0],
            "qopp_qb_interceptions": [0.8, 0.8],
            "qopp_qb_aggressiveness": [8.0, 20.0],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined)

    qsos_values = ratings.sort("qb_id").select("QSoS").to_series().to_list()
    assert qsos_values[0] == qsos_values[1]


def test_compute_qb_ratings_anchors_final_composite_to_qb_outcomes() -> None:
    """Verify QSaCR can reflect QB results while QSaOR remains passing-context based."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["LOW_RESULT", "HIGH_RESULT", "AVG"],
            "qb_name": ["Good Stats Bad Record", "Modest Stats Great Record", "Average"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "qb_win_pct": [0.083, 0.824, 0.5],
            "qb_passer_rating": [92.0, 91.0, 91.0],
            "qopp_qb_passer_rating": [91.0, 91.0, 91.0],
            "qb_completion_percentage_above_expectation": [0.5, 0.0, 0.0],
            "qopp_qb_completion_percentage_above_expectation": [0.0, 0.0, 0.0],
            "qb_yards_per_attempt": [6.6, 6.5, 6.5],
            "qopp_qb_yards_per_attempt": [6.5, 6.5, 6.5],
            "qb_touchdown_rate": [0.041, 0.04, 0.04],
            "qopp_qb_touchdown_rate": [0.04, 0.04, 0.04],
            "qb_interception_rate": [0.019, 0.019, 0.019],
            "qopp_qb_interception_rate": [0.02, 0.02, 0.02],
            "qopp_points_allowed": [22.0, 22.0, 22.0],
        }
    )

    ratings = qb_ratings.compute_qb_ratings(qb_combined, sos_weight=0.0, outcome_weight=2.0)

    assert (
        ratings.filter(pl.col("qb_id") == "LOW_RESULT").select("QSaOR").item()
        > ratings.filter(pl.col("qb_id") == "HIGH_RESULT").select("QSaOR").item()
    )
    assert (
        ratings.filter(pl.col("qb_id") == "HIGH_RESULT").select("QSaCR").item()
        > ratings.filter(pl.col("qb_id") == "LOW_RESULT").select("QSaCR").item()
    )


def test_calibrate_qb_model_returns_candidate_constants() -> None:
    """Verify calibration returns constants from provided candidate grids."""
    calibration_df = pl.DataFrame(
        {
            "qb_win_pct": [0.2, 0.4, 0.6, 0.8],
            "qb_passer_rating": [80.0, 90.0, 100.0, 110.0],
            "qb_completion_percentage_above_expectation": [-2.0, 0.0, 1.0, 3.0],
            "qopp_points_allowed": [24.0, 22.0, 20.0, 18.0],
            "qopp_def_sacks": [2.0, 2.5, 3.0, 3.5],
            "qopp_def_interceptions": [0.8, 1.0, 1.2, 1.4],
        }
    )

    min_corr, sos_weight, outcome_weight = qb_ratings.calibrate_qb_model(
        calibration_df,
        correlation_grid=[0.05, 0.1],
        sos_weight_grid=[0.1, 0.2, 0.3],
        outcome_weight_grid=[0.2, 0.4],
    )

    assert min_corr in {0.05, 0.1}
    assert sos_weight in {0.1, 0.2, 0.3}
    assert outcome_weight in {0.2, 0.4}


def test_calibrate_qb_model_keeps_material_schedule_weight_by_default() -> None:
    """Verify calibration cannot wash out schedule context with a near-zero SOS weight."""
    calibration_df = pl.DataFrame(
        {
            "qb_win_pct": [0.2, 0.4, 0.6, 0.8],
            "qb_passer_rating": [80.0, 90.0, 100.0, 110.0],
            "qb_completion_percentage_above_expectation": [-2.0, 0.0, 1.0, 3.0],
            "qb_yards_per_attempt": [5.8, 6.5, 7.1, 7.8],
            "qb_touchdown_rate": [0.025, 0.035, 0.045, 0.055],
            "qb_interception_rate": [0.04, 0.03, 0.02, 0.01],
            "qopp_qb_passer_rating": [98.0, 95.0, 92.0, 89.0],
            "qopp_qb_completion_percentage_above_expectation": [2.0, 1.0, 0.0, -1.0],
            "qopp_qb_yards_per_attempt": [7.4, 7.0, 6.6, 6.2],
            "qopp_qb_touchdown_rate": [0.05, 0.045, 0.04, 0.035],
            "qopp_qb_interception_rate": [0.015, 0.02, 0.025, 0.03],
            "qopp_points_allowed": [25.0, 23.0, 21.0, 19.0],
            "qopp_def_sacks": [2.0, 2.5, 3.0, 3.5],
            "qopp_def_interceptions": [0.6, 0.8, 1.0, 1.2],
        }
    )

    _, sos_weight, outcome_weight = qb_ratings.calibrate_qb_model(calibration_df)

    assert sos_weight >= 2.0
    assert outcome_weight <= 0.75
