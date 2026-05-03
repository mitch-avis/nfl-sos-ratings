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
        ("qb_passer_rating", 0.5, True),
        ("qb_aggressiveness", 0.5, True),
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
    ]
    assert ratings.height == 3
    assert ratings.filter(pl.col("qb_id") == "A").select("QSaCR").item() > 0
    assert ratings.filter(pl.col("qb_id") == "A").select("QSaCR_pct").item() > 50.0


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

    min_corr, sos_weight = qb_ratings.calibrate_qb_model(
        calibration_df,
        correlation_grid=[0.05, 0.1],
        sos_weight_grid=[0.1, 0.2, 0.3],
    )

    assert min_corr in {0.05, 0.1}
    assert sos_weight in {0.1, 0.2, 0.3}
