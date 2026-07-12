"""Tests for nfl_sos_ratings.ratings."""

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings import ratings


def test_rating_helpers_cover_edge_cases() -> None:
    """Verify helper functions handle standard and edge-case numeric inputs."""
    assert np.allclose(ratings._zscore([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(ratings._zscore([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 0.0]))

    df = pl.DataFrame({"team": ["DEN"], "value": [1.5]})
    value_col = ratings._col(df, "value")
    assert value_col is not None
    assert np.allclose(value_col, np.array([1.5]))
    assert ratings._col(df, "missing") is None

    composite = ratings._build_composite(
        pl.DataFrame({"value": [1.0, 2.0, 3.0]}),
        [("missing", 0.5, True), ("value", 0.5, True)],
    )
    assert np.allclose(composite, np.array([-0.5, 0.0, 0.5]))


def test_derive_weights_builds_weighted_composite_and_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify stat pools are weighted equally across the present columns."""
    df = pl.DataFrame(
        {
            "stat_a": [1.0, 2.0, 3.0, 4.0],
            "stat_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    win_pct = np.array([0.25, 0.5, 0.75, 1.0])

    weighted = ratings._derive_weights(
        df,
        [("stat_a", True), ("stat_b", False)],
        win_pct,
        "Offensive",
    )
    composite = ratings._build_composite(df, weighted)

    assert weighted == [("stat_a", 0.5, True), ("stat_b", 0.5, False)]
    assert np.allclose(composite, np.array([-1.161895, -0.387298, 0.387298, 1.161895]))
    assert capsys.readouterr().out == ""

    fallback = ratings._derive_weights(
        pl.DataFrame({"stat_c": [1.0, 2.0, 1.0]}),
        [("stat_c", True)],
        np.array([0.3, 0.5, 0.7]),
        "Defensive",
    )

    assert fallback == [("stat_c", 1.0, True)]
    assert capsys.readouterr().out == ""


def test_compute_ratings_with_real_inputs() -> None:
    """Verify compute_ratings emits overall ratings and sensible ranking direction."""
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "win_pct": [0.25, 0.5, 0.75, 1.0],
            "points_per_offensive_snap": [0.2, 0.3, 0.4, 0.5],
            "total_yards_per_offensive_snap": [4.0, 4.5, 5.0, 5.5],
            "passing_yards_per_offensive_snap": [2.5, 2.8, 3.1, 3.4],
            "rushing_yards_per_offensive_snap": [1.5, 1.7, 1.9, 2.1],
            "passing_epa_per_offensive_snap": [0.0, 0.1, 0.2, 0.3],
            "rushing_epa_per_offensive_snap": [0.0, 0.05, 0.1, 0.15],
            "passing_tds_per_offensive_snap": [0.01, 0.02, 0.03, 0.04],
            "rushing_tds_per_offensive_snap": [0.0, 0.01, 0.01, 0.02],
            "passing_first_downs_per_offensive_snap": [0.12, 0.15, 0.18, 0.21],
            "rushing_first_downs_per_offensive_snap": [0.06, 0.07, 0.08, 0.09],
            "passing_cpoe": [-1.0, 0.0, 1.0, 2.0],
            "sacks_suffered_per_offensive_snap": [0.08, 0.06, 0.04, 0.02],
            "passing_interceptions_per_offensive_snap": [0.04, 0.02, 0.02, 0.0],
            "sack_fumbles_lost_per_offensive_snap": [0.02, 0.02, 0.0, 0.0],
            "rushing_fumbles_lost_per_offensive_snap": [0.02, 0.0, 0.0, 0.0],
            "points_allowed_per_defensive_snap": [0.5, 0.4, 0.3, 0.2],
            "total_yards_allowed_per_defensive_snap": [5.5, 5.0, 4.5, 4.0],
            "passing_yards_allowed_per_defensive_snap": [3.4, 3.1, 2.8, 2.5],
            "rushing_yards_allowed_per_defensive_snap": [2.1, 1.9, 1.7, 1.5],
            "passing_epa_allowed_per_defensive_snap": [0.3, 0.2, 0.1, 0.0],
            "rushing_epa_allowed_per_defensive_snap": [0.15, 0.1, 0.05, 0.0],
            "passing_tds_allowed_per_defensive_snap": [0.04, 0.03, 0.02, 0.01],
            "rushing_tds_allowed_per_defensive_snap": [0.02, 0.01, 0.01, 0.0],
            "passing_first_downs_allowed_per_defensive_snap": [0.21, 0.18, 0.15, 0.12],
            "rushing_first_downs_allowed_per_defensive_snap": [0.09, 0.08, 0.07, 0.06],
            "passing_cpoe_allowed": [2.0, 1.0, 0.0, -1.0],
            "def_sacks_per_defensive_snap": [0.02, 0.04, 0.06, 0.08],
            "def_interceptions_per_defensive_snap": [0.0, 0.02, 0.02, 0.04],
            "def_pass_defended_per_defensive_snap": [0.06, 0.08, 0.10, 0.12],
            "def_tackles_for_loss_per_defensive_snap": [0.08, 0.10, 0.12, 0.14],
            "def_qb_hits_per_defensive_snap": [0.10, 0.12, 0.14, 0.16],
            "def_fumbles_forced_per_defensive_snap": [0.0, 0.02, 0.02, 0.04],
            "def_safeties_per_defensive_snap": [0.0, 0.0, 0.0, 0.01],
            "opp_points_allowed": [24, 22, 20, 18],
            "opp_points_for": [18, 21, 24, 27],
            "opp_passing_epa": [0.0, 0.05, 0.1, 0.15],
            "opp_win_value": [0.3, 0.45, 0.6, 0.75],
            "opp_turnover_margin": [-0.2, 0.0, 0.2, 0.4],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.columns == ["team", "SaOR", "SaDR", "SaOvR", "SaCR"]
    assert result.select("team").to_series().to_list() == ["A", "B", "C", "D"]
    assert result.filter(pl.col("team") == "D").select("SaCR").item() > 0
    assert result.filter(pl.col("team") == "D").select("SaOvR").item() > 0


def test_compute_ratings_ignores_raw_total_only_columns() -> None:
    """Verify raw totals alone do not move ratings when rate fields are absent."""
    df = pl.DataFrame(
        {
            "team": ["A", "B"],
            "points_for": [400.0, 100.0],
            "points_allowed": [100.0, 400.0],
            "total_yards": [6000.0, 3000.0],
            "passing_yards": [4200.0, 1800.0],
            "rushing_yards": [1800.0, 1200.0],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.select("SaOR").to_series().to_list() == [0.0, 0.0]
    assert result.select("SaDR").to_series().to_list() == [0.0, 0.0]
    assert result.select("SaOvR").to_series().to_list() == [0.0, 0.0]
    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0]


def test_compute_ratings_without_win_pct_and_without_sos_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify compute_ratings falls back cleanly when no rating inputs are present."""
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "unrelated_metric": [1.0, 2.0, 3.0],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.height == 3
    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaOvR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert capsys.readouterr().out == ""


def test_compute_ratings_makes_overall_secondary_to_offense_and_defense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify compute_ratings treats SaOvR as secondary inside the final team composite."""
    df = pl.DataFrame(
        {
            "team": ["A", "B"],
            "win_pct": [6.0, -6.0],
            "points_for": [10, 20],
            "points_allowed": [20, 10],
            "def_interceptions": [3.0, -3.0],
            "def_fumbles_forced": [3.0, -3.0],
            "passing_interceptions": [0.0, 0.0],
            "sack_fumbles_lost": [0.0, 0.0],
            "rushing_fumbles_lost": [0.0, 0.0],
        }
    )

    composites = iter([np.array([2.0, -2.0]), np.array([4.0, -4.0])])

    monkeypatch.setattr(
        ratings,
        "_derive_weights",
        lambda *args, **kwargs: [("points_for", 1.0, True)],
    )
    monkeypatch.setattr(ratings, "_build_composite", lambda *args, **kwargs: next(composites))
    monkeypatch.setattr(ratings, "_zscore", lambda values: np.array(values, dtype=np.float64))

    result = ratings.compute_ratings(df)

    assert result.select("SaOR").to_series().to_list() == [2.0, -2.0]
    assert result.select("SaDR").to_series().to_list() == [4.0, -4.0]
    assert result.select("SaOvR").to_series().to_list() == [6.0, -6.0]
    assert result.select("SaCR").to_series().to_list() == [3.333, -3.333]
