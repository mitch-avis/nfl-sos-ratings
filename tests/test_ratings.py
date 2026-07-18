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
    """Verify published team ratings now follow the ridge-adjusted EPA surfaces."""
    df = pl.DataFrame(
        {
            "team": ["DOM", "OFF", "DEF", "BAD"],
            "win_pct": [0.25, 1.0, 0.75, 0.5],
            "adj_off_passing_epa_per_offensive_snap": [0.32, 0.28, 0.04, -0.20],
            "adj_off_rushing_epa_per_offensive_snap": [0.18, 0.14, -0.02, -0.12],
            "adj_def_passing_epa_per_offensive_snap": [0.30, -0.06, 0.22, -0.18],
            "adj_def_rushing_epa_per_offensive_snap": [0.16, -0.02, 0.18, -0.10],
            "st_rating": [0.20, -0.06, 0.08, -0.12],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.columns == ["team", "SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]
    assert result.select("team").to_series().to_list() == ["BAD", "DEF", "DOM", "OFF"]
    assert result.sort("SaOR", descending=True).select("team").to_series().to_list()[0] == "DOM"
    assert result.sort("SaDR", descending=True).select("team").to_series().to_list()[0] == "DOM"
    assert result.sort("SaSTR", descending=True).select("team").to_series().to_list()[0] == "DOM"
    assert result.sort("SaOvR", descending=True).select("team").to_series().to_list()[0] == "DOM"
    assert result.sort("SaCR", descending=True).select("team").to_series().to_list()[0] == "DOM"


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
    assert result.select("SaSTR").to_series().to_list() == [0.0, 0.0]
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
    assert result.select("SaSTR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaOvR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert capsys.readouterr().out == ""


def test_compute_ratings_excludes_outcome_only_fields_from_quality_ratings() -> None:
    """Verify outcome-only fields do not move any published team quality rating."""
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "win_pct": [1.0, 0.5, 0.0],
            "win_value": [1.0, 0.5, 0.0],
            "turnover_margin": [2.0, 0.0, -2.0],
            "opp_win_value": [0.2, 0.5, 0.8],
            "opp_turnover_margin": [-1.0, 0.0, 1.0],
        }
    )

    result = ratings.compute_ratings(df).sort("team")

    assert result.select("SaOR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaDR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaSTR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaOvR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0, 0.0]


def test_compute_ratings_builds_overall_and_composite_from_standardized_offense_and_defense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Team overall and composite assembly should stay in one place."""
    df = pl.DataFrame(
        {
            "team": ["A", "B"],
            "adj_off_passing_epa_per_offensive_snap": [1.0, 0.0],
            "adj_off_rushing_epa_per_offensive_snap": [1.0, 0.0],
            "adj_def_passing_epa_per_offensive_snap": [3.0, 0.0],
            "adj_def_rushing_epa_per_offensive_snap": [3.0, 0.0],
            "st_rating": [5.0, 0.0],
        }
    )

    monkeypatch.setattr(ratings, "_zscore", lambda values: np.array(values, dtype=np.float64))
    monkeypatch.setattr(
        ratings,
        "_zscore_against",
        lambda values, reference_values: np.array(values, dtype=np.float64),
    )
    monkeypatch.setattr(
        ratings.composite_weights,
        "_zscore_against",
        lambda values, reference_values: np.array(values, dtype=np.float64),
    )

    result = ratings.compute_ratings(df)
    weights = ratings.composite_weights.TEAM_SACR_FROZEN_SPEC.weight_map()
    expected_sacr = [
        (
            weights["adj_off_passing_epa_per_offensive_snap"] * 1.0
            + weights["adj_off_rushing_epa_per_offensive_snap"] * 1.0
            + weights["adj_def_passing_epa_per_offensive_snap"] * 3.0
            + weights["adj_def_rushing_epa_per_offensive_snap"] * 3.0
            + weights["st_rating"] * 5.0
        ),
        0.0,
    ]
    expected_sacr = [round(value, 3) for value in expected_sacr]

    assert result.select("SaOR").to_series().to_list() == [1.0, 0.0]
    assert result.select("SaDR").to_series().to_list() == [3.0, 0.0]
    assert result.select("SaSTR").to_series().to_list() == [5.0, 0.0]
    assert result.select("SaOvR").to_series().to_list() == [9.0, 0.0]
    assert result.select("SaCR").to_series().to_list() == pytest.approx(expected_sacr)


def test_compute_ratings_uses_frozen_stage_two_component_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SaCR should use the published weighted blend, not the interim equal average."""
    df = pl.DataFrame(
        {
            "team": ["A", "B"],
            "adj_off_passing_epa_per_offensive_snap": [1.0, -1.0],
            "adj_off_rushing_epa_per_offensive_snap": [2.0, -2.0],
            "adj_def_passing_epa_per_offensive_snap": [3.0, -3.0],
            "adj_def_rushing_epa_per_offensive_snap": [4.0, -4.0],
            "st_rating": [5.0, -5.0],
        }
    )

    monkeypatch.setattr(ratings, "_zscore", lambda values: np.array(values, dtype=np.float64))
    monkeypatch.setattr(
        ratings,
        "_zscore_against",
        lambda values, reference_values: np.array(values, dtype=np.float64),
    )
    monkeypatch.setattr(
        ratings.composite_weights,
        "_zscore_against",
        lambda values, reference_values: np.array(values, dtype=np.float64),
    )

    result = ratings.compute_ratings(df).sort("team")
    weights = ratings.composite_weights.TEAM_SACR_FROZEN_SPEC.weight_map()
    expected = [
        (
            weights["adj_off_passing_epa_per_offensive_snap"] * 1.0
            + weights["adj_off_rushing_epa_per_offensive_snap"] * 2.0
            + weights["adj_def_passing_epa_per_offensive_snap"] * 3.0
            + weights["adj_def_rushing_epa_per_offensive_snap"] * 4.0
            + weights["st_rating"] * 5.0
        ),
        (
            weights["adj_off_passing_epa_per_offensive_snap"] * -1.0
            + weights["adj_off_rushing_epa_per_offensive_snap"] * -2.0
            + weights["adj_def_passing_epa_per_offensive_snap"] * -3.0
            + weights["adj_def_rushing_epa_per_offensive_snap"] * -4.0
            + weights["st_rating"] * -5.0
        ),
    ]
    expected = [round(value, 3) for value in expected]

    assert result.select("SaCR").to_series().to_list() == pytest.approx(expected)


def test_compute_ratings_standardize_within_the_current_season() -> None:
    """Published team ratings should be centered and scaled within the current season."""
    current_df = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "adj_off_passing_epa_per_offensive_snap": [0.30, 0.12, -0.05, -0.20],
            "adj_off_rushing_epa_per_offensive_snap": [0.15, 0.08, -0.01, -0.10],
            "adj_def_passing_epa_per_offensive_snap": [0.22, 0.09, -0.04, -0.16],
            "adj_def_rushing_epa_per_offensive_snap": [0.14, 0.05, -0.02, -0.08],
            "st_rating": [0.11, 0.03, -0.02, -0.09],
        }
    )

    result = ratings.compute_ratings(current_df)

    for column in ("SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"):
        values = np.asarray(
            result.select(column).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        assert float(values.mean()) == pytest.approx(0.0, abs=0.001)
        assert float(values.std(ddof=1)) == pytest.approx(1.0, abs=0.001)
