"""Tests for partial-season team rating snapshots."""

import math

import polars as pl

from nfl_sos_ratings import ratings, simultaneous_adjustment
from nfl_sos_ratings.validation.snapshots import build_team_rating_snapshot

_TEAM_RIDGE_RESPONSE_COLS: list[str] = [
    "passing_epa_per_offensive_snap",
    "rushing_epa_per_offensive_snap",
]


def _weekly_team_rows() -> pl.DataFrame:
    """Build a minimal multi-week team-game fixture for snapshot tests."""
    return pl.DataFrame(
        {
            "team": ["A", "B", "A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A", "B", "A"],
            "week": [1, 1, 2, 2, 3, 3],
            "is_home": [True, False, False, True, True, False],
            "passing_epa_per_offensive_snap": [0.28, -0.10, 0.16, 0.02, 0.75, -0.55],
            "rushing_epa_per_offensive_snap": [0.12, -0.04, 0.08, 0.00, 0.48, -0.30],
        }
    )


def test_build_team_rating_snapshot_matches_live_team_rating_path() -> None:
    """Verify snapshots reuse the published team rating derivation on pre-cutoff rows."""
    weekly_df = _weekly_team_rows()
    filtered = weekly_df.filter(pl.col("week") < 3)

    expected = ratings.compute_ratings(
        simultaneous_adjustment.compute_team_adjusted_stats(
            filtered,
            response_cols=_TEAM_RIDGE_RESPONSE_COLS,
        )
    ).sort("team")

    result = build_team_rating_snapshot(weekly_df, cutoff_week=3).sort("team")

    assert result.select(["team", "SaOR", "SaDR", "SaOvR"]).to_dict(as_series=False) == (
        expected.select(["team", "SaOR", "SaDR", "SaOvR"]).to_dict(as_series=False)
    )


def test_build_team_rating_snapshot_ignores_cutoff_week_and_later_rows() -> None:
    """Verify future-week perturbations cannot influence an earlier snapshot."""
    weekly_df = _weekly_team_rows()
    perturbed = weekly_df.with_columns(
        pl.when(pl.col("week") >= 3)
        .then(pl.col("passing_epa_per_offensive_snap") * 100.0)
        .otherwise(pl.col("passing_epa_per_offensive_snap"))
        .alias("passing_epa_per_offensive_snap"),
        pl.when(pl.col("week") >= 3)
        .then(pl.col("rushing_epa_per_offensive_snap") * -100.0)
        .otherwise(pl.col("rushing_epa_per_offensive_snap"))
        .alias("rushing_epa_per_offensive_snap"),
    )

    baseline = build_team_rating_snapshot(weekly_df, cutoff_week=3).sort("team")
    future_perturbed = build_team_rating_snapshot(perturbed, cutoff_week=3).sort("team")

    assert baseline.to_dict(as_series=False) == future_perturbed.to_dict(as_series=False)


def test_build_team_rating_snapshot_handles_early_cutoff() -> None:
    """Verify an early-week snapshot returns finite ratings instead of failing."""
    result = build_team_rating_snapshot(_weekly_team_rows(), cutoff_week=2).sort("team")

    assert result.columns == ["team", "SaOR", "SaDR", "SaOvR", "SaCR"]
    assert result.select("team").to_series().to_list() == ["A", "B"]
    assert all(
        math.isfinite(value)
        for value in result.select(["SaOR", "SaDR", "SaOvR", "SaCR"]).row(0)
        + result.select(["SaOR", "SaDR", "SaOvR", "SaCR"]).row(1)
    )
