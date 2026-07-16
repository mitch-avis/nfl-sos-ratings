"""Tests for partial-season team rating snapshots."""

import math

import polars as pl
import pytest

from nfl_sos_ratings import ratings, simultaneous_adjustment
from nfl_sos_ratings.validation.snapshots import (
    build_play_level_team_adjusted_snapshot,
    build_play_level_team_frame_from_pbp,
    build_special_teams_game_frame_from_pbp,
    build_special_teams_rating_snapshot,
    build_team_adjusted_snapshot,
    build_team_rating_snapshot,
    build_team_weighted_rating_snapshot,
)

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

    assert result.select(["team", "SaOR", "SaDR", "SaSTR", "SaOvR"]).to_dict(as_series=False) == (
        expected.select(["team", "SaOR", "SaDR", "SaSTR", "SaOvR"]).to_dict(as_series=False)
    )


def test_build_team_adjusted_snapshot_matches_live_adjustment_path() -> None:
    """Verify adjusted snapshots reuse the published pre-cutoff team adjustment path."""
    weekly_df = _weekly_team_rows()
    filtered = weekly_df.filter(pl.col("week") < 3)

    expected = simultaneous_adjustment.compute_team_adjusted_stats(
        filtered,
        response_cols=_TEAM_RIDGE_RESPONSE_COLS,
    ).sort("team")

    result = build_team_adjusted_snapshot(weekly_df, cutoff_week=3).sort("team")

    assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)


def test_build_team_weighted_rating_snapshot_applies_supplied_weights() -> None:
    """Verify weighted team snapshots z-score the supplied adjusted components."""
    weekly_df = _weekly_team_rows()

    result = build_team_weighted_rating_snapshot(
        weekly_df,
        cutoff_week=3,
        weight_map={
            "adj_off_passing_epa_per_offensive_snap": 1.0,
            "adj_off_rushing_epa_per_offensive_snap": 0.0,
            "adj_def_passing_epa_per_offensive_snap": 0.0,
            "adj_def_rushing_epa_per_offensive_snap": 0.0,
        },
        output_col="T1SaOvR",
    ).sort("team")

    adjusted = build_team_adjusted_snapshot(weekly_df, cutoff_week=3).sort("team")
    passing_values = adjusted.select("adj_off_passing_epa_per_offensive_snap").to_series().to_list()
    mean = sum(passing_values) / len(passing_values)
    variance = sum((value - mean) ** 2 for value in passing_values) / (len(passing_values) - 1)
    std = variance**0.5
    expected = [(value - mean) / std for value in passing_values]

    assert result.columns == ["team", "T1SaOvR"]
    assert result.select("T1SaOvR").to_series().to_list() == pytest.approx(expected)


def test_build_team_weighted_rating_snapshot_standardizes_components_before_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling T1 weights must apply to standardized component columns, not raw magnitudes."""
    adjusted_snapshot = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "adj_off_passing_epa_per_offensive_snap": [10.0, 0.0, -10.0],
            "adj_off_rushing_epa_per_offensive_snap": [-0.1, 0.0, 0.1],
            "adj_def_passing_epa_per_offensive_snap": [0.0, 0.0, 0.0],
            "adj_def_rushing_epa_per_offensive_snap": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        "nfl_sos_ratings.validation.snapshots.build_team_adjusted_snapshot",
        lambda weekly_team_rows, cutoff_week, response_cols=None: adjusted_snapshot,
    )

    result = build_team_weighted_rating_snapshot(
        pl.DataFrame({"team": ["A", "B", "C"], "week": [1, 1, 1]}),
        cutoff_week=2,
        weight_map={
            "adj_off_passing_epa_per_offensive_snap": 0.5,
            "adj_off_rushing_epa_per_offensive_snap": 0.5,
            "adj_def_passing_epa_per_offensive_snap": 0.0,
            "adj_def_rushing_epa_per_offensive_snap": 0.0,
        },
        output_col="T1SaOvR",
    )

    assert result.select("T1SaOvR").to_series().to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_build_special_teams_game_frame_from_pbp_builds_team_margins() -> None:
    """Special-play PBP should become one net special-teams margin row per team-game."""
    pbp = pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "week": [1, 1],
            "posteam": ["A", "B"],
            "defteam": ["B", "A"],
            "special": [1, 1],
            "epa": [-0.3, 1.0],
        }
    )

    result = build_special_teams_game_frame_from_pbp(pbp).sort("team")

    assert result.select("team").to_series().to_list() == ["A", "B"]
    assert result.select("st_epa_margin_per_play").to_series().to_list() == pytest.approx(
        [-0.65, 0.65]
    )


def test_build_play_level_team_frame_from_pbp_builds_per_snap_pass_rush_rows() -> None:
    """Scrimmage PBP should become one row per offensive snap with pass/rush EPA channels."""
    pbp = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g1", "g1", "g1"],
            "week": [1, 1, 1, 1, 1, 1],
            "posteam": ["A", "A", "A", "B", "B", "B"],
            "defteam": ["B", "B", "B", "A", "A", "A"],
            "home_team": ["A", "A", "A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B", "B", "B"],
            "qb_dropback": [1, 0, 0, 1, 0, 0],
            "rush": [0, 1, 0, 0, 1, 0],
            "qb_kneel": [0, 0, 1, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0, 0, 0],
            "epa": [0.4, 0.2, -0.1, -0.3, 0.1, 1.0],
        }
    )

    result = build_play_level_team_frame_from_pbp(pbp)

    assert result.columns == [
        "game_id",
        "week",
        "team",
        "opponent_team",
        "is_home",
        "passing_epa_per_offensive_snap",
        "rushing_epa_per_offensive_snap",
    ]
    assert result.height == 5
    assert result.filter(pl.col("team") == "A").select("is_home").to_series().to_list() == [
        True,
        True,
        True,
    ]
    assert result.select("passing_epa_per_offensive_snap").to_series().to_list() == pytest.approx(
        [0.4, 0.0, 0.0, -0.3, 0.0]
    )
    assert result.select("rushing_epa_per_offensive_snap").to_series().to_list() == pytest.approx(
        [0.0, 0.2, 0.0, 0.0, 0.1]
    )


def test_build_play_level_team_adjusted_snapshot_uses_only_pre_cutoff_rows() -> None:
    """Play-level snapshots should match the live ridge wrapper on pre-cutoff play rows only."""
    pbp = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "week": [1, 1, 2, 2, 3, 3],
            "posteam": ["A", "B", "A", "B", "A", "B"],
            "defteam": ["B", "A", "B", "A", "B", "A"],
            "home_team": ["A", "A", "B", "B", "A", "A"],
            "away_team": ["B", "B", "A", "A", "B", "B"],
            "qb_dropback": [1, 1, 1, 1, 1, 1],
            "rush": [0, 0, 0, 0, 0, 0],
            "qb_kneel": [0, 0, 0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0, 0, 0],
            "epa": [0.3, -0.2, 0.1, 0.0, 99.0, -99.0],
        }
    )
    play_rows = build_play_level_team_frame_from_pbp(pbp)
    expected = simultaneous_adjustment.compute_team_adjusted_stats(
        play_rows.filter(pl.col("week") < 3),
        response_cols=_TEAM_RIDGE_RESPONSE_COLS,
    ).sort("team")

    result = build_play_level_team_adjusted_snapshot(play_rows, cutoff_week=3).sort("team")

    assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)


def test_build_special_teams_rating_snapshot_uses_prior_games_only() -> None:
    """Special-teams snapshots should use only games before the requested cutoff."""
    st_games = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "week": [1, 1, 2, 2, 3, 3],
            "team": ["A", "B", "A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A", "B", "A"],
            "st_epa_margin_per_play": [0.8, -0.8, 0.2, -0.2, -0.6, 0.6],
        }
    )

    snapshot = build_special_teams_rating_snapshot(st_games, cutoff_week=3).sort("team")

    assert snapshot.select("st_rating").to_series().to_list() == pytest.approx([0.25, -0.25])


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

    assert result.columns == ["team", "SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]
    assert result.select("team").to_series().to_list() == ["A", "B"]
    assert all(
        math.isfinite(value)
        for value in result.select(["SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]).row(0)
        + result.select(["SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]).row(1)
    )
