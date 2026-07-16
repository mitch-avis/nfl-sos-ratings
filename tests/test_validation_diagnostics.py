"""Tests for Stage 3b validation diagnostics helpers."""

import polars as pl
import pytest

from nfl_sos_ratings.validation.diagnostics import (
    build_qb_adjustment_audit_frame,
    compute_season_mae_deltas,
    compute_weekly_mae_curves,
    summarize_defense_spread,
    summarize_qb_adjustment_slopes,
)


def test_compute_weekly_mae_curves_aggregates_by_baseline_and_week() -> None:
    """Weekly MAE curves should aggregate matching week numbers across rows."""
    predictions = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [5, 5, 6, 6],
            "baseline": ["SaOvR", "SRS", "SaOvR", "SRS"],
            "predicted_margin": [3.0, 5.0, 0.0, 1.0],
            "home_margin": [1.0, 1.0, 2.0, 2.0],
        }
    )

    curves = compute_weekly_mae_curves(predictions).sort(["baseline", "week"])

    assert curves.select("baseline").to_series().to_list() == ["SRS", "SRS", "SaOvR", "SaOvR"]
    assert curves.select("mae").to_series().to_list() == pytest.approx([4.0, 1.0, 2.0, 2.0])


def test_compute_season_mae_deltas_compares_requested_baselines() -> None:
    """Per-season deltas should subtract baseline B from baseline A."""
    metrics = pl.DataFrame(
        {
            "baseline": ["SaOvR", "SRS", "SaOvR", "SRS"],
            "season": [2024, 2024, 2025, 2025],
            "split": ["season", "season", "season", "season"],
            "games": [10, 10, 12, 12],
            "mae": [9.8, 10.1, 10.4, 10.0],
            "rmse": [12.0, 12.4, 13.1, 12.8],
        }
    )

    deltas = compute_season_mae_deltas(metrics, baseline_a="SaOvR", baseline_b="SRS")

    assert deltas.select("season").to_series().to_list() == [2024, 2025]
    assert deltas.select("mae_delta").to_series().to_list() == pytest.approx([-0.3, 0.4])
    assert deltas.select("rmse_delta").to_series().to_list() == pytest.approx([-0.4, 0.3])


def test_build_qb_adjustment_audit_frame_recovers_weighted_schedule_effect() -> None:
    """The audit frame should expose the weighted faced-defense effect in EPA units."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["A", "A", "B", "B"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B"],
            "team": ["TA", "TA", "TB", "TB"],
            "opponent_team": ["D1", "D2", "D1", "D2"],
            "qb_dropbacks": [40.0, 10.0, 10.0, 40.0],
            "qb_epa_per_dropback": [0.10, 0.30, -0.30, -0.10],
        }
    )

    audit = build_qb_adjustment_audit_frame(
        qb_games,
        response_col="qb_epa_per_dropback",
        ridge_lambda=0.0,
    ).sort("qb_id")

    assert audit.select("raw_value").to_series().to_list() == pytest.approx([0.14, -0.14])
    assert audit.select("adjusted_value").to_series().to_list() == pytest.approx([0.2, -0.2])
    assert audit.select("weighted_faced_defense").to_series().to_list() == pytest.approx(
        [0.06, -0.06]
    )
    assert audit.select("adjustment_delta").to_series().to_list() == pytest.approx([0.06, -0.06])


def test_summarize_qb_adjustment_slopes_reports_season_level_fit() -> None:
    """Season summaries should expose slope, correlation, and residual size."""
    audit = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "weighted_faced_defense": [-0.1, 0.0, 0.1],
            "adjustment_delta": [-0.1, 0.0, 0.1],
            "identity_residual": [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_qb_adjustment_slopes(audit)

    assert summary.select("season").to_series().to_list() == [2025]
    assert summary.select("slope").item() == pytest.approx(1.0)
    assert summary.select("correlation").item() == pytest.approx(1.0)
    assert summary.select("mean_abs_identity_residual").item() == pytest.approx(0.0)


def test_summarize_defense_spread_reports_ratio() -> None:
    """Defense spread summaries should report the two standard deviations and their ratio."""
    team_defense = pl.DataFrame({"team": ["A", "B"], "defense_rating": [0.10, -0.10]})
    qb_defense = pl.DataFrame({"team": ["A", "B"], "defense_rating": [0.05, -0.05]})

    summary = summarize_defense_spread(team_defense, qb_defense)

    assert summary.select("team_defense_sd").item() == pytest.approx(0.14142135623730953)
    assert summary.select("qb_defense_sd").item() == pytest.approx(0.07071067811865477)
    assert summary.select("qb_to_team_spread_ratio").item() == pytest.approx(0.5)
