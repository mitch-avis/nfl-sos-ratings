"""Tests for validation diagnostics helpers."""

from pathlib import Path

import polars as pl
import pytest

from nfl_sos_ratings.validation.diagnostics import (
    build_qb_adjustment_audit_frame,
    build_qb_leverage_profile_frame,
    build_qb_opponent_offense_frame,
    build_qb_split_half_frame,
    compute_qb_playoff_season_summary,
    compute_qb_playoff_validation_frame,
    compute_season_mae_deltas,
    compute_weekly_mae_curves,
    evaluate_qb_split_half_decision,
    summarize_defense_spread,
    summarize_qb_adjustment_slopes,
    summarize_qb_leverage_signal,
    summarize_qb_opponent_offense_signal,
    summarize_qb_split_half_signal,
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
            "opponent_team": ["DEF1", "DEF2", "DEF1", "DEF2"],
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


def test_build_qb_split_half_frame_aggregates_top_and_bottom_halves() -> None:
    """Split-half QB diagnostics should produce weighted raw, adjusted, and residual outputs."""
    qb_games = pl.DataFrame(
        {
            "season": [2025] * 8,
            "qb_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "qb_name": ["QB A", "QB A", "QB A", "QB A", "QB B", "QB B", "QB B", "QB B"],
            "team": ["TA", "TA", "TA", "TA", "TB", "TB", "TB", "TB"],
            "opponent_team": [
                "DEF1",
                "DEF2",
                "DEF3",
                "DEF4",
                "DEF1",
                "DEF2",
                "DEF3",
                "DEF4",
            ],
            "qb_dropbacks": [20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0],
            "qb_epa_per_dropback": [0.0, 0.1, 0.3, 0.4, -0.2, 0.0, 0.3, 0.5],
        }
    )
    defense_ratings = pl.DataFrame(
        {
            "team": ["DEF1", "DEF2", "DEF3", "DEF4"],
            "defense_coefficient": [0.3, 0.1, -0.1, -0.2],
        }
    )
    qb_meta = pl.DataFrame(
        {
            "qb_id": ["A", "B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["TA", "TB"],
            "qb_is_eligible": [True, True],
        }
    )

    split_frame = build_qb_split_half_frame(qb_games, defense_ratings, qb_meta=qb_meta).sort(
        "qb_id"
    )

    qb_a = split_frame.filter(pl.col("qb_id") == "A").row(0, named=True)
    qb_b = split_frame.filter(pl.col("qb_id") == "B").row(0, named=True)

    assert qb_a["faced_difficulty"] == pytest.approx(0.08333333333333333)
    assert qb_a["vs_top_half_raw_epa_per_dropback"] == pytest.approx(0.05)
    assert qb_a["vs_top_half_adjusted_epa_per_dropback"] == pytest.approx(0.25)
    assert qb_a["vs_top_half_residual"] == pytest.approx(0.016666666666666663)
    assert qb_a["vs_bottom_half_adjusted_epa_per_dropback"] == pytest.approx(0.2)
    assert qb_a["vs_top_half_dropbacks"] == pytest.approx(40.0)
    assert qb_b["faced_difficulty"] == pytest.approx(-0.03333333333333333)
    assert qb_b["vs_top_half_adjusted_epa_per_dropback"] == pytest.approx(0.1)
    assert qb_b["vs_top_half_residual"] == pytest.approx(-0.1)


def test_summarize_qb_split_half_signal_and_placebo_gate() -> None:
    """The split-half gate should reject a symmetric positive placebo effect."""
    split_frame = pl.DataFrame(
        {
            "season": [
                2020,
                2020,
                2020,
                2021,
                2021,
                2021,
                2022,
                2022,
                2022,
                2023,
                2023,
                2023,
                2024,
                2024,
                2024,
                2025,
                2025,
                2025,
            ],
            "faced_difficulty": [-1.0, 0.0, 1.0] * 6,
            "vs_top_half_residual": [
                -0.2,
                0.0,
                0.2,
                -0.18,
                0.0,
                0.18,
                -0.16,
                0.0,
                0.16,
                -0.14,
                0.0,
                0.14,
                -0.12,
                0.0,
                0.12,
                -0.1,
                0.0,
                0.1,
            ],
            "vs_top_half_dropbacks": [30.0] * 18,
            "vs_bottom_half_residual": [
                -0.18,
                0.0,
                0.18,
                -0.16,
                0.0,
                0.16,
                -0.14,
                0.0,
                0.14,
                -0.12,
                0.0,
                0.12,
                -0.1,
                0.0,
                0.1,
                -0.08,
                0.0,
                0.08,
            ],
            "vs_bottom_half_dropbacks": [30.0] * 18,
        }
    )

    primary = summarize_qb_split_half_signal(
        split_frame,
        residual_col="vs_top_half_residual",
        weight_col="vs_top_half_dropbacks",
        resamples=256,
        seed=0,
    )
    placebo = summarize_qb_split_half_signal(
        split_frame,
        residual_col="vs_bottom_half_residual",
        weight_col="vs_bottom_half_dropbacks",
        resamples=256,
        seed=0,
    )
    decision = evaluate_qb_split_half_decision(primary, placebo)

    assert primary.filter(pl.col("scope") == "pooled").select("slope").item() > 0.0
    assert primary.filter(pl.col("scope") == "season").height == 6
    assert decision["primary_gate_supported"] is True
    assert decision["placebo_is_symmetric"] is True
    assert decision["decision"] == "not_supported"


def test_compute_qb_playoff_season_summary_filters_to_eligible_qbs_and_adjusts_epa() -> None:
    """Playoff summaries should use regular-season defense coefficients and eligibility."""
    playoff_qb_games = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "qb_id": ["A", "A", "B"],
            "qb_name": ["QB A", "QB A", "QB B"],
            "team": ["TA", "TA", "TB"],
            "opponent_team": ["DEF1", "DEF2", "DEF1"],
            "qb_dropbacks": [10.0, 30.0, 20.0],
            "qb_epa_per_dropback": [0.0, 0.2, 0.3],
        }
    )
    defense_ratings = pl.DataFrame(
        {
            "team": ["DEF1", "DEF2"],
            "defense_coefficient": [0.3, -0.1],
        }
    )
    qb_meta = pl.DataFrame(
        {
            "qb_id": ["A", "B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["TA", "TB"],
            "qb_is_eligible": [True, False],
        }
    )

    summary = compute_qb_playoff_season_summary(playoff_qb_games, defense_ratings, qb_meta=qb_meta)

    assert summary.height == 1
    assert summary.select("qb_id").item() == "A"
    assert summary.select("playoff_dropbacks").item() == pytest.approx(40.0)
    assert summary.select("playoff_raw_epa_per_dropback").item() == pytest.approx(0.15)
    assert summary.select("playoff_adjusted_epa_per_dropback").item() == pytest.approx(0.15)


def test_compute_qb_playoff_validation_frame_canonicalizes_playoff_qb_ids_for_opponent_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playoff validation should retain opponent joins after QB identity canonicalization."""
    pl.DataFrame(
        {
            "qb_id": ["canon-a"],
            "qb_name": ["QB A"],
            "team": ["TA"],
            "qb_is_eligible": [True],
            "QSaCR": [1.0],
            "QSaOR": [0.8],
            "QRaw": [0.7],
            "qb_passer_rating": [100.0],
            "qb_any_a": [7.5],
        }
    ).write_parquet(tmp_path / "2025_qb_combined.parquet")
    pl.DataFrame(
        {
            "team": ["TB"],
            "adj_def_passing_epa_per_offensive_snap": [0.25],
        }
    ).write_parquet(tmp_path / "2025_simultaneous_team_adjustments.parquet")

    playoff_pbp = pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "week": [20, 20],
            "season_type": ["POST", "POST"],
            "posteam": ["TA", "TB"],
            "defteam": ["TB", "TA"],
            "passer_player_id": ["canon-a", "canon-b"],
            "passer_player_name": ["Q.A", "Q.B"],
            "qb_dropback": [1, 1],
            "pass": [1, 1],
            "complete_pass": [1, 0],
            "passing_yards": [12.0, 0.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [0, 0],
            "fumble_lost": [0, 0],
            "qb_epa": [0.4, -0.3],
            "cpoe": [2.0, -1.0],
            "yards_gained": [12.0, 0.0],
        }
    )
    crosswalk = pl.DataFrame(
        {
            "qb_id": ["canon-a", "canon-b"],
            "snap_player_id": [None, None],
            "qb_name": ["QB A", "QB B"],
            "qb_position": ["QB", "QB"],
        }
    )

    monkeypatch.setattr(
        "nfl_sos_ratings.validation.diagnostics.load_playoff_pbp_data",
        lambda season: playoff_pbp,
    )
    monkeypatch.setattr(
        "nfl_sos_ratings.validation.diagnostics.load_qb_identity_crosswalk",
        lambda season: crosswalk,
    )

    validation = compute_qb_playoff_validation_frame(tmp_path, [2025])

    assert validation.height == 1
    assert validation.select("qb_id").item() == "canon-a"
    assert validation.select("playoff_adjusted_epa_per_dropback").item() == pytest.approx(0.65)


def test_compute_qb_playoff_validation_frame_handles_pre_2006_null_qsacr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playoff validation should tolerate null pre-2006 QSaCR rows across seasons."""
    pl.DataFrame(
        {
            "qb_id": ["canon-a"],
            "qb_name": ["QB A"],
            "team": ["TA"],
            "qb_is_eligible": [True],
            "QSaCR": [None],
            "QSaOR": [0.5],
            "QRaw": [None],
            "qb_passer_rating": [92.0],
            "qb_any_a": [6.8],
        }
    ).write_parquet(tmp_path / "2005_qb_combined.parquet")
    pl.DataFrame(
        {
            "qb_id": ["canon-b"],
            "qb_name": ["QB B"],
            "team": ["TB"],
            "qb_is_eligible": [True],
            "QSaCR": [1.0],
            "QSaOR": [0.8],
            "QRaw": [0.7],
            "qb_passer_rating": [100.0],
            "qb_any_a": [7.5],
        }
    ).write_parquet(tmp_path / "2006_qb_combined.parquet")
    pl.DataFrame(
        {
            "team": ["TB"],
            "adj_def_passing_epa_per_offensive_snap": [0.20],
        }
    ).write_parquet(tmp_path / "2005_simultaneous_team_adjustments.parquet")
    pl.DataFrame(
        {
            "team": ["TA"],
            "adj_def_passing_epa_per_offensive_snap": [0.15],
        }
    ).write_parquet(tmp_path / "2006_simultaneous_team_adjustments.parquet")

    playoff_frames = {
        2005: pl.DataFrame(
            {
                "game_id": ["g1", "g1"],
                "week": [20, 20],
                "season_type": ["POST", "POST"],
                "posteam": ["TA", "TB"],
                "defteam": ["TB", "TA"],
                "passer_player_id": ["canon-a", "canon-x"],
                "passer_player_name": ["Q.A", "Q.X"],
                "qb_dropback": [1, 1],
                "pass": [1, 1],
                "complete_pass": [1, 0],
                "passing_yards": [10.0, 0.0],
                "pass_touchdown": [0, 0],
                "interception": [0, 0],
                "sack": [0, 0],
                "fumble_lost": [0, 0],
                "qb_epa": [0.3, -0.2],
                "cpoe": [None, None],
                "yards_gained": [10.0, 0.0],
            }
        ),
        2006: pl.DataFrame(
            {
                "game_id": ["g2", "g2"],
                "week": [20, 20],
                "season_type": ["POST", "POST"],
                "posteam": ["TB", "TA"],
                "defteam": ["TA", "TB"],
                "passer_player_id": ["canon-b", "canon-y"],
                "passer_player_name": ["Q.B", "Q.Y"],
                "qb_dropback": [1, 1],
                "pass": [1, 1],
                "complete_pass": [1, 0],
                "passing_yards": [12.0, 0.0],
                "pass_touchdown": [0, 0],
                "interception": [0, 0],
                "sack": [0, 0],
                "fumble_lost": [0, 0],
                "qb_epa": [0.4, -0.1],
                "cpoe": [2.0, -1.0],
                "yards_gained": [12.0, 0.0],
            }
        ),
    }
    crosswalk_frames = {
        2005: pl.DataFrame(
            {
                "qb_id": ["canon-a", "canon-x"],
                "snap_player_id": [None, None],
                "qb_name": ["QB A", "QB X"],
                "qb_position": ["QB", "QB"],
            }
        ),
        2006: pl.DataFrame(
            {
                "qb_id": ["canon-b", "canon-y"],
                "snap_player_id": [None, None],
                "qb_name": ["QB B", "QB Y"],
                "qb_position": ["QB", "QB"],
            }
        ),
    }

    monkeypatch.setattr(
        "nfl_sos_ratings.validation.diagnostics.load_playoff_pbp_data",
        lambda season: playoff_frames[int(season)],
    )
    monkeypatch.setattr(
        "nfl_sos_ratings.validation.diagnostics.load_qb_identity_crosswalk",
        lambda season: crosswalk_frames[int(season)],
    )

    validation = compute_qb_playoff_validation_frame(tmp_path, [2005, 2006]).sort("season")

    assert validation.height == 2
    assert validation.select("QSaCR").to_series().to_list() == [None, 1.0]


def test_build_qb_opponent_offense_frame_adjusts_for_defense_and_home_context() -> None:
    """Opponent-offense rows should keep game-level adjusted residuals and offense context."""
    qb_games = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "qb_id": ["A", "A", "B", "B"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B"],
            "team": ["TA", "TA", "TB", "TB"],
            "opponent_team": ["OFF1", "OFF2", "OFF1", "OFF2"],
            "qb_dropbacks": [20.0, 20.0, 20.0, 20.0],
            "qb_epa_per_dropback": [0.10, 0.30, -0.10, 0.10],
            "is_home": [True, False, False, True],
        }
    )
    offense_ratings = pl.DataFrame(
        {
            "team": ["OFF1", "OFF2"],
            "offense_rating": [0.20, -0.10],
        }
    )
    defense_ratings = pl.DataFrame(
        {
            "team": ["OFF1", "OFF2"],
            "defense_rating": [0.05, -0.05],
        }
    )

    frame = build_qb_opponent_offense_frame(
        qb_games,
        offense_ratings,
        defense_ratings,
        home_field_advantage=0.02,
    ).sort(["qb_id", "opponent_team"])

    qb_a_home = frame.filter((pl.col("qb_id") == "A") & (pl.col("opponent_team") == "OFF1")).row(
        0, named=True
    )
    qb_a_away = frame.filter((pl.col("qb_id") == "A") & (pl.col("opponent_team") == "OFF2")).row(
        0, named=True
    )

    assert qb_a_home["adjusted_game_epa_per_dropback"] == pytest.approx(0.13)
    assert qb_a_away["adjusted_game_epa_per_dropback"] == pytest.approx(0.27)
    assert qb_a_home["season_adjusted_epa_per_dropback"] == pytest.approx(0.20)
    assert qb_a_home["adjusted_residual"] == pytest.approx(-0.07)
    assert qb_a_away["adjusted_residual"] == pytest.approx(0.07)
    assert qb_a_home["opponent_offense_coefficient"] == pytest.approx(0.20)


def test_summarize_qb_opponent_offense_signal_reports_ci_and_sign_consistency() -> None:
    """Opponent-offense summaries should return pooled confidence intervals and sign counts."""
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
            "opponent_offense_coefficient": [-1.0, 0.0, 1.0] * 3,
            "adjusted_residual": [-0.2, 0.0, 0.2, -0.18, 0.0, 0.18, -0.16, 0.0, 0.16],
            "qb_dropbacks": [30.0] * 9,
        }
    )

    summary = summarize_qb_opponent_offense_signal(frame, resamples=256, seed=0)
    pooled = summary.filter(pl.col("scope") == "pooled").row(0, named=True)

    assert pooled["slope"] > 0.0
    assert pooled["ci_lower"] > 0.0
    assert pooled["direction_positive_count"] == 3
    assert pooled["direction_total_count"] == 3


def test_build_qb_leverage_profile_frame_computes_low_and_moderate_shares() -> None:
    """Leverage profiles should split dropbacks into low- and moderate-leverage shares."""
    dropback_plays = pl.DataFrame(
        {
            "season": [2025] * 6,
            "qb_id": ["A", "A", "A", "B", "B", "B"],
            "qb_name": ["QB A", "QB A", "QB A", "QB B", "QB B", "QB B"],
            "team": ["TA", "TA", "TA", "TB", "TB", "TB"],
            "wp": [0.01, 0.50, 0.99, 0.10, 0.20, 0.90],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2025, 2025],
            "qb_id": ["A", "B"],
            "schedule_softness": [0.30, -0.20],
        }
    )

    profile = build_qb_leverage_profile_frame(dropback_plays, schedule).sort("qb_id")

    qb_a = profile.filter(pl.col("qb_id") == "A").row(0, named=True)
    qb_b = profile.filter(pl.col("qb_id") == "B").row(0, named=True)

    assert qb_a["low_leverage_share"] == pytest.approx(2.0 / 3.0)
    assert qb_a["moderate_leverage_share"] == pytest.approx(1.0 / 3.0)
    assert qb_b["low_leverage_share"] == pytest.approx(0.0)
    assert qb_b["schedule_softness"] == pytest.approx(-0.20)


def test_summarize_qb_leverage_signal_reports_supported_direction() -> None:
    """Leverage summaries should report pooled support when low-leverage share tracks softness."""
    profile = pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
            "schedule_softness": [-1.0, 0.0, 1.0] * 3,
            "low_leverage_share": [0.1, 0.2, 0.3, 0.12, 0.22, 0.32, 0.14, 0.24, 0.34],
            "total_dropbacks": [100.0] * 9,
        }
    )

    summary = summarize_qb_leverage_signal(profile, resamples=256, seed=0)
    pooled = summary.filter(pl.col("scope") == "pooled").row(0, named=True)

    assert pooled["slope"] > 0.0
    assert pooled["ci_lower"] > 0.0
    assert pooled["direction_positive_count"] == 3
