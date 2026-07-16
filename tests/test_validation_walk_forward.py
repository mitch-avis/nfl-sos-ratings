"""Tests for walk-forward validation helpers."""

import math

import polars as pl
import pytest

from nfl_sos_ratings.validation.snapshots import build_team_rating_snapshot
from nfl_sos_ratings.validation.walk_forward import (
    EloConfig,
    build_elo_feature_rows,
    build_play_level_team_training_rows_with_special_teams,
    build_raw_epa_feature_rows,
    build_rolling_team_weight_maps,
    build_snapshot_feature_rows,
    build_srs_feature_rows,
    build_validation_report_text,
    build_weighted_team_feature_rows,
    compute_pairwise_mae_bootstrap,
    compute_qbr_correlations,
    compute_stability_metrics,
    evaluate_feature_rows,
    run_play_level_team_special_teams_backtest,
    run_walk_forward_backtest,
    score_prediction_rows,
)


def _weekly_team_rows() -> pl.DataFrame:
    """Build a compact three-week team-game fixture."""
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "week": [1, 1, 2, 2, 3, 3],
            "team": ["A", "B", "B", "A", "A", "B"],
            "opponent_team": ["B", "A", "A", "B", "B", "A"],
            "is_home": [True, False, True, False, True, False],
            "point_margin": [7.0, -7.0, -3.0, 3.0, 10.0, -10.0],
            "epa_margin_per_play": [0.12, -0.12, -0.04, 0.04, 0.18, -0.18],
            "passing_epa_per_offensive_snap": [0.24, -0.08, 0.02, 0.16, 0.32, -0.14],
            "rushing_epa_per_offensive_snap": [0.10, -0.02, 0.01, 0.08, 0.12, -0.05],
        }
    )


def test_build_snapshot_feature_rows_uses_week_specific_pregame_snapshot() -> None:
    """Verify snapshot feature rows join home and away ratings from the right cutoff."""
    weekly_df = _weekly_team_rows()

    feature_rows = build_snapshot_feature_rows(
        weekly_df,
        season=2025,
        baseline_name="SaOvR",
        snapshot_builder=build_team_rating_snapshot,
        rating_column="SaOvR",
    )
    week_three_row = feature_rows.filter(pl.col("week") == 3).row(0, named=True)
    snapshot = build_team_rating_snapshot(weekly_df, cutoff_week=3).sort("team")
    manual_diff = (
        snapshot.filter(pl.col("team") == "A").select("SaOvR").item()
        - snapshot.filter(pl.col("team") == "B").select("SaOvR").item()
    )

    assert week_three_row["home_team"] == "A"
    assert week_three_row["away_team"] == "B"
    assert week_three_row["rating_diff"] == pytest.approx(manual_diff)
    assert week_three_row["home_margin"] == 10.0


def test_build_srs_feature_rows_uses_prior_games_only() -> None:
    """Verify the SRS baseline uses only pre-cutoff game results for each week."""
    weekly_df = _weekly_team_rows()

    feature_rows = build_srs_feature_rows(weekly_df, season=2025)
    week_three_row = feature_rows.filter(pl.col("week") == 3).row(0, named=True)

    assert week_three_row["rating_diff"] == pytest.approx(5.0)
    assert week_three_row["home_margin"] == 10.0


def test_build_raw_epa_feature_rows_uses_pre_cutoff_team_means() -> None:
    """Verify the raw-EPA baseline is the pregame mean EPA margin differential."""
    weekly_df = _weekly_team_rows()

    feature_rows = build_raw_epa_feature_rows(weekly_df, season=2025)
    week_three_row = feature_rows.filter(pl.col("week") == 3).row(0, named=True)

    assert week_three_row["rating_diff"] == pytest.approx(0.16)
    assert week_three_row["home_margin"] == 10.0


def test_build_weighted_team_feature_rows_uses_weighted_snapshot() -> None:
    """Verify the weighted team feature rows use the supplied rolling weight map."""
    weekly_df = _weekly_team_rows()

    feature_rows = build_weighted_team_feature_rows(
        weekly_df,
        season=2025,
        weight_map={
            "adj_off_passing_epa_per_offensive_snap": 1.0,
            "adj_off_rushing_epa_per_offensive_snap": 0.0,
            "adj_def_passing_epa_per_offensive_snap": 0.0,
            "adj_def_rushing_epa_per_offensive_snap": 0.0,
        },
        baseline_name="T1Weighted",
    )
    week_three_row = feature_rows.filter(pl.col("week") == 3).row(0, named=True)

    assert week_three_row["baseline"] == "T1Weighted"
    assert week_three_row["home_margin"] == 10.0
    assert week_three_row["rating_diff"] > 0.0


def test_build_rolling_team_weight_maps_uses_only_prior_season_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling team weights should fit on season pairs that end before the target season."""
    training_rows = pl.DataFrame(
        {
            "season": [2022, 2023],
            "next_season": [2023, 2024],
            "team": ["A", "B"],
            "adj_off_passing_epa_per_offensive_snap": [0.1, 0.2],
            "adj_off_rushing_epa_per_offensive_snap": [0.0, 0.1],
            "adj_def_passing_epa_per_offensive_snap": [0.1, 0.2],
            "adj_def_rushing_epa_per_offensive_snap": [0.0, 0.1],
            "target": [0.2, 0.3],
        }
    )
    captured: dict[str, object] = {}

    def fake_fit_linear_weights(
        df: pl.DataFrame,
        feature_columns: list[str],
        target_column: str,
        sample_weight_column: str | None = None,
    ) -> dict[str, float]:
        captured["next_seasons"] = df.select("next_season").to_series().to_list()
        captured["feature_columns"] = feature_columns
        captured["target_column"] = target_column
        captured["sample_weight_column"] = sample_weight_column
        return {column: float(index + 1) for index, column in enumerate(feature_columns)}

    monkeypatch.setattr(
        "nfl_sos_ratings.validation.walk_forward.composite_weights.fit_linear_weights",
        fake_fit_linear_weights,
    )

    weight_maps = build_rolling_team_weight_maps(training_rows, seasons=[2023, 2024, 2025])

    assert weight_maps[2023]["adj_off_passing_epa_per_offensive_snap"] == pytest.approx(0.25)
    assert captured["next_seasons"] == [2023, 2024]
    assert weight_maps[2025]["adj_def_rushing_epa_per_offensive_snap"] == pytest.approx(0.4)


def test_build_elo_feature_rows_updates_week_to_week() -> None:
    """Verify the Elo baseline carries team ratings forward within a season."""
    home_games = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "game_id": ["g1", "g2"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_margin": [10.0, -3.0],
        }
    )

    feature_rows = build_elo_feature_rows(
        home_games,
        config=EloConfig(
            initial_rating=1500.0,
            k_factor=20.0,
            home_field_elo=0.0,
            regression_to_mean=0.5,
            use_margin_multiplier=False,
        ),
    )

    assert feature_rows.filter(pl.col("week") == 1).select("rating_diff").item() == 0.0
    assert feature_rows.filter(pl.col("week") == 2).select("rating_diff").item() == pytest.approx(
        -20.0
    )


def test_build_elo_feature_rows_regresses_between_seasons() -> None:
    """Verify the Elo baseline regresses ratings toward 1500 at season boundaries."""
    home_games = pl.DataFrame(
        {
            "season": [2024, 2025],
            "week": [1, 1],
            "game_id": ["g1", "g2"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_margin": [10.0, 0.0],
        }
    )

    feature_rows = build_elo_feature_rows(
        home_games,
        config=EloConfig(
            initial_rating=1500.0,
            k_factor=20.0,
            home_field_elo=0.0,
            regression_to_mean=0.5,
            use_margin_multiplier=False,
        ),
    )

    assert feature_rows.filter(pl.col("season") == 2025).select(
        "rating_diff"
    ).item() == pytest.approx(10.0)


def test_evaluate_feature_rows_fits_only_on_prior_weeks() -> None:
    """Verify target-week actuals cannot leak into the fitted projection coefficients."""
    feature_rows = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 4],
            "baseline": ["SaOvR", "SaOvR", "SaOvR", "SaOvR"],
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_team": ["A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B"],
            "rating_diff": [1.0, -1.0, 2.0, -2.0],
            "home_margin": [5.0, -1.0, 999.0, -999.0],
        }
    )
    perturbed = feature_rows.with_columns(
        pl.when(pl.col("week") >= 3)
        .then(pl.col("home_margin") * -1.0)
        .otherwise(pl.col("home_margin"))
        .alias("home_margin")
    )

    baseline_predictions = evaluate_feature_rows(feature_rows, start_week=3)
    perturbed_predictions = evaluate_feature_rows(perturbed, start_week=3)

    baseline_week_three = baseline_predictions.filter(pl.col("week") == 3).row(0, named=True)
    perturbed_week_three = perturbed_predictions.filter(pl.col("week") == 3).row(0, named=True)

    assert baseline_week_three["training_row_count"] == 2
    assert baseline_week_three["predicted_margin"] == pytest.approx(8.0)
    assert baseline_week_three["predicted_margin"] == pytest.approx(
        perturbed_week_three["predicted_margin"]
    )
    assert baseline_week_three["fitted_k"] == pytest.approx(3.0)
    assert baseline_week_three["fitted_hfa_points"] == pytest.approx(2.0)


def test_score_prediction_rows_reports_overall_and_early_late_splits() -> None:
    """Verify the scoring helper returns overall and early/late MAE and RMSE."""
    predictions = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [5, 7, 8, 10],
            "baseline": ["SaOvR", "SaOvR", "SaOvR", "SaOvR"],
            "predicted_margin": [3.0, 1.0, 5.0, -1.0],
            "home_margin": [1.0, 2.0, 1.0, -4.0],
        }
    )

    metrics = score_prediction_rows(predictions)

    assert metrics.filter(pl.col("split") == "overall").select("mae").item() == pytest.approx(2.5)
    assert metrics.filter(pl.col("split") == "early").select("mae").item() == pytest.approx(1.5)
    assert metrics.filter(pl.col("split") == "late").select("mae").item() == pytest.approx(3.5)

    overall_rmse = metrics.filter(pl.col("split") == "overall").select("rmse").item()
    assert math.isclose(overall_rmse, math.sqrt(7.5), rel_tol=1e-9)


def test_compute_pairwise_mae_bootstrap_returns_deterministic_delta_intervals() -> None:
    """Verify paired-game bootstrap deltas report stable confidence intervals."""
    predictions = pl.DataFrame(
        {
            "season": [2025] * 8,
            "week": [5, 5, 6, 6, 5, 5, 6, 6],
            "baseline": ["SaOvR", "SaOvR", "SaOvR", "SaOvR", "SRS", "SRS", "SRS", "SRS"],
            "game_id": ["g1", "g2", "g3", "g4", "g1", "g2", "g3", "g4"],
            "home_team": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H", "E", "F", "G", "H"],
            "predicted_margin": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
            "home_margin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).with_columns((pl.col("predicted_margin") - pl.col("home_margin")).alias("error"))

    deltas = compute_pairwise_mae_bootstrap(predictions, baselines=["SaOvR", "SRS"], resamples=128)
    overall_row = deltas.filter(pl.col("split") == "overall").row(0, named=True)

    assert overall_row["baseline_a"] == "SaOvR"
    assert overall_row["baseline_b"] == "SRS"
    assert overall_row["mae_delta"] == pytest.approx(-2.0)
    assert overall_row["ci_lower"] == pytest.approx(-2.0)
    assert overall_row["ci_upper"] == pytest.approx(-2.0)
    assert overall_row["probability_baseline_a_not_worse"] == pytest.approx(1.0)
    assert overall_row["distinguishable_from_zero"] is True


def test_build_validation_report_text_includes_bootstrap_delta_section() -> None:
    """Verify the report renders paired-bootstrap MAE deltas."""
    metrics = pl.DataFrame(
        {
            "baseline": ["SaOvR", "SRS"],
            "season": [None, None],
            "split": ["overall", "overall"],
            "games": [10, 10],
            "mae": [7.0, 7.5],
            "rmse": [9.0, 9.4],
        }
    )
    stability = pl.DataFrame(
        {
            "entity": ["qb", "team"],
            "metric": ["QSaCR", "SaOvR"],
            "paired_rows": [100, 200],
            "pearson": [0.61, 0.42],
            "spearman": [0.60, 0.40],
        }
    )
    qbr = pl.DataFrame(
        {
            "season": [2006],
            "joined_rows": [24],
            "pearson": [0.72],
            "spearman": [0.69],
        }
    )
    deltas = pl.DataFrame(
        {
            "baseline_a": ["SaOvR"],
            "baseline_b": ["SRS"],
            "split": ["overall"],
            "games": [10],
            "mae_delta": [-0.5],
            "ci_lower": [-0.8],
            "ci_upper": [-0.2],
            "probability_baseline_a_not_worse": [0.94],
            "distinguishable_from_zero": [True],
        }
    )

    report = build_validation_report_text(
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr,
        mae_deltas=deltas,
        seasons=[1999, 2000],
        start_week=5,
        command="uv run python -m nfl_sos_ratings.validation.walk_forward --start-week 5",
    )

    assert "## Paired Bootstrap MAE Deltas" in report
    assert (
        "| Baseline A | Baseline B | Split | Games | MAE Delta | CI Lower | CI Upper | "
        "P(A<=B) | Distinguishable |" in report
    )
    assert "| SaOvR | SRS | overall | 10 | -0.500 | -0.800 | -0.200 | 0.940 | True |" in report


def test_run_walk_forward_backtest_stacks_all_team_baselines(tmp_path) -> None:
    """Verify the orchestration path evaluates SaOvR, SRS, raw EPA, and Elo together."""
    _weekly_team_rows().write_parquet(tmp_path / "2025_team_game_logs.parquet")

    predictions, metrics = run_walk_forward_backtest(
        tmp_path,
        seasons=[2025],
        start_week=3,
        elo_config=EloConfig(home_field_elo=0.0, use_margin_multiplier=False),
    )

    assert set(predictions.select("baseline").to_series().to_list()) == {
        "Elo",
        "RawEPA",
        "SRS",
        "SaOvR",
    }
    assert predictions.select("week").min().item() == 3
    assert set(
        metrics.filter(pl.col("split") != "season").select("split").to_series().to_list()
    ) == {"overall", "early", "late"}


def test_run_play_level_team_special_teams_backtest_builds_t4_baseline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the T4 orchestration path evaluates a play-level weighted team baseline."""
    _weekly_team_rows().write_parquet(tmp_path / "2025_team_game_logs.parquet")
    pbp = pl.DataFrame(
        {
            "game_id": [
                "g1",
                "g1",
                "g1",
                "g1",
                "g2",
                "g2",
                "g2",
                "g2",
                "g3",
                "g3",
                "g3",
                "g3",
            ],
            "week": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "posteam": ["A", "B", "A", "B", "B", "A", "B", "A", "A", "B", "A", "B"],
            "defteam": ["B", "A", "B", "A", "A", "B", "A", "B", "B", "A", "B", "A"],
            "home_team": ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B", "A", "A", "A", "A", "B", "B", "B", "B"],
            "qb_dropback": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            "rush": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "qb_kneel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "special": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "epa": [0.3, -0.2, -0.1, 0.1, -0.1, 0.2, 0.2, -0.2, 0.4, -0.3, 0.1, -0.1],
        }
    )
    monkeypatch.setattr(
        "nfl_sos_ratings.validation.walk_forward.load_pbp_data",
        lambda season: pbp,
    )

    predictions, metrics, weight_maps = run_play_level_team_special_teams_backtest(
        tmp_path,
        seasons=[2025],
        start_week=3,
    )

    assert set(predictions.select("baseline").to_series().to_list()) == {"T4Weighted"}
    assert predictions.select("week").min().item() == 3
    assert set(
        metrics.filter(pl.col("split") != "season").select("split").to_series().to_list()
    ) == {"overall", "early", "late"}
    assert 2025 in weight_maps


def test_build_play_level_team_training_rows_with_special_teams_uses_next_season_targets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Play-level T4 training rows should pair current-season features with next-season targets."""
    pl.DataFrame({"team": ["A", "B"], "SaOvR": [1.0, -1.0]}).write_parquet(
        tmp_path / "2025_combined.parquet"
    )
    pl.DataFrame({"team": ["A", "B"], "SaOvR": [0.5, -0.5]}).write_parquet(
        tmp_path / "2026_combined.parquet"
    )
    pbp = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "week": [1, 1, 1, 1],
            "posteam": ["A", "B", "A", "B"],
            "defteam": ["B", "A", "B", "A"],
            "home_team": ["A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B"],
            "qb_dropback": [1, 1, 0, 0],
            "rush": [0, 0, 1, 1],
            "qb_kneel": [0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0],
            "special": [0, 0, 1, 1],
            "epa": [0.3, -0.2, 0.1, -0.1],
        }
    )
    monkeypatch.setattr(
        "nfl_sos_ratings.validation.walk_forward.load_pbp_data",
        lambda season: pbp,
    )

    training_rows = build_play_level_team_training_rows_with_special_teams(
        tmp_path,
        seasons=[2025, 2026],
    )

    assert training_rows.select("season").to_series().to_list() == [2025, 2025]
    assert training_rows.select("next_season").to_series().to_list() == [2026, 2026]
    assert training_rows.select("target").to_series().to_list() == pytest.approx([0.5, -0.5])


def test_compute_stability_metrics_matches_adjacent_season_pairs(tmp_path) -> None:
    """Verify stability metrics match consecutive-season team and QB joins."""
    pl.DataFrame(
        {
            "qb_id": ["A", "B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["X", "Y"],
            "qb_is_eligible": [True, True],
            "QSaCR": [1.0, -1.0],
            "qb_passer_rating": [100.0, 80.0],
            "qb_any_a": [7.0, 5.0],
        }
    ).write_parquet(tmp_path / "2006_qb_combined.parquet")
    pl.DataFrame(
        {
            "qb_id": ["A", "B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["X", "Y"],
            "qb_is_eligible": [True, True],
            "QSaCR": [2.0, -2.0],
            "qb_passer_rating": [110.0, 70.0],
            "qb_any_a": [8.0, 4.0],
        }
    ).write_parquet(tmp_path / "2007_qb_combined.parquet")
    pl.DataFrame({"team": ["X", "Y"], "SaOvR": [1.5, -1.5]}).write_parquet(
        tmp_path / "2006_combined.parquet"
    )
    pl.DataFrame({"team": ["X", "Y"], "SaOvR": [2.5, -2.5]}).write_parquet(
        tmp_path / "2007_combined.parquet"
    )

    stability = compute_stability_metrics(tmp_path, seasons=[2006, 2007])

    assert set(stability.select("metric").to_series().to_list()) == {
        "QSaCR",
        "SaOvR",
        "qb_any_a",
        "qb_passer_rating",
    }
    assert all(
        value == pytest.approx(1.0)
        for value in stability.select("pearson").to_series().to_list()
        + stability.select("spearman").to_series().to_list()
    )


def test_compute_qbr_correlations_joins_qbs_by_team_and_name(tmp_path, monkeypatch) -> None:
    """Verify QBR correlations join season-end QB rows to the loaded ESPN reference."""
    pl.DataFrame(
        {
            "qb_id": ["A", "B"],
            "qb_name": ["DJ Smith", "Pat O'Brien"],
            "team": ["KC", "WAS"],
            "qb_is_eligible": [True, True],
            "QSaCR": [1.0, -1.0],
        }
    ).write_parquet(tmp_path / "2006_qb_combined.parquet")

    monkeypatch.setattr(
        "nfl_sos_ratings.validation.walk_forward.load_espn_qbr",
        lambda level, seasons: pl.DataFrame(
            {
                "season": [2006, 2006],
                "team_abb": ["KC", "WAS"],
                "name_display": ["D.J. Smith", "Pat OBrien"],
                "qbr_total": [80.0, 20.0],
            }
        ),
    )

    correlations = compute_qbr_correlations(tmp_path, seasons=[2006])

    assert correlations.select("joined_rows").item() == 2
    assert correlations.select("pearson").item() == pytest.approx(1.0)
    assert correlations.select("spearman").item() == pytest.approx(1.0)


def test_build_validation_report_text_includes_command_tables_and_sacr_caveat() -> None:
    """Verify the report renderer includes the required Stage 3 sections."""
    metrics = pl.DataFrame(
        {
            "baseline": ["SaOvR", "Elo"],
            "season": [None, None],
            "split": ["overall", "overall"],
            "games": [10, 10],
            "mae": [7.0, 7.5],
            "rmse": [9.0, 9.4],
        }
    )
    stability = pl.DataFrame(
        {
            "entity": ["qb", "team"],
            "metric": ["QSaCR", "SaOvR"],
            "paired_rows": [100, 200],
            "pearson": [0.61, 0.42],
            "spearman": [0.60, 0.40],
        }
    )
    qbr = pl.DataFrame(
        {
            "season": [2006],
            "joined_rows": [24],
            "pearson": [0.72],
            "spearman": [0.69],
        }
    )

    report = build_validation_report_text(
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr,
        seasons=[1999, 2000],
        start_week=5,
        command="uv run python -m nfl_sos_ratings.validation.walk_forward --start-week 5",
    )

    assert "# Validation Report" in report
    assert "## Acceptance Check" in report
    assert "uv run python -m nfl_sos_ratings.validation.walk_forward --start-week 5" in report
    assert "SaCR may be evaluated as a secondary line" in report
    assert "| Baseline | Split | Games | MAE | RMSE |" in report
    assert "| Metric | Entity | Paired Rows | Pearson | Spearman |" in report
    assert "| Season | Joined Rows | Pearson | Spearman |" in report


def test_build_validation_report_text_can_render_stage3c_and_qb_status_sections() -> None:
    """Verify the report renderer can include Stage 3c and QB-open-status narrative blocks."""
    metrics = pl.DataFrame(
        {
            "baseline": ["SaOvR"],
            "season": [None],
            "split": ["overall"],
            "games": [10],
            "mae": [7.0],
            "rmse": [9.0],
        }
    )
    stability = pl.DataFrame(
        {
            "entity": ["team"],
            "metric": ["SaOvR"],
            "paired_rows": [200],
            "pearson": [0.42],
            "spearman": [0.40],
        }
    )
    qbr = pl.DataFrame(
        {
            "season": [2006],
            "joined_rows": [24],
            "pearson": [0.72],
            "spearman": [0.69],
        }
    )

    report = build_validation_report_text(
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr,
        seasons=[1999, 2000],
        start_week=5,
        command="uv run python -m nfl_sos_ratings.validation.walk_forward --start-week 5",
        stage3c_lines=["## Stage 3c Decision Rule", "", "- Candidate rule line."],
        qb_open_status_lines=["## QB Open Status", "", "- Stage 3d next."],
    )

    assert "## Stage 3c Decision Rule" in report
    assert "Candidate rule line." in report
    assert "## QB Open Status" in report
    assert "Stage 3d next." in report
