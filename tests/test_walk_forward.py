"""Tests for the walk-forward validation of the team rating and the QB checks."""

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.validation import walk_forward
from nfl_sos_ratings.validation.report import ValidationReportInputs, build_validation_report_text
from nfl_sos_ratings.validation.walk_forward import (
    TEAM_RATING_BASELINE,
    EloConfig,
    build_elo_feature_rows,
    build_team_rating_feature_rows,
    compute_pairwise_mae_bootstrap,
    compute_qbr_correlations,
    compute_stability_metrics,
    evaluate_feature_rows,
    evaluate_team_decision,
    score_prediction_rows,
)
from tests.stubs import stub

if TYPE_CHECKING:
    from pathlib import Path

_FIXED_ELO = EloConfig(
    initial_rating=1500.0,
    k_factor=20.0,
    home_field_elo=0.0,
    regression_to_mean=0.5,
    use_margin_multiplier=False,
)
# (week, home, away, home EPA per play, away EPA per play, home margin)
_GAMES = (
    (1, "A", "B", 0.10, -0.05, 7),
    (1, "C", "D", 0.02, 0.01, 3),
    (2, "B", "C", 0.04, 0.00, 1),
    (2, "D", "A", -0.06, 0.12, -10),
    (3, "A", "C", 0.08, -0.02, 6),
    (3, "B", "D", 0.03, 0.05, -2),
)


def _team_game_logs() -> pl.DataFrame:
    """Return three weeks of team-game rows with every column the team rating reads."""
    rows: list[dict[str, object]] = []
    for week, home, away, home_epa, away_epa, margin in _GAMES:
        game_id = f"w{week}_{home}_{away}"
        for team, opponent, is_home, epa, team_margin in (
            (home, away, True, home_epa, margin),
            (away, home, False, away_epa, -margin),
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "point_margin": team_margin,
                    "offensive_snaps": 60,
                    "offensive_epa": epa * 60,
                    "st_plays": 12,
                    "st_epa": 0.1 if is_home else -0.1,
                }
            )
    return pl.DataFrame(rows)


def _delta_rows(*rows: tuple[str, str, float, float, float]) -> pl.DataFrame:
    """Return overall-split bootstrap rows from (a, b, delta, ci_lower, ci_upper) tuples."""
    return pl.DataFrame(
        [
            {
                "baseline_a": a,
                "baseline_b": b,
                "split": "overall",
                "games": 100,
                "mae_delta": delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "probability_baseline_a_not_worse": 0.5,
                "distinguishable_from_zero": lower > 0.0 or upper < 0.0,
            }
            for a, b, delta, lower, upper in rows
        ]
    )


def test_build_elo_feature_rows_updates_week_to_week() -> None:
    # Arrange
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

    # Act
    feature_rows = build_elo_feature_rows(home_games, config=_FIXED_ELO)

    # Assert
    assert feature_rows.get_column("rating_diff").to_list() == pytest.approx([0.0, -20.0])


def test_build_elo_feature_rows_regresses_between_seasons() -> None:
    # Arrange
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

    # Act
    feature_rows = build_elo_feature_rows(home_games, config=_FIXED_ELO)

    # Assert
    second_season = feature_rows.filter(pl.col("season") == 2025)
    assert second_season.get_column("rating_diff").item() == pytest.approx(10.0)


def test_build_team_rating_feature_rows_ignores_the_predicted_week_and_later() -> None:
    # Arrange
    game_logs = _team_game_logs()
    perturbed = game_logs.with_columns(
        pl.when(pl.col("week") >= 3)
        .then(pl.col("offensive_epa") * -5.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    baseline = build_team_rating_feature_rows(game_logs, 2025).filter(pl.col("week") == 3)

    # Act
    result = build_team_rating_feature_rows(perturbed, 2025).filter(pl.col("week") == 3)

    # Assert
    assert result.get_column("rating_diff").to_list() == pytest.approx(
        baseline.get_column("rating_diff").to_list()
    )


def test_build_team_rating_feature_rows_labels_the_team_rating_baseline() -> None:
    # Arrange
    game_logs = _team_game_logs()

    # Act
    feature_rows = build_team_rating_feature_rows(game_logs, 2025)

    # Assert
    assert feature_rows.get_column("baseline").unique().to_list() == [TEAM_RATING_BASELINE]


def test_evaluate_feature_rows_fits_only_on_prior_weeks() -> None:
    # Arrange
    feature_rows = pl.DataFrame(
        {
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "baseline": ["X"] * 4,
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_team": ["A"] * 4,
            "away_team": ["B"] * 4,
            "rating_diff": [1.0, -1.0, 2.0, -2.0],
            "home_margin": [5.0, -1.0, 999.0, -999.0],
        }
    )

    # Act
    predictions = evaluate_feature_rows(feature_rows, start_week=3)

    # Assert
    week_three = predictions.filter(pl.col("week") == 3).row(0, named=True)
    assert (week_three["fitted_k"], week_three["fitted_hfa_points"]) == pytest.approx((3.0, 2.0))
    assert week_three["predicted_margin"] == pytest.approx(8.0)


def test_score_prediction_rows_reports_overall_and_early_late_splits() -> None:
    # Arrange
    predictions = pl.DataFrame(
        {
            "season": [2024] * 4,
            "week": [5, 7, 8, 10],
            "baseline": ["X"] * 4,
            "predicted_margin": [3.0, 1.0, 5.0, -1.0],
            "home_margin": [1.0, 2.0, 1.0, -4.0],
        }
    )

    # Act
    metrics = score_prediction_rows(predictions)

    # Assert
    mae = {
        split: metrics.filter(pl.col("split") == split).get_column("mae").item()
        for split in ("overall", "early", "late")
    }
    assert mae == pytest.approx({"overall": 2.5, "early": 1.5, "late": 3.5})
    overall_rmse = metrics.filter(pl.col("split") == "overall").get_column("rmse").item()
    assert math.isclose(overall_rmse, math.sqrt(7.5), rel_tol=1e-9)


def test_compute_pairwise_mae_bootstrap_returns_deterministic_delta_intervals() -> None:
    # Arrange
    predictions = pl.DataFrame(
        {
            "season": [2025] * 8,
            "week": [5, 5, 6, 6] * 2,
            "baseline": ["X"] * 4 + ["SRS"] * 4,
            "game_id": ["g1", "g2", "g3", "g4"] * 2,
            "home_team": ["A", "B", "C", "D"] * 2,
            "away_team": ["E", "F", "G", "H"] * 2,
            "predicted_margin": [1.0] * 4 + [3.0] * 4,
            "home_margin": [0.0] * 8,
        }
    ).with_columns((pl.col("predicted_margin") - pl.col("home_margin")).alias("error"))

    # Act
    deltas = compute_pairwise_mae_bootstrap(predictions, baselines=["X", "SRS"], resamples=128)

    # Assert
    overall = deltas.filter(pl.col("split") == "overall").row(0, named=True)
    assert (overall["mae_delta"], overall["ci_lower"], overall["ci_upper"]) == pytest.approx(
        (-2.0, -2.0, -2.0)
    )
    assert overall["distinguishable_from_zero"] is True


def test_evaluate_team_decision_parity_with_both_comparators_adopts() -> None:
    # Arrange
    deltas = _delta_rows(
        (TEAM_RATING_BASELINE, "RawEPA", 0.02, -0.03, 0.07),
        (TEAM_RATING_BASELINE, "SRS", -0.01, -0.06, 0.04),
    )

    # Act
    decision = evaluate_team_decision(deltas)

    # Assert
    assert decision.adopted is True


def test_evaluate_team_decision_significantly_worse_than_one_comparator_rejects() -> None:
    # Arrange
    deltas = _delta_rows(
        (TEAM_RATING_BASELINE, "RawEPA", 0.02, -0.03, 0.07),
        (TEAM_RATING_BASELINE, "SRS", 0.09, 0.03, 0.15),
    )

    # Act
    decision = evaluate_team_decision(deltas)

    # Assert
    assert decision.adopted is False


def test_evaluate_team_decision_reads_comparisons_listed_in_either_order() -> None:
    # Arrange
    deltas = _delta_rows(
        ("RawEPA", TEAM_RATING_BASELINE, 0.10, 0.04, 0.16),
        (TEAM_RATING_BASELINE, "SRS", -0.01, -0.06, 0.04),
    )

    # Act
    decision = evaluate_team_decision(deltas)

    # Assert
    raw_epa = next(result for result in decision.comparisons if result.comparator == "RawEPA")
    assert (raw_epa.mae_delta, raw_epa.ci_lower, raw_epa.ci_upper) == pytest.approx(
        (-0.10, -0.16, -0.04)
    )


def test_evaluate_team_decision_missing_comparator_raises_value_error() -> None:
    # Arrange
    deltas = _delta_rows((TEAM_RATING_BASELINE, "SRS", -0.01, -0.06, 0.04))

    # Act & Assert
    with pytest.raises(ValueError, match="RawEPA"):
        evaluate_team_decision(deltas)


def _report_inputs() -> ValidationReportInputs:
    """Return small but complete report inputs."""
    deltas = _delta_rows(
        (TEAM_RATING_BASELINE, "RawEPA", 0.02, -0.03, 0.07),
        (TEAM_RATING_BASELINE, "SRS", -0.01, -0.06, 0.04),
        ("RawEPA", "SRS", -0.03, -0.05, -0.01),
    )
    metrics = pl.DataFrame(
        {
            "baseline": [TEAM_RATING_BASELINE, "SRS"],
            "season": [None, None],
            "split": ["overall", "overall"],
            "games": [10, 10],
            "mae": [10.5, 10.6],
            "rmse": [13.5, 13.6],
        }
    )
    return ValidationReportInputs(
        command="nfl-sos-ratings validate",
        seasons=[2024, 2025],
        start_week=5,
        metrics=metrics,
        mae_deltas=deltas,
        weekly_curves=pl.DataFrame(
            {"baseline": ["SRS"], "week": [5], "games": [10], "mae": [10.0], "rmse": [13.0]}
        ),
        decision=evaluate_team_decision(deltas),
        stability=pl.DataFrame(
            {
                "entity": ["team"],
                "metric": ["team_rating"],
                "paired_rows": [32],
                "pearson": [0.5],
                "spearman": [0.5],
            }
        ),
        qbr_correlations=pl.DataFrame(
            {"season": [2025], "joined_rows": [30], "pearson": [0.9], "spearman": [0.88]}
        ),
    )


def test_build_validation_report_text_states_the_decision() -> None:
    # Arrange
    inputs = _report_inputs()

    # Act
    text = build_validation_report_text(inputs)

    # Assert
    assert "Decision: adopt" in text


def test_build_validation_report_text_lists_every_distinguishable_interval() -> None:
    # Arrange
    inputs = _report_inputs()

    # Act
    text = build_validation_report_text(inputs)

    # Assert
    assert "RawEPA vs SRS" in text


def test_build_validation_report_text_keeps_prose_within_the_line_limit() -> None:
    # Arrange
    inputs = _report_inputs()

    # Act
    text = build_validation_report_text(inputs)

    # Assert
    long_prose = [
        line for line in text.splitlines() if len(line) > 100 and not line.startswith(("|", "nfl"))
    ]
    assert long_prose == []


def _write_season_files(tmp_path: Path, season: int, offset: float) -> None:
    """Write minimal ratings and QB files for one season of the stability check."""
    pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "team_rating": [1.0 + offset, 0.0, -1.0],
            "SRS": [2.0, 0.0, -2.0 + offset],
        }
    ).write_parquet(tmp_path / f"{season}_ratings.parquet")
    pl.DataFrame(
        {
            "qb_id": ["q1", "q2", "q3"],
            "qb_name": ["One", "Two", "Three"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "adj_qb_epa_per_dropback": [0.2, 0.1 + offset, 0.0],
            "qb_passer_rating": [100.0, 90.0, 80.0],
            "qb_any_a": [7.0, 6.0, 5.0 + offset],
        }
    ).write_parquet(tmp_path / f"{season}_qb_combined.parquet")


def test_compute_stability_metrics_reports_team_and_qb_metrics(tmp_path: Path) -> None:
    # Arrange
    _write_season_files(tmp_path, 2024, 0.0)
    _write_season_files(tmp_path, 2025, 0.0)

    # Act
    stability = compute_stability_metrics(tmp_path, [2024, 2025])

    # Assert
    assert dict(zip(stability["metric"], stability["pearson"], strict=True)) == pytest.approx(
        {
            "SRS": 1.0,
            "adj_qb_epa_per_dropback": 1.0,
            "qb_any_a": 1.0,
            "qb_passer_rating": 1.0,
            "team_rating": 1.0,
        }
    )


def test_compute_qbr_correlations_joins_qbs_by_team_and_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    _write_season_files(tmp_path, 2025, 0.0)
    qbr = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "game_week": ["Season Total"] * 3,
            "team_abb": ["A", "B", "C"],
            "name_display": ["One", "Two", "Three"],
            "qbr_total": [70.0, 60.0, 50.0],
        }
    )
    monkeypatch.setattr(walk_forward, "load_espn_qbr", stub(lambda: qbr))

    # Act
    correlations = compute_qbr_correlations(tmp_path, [2025])

    # Assert
    assert correlations.row(0, named=True)["joined_rows"] == 3
    assert correlations.row(0, named=True)["pearson"] == pytest.approx(1.0)
