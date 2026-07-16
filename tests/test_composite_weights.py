"""Tests for Stage 2 composite-weight fitting helpers."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings import composite_weights


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    """Write one synthetic parquet fixture for a season artifact."""
    frame.write_parquet(path)


def test_build_team_training_rows_matches_aliases_and_standardizes_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Team training rows should pair relocated clubs and z-score season-t predictors."""
    monkeypatch.setattr(
        composite_weights,
        "load_pbp_data",
        lambda season: pl.DataFrame(
            {
                "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
                "week": [1, 1, 1, 1, 1, 1],
                "posteam": ["LV", "LAC", "NE", "LV", "LAC", "NE"],
                "defteam": ["LAC", "LV", "LV", "NE", "NE", "LAC"],
                "special": [1, 1, 1, 1, 1, 1],
                "epa": [0.4, -0.1, -0.3, 0.2, 0.1, -0.2],
            }
        ),
    )
    _write_parquet(
        tmp_path / "2020_combined.parquet",
        pl.DataFrame(
            {
                "team": ["OAK", "SD", "NE"],
                "adj_off_passing_epa_per_offensive_snap": [0.30, 0.10, -0.10],
                "adj_off_rushing_epa_per_offensive_snap": [0.15, 0.05, -0.05],
                "adj_def_passing_epa_per_offensive_snap": [0.20, 0.00, -0.20],
                "adj_def_rushing_epa_per_offensive_snap": [0.12, 0.02, -0.08],
                "adj_def_def_interceptions_per_defensive_snap": [0.030, 0.020, 0.010],
                "adj_def_def_fumbles_forced_per_defensive_snap": [0.020, 0.010, 0.000],
                "SaOvR": [1.2, 0.2, -1.0],
            }
        ),
    )
    _write_parquet(
        tmp_path / "2021_combined.parquet",
        pl.DataFrame(
            {
                "team": ["LV", "LAC", "NE"],
                "SaOvR": [1.5, 0.4, -0.6],
            }
        ),
    )

    rows = composite_weights.build_team_training_rows(tmp_path, seasons=[2020, 2021]).sort("team")

    assert rows.select("team").to_series().to_list() == ["LAC", "LV", "NE"]
    assert rows.filter(pl.col("team") == "LV").select("target").item() == pytest.approx(1.5)

    feature_names = [component.name for component in composite_weights.TEAM_SACR_COMPONENTS]
    for feature_name in feature_names:
        values = rows.select(feature_name).to_series()
        assert values.mean() == pytest.approx(0.0, abs=1e-9)
        assert values.std() == pytest.approx(1.0)


def test_build_qb_training_rows_filters_to_eligible_and_carries_dropback_weights(
    tmp_path: Path,
) -> None:
    """QB training rows should keep only eligible passers and preserve dropback weights."""
    _write_parquet(
        tmp_path / "2006_qb_combined.parquet",
        pl.DataFrame(
            {
                "qb_id": ["qb-a", "qb-b", "qb-c"],
                "qb_is_eligible": [True, False, True],
                "qb_dropbacks": [520, 180, 410],
                "adj_qb_epa_per_dropback": [0.24, 0.05, 0.12],
                "adj_qb_completion_percentage_above_expectation": [4.0, 1.0, 2.0],
                "adj_qb_sack_rate": [0.03, 0.07, 0.05],
                "adj_qb_td_int_margin_rate": [0.08, 0.01, 0.04],
            }
        ),
    )
    _write_parquet(
        tmp_path / "2007_qb_combined.parquet",
        pl.DataFrame(
            {
                "qb_id": ["qb-a", "qb-c"],
                "adj_qb_epa_per_dropback": [0.18, 0.10],
            }
        ),
    )

    rows = composite_weights.build_qb_training_rows(tmp_path, seasons=[2006, 2007]).sort("qb_id")

    assert rows.select("qb_id").to_series().to_list() == ["qb-a", "qb-c"]
    assert rows.select("qb_dropbacks").to_series().to_list() == [520, 410]


def test_fit_linear_weights_and_holdout_diagnostics_prefer_the_planted_signal() -> None:
    """OLS fit and leave-one-season-out diagnostics should beat an equal-weight blend."""
    rows = pl.DataFrame(
        {
            "season": [2018, 2018, 2018, 2018, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020],
            "feature_a": [1.0, -1.0, 0.5, -0.5, 1.2, -1.2, 0.4, -0.4, 0.8, -0.8, 0.2, -0.2],
            "feature_b": [0.5, 0.5, -0.5, -0.5, 0.7, 0.7, -0.7, -0.7, 0.3, 0.3, -0.3, -0.3],
            "feature_c": [1.0, 1.0, -1.0, -1.0, -0.2, -0.2, 0.2, 0.2, 0.6, 0.6, -0.6, -0.6],
        }
    ).with_columns(
        (
            (pl.col("feature_a") * 2.0) + (pl.col("feature_b") * 0.5) + (pl.col("feature_c") * 0.0)
        ).alias("target")
    )

    weights = composite_weights.fit_linear_weights(
        rows,
        feature_columns=("feature_a", "feature_b", "feature_c"),
        target_column="target",
    )
    diagnostics = composite_weights.evaluate_leave_one_season_out(
        rows,
        feature_columns=("feature_a", "feature_b", "feature_c"),
        target_column="target",
        holdout_column="season",
    )

    assert weights["feature_a"] == pytest.approx(2.0)
    assert weights["feature_b"] == pytest.approx(0.5)
    assert weights["feature_c"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["weighted_rmse"] < diagnostics["equal_weight_rmse"]
    assert diagnostics["weighted_mae"] < diagnostics["equal_weight_mae"]


def test_frozen_stage_two_specs_match_the_committed_weight_snapshot() -> None:
    """Frozen Stage 2 weight specs should not drift without an explicit refit."""
    assert composite_weights.TEAM_SACR_FROZEN_SPEC.fit_window == (1999, 2025)
    assert composite_weights.TEAM_SACR_FROZEN_SPEC.feature_columns == (
        "adj_off_passing_epa_per_offensive_snap",
        "adj_off_rushing_epa_per_offensive_snap",
        "adj_def_passing_epa_per_offensive_snap",
        "adj_def_rushing_epa_per_offensive_snap",
        "st_rating",
    )
    assert composite_weights.TEAM_SACR_FROZEN_SPEC.weight_map() == pytest.approx(
        {
            "adj_off_passing_epa_per_offensive_snap": 0.3828739475913225,
            "adj_off_rushing_epa_per_offensive_snap": 0.19062479977967036,
            "adj_def_passing_epa_per_offensive_snap": 0.27163464954613765,
            "adj_def_rushing_epa_per_offensive_snap": 0.0973640754908631,
            "st_rating": 0.0575025275920063,
        }
    )

    assert composite_weights.QB_QSACR_FROZEN_SPEC.fit_window == (2006, 2025)
    assert composite_weights.QB_QSACR_FROZEN_SPEC.feature_columns == (
        "adj_qb_epa_per_dropback",
        "adj_qb_completion_percentage_above_expectation",
        "adj_qb_sack_rate",
        "adj_qb_td_int_margin_rate",
    )
    assert composite_weights.QB_QSACR_FROZEN_SPEC.weight_map() == pytest.approx(
        {
            "adj_qb_epa_per_dropback": 0.6687790473858877,
            "adj_qb_completion_percentage_above_expectation": 0.21464381898367774,
            "adj_qb_sack_rate": 0.06725872314827445,
            "adj_qb_td_int_margin_rate": 0.04931841048216012,
        }
    )


def test_builders_return_typed_empty_frames_when_no_pairs_are_available(tmp_path: Path) -> None:
    """Empty season windows should return typed empty training frames instead of raising."""
    team_rows = composite_weights.build_team_training_rows(tmp_path, seasons=[2020])
    qb_rows = composite_weights.build_qb_training_rows(tmp_path, seasons=[1999, 2005])

    assert team_rows.is_empty()
    assert team_rows.columns == [
        "season",
        "next_season",
        "team",
        *[component.name for component in composite_weights.TEAM_SACR_COMPONENTS],
        "target",
    ]
    assert qb_rows.is_empty()
    assert qb_rows.columns == [
        "season",
        "next_season",
        "qb_id",
        "qb_dropbacks",
        *[component.name for component in composite_weights.QB_QSACR_COMPONENTS],
        "target",
    ]


def test_build_team_training_rows_rejects_missing_stage_two_columns(tmp_path: Path) -> None:
    """Team training rows should fail fast when a consumed season file is stale or incomplete."""
    _write_parquet(
        tmp_path / "2020_combined.parquet",
        pl.DataFrame(
            {
                "team": ["A"],
                "adj_off_passing_epa_per_offensive_snap": [0.1],
                "adj_off_rushing_epa_per_offensive_snap": [0.1],
            }
        ),
    )
    _write_parquet(
        tmp_path / "2021_combined.parquet", pl.DataFrame({"team": ["A"], "SaOvR": [1.0]})
    )

    with pytest.raises(ValueError, match="missing required Stage 2 columns"):
        composite_weights.build_team_training_rows(tmp_path, seasons=[2020, 2021])


def test_helper_branches_cover_missing_components_reference_edges_and_weight_columns() -> None:
    """Low-level helpers should handle degenerate references and missing component inputs."""
    assert (
        composite_weights._resolve_qb_weight_column(pl.DataFrame({"qb_dropbacks_total": [12.0]}))
        == "qb_dropbacks_total"
    )
    with pytest.raises(ValueError, match="qb_dropbacks"):
        composite_weights._resolve_qb_weight_column(pl.DataFrame({"other": [1.0]}))

    component_a = composite_weights.CompositeComponent("a", ("a",), True)
    component_b = composite_weights.CompositeComponent("b", ("b",), False)
    spec = composite_weights.FrozenCompositeSpec(
        name="demo",
        components=(component_a, component_b),
        weights=(("a", 0.75), ("b", 0.25)),
        target_column="target",
        fit_window=(2020, 2021),
        fitting_command="demo",
        refit_policy="demo",
    )
    composite = composite_weights.build_weighted_composite(
        pl.DataFrame({"a": [3.0, 1.0]}),
        spec,
        pl.DataFrame({"a": [2.0]}),
    )

    assert composite.tolist() == pytest.approx([1.0, -1.0])
    assert composite_weights._zscore_against(
        np.array([1.0, 2.0], dtype=np.float64), np.array([], dtype=np.float64)
    ).tolist() == [1.0, 2.0]
    assert composite_weights._normalize_weight_map({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_fit_helpers_cover_weighted_and_empty_holdout_paths() -> None:
    """Weighted fits and empty holdout splits should both behave deterministically."""
    weighted_rows = pl.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "target": [2.0, 4.0, 6.0],
            "sample_weight": [1.0, 2.0, 3.0],
        }
    )
    weights = composite_weights.fit_linear_weights(
        weighted_rows,
        feature_columns=("feature",),
        target_column="target",
        sample_weight_column="sample_weight",
    )
    empty_diag = composite_weights.evaluate_leave_one_season_out(
        pl.DataFrame({"season": [2020], "feature": [1.0], "target": [2.0]}),
        feature_columns=("feature",),
        target_column="target",
        holdout_column="season",
    )

    assert weights["feature"] == pytest.approx(2.0)
    assert empty_diag == {
        "weighted_mae": 0.0,
        "weighted_rmse": 0.0,
        "equal_weight_mae": 0.0,
        "equal_weight_rmse": 0.0,
    }


def test_main_prints_the_reproducible_stage_two_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI report should summarize the candidate and frozen Stage 2 fits."""
    team_rows = pl.DataFrame(
        {
            "season": [2020],
            "next_season": [2021],
            "team": ["A"],
            **{component.name: [0.0] for component in composite_weights.TEAM_SACR_COMPONENTS},
            "target": [0.0],
        }
    )
    qb_rows = pl.DataFrame(
        {
            "season": [2020],
            "next_season": [2021],
            "qb_id": ["qb-a"],
            "qb_dropbacks": [30.0],
            **{component.name: [0.0] for component in composite_weights.QB_QSACR_COMPONENTS},
            "target": [0.0],
        }
    )

    def fake_fit(
        df: pl.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        sample_weight_column: str | None = None,
    ) -> dict[str, float]:
        del df, target_column, sample_weight_column
        return {column: float(index + 1) for index, column in enumerate(feature_columns)}

    def fake_eval(
        df: pl.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        holdout_column: str,
        sample_weight_column: str | None = None,
    ) -> dict[str, float]:
        del df, feature_columns, target_column, holdout_column, sample_weight_column
        return {
            "weighted_mae": 0.1,
            "weighted_rmse": 0.2,
            "equal_weight_mae": 0.3,
            "equal_weight_rmse": 0.4,
        }

    monkeypatch.setattr(
        composite_weights, "build_team_training_rows", lambda data_dir, seasons: team_rows
    )
    monkeypatch.setattr(
        composite_weights, "build_qb_training_rows", lambda data_dir, seasons: qb_rows
    )
    monkeypatch.setattr(composite_weights, "fit_linear_weights", fake_fit)
    monkeypatch.setattr(composite_weights, "evaluate_leave_one_season_out", fake_eval)

    composite_weights.main()
    output = capsys.readouterr().out

    assert "Stage 2 composite-weight fit" in output
    assert "Team candidate fit (includes turnover-creation test component):" in output
    assert "Frozen SaCR weights:" in output
    assert "Frozen QSaCR weights:" in output
    assert "adj_def_takeaway_creation_rate_per_defensive_snap" in output
