"""Tests for nfl_sos_ratings.visualize plotting behavior."""

import os
from pathlib import Path

import polars as pl
import pytest

from nfl_sos_ratings import visualize


def test_plot_functions_skip_when_inputs_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify plotting functions print skip messages when required columns are absent."""
    df = pl.DataFrame({"team": ["DEN", "KC"]})

    visualize.PLOTS_DIR = str(tmp_path)
    visualize.plot_diff_grid(df, visualize.OFFENSE_SPECS, "Offense", "diffs.png")
    visualize.plot_sos_overview(df, "overview.png")
    visualize.plot_diff_heatmap(df, "heatmap.png")
    visualize.plot_composite_sos(df, "composite.png")
    visualize.plot_adjusted_ratings(df, "ratings.png")
    visualize.plot_qb_raw_vs_adjusted(df, "qb_raw_vs_adjusted.png")
    visualize.plot_qb_schedule_vs_performance(df, "qb_schedule_vs_performance.png")
    visualize.plot_qb_raw_vs_schedule(df, "qb_raw_vs_schedule.png")

    output = capsys.readouterr().out
    assert "Skipping diffs.png" in output
    assert "Skipping overview.png" in output
    assert "Skipping heatmap.png" in output
    assert "Skipping composite.png" in output
    assert "Skipping ratings.png" in output
    assert "Skipping qb_raw_vs_adjusted.png" in output
    assert "Skipping qb_schedule_vs_performance.png" in output
    assert "Skipping qb_raw_vs_schedule.png" in output


def test_qb_plot_helpers_use_player_labels_and_filter_ineligible() -> None:
    """Verify QB plots operate on qualified player rows, not team-only labels."""
    qb_df = pl.DataFrame(
        {
            "qb_name": ["Bo Nix", "Backup QB"],
            "team": ["DEN", "DEN"],
            "qb_is_eligible": [True, False],
            "QSaCR": [0.5, 2.0],
        }
    )

    filtered = visualize._filter_qb_plot_df(qb_df)

    assert filtered.height == 1
    assert visualize._qb_display_labels(filtered) == ["Bo Nix (DEN)"]


def test_sorted_rating_rows_orders_descending() -> None:
    """Verify adjusted-rating row sorting is highest-to-lowest by selected column."""
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "SaOR": [0.1, 1.2, -0.3],
        }
    )

    teams, values = visualize._sorted_rating_rows(df, "SaOR")

    assert teams == ["B", "A", "C"]
    assert values == [1.2, 0.1, -0.3]


def test_plot_functions_create_expected_files(tmp_path: Path, visualize_df: pl.DataFrame) -> None:
    """Verify plotting helpers create expected output image files."""
    visualize.PLOTS_DIR = str(tmp_path)

    visualize.plot_diff_grid(visualize_df, visualize.OFFENSE_SPECS, "Offense", "diffs_offense.png")
    visualize.plot_sos_overview(visualize_df, "sos_opponent_strength.png")
    visualize.plot_diff_heatmap(visualize_df, "heatmap_diffs.png")
    visualize.plot_composite_sos(visualize_df, "sos_composite_ranking.png")
    visualize.plot_adjusted_ratings(visualize_df, "adjusted_ratings.png")
    visualize.plot_qb_adjusted_ratings(
        pl.DataFrame({"team": ["DEN", "KC"], "QSaCR": [0.6, -0.4]}),
        "qb_adjusted_ratings.png",
    )
    qb_df = pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "QRaw": [0.2, -0.1],
            "QSaOR": [0.6, -0.3],
            "QSoS": [0.4, -0.2],
            "QSaCR": [0.6, -0.3],
        }
    )
    visualize.plot_qb_raw_vs_adjusted(qb_df, "qb_raw_vs_adjusted.png")
    visualize.plot_qb_schedule_vs_performance(qb_df, "qb_schedule_vs_performance.png")
    visualize.plot_qb_raw_vs_schedule(qb_df, "qb_raw_vs_schedule.png")

    assert (tmp_path / "diffs_offense.png").exists()
    assert (tmp_path / "sos_opponent_strength.png").exists()
    assert (tmp_path / "heatmap_diffs.png").exists()
    assert (tmp_path / "sos_composite_ranking.png").exists()
    assert (tmp_path / "adjusted_ratings.png").exists()
    assert (tmp_path / "qb_adjusted_ratings.png").exists()
    assert (tmp_path / "qb_raw_vs_adjusted.png").exists()
    assert (tmp_path / "qb_schedule_vs_performance.png").exists()
    assert (tmp_path / "qb_raw_vs_schedule.png").exists()


def test_plot_functions_hide_unused_axes(tmp_path: Path) -> None:
    """Verify plotting grids still render when only a subset of panels is available."""
    visualize.PLOTS_DIR = str(tmp_path)
    df = pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "diff_points_for": [1.0, -1.0],
            "opp_points_for": [20.0, 24.0],
        }
    )

    visualize.plot_diff_grid(df, visualize.OFFENSE_SPECS, "Offense", "partial_diffs.png")
    visualize.plot_sos_overview(df, "partial_overview.png")

    assert (tmp_path / "partial_diffs.png").exists()
    assert (tmp_path / "partial_overview.png").exists()


def test_visualize_main_handles_missing_and_invalid_combined_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify visualize.main handles missing combined file and sparse combined columns."""
    visualize.OUTPUT_DIR = str(tmp_path)
    visualize.PLOTS_DIR = os.path.join(str(tmp_path), "plots")

    visualize.main()
    missing_output = capsys.readouterr().out
    assert "not found" in missing_output

    combined_path = tmp_path / f"{visualize.SEASON}_combined.csv"
    pl.DataFrame({"team": ["DEN"]}).write_csv(combined_path)
    visualize.main()

    output = capsys.readouterr().out
    assert "Skipping" in output


def test_visualize_main_generates_all_plots(tmp_path: Path, visualize_df: pl.DataFrame) -> None:
    """Verify visualize.main generates only the selected team plots."""
    visualize.OUTPUT_DIR = str(tmp_path)
    visualize.PLOTS_DIR = os.path.join(str(tmp_path), "plots")
    pl.DataFrame(visualize_df).write_csv(tmp_path / f"{visualize.SEASON}_combined.csv")

    visualize.main()

    assert sorted(os.listdir(visualize.PLOTS_DIR)) == [
        f"{visualize.SEASON}_adjusted_ratings_defense.png",
        f"{visualize.SEASON}_adjusted_ratings_offense.png",
        f"{visualize.SEASON}_adjusted_ratings_overall.png",
        f"{visualize.SEASON}_sos_composite_ranking.png",
    ]


def test_visualize_main_generates_qb_plot_when_qb_combined_exists(
    tmp_path: Path, visualize_df: pl.DataFrame
) -> None:
    """Verify visualize.main adds selected QB plots when qb_combined output exists."""
    visualize.OUTPUT_DIR = str(tmp_path)
    visualize.PLOTS_DIR = os.path.join(str(tmp_path), "plots")
    visualize_df.write_csv(tmp_path / f"{visualize.SEASON}_combined.csv")
    pl.DataFrame({"team": ["DEN", "KC"], "QSaCR": [0.4, -0.2]}).write_csv(
        tmp_path / f"{visualize.SEASON}_qb_combined.csv"
    )

    pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "QRaw": [0.2, -0.1],
            "QSaOR": [0.4, -0.2],
            "QSoS": [0.3, -0.1],
            "QSaCR": [0.4, -0.2],
        }
    ).write_csv(tmp_path / f"{visualize.SEASON}_qb_combined.csv")

    visualize.main()

    assert (tmp_path / "plots" / f"{visualize.SEASON}_qb_adjusted_ratings.png").exists()
    assert (tmp_path / "plots" / f"{visualize.SEASON}_qb_raw_vs_schedule.png").exists()
