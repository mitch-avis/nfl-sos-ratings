import os
from pathlib import Path

import polars as pl
import pytest

from nfl_sos_ratings import visualize


def test_plot_functions_skip_when_inputs_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    df = pl.DataFrame({"team": ["DEN", "KC"]})

    visualize.PLOTS_DIR = str(tmp_path)
    visualize.plot_diff_grid(df, visualize.OFFENSE_SPECS, "Offense", "diffs.png")
    visualize.plot_sos_overview(df, "overview.png")
    visualize.plot_diff_heatmap(df, "heatmap.png")
    visualize.plot_composite_sos(df, "composite.png")
    visualize.plot_adjusted_ratings(df, "ratings.png")

    output = capsys.readouterr().out
    assert "Skipping diffs.png" in output
    assert "Skipping overview.png" in output
    assert "Skipping heatmap.png" in output
    assert "Skipping composite.png" in output
    assert "Skipping ratings.png" in output


def test_plot_functions_create_expected_files(tmp_path: Path, visualize_df: pl.DataFrame) -> None:
    visualize.PLOTS_DIR = str(tmp_path)

    visualize.plot_diff_grid(visualize_df, visualize.OFFENSE_SPECS, "Offense", "diffs_offense.png")
    visualize.plot_sos_overview(visualize_df, "sos_opponent_strength.png")
    visualize.plot_diff_heatmap(visualize_df, "heatmap_diffs.png")
    visualize.plot_composite_sos(visualize_df, "sos_composite_ranking.png")
    visualize.plot_adjusted_ratings(visualize_df, "adjusted_ratings.png")

    assert (tmp_path / "diffs_offense.png").exists()
    assert (tmp_path / "sos_opponent_strength.png").exists()
    assert (tmp_path / "heatmap_diffs.png").exists()
    assert (tmp_path / "sos_composite_ranking.png").exists()
    assert (tmp_path / "adjusted_ratings.png").exists()


def test_plot_functions_hide_unused_axes(tmp_path: Path) -> None:
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
    visualize.OUTPUT_DIR = str(tmp_path)
    visualize.PLOTS_DIR = os.path.join(str(tmp_path), "plots")

    visualize.main()
    missing_output = capsys.readouterr().out
    assert "not found" in missing_output

    combined_path = tmp_path / f"{visualize.SEASON}_combined.csv"
    pl.DataFrame({"team": ["DEN"]}).write_csv(combined_path)
    visualize.main()

    assert "No diff_ columns found" in capsys.readouterr().out


def test_visualize_main_generates_all_plots(tmp_path: Path, visualize_df: pl.DataFrame) -> None:
    visualize.OUTPUT_DIR = str(tmp_path)
    visualize.PLOTS_DIR = os.path.join(str(tmp_path), "plots")
    pl.DataFrame(visualize_df).write_csv(tmp_path / f"{visualize.SEASON}_combined.csv")

    visualize.main()

    assert sorted(os.listdir(visualize.PLOTS_DIR)) == [
        f"{visualize.SEASON}_adjusted_ratings.png",
        f"{visualize.SEASON}_diffs_defense.png",
        f"{visualize.SEASON}_diffs_offense.png",
        f"{visualize.SEASON}_diffs_overall.png",
        f"{visualize.SEASON}_diffs_qb.png",
        f"{visualize.SEASON}_heatmap_diffs.png",
        f"{visualize.SEASON}_opponent_stats_defense.png",
        f"{visualize.SEASON}_opponent_stats_offense.png",
        f"{visualize.SEASON}_opponent_stats_overall.png",
        f"{visualize.SEASON}_opponent_stats_qb.png",
        f"{visualize.SEASON}_sos_composite_ranking.png",
        f"{visualize.SEASON}_team_stats_defense.png",
        f"{visualize.SEASON}_team_stats_offense.png",
        f"{visualize.SEASON}_team_stats_overall.png",
        f"{visualize.SEASON}_team_stats_qb.png",
    ]
