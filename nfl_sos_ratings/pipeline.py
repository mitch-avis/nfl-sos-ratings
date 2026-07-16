"""Multi-season data and visualization pipeline.

Runs the single-season pipeline for every configured season, then runs the
visualization pass for those seasons after all data outputs are available.

Usage:
    uv run nfl-sos-pipeline          # uses START_YEAR / END_YEAR from config
    uv run python -m nfl_sos_ratings.pipeline
"""

import io
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/pipeline.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nfl_sos_ratings.visualize as visualize
from nfl_sos_ratings.config import END_YEAR, START_YEAR
from nfl_sos_ratings.main import run_season


def main() -> None:
    """Run data gathering then visualization for all seasons from START_YEAR to END_YEAR."""
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    seasons = list(range(START_YEAR, END_YEAR + 1))
    print(
        f"=== NFL SoS Pipeline: {START_YEAR}–{END_YEAR} "
        f"({len(seasons)} season{'s' if len(seasons) != 1 else ''}) ===\n"
    )

    # Phase 1: data for every season
    print(f"{'─' * 70}")
    print("Phase 1 of 2: Data gathering")
    print(f"{'─' * 70}\n")
    failed_data_seasons: list[int] = []
    for season in seasons:
        try:
            run_season(season)
        except Exception as exc:
            failed_data_seasons.append(season)
            print(f"\nERROR: season {season} data step failed — {exc}\n")

    # Phase 2: visualizations for every season
    print(f"\n{'─' * 70}")
    print("Phase 2 of 2: Visualizations")
    print(f"{'─' * 70}\n")
    failed_visualization_seasons: list[int] = []
    for season in seasons:
        if season in failed_data_seasons:
            print(f"Skipping visualization for season {season} due to failed data step.")
            continue
        try:
            visualize.main(season)
        except Exception as exc:
            failed_visualization_seasons.append(season)
            print(f"\nERROR: season {season} visualization failed — {exc}\n")

    if failed_data_seasons or failed_visualization_seasons:
        print(f"\n{'=' * 70}")
        print("Pipeline finished with failures.")
        if failed_data_seasons:
            data_failures = ", ".join(str(season) for season in failed_data_seasons)
            print(f"Data step failures: {data_failures}")
        if failed_visualization_seasons:
            visualization_failures = ", ".join(
                str(season) for season in failed_visualization_seasons
            )
            print(f"Visualization failures: {visualization_failures}")
        print(f"{'=' * 70}")
        raise SystemExit(1)

    print(f"\n{'=' * 70}")
    print(f"Pipeline complete — {len(seasons)} seasons processed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
