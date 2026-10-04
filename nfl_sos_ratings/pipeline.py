"""Multi-season data pipeline.

Runs the single-season pipeline for every configured season.

Usage:
    uv run nfl-sos-pipeline          # uses START_YEAR / END_YEAR from config
    uv run python -m nfl_sos_ratings.pipeline
"""

import argparse
import io
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/pipeline.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_sos_ratings.config import END_YEAR, START_YEAR
from nfl_sos_ratings.main import run_season


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``pipeline`` command's options (it has none beyond ``--help``)."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings pipeline",
        description=(
            f"Build every season {START_YEAR}-{END_YEAR} into data/. Exits 1 if any season fails."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run data gathering for all seasons from START_YEAR to END_YEAR."""
    _parse_args(argv)
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    seasons = list(range(START_YEAR, END_YEAR + 1))
    print(
        f"=== NFL SoS Pipeline: {START_YEAR}-{END_YEAR} "
        f"({len(seasons)} season{'s' if len(seasons) != 1 else ''}) ===\n"
    )

    # Data for every season
    print(f"{'─' * 70}")
    print("Phase 1 of 1: Data gathering")
    print(f"{'─' * 70}\n")
    failed_data_seasons: list[int] = []
    for season in seasons:
        try:
            run_season(season)
        except Exception as exc:  # noqa: BLE001 - report every failed season, then exit 1
            failed_data_seasons.append(season)
            print(f"\nERROR: season {season} data step failed — {exc}\n")

    if failed_data_seasons:
        failed_season_summary = ", ".join(str(season) for season in failed_data_seasons)
        print(f"Data step failures: {failed_season_summary}")
        print("Pipeline finished with failures.")
        raise SystemExit(1)

    print(f"\n{'=' * 70}")
    print(f"Pipeline complete — {len(seasons)} seasons processed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
