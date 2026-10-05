"""Multi-season data pipeline.

Runs the single-season pipeline for every configured season.

Usage:
    nfl-sos-ratings pipeline         # uses START_YEAR / END_YEAR from config
"""

import argparse
import io
import logging
import sys

from nfl_sos_ratings.config import END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured
from nfl_sos_ratings.logger import configure_logging
from nfl_sos_ratings.main import run_season

logger = logging.getLogger(__name__)


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
    use_disk_cache_unless_configured()
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    seasons = list(range(START_YEAR, END_YEAR + 1))
    plural = "s" if len(seasons) != 1 else ""
    logger.info(
        "=== NFL SoS Pipeline: %s-%s (%d season%s) ===", START_YEAR, END_YEAR, len(seasons), plural
    )
    logger.info("%s", "-" * 70)
    logger.info("Phase 1 of 1: Data gathering")
    logger.info("%s", "-" * 70)
    failed_data_seasons: list[int] = []
    for season in seasons:
        try:
            run_season(season)
        except Exception:
            # Report every failed season with its traceback, then exit 1 after the rest.
            failed_data_seasons.append(season)
            logger.exception("season %s data step failed", season)

    if failed_data_seasons:
        failed_season_summary = ", ".join(str(season) for season in failed_data_seasons)
        logger.error("Data step failures: %s", failed_season_summary)
        logger.error("Pipeline finished with failures.")
        raise SystemExit(1)

    logger.info("%s", "=" * 70)
    logger.info("Pipeline complete: %d seasons processed.", len(seasons))
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    configure_logging()
    main()
