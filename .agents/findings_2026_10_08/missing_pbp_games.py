"""List regular-season games on nflverse's schedule that have no rows in the team game logs.

Written for the season-in-progress bug (roadmap, "Data notes"): for each season it compares
nflverse's schedule with ``data/{season}_team_game_logs.parquet`` and says, for every scheduled
game missing from the logs, whether nflverse play-by-play has any rows for it. Downloads the
schedules and play-by-play through nflreadpy (cached); reads ``data/``; writes to stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/missing_pbp_games.py \
        1999 2000 2022
"""

from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured

DATA = Path("data")


def missing_games(season: int) -> pl.DataFrame:
    """Return the scheduled regular-season games of ``season`` absent from its team game logs."""
    schedule = nfl.load_schedules([season]).filter(pl.col("game_type") == "REG")
    logged = pl.read_parquet(DATA / f"{season}_team_game_logs.parquet").get_column("game_id")
    return schedule.filter(~pl.col("game_id").is_in(logged.unique().to_list())).select(
        "game_id", "week", "away_team", "home_team", "away_score", "home_score"
    )


def main() -> None:
    """Print each requested season's missing games and their play-by-play row counts."""
    use_disk_cache_unless_configured()
    for season in (int(argument) for argument in sys.argv[1:]):
        missing = missing_games(season)
        ids = missing.get_column("game_id").to_list()
        pbp_rows = nfl.load_pbp([season]).filter(pl.col("game_id").is_in(ids)).height
        sys.stdout.write(
            f"{season}: {missing.height} missing; play-by-play rows for them: {pbp_rows}\n"
        )
        for row in missing.iter_rows(named=True):
            sys.stdout.write(
                f"  {row['game_id']} (week {row['week']}): {row['away_team']} {row['away_score']} "
                f"at {row['home_team']} {row['home_score']}\n"
            )


if __name__ == "__main__":
    main()
