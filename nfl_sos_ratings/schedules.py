"""All-time strength-of-schedule leaderboard across every season written to ``data/``.

``sos`` is the average ``team_rating`` of a team's opponents, in points per game, with each
opponent rated without its games against that team. This module ranks every team-season on it, so
one season's schedule can be placed in the full history the data covers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

from nfl_sos_ratings.config import DATA_DIR

_RATINGS_FILE = re.compile(r"^(?P<season>\d{4})_ratings\.parquet$")
_DEFAULT_TOP = 10


def rank_schedules(data_dir: Path) -> pl.DataFrame:
    """Return every team-season with its ``sos`` and its softest and hardest rank overall.

    Rank 1 is the softest (``softest_rank``) or hardest (``hardest_rank``) schedule in the data.
    Ratings files without an ``sos`` column (older outputs) are skipped.
    """
    frames: list[pl.DataFrame] = []
    for path in sorted(data_dir.glob("*_ratings.parquet")):
        match = _RATINGS_FILE.match(path.name)
        if match is None or "sos" not in pl.read_parquet_schema(path):
            continue
        frames.append(
            pl.read_parquet(path, columns=["team", "sos"]).with_columns(
                pl.lit(int(match["season"])).alias("season")
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "team": pl.String,
                "sos": pl.Float64,
                "softest_rank": pl.UInt32,
                "hardest_rank": pl.UInt32,
            }
        )
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("sos").rank("ordinal").alias("softest_rank"),
            pl.col("sos").rank("ordinal", descending=True).alias("hardest_rank"),
        )
        .select("season", "team", "sos", "softest_rank", "hardest_rank")
        .sort("softest_rank")
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``schedules`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings schedules",
        description=(
            "Rank every team-season in data/ by strength of schedule (sos, points per game). "
            "With --team and --season, report where that schedule ranks."
        ),
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory of Parquet outputs.")
    parser.add_argument("--top", type=int, default=_DEFAULT_TOP, help="Rows per list.")
    parser.add_argument("--team", help="Team abbreviation to look up, such as NE.")
    parser.add_argument("--season", type=int, help="Season of the team to look up.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print the softest and hardest schedules, or one team-season's rank."""
    args = _parse_args(argv)
    ranked = rank_schedules(Path(args.data_dir))
    total = ranked.height
    if args.team is not None and args.season is not None:
        row = ranked.filter((pl.col("team") == args.team) & (pl.col("season") == args.season))
        if row.is_empty():
            sys.stdout.write(f"No sos for {args.season} {args.team} in {args.data_dir}.\n")
            return
        found = row.row(0, named=True)
        sys.stdout.write(
            f"{found['season']} {found['team']}: sos {found['sos']:.2f}, softest "
            f"{found['softest_rank']} of {total}, hardest {found['hardest_rank']} of {total}\n"
        )
        return
    with pl.Config(tbl_rows=args.top, float_precision=2):
        sys.stdout.write(f"Softest schedules of {total} team-seasons:\n")
        sys.stdout.write(f"{ranked.head(args.top)}\n\n")
        sys.stdout.write(f"Hardest schedules of {total} team-seasons:\n")
        sys.stdout.write(f"{ranked.sort('hardest_rank').head(args.top)}\n")


__all__ = ["main", "rank_schedules"]
