"""Show how nflverse codes scrambles by season, and what the scrimmage-snap filter made of them.

Written to confirm that team scrimmage plays left out scrambles before 2006. For each season given
it prints:

- the flags on every run play nflverse marks ``qb_scramble`` (``qb_dropback``, ``rush``,
  ``rush_attempt``, ``pass``, ``pass_attempt``), as counts per combination;
- the plays nflverse flags ``pass`` that ``pbp_expressions.scrimmage_snap_expr`` leaves out, by
  ``play_type``, and how many of them are run-play scrambles;
- play-by-play rushing yards over the filter's plays, with and without the run-play scrambles it
  leaves out, beside nflverse's official team rushing yards, which count scramble yards.

Run it against the filter as it stands; on the fixed filter the left-out scrambles are 0. Reads
play-by-play and official team stats through the package loaders (nflreadpy, cached); writes to
stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/scramble_coding.py \
        1999 2003 2004 2005 2006 2010 2025
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from nfl_sos_ratings.data_loader import (
    load_official_weekly_team_stats,
    load_pbp_data,
    use_disk_cache_unless_configured,
)
from nfl_sos_ratings.pbp_expressions import scrimmage_snap_expr

FLAGS = ("qb_dropback", "rush", "rush_attempt", "pass", "pass_attempt")


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def season_lines(season: int) -> list[str]:
    """Return the printed lines for one season."""
    pbp = load_pbp_data(season).filter(
        pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    )
    snap = scrimmage_snap_expr(pbp.columns)
    run_scramble = _flag("qb_scramble") & (pl.col("play_type") == "run")
    scrambles = pbp.filter(run_scramble)
    combos = (
        scrambles.select(pl.col(flag).fill_null(-1).cast(pl.Int8) for flag in FLAGS)
        .group_by(pl.all())
        .len()
        .sort("len", descending=True)
    )
    combo_text = "; ".join(
        ", ".join(f"{flag}={value}" for flag, value in zip(FLAGS, row[:-1], strict=True))
        + f": {row[-1]}"
        for row in combos.iter_rows()
    )
    pass_left_out = pbp.filter(_flag("pass") & ~snap)
    by_type = dict(
        pass_left_out.group_by(pl.col("play_type").fill_null("none")).len().sort("play_type").rows()
    )
    yards = pl.col("rushing_yards").fill_null(0.0)
    filter_yards = pbp.filter(snap).select(yards.sum()).item()
    with_scrambles = pbp.filter(snap | (run_scramble & ~snap)).select(yards.sum()).item()
    official = load_official_weekly_team_stats(season)["rushing_yards"].sum()
    return [
        f"{season}: {scrambles.height} run-play scrambles; flags {combo_text}",
        (
            f"  pass-flagged plays the filter leaves out, by play type: {by_type}; run-play "
            f"scrambles among them: {pass_left_out.filter(run_scramble).height}"
        ),
        (
            f"  rushing yards: filter's plays {filter_yards:.0f}, plus left-out scrambles "
            f"{with_scrambles:.0f}, official {official:.0f}"
        ),
    ]


def main() -> None:
    """Print the scramble coding for each season given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("seasons", type=int, nargs="+", help="Seasons to show.")
    arguments = parser.parse_args()
    use_disk_cache_unless_configured()
    for season in arguments.seasons:
        sys.stdout.write("\n".join(season_lines(season)) + "\n")


if __name__ == "__main__":
    main()
