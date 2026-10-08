"""Per season, the punts nflverse books as the punting team's lost fumble (roadmap, "Data notes").

On a punt, nflverse's ``posteam`` is the punting team, yet ``fumble_lost`` is also set when the
receiving team muffs or fumbles the punt and the punting team recovers it. ``turnover_epa`` sums EPA
over every play of the team with the ball that is an interception or a lost fumble, so such a play
counts as the punting team's giveaway (with positive EPA, since it got the ball), and the mirrored
``takeaway_epa`` counts it as the receiving team's takeaway. Reads nflverse's raw regular-season
play-by-play through nflreadpy (cached) and prints one row per season: punts with a lost fumble,
how many the punting team recovered (the receiving team's giveaways), and the summed EPA of those
from the punting team's side.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/punt_fumbles.py 1999 2025
"""

from __future__ import annotations

import sys

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def season_row(season: int) -> dict[str, int | float]:
    """Return one season's counts of punts with a lost fumble."""
    pbp = nfl.load_pbp([season]).filter(pl.col("season_type") == "REG")
    lost = pbp.filter(_flag("punt_attempt") & _flag("fumble_lost"))
    receiving_giveaways = lost.filter(pl.col("fumble_recovery_1_team") == pl.col("posteam"))
    return {
        "season": season,
        "punts_with_fumble_lost": lost.height,
        "recovered_by_punting_team": receiving_giveaways.height,
        "their_epa_for_punting_team": round(float(receiving_giveaways.get_column("epa").sum()), 2),
    }


def main() -> None:
    """Print the table for the seasons from the first argument to the second, with a total row."""
    use_disk_cache_unless_configured()
    first, last = (int(argument) for argument in sys.argv[1:3])
    table = pl.DataFrame([season_row(season) for season in range(first, last + 1)])
    total = table.select(
        pl.lit(None, dtype=pl.Int64).alias("season"),
        pl.exclude("season").sum(),
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f"{pl.concat([table, total], how='vertical_relaxed')}\n")


if __name__ == "__main__":
    main()
