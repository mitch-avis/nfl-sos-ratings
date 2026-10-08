"""Per season, which side of the ball lost each fumble nflverse flags (roadmap, "Data notes").

nflverse's ``fumble_lost`` marks a play on which a fumble was lost, whichever team fumbled, and
``fumbled_1_team`` names the fumbling team. Reads nflverse's raw regular-season play-by-play
through nflreadpy (cached) and prints, per season, the plays with ``fumble_lost`` by kind of play
(punt, kickoff, interception, other) and by whether the fumbler was the team listed with the ball
(``posteam``) or the other team (``defteam``), with a total row. On a punt ``posteam`` is the
punting team, so a returner's lost muff or fumble shows as ``punt_defteam``.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/lost_fumbles.py 1999 2025
"""

from __future__ import annotations

import sys

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured

KINDS = ("punt", "kickoff", "interception", "other")
SIDES = ("posteam", "defteam", "unknown")


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def season_row(season: int) -> dict[str, int]:
    """Return one season's lost-fumble counts by kind of play and fumbling side."""
    pbp = nfl.load_pbp([season]).filter((pl.col("season_type") == "REG") & _flag("fumble_lost"))
    labelled = pbp.select(
        pl.when(_flag("punt_attempt"))
        .then(pl.lit("punt"))
        .when(_flag("kickoff_attempt"))
        .then(pl.lit("kickoff"))
        .when(_flag("interception"))
        .then(pl.lit("interception"))
        .otherwise(pl.lit("other"))
        .alias("kind"),
        pl.when(pl.col("fumbled_1_team") == pl.col("posteam"))
        .then(pl.lit("posteam"))
        .when(pl.col("fumbled_1_team") == pl.col("defteam"))
        .then(pl.lit("defteam"))
        .otherwise(pl.lit("unknown"))
        .alias("side"),
    )
    row = {"season": season}
    for kind in KINDS:
        for side in SIDES:
            count = labelled.filter((pl.col("kind") == kind) & (pl.col("side") == side)).height
            row[f"{kind}_{side}"] = count
    return row


def main() -> None:
    """Print the table for the seasons from the first argument to the second, with a total row."""
    use_disk_cache_unless_configured()
    first, last = (int(argument) for argument in sys.argv[1:3])
    table = pl.DataFrame([season_row(season) for season in range(first, last + 1)])
    table = table.select(
        "season", *(column for column in table.columns[1:] if table.get_column(column).sum() > 0)
    )
    total = table.select(pl.lit(None, dtype=pl.Int64).alias("season"), pl.exclude("season").sum())
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f"{pl.concat([table, total], how='vertical_relaxed')}\n")


if __name__ == "__main__":
    main()
