"""Split aborted snaps by who the official scorer charged with the fumble, per season.

Written for the QB all-plays protocol's question on fumbled snap exchanges. nflverse flags a
botched snap ``aborted_play``; its ``fumbled_1_player_id`` is the player the gamebook charges with
the fumble: the quarterback ("9-J.Burrow FUMBLES (Aborted)") or, on a bad snap, the center
("18-K.Cousins Aborted. 67-D.Dalman FUMBLES"). For every aborted play of a regular season (two-point
tries and penalty-wiped plays out) it classifies:

- the charged player: ``qb`` (a player with a QB row for that team in that game, from
  ``data/{season}_qb_game_logs.parquet``, or listed at QB in nflreadpy's players file, which
  catches a backup whose only snap was the aborted one), ``line`` (a center, guard, or tackle in
  the players file), ``other`` (any other player), or ``none`` (no fumble charged);
- the rusher: ``qb``, ``other``, or ``none``, the same way.

It prints the counts and summed EPA of each pair per season, the totals over the seasons given,
and three sample descriptions of each pair. Reads ``data/``, play-by-play, and the players file
(nflreadpy, cached); writes to stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/aborted_snaps.py 2006 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import load_pbp_data, use_disk_cache_unless_configured

DATA = Path("data")
LINE_POSITIONS = ("C", "G", "T", "OL", "OT", "OG")
SAMPLES = 3


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def classify(season: int, positions: pl.DataFrame) -> pl.DataFrame:
    """Return the season's aborted plays with the charged player's and the rusher's roles."""
    qb_rows = (
        pl.read_parquet(DATA / f"{season}_qb_game_logs.parquet")
        .select("game_id", "team", "qb_id")
        .unique()
        .with_columns(pl.lit(value=True).alias("_is_qb"))
    )
    plays = load_pbp_data(season).filter(
        _flag("aborted_play")
        & ~_flag("two_point_attempt")
        & (pl.col("play_type") != "no_play").fill_null(value=True)
        & pl.col("posteam").is_not_null()
    )

    def role(id_column: str, alias: str, *, line: bool) -> pl.DataFrame:
        joined = (
            plays.select("play_id", "game_id", "posteam", id_column)
            .join(
                qb_rows,
                left_on=["game_id", "posteam", id_column],
                right_on=["game_id", "team", "qb_id"],
                how="left",
            )
            .join(positions, left_on=id_column, right_on="player_id", how="left")
            .with_columns(
                (pl.col("_is_qb").fill_null(value=False) | (pl.col("position") == "QB")).fill_null(
                    value=False
                )
            )
        )
        on_line = (
            pl.col("position").is_in(LINE_POSITIONS).fill_null(value=False)
            if line
            else pl.lit(value=False)
        )
        return joined.select(
            "play_id",
            "game_id",
            pl.when(pl.col(id_column).is_null())
            .then(pl.lit("none"))
            .when(pl.col("_is_qb").fill_null(value=False))
            .then(pl.lit("qb"))
            .when(on_line)
            .then(pl.lit("line"))
            .otherwise(pl.lit("other"))
            .alias(alias),
        )

    return (
        plays.select("play_id", "game_id", "play_type", "epa", "desc")
        .join(role("fumbled_1_player_id", "charged", line=True), on=["game_id", "play_id"])
        .join(role("rusher_player_id", "rusher", line=False), on=["game_id", "play_id"])
        .with_columns(pl.lit(season).alias("season"))
    )


def main() -> None:
    """Print the aborted-snap split for a range of seasons."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("first", type=int, help="First season.")
    parser.add_argument("last", type=int, help="Last season.")
    arguments = parser.parse_args()
    use_disk_cache_unless_configured()
    positions = (
        nfl.load_players()
        .select(pl.col("gsis_id").alias("player_id"), "position")
        .drop_nulls("player_id")
    )
    plays = pl.concat(
        [classify(season, positions) for season in range(arguments.first, arguments.last + 1)]
    )
    pairs = ["charged", "rusher"]
    summary = (
        plays.group_by(pairs)
        .agg(pl.len().alias("plays"), pl.col("epa").fill_null(0.0).sum().alias("epa"))
        .with_columns((pl.col("epa") / pl.col("plays")).alias("epa_per_play"))
        .sort("plays", descending=True)
    )
    by_season = (
        plays.group_by("season", "charged")
        .len()
        .pivot(on="charged", index="season", values="len")
        .fill_null(0)
        .sort("season")
    )
    samples = (
        plays.sort("season", "game_id", "play_id")
        .group_by(pairs, maintain_order=True)
        .head(SAMPLES)
        .select(*pairs, "season", pl.col("desc").str.slice(0, 130))
    )
    with pl.Config(
        float_precision=1,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=250,
        fmt_str_lengths=130,
        tbl_hide_dataframe_shape=True,
    ):
        sys.stdout.write(
            f"Aborted plays {arguments.first}-{arguments.last} by charged fumbler and rusher:\n"
            f"{summary}\nBy season and charged fumbler:\n{by_season}\nSamples:\n{samples}\n"
        )


if __name__ == "__main__":
    main()
