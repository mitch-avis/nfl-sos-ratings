"""Count the play-by-play facts the QB all-plays protocol's review turned up, per season.

Written for the QB all-plays protocol (roadmap, "Q1"), after its independent review. For each
regular season it prints:

- scrambles (``qb_scramble``) and those on run plays (``play_type == "run"``), and the scrambles
  the team scrimmage-snap filter (``pbp_expressions.scrimmage_snap_expr``: a dropback, rush, kneel,
  or spike) leaves out, split into run plays (missing from every published team column built on
  it) and penalty-wiped ones (``no_play``, which every QB candidate leaves out too), with their EPA;
- run plays flagged neither as a rush nor as a scramble (two-point tries out), and how many of them
  the filter counts as scrimmage snaps;
- penalty-wiped runs the same filter keeps (``play_type == "no_play"`` with ``rush``), and their
  EPA;
- aborted snaps that became dropbacks with a passer (``aborted_play`` and ``qb_dropback``, two-point
  tries out), which the published QB rating counts, and their ``qb_epa``;
- QB-game rows in ``data/{season}_qb_game_logs.parquet`` without a dropback;
- team-games whose schedule-listed starter (``home_qb_id`` / ``away_qb_id``) has no QB row in that
  game; team-games without a first-quarter quarterback snap (a first-quarter play whose passer or
  rusher has a QB row with at least one dropback, the rows the QB fit uses, for that team in that
  game); and team-games whose first such snap, by ``play_id``, names a different player than the
  schedule.

Then it prints, for the players named in the protocol's "QB rows by career position" note, the
position nflverse's weekly rosters list for each season and the latest position in
``nflreadpy.load_players()``, which the QB rows use today. Reads ``data/``, play-by-play,
schedules, weekly rosters, and the players file (nflreadpy, cached, through the package loaders
where they exist); writes to stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/qb_protocol_checks.py \
        1999 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import (
    load_pbp_data,
    load_schedule,
    use_disk_cache_unless_configured,
)
from nfl_sos_ratings.pbp_expressions import scrimmage_snap_expr

DATA = Path("data")
GAME_KEYS = ["game_id", "team", "qb_id"]
# The players the protocol's "QB rows by career position" note names, with the seasons it names.
NAMED_PLAYERS = {
    "T.Pryor": (2012, 2013, 2016),
    "T.Hill": (2020, 2021, 2022, 2023),
    "L.Thomas": (2014,),
    "K.Hinton": (2020,),
}


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def _count_and_epa(prefix: str, condition: pl.Expr, epa: str = "epa") -> list[pl.Expr]:
    """Return the number of plays meeting ``condition`` and their summed ``epa`` column."""
    return [
        condition.cast(pl.Int64).sum().alias(prefix),
        pl.col(epa).fill_null(0.0).filter(condition).sum().alias(f"{prefix}_epa"),
    ]


def play_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return the season's scramble, wiped-run, other-run, and aborted-dropback counts and EPA."""
    snap = scrimmage_snap_expr(pbp.columns)
    scramble = _flag("qb_scramble")
    wiped = pl.col("play_type") == "no_play"
    other_run = (
        (pl.col("play_type") == "run") & ~_flag("rush") & ~scramble & ~_flag("two_point_attempt")
    )
    return pbp.select(
        scramble.cast(pl.Int64).sum().alias("scrambles"),
        (scramble & (pl.col("play_type") == "run")).cast(pl.Int64).sum().alias("scrambles_run"),
        *_count_and_epa("scrambles_missed_run", scramble & ~snap & (pl.col("play_type") == "run")),
        *_count_and_epa("scrambles_missed_wiped", scramble & ~snap & wiped),
        *_count_and_epa("wiped_runs_kept", snap & wiped & _flag("rush")),
        other_run.cast(pl.Int64).sum().alias("other_runs"),
        (other_run & snap).cast(pl.Int64).sum().alias("other_run_snaps"),
        *_count_and_epa(
            "aborted_dropbacks",
            _flag("aborted_play")
            & _flag("qb_dropback")
            & pl.col("passer_player_id").is_not_null()
            & ~_flag("two_point_attempt"),
            epa="qb_epa",
        ),
    )


def first_qb_snaps(pbp: pl.DataFrame, logs: pl.DataFrame) -> pl.DataFrame:
    """Return each team-game's first first-quarter snap by a passer or rusher with a fit row."""
    rows = logs.filter(pl.col("qb_dropbacks").fill_null(0) > 0).select(GAME_KEYS).unique()
    plays = pbp.filter(pl.col("posteam").is_not_null() & (pl.col("qtr") == 1)).select(
        "game_id",
        pl.col("posteam").alias("team"),
        "play_id",
        "passer_player_id",
        "rusher_player_id",
    )
    candidates = pl.concat(
        [
            plays.select("game_id", "team", "play_id", pl.col(column).alias("qb_id"))
            for column in ("passer_player_id", "rusher_player_id")
        ]
    ).join(rows, on=GAME_KEYS, how="semi")
    return candidates.sort("play_id").group_by("game_id", "team", maintain_order=True).first()


def starter_checks(season: int, pbp: pl.DataFrame, logs: pl.DataFrame) -> dict[str, int]:
    """Return the season's team-games and their starter disagreements and unknown starters."""
    schedule = load_schedule(season).filter(pl.col("game_id").is_in(logs["game_id"].unique()))
    listed = pl.concat(
        [
            schedule.select(
                "game_id",
                pl.col(f"{side}_team").alias("team"),
                pl.col(f"{side}_qb_id").alias("qb_id"),
            )
            for side in ("home", "away")
        ]
    )
    no_row = listed.join(logs.select(GAME_KEYS), on=GAME_KEYS, how="anti").height
    first = first_qb_snaps(pbp, logs).select("game_id", "team", pl.col("qb_id").alias("first_id"))
    paired = listed.join(first, on=["game_id", "team"], how="left")
    known = paired.filter(pl.col("first_id").is_not_null())
    return {
        "team_games": listed.height,
        "listed_without_qb_row": no_row,
        "starter_unknown": paired.height - known.height,
        "first_snap_differs": known.filter(pl.col("qb_id").ne_missing(pl.col("first_id"))).height,
    }


def season_row(season: int) -> dict[str, float | int]:
    """Return one season's counts."""
    pbp = load_pbp_data(season)
    logs = pl.read_parquet(DATA / f"{season}_qb_game_logs.parquet")
    counts = play_counts(pbp).row(0, named=True)
    return {
        "season": season,
        **counts,
        "qb_rows_without_dropback": logs.filter(pl.col("qb_dropbacks").fill_null(0) == 0).height,
        **starter_checks(season, pbp, logs),
    }


def named_positions(first: int, last: int) -> pl.DataFrame:
    """Return the weekly-roster and latest positions of the players the protocol names."""
    players = nfl.load_players().select(
        pl.col("gsis_id").alias("qb_id"), pl.col("position").alias("latest_position")
    )
    frames: list[pl.DataFrame] = []
    for name, seasons in NAMED_PLAYERS.items():
        for season in seasons:
            if not first <= season <= last:
                continue
            logs_path = DATA / f"{season}_qb_game_logs.parquet"
            pbp = load_pbp_data(season)
            ids = (
                pbp.filter(
                    (pl.col("passer_player_name") == name) | (pl.col("rusher_player_name") == name)
                )
                .select(pl.coalesce("passer_player_id", "rusher_player_id").alias("qb_id"))
                .drop_nulls()
                .unique()
            )
            rosters = (
                nfl.load_rosters_weekly(seasons=season)
                .filter(pl.col("game_type") == "REG")
                .select(
                    pl.col("gsis_id").alias("qb_id"), pl.col("position").alias("roster_position")
                )
                .join(ids, on="qb_id", how="semi")
                .group_by("qb_id")
                .agg(pl.col("roster_position").drop_nulls().unique().sort().str.join("/"))
            )
            in_rows = pl.read_parquet(logs_path).join(ids, on="qb_id", how="semi").height
            frames.append(
                ids.join(rosters, on="qb_id", how="left")
                .join(players, on="qb_id", how="left")
                .select(
                    pl.lit(name).alias("player"),
                    pl.lit(season).alias("season"),
                    "qb_id",
                    "roster_position",
                    "latest_position",
                    pl.lit(in_rows).alias("qb_rows"),
                )
            )
    return pl.concat(frames)


def main() -> None:
    """Print the per-season counts and the named players' positions."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("first", type=int, help="First season.")
    parser.add_argument("last", type=int, help="Last season.")
    arguments = parser.parse_args()
    use_disk_cache_unless_configured()
    table = pl.DataFrame(
        [season_row(season) for season in range(arguments.first, arguments.last + 1)]
    )
    with pl.Config(
        float_precision=1,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=300,
        tbl_hide_dataframe_shape=True,
    ):
        sys.stdout.write(f"{table}\n")
        for window in ((2003, 2025), (2006, 2025)):
            sums = table.filter(pl.col("season").is_between(*window)).drop("season").sum()
            sys.stdout.write(f"Total {window[0]}-{window[1]}:\n{sums}\n")
        sys.stdout.write(
            f"Named players' positions:\n{named_positions(arguments.first, arguments.last)}\n"
        )


if __name__ == "__main__":
    main()
