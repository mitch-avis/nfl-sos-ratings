"""Count quarterbacks' plays by type, then refit the QB rating on each candidate set of plays.

Written for the QB all-plays protocol (roadmap, "Q1"). The published QB rating counts a
quarterback's pass attempts and sacks (plays with a passer); nflverse leaves the passer empty on
scrambles, and designed runs are not counted. For every quarterback-game in
``data/{season}_qb_game_logs.parquet`` this counts what regular-season play-by-play (nflreadpy,
cached, through the package loader) credits to the same player id, game, and team:

- scrambles: ``play_type == "run"`` with ``qb_scramble``, the quarterback as ``rusher_player_id``;
- designed runs: his other runs nflverse marks as a rush (``rush``), aborted snaps left out;
- aborted snaps (``aborted_play``), kneel-downs (``play_type == "qb_kneel"``), and spikes
  (``play_type == "qb_spike"``, the quarterback as ``passer_player_id``), which no candidate
  counts as plays.

Two-point tries are left out of every count, as the published scramble and carry columns leave
them out. EPA is nflverse's play-by-play ``epa``. The first table gives, per season, each play
type's count and summed EPA over the quarterback-game rows (dropbacks and their EPA are the
published ``qb_dropbacks`` and ``qb_passing_epa``). The checks table counts the dropbacks,
scrambles, and aborted snaps of players in games where they have no QB row, and compares the
play-by-play counts with the published ``qb_scrambles``, ``qb_kneels``, and
``qb_designed_carries`` (which counts aborted snaps as designed runs), the official weekly rushing
EPA (``qb_rushing_epa``) with the play-by-play parts that make it up, and the published passing
EPA with the passer's ``qb_epa`` over his passes and spikes. Then it lists the players with 10 or
more dropbacks or scrambles in a season in games where they have no QB row.

With ``--compare SEASON`` it also refits the published QB fit (``fit_qb_ratings``, penalty
cross-validated each time) on each candidate's rows and prints every qualified passer's adjusted
EPA per play and rank: A, the published rows; B, scrambles added to dropbacks and dropback EPA;
C, B plus designed runs; and C with aborted snaps, for reference. It ends with the least, median,
and most of each play type a defense faced. Reads ``data/`` and play-by-play; writes to stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/qb_play_types.py \
        1999 2025 --compare 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_sos_ratings.data_loader import load_pbp_data, use_disk_cache_unless_configured
from nfl_sos_ratings.qb_rating import fit_qb_ratings

DATA = Path("data")
GAME_KEYS = ["game_id", "team", "qb_id"]
CANDIDATES = ("A", "B", "C", "C_aborted")
RUSHING_GAP_TOLERANCE = 1e-6
# Each play type the play-by-play pass counts, with the column of its summed EPA.
PLAY_TYPES = {
    "scrambles": "scramble_epa",
    "designed_runs": "designed_run_epa",
    "aborted": "aborted_epa",
    "kneels": "kneel_epa",
    "spikes": "spike_epa",
    "two_point_runs": "two_point_run_epa",
    "other_runs": "other_run_epa",
}
# Plus the passer's ``qb_epa`` over his passes and spikes, two-point tries included, which is how
# nflverse's official weekly ``passing_epa`` (the published ``qb_passing_epa``) is summed.
PART_COLUMNS = (*(column for pair in PLAY_TYPES.items() for column in pair), "pass_spike_qb_epa")
OFF_ROW_MINIMUM = 10
CHECK_COLUMNS = (
    "dropbacks_off_rows",
    "scrambles_off_rows",
    "aborted_off_rows",
    "scramble_mismatches",
    "kneel_mismatches",
    "carry_mismatches",
    "rushing_epa_gap_rows",
    "rushing_epa_gap_max",
    "passing_epa_gap_rows",
)


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def _count_and_epa(name: str, condition: pl.Expr) -> list[pl.Expr]:
    """Return the number of plays meeting ``condition`` and their summed EPA."""
    return [
        condition.cast(pl.Int64).sum().alias(name),
        pl.col("epa").fill_null(0.0).filter(condition).sum().alias(PLAY_TYPES[name]),
    ]


def qb_play_parts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return each player's plays of every type in ``PLAY_TYPES`` and their EPA, per game.

    Runs, scrambles, aborted snaps, and kneels go to ``rusher_player_id``; spikes to
    ``passer_player_id``. Rows are keyed like the QB game logs (``game_id``, ``team``,
    ``qb_id``). ``other_runs`` are runs nflverse marks neither as a rush nor as a scramble, which
    the published carry columns leave out; ``two_point_runs`` are two-point tries he ran.
    """
    two_point = _flag("two_point_attempt")
    run = (pl.col("play_type") == "run") & ~two_point
    carry = run & _flag("rush") & ~_flag("qb_scramble")
    rushes = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("rusher_player_id").is_not_null())
        .group_by("game_id", "posteam", "rusher_player_id")
        .agg(
            *_count_and_epa("scrambles", run & _flag("qb_scramble")),
            *_count_and_epa("designed_runs", carry & ~_flag("aborted_play")),
            *_count_and_epa("aborted", carry & _flag("aborted_play")),
            *_count_and_epa("kneels", (pl.col("play_type") == "qb_kneel") & ~two_point),
            *_count_and_epa(
                "two_point_runs", pl.col("play_type").is_in(["run", "qb_kneel"]) & two_point
            ),
            *_count_and_epa("other_runs", run & ~_flag("rush") & ~_flag("qb_scramble")),
        )
        .rename({"posteam": "team", "rusher_player_id": "qb_id"})
    )
    spikes = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("passer_player_id").is_not_null())
        .group_by("game_id", "posteam", "passer_player_id")
        .agg(
            *_count_and_epa("spikes", pl.col("play_type") == "qb_spike"),
            pl.col("qb_epa")
            .fill_null(0.0)
            .filter(pl.col("play_type").is_in(["pass", "qb_spike"]))
            .sum()
            .alias("pass_spike_qb_epa"),
        )
        .rename({"posteam": "team", "passer_player_id": "qb_id"})
    )
    return rushes.join(spikes, on=GAME_KEYS, how="full", coalesce=True).with_columns(
        pl.col(column).fill_null(0) for column in PART_COLUMNS
    )


def off_row_plays(pbp: pl.DataFrame, logs: pl.DataFrame) -> pl.DataFrame:
    """Return each player's dropbacks and scrambles in games where he has no QB game-log row.

    Dropbacks are plays with ``qb_dropback`` and a passer, scrambles as in :func:`qb_play_parts`,
    so these are the plays neither the published rows nor the added play types can count.
    """
    with_team = pbp.filter(pl.col("posteam").is_not_null())
    plays = pl.concat(
        [
            with_team.filter(
                _flag("qb_dropback") & pl.col("passer_player_id").is_not_null()
            ).select(
                "game_id",
                pl.col("posteam").alias("team"),
                pl.col("passer_player_id").alias("qb_id"),
                pl.col("passer_player_name").alias("name"),
                pl.lit(1).alias("dropbacks"),
                pl.lit(0).alias("scrambles"),
            ),
            with_team.filter(
                (pl.col("play_type") == "run")
                & _flag("qb_scramble")
                & ~_flag("two_point_attempt")
                & pl.col("rusher_player_id").is_not_null()
            ).select(
                "game_id",
                pl.col("posteam").alias("team"),
                pl.col("rusher_player_id").alias("qb_id"),
                pl.col("rusher_player_name").alias("name"),
                pl.lit(0).alias("dropbacks"),
                pl.lit(1).alias("scrambles"),
            ),
        ]
    )
    return (
        plays.join(logs.select(GAME_KEYS), on=GAME_KEYS, how="anti")
        .group_by("qb_id")
        .agg(
            pl.col("name").drop_nulls().first(),
            pl.col("team").unique().sort().str.join("/").alias("teams"),
            pl.col("game_id").n_unique().alias("games"),
            pl.col("dropbacks").sum(),
            pl.col("scrambles").sum(),
        )
        .sort("dropbacks", "scrambles", "qb_id", descending=[True, True, False])
    )


def with_parts(logs: pl.DataFrame, parts: pl.DataFrame) -> pl.DataFrame:
    """Return the QB game logs with the play-by-play parts of each quarterback-game joined on."""
    return logs.join(parts, on=GAME_KEYS, how="left").with_columns(
        pl.col(column).fill_null(0) for column in PART_COLUMNS
    )


def season_row(
    season: int, logs: pl.DataFrame, parts: pl.DataFrame, off_rows: pl.DataFrame
) -> dict[str, float | int]:
    """Return one season's play counts and EPA by type over the QB game-log rows, with checks."""
    rows = with_parts(logs, parts)

    def total(frame: pl.DataFrame, column: str) -> float:
        return float(frame.get_column(column).cast(pl.Float64).sum())

    rushing_parts = pl.sum_horizontal(
        "scramble_epa",
        "designed_run_epa",
        "aborted_epa",
        "kneel_epa",
        "two_point_run_epa",
        "other_run_epa",
    )
    rushing_gap = (pl.col("qb_rushing_epa") - rushing_parts).abs()
    carries = pl.col("designed_runs") + pl.col("aborted")
    return {
        "season": season,
        "dropbacks": int(total(rows, "qb_dropbacks")),
        "dropback_epa": total(rows, "qb_passing_epa"),
        **{column: total(rows, column) for column in PART_COLUMNS if column in rows.columns},
        "dropbacks_off_rows": int(total(off_rows, "dropbacks")),
        "scrambles_off_rows": int(total(off_rows, "scrambles")),
        "aborted_off_rows": int(total(parts, "aborted") - total(rows, "aborted")),
        "scramble_mismatches": rows.filter(pl.col("scrambles") != pl.col("qb_scrambles")).height,
        "kneel_mismatches": rows.filter(pl.col("kneels") != pl.col("qb_kneels")).height,
        "carry_mismatches": rows.filter(
            (carries != pl.col("qb_designed_carries"))
            | (
                (
                    pl.col("designed_run_epa")
                    + pl.col("aborted_epa")
                    - pl.col("qb_designed_rush_epa")
                )
                .abs()
                .gt(RUSHING_GAP_TOLERANCE)
            )
        ).height,
        "rushing_epa_gap_rows": int(
            rows.select((rushing_gap > RUSHING_GAP_TOLERANCE).sum()).item()
        ),
        "rushing_epa_gap_max": float(rows.select(rushing_gap.max()).item() or 0.0),
        "passing_epa_gap_rows": rows.filter(
            (pl.col("qb_passing_epa") - pl.col("pass_spike_qb_epa")).abs() > RUSHING_GAP_TOLERANCE
        ).height,
    }


def candidate_rows(logs: pl.DataFrame, parts: pl.DataFrame, candidate: str) -> pl.DataFrame:
    """Return the QB fit's rows for one candidate, with plays as weight and EPA per play.

    A is the published rows unchanged. B adds scrambles to dropbacks and their EPA to the dropback
    EPA; C adds designed runs on top of B; ``C_aborted`` adds aborted snaps on top of C. The
    columns keep their published names so the fit reads them unchanged.
    """
    if candidate == "A":
        return logs
    added = {
        "B": ("scrambles",),
        "C": ("scrambles", "designed_runs"),
        "C_aborted": ("scrambles", "designed_runs", "aborted"),
    }[candidate]
    plays = pl.sum_horizontal("qb_dropbacks", *added)
    epa = pl.sum_horizontal("qb_passing_epa", *(PLAY_TYPES[name] for name in added))
    return with_parts(logs, parts).with_columns(
        pl.when(plays > 0).then(epa / plays).otherwise(None).alias("qb_epa_per_dropback"),
        plays.alias("qb_dropbacks"),
    )


def comparison(
    logs: pl.DataFrame, parts: pl.DataFrame, combined: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Return every qualified passer's rating and rank under each candidate, and the penalties."""
    season_parts = (
        with_parts(logs, parts)
        .group_by("qb_id")
        .agg(
            pl.col("qb_dropbacks").sum().alias("dropbacks"),
            pl.col("scrambles").sum(),
            pl.col("designed_runs").sum(),
            pl.col("aborted").sum(),
        )
    )
    table = (
        combined.filter(pl.col("qb_is_eligible"))
        .select("qb_id", "qb_name", "team", "adj_qb_epa_per_dropback")
        .join(season_parts, on="qb_id", how="left")
    )
    penalties: dict[str, float] = {}
    for candidate in CANDIDATES:
        fit = fit_qb_ratings(candidate_rows(logs, parts, candidate))
        penalties[candidate] = fit.ridge_lambda
        table = table.join(
            fit.ratings.rename({"adj_qb_epa_per_dropback": f"adj_{candidate}"}),
            on="qb_id",
            how="left",
        ).with_columns(
            pl.col(f"adj_{candidate}")
            .rank("min", descending=True)
            .cast(pl.Int64)
            .alias(f"rank_{candidate}")
        )
    return table.sort("rank_A", "qb_id"), penalties


def _moves(table: pl.DataFrame, candidate: str) -> str:
    """Describe how ranks under ``candidate`` differ from A's."""
    moved = table.filter(pl.col(f"rank_{candidate}") != pl.col("rank_A"))
    largest = moved.select((pl.col(f"rank_{candidate}") - pl.col("rank_A")).abs().max()).item()
    spearman = table.select(pl.corr("adj_A", f"adj_{candidate}", method="spearman")).item()
    return (
        f"{candidate} vs A: {moved.height} rank changes (largest {largest or 0}), "
        f"Spearman {spearman:.3f}"
    )


def defense_volume(logs: pl.DataFrame, parts: pl.DataFrame) -> pl.DataFrame:
    """Return the least, median, and most of each QB play type a defense faced in the season."""
    faced = (
        with_parts(logs, parts)
        .group_by("opponent_team")
        .agg(
            pl.col("qb_dropbacks").sum().alias("dropbacks"),
            pl.col("scrambles").sum(),
            pl.col("designed_runs").sum(),
        )
    )
    volumes = pl.exclude("opponent_team")
    return pl.concat(
        [
            faced.select(pl.lit(label).alias("faced"), summary)
            for label, summary in (
                ("least", volumes.min()),
                ("median", volumes.median()),
                ("most", volumes.max()),
            )
        ],
        how="vertical_relaxed",
    )


def print_compare(season: int) -> None:
    """Print the candidate refits and the per-defense play volumes for ``season``."""
    logs = pl.read_parquet(DATA / f"{season}_qb_game_logs.parquet")
    combined = pl.read_parquet(DATA / f"{season}_qb_combined.parquet")
    parts = qb_play_parts(load_pbp_data(season))
    table, penalties = comparison(logs, parts, combined)
    gap = table.select((pl.col("adj_A") - pl.col("adj_qb_epa_per_dropback")).abs().max()).item()
    sys.stdout.write(
        f"\n{season}: {table.height} qualified passers; A refit against the published rating, "
        f"largest gap {gap:.1e}\nPenalties: "
        + ", ".join(f"{candidate} {value:.3f}" for candidate, value in penalties.items())
        + "\n"
        + "; ".join(_moves(table, candidate) for candidate in CANDIDATES[1:])
        + "\n"
    )
    with pl.Config(
        float_precision=3,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=250,
        tbl_hide_dataframe_shape=True,
    ):
        sys.stdout.write(f"{table.drop('qb_id', 'adj_qb_epa_per_dropback')}\n")
        sys.stdout.write(f"QB plays faced per defense, {season}:\n{defense_volume(logs, parts)}\n")


def main() -> None:
    """Print the play-type table for a range of seasons and, optionally, one season's refits."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("first", type=int, help="First season of the play-type table.")
    parser.add_argument("last", type=int, help="Last season of the play-type table.")
    parser.add_argument("--compare", type=int, help="Season to refit under each candidate.")
    arguments = parser.parse_args()
    use_disk_cache_unless_configured()
    rows: list[dict[str, float | int]] = []
    off_players: list[pl.DataFrame] = []
    for season in range(arguments.first, arguments.last + 1):
        logs = pl.read_parquet(DATA / f"{season}_qb_game_logs.parquet")
        pbp = load_pbp_data(season)
        off_rows = off_row_plays(pbp, logs)
        rows.append(season_row(season, logs, qb_play_parts(pbp), off_rows))
        off_players.append(
            off_rows.filter(
                (pl.col("dropbacks") >= OFF_ROW_MINIMUM) | (pl.col("scrambles") >= OFF_ROW_MINIMUM)
            ).select(pl.lit(season).alias("season"), pl.all())
        )
    table = pl.DataFrame(rows)
    shown = ["dropbacks", "scrambles", "designed_runs", "aborted", "kneels", "spikes"]
    epa_columns = {"dropbacks": "dropback_epa", **PLAY_TYPES}
    plays = table.select(
        "season", *(column for name in shown for column in (name, epa_columns[name]))
    )
    sums = plays.drop("season").sum()
    per_play = ", ".join(
        f"{name} {sums.get_column(epa_columns[name]).item() / sums.get_column(name).item():+.3f}"
        for name in shown
    )
    with pl.Config(
        float_precision=1,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=250,
        tbl_hide_dataframe_shape=True,
    ):
        sys.stdout.write(f"{plays}\nTotal:\n{sums}\nEPA per play, all seasons: {per_play}\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        sys.stdout.write(
            "Checks: dropbacks, scrambles, and aborted snaps by players in games where they have "
            "no QB row; QB-games whose "
            "play-by-play scramble, kneel, or carry count or carry EPA differs from the published "
            "column; official rushing EPA minus its play-by-play parts (rows above "
            f"{RUSHING_GAP_TOLERANCE:g}, largest gap); QB-games whose published passing EPA "
            "differs from the passer's qb_epa over his passes and spikes (rows above "
            f"{RUSHING_GAP_TOLERANCE:g}):\n{table.select('season', *CHECK_COLUMNS)}\n"
            f"Players with {OFF_ROW_MINIMUM} or more dropbacks or scrambles in a season in games "
            f"where they have no QB row:\n{pl.concat(off_players)}\n"
        )
    if arguments.compare is not None:
        print_compare(arguments.compare)


if __name__ == "__main__":
    main()
