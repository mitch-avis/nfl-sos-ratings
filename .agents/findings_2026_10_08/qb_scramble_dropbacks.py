"""Compare the QB rating with and without scrambles counted as dropbacks, for one season.

Written for the open question on QB dropbacks (roadmap, "Data notes"): the published QB rating
counts only plays with a passer (pass attempts and sacks), because nflverse leaves the passer empty
on scrambles, while team dropbacks include scrambles. This adds each quarterback's scrambles
(nflverse ``qb_scramble``, regular season, two-point tries left out) and their EPA to his dropbacks
and dropback EPA, refits the published QB rating both ways (the penalty cross-validated each time),
and prints the league totals and every qualified quarterback's raw and adjusted EPA per dropback
and rank both ways. Reads ``data/`` and loads play-by-play through nflreadpy (cached); writes to
stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/qb_scramble_dropbacks.py \
        --season 2025
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


def scramble_totals(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return each rusher's scrambles and their EPA per game, keyed like the QB game logs."""
    return (
        pbp.filter(
            (pl.col("qb_scramble").fill_null(0) == 1)
            & pl.col("rusher_player_id").is_not_null()
            & (pl.col("two_point_attempt").fill_null(0) == 0)
        )
        .group_by("game_id", "posteam", "rusher_player_id")
        .agg(
            pl.len().cast(pl.Int64).alias("scrambles"),
            pl.col("epa").fill_null(0.0).sum().alias("scramble_epa"),
        )
        .rename({"posteam": "team", "rusher_player_id": "qb_id"})
    )


def with_scrambles(logs: pl.DataFrame, scrambles: pl.DataFrame) -> pl.DataFrame:
    """Return the QB game logs with scrambles added to dropbacks and to dropback EPA."""
    dropbacks = pl.col("qb_dropbacks") + pl.col("scrambles")
    return (
        logs.join(scrambles, on=GAME_KEYS, how="left")
        .with_columns(pl.col("scrambles").fill_null(0), pl.col("scramble_epa").fill_null(0.0))
        .with_columns(
            pl.when(dropbacks > 0)
            .then((pl.col("qb_passing_epa") + pl.col("scramble_epa")) / dropbacks)
            .otherwise(None)
            .alias("qb_epa_per_dropback"),
            dropbacks.alias("qb_dropbacks"),
        )
    )


def raw_epa_per_dropback(logs: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return each passer's season EPA per dropback (dropback-weighted) as ``name``."""
    rated = logs.filter(pl.col("qb_dropbacks") > 0)
    weighted = (pl.col("qb_epa_per_dropback") * pl.col("qb_dropbacks")).sum()
    return rated.group_by("qb_id").agg((weighted / pl.col("qb_dropbacks").sum()).alias(name))


def comparison(logs: pl.DataFrame, scrambles: pl.DataFrame, combined: pl.DataFrame) -> pl.DataFrame:
    """Return every qualified passer's rating and rank with and without scrambles as dropbacks."""
    alternative = with_scrambles(logs, scrambles)
    now = fit_qb_ratings(logs).ratings.rename({"adj_qb_epa_per_dropback": "adj_now"})
    alt = fit_qb_ratings(alternative).ratings.rename({"adj_qb_epa_per_dropback": "adj_scr"})
    season_scrambles = alternative.group_by("qb_id").agg(
        pl.col("scrambles").sum(), pl.col("scramble_epa").sum()
    )
    return (
        combined.filter(pl.col("qb_is_eligible"))
        .select("qb_id", "qb_name", "team")
        .join(season_scrambles, on="qb_id", how="left")
        .join(raw_epa_per_dropback(logs, "epa_db_now"), on="qb_id", how="left")
        .join(raw_epa_per_dropback(alternative, "epa_db_scr"), on="qb_id", how="left")
        .join(now, on="qb_id", how="left")
        .join(alt, on="qb_id", how="left")
        .with_columns(
            pl.col("adj_now").rank("min", descending=True).cast(pl.Int64).alias("rank_now"),
            pl.col("adj_scr").rank("min", descending=True).cast(pl.Int64).alias("rank_scr"),
        )
        .sort("rank_now", "qb_id")
    )


def main() -> None:
    """Print the league totals and the per-passer comparison for the requested season."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True, help="Season to compare.")
    season = parser.parse_args().season
    use_disk_cache_unless_configured()
    logs = pl.read_parquet(DATA / f"{season}_qb_game_logs.parquet")
    combined = pl.read_parquet(DATA / f"{season}_qb_combined.parquet")
    scrambles = scramble_totals(load_pbp_data(season))
    table = comparison(logs, scrambles, combined)
    matched = logs.join(scrambles, on=GAME_KEYS, how="inner")
    moved = table.filter(pl.col("rank_now") != pl.col("rank_scr"))
    spearman = table.select(pl.corr("adj_now", "adj_scr", method="spearman")).item()
    sys.stdout.write(
        f"{season} QB game logs: {logs.get_column('qb_dropbacks').sum()} dropbacks with "
        f"{logs.get_column('qb_passing_epa').sum():.1f} passing EPA; scrambles on those rows: "
        f"{matched.get_column('scrambles').sum()} with "
        f"{matched.get_column('scramble_epa').sum():.1f} EPA\n"
        f"Qualified passers: {table.height}; rank changes: {moved.height} (largest "
        f"{(moved.get_column('rank_now') - moved.get_column('rank_scr')).abs().max() or 0}); "
        f"Spearman between the two ratings: {spearman:.3f}\n"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        sys.stdout.write(f"{table.drop('qb_id')}\n")


if __name__ == "__main__":
    main()
