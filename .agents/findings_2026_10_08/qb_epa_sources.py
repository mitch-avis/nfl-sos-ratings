"""Compare the published QB EPA per dropback with the play-level EPA the garbage-time filter uses.

The published rate is nflverse's official passing EPA, less the passer's spike EPA, over dropbacks
(``qb_combined``'s season ``qb_epa_per_dropback``). The filter's quarterback refits read
play-by-play EPA credited to the passer on his dropbacks, in win-probability bins
(``qb_wp_bins``). For each season given, over the qualifying passers, this prints the largest
absolute difference between the two season rates and the passer it belongs to. Reads ``data/``;
writes to stdout.

Run from the repository root:

    .venv/bin/python .agents/findings_2026_10_08/qb_epa_sources.py 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

DATA = Path("data")


def season_gap(season: int) -> tuple[float, str]:
    """Return the largest published-versus-play-level gap among qualifying passers, and whose."""
    published = (
        pl.read_parquet(DATA / f"{season}_qb_combined.parquet")
        .filter(pl.col("qb_is_eligible"))
        .select("qb_id", "qb_name", "qb_epa_per_dropback")
    )
    play_level = (
        pl.read_parquet(DATA / f"{season}_qb_wp_bins.parquet")
        .group_by("qb_id")
        .agg(
            (pl.col("qb_wp_bin_epa").sum() / pl.col("qb_wp_bin_dropbacks").sum()).alias(
                "play_level"
            )
        )
    )
    gaps = (
        published.join(play_level, on="qb_id", how="inner")
        .with_columns((pl.col("qb_epa_per_dropback") - pl.col("play_level")).abs().alias("gap"))
        .sort("gap", descending=True)
    )
    top = gaps.row(0, named=True)
    return float(top["gap"]), str(top["qb_name"])


def main() -> None:
    """Print the largest gap for each season given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("seasons", type=int, nargs="+", help="Seasons to compare.")
    arguments = parser.parse_args()
    for season in arguments.seasons:
        gap, name = season_gap(season)
        sys.stdout.write(f"{season}: largest gap {gap:.1e} EPA per dropback ({name})\n")


if __name__ == "__main__":
    main()
