"""Rank ranges: how much a team's or quarterback's rating and rank move under game resampling.

A season's games are resampled with replacement and the published fit is rerun on each resample
with the season's fixed ridge penalties (``team_rating.bootstrap_team_ratings``,
``qb_rating.bootstrap_qb_ratings``). Every resample is ranked, and this module summarizes the
draws per team or quarterback: rating and rank quantiles, the probability of each rank, and the
chance of a top-5 or top-10 finish.

The ranges describe game-to-game sampling noise in the shrunken estimate. They say nothing about
whether the model itself is right, and a season in progress shows very wide ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
RANGE_QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
TOP_RANKS = (5, 10)


@dataclass(frozen=True, slots=True)
class RangeColumns:
    """Column names for one entity's rank ranges."""

    id: str
    rating: str
    rank: str


TEAM_RANGE_COLUMNS = RangeColumns(id="team", rating="team_rating", rank="team_rank")
QB_RANGE_COLUMNS = RangeColumns(id="qb_id", rating="adj_qb_epa_per_dropback", rank="qb_rank")


def quantile_suffix(level: float) -> str:
    """Return the column suffix for a quantile level, ``_q025`` for 0.025 through ``_q975``."""
    return f"_q{round(level * 1000):03d}"


def summarize_rank_ranges(
    draws: pl.DataFrame,
    published: pl.DataFrame,
    columns: RangeColumns,
    *,
    eligible: Collection[str] | None = None,
) -> pl.DataFrame:
    """Summarize bootstrap draws into rating and rank ranges, one row per ranked unit.

    Args:
        draws: Long rows with ``draw`` and the ``columns`` id and rating (higher is better).
        published: The published id and rating from the full fit.
        columns: The id column, the rating to rank on, and the name for the rank columns.
        eligible: Units to rank (for example published-eligible quarterbacks); every published
            unit when ``None``. Ranks in each draw are among the eligible units present in it.

    Returns:
        Per unit, ordered by published rank: the published rank (``rank_column``), rating and rank
        quantiles (``{rating_column}_q025`` ... and ``{rank_column}_q025`` ...), the share of
        draws without the unit, P(top 5) and P(top 10), and ``{rank_column}_probabilities``, whose
        k-th entry is P(rank = k + 1). Probabilities count every draw, so a unit missing from some
        draws has probabilities summing to less than one.

    """
    id_column, rating_column, rank_column = columns.id, columns.rating, columns.rank
    units = published.filter(
        pl.lit(value=True) if eligible is None else pl.col(id_column).is_in(list(eligible))
    ).with_columns(
        pl.col(rating_column).rank(method="min", descending=True).cast(pl.Int64).alias(rank_column)
    )
    unit_count = units.height
    draw_count = draws.get_column("draw").n_unique()
    ranked = draws.filter(
        pl.col(id_column).is_in(units.get_column(id_column).to_list())
    ).with_columns(
        pl.col(rating_column)
        .rank(method="min", descending=True)
        .over("draw")
        .cast(pl.Int64)
        .alias("_rank")
    )
    by_unit = {str(key[0]): frame for key, frame in ranked.group_by(id_column, maintain_order=True)}
    rows: list[dict[str, object]] = []
    for unit in units.sort(rank_column).iter_rows(named=True):
        frame = by_unit.get(str(unit[id_column]))
        ratings = np.empty(0) if frame is None else frame.get_column(rating_column).to_numpy()
        ranks = (
            np.empty(0, dtype=np.int64) if frame is None else frame.get_column("_rank").to_numpy()
        )
        row: dict[str, object] = {id_column: unit[id_column], rank_column: unit[rank_column]}
        for level in RANGE_QUANTILES:
            suffix = quantile_suffix(level)
            row[f"{rating_column}{suffix}"] = (
                float(np.quantile(ratings, level)) if ratings.size else None
            )
            row[f"{rank_column}{suffix}"] = (
                int(np.quantile(ranks, level, method="inverted_cdf")) if ranks.size else None
            )
        row[f"{rank_column}_missing_share"] = 1.0 - ranks.size / draw_count
        for top in TOP_RANKS:
            row[f"{rank_column}_top{top}_probability"] = float((ranks <= top).sum() / draw_count)
        row[f"{rank_column}_probabilities"] = (
            np.bincount(ranks, minlength=unit_count + 1)[1:] / draw_count
        ).tolist()
        rows.append(row)
    return pl.DataFrame(rows)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "QB_RANGE_COLUMNS",
    "RANGE_QUANTILES",
    "TEAM_RANGE_COLUMNS",
    "TOP_RANKS",
    "RangeColumns",
    "quantile_suffix",
    "summarize_rank_ranges",
]
