"""Passer holdout: were a passer's later games below what his rating season's model predicted?

The QB rating for a season is one ridge fit: ``EPA per dropback = league average + passer -
defense + home field``. This check takes that fit as fixed and predicts games it never saw: the
passer's postseason games that year and his games in the following regular season, each against
the opponent's pass-defense effect from the rating season. The statistic is the dropback-weighted
mean residual (actual minus predicted EPA per dropback) and its z-score,
``residual / (sigma / sqrt(dropbacks))``, with sigma estimated from the rating season's residuals
for qualifying passers. A z at or below -2 says the rating overstated the passer against these
opponents; anything above is within the noise the model expects.

Run ``nfl-sos-ratings check-passer``; it reads ``data/`` and downloads the postseason games.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR
from nfl_sos_ratings.data_loader import (
    load_playoff_qb_stats,
    load_playoff_schedule,
    use_disk_cache_unless_configured,
)
from nfl_sos_ratings.qb_rating import (
    QB_DROPBACKS_COLUMN,
    QB_EPA_PER_DROPBACK_COLUMN,
    QB_ID_COLUMN,
    QB_UNIT_COLUMNS,
    fit_qb_ratings,
    qb_rating_rows,
)
from nfl_sos_ratings.ridge import UnitFit, fit_unit_ridge, predict_unit

if TYPE_CHECKING:
    from collections.abc import Collection

DEFAULT_MODEL_SEASON = 2025
Z_THRESHOLD = -2.0

_ROW_COLUMNS = (
    "game_id",
    QB_ID_COLUMN,
    "opponent_team",
    "is_home",
    QB_DROPBACKS_COLUMN,
    QB_EPA_PER_DROPBACK_COLUMN,
)


@dataclass(frozen=True, slots=True)
class HoldoutPart:
    """One set of held-out games scored against the rating season's fit."""

    label: str
    games: int
    dropbacks: int
    actual: float
    predicted: float
    residual: float
    z: float


def residual_sigma(rows: pl.DataFrame, fit: UnitFit, qualifying_ids: Collection[str]) -> float:
    """Return the per-dropback residual standard deviation of the qualifying passers' games.

    A game's mean residual over ``n`` dropbacks has variance ``sigma**2 / n``, so ``sigma**2`` is
    estimated as the mean of ``n * residual**2`` over the qualifying passers' rows (in-sample, no
    degrees-of-freedom correction).
    """
    qualifying = rows.filter(pl.col(QB_ID_COLUMN).is_in(list(qualifying_ids)))
    residuals = qualifying.get_column(QB_EPA_PER_DROPBACK_COLUMN).to_numpy() - predict_unit(
        fit, qualifying, QB_UNIT_COLUMNS
    )
    weights = qualifying.get_column(QB_DROPBACKS_COLUMN).cast(pl.Float64).to_numpy()
    return float(np.sqrt(np.mean(weights * residuals**2)))


def score_part(label: str, rows: pl.DataFrame, fit: UnitFit, *, sigma: float) -> HoldoutPart:
    """Score held-out passer-game rows against ``fit``.

    Raises:
        ValueError: If the fit has no effect for a row's passer or opponent.

    """
    predicted = predict_unit(fit, rows, QB_UNIT_COLUMNS)
    if np.isnan(predicted).any():
        unknown = rows.filter(pl.Series(np.isnan(predicted))).select(QB_ID_COLUMN, "opponent_team")
        msg = f"The model season has no effect for {unknown.rows()}"
        raise ValueError(msg)
    weights = rows.get_column(QB_DROPBACKS_COLUMN).cast(pl.Float64).to_numpy()
    actual = rows.get_column(QB_EPA_PER_DROPBACK_COLUMN).cast(pl.Float64).to_numpy()
    dropbacks = float(weights.sum())
    residual = float(np.average(actual - predicted, weights=weights))
    return HoldoutPart(
        label=label,
        games=rows.height,
        dropbacks=int(dropbacks),
        actual=float(np.average(actual, weights=weights)),
        predicted=float(np.average(predicted, weights=weights)),
        residual=residual,
        z=residual / (sigma / math.sqrt(dropbacks)),
    )


def postseason_rows(playoff_qb: pl.DataFrame, schedule: pl.DataFrame, qb_id: str) -> pl.DataFrame:
    """Return one passer's postseason rows with the opponent and venue.

    The host is home and the visitor away; the Super Bowl is a neutral site (null ``is_home``).
    """
    team = pl.col("team_abbr")
    return (
        playoff_qb.filter((pl.col(QB_ID_COLUMN) == qb_id) & (pl.col(QB_DROPBACKS_COLUMN) > 0))
        .join(schedule.select("game_id", "game_type", "home_team", "away_team"), on="game_id")
        .with_columns(
            pl.when(team == pl.col("home_team"))
            .then(pl.col("away_team"))
            .otherwise(pl.col("home_team"))
            .alias("opponent_team"),
            pl.when(pl.col("game_type") != "SB").then(team == pl.col("home_team")).alias("is_home"),
        )
        .sort("week")
        .select(_ROW_COLUMNS)
    )


def reading(z: float) -> str:
    """Return the pre-registered reading of one part's z-score."""
    if z <= Z_THRESHOLD:
        return "the model season's rating overstated the passer against these opponents"
    return "within the noise the model expects"


def _say(text: str) -> None:
    """Write one line of the report to standard output."""
    sys.stdout.write(f"{text}\n")


def _resolve_qb_id(model_logs: pl.DataFrame, name: str, season: int) -> str:
    """Return the one passer ID with ``name`` in the model season, or exit with a message."""
    ids = model_logs.filter(pl.col("qb_name") == name).get_column(QB_ID_COLUMN).unique().to_list()
    if len(ids) != 1:
        msg = (
            f"No {season} passer named {name}"
            if not ids
            else f"{name} matches several {season} passer IDs: {', '.join(sorted(ids))}"
        )
        raise SystemExit(msg)
    return str(ids[0])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``check-passer`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings check-passer",
        description=(
            "Score a passer's postseason games and next regular season against his rating "
            "season's QB model (reads data/, downloads the postseason, writes nothing)."
        ),
    )
    parser.add_argument("--name", required=True, help='Passer name as in qb_name ("Drake Maye").')
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs (default: {DATA_DIR})."
    )
    parser.add_argument(
        "--model-season",
        type=int,
        default=DEFAULT_MODEL_SEASON,
        help=f"Season whose QB fit makes the predictions (default: {DEFAULT_MODEL_SEASON}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the passer holdout and print each part with its reading."""
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    season: int = args.model_season
    later = season + 1
    model_logs = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet")
    qb_id = _resolve_qb_id(model_logs, args.name, season)
    qualifying = set(pl.read_parquet(data_dir / f"{season}_qb_ratings.parquet")[QB_ID_COLUMN])

    rows = qb_rating_rows(model_logs)
    ridge_lambda = fit_qb_ratings(model_logs).ridge_lambda
    fit = fit_unit_ridge(rows, QB_UNIT_COLUMNS, ridge_lambda=ridge_lambda)
    sigma = residual_sigma(rows, fit, qualifying)

    use_disk_cache_unless_configured()
    postseason = postseason_rows(
        load_playoff_qb_stats(season), load_playoff_schedule(season), qb_id
    )
    later_path = data_dir / f"{later}_qb_game_logs.parquet"
    later_rows = (
        qb_rating_rows(pl.read_parquet(later_path))
        .filter(pl.col(QB_ID_COLUMN) == qb_id)
        .sort("week")
        .select(_ROW_COLUMNS)
        if later_path.exists()
        else postseason.clear()
    )

    _say(f"Passer holdout: {args.name} ({qb_id}) against the {season} QB fit")
    _say(
        f"  {season} adjusted EPA per dropback {fit.intercept + fit.offense[qb_id]:+.3f}; "
        f"ridge penalty {ridge_lambda:g}; sigma {sigma:.3f} per dropback"
    )
    parts = [
        (f"{season} postseason", postseason),
        (f"{later} regular season", later_rows),
        ("both", pl.concat([postseason, later_rows], how="vertical_relaxed")),
    ]
    for label, part_rows in parts:
        if part_rows.is_empty():
            _say(f"\n{label}: no games")
            continue
        part = score_part(label, part_rows, fit, sigma=sigma)
        _say(f"\n{part.label}: {part.games} games, {part.dropbacks} dropbacks")
        _say(f"  actual EPA per dropback {part.actual:+.3f}, predicted {part.predicted:+.3f}")
        _say(f"  mean residual {part.residual:+.3f}, z {part.z:+.2f}")
        _say(f"  Reading: {reading(part.z)}")


__all__ = [
    "HoldoutPart",
    "main",
    "postseason_rows",
    "reading",
    "residual_sigma",
    "score_part",
]
