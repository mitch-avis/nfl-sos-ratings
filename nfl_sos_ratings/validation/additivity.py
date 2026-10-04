"""Additivity check: do strong offenses beat the additive prediction against weak defenses?

The ratings assume ``EPA per play = league average + offense - defense + home field``: a strong
offense gains the same amount against every defense. If that is wrong in the direction critics
suspect, strong offenses run up more than predicted against weak defenses and fall short against
strong ones, and ratings built on soft schedules run high.

For each pair of teams that met in a season, the ridge is refit without their games (at the
season's full-fit penalty) and their games are predicted from the refit, so no game helps predict
itself. Each offense and defense is placed in a tercile by its refit effect against cut points from
the season's full fit. The statistic is the weighted mean residual (actual minus predicted) of
top-tercile offenses against bottom-tercile defenses minus the same against top-tercile defenses.
Additivity predicts zero. A season-block bootstrap gives the interval. The same contrast is run
for passers against pass defenses.

Run ``nfl-sos-ratings check-additivity``; it only reads ``data/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.qb_rating import QB_UNIT_COLUMNS, fit_qb_ratings, qb_rating_rows
from nfl_sos_ratings.ridge import UnitColumns, fit_unit_ridge, predict_unit
from nfl_sos_ratings.team_rating import TEAM_UNIT_COLUMNS, fit_team_ratings, scrimmage_rows

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
TERCILE_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)

type Tier = Literal["bottom", "middle", "top"]

# Both teams in a game, whichever unit a row rates: a pair's games are held out together.
_PAIR_COLUMNS = ("team", "opponent_team")
_CELLS: tuple[tuple[Tier, Tier], ...] = (("top", "bottom"), ("top", "top"))


@dataclass(frozen=True, slots=True)
class TercileCuts:
    """The cut points that split one season's unit effects into thirds."""

    lower: float
    upper: float

    @classmethod
    def from_effects(cls, effects: Iterable[float]) -> TercileCuts:
        """Return the 1/3 and 2/3 quantiles (NumPy linear interpolation) of ``effects``."""
        lower, upper = np.quantile(np.fromiter(effects, dtype=np.float64), TERCILE_QUANTILES)
        return cls(lower=float(lower), upper=float(upper))

    def tier(self, effect: float) -> Tier:
        """Return ``top`` above the upper cut, ``bottom`` below the lower cut, else ``middle``."""
        if effect > self.upper:
            return "top"
        if effect < self.lower:
            return "bottom"
        return "middle"


def _pair_keys(rows: pl.DataFrame) -> pl.Series:
    """Return an order-free key naming the two teams of each row's game."""
    home, away = (pl.col(column) for column in _PAIR_COLUMNS)
    return rows.select(
        pl.when(home < away).then(home + "|" + away).otherwise(away + "|" + home).alias("pair")
    ).to_series()


def pair_holdout_residuals(
    rows: pl.DataFrame,
    columns: UnitColumns,
    *,
    ridge_lambda: float,
    offense_cuts: TercileCuts,
    defense_cuts: TercileCuts,
) -> pl.DataFrame:
    """Score every row against a refit that leaves out all games between the row's two teams.

    Args:
        rows: Offense-versus-defense rows with ``team`` and ``opponent_team`` naming the game's two
            teams, plus the columns named in ``columns``.
        columns: Ridge column names for ``rows``.
        ridge_lambda: The season's full-fit penalty, reused by every refit.
        offense_cuts: Tercile cut points for offense effects, from the season's full fit.
        defense_cuts: Tercile cut points for defense effects, from the season's full fit.

    Returns:
        ``rows``'s identity columns plus ``weight``, ``residual`` (actual minus predicted),
        ``offense_tier``, and ``defense_tier``. A row whose offense or defense has no other games
        cannot be predicted; its residual and tiers are null.

    """
    keyed = rows.with_columns(_pair_keys(rows))
    scored: list[pl.DataFrame] = []
    for pair in sorted(keyed.get_column("pair").unique().to_list()):
        held_out = keyed.filter(pl.col("pair") == pair)
        refit = fit_unit_ridge(
            keyed.filter(pl.col("pair") != pair), columns, ridge_lambda=ridge_lambda
        )
        predicted = predict_unit(refit, held_out, columns)
        offense_effects = [refit.offense.get(unit) for unit in held_out[columns.offense]]
        defense_effects = [refit.defense.get(unit) for unit in held_out[columns.defense]]
        scored.append(
            held_out.with_columns(
                pl.Series("predicted", predicted).fill_nan(None),
                pl.Series(
                    "offense_tier",
                    [None if e is None else offense_cuts.tier(e) for e in offense_effects],
                    dtype=pl.String,
                ),
                pl.Series(
                    "defense_tier",
                    [None if e is None else defense_cuts.tier(e) for e in defense_effects],
                    dtype=pl.String,
                ),
            )
        )
    identity = list(dict.fromkeys(("game_id", *_PAIR_COLUMNS, columns.offense)))
    weight = pl.lit(1.0) if columns.weight is None else pl.col(columns.weight).cast(pl.Float64)
    return (
        pl.concat(scored)
        .select(
            *identity,
            weight.alias("weight"),
            (pl.col(columns.response) - pl.col("predicted")).alias("residual"),
            pl.when(pl.col("predicted").is_not_null()).then(pl.col("offense_tier")),
            pl.when(pl.col("predicted").is_not_null()).then(pl.col("defense_tier")),
        )
        .sort(identity)
    )


def team_residuals(game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return pair-held-out scrimmage residuals for one season of team-game logs."""
    rows = scrimmage_rows(game_logs)
    ridge_lambda = fit_team_ratings(game_logs).scrimmage_lambda
    full = fit_unit_ridge(rows, TEAM_UNIT_COLUMNS, ridge_lambda=ridge_lambda)
    return pair_holdout_residuals(
        rows,
        TEAM_UNIT_COLUMNS,
        ridge_lambda=ridge_lambda,
        offense_cuts=TercileCuts.from_effects(full.offense.values()),
        defense_cuts=TercileCuts.from_effects(full.defense.values()),
    )


def passer_residuals(qb_game_logs: pl.DataFrame, qualifying_ids: Collection[str]) -> pl.DataFrame:
    """Return pair-held-out residuals for one season of passer-game rows.

    Passer cut points come from the full-fit effects of the ``qualifying_ids`` passers only, so
    the tiers compare starters rather than starters against shrunken backups.
    """
    rows = qb_rating_rows(qb_game_logs)
    ridge_lambda = fit_qb_ratings(qb_game_logs).ridge_lambda
    full = fit_unit_ridge(rows, QB_UNIT_COLUMNS, ridge_lambda=ridge_lambda)
    return pair_holdout_residuals(
        rows,
        QB_UNIT_COLUMNS,
        ridge_lambda=ridge_lambda,
        offense_cuts=TercileCuts.from_effects(
            effect for passer, effect in full.offense.items() if passer in qualifying_ids
        ),
        defense_cuts=TercileCuts.from_effects(full.defense.values()),
    )


def _cell_sums(rows: pl.DataFrame) -> pl.DataFrame:
    """Return per-season weight and weighted-residual sums for the two contrast cells."""
    sums = [
        pl.when((pl.col("offense_tier") == offense) & (pl.col("defense_tier") == defense)).then(
            value
        )
        for offense, defense in _CELLS
        for value in (pl.col("weight"), pl.col("weight") * pl.col("residual"))
    ]
    names = [f"{o}_{d}_{part}" for o, d in _CELLS for part in ("weight", "weighted")]
    return (
        rows.drop_nulls("residual")
        .group_by("season")
        .agg(*(expr.sum().alias(name) for expr, name in zip(sums, names, strict=True)))
        .sort("season")
    )


def _contrast_from_sums(sums: np.ndarray) -> np.ndarray:
    """Return the contrast for rows of summed ``[w_TB, wr_TB, w_TT, wr_TT]`` cell totals."""
    return sums[..., 1] / sums[..., 0] - sums[..., 3] / sums[..., 2]


def contrast(rows: pl.DataFrame) -> float:
    """Return the top-offense mean residual against bottom- minus top-tercile defenses.

    Args:
        rows: Residual rows with ``weight``, ``residual``, ``offense_tier``, and ``defense_tier``
            (``season`` is added when missing).

    """
    if "season" not in rows.columns:
        rows = rows.with_columns(pl.lit(0).alias("season"))
    totals = _cell_sums(rows).drop("season").to_numpy().sum(axis=0)
    return float(_contrast_from_sums(totals))


def season_block_interval(
    rows: pl.DataFrame, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return the 95% interval of :func:`contrast` over seasons resampled with replacement."""
    sums = _cell_sums(rows).drop("season").to_numpy()
    rng = np.random.default_rng(seed)
    sampled = np.array(
        [
            _contrast_from_sums(sums[rng.integers(0, sums.shape[0], sums.shape[0])].sum(axis=0))
            for _ in range(resamples)
        ]
    )
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return float(lower), float(upper)


def reading(lower: float, upper: float) -> str:
    """Return the pre-registered reading of a 95% interval for the contrast."""
    if lower > 0.0:
        return (
            "the additive model misses the claimed effect; a non-additive model becomes a new "
            "candidate for the walk-forward rule"
        )
    if upper < 0.0:
        return "the opposite of the claim: strong units gain less against weak opponents"
    return "no evidence against additivity"


def _cell_mean(rows: pl.DataFrame, offense: Tier, defense: Tier) -> tuple[float, float]:
    """Return the weighted mean residual and total weight of one offense-defense cell."""
    cell = rows.drop_nulls("residual").filter(
        (pl.col("offense_tier") == offense) & (pl.col("defense_tier") == defense)
    )
    total = float(cell.get_column("weight").sum())
    weighted = float((cell.get_column("weight") * cell.get_column("residual")).sum())
    return weighted / total, total


def _say(text: str) -> None:
    """Write one line of the report to standard output."""
    sys.stdout.write(f"{text}\n")


def _report(title: str, rows: pl.DataFrame, unit: str) -> None:
    """Print one contrast with its cells, interval, and reading."""
    lower, upper = season_block_interval(rows)
    _say(f"\n{title}")
    for offense, defense in _CELLS:
        mean, total = _cell_mean(rows, offense, defense)
        _say(
            f"  top offense vs {defense:<6} defense: mean residual {mean:+.4f} "
            f"over {total:,.0f} {unit}"
        )
    unscored = rows.get_column("residual").null_count()
    if unscored:
        _say(f"  rows without a prediction (unit seen only in the held-out games): {unscored}")
    _say(f"  contrast: {contrast(rows):+.4f}")
    _say(f"  95% season-block interval: {lower:+.4f} to {upper:+.4f}")
    _say(f"  Reading: {reading(lower, upper)}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``check-additivity`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings check-additivity",
        description=(
            "Test whether strong offenses and passers beat the additive prediction against weak "
            "defenses (reads data/, writes nothing)."
        ),
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs (default: {DATA_DIR})."
    )
    parser.add_argument("--start-season", type=int, default=START_YEAR, help="First season.")
    parser.add_argument("--end-season", type=int, default=END_YEAR, help="Last season.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the additivity check for teams and passers and print the results."""
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    seasons = range(args.start_season, args.end_season + 1)
    team_rows: list[pl.DataFrame] = []
    passer_rows: list[pl.DataFrame] = []
    for season in seasons:
        _say(f"Scoring {season}...")
        season_column = pl.lit(season).alias("season")
        team_rows.append(
            team_residuals(
                pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
            ).with_columns(season_column)
        )
        qualifying = set(
            pl.read_parquet(data_dir / f"{season}_qb_ratings.parquet")["qb_id"].to_list()
        )
        passer_rows.append(
            passer_residuals(
                pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet"), qualifying
            ).with_columns(season_column)
        )
    _say(
        f"\nAdditivity check, {args.start_season}-{args.end_season} "
        f"({BOOTSTRAP_RESAMPLES} season-block resamples, seed {BOOTSTRAP_SEED})"
    )
    _report("Teams (scrimmage EPA per play)", pl.concat(team_rows), "plays")
    _report("Passers (EPA per dropback)", pl.concat(passer_rows), "dropbacks")


__all__ = [
    "TercileCuts",
    "contrast",
    "main",
    "pair_holdout_residuals",
    "passer_residuals",
    "reading",
    "season_block_interval",
    "team_residuals",
]
