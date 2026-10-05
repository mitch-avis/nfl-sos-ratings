"""Team and QB ratings refit on the plays a garbage-time filter keeps, for thresholds 0-30%.

A filter at X% keeps the plays whose win-probability bin (``wp_bins``) is X or above, plus the
plays without a bin, so 0% keeps every play. A bin's plays and EPA add up, so every design row's
kept plays and kept EPA at every threshold are a cumulative sum over its bins, computed once. A
threshold then swaps those weights (kept plays) and responses (kept EPA per kept play) into the
season's prebuilt design and solves; nothing is rebuilt per threshold.

What the view holds fixed, as an unvalidated exploration view rather than a published rating:

- Penalties and per-game scales are the season fit's, as the rating history and the
  head-to-head refits use. A filtered rating is the per-play estimate on the kept plays over a
  full game's worth of plays, so it reads on the published scale, and at 0% the team ratings and
  ``sos`` reproduce the published ones.
- ``sos`` and ``qb_faced_pass_defense`` repeat their head-to-head exclusion at every threshold.
- QB EPA is the play-by-play ``qb_epa`` the bins carry, at every threshold including 0%. The
  published QB rating uses official weekly passing EPA, so the two differ slightly even at 0%.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_sos_ratings.qb_rating import (
    QB_DROPBACKS_COLUMN,
    QB_EPA_PER_DROPBACK_COLUMN,
    QB_ID_COLUMN,
    QB_UNIT_COLUMNS,
    qb_rating_rows,
)
from nfl_sos_ratings.ridge import build_unit_design, solve_unit_design
from nfl_sos_ratings.team_rating import (
    TEAM_UNIT_COLUMNS,
    scrimmage_rows,
    special_teams_rows,
    team_ratings_from_unit_fits,
)
from nfl_sos_ratings.wp_bins import SCRIMMAGE_UNIT, SPECIAL_TEAMS_UNIT

if TYPE_CHECKING:
    from nfl_sos_ratings.qb_rating import QbRatingFit
    from nfl_sos_ratings.ridge import UnitColumns, UnitDesign, UnitFit
    from nfl_sos_ratings.team_rating import TeamRatingFit

type FloatArray = npt.NDArray[np.float64]

MAX_WP_THRESHOLD = 30
# Bins run 0-50 (``wp_bins``); one more slot holds the plays without a bin.
_LAST_BIN = 50
_UNBINNED_SLOT = _LAST_BIN + 1


def _check_threshold(threshold: int) -> None:
    """Raise ``ValueError`` unless ``threshold`` is a whole percentage from 0 to 30."""
    if not 0 <= threshold <= MAX_WP_THRESHOLD:
        msg = f"WP threshold must be from 0 to {MAX_WP_THRESHOLD}, got {threshold}"
        raise ValueError(msg)


def _kept_by_threshold(row_keys: pl.DataFrame, bins: pl.DataFrame, value: str) -> FloatArray:
    """Return, for each row of ``row_keys`` and each threshold 0-30, the kept total of ``value``.

    Bins match rows on every ``row_keys`` column; a row without bins keeps nothing.
    """
    indexed = row_keys.with_row_index("_row")
    joined = bins.join(indexed, on=row_keys.columns, how="inner")
    by_bin = np.zeros((row_keys.height, _UNBINNED_SLOT + 1), dtype=np.float64)
    np.add.at(
        by_bin,
        (
            np.asarray(joined.get_column("_row").to_numpy(), dtype=np.int64),
            np.asarray(
                joined.get_column("wp_bin").fill_null(_UNBINNED_SLOT).to_numpy(), dtype=np.int64
            ),
        ),
        np.asarray(joined.get_column(value).cast(pl.Float64).to_numpy(), dtype=np.float64),
    )
    at_or_above = np.cumsum(by_bin[:, _LAST_BIN::-1], axis=1)[:, ::-1]
    return at_or_above[:, : MAX_WP_THRESHOLD + 1] + by_bin[:, [_UNBINNED_SLOT]]


def _labels(rows: pl.DataFrame, column: str) -> npt.NDArray[np.str_]:
    """Return one column of unit labels as a NumPy string array."""
    return np.asarray(rows.get_column(column).cast(pl.String).to_list(), dtype=np.str_)


@dataclass(frozen=True, slots=True)
class _ThresholdDesign:
    """One fit's prebuilt design plus every row's kept weight and response sum by threshold."""

    design: UnitDesign
    ridge_lambda: float
    kept_weight: FloatArray
    kept_response_sum: FloatArray
    offense: npt.NDArray[np.str_]
    defense: npt.NDArray[np.str_]

    def solve(self, threshold: int, multipliers: FloatArray | None = None) -> UnitFit:
        """Solve on the plays kept at ``threshold``, rows scaled by ``multipliers`` (default 1)."""
        weights = self.kept_weight[:, threshold]
        response = np.divide(
            self.kept_response_sum[:, threshold],
            weights,
            out=np.zeros_like(weights),
            where=weights > 0.0,
        )
        return solve_unit_design(
            replace(self.design, weights=weights, response=response),
            self.ridge_lambda,
            np.ones_like(weights) if multipliers is None else multipliers,
        )

    def without(self, unit: str) -> FloatArray:
        """Return row multipliers that leave out every row involving ``unit`` on either side."""
        return ((self.offense != unit) & (self.defense != unit)).astype(np.float64)


def _threshold_design(
    rows: pl.DataFrame,
    columns: UnitColumns,
    ridge_lambda: float,
    bins: pl.DataFrame,
    values: tuple[str, str],
) -> _ThresholdDesign:
    """Build one fit's threshold design from its rows and the bins holding ``values``.

    ``values`` names the bins' play (weight) and EPA (response sum) columns. Rows lose the nulls
    ``build_unit_design`` would drop first, so the kept totals stay aligned with design rows.
    """
    rows = rows.drop_nulls([columns.offense, columns.defense, columns.response])
    keys = rows.select("game_id", columns.offense)
    return _ThresholdDesign(
        design=build_unit_design(rows, columns),
        ridge_lambda=ridge_lambda,
        kept_weight=_kept_by_threshold(keys, bins, values[0]),
        kept_response_sum=_kept_by_threshold(keys, bins, values[1]),
        offense=_labels(rows, columns.offense),
        defense=_labels(rows, columns.defense),
    )


class TeamWpFilter:
    """Team ratings and ``sos`` on the plays kept at any threshold, for one season.

    Built once per season from the team game logs, their win-probability bins, and the season fit
    (penalties and per-game scales); :meth:`ratings` then costs a few dozen small solves.
    """

    def __init__(self, game_logs: pl.DataFrame, bins: pl.DataFrame, fit: TeamRatingFit) -> None:
        """Build the scrimmage and special-teams designs and their kept plays by threshold."""
        self._fit = fit
        values = ("wp_bin_plays", "wp_bin_epa")
        self._units = (
            _threshold_design(
                scrimmage_rows(game_logs),
                TEAM_UNIT_COLUMNS,
                fit.scrimmage_lambda,
                bins.filter(pl.col("wp_unit") == SCRIMMAGE_UNIT),
                values,
            ),
            _threshold_design(
                special_teams_rows(game_logs),
                TEAM_UNIT_COLUMNS,
                fit.special_teams_lambda,
                bins.filter(pl.col("wp_unit") == SPECIAL_TEAMS_UNIT),
                values,
            ),
        )
        self._opponents: dict[str, list[str]] = {
            str(team): frame.get_column("opponent_team").cast(pl.String).to_list()
            for (team,), frame in game_logs.group_by("team")
        }

    def _rated(self, threshold: int, without: str | None = None) -> pl.DataFrame:
        """Return the four rating columns at ``threshold``, optionally refit without a team."""
        scrimmage, special_teams = self._units
        return team_ratings_from_unit_fits(
            scrimmage.solve(threshold, None if without is None else scrimmage.without(without)),
            special_teams.solve(
                threshold, None if without is None else special_teams.without(without)
            ),
            self._fit,
        )

    def _schedule_strength(self, threshold: int) -> pl.DataFrame:
        """Return each team's mean opponent rating, each opponent refit without the team."""
        teams = sorted(self._opponents)
        values: list[float | None] = []
        for team in teams:
            refit = self._rated(threshold, without=team)
            ratings = dict(
                zip(
                    refit.get_column("team").to_list(),
                    refit.get_column("team_rating").to_list(),
                    strict=True,
                )
            )
            faced = [ratings[opponent] for opponent in self._opponents[team] if opponent in ratings]
            values.append(float(np.mean(faced)) if faced else None)
        return pl.DataFrame(
            {"team": teams, "sos": values}, schema={"team": pl.String, "sos": pl.Float64}
        )

    def _kept_play_share(self, threshold: int) -> pl.DataFrame:
        """Return each team's kept share of its scrimmage and special-teams plays."""
        return (
            pl.DataFrame(
                {
                    "team": np.concatenate([unit.offense for unit in self._units]),
                    "kept": np.concatenate(
                        [unit.kept_weight[:, threshold] for unit in self._units]
                    ),
                    "total": np.concatenate([unit.kept_weight[:, 0] for unit in self._units]),
                }
            )
            .group_by("team")
            .agg((pl.col("kept").sum() / pl.col("total").sum()).alias("wp_kept_play_share"))
        )

    def ratings(self, threshold: int) -> pl.DataFrame:
        """Return every team's ratings on the plays kept at ``threshold``.

        Returns:
            One row per team: ``team``, ``offense_rating``, ``defense_rating``,
            ``special_teams_rating``, ``team_rating``, ``sos``, and ``wp_kept_play_share`` (kept
            plays over all of the team's scrimmage and special-teams plays), sorted by team.

        Raises:
            ValueError: If ``threshold`` is outside 0-30.

        """
        _check_threshold(threshold)
        return (
            self._rated(threshold)
            .join(self._schedule_strength(threshold), on="team", how="left")
            .join(self._kept_play_share(threshold), on="team", how="left")
            .sort("team")
        )


class QbWpFilter:
    """Adjusted EPA per dropback and faced pass defense on the dropbacks kept at any threshold.

    Built once per season from the QB game logs, their win-probability bins, and the season fit's
    penalty. Every threshold uses the play-level EPA in the bins (see the module docstring).
    """

    def __init__(self, qb_games: pl.DataFrame, bins: pl.DataFrame, fit: QbRatingFit) -> None:
        """Build the passer-versus-defense design and its kept dropbacks by threshold."""
        self._unit = _threshold_design(
            qb_rating_rows(qb_games),
            QB_UNIT_COLUMNS,
            fit.ridge_lambda,
            bins,
            ("qb_wp_bin_dropbacks", "qb_wp_bin_epa"),
        )

    def _faced_pass_defense(self, threshold: int, passers: list[str]) -> list[float | None]:
        """Return each passer's kept-dropback-weighted defense quality, refit without him."""
        unit = self._unit
        kept = unit.kept_weight[:, threshold]
        values: list[float | None] = []
        for passer in passers:
            defense = unit.solve(threshold, (unit.offense != passer).astype(np.float64)).defense
            own = (unit.offense == passer) & (kept > 0.0)
            faced = [
                (defense[str(opponent)], float(weight))
                for opponent, weight in zip(unit.defense[own], kept[own], strict=True)
                if str(opponent) in defense
            ]
            values.append(
                float(
                    np.average(
                        [quality for quality, _ in faced], weights=[weight for _, weight in faced]
                    )
                )
                if faced
                else None
            )
        return values

    def ratings(self, threshold: int) -> pl.DataFrame:
        """Return every passer's rating on the dropbacks kept at ``threshold``.

        Returns:
            One row per passer with a kept dropback: ``qb_id``, ``qb_dropbacks`` (kept),
            ``qb_epa_per_dropback`` (play-level, kept), ``adj_qb_epa_per_dropback``,
            ``qb_faced_pass_defense``, and ``wp_kept_dropback_share``, sorted by ``qb_id``.

        Raises:
            ValueError: If ``threshold`` is outside 0-30.

        """
        _check_threshold(threshold)
        unit = self._unit
        fit = unit.solve(threshold)
        totals = (
            pl.DataFrame(
                {
                    QB_ID_COLUMN: unit.offense,
                    "kept": unit.kept_weight[:, threshold],
                    "kept_epa": unit.kept_response_sum[:, threshold],
                    "total": unit.kept_weight[:, 0],
                }
            )
            .group_by(QB_ID_COLUMN)
            .agg(pl.col("kept").sum(), pl.col("kept_epa").sum(), pl.col("total").sum())
            .filter(pl.col("kept") > 0.0)
            .sort(QB_ID_COLUMN)
        )
        passers: list[str] = totals.get_column(QB_ID_COLUMN).to_list()
        return totals.select(
            QB_ID_COLUMN,
            pl.col("kept").round(0).cast(pl.Int64).alias(QB_DROPBACKS_COLUMN),
            (pl.col("kept_epa") / pl.col("kept")).alias(QB_EPA_PER_DROPBACK_COLUMN),
            pl.Series(
                "adj_qb_epa_per_dropback",
                [fit.intercept + fit.offense[passer] for passer in passers],
                dtype=pl.Float64,
            ),
            pl.Series(
                "qb_faced_pass_defense",
                self._faced_pass_defense(threshold, passers),
                dtype=pl.Float64,
            ),
            (pl.col("kept") / pl.col("total")).alias("wp_kept_dropback_share"),
        )


__all__ = ["MAX_WP_THRESHOLD", "QbWpFilter", "TeamWpFilter"]
