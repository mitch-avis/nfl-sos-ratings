"""Quarterback rating: EPA per dropback adjusted for the pass defenses actually faced.

``adj_qb_epa_per_dropback`` is the passer's EPA per dropback after the simultaneous solve in
``nfl_sos_ratings.ridge``: every passer is judged against the defenses he faced, and every defense
against all the passers it faced. It reads on the same scale as raw EPA per dropback, with the
league-average passer at the fitted intercept. Rows are weighted by dropbacks, and the ridge
penalty pulls small samples toward average, so a few backup snaps cannot produce an extreme value.

``qb_faced_pass_defense`` is the dropback-weighted average quality of the pass defenses the
passer faced, in EPA per dropback (positive means tougher defenses). Each defense is rated from a
refit that leaves out the passer's own dropbacks, so a passer who shreds a defense cannot make
that defense look weaker in his own schedule.

The rating history (``fit_qb_ratings_by_week``) refits the rating on the games through each week
with the season's penalty, so each week's row shows the rating as the evidence then supported it.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_sos_ratings.ridge import UnitColumns, build_unit_design, fit_unit_ridge, solve_unit_design

QB_ID_COLUMN = "qb_id"
QB_DROPBACKS_COLUMN = "qb_dropbacks"
QB_EPA_PER_DROPBACK_COLUMN = "qb_epa_per_dropback"

_REQUIRED_COLUMNS = (
    "game_id",
    QB_ID_COLUMN,
    "opponent_team",
    QB_DROPBACKS_COLUMN,
    QB_EPA_PER_DROPBACK_COLUMN,
)
QB_UNIT_COLUMNS = UnitColumns(
    response=QB_EPA_PER_DROPBACK_COLUMN,
    weight=QB_DROPBACKS_COLUMN,
    offense=QB_ID_COLUMN,
)


@dataclass(frozen=True, slots=True)
class QbRatingFit:
    """Adjusted QB ratings plus the penalty a head-to-head-excluded refit must reuse."""

    ratings: pl.DataFrame
    ridge_lambda: float


def _require_columns(qb_games: pl.DataFrame) -> None:
    """Raise ``ValueError`` naming every column the QB rating needs but lacks."""
    missing = sorted(set(_REQUIRED_COLUMNS) - set(qb_games.columns))
    if missing:
        msg = f"QB game rows are missing QB-rating columns: {', '.join(missing)}"
        raise ValueError(msg)


def _rated_rows(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return the QB-game rows with at least one dropback."""
    return qb_games.filter(pl.col(QB_DROPBACKS_COLUMN) > 0)


def qb_rating_rows(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return the passer-game rows the QB fit uses: those with a dropback.

    Fit them with ``QB_UNIT_COLUMNS``.

    Raises:
        ValueError: If a QB-rating column is missing.

    """
    _require_columns(qb_games)
    return _rated_rows(qb_games)


def fit_qb_ratings(qb_games: pl.DataFrame, *, ridge_lambda: float | None = None) -> QbRatingFit:
    """Fit adjusted EPA per dropback for every passer with a dropback.

    Args:
        qb_games: One row per passer-game with ``game_id``, ``qb_id``, ``opponent_team``,
            ``qb_dropbacks``, ``qb_epa_per_dropback``, and optionally ``is_home``.
        ridge_lambda: Fixed ridge penalty; cross-validated when ``None``.

    Returns:
        One row per passer with ``qb_id`` and ``adj_qb_epa_per_dropback``, plus the penalty.

    Raises:
        ValueError: If a required column is missing or no row has a dropback.

    """
    _require_columns(qb_games)
    fit = fit_unit_ridge(_rated_rows(qb_games), QB_UNIT_COLUMNS, ridge_lambda=ridge_lambda)
    passers = sorted(fit.offense)
    return QbRatingFit(
        ratings=pl.DataFrame(
            {
                QB_ID_COLUMN: passers,
                "adj_qb_epa_per_dropback": [fit.intercept + fit.offense[qb] for qb in passers],
            }
        ),
        ridge_lambda=fit.ridge_lambda,
    )


def fit_qb_ratings_by_week(qb_games: pl.DataFrame, fit: QbRatingFit) -> pl.DataFrame:
    """Return each passer's adjusted EPA per dropback as of every week.

    Every week refits the passer-games through that week with the season fit's penalty, for the
    reason :func:`nfl_sos_ratings.team_rating.fit_team_ratings_by_week` gives: early weeks are too
    small for cross-validation to choose one reliably. Faced pass defense is not refit week by
    week.

    Args:
        qb_games: The passer-game rows passed to :func:`fit_qb_ratings`, plus ``week``.
        fit: The season fit of ``qb_games`` whose penalty every week reuses.

    Returns:
        One row per week and passer with a dropback by then: ``week``, ``qb_id``,
        ``qb_games_played`` and ``qb_dropbacks`` through that week, and
        ``adj_qb_epa_per_dropback``.

    Raises:
        ValueError: If ``week`` or a QB-rating column is missing.

    """
    _require_columns(qb_games)
    if "week" not in qb_games.columns:
        msg = "QB game rows need a week column for the weekly rating history"
        raise ValueError(msg)
    rows = _rated_rows(qb_games)
    weeks: list[int] = sorted(rows.get_column("week").unique().to_list())
    frames: list[pl.DataFrame] = []
    for week in weeks:
        through_week = rows.filter(pl.col("week") <= week)
        volume = through_week.group_by(QB_ID_COLUMN).agg(
            pl.len().cast(pl.Int64).alias("qb_games_played"),
            pl.col(QB_DROPBACKS_COLUMN).sum().cast(pl.Int64).alias("qb_dropbacks"),
        )
        ratings = fit_qb_ratings(through_week, ridge_lambda=fit.ridge_lambda).ratings
        frames.append(
            ratings.join(volume, on=QB_ID_COLUMN).select(
                pl.lit(week, dtype=pl.Int64).alias("week"),
                QB_ID_COLUMN,
                "qb_games_played",
                "qb_dropbacks",
                "adj_qb_epa_per_dropback",
            )
        )
    return pl.concat(frames)


class QbRatingResampler:
    """Refit the QB rating on resampled games, reusing one fit's fixed penalty.

    The QB counterpart of ``team_rating.TeamRatingResampler``: :meth:`ratings` takes a count per
    game and returns what :func:`fit_qb_ratings` would return on the resampled games.
    """

    def __init__(self, qb_games: pl.DataFrame, fit: QbRatingFit) -> None:
        """Build the passer-versus-defense design for ``qb_games``."""
        _require_columns(qb_games)
        self._ridge_lambda = fit.ridge_lambda
        rows = _rated_rows(qb_games).drop_nulls(
            [QB_ID_COLUMN, "opponent_team", QB_EPA_PER_DROPBACK_COLUMN]
        )
        self.game_ids: list[str] = sorted(rows.get_column("game_id").unique().to_list())
        index = {game_id: position for position, game_id in enumerate(self.game_ids)}
        self._design = build_unit_design(rows, QB_UNIT_COLUMNS)
        self._games = np.array(
            [index[game_id] for game_id in rows.get_column("game_id")], dtype=np.int64
        )

    def ratings(self, game_counts: npt.ArrayLike) -> pl.DataFrame:
        """Return ``qb_id`` and ``adj_qb_epa_per_dropback`` for passers in the drawn games."""
        counts = np.asarray(game_counts, dtype=np.float64)
        fit = solve_unit_design(self._design, self._ridge_lambda, counts[self._games])
        passers = sorted(fit.offense)
        return pl.DataFrame(
            {
                QB_ID_COLUMN: passers,
                "adj_qb_epa_per_dropback": [fit.intercept + fit.offense[qb] for qb in passers],
            },
            schema={QB_ID_COLUMN: pl.String, "adj_qb_epa_per_dropback": pl.Float64},
        )


def bootstrap_qb_ratings(
    qb_games: pl.DataFrame, fit: QbRatingFit, *, resamples: int, seed: int
) -> pl.DataFrame:
    """Return ``adj_qb_epa_per_dropback`` for every passer in each game-bootstrap resample.

    Returns:
        Long rows with ``draw``, ``qb_id``, and ``adj_qb_epa_per_dropback``; a passer with no
        dropbacks in a resample has no row for that draw.

    """
    resampler = QbRatingResampler(qb_games, fit)
    rng = np.random.default_rng(seed)
    game_count = len(resampler.game_ids)
    frames = [
        resampler.ratings(
            np.bincount(rng.integers(0, game_count, game_count), minlength=game_count)
        ).select(pl.lit(draw).alias("draw"), QB_ID_COLUMN, "adj_qb_epa_per_dropback")
        for draw in range(resamples)
    ]
    return pl.concat(frames)


def compute_qb_faced_pass_defense(qb_games: pl.DataFrame, fit: QbRatingFit) -> pl.DataFrame:
    """Return each passer's dropback-weighted faced pass defense, head-to-head excluded.

    Args:
        qb_games: The same passer-game rows passed to :func:`fit_qb_ratings`.
        fit: The full fit whose penalty the refits reuse.

    Returns:
        One row per passer with ``qb_id`` and ``qb_faced_pass_defense`` in EPA per dropback.
        Defenses that have faced no other passer yet are skipped, and the value is null when
        none of the passer's defenses can be rated.

    """
    _require_columns(qb_games)
    rows = _rated_rows(qb_games)
    passers: list[str] = fit.ratings.get_column(QB_ID_COLUMN).to_list()
    values: list[float | None] = []
    for passer in passers:
        others = rows.filter(pl.col(QB_ID_COLUMN) != passer)
        defense = (
            fit_unit_ridge(others, QB_UNIT_COLUMNS, ridge_lambda=fit.ridge_lambda).defense
            if not others.is_empty()
            else {}
        )
        own = rows.filter(
            (pl.col(QB_ID_COLUMN) == passer) & pl.col("opponent_team").is_in(list(defense))
        )
        if own.is_empty():
            values.append(None)
            continue
        weights = own.get_column(QB_DROPBACKS_COLUMN).cast(pl.Float64).to_numpy()
        effects = np.array(
            [defense[team] for team in own.get_column("opponent_team").to_list()], dtype=np.float64
        )
        values.append(float(np.average(effects, weights=weights)))
    return pl.DataFrame({QB_ID_COLUMN: passers, "qb_faced_pass_defense": values})


__all__ = [
    "QB_DROPBACKS_COLUMN",
    "QB_EPA_PER_DROPBACK_COLUMN",
    "QB_ID_COLUMN",
    "QB_UNIT_COLUMNS",
    "QbRatingFit",
    "QbRatingResampler",
    "bootstrap_qb_ratings",
    "compute_qb_faced_pass_defense",
    "fit_qb_ratings",
    "fit_qb_ratings_by_week",
    "qb_rating_rows",
]
