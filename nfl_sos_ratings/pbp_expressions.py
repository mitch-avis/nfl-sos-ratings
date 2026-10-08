"""Shared Polars expressions used by the team and QB stat builders.

These definitions are correctness invariants: the scrimmage-snap definition
here is the single definition of an offensive snap for the whole project, and
the passer-rating formula the single definition of passer rating.
"""

import polars as pl


def scrimmage_snap_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that flags offensive scrimmage snaps in PBP data.

    A scrimmage snap is a dropback, rush, kneel, or spike (the established
    pipeline definition).
    """

    def _flag(column: str) -> pl.Expr:
        if column in columns:
            return pl.col(column).fill_null(0).cast(pl.Int8)
        return pl.lit(0)

    return (_flag("qb_dropback") + _flag("rush") + _flag("qb_kneel") + _flag("qb_spike")) > 0


def special_teams_play_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that flags special-teams plays in PBP data.

    nflverse marks kickoffs, punts, field goals, and extra points with ``special`` (older files
    name it ``special_teams_play``); a file with neither flags no play.
    """
    flag = "special" if "special" in columns else "special_teams_play"
    return value_expr(columns, flag) > 0


def value_expr(columns: list[str], column: str, default: float = 0) -> pl.Expr:
    """Return a null-safe column expression or a literal default when absent."""
    if column in columns:
        return pl.col(column).fill_null(default)
    return pl.lit(default)


def passer_rating_from_rates(
    completion_rate: pl.Expr,
    yards_per_attempt: pl.Expr,
    touchdown_rate: pl.Expr,
    interception_rate: pl.Expr,
) -> pl.Expr:
    """Return the official NFL passer rating from its four per-attempt rates.

    Null when any rate is null (no attempts).
    """

    def _clamp(component: pl.Expr) -> pl.Expr:
        return component.clip(0.0, 2.375)

    a = _clamp((completion_rate - 0.3) * 5.0)
    b = _clamp((yards_per_attempt - 3.0) * 0.25)
    c = _clamp(touchdown_rate * 20.0)
    d = _clamp(2.375 - (interception_rate * 25.0))
    return (a + b + c + d) / 6.0 * 100.0
