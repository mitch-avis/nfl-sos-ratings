"""Shared play-by-play Polars expressions used by the team stat builders.

These definitions are correctness invariants: the scrimmage-snap definition
here is the single definition of an offensive snap for the whole project.
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


def value_expr(columns: list[str], column: str, default: int | float = 0) -> pl.Expr:
    """Return a null-safe column expression or a literal default when absent."""
    if column in columns:
        return pl.col(column).fill_null(default)
    return pl.lit(default)


def rate_expr(numerator: str, denominator: str, output: str) -> pl.Expr:
    """Return a null-safe rate expression for a numerator and denominator pair."""
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(output)
    )
