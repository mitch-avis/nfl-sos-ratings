"""Partial-season rating snapshot helpers."""

from collections.abc import Sequence

import polars as pl

from nfl_sos_ratings.ratings import compute_ratings
from nfl_sos_ratings.simultaneous_adjustment import compute_team_adjusted_stats

_DEFAULT_TEAM_RESPONSE_COLS: tuple[str, str] = (
    "passing_epa_per_offensive_snap",
    "rushing_epa_per_offensive_snap",
)


def _empty_team_snapshot(teams: list[str]) -> pl.DataFrame:
    """Return a zeroed team snapshot when no games qualify before the cutoff."""
    return pl.DataFrame(
        {
            "team": teams,
            "SaOR": [0.0] * len(teams),
            "SaDR": [0.0] * len(teams),
            "SaOvR": [0.0] * len(teams),
            "SaCR": [0.0] * len(teams),
        }
    ).sort("team")


def build_team_rating_snapshot(
    weekly_team_rows: pl.DataFrame,
    cutoff_week: int,
    response_cols: Sequence[str] | None = None,
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a team rating snapshot using only games before ``cutoff_week``.

    Args:
        weekly_team_rows: Team-game rows with ``team``, ``opponent_team``, ``week``,
            ``is_home``, and the selected response columns.
        cutoff_week: Snapshot cutoff. Only rows with ``week < cutoff_week`` are used.
        response_cols: Optional override for the ridge response columns.
        reference_df: Optional reference frame for rating standardization.

    Returns:
        A ``compute_ratings``-compatible team ratings table for the pre-cutoff games.

    """
    if "week" not in weekly_team_rows.columns:
        raise ValueError("weekly_team_rows must include a week column")

    selected_response_cols = list(response_cols or _DEFAULT_TEAM_RESPONSE_COLS)
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        teams = sorted(
            weekly_team_rows.select("team")
            .drop_nulls()
            .to_series()
            .cast(pl.String)
            .unique()
            .to_list()
        )
        return _empty_team_snapshot(teams)

    adjusted = compute_team_adjusted_stats(
        filtered_rows,
        response_cols=selected_response_cols,
    )
    return compute_ratings(adjusted, reference_df=reference_df)


__all__ = ["build_team_rating_snapshot"]
