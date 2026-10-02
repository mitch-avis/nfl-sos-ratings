"""Schedule-adjusted team ratings built from equal stat weighting."""

import numpy as np
import polars as pl

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.metrics import get_registry

# ---------------------------------------------------------------------------
# Stat pools: (column_name, True if higher value = better for the team).
#
# Membership lives in the metric registry (nfl_sos_ratings/metrics/catalog.py),
# the single source of truth; higher-is-better derives from each metric's
# polarity.
# ---------------------------------------------------------------------------

_OFF_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("team_offense")

_DEF_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("team_defense")

_RIDGE_OFFENSE_COMPONENTS: tuple[str, str] = (
    "adj_off_passing_epa_per_offensive_snap",
    "adj_off_rushing_epa_per_offensive_snap",
)

_RIDGE_DEFENSE_COMPONENTS: tuple[str, str] = (
    "adj_def_passing_epa_per_offensive_snap",
    "adj_def_rushing_epa_per_offensive_snap",
)

# How strongly schedule difficulty shifts the raw composite.
# 0 = ignore schedule; 1 = equal weight to raw performance.
SOS_WEIGHT: float = 0.25
OVERALL_COMPOSITE_WEIGHT: float = 0.25


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore(values: list[float]) -> np.ndarray:
    """Z-score using sample standard deviation (ddof=1)."""
    arr = np.array(values, dtype=np.float64)
    std = float(arr.std(ddof=1))
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def _col(df: pl.DataFrame, name: str) -> np.ndarray | None:
    """Return a DataFrame column as float64 ndarray, or None if absent."""
    if name not in df.columns:
        return None
    return np.array(
        df.select(name).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
        dtype=np.float64,
    )


def _season_frames(df: pl.DataFrame) -> list[pl.DataFrame]:
    """Partition a rating frame by season when a season column is present."""
    if "season" not in df.columns or df.is_empty():
        return [df]
    return list(df.partition_by("season", maintain_order=True))


def _build_ridge_epa_composite(df: pl.DataFrame, component_cols: tuple[str, ...]) -> np.ndarray:
    """Average the present ridge-adjusted EPA components for one rating side."""
    present_components = [
        values for column in component_cols if (values := _col(df, column)) is not None
    ]
    if not present_components:
        return np.zeros(df.height, dtype=np.float64)
    return np.mean(np.column_stack(present_components), axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ratings(
    df: pl.DataFrame,
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute team ratings from ridge-backed offense, defense, and special-teams components."""
    del reference_df
    frames: list[pl.DataFrame] = []

    for season_df in _season_frames(df):
        teams = season_df.select("team").to_series().to_list()

        raw_off = _build_ridge_epa_composite(season_df, _RIDGE_OFFENSE_COMPONENTS)
        raw_def = _build_ridge_epa_composite(season_df, _RIDGE_DEFENSE_COMPONENTS)
        raw_st = _col(season_df, "st_rating")
        existing_sastr = _col(season_df, "SaSTR")

        saor = _zscore(raw_off.tolist())
        sadr = _zscore(raw_def.tolist())

        if raw_st is not None:
            sastr = _zscore(raw_st.tolist())
        elif existing_sastr is not None:
            sastr = np.array(existing_sastr, dtype=np.float64)
        else:
            sastr = np.zeros(season_df.height, dtype=np.float64)

        saovr = _zscore((saor + sadr + sastr).tolist())

        composite_df = season_df
        if "st_rating" not in composite_df.columns:
            composite_df = composite_df.with_columns(pl.Series("st_rating", sastr.tolist()))

        sacr_input = composite_weights.build_weighted_composite(
            composite_df,
            composite_weights.TEAM_SACR_FROZEN_SPEC,
            composite_df,
        )
        sacr = _zscore(sacr_input.tolist())

        frames.append(
            pl.DataFrame(
                {
                    "team": teams,
                    "SaOR": np.round(saor, 3).tolist(),
                    "SaDR": np.round(sadr, 3).tolist(),
                    "SaSTR": np.round(sastr, 3).tolist(),
                    "SaOvR": np.round(saovr, 3).tolist(),
                    "SaCR": np.round(sacr, 3).tolist(),
                }
            )
        )

    return pl.concat(frames, how="vertical_relaxed").sort("team")
