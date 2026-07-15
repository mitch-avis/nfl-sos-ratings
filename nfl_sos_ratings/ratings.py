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


def _zscore_against(values: list[float], reference_values: list[float]) -> np.ndarray:
    """Z-score *values* against the mean and spread of *reference_values*."""
    arr = np.array(values, dtype=np.float64)
    reference = np.array(reference_values, dtype=np.float64)
    if len(reference) == 0:
        return arr
    std = float(reference.std(ddof=1)) if len(reference) > 1 else 0.0
    mean = float(reference.mean())
    return (arr - mean) / std if std > 0 else arr - mean


def _col(df: pl.DataFrame, name: str) -> np.ndarray | None:
    """Return a DataFrame column as float64 ndarray, or None if absent."""
    if name not in df.columns:
        return None
    return np.array(
        df.select(name).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
        dtype=np.float64,
    )


def _resolve_reference_df(df: pl.DataFrame, reference_df: pl.DataFrame | None) -> pl.DataFrame:
    """Return the reference frame used to standardize rating components."""
    if reference_df is None or reference_df.is_empty():
        return df
    return reference_df


def _derive_weights(
    df: pl.DataFrame,
    stat_pool: list[tuple[str, bool]],
    win_pct: np.ndarray,
    label: str,
) -> list[tuple[str, float, bool]]:
    """Return equal weights for every present stat in the pool."""
    del win_pct, label
    present = [
        (stat, higher_better) for stat, higher_better in stat_pool if _col(df, stat) is not None
    ]
    count = len(present)
    if count == 0:
        return []
    weight = 1.0 / count
    return [(stat, weight, higher_better) for stat, higher_better in present]


def _build_composite(
    df: pl.DataFrame,
    weights: list[tuple[str, float, bool]],
    reference_df: pl.DataFrame | None = None,
) -> np.ndarray:
    """Weighted average of signed z-scores for the given stat weights."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    composite = np.zeros(df.height)
    for stat, weight, higher_better in weights:
        v = _col(df, stat)
        if v is None:
            continue
        reference_values = _col(resolved_reference_df, stat)
        z = _zscore_against(
            v.tolist(), (reference_values if reference_values is not None else v).tolist()
        )
        composite += (z if higher_better else -z) * weight
    return composite


def _build_ridge_epa_composite(df: pl.DataFrame, component_cols: tuple[str, ...]) -> np.ndarray:
    """Average the present ridge-adjusted EPA components for one rating side."""
    present_components = [
        values for column in component_cols if (values := _col(df, column)) is not None
    ]
    if not present_components:
        return np.zeros(df.height, dtype=np.float64)
    return np.mean(np.column_stack(present_components), axis=1)


def _derive_turnover_margin(df: pl.DataFrame) -> np.ndarray | None:
    """Derive turnover margin from present takeaway and giveaway columns."""
    takeaways: list[np.ndarray] = []
    giveaways: list[np.ndarray] = []

    for column in ("def_interceptions", "def_fumbles_forced"):
        values = _col(df, column)
        if values is not None:
            takeaways.append(values)

    for column in ("passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost"):
        values = _col(df, column)
        if values is not None:
            giveaways.append(values)

    if not takeaways and not giveaways:
        return None

    takeaways_total = (
        np.sum(np.column_stack(takeaways), axis=1) if takeaways else np.zeros(df.height)
    )
    giveaways_total = (
        np.sum(np.column_stack(giveaways), axis=1) if giveaways else np.zeros(df.height)
    )
    return takeaways_total - giveaways_total


def _build_overall_raw(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Return a neutral placeholder until Stage 1 redefines team overall quality."""
    del reference_df
    return np.zeros(df.height, dtype=np.float64)


def _build_overall_sos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Return a neutral placeholder until Stage 1 redefines team overall schedule context."""
    del reference_df
    return np.zeros(df.height, dtype=np.float64)


def _build_off_sos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Build the offensive schedule-strength adjustment from opponent defense context."""
    opp_pa = _col(df, "opp_points_allowed")
    if opp_pa is None:
        return np.zeros(df.height)

    resolved_reference_df = _resolve_reference_df(df, reference_df)
    reference_opp_pa = _col(resolved_reference_df, "opp_points_allowed")
    return -_zscore_against(
        opp_pa.tolist(),
        (reference_opp_pa if reference_opp_pa is not None else opp_pa).tolist(),
    )


def _build_def_sos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Build the defensive schedule-strength adjustment from opponent offense context."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    sos_def_parts: list[np.ndarray] = []
    for col_name in ("opp_points_for", "opp_passing_epa"):
        values = _col(df, col_name)
        if values is None:
            continue
        reference_values = _col(resolved_reference_df, col_name)
        sos_def_parts.append(
            _zscore_against(
                values.tolist(),
                (reference_values if reference_values is not None else values).tolist(),
            )
        )
    return np.mean(sos_def_parts, axis=0) if sos_def_parts else np.zeros(df.height)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ratings(
    df: pl.DataFrame,
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute team ratings from ridge-adjusted passing and rushing EPA components."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    teams = df.select("team").to_series().to_list()

    raw_off = _build_ridge_epa_composite(df, _RIDGE_OFFENSE_COMPONENTS)
    raw_def = _build_ridge_epa_composite(df, _RIDGE_DEFENSE_COMPONENTS)
    reference_raw_off = _build_ridge_epa_composite(resolved_reference_df, _RIDGE_OFFENSE_COMPONENTS)
    reference_raw_def = _build_ridge_epa_composite(resolved_reference_df, _RIDGE_DEFENSE_COMPONENTS)

    saor = _zscore_against(raw_off.tolist(), reference_raw_off.tolist())
    sadr = _zscore_against(raw_def.tolist(), reference_raw_def.tolist())
    reference_saor = _zscore(reference_raw_off.tolist())
    reference_sadr = _zscore(reference_raw_def.tolist())

    saovr_input = saor + sadr
    reference_saovr_input = reference_saor + reference_sadr
    saovr = _zscore_against(saovr_input.tolist(), reference_saovr_input.tolist())

    sacr_input = composite_weights.build_weighted_composite(
        df,
        composite_weights.TEAM_SACR_FROZEN_SPEC,
        resolved_reference_df,
    )
    reference_sacr_input = composite_weights.build_weighted_composite(
        resolved_reference_df,
        composite_weights.TEAM_SACR_FROZEN_SPEC,
        resolved_reference_df,
    )
    sacr = _zscore_against(sacr_input.tolist(), reference_sacr_input.tolist())

    return pl.DataFrame(
        {
            "team": teams,
            "SaOR": np.round(saor, 3).tolist(),
            "SaDR": np.round(sadr, 3).tolist(),
            "SaOvR": np.round(saovr, 3).tolist(),
            "SaCR": np.round(sacr, 3).tolist(),
        }
    ).sort("team")
