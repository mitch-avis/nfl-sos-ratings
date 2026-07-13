"""Schedule-adjusted team ratings built from equal stat weighting."""

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Stat pools: (column_name, True if higher value = better for the team)
#
# Include all plausibly relevant stats; correlation-threshold filtering removes
# anything that doesn't actually predict winning.  Don't pre-select — let the
# data decide.
# ---------------------------------------------------------------------------

_OFF_STAT_POOL: list[tuple[str, bool]] = [
    ("points_per_offensive_snap", True),
    ("total_yards_per_offensive_snap", True),
    ("passing_yards_per_offensive_snap", True),
    ("rushing_yards_per_offensive_snap", True),
    ("passing_epa_per_offensive_snap", True),
    ("rushing_epa_per_offensive_snap", True),
    ("passing_tds_per_offensive_snap", True),
    ("rushing_tds_per_offensive_snap", True),
    ("passing_first_downs_per_offensive_snap", True),
    ("rushing_first_downs_per_offensive_snap", True),
    ("passing_cpoe", True),  # completion % over expectation
    ("sacks_suffered_per_offensive_snap", False),
    ("passing_interceptions_per_offensive_snap", False),
    ("sack_fumbles_lost_per_offensive_snap", False),
    ("rushing_fumbles_lost_per_offensive_snap", False),
]

_DEF_STAT_POOL: list[tuple[str, bool]] = [
    ("points_allowed_per_defensive_snap", False),
    ("total_yards_allowed_per_defensive_snap", False),
    ("passing_yards_allowed_per_defensive_snap", False),
    ("rushing_yards_allowed_per_defensive_snap", False),
    ("passing_epa_allowed_per_defensive_snap", False),
    ("rushing_epa_allowed_per_defensive_snap", False),
    ("passing_tds_allowed_per_defensive_snap", False),
    ("rushing_tds_allowed_per_defensive_snap", False),
    ("passing_first_downs_allowed_per_defensive_snap", False),
    ("rushing_first_downs_allowed_per_defensive_snap", False),
    ("passing_cpoe_allowed", False),
    ("def_sacks_per_defensive_snap", True),
    ("def_interceptions_per_defensive_snap", True),
    ("def_pass_defended_per_defensive_snap", True),
    ("def_tackles_for_loss_per_defensive_snap", True),
    ("def_qb_hits_per_defensive_snap", True),
    ("def_fumbles_forced_per_defensive_snap", True),
    ("def_safeties_per_defensive_snap", True),
]

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
    """Build the raw overall signal from win rate and turnover margin."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    components: list[np.ndarray] = []

    win_values = _col(df, "win_pct")
    if win_values is None:
        win_values = _col(df, "win_value")
    if win_values is not None:
        reference_win_values = _col(resolved_reference_df, "win_pct")
        if reference_win_values is None:
            reference_win_values = _col(resolved_reference_df, "win_value")
        components.append(
            _zscore_against(
                win_values.tolist(),
                (reference_win_values if reference_win_values is not None else win_values).tolist(),
            )
        )

    turnover_margin = _col(df, "turnover_margin")
    if turnover_margin is None:
        turnover_margin = _derive_turnover_margin(df)
    if turnover_margin is not None:
        reference_turnover_margin = _col(resolved_reference_df, "turnover_margin")
        if reference_turnover_margin is None:
            reference_turnover_margin = _derive_turnover_margin(resolved_reference_df)
        components.append(
            _zscore_against(
                turnover_margin.tolist(),
                (
                    reference_turnover_margin
                    if reference_turnover_margin is not None
                    else turnover_margin
                ).tolist(),
            )
        )

    if not components:
        return np.zeros(df.height)
    return np.mean(np.column_stack(components), axis=1)


def _build_overall_sos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Build the overall schedule-strength signal from opponent overall fields."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    parts: list[np.ndarray] = []
    for column in ("opp_win_value", "opp_turnover_margin"):
        values = _col(df, column)
        if values is not None:
            reference_values = _col(resolved_reference_df, column)
            parts.append(
                _zscore_against(
                    values.tolist(),
                    (reference_values if reference_values is not None else values).tolist(),
                )
            )
    return np.mean(np.column_stack(parts), axis=1) if parts else np.zeros(df.height)


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
    """Compute SaOR, SaDR, SaOvR, and SaCR for every team in *df*."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    n = df.height
    teams = df.select("team").to_series().to_list()

    win_pct_arr = _col(df, "win_pct")
    if win_pct_arr is None:
        win_pct_arr = np.full(n, 0.5)

    off_weights = _derive_weights(df, _OFF_STAT_POOL, win_pct_arr, "Offensive")
    def_weights = _derive_weights(df, _DEF_STAT_POOL, win_pct_arr, "Defensive")

    sos_off = _build_off_sos(df, resolved_reference_df)
    sos_def = _build_def_sos(df, resolved_reference_df)
    sos_ovr = _build_overall_sos(df, resolved_reference_df)

    reference_sos_off = _build_off_sos(resolved_reference_df, resolved_reference_df)
    reference_sos_def = _build_def_sos(resolved_reference_df, resolved_reference_df)
    reference_sos_ovr = _build_overall_sos(resolved_reference_df, resolved_reference_df)

    raw_off = _build_composite(df, off_weights, resolved_reference_df)
    raw_def = _build_composite(df, def_weights, resolved_reference_df)
    raw_ovr = _build_overall_raw(df, resolved_reference_df)
    reference_raw_off = _build_composite(resolved_reference_df, off_weights, resolved_reference_df)
    reference_raw_def = _build_composite(resolved_reference_df, def_weights, resolved_reference_df)
    reference_raw_ovr = _build_overall_raw(resolved_reference_df, resolved_reference_df)
    adj_off = raw_off + SOS_WEIGHT * sos_off
    adj_def = raw_def + SOS_WEIGHT * sos_def
    adj_ovr = raw_ovr + SOS_WEIGHT * sos_ovr
    reference_adj_off = reference_raw_off + SOS_WEIGHT * reference_sos_off
    reference_adj_def = reference_raw_def + SOS_WEIGHT * reference_sos_def
    reference_adj_ovr = reference_raw_ovr + SOS_WEIGHT * reference_sos_ovr

    saor = _zscore_against(adj_off.tolist(), reference_adj_off.tolist())
    sadr = _zscore_against(adj_def.tolist(), reference_adj_def.tolist())
    saovr = _zscore_against(adj_ovr.tolist(), reference_adj_ovr.tolist())
    reference_saor = _zscore(reference_adj_off.tolist())
    reference_sadr = _zscore(reference_adj_def.tolist())
    reference_saovr = _zscore(reference_adj_ovr.tolist())
    sacr = _zscore(
        (
            ((saor + sadr) + OVERALL_COMPOSITE_WEIGHT * saovr) / (2.0 + OVERALL_COMPOSITE_WEIGHT)
        ).tolist()
    )
    sacr = _zscore_against(
        (
            ((saor + sadr) + OVERALL_COMPOSITE_WEIGHT * saovr) / (2.0 + OVERALL_COMPOSITE_WEIGHT)
        ).tolist(),
        (
            ((reference_saor + reference_sadr) + OVERALL_COMPOSITE_WEIGHT * reference_saovr)
            / (2.0 + OVERALL_COMPOSITE_WEIGHT)
        ).tolist(),
    )

    return pl.DataFrame(
        {
            "team": teams,
            "SaOR": np.round(saor, 3).tolist(),
            "SaDR": np.round(sadr, 3).tolist(),
            "SaOvR": np.round(saovr, 3).tolist(),
            "SaCR": np.round(sacr, 3).tolist(),
        }
    ).sort("team")
