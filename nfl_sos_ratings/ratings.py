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


def _col(df: pl.DataFrame, name: str) -> np.ndarray | None:
    """Return a DataFrame column as float64 ndarray, or None if absent."""
    if name not in df.columns:
        return None
    return np.array(
        df.select(name).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
        dtype=np.float64,
    )


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


def _build_composite(df: pl.DataFrame, weights: list[tuple[str, float, bool]]) -> np.ndarray:
    """Weighted average of signed z-scores for the given stat weights."""
    composite = np.zeros(df.height)
    for stat, weight, higher_better in weights:
        v = _col(df, stat)
        if v is None:
            continue
        z = _zscore(v.tolist())
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


def _build_overall_raw(df: pl.DataFrame) -> np.ndarray:
    """Build the raw overall signal from win rate and turnover margin."""
    components: list[np.ndarray] = []

    win_values = _col(df, "win_pct")
    if win_values is None:
        win_values = _col(df, "win_value")
    if win_values is not None:
        components.append(_zscore(win_values.tolist()))

    turnover_margin = _col(df, "turnover_margin")
    if turnover_margin is None:
        turnover_margin = _derive_turnover_margin(df)
    if turnover_margin is not None:
        components.append(_zscore(turnover_margin.tolist()))

    if not components:
        return np.zeros(df.height)
    return np.mean(np.column_stack(components), axis=1)


def _build_overall_sos(df: pl.DataFrame) -> np.ndarray:
    """Build the overall schedule-strength signal from opponent overall fields."""
    parts: list[np.ndarray] = []
    for column in ("opp_win_value", "opp_turnover_margin"):
        values = _col(df, column)
        if values is not None:
            parts.append(_zscore(values.tolist()))
    return np.mean(np.column_stack(parts), axis=1) if parts else np.zeros(df.height)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ratings(df: pl.DataFrame) -> pl.DataFrame:
    """Compute SaOR, SaDR, SaOvR, and SaCR for every team in *df*."""
    n = df.height
    teams = df.select("team").to_series().to_list()

    win_pct_arr = _col(df, "win_pct")
    if win_pct_arr is None:
        win_pct_arr = np.full(n, 0.5)

    off_weights = _derive_weights(df, _OFF_STAT_POOL, win_pct_arr, "Offensive")
    def_weights = _derive_weights(df, _DEF_STAT_POOL, win_pct_arr, "Defensive")

    sos_off = np.zeros(n)
    opp_pa = _col(df, "opp_points_allowed")
    if opp_pa is not None:
        sos_off = -_zscore(opp_pa.tolist())

    sos_def_parts: list[np.ndarray] = []
    for col_name in ("opp_points_for", "opp_passing_epa"):
        v = _col(df, col_name)
        if v is not None:
            sos_def_parts.append(_zscore(v.tolist()))
    sos_def = np.mean(sos_def_parts, axis=0) if sos_def_parts else np.zeros(n)
    sos_ovr = _build_overall_sos(df)

    raw_off = _build_composite(df, off_weights)
    raw_def = _build_composite(df, def_weights)
    raw_ovr = _build_overall_raw(df)
    adj_off = raw_off + SOS_WEIGHT * sos_off
    adj_def = raw_def + SOS_WEIGHT * sos_def
    adj_ovr = raw_ovr + SOS_WEIGHT * sos_ovr

    saor = _zscore(adj_off.tolist())
    sadr = _zscore(adj_def.tolist())
    saovr = _zscore(adj_ovr.tolist())
    sacr = _zscore(
        (
            ((saor + sadr) + OVERALL_COMPOSITE_WEIGHT * saovr) / (2.0 + OVERALL_COMPOSITE_WEIGHT)
        ).tolist()
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
