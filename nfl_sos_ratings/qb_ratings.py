"""Quarterback schedule-adjusted rating helpers."""

import numpy as np
import polars as pl

_SOS_WEIGHT: float = 0.0
_OUTCOME_WEIGHT: float = 0.75
_MIN_CORRELATION: float = 0.1

_QB_STAT_POOL: list[tuple[str, bool]] = [
    ("qb_epa_per_dropback", True),
    ("qb_any_a", True),
    ("qb_completion_percentage_above_expectation", True),
    ("qb_td_int_margin_rate", True),
    ("qb_sack_rate", False),
    ("qb_pass_yards_per_dropback", True),
    ("qb_sacks", False),
    ("qb_passer_rating", True),
]

_QB_DIFF_STAT_POOL: list[tuple[str, bool]] = [
    ("diff_qb_epa_per_dropback", True),
    ("diff_qb_any_a", True),
    ("diff_qb_completion_percentage_above_expectation", True),
    ("diff_qb_td_int_margin_rate", True),
    ("diff_qb_sack_rate", False),
    ("diff_qb_pass_yards_per_dropback", True),
    ("diff_qb_sacks", False),
    ("diff_qb_passer_rating", True),
]

_QB_PAIRED_STAT_POOL: list[tuple[str, bool]] = [
    ("qb_epa_per_dropback", True),
    ("qb_any_a", True),
    ("qb_completion_percentage_above_expectation", True),
    ("qb_td_int_margin_rate", True),
    ("qb_sack_rate", False),
    ("qb_pass_yards_per_dropback", True),
    ("qb_passer_rating", True),
]


def _zscore(values: list[float]) -> np.ndarray:
    """Return a z-scored array using sample standard deviation."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return arr - arr.mean() if len(arr) == 1 else arr
    std = float(arr.std(ddof=1))
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def _col(df: pl.DataFrame, name: str) -> np.ndarray | None:
    """Return a float64 ndarray for a DataFrame column, or None when missing."""
    if name not in df.columns:
        return None
    values = np.array(
        df.select(pl.col(name).cast(pl.Float64).fill_nan(None).fill_null(0.0))
        .to_series()
        .to_list(),
        dtype=np.float64,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _reliability_weights(df: pl.DataFrame) -> np.ndarray:
    """Return 0-1 reliability weights based on games played share of a full season."""
    games = _col(df, "qb_games_played")
    if games is None:
        return np.ones(df.height, dtype=np.float64)

    full_season_games = float(np.max(games)) if len(games) > 0 else 0.0
    if full_season_games <= 0.0:
        return np.ones(df.height, dtype=np.float64)
    return np.clip(games / full_season_games, 0.0, 1.0)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or 0.0 when undefined/unstable."""
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    x_std = float(x.std(ddof=1))
    y_std = float(y.std(ddof=1))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _percentile(values: np.ndarray) -> np.ndarray:
    """Return percentile ranks from 0 to 100 with higher values ranking better."""
    if len(values) <= 1:
        return np.array([100.0] * len(values), dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return np.round((ranks / (len(values) - 1)) * 100.0, 1)


def _derive_qb_weights(
    df: pl.DataFrame,
    min_correlation: float = _MIN_CORRELATION,
    stat_pool: list[tuple[str, bool]] | None = None,
) -> list[tuple[str, float, bool]]:
    """Return equal weights for the present QB stat-pool columns."""
    del min_correlation
    selected_pool = stat_pool or (
        _QB_DIFF_STAT_POOL
        if any(_col(df, stat) is not None for stat, _ in _QB_DIFF_STAT_POOL)
        else _QB_STAT_POOL
    )
    if stat_pool is not None and not any(_col(df, stat) is not None for stat, _ in selected_pool):
        selected_pool = _QB_STAT_POOL

    present = [(stat, higher) for stat, higher in selected_pool if _col(df, stat) is not None]
    if not present:
        return []
    weight = 1.0 / len(present)
    return [(stat, weight, higher) for stat, higher in present]


def _build_qb_raw_composite(df: pl.DataFrame, weights: list[tuple[str, float, bool]]) -> np.ndarray:
    """Build a weighted raw QB composite from oriented z-scored stat columns."""
    if not weights:
        return np.zeros(df.height, dtype=np.float64)

    composite = np.zeros(df.height, dtype=np.float64)
    for stat, weight, higher_is_better in weights:
        values = _col(df, stat)
        if values is None:
            continue
        zscore = _zscore(values.tolist())
        composite += (zscore if higher_is_better else -zscore) * weight
    return composite


def _build_paired_adjusted_frame(df: pl.DataFrame) -> pl.DataFrame | None:
    """Return standardized paired QB-vs-opponent columns when matched context exists."""
    payload: dict[str, list[float]] = {}
    for stat, higher_is_better in _QB_PAIRED_STAT_POOL:
        qopp_stat = f"qopp_{stat}"
        values = _col(df, stat)
        qopp_values = _col(df, qopp_stat)
        if values is None or qopp_values is None:
            continue

        value_zscore = _zscore(values.tolist())
        qopp_zscore = _zscore(qopp_values.tolist())
        adjusted = value_zscore - qopp_zscore if higher_is_better else -value_zscore + qopp_zscore
        payload[f"adj_{stat}"] = adjusted.tolist()

    if not payload:
        return None

    for col_name in ("qb_win_pct", "win_pct"):
        values = _col(df, col_name)
        if values is not None:
            payload[col_name] = values.tolist()
            break
    return pl.DataFrame(payload)


def _build_qb_adjusted_composite(
    df: pl.DataFrame,
    min_correlation: float = _MIN_CORRELATION,
) -> np.ndarray:
    """Build the schedule-adjusted QB base from paired context or fallback differentials."""
    paired_df = _build_paired_adjusted_frame(df)
    if paired_df is not None:
        paired_pool = [
            (f"adj_{stat}", True)
            for stat, _ in _QB_PAIRED_STAT_POOL
            if f"adj_{stat}" in paired_df.columns
        ]
        weights = _derive_qb_weights(
            paired_df,
            min_correlation=min_correlation,
            stat_pool=paired_pool,
        )
        return _build_qb_raw_composite(paired_df, weights)

    weights = _derive_qb_weights(df, min_correlation=min_correlation)
    return _build_qb_raw_composite(df, weights)


def _build_qsos(df: pl.DataFrame) -> np.ndarray:
    """Build QB schedule strength signal from available qopp_* columns."""
    n_teams = df.height
    sos_parts: list[np.ndarray] = []
    qopp_pa = _col(df, "qopp_points_allowed")
    if qopp_pa is not None:
        sos_parts.append(-_zscore(qopp_pa.tolist()))

    for col_name in ("qopp_def_sacks", "qopp_def_interceptions"):
        values = _col(df, col_name)
        if values is not None:
            sos_parts.append(_zscore(values.tolist()))

    for col_name in (
        "qopp_qb_epa_per_dropback",
        "qopp_qb_any_a",
        "qopp_qb_passer_rating",
        "qopp_qb_completion_percentage_above_expectation",
        "qopp_qb_td_int_margin_rate",
        "qopp_qb_pass_yards_per_dropback",
    ):
        values = _col(df, col_name)
        if values is not None:
            sos_parts.append(-_zscore(values.tolist()))

    qopp_sack_rate = _col(df, "qopp_qb_sack_rate")
    if qopp_sack_rate is not None:
        sos_parts.append(_zscore(qopp_sack_rate.tolist()))

    return np.mean(np.column_stack(sos_parts), axis=1) if sos_parts else np.zeros(n_teams)


def _build_outcome_signal(df: pl.DataFrame) -> tuple[np.ndarray, bool]:
    """Build the QB outcome signal used only by the final composite."""
    outcome_parts: list[np.ndarray] = []

    primary_win_column = None
    for column in ("qb_win_pct", "qb_wins"):
        values = _col(df, column)
        if values is not None:
            primary_win_column = column
            outcome_parts.append(_zscore(values.tolist()))
            break

    del primary_win_column

    for column in (
        "qb_fourth_quarter_comebacks",
        "qb_game_winning_drives",
    ):
        values = _col(df, column)
        if values is not None:
            outcome_parts.append(_zscore(values.tolist()))

    if not outcome_parts:
        return np.zeros(df.height, dtype=np.float64), False

    outcome = np.mean(np.column_stack(outcome_parts), axis=1)
    return outcome, True


def _filter_rating_pool(df: pl.DataFrame) -> pl.DataFrame:
    """Return qualified QB rows when eligibility is available."""
    if "qb_is_eligible" in df.columns:
        return df.filter(pl.col("qb_is_eligible"))
    return df


def calibrate_qb_model(
    historical_df: pl.DataFrame,
    correlation_grid: list[float] | None = None,
    sos_weight_grid: list[float] | None = None,
    outcome_weight_grid: list[float] | None = None,
) -> tuple[float, float, float]:
    """Return fixed QB model constants; historical win-based calibration is disabled."""
    del historical_df, correlation_grid, sos_weight_grid, outcome_weight_grid
    return _MIN_CORRELATION, _SOS_WEIGHT, _OUTCOME_WEIGHT


def compute_qb_ratings(
    df: pl.DataFrame,
    min_correlation: float = _MIN_CORRELATION,
    sos_weight: float = _SOS_WEIGHT,
    outcome_weight: float = _OUTCOME_WEIGHT,
) -> pl.DataFrame:
    """Compute QB raw, adjusted, and percentile outputs for each team row."""
    df = _filter_rating_pool(df)
    id_cols = [col for col in ("qb_id", "qb_name", "team") if col in df.columns]
    n_teams = df.height

    raw_weights = _derive_qb_weights(
        df,
        min_correlation=min_correlation,
        stat_pool=_QB_STAT_POOL,
    )
    raw = _build_qb_raw_composite(df, raw_weights)
    adjusted_base = _build_qb_adjusted_composite(df, min_correlation=min_correlation)
    qraw = _zscore(raw.tolist()) if n_teams > 0 else np.zeros(0, dtype=np.float64)

    qsos = _build_qsos(df)
    qoutcome, has_outcome = _build_outcome_signal(df)
    reliability = _reliability_weights(df)
    qsaor = _zscore((reliability * (adjusted_base + sos_weight * qsos)).tolist())
    qsacr = _zscore(
        (reliability * (adjusted_base + sos_weight * qsos + outcome_weight * qoutcome)).tolist()
    )

    payload: dict[str, list[float] | list[str]] = {}
    for col in id_cols:
        payload[col] = df.select(col).to_series().cast(pl.String).to_list()

    payload.update(
        {
            "QRaw": np.round(qraw, 3).tolist(),
            "QSaOR": np.round(qsaor, 3).tolist(),
            "QSoS": np.round(qsos, 3).tolist(),
            "QSaCR": np.round(qsacr, 3).tolist(),
            "QRaw_pct": _percentile(qraw).tolist(),
            "QSaOR_pct": _percentile(qsaor).tolist(),
            "QSoS_pct": _percentile(qsos).tolist(),
            "QSaCR_pct": _percentile(qsacr).tolist(),
        }
    )
    if has_outcome:
        payload["QOutcome"] = np.round(qoutcome, 3).tolist()
        payload["QOutcome_pct"] = _percentile(qoutcome).tolist()

    result = pl.DataFrame(payload)
    sort_key = "QSaCR" if "QSaCR" in result.columns else (id_cols[0] if id_cols else None)
    return result.sort(sort_key) if sort_key is not None else result
