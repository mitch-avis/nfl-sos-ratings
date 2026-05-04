"""Quarterback schedule-adjusted rating helpers."""

import numpy as np
import polars as pl

_SOS_WEIGHT: float = 2.0
_OUTCOME_WEIGHT: float = 0.75
_MIN_CORRELATION: float = 0.1

_QB_STAT_POOL: list[tuple[str, bool]] = [
    ("qb_passer_rating", True),
    ("qb_completion_percentage_above_expectation", True),
    ("qb_yards_per_attempt", True),
    ("qb_touchdown_rate", True),
    ("qb_interception_rate", False),
]

_QB_DIFF_STAT_POOL: list[tuple[str, bool]] = [
    ("diff_qb_passer_rating", True),
    ("diff_qb_completion_percentage_above_expectation", True),
    ("diff_qb_yards_per_attempt", True),
    ("diff_qb_touchdown_rate", True),
    ("diff_qb_interception_rate", False),
]

_QB_PAIRED_STAT_POOL: list[tuple[str, bool]] = [
    ("qb_passer_rating", True),
    ("qb_completion_percentage_above_expectation", True),
    ("qb_yards_per_attempt", True),
    ("qb_touchdown_rate", True),
    ("qb_interception_rate", False),
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
    return np.array(
        df.select(name).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
        dtype=np.float64,
    )


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
    """Derive QB stat weights from correlations with team win percentage."""
    selected_pool = stat_pool or (
        _QB_DIFF_STAT_POOL
        if any(_col(df, stat) is not None for stat, _ in _QB_DIFF_STAT_POOL)
        else _QB_STAT_POOL
    )
    if stat_pool is not None and not any(_col(df, stat) is not None for stat, _ in selected_pool):
        selected_pool = _QB_STAT_POOL

    win_pct = _col(df, "qb_win_pct")
    if win_pct is None:
        win_pct = _col(df, "win_pct")
    if win_pct is None:
        present = [(stat, higher) for stat, higher in selected_pool if _col(df, stat) is not None]
        if not present:
            return []
        weight = 1.0 / len(present)
        return [(stat, weight, higher) for stat, higher in present]

    weighted: list[tuple[str, float, bool]] = []
    for stat, higher_is_better in selected_pool:
        values = _col(df, stat)
        if values is None:
            continue
        oriented = values if higher_is_better else -values
        corr = _safe_corr(oriented, win_pct)
        if corr >= min_correlation:
            weighted.append((stat, corr, higher_is_better))

    if not weighted:
        present = [(stat, higher) for stat, higher in selected_pool if _col(df, stat) is not None]
        if not present:
            return []
        weight = 1.0 / len(present)
        return [(stat, weight, higher) for stat, higher in present]

    total = sum(weight for _, weight, _ in weighted)
    return [(stat, weight / total, higher) for stat, weight, higher in weighted]


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
        "qopp_qb_yards_per_attempt",
        "qopp_qb_touchdown_rate",
        "qopp_qb_passer_rating",
        "qopp_qb_completion_percentage_above_expectation",
    ):
        values = _col(df, col_name)
        if values is not None:
            sos_parts.append(-_zscore(values.tolist()))

    qopp_interceptions = _col(df, "qopp_qb_interception_rate")
    if qopp_interceptions is not None:
        sos_parts.append(_zscore(qopp_interceptions.tolist()))

    return np.mean(np.column_stack(sos_parts), axis=1) if sos_parts else np.zeros(n_teams)


def _build_outcome_signal(df: pl.DataFrame) -> tuple[np.ndarray, bool]:
    """Build the QB outcome signal used only by the final composite."""
    outcome = _col(df, "qb_win_pct")
    if outcome is None:
        outcome = _col(df, "win_pct")
    if outcome is None:
        return np.zeros(df.height, dtype=np.float64), False
    return _zscore(outcome.tolist()), True


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
    """Calibrate QB model constants against historical quarterback outcomes."""
    if historical_df.is_empty():
        return _MIN_CORRELATION, _SOS_WEIGHT, _OUTCOME_WEIGHT

    target = _col(historical_df, "qb_win_pct")
    if target is None:
        target = _col(historical_df, "win_pct")
    if target is None:
        return _MIN_CORRELATION, _SOS_WEIGHT, _OUTCOME_WEIGHT

    corr_candidates = correlation_grid or [0.05, 0.1, 0.15, 0.2]
    sos_candidates = sos_weight_grid or [2.0, 2.5, 3.0, 3.5, 4.0]
    outcome_candidates = outcome_weight_grid or [0.0, 0.25, 0.5, 0.75]

    best_score = float("-inf")
    best_tuple = (_MIN_CORRELATION, _SOS_WEIGHT, _OUTCOME_WEIGHT)

    for min_corr in corr_candidates:
        adjusted_base = _build_qb_adjusted_composite(historical_df, min_correlation=min_corr)
        qsos = _build_qsos(historical_df)
        outcome, _ = _build_outcome_signal(historical_df)
        for sos_weight in sos_candidates:
            for outcome_weight in outcome_candidates:
                qsacr = _zscore(
                    (adjusted_base + sos_weight * qsos + outcome_weight * outcome).tolist()
                )
                score = _safe_corr(qsacr, target)
                if score > best_score:
                    best_score = score
                    best_tuple = (min_corr, sos_weight, outcome_weight)

    return best_tuple


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
    qsaor = _zscore((adjusted_base + sos_weight * qsos).tolist())
    qsacr = _zscore((adjusted_base + sos_weight * qsos + outcome_weight * qoutcome).tolist())

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
