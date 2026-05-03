"""Quarterback schedule-adjusted rating helpers."""

import numpy as np
import polars as pl

_SOS_WEIGHT: float = 0.25
_MIN_CORRELATION: float = 0.1

_QB_STAT_POOL: list[tuple[str, bool]] = [
    ("qb_passer_rating", True),
    ("qb_completion_percentage_above_expectation", True),
    ("qb_aggressiveness", True),
    ("qb_avg_intended_air_yards", True),
    ("qb_avg_time_to_throw", False),
    ("qb_avg_air_yards_to_sticks", True),
]


def _zscore(values: list[float]) -> np.ndarray:
    """Return a z-scored array using sample standard deviation."""
    arr = np.array(values, dtype=np.float64)
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
) -> list[tuple[str, float, bool]]:
    """Derive QB stat weights from correlations with team win percentage."""
    win_pct = _col(df, "qb_win_pct")
    if win_pct is None:
        win_pct = _col(df, "win_pct")
    if win_pct is None:
        present = [(stat, higher) for stat, higher in _QB_STAT_POOL if _col(df, stat) is not None]
        if not present:
            return []
        weight = 1.0 / len(present)
        return [(stat, weight, higher) for stat, higher in present]

    weighted: list[tuple[str, float, bool]] = []
    for stat, higher_is_better in _QB_STAT_POOL:
        values = _col(df, stat)
        if values is None:
            continue
        oriented = values if higher_is_better else -values
        corr = _safe_corr(oriented, win_pct)
        if corr >= min_correlation:
            weighted.append((stat, corr, higher_is_better))

    if not weighted:
        present = [(stat, higher) for stat, higher in _QB_STAT_POOL if _col(df, stat) is not None]
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
        "qopp_qb_passer_rating",
        "qopp_qb_completion_percentage_above_expectation",
        "qopp_qb_aggressiveness",
    ):
        values = _col(df, col_name)
        if values is not None:
            sos_parts.append(-_zscore(values.tolist()))

    return np.mean(np.column_stack(sos_parts), axis=1) if sos_parts else np.zeros(n_teams)


def calibrate_qb_model(
    historical_df: pl.DataFrame,
    correlation_grid: list[float] | None = None,
    sos_weight_grid: list[float] | None = None,
) -> tuple[float, float]:
    """Calibrate QB model constants against historical quarterback outcomes."""
    if historical_df.is_empty():
        return _MIN_CORRELATION, _SOS_WEIGHT

    target = _col(historical_df, "qb_win_pct")
    if target is None:
        target = _col(historical_df, "win_pct")
    if target is None:
        return _MIN_CORRELATION, _SOS_WEIGHT

    corr_candidates = correlation_grid or [0.05, 0.1, 0.15, 0.2]
    sos_candidates = sos_weight_grid or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

    best_score = float("-inf")
    best_pair = (_MIN_CORRELATION, _SOS_WEIGHT)

    for min_corr in corr_candidates:
        weights = _derive_qb_weights(historical_df, min_correlation=min_corr)
        raw = _build_qb_raw_composite(historical_df, weights)
        qsos = _build_qsos(historical_df)
        for sos_weight in sos_candidates:
            qsacr = _zscore((raw + sos_weight * qsos).tolist())
            score = _safe_corr(qsacr, target)
            if score > best_score:
                best_score = score
                best_pair = (min_corr, sos_weight)

    return best_pair


def compute_qb_ratings(
    df: pl.DataFrame,
    min_correlation: float = _MIN_CORRELATION,
    sos_weight: float = _SOS_WEIGHT,
) -> pl.DataFrame:
    """Compute QB raw, adjusted, and percentile outputs for each team row."""
    id_cols = [col for col in ("qb_id", "qb_name", "team") if col in df.columns]
    n_teams = df.height

    weights = _derive_qb_weights(df, min_correlation=min_correlation)
    raw = _build_qb_raw_composite(df, weights)
    qraw = _zscore(raw.tolist()) if n_teams > 0 else np.zeros(0, dtype=np.float64)

    qsos = _build_qsos(df)
    qsaor = _zscore((raw + sos_weight * qsos).tolist())
    qsacr = qsaor.copy()

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

    result = pl.DataFrame(payload)
    sort_key = "QSaCR" if "QSaCR" in result.columns else (id_cols[0] if id_cols else None)
    return result.sort(sort_key) if sort_key is not None else result
