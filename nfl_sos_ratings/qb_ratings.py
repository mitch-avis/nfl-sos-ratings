"""Quarterback schedule-adjusted rating helpers."""

import numpy as np
import polars as pl

from nfl_sos_ratings.metrics import get_registry

_SOS_WEIGHT: float = 0.0
_OUTCOME_WEIGHT: float = 0.75
_MIN_CORRELATION: float = 0.1

# Pool membership lives in the metric registry (the single source of truth);
# higher-is-better derives from each metric's polarity.
_QB_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("qb_primary")

_QB_DIFF_STAT_POOL: list[tuple[str, bool]] = [
    (f"diff_{stat}", higher_is_better) for stat, higher_is_better in _QB_STAT_POOL
]

_QB_PAIRED_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("qb_paired")


def _zscore(values: list[float]) -> np.ndarray:
    """Return a z-scored array using sample standard deviation."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return arr - arr.mean() if len(arr) == 1 else arr
    std = float(arr.std(ddof=1))
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def _zscore_against(values: list[float], reference_values: list[float]) -> np.ndarray:
    """Return a z-scored array using the mean and spread of *reference_values*."""
    arr = np.array(values, dtype=np.float64)
    reference = np.array(reference_values, dtype=np.float64)
    if len(reference) == 0:
        return arr
    std = float(reference.std(ddof=1)) if len(reference) > 1 else 0.0
    mean = float(reference.mean())
    return (arr - mean) / std if std > 0 else arr - mean


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


def _resolve_reference_df(df: pl.DataFrame, reference_df: pl.DataFrame | None) -> pl.DataFrame:
    """Return the reference frame used to standardize QB rating components."""
    if reference_df is None or reference_df.is_empty():
        return df
    return reference_df


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


def _build_qb_raw_composite(
    df: pl.DataFrame,
    weights: list[tuple[str, float, bool]],
    reference_df: pl.DataFrame | None = None,
) -> np.ndarray:
    """Build a weighted raw QB composite from oriented z-scored stat columns."""
    if not weights:
        return np.zeros(df.height, dtype=np.float64)

    resolved_reference_df = _resolve_reference_df(df, reference_df)
    composite = np.zeros(df.height, dtype=np.float64)
    for stat, weight, higher_is_better in weights:
        values = _col(df, stat)
        if values is None:
            continue
        reference_values = _col(resolved_reference_df, stat)
        zscore = _zscore_against(
            values.tolist(),
            (reference_values if reference_values is not None else values).tolist(),
        )
        composite += (zscore if higher_is_better else -zscore) * weight
    return composite


def _build_paired_adjusted_frame(
    df: pl.DataFrame,
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame | None:
    """Return standardized paired QB-vs-opponent columns when matched context exists."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    payload: dict[str, list[float]] = {}
    for stat, higher_is_better in _QB_PAIRED_STAT_POOL:
        qopp_stat = f"qopp_{stat}"
        values = _col(df, stat)
        qopp_values = _col(df, qopp_stat)
        if values is None or qopp_values is None:
            continue

        reference_values = _col(resolved_reference_df, stat)
        reference_qopp_values = _col(resolved_reference_df, qopp_stat)
        value_zscore = _zscore_against(
            values.tolist(),
            (reference_values if reference_values is not None else values).tolist(),
        )
        qopp_zscore = _zscore_against(
            qopp_values.tolist(),
            (reference_qopp_values if reference_qopp_values is not None else qopp_values).tolist(),
        )
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
    reference_df: pl.DataFrame | None = None,
) -> np.ndarray:
    """Build the schedule-adjusted QB base from paired context or fallback differentials."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    paired_df = _build_paired_adjusted_frame(df, resolved_reference_df)
    if paired_df is not None:
        reference_paired_df = _build_paired_adjusted_frame(
            resolved_reference_df,
            resolved_reference_df,
        )
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
        return _build_qb_raw_composite(
            paired_df,
            weights,
            reference_paired_df if reference_paired_df is not None else paired_df,
        )

    weights = _derive_qb_weights(df, min_correlation=min_correlation)
    return _build_qb_raw_composite(df, weights, resolved_reference_df)


def _build_qsos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Build QB schedule strength signal from available qopp_* columns."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    n_teams = df.height
    sos_parts: list[np.ndarray] = []
    qopp_pa = _col(df, "qopp_points_allowed")
    if qopp_pa is not None:
        reference_qopp_pa = _col(resolved_reference_df, "qopp_points_allowed")
        sos_parts.append(
            -_zscore_against(
                qopp_pa.tolist(),
                (reference_qopp_pa if reference_qopp_pa is not None else qopp_pa).tolist(),
            )
        )

    for col_name in ("qopp_def_sacks", "qopp_def_interceptions"):
        values = _col(df, col_name)
        if values is not None:
            reference_values = _col(resolved_reference_df, col_name)
            sos_parts.append(
                _zscore_against(
                    values.tolist(),
                    (reference_values if reference_values is not None else values).tolist(),
                )
            )

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
            reference_values = _col(resolved_reference_df, col_name)
            sos_parts.append(
                -_zscore_against(
                    values.tolist(),
                    (reference_values if reference_values is not None else values).tolist(),
                )
            )

    qopp_sack_rate = _col(df, "qopp_qb_sack_rate")
    if qopp_sack_rate is not None:
        reference_qopp_sack_rate = _col(resolved_reference_df, "qopp_qb_sack_rate")
        sos_parts.append(
            _zscore_against(
                qopp_sack_rate.tolist(),
                (
                    reference_qopp_sack_rate
                    if reference_qopp_sack_rate is not None
                    else qopp_sack_rate
                ).tolist(),
            )
        )

    return np.mean(np.column_stack(sos_parts), axis=1) if sos_parts else np.zeros(n_teams)


def _build_outcome_signal(
    df: pl.DataFrame,
    reference_df: pl.DataFrame | None = None,
) -> tuple[np.ndarray, bool]:
    """Build the QB outcome signal used only by the final composite."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    outcome_parts: list[np.ndarray] = []

    primary_win_column = None
    for column in ("qb_win_pct", "qb_wins"):
        values = _col(df, column)
        if values is not None:
            primary_win_column = column
            reference_values = _col(resolved_reference_df, column)
            outcome_parts.append(
                _zscore_against(
                    values.tolist(),
                    (reference_values if reference_values is not None else values).tolist(),
                )
            )
            break

    del primary_win_column

    for column in (
        "qb_fourth_quarter_comebacks",
        "qb_game_winning_drives",
    ):
        values = _col(df, column)
        if values is not None:
            reference_values = _col(resolved_reference_df, column)
            outcome_parts.append(
                _zscore_against(
                    values.tolist(),
                    (reference_values if reference_values is not None else values).tolist(),
                )
            )

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
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute QB raw, adjusted, and percentile outputs for each team row."""
    df = _filter_rating_pool(df)
    resolved_reference_df = _filter_rating_pool(_resolve_reference_df(df, reference_df))
    id_cols = [col for col in ("qb_id", "qb_name", "team") if col in df.columns]
    n_teams = df.height

    raw_weights = _derive_qb_weights(
        df,
        min_correlation=min_correlation,
        stat_pool=_QB_STAT_POOL,
    )
    raw = _build_qb_raw_composite(df, raw_weights, resolved_reference_df)
    reference_raw = _build_qb_raw_composite(
        resolved_reference_df,
        raw_weights,
        resolved_reference_df,
    )
    adjusted_base = _build_qb_adjusted_composite(
        df,
        min_correlation=min_correlation,
        reference_df=resolved_reference_df,
    )
    reference_adjusted_base = _build_qb_adjusted_composite(
        resolved_reference_df,
        min_correlation=min_correlation,
        reference_df=resolved_reference_df,
    )
    qraw = (
        _zscore_against(raw.tolist(), reference_raw.tolist())
        if n_teams > 0
        else np.zeros(0, dtype=np.float64)
    )

    qsos = _build_qsos(df, resolved_reference_df)
    reference_qsos = _build_qsos(resolved_reference_df, resolved_reference_df)
    qoutcome, has_outcome = _build_outcome_signal(df, resolved_reference_df)
    reference_qoutcome, _ = _build_outcome_signal(resolved_reference_df, resolved_reference_df)
    reliability = _reliability_weights(df)
    reference_reliability = _reliability_weights(resolved_reference_df)
    qsaor = _zscore_against(
        (reliability * (adjusted_base + sos_weight * qsos)).tolist(),
        (reference_reliability * (reference_adjusted_base + sos_weight * reference_qsos)).tolist(),
    )
    qsacr = _zscore_against(
        (reliability * (adjusted_base + sos_weight * qsos + outcome_weight * qoutcome)).tolist(),
        (
            reference_reliability
            * (
                reference_adjusted_base
                + sos_weight * reference_qsos
                + outcome_weight * reference_qoutcome
            )
        ).tolist(),
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
