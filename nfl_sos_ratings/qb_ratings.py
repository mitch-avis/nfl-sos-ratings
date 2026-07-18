"""Quarterback schedule-adjusted rating helpers."""

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.metrics import get_registry

_OUTCOME_WEIGHT: float = 0.0
_MIN_CORRELATION: float = 0.1

FloatArray = npt.NDArray[np.float64]

_QB_RIDGE_ADJUSTED_COLUMN = "adj_qb_epa_per_dropback"
_QB_RIDGE_SOS_COLUMN = "adj_def_qb_epa_per_dropback_faced"

# Pool membership lives in the metric registry (the single source of truth);
# higher-is-better derives from each metric's polarity.
_QB_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("qb_primary")

_QB_DIFF_STAT_POOL: list[tuple[str, bool]] = [
    (f"diff_{stat}", higher_is_better) for stat, higher_is_better in _QB_STAT_POOL
]

_QB_PAIRED_STAT_POOL: list[tuple[str, bool]] = get_registry().pool_stats("qb_paired")


def _zscore(values: list[float]) -> FloatArray:
    """Return a z-scored array using sample standard deviation."""
    arr: FloatArray = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return arr - arr.mean() if len(arr) == 1 else arr
    std = float(arr.std(ddof=1))
    centered = arr - arr.mean()
    return np.divide(centered, np.float64(std)) if std > 0 else centered


def _zscore_against(values: list[float], reference_values: list[float]) -> FloatArray:
    """Return a z-scored array using the mean and spread of *reference_values*."""
    arr: FloatArray = np.asarray(values, dtype=np.float64)
    reference: FloatArray = np.asarray(reference_values, dtype=np.float64)
    if len(reference) == 0:
        return arr
    std = float(reference.std(ddof=1)) if len(reference) > 1 else 0.0
    mean = float(reference.mean())
    centered = arr - mean
    return np.divide(centered, np.float64(std)) if std > 0 else centered


def _col(df: pl.DataFrame, name: str) -> FloatArray | None:
    """Return a float64 ndarray for a DataFrame column, or None when missing."""
    if name not in df.columns:
        return None
    values: FloatArray = np.asarray(
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


def _season_frames(df: pl.DataFrame) -> list[pl.DataFrame]:
    """Partition a QB rating frame by season when a season column is present."""
    if "season" not in df.columns or df.is_empty():
        return [df]
    return list(df.partition_by("season", maintain_order=True))


def _has_non_null_values(df: pl.DataFrame, column: str) -> bool:
    """Return whether a column exists and contains at least one non-null value."""
    if column not in df.columns:
        return False
    return bool(
        df.select(pl.col(column).cast(pl.Float64).fill_nan(None).is_not_null().any()).item()
    )


def _season_supports_cpoe(df: pl.DataFrame) -> bool:
    """Return whether this rating frame belongs to the CPOE era."""
    if "season" not in df.columns or df.is_empty():
        return True
    season_values = df.select("season").to_series().drop_nulls().unique().to_list()
    return all(
        int(season) >= composite_weights.QB_COMPOSITE_START_SEASON for season in season_values
    )


def _safe_corr(x: FloatArray, y: FloatArray) -> float:
    """Return Pearson correlation, or 0.0 when undefined/unstable."""
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    x_std = float(x.std(ddof=1))
    y_std = float(y.std(ddof=1))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _percentile(values: FloatArray) -> FloatArray:
    """Return percentile ranks from 0 to 100 with higher values ranking better."""
    if len(values) <= 1:
        return np.array([100.0] * len(values), dtype=np.float64)
    order = np.argsort(values)
    ranks: FloatArray = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    percentile_scale = np.float64(len(values) - 1)
    percentiles = np.multiply(np.divide(ranks, percentile_scale), np.float64(100.0))
    return np.round(percentiles, 1).astype(np.float64)


def _derive_qb_weights(
    df: pl.DataFrame,
    stat_pool: list[tuple[str, bool]] | None = None,
) -> list[tuple[str, float, bool]]:
    """Return equal weights for the present QB stat-pool columns."""
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
        weights = _derive_qb_weights(paired_df, stat_pool=paired_pool)
        return _build_qb_raw_composite(
            paired_df,
            weights,
            reference_paired_df if reference_paired_df is not None else paired_df,
        )

    weights = _derive_qb_weights(df)
    return _build_qb_raw_composite(df, weights, resolved_reference_df)


def _build_qsos(df: pl.DataFrame, reference_df: pl.DataFrame | None = None) -> np.ndarray:
    """Build QB schedule strength from the faced-defense ridge coefficient."""
    resolved_reference_df = _resolve_reference_df(df, reference_df)
    faced_defense_rating = _col(df, _QB_RIDGE_SOS_COLUMN)
    if faced_defense_rating is None:
        return np.zeros(df.height, dtype=np.float64)

    reference_faced_defense_rating = _col(resolved_reference_df, _QB_RIDGE_SOS_COLUMN)
    return _zscore_against(
        faced_defense_rating.tolist(),
        (
            reference_faced_defense_rating
            if reference_faced_defense_rating is not None
            else faced_defense_rating
        ).tolist(),
    )


def _build_qb_adjusted_epa(df: pl.DataFrame) -> np.ndarray:
    """Return the ridge-adjusted QB EPA surface used for published QB quality ratings."""
    adjusted_epa = _col(df, _QB_RIDGE_ADJUSTED_COLUMN)
    if adjusted_epa is None:
        return np.zeros(df.height, dtype=np.float64)
    return adjusted_epa


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
    return _MIN_CORRELATION, 0.0, _OUTCOME_WEIGHT


def compute_qb_ratings(
    df: pl.DataFrame,
    sos_weight: float = 0.0,
    outcome_weight: float = _OUTCOME_WEIGHT,
    reference_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute QB raw, adjusted, and percentile outputs for each team row."""
    del sos_weight, outcome_weight, reference_df
    df = _filter_rating_pool(df)
    frames: list[pl.DataFrame] = []

    for season_df in _season_frames(df):
        id_cols = [col for col in ("qb_id", "qb_name", "team") if col in season_df.columns]
        n_qbs = season_df.height

        raw_weights = _derive_qb_weights(
            season_df,
            stat_pool=_QB_STAT_POOL,
        )
        raw = _build_qb_raw_composite(season_df, raw_weights, season_df)
        qraw = _zscore(raw.tolist()) if n_qbs > 0 else np.zeros(0, dtype=np.float64)

        adjusted_epa = _build_qb_adjusted_epa(season_df)
        qsaor = _zscore(adjusted_epa.tolist())
        qsos = _build_qsos(season_df, season_df)
        qoutcome, has_outcome = _build_outcome_signal(season_df, season_df)
        qsacr_input = composite_weights.build_weighted_composite(
            season_df,
            composite_weights.QB_QSACR_FROZEN_SPEC,
            season_df,
        )
        qsacr = _zscore(qsacr_input.tolist())

        publish_cpoe_composites = _season_supports_cpoe(season_df)
        publish_qraw = publish_cpoe_composites and _has_non_null_values(
            season_df,
            "qb_completion_percentage_above_expectation",
        )
        publish_qsacr = publish_cpoe_composites and _has_non_null_values(
            season_df,
            "adj_qb_completion_percentage_above_expectation",
        )

        null_metric_values: list[float | None] = [None for _ in range(n_qbs)]
        qraw_values: list[float | None] = (
            [float(value) for value in np.round(qraw, 3).tolist()]
            if publish_qraw
            else null_metric_values.copy()
        )
        qsaor_values: list[float | None] = [float(value) for value in np.round(qsaor, 3).tolist()]
        qsos_values: list[float | None] = [float(value) for value in np.round(qsos, 3).tolist()]
        qsacr_values: list[float | None] = (
            [float(value) for value in np.round(qsacr, 3).tolist()]
            if publish_qsacr
            else null_metric_values.copy()
        )
        qraw_pct_values: list[float | None] = (
            [float(value) for value in _percentile(qraw).tolist()]
            if publish_qraw
            else null_metric_values.copy()
        )
        qsaor_pct_values: list[float | None] = [
            float(value) for value in _percentile(qsaor).tolist()
        ]
        qsos_pct_values: list[float | None] = [float(value) for value in _percentile(qsos).tolist()]
        qsacr_pct_values: list[float | None] = (
            [float(value) for value in _percentile(qsacr).tolist()]
            if publish_qsacr
            else null_metric_values.copy()
        )

        payload: dict[str, list[float | None] | list[str]] = {}
        for col in id_cols:
            payload[col] = season_df.select(col).to_series().cast(pl.String).to_list()

        payload["QRaw"] = qraw_values
        payload["QSaOR"] = qsaor_values
        payload["QSoS"] = qsos_values
        payload["QSaCR"] = qsacr_values
        payload["QRaw_pct"] = qraw_pct_values
        payload["QSaOR_pct"] = qsaor_pct_values
        payload["QSoS_pct"] = qsos_pct_values
        payload["QSaCR_pct"] = qsacr_pct_values
        if has_outcome:
            payload["QOutcome"] = np.round(qoutcome, 3).tolist()
            payload["QOutcome_pct"] = _percentile(qoutcome).tolist()

        frames.append(pl.DataFrame(payload))

    result = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    if result.is_empty():
        return result

    has_qsacr_values = bool(result.select(pl.col("QSaCR").drop_nulls().len()).item())
    sort_key = "QSaCR" if has_qsacr_values else ("qb_id" if "qb_id" in result.columns else None)
    if sort_key is None:
        return result
    return result.sort(sort_key, descending=(sort_key == "QSaCR"))
