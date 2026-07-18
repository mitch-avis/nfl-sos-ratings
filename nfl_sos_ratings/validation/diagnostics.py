"""Validation diagnostics helpers for team and quarterback analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from nfl_sos_ratings.data_loader import (
    load_pbp_data,
    load_playoff_pbp_data,
    load_qb_identity_crosswalk,
)
from nfl_sos_ratings.qb_stats import compute_qb_game_stats_from_pbp
from nfl_sos_ratings.simultaneous_adjustment import (
    solve_qb_stat_ridge,
    solve_qb_stat_with_fixed_defense_offsets,
    solve_team_stat_ridge,
    tune_ridge_lambda,
)


def _ensure_error_column(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return predictions with an explicit residual column."""
    if "error" in predictions.columns:
        return predictions
    return predictions.with_columns(
        (pl.col("predicted_margin") - pl.col("home_margin")).alias("error")
    )


def _weighted_mean_frame(
    frame: pl.DataFrame,
    *,
    group_keys: Sequence[str],
    value_col: str,
    weight_col: str,
    alias: str,
) -> pl.DataFrame:
    """Return grouped weighted means with a simple-mean fallback for zero total weight."""
    weighted = (
        frame.with_columns(pl.col(weight_col).cast(pl.Float64).fill_null(0.0))
        .group_by(list(group_keys))
        .agg(
            (pl.col(value_col) * pl.col(weight_col)).sum().alias("_weighted_value"),
            pl.col(weight_col).sum().alias("_total_weight"),
            pl.col(value_col).mean().alias("_fallback_mean"),
        )
        .with_columns(
            pl.when(pl.col("_total_weight") > 0.0)
            .then(pl.col("_weighted_value") / pl.col("_total_weight"))
            .otherwise(pl.col("_fallback_mean"))
            .alias(alias)
        )
        .select([*group_keys, alias, pl.col("_total_weight").alias("total_dropbacks")])
    )
    return weighted


def _one_sided_binomial_tail(successes: int, trials: int) -> float:
    """Return ``P(X >= successes)`` for ``X ~ Binomial(trials, 0.5)``."""
    if trials <= 0:
        return 1.0
    numerator = sum(math.comb(trials, value) for value in range(successes, trials + 1))
    return float(numerator / (2**trials))


def _weighted_slope(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted least-squares slope for one predictor and response."""
    if x_values.size < 2 or y_values.size < 2 or weights.size < 2:
        return 0.0
    x_mean = float(np.average(x_values, weights=weights))
    y_mean = float(np.average(y_values, weights=weights))
    denominator = float(np.sum(weights * (x_values - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(weights * (x_values - x_mean) * (y_values - y_mean)))
    return numerator / denominator


def _weighted_correlation(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted Pearson correlation, or 0.0 when undefined."""
    if x_values.size < 2 or y_values.size < 2 or weights.size < 2:
        return 0.0
    x_mean = float(np.average(x_values, weights=weights))
    y_mean = float(np.average(y_values, weights=weights))
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    covariance = float(np.sum(weights * x_centered * y_centered))
    x_scale = float(np.sqrt(np.sum(weights * x_centered**2)))
    y_scale = float(np.sqrt(np.sum(weights * y_centered**2)))
    if np.isclose(x_scale, 0.0) or np.isclose(y_scale, 0.0):
        return 0.0
    return covariance / (x_scale * y_scale)


def _summarize_weighted_signal_by_season(
    frame: pl.DataFrame,
    *,
    predictor_col: str,
    value_col: str,
    weight_col: str,
    resamples: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Return pooled and per-season weighted slopes, correlations, and pooled CIs."""
    required_columns = {predictor_col, value_col, weight_col}
    if not required_columns.issubset(set(frame.columns)):
        return pl.DataFrame()

    filtered = frame.drop_nulls([predictor_col, value_col, weight_col]).filter(
        pl.col(weight_col).cast(pl.Float64) > 0.0
    )
    if filtered.is_empty():
        return pl.DataFrame()

    def summarize_one(
        summary_frame: pl.DataFrame,
        *,
        scope: str,
        season: int | None,
    ) -> dict[str, object]:
        x_values = np.asarray(
            summary_frame.select(predictor_col).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            summary_frame.select(value_col).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        weights = np.asarray(
            summary_frame.select(pl.col(weight_col).cast(pl.Float64)).to_series().to_list(),
            dtype=np.float64,
        )
        return {
            "scope": scope,
            "season": season,
            "rows": int(summary_frame.height),
            "total_dropbacks": float(weights.sum()),
            "slope": _weighted_slope(x_values, y_values, weights),
            "correlation": _weighted_correlation(x_values, y_values, weights),
            "ci_lower": None,
            "ci_upper": None,
            "direction_positive_count": None,
            "direction_total_count": None,
            "direction_p_value": None,
        }

    season_rows: list[dict[str, object]] = []
    if "season" in filtered.columns:
        for season_key, season_frame in filtered.group_by("season", maintain_order=True):
            season_value = season_key[0] if isinstance(season_key, tuple) else season_key
            season_rows.append(
                summarize_one(season_frame, scope="season", season=int(season_value))
            )

    pooled_row = summarize_one(filtered, scope="pooled", season=None)
    positive_slopes = sum(1 for row in season_rows if cast(float, row["slope"]) > 0.0)
    pooled_row["direction_positive_count"] = positive_slopes
    pooled_row["direction_total_count"] = len(season_rows)
    pooled_row["direction_p_value"] = _one_sided_binomial_tail(positive_slopes, len(season_rows))

    rng = np.random.default_rng(seed)
    bootstrap_slopes = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        sample = filtered[rng.integers(0, filtered.height, size=filtered.height).tolist()]
        x_values = np.asarray(
            sample.select(predictor_col).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            sample.select(value_col).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        weights = np.asarray(
            sample.select(pl.col(weight_col).cast(pl.Float64)).to_series().to_list(),
            dtype=np.float64,
        )
        bootstrap_slopes[sample_index] = _weighted_slope(x_values, y_values, weights)
    pooled_row["ci_lower"] = float(np.quantile(bootstrap_slopes, 0.025))
    pooled_row["ci_upper"] = float(np.quantile(bootstrap_slopes, 0.975))

    return pl.DataFrame([pooled_row, *season_rows])


def build_qb_split_half_frame(
    qb_games: pl.DataFrame,
    defense_ratings: pl.DataFrame,
    *,
    qb_meta: pl.DataFrame | None = None,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
) -> pl.DataFrame:
    """Return one QB-season row with split-half raw, adjusted, and residual diagnostics."""
    required_game_columns = {
        "qb_id",
        "qb_name",
        "team",
        "opponent_team",
        response_col,
        dropback_col,
    }
    if not required_game_columns.issubset(set(qb_games.columns)):
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "qb_id": pl.String,
                "qb_name": pl.String,
                "team": pl.String,
                "faced_difficulty": pl.Float64,
                "additive_prediction": pl.Float64,
                "total_dropbacks": pl.Float64,
                "vs_top_half_raw_epa_per_dropback": pl.Float64,
                "vs_top_half_adjusted_epa_per_dropback": pl.Float64,
                "vs_top_half_residual": pl.Float64,
                "vs_top_half_dropbacks": pl.Float64,
                "vs_bottom_half_raw_epa_per_dropback": pl.Float64,
                "vs_bottom_half_adjusted_epa_per_dropback": pl.Float64,
                "vs_bottom_half_residual": pl.Float64,
                "vs_bottom_half_dropbacks": pl.Float64,
            }
        )

    required_defense_columns = {"team", "defense_coefficient"}
    if not required_defense_columns.issubset(set(defense_ratings.columns)):
        return pl.DataFrame()

    top_half_count = (defense_ratings.height + 1) // 2
    defense_halves = (
        defense_ratings.select(
            pl.col("team").cast(pl.String).alias("opponent_team"),
            pl.col("defense_coefficient").cast(pl.Float64).alias("defense_coefficient"),
        )
        .sort("defense_coefficient", descending=True)
        .with_row_index("_rank")
        .with_columns((pl.col("_rank") < top_half_count).alias("is_top_half"))
        .drop("_rank")
    )

    joined = qb_games.join(defense_halves, on="opponent_team", how="left")
    if qb_meta is not None and not qb_meta.is_empty():
        join_keys = [
            column for column in ("season", "qb_id", "qb_name", "team") if column in qb_meta.columns
        ]
        join_keys = [column for column in join_keys if column in joined.columns]
        if join_keys:
            joined = joined.join(qb_meta, on=join_keys, how="left")
            if "qb_is_eligible" in joined.columns:
                joined = joined.filter(pl.col("qb_is_eligible"))
    joined = joined.drop_nulls(["defense_coefficient"]).with_columns(
        pl.col(dropback_col).cast(pl.Float64).fill_null(0.0).alias(dropback_col),
        pl.col(response_col).cast(pl.Float64).alias(response_col),
        (pl.col(response_col) + pl.col("defense_coefficient")).alias("adjusted_game"),
    )
    if joined.is_empty():
        return pl.DataFrame()

    group_keys = [
        column for column in ("season", "qb_id", "qb_name", "team") if column in joined.columns
    ]
    aggregated = (
        joined.group_by(group_keys)
        .agg(
            (
                (pl.col("defense_coefficient") * pl.col(dropback_col)).sum()
                / pl.col(dropback_col).sum()
            ).alias("faced_difficulty"),
            (
                (pl.col("adjusted_game") * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()
            ).alias("additive_prediction"),
            pl.col(dropback_col).sum().alias("total_dropbacks"),
            pl.when(pl.col("is_top_half"))
            .then(pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("vs_top_half_dropbacks"),
            pl.when(~pl.col("is_top_half"))
            .then(pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("vs_bottom_half_dropbacks"),
            pl.when(pl.col("is_top_half"))
            .then(pl.col(response_col) * pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("_top_raw_weighted"),
            pl.when(pl.col("is_top_half"))
            .then(pl.col("adjusted_game") * pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("_top_adjusted_weighted"),
            pl.when(~pl.col("is_top_half"))
            .then(pl.col(response_col) * pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("_bottom_raw_weighted"),
            pl.when(~pl.col("is_top_half"))
            .then(pl.col("adjusted_game") * pl.col(dropback_col))
            .otherwise(0.0)
            .sum()
            .alias("_bottom_adjusted_weighted"),
        )
        .with_columns(
            pl.when(pl.col("vs_top_half_dropbacks") > 0.0)
            .then(pl.col("_top_raw_weighted") / pl.col("vs_top_half_dropbacks"))
            .otherwise(None)
            .alias("vs_top_half_raw_epa_per_dropback"),
            pl.when(pl.col("vs_top_half_dropbacks") > 0.0)
            .then(pl.col("_top_adjusted_weighted") / pl.col("vs_top_half_dropbacks"))
            .otherwise(None)
            .alias("vs_top_half_adjusted_epa_per_dropback"),
            pl.when(pl.col("vs_bottom_half_dropbacks") > 0.0)
            .then(pl.col("_bottom_raw_weighted") / pl.col("vs_bottom_half_dropbacks"))
            .otherwise(None)
            .alias("vs_bottom_half_raw_epa_per_dropback"),
            pl.when(pl.col("vs_bottom_half_dropbacks") > 0.0)
            .then(pl.col("_bottom_adjusted_weighted") / pl.col("vs_bottom_half_dropbacks"))
            .otherwise(None)
            .alias("vs_bottom_half_adjusted_epa_per_dropback"),
        )
        .with_columns(
            (pl.col("vs_top_half_adjusted_epa_per_dropback") - pl.col("additive_prediction")).alias(
                "vs_top_half_residual"
            ),
            (
                pl.col("vs_bottom_half_adjusted_epa_per_dropback") - pl.col("additive_prediction")
            ).alias("vs_bottom_half_residual"),
        )
        .drop(
            [
                "_top_raw_weighted",
                "_top_adjusted_weighted",
                "_bottom_raw_weighted",
                "_bottom_adjusted_weighted",
            ]
        )
    )
    return aggregated.sort(group_keys)


def summarize_qb_split_half_signal(
    split_frame: pl.DataFrame,
    *,
    residual_col: str,
    weight_col: str,
    resamples: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Summarize one split-half residual signal with pooled and per-season weighted slopes."""
    summary = _summarize_weighted_signal_by_season(
        split_frame,
        predictor_col="faced_difficulty",
        value_col=residual_col,
        weight_col=weight_col,
        resamples=resamples,
        seed=seed,
    )
    if summary.is_empty():
        return summary
    return summary.with_columns(pl.lit(residual_col).alias("residual_label"))


def evaluate_qb_split_half_decision(
    primary_summary: pl.DataFrame,
    placebo_summary: pl.DataFrame,
) -> dict[str, object]:
    """Return the split-half gate reading and whether the placebo looks symmetric."""
    primary_pooled = primary_summary.filter(pl.col("scope") == "pooled")
    placebo_pooled = placebo_summary.filter(pl.col("scope") == "pooled")
    if primary_pooled.is_empty() or placebo_pooled.is_empty():
        return {
            "decision": "not_supported",
            "primary_gate_supported": False,
            "placebo_is_symmetric": False,
        }

    primary_row = primary_pooled.row(0, named=True)
    placebo_row = placebo_pooled.row(0, named=True)
    primary_gate_supported = bool(
        float(primary_row["ci_lower"]) > 0.0 and float(primary_row["direction_p_value"]) < 0.05
    )
    placebo_is_symmetric = bool(float(placebo_row["ci_lower"]) > 0.0)
    return {
        "decision": "supported"
        if primary_gate_supported and not placebo_is_symmetric
        else "not_supported",
        "primary_gate_supported": primary_gate_supported,
        "placebo_is_symmetric": placebo_is_symmetric,
    }


def build_qb_opponent_offense_frame(
    qb_games: pl.DataFrame,
    offense_ratings: pl.DataFrame,
    defense_ratings: pl.DataFrame,
    *,
    qb_meta: pl.DataFrame | None = None,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
    home_field_col: str = "is_home",
    home_field_advantage: float = 0.0,
) -> pl.DataFrame:
    """Return QB-game rows with opponent-offense context and adjusted residuals."""
    required_columns = {"qb_id", "qb_name", "team", "opponent_team", response_col, dropback_col}
    if not required_columns.issubset(set(qb_games.columns)):
        return pl.DataFrame()

    joined = (
        qb_games.join(
            offense_ratings.select(
                pl.col("team").cast(pl.String).alias("opponent_team"),
                pl.col("offense_rating").cast(pl.Float64).alias("opponent_offense_coefficient"),
            ),
            on="opponent_team",
            how="left",
        )
        .join(
            defense_ratings.select(
                pl.col("team").cast(pl.String).alias("opponent_team"),
                pl.col("defense_rating").cast(pl.Float64).alias("opponent_defense_coefficient"),
            ),
            on="opponent_team",
            how="left",
        )
        .drop_nulls(
            [
                response_col,
                dropback_col,
                "opponent_offense_coefficient",
                "opponent_defense_coefficient",
            ]
        )
    )
    if joined.is_empty():
        return pl.DataFrame()

    if qb_meta is not None and not qb_meta.is_empty():
        join_keys = [
            column
            for column in ("season", "qb_id", "qb_name", "team")
            if column in qb_meta.columns and column in joined.columns
        ]
        if join_keys:
            joined = joined.join(qb_meta, on=join_keys, how="left")
            if "qb_is_eligible" in joined.columns:
                joined = joined.filter(pl.col("qb_is_eligible"))
    if joined.is_empty():
        return pl.DataFrame()

    home_effect = pl.lit(0.0)
    if home_field_col in joined.columns:
        home_effect = (
            pl.when(pl.col(home_field_col).is_null())
            .then(0.0)
            .when(pl.col(home_field_col))
            .then(home_field_advantage)
            .otherwise(-home_field_advantage)
        )

    group_keys = [
        column for column in ("season", "qb_id", "qb_name", "team") if column in joined.columns
    ]
    adjusted = joined.with_columns(
        pl.col(dropback_col).cast(pl.Float64).alias(dropback_col),
        (
            pl.col(response_col).cast(pl.Float64)
            + pl.col("opponent_defense_coefficient")
            - home_effect
        ).alias("adjusted_game_epa_per_dropback"),
    )
    season_adjusted = adjusted.group_by(group_keys).agg(
        (
            (pl.col("adjusted_game_epa_per_dropback") * pl.col(dropback_col)).sum()
            / pl.col(dropback_col).sum()
        ).alias("season_adjusted_epa_per_dropback")
    )
    return (
        adjusted.join(season_adjusted, on=group_keys, how="left")
        .with_columns(
            (
                pl.col("adjusted_game_epa_per_dropback")
                - pl.col("season_adjusted_epa_per_dropback")
            ).alias("adjusted_residual")
        )
        .select(
            [
                *group_keys,
                "opponent_team",
                dropback_col,
                "opponent_offense_coefficient",
                "opponent_defense_coefficient",
                "adjusted_game_epa_per_dropback",
                "season_adjusted_epa_per_dropback",
                "adjusted_residual",
            ]
        )
        .sort([*group_keys, "opponent_team"])
    )


def summarize_qb_opponent_offense_signal(
    frame: pl.DataFrame,
    *,
    predictor_col: str = "opponent_offense_coefficient",
    value_col: str = "adjusted_residual",
    weight_col: str = "qb_dropbacks",
    resamples: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Return pooled and per-season support for the opponent-offense residual effect."""
    return _summarize_weighted_signal_by_season(
        frame,
        predictor_col=predictor_col,
        value_col=value_col,
        weight_col=weight_col,
        resamples=resamples,
        seed=seed,
    )


def build_qb_leverage_profile_frame(
    dropback_plays: pl.DataFrame,
    schedule_frame: pl.DataFrame,
    *,
    wp_col: str = "wp",
    moderate_wp_low: float = 0.05,
    moderate_wp_high: float = 0.95,
) -> pl.DataFrame:
    """Return QB-season leverage-share profiles joined to schedule-softness context."""
    required_columns = {"season", "qb_id", "qb_name", "team", wp_col}
    if not required_columns.issubset(set(dropback_plays.columns)):
        return pl.DataFrame()

    profile = (
        dropback_plays.select(
            pl.col("season").cast(pl.Int64),
            pl.col("qb_id").cast(pl.String),
            pl.col("qb_name").cast(pl.String),
            pl.col("team").cast(pl.String),
            pl.col(wp_col).cast(pl.Float64).alias(wp_col),
        )
        .group_by(["season", "qb_id", "qb_name", "team"])
        .agg(
            pl.len().cast(pl.Float64).alias("total_dropbacks"),
            pl.when((pl.col(wp_col) <= moderate_wp_low) | (pl.col(wp_col) >= moderate_wp_high))
            .then(1.0)
            .otherwise(0.0)
            .sum()
            .alias("low_leverage_dropbacks"),
            pl.when((pl.col(wp_col) > moderate_wp_low) & (pl.col(wp_col) < moderate_wp_high))
            .then(1.0)
            .otherwise(0.0)
            .sum()
            .alias("moderate_leverage_dropbacks"),
        )
        .with_columns(
            (pl.col("low_leverage_dropbacks") / pl.col("total_dropbacks")).alias(
                "low_leverage_share"
            ),
            (pl.col("moderate_leverage_dropbacks") / pl.col("total_dropbacks")).alias(
                "moderate_leverage_share"
            ),
        )
    )
    join_keys = [
        column
        for column in ("season", "qb_id", "qb_name", "team")
        if column in schedule_frame.columns and column in profile.columns
    ]
    if join_keys:
        profile = profile.join(schedule_frame, on=join_keys, how="left")
    return profile.sort(["season", "qb_id"])


def summarize_qb_leverage_signal(
    frame: pl.DataFrame,
    *,
    predictor_col: str = "schedule_softness",
    value_col: str = "low_leverage_share",
    weight_col: str = "total_dropbacks",
    resamples: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Return pooled and per-season support for the leverage-softness hypothesis."""
    return _summarize_weighted_signal_by_season(
        frame,
        predictor_col=predictor_col,
        value_col=value_col,
        weight_col=weight_col,
        resamples=resamples,
        seed=seed,
    )


def compute_qb_split_half_diagnostics(
    data_dir: Path,
    seasons: Sequence[int],
    *,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
) -> pl.DataFrame:
    """Return one QB-season split-half diagnostics row per eligible quarterback season."""
    rows: list[pl.DataFrame] = []
    for season in sorted(seasons):
        qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet").with_columns(
            pl.lit(int(season)).cast(pl.Int64).alias("season")
        )
        qb_meta = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet").select(
            [
                column
                for column in ("season", "qb_id", "qb_name", "team", "qb_is_eligible")
                if column in pl.read_parquet(data_dir / f"{season}_qb_combined.parquet").columns
            ]
        )
        if "season" not in qb_meta.columns:
            qb_meta = qb_meta.with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        defense_ratings = pl.read_parquet(
            data_dir / f"{season}_simultaneous_team_adjustments.parquet"
        ).select(
            pl.col("team"),
            pl.col("adj_def_passing_epa_per_offensive_snap").alias("defense_coefficient"),
        )
        frame = build_qb_split_half_frame(
            qb_games,
            defense_ratings,
            qb_meta=qb_meta,
            response_col=response_col,
            dropback_col=dropback_col,
        )
        if not frame.is_empty():
            rows.append(frame)
    return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()


def compute_qb_playoff_season_summary(
    playoff_qb_games: pl.DataFrame,
    defense_ratings: pl.DataFrame,
    *,
    qb_meta: pl.DataFrame | None = None,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
) -> pl.DataFrame:
    """Aggregate one playoff adjusted EPA/dropback row per eligible QB season."""
    required_columns = {
        "season",
        "qb_id",
        "qb_name",
        "team",
        "opponent_team",
        response_col,
        dropback_col,
    }
    if not required_columns.issubset(set(playoff_qb_games.columns)):
        return pl.DataFrame()
    if not {"team", "defense_coefficient"}.issubset(set(defense_ratings.columns)):
        return pl.DataFrame()

    joined = playoff_qb_games.join(
        defense_ratings.select(
            pl.col("team").cast(pl.String).alias("opponent_team"),
            pl.col("defense_coefficient").cast(pl.Float64).alias("defense_coefficient"),
        ),
        on="opponent_team",
        how="left",
    )
    if qb_meta is not None and not qb_meta.is_empty():
        join_keys = [
            column
            for column in ("season", "qb_id", "qb_name", "team")
            if column in joined.columns and column in qb_meta.columns
        ]
        if join_keys:
            joined = joined.join(qb_meta, on=join_keys, how="left")
            if "qb_is_eligible" in joined.columns:
                joined = joined.filter(pl.col("qb_is_eligible"))
    joined = joined.drop_nulls(["defense_coefficient"]).with_columns(
        pl.col(dropback_col).cast(pl.Float64).fill_null(0.0).alias(dropback_col),
        pl.col(response_col).cast(pl.Float64).alias(response_col),
        (pl.col(response_col) + pl.col("defense_coefficient")).alias("adjusted_game"),
    )
    if joined.is_empty():
        return pl.DataFrame()

    return (
        joined.group_by(["season", "qb_id", "qb_name", "team"])
        .agg(
            pl.col(dropback_col).sum().alias("playoff_dropbacks"),
            (
                (pl.col(response_col) * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()
            ).alias("playoff_raw_epa_per_dropback"),
            (
                (pl.col("adjusted_game") * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()
            ).alias("playoff_adjusted_epa_per_dropback"),
            (
                (pl.col("defense_coefficient") * pl.col(dropback_col)).sum()
                / pl.col(dropback_col).sum()
            ).alias("playoff_faced_difficulty"),
        )
        .sort(["season", "qb_name"])
    )


def compute_qb_playoff_validation_frame(
    data_dir: Path,
    seasons: Sequence[int],
    *,
    qb_split_half: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return playoff QB-season summaries joined to regular-season metrics for validation."""
    rows: list[pl.DataFrame] = []
    for season in sorted(seasons):
        playoff_pbp = load_playoff_pbp_data(int(season))
        if playoff_pbp.is_empty():
            continue

        qb_identity = load_qb_identity_crosswalk(int(season))
        playoff_qb_games = compute_qb_game_stats_from_pbp(playoff_pbp, qb_identity_df=qb_identity)
        opponent_map = (
            playoff_pbp.filter(
                pl.col("posteam").is_not_null()
                & pl.col("defteam").is_not_null()
                & pl.col("passer_player_name").is_not_null()
                & (pl.col("qb_dropback").fill_null(0) > 0)
            )
            .group_by(["game_id", "week", "posteam", "passer_player_id", "passer_player_name"])
            .agg(pl.col("defteam").drop_nulls().first().alias("opponent_team"))
            .rename(
                {
                    "posteam": "team_abbr",
                    "passer_player_id": "qb_id",
                    "passer_player_name": "qb_name",
                }
            )
        )
        playoff_qb_games = (
            playoff_qb_games.join(
                opponent_map,
                on=["game_id", "week", "team_abbr", "qb_id"],
                how="left",
            )
            .rename({"team_abbr": "team"})
            .with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        )

        defense_ratings = pl.read_parquet(
            data_dir / f"{season}_simultaneous_team_adjustments.parquet"
        ).select(
            pl.col("team"),
            pl.col("adj_def_passing_epa_per_offensive_snap").alias("defense_coefficient"),
        )
        regular_metrics = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        qb_meta = regular_metrics.select(
            [
                column
                for column in ("season", "qb_id", "qb_name", "team", "qb_is_eligible")
                if column in regular_metrics.columns
            ]
        )
        if "season" not in qb_meta.columns:
            qb_meta = qb_meta.with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        playoff_summary = compute_qb_playoff_season_summary(
            playoff_qb_games,
            defense_ratings,
            qb_meta=qb_meta,
        )
        if playoff_summary.is_empty():
            continue

        metric_columns = [
            column
            for column in (
                "qb_id",
                "qb_name",
                "team",
                "QSaCR",
                "QSaOR",
                "QRaw",
                "qb_passer_rating",
                "qb_any_a",
            )
            if column in regular_metrics.columns
        ]
        joined = playoff_summary.join(
            regular_metrics.select(metric_columns), on=["qb_id", "qb_name", "team"], how="left"
        )
        if qb_split_half is not None and not qb_split_half.is_empty():
            split_columns = [
                column
                for column in (
                    "season",
                    "qb_id",
                    "team",
                    "vs_top_half_adjusted_epa_per_dropback",
                )
                if column in qb_split_half.columns
            ]
            joined = joined.join(
                qb_split_half.select(split_columns),
                on=[column for column in ("season", "qb_id", "team") if column in split_columns],
                how="left",
            )
        rows.append(joined)
    return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()


def _build_qb_game_frame_from_pbp(
    pbp_df: pl.DataFrame,
    *,
    season: int,
    qb_identity_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return a QB-game frame with opponent and home/away context from PBP."""
    if pbp_df.is_empty():
        return pl.DataFrame()

    qb_games = compute_qb_game_stats_from_pbp(pbp_df, qb_identity_df=qb_identity_df)
    if qb_games.is_empty():
        return pl.DataFrame()

    opponent_map = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
        )
        .group_by(["game_id", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg(
            pl.col("defteam").drop_nulls().first().alias("opponent_team"),
            pl.col("home_team").drop_nulls().first().alias("home_team"),
            pl.col("away_team").drop_nulls().first().alias("away_team"),
        )
        .rename(
            {
                "posteam": "team_abbr",
                "passer_player_id": "qb_id",
                "passer_player_name": "qb_name",
            }
        )
    )
    return (
        qb_games.join(
            opponent_map,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .rename({"team_abbr": "team"})
        .with_columns(
            pl.lit(int(season)).cast(pl.Int64).alias("season"),
            (pl.col("team") == pl.col("home_team")).alias("is_home"),
        )
        .drop(["home_team", "away_team"])
    )


def compute_qb_metric_stability_from_history(
    history: pl.DataFrame,
    value_col: str,
) -> dict[str, float | int] | None:
    """Return adjacent-season Pearson and Spearman stability for one QB metric history."""
    if history.is_empty() or value_col not in history.columns:
        return None

    season_frames: list[pl.DataFrame] = []
    seasons = sorted(history.select("season").to_series().cast(pl.Int64).unique().to_list())
    for season, next_season in zip(seasons, seasons[1:], strict=False):
        current = history.filter(pl.col("season") == int(season)).select(
            "qb_id", pl.col(value_col).alias("value_t")
        )
        nxt = history.filter(pl.col("season") == int(next_season)).select(
            "qb_id", pl.col(value_col).alias("value_t1")
        )
        season_frames.append(current.join(nxt, on="qb_id", how="inner"))

    if not season_frames:
        return None
    pair_frame = pl.concat(season_frames, how="diagonal_relaxed").drop_nulls(
        ["value_t", "value_t1"]
    )
    if pair_frame.is_empty():
        return None

    x_values = np.asarray(
        pair_frame.select("value_t").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    y_values = np.asarray(
        pair_frame.select("value_t1").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    weights = np.ones(len(x_values), dtype=np.float64)
    x_ranks = np.asarray(pl.Series(x_values).rank(method="average").to_list(), dtype=np.float64)
    y_ranks = np.asarray(pl.Series(y_values).rank(method="average").to_list(), dtype=np.float64)
    return {
        "paired_rows": int(len(x_values)),
        "pearson": _weighted_correlation(x_values, y_values, weights),
        "spearman": _weighted_correlation(x_ranks, y_ranks, weights),
    }


def compute_qb_opponent_offense_diagnostics(
    data_dir: Path,
    seasons: Sequence[int],
    *,
    qb_names: Sequence[str] = ("Drake Maye", "Matthew Stafford"),
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
    team_response_col: str = "passing_epa_per_offensive_snap",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Return opponent-offense summary rows, named cases, and the support decision."""
    rows: list[pl.DataFrame] = []
    latest_season = max(seasons) if seasons else None

    for season in sorted(seasons):
        team_games = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet")
        qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")

        if "team" not in qb_games.columns and "team_abbr" in qb_games.columns:
            qb_games = qb_games.rename({"team_abbr": "team"})
        if {"game_id", "week", "team"}.issubset(set(qb_games.columns)) and (
            "opponent_team" not in qb_games.columns or "is_home" not in qb_games.columns
        ):
            context_columns = [
                column
                for column in ("game_id", "week", "team", "opponent_team", "is_home")
                if column in team_games.columns
            ]
            qb_games = qb_games.join(
                team_games.select(context_columns),
                on=[column for column in ("game_id", "week", "team") if column in context_columns],
                how="left",
            )

        qb_meta = qb_combined.select(
            [
                column
                for column in ("season", "qb_id", "qb_name", "team", "qb_is_eligible")
                if column in qb_combined.columns
            ]
        )
        if "season" not in qb_meta.columns:
            qb_meta = qb_meta.with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        if "season" not in qb_games.columns:
            qb_games = qb_games.with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))

        offense_ratings, home_field = solve_team_stat_ridge(
            team_games,
            response_col=team_response_col,
        )
        _, defense_ratings = solve_qb_stat_ridge(qb_games, response_col=response_col)
        frame = build_qb_opponent_offense_frame(
            qb_games,
            offense_ratings.select(["team", "offense_rating"]),
            defense_ratings.select(["team", "defense_rating"]),
            qb_meta=qb_meta,
            response_col=response_col,
            dropback_col=dropback_col,
            home_field_advantage=home_field,
        )
        if not frame.is_empty():
            rows.append(frame)

    all_rows = pl.concat(rows, how="vertical") if rows else pl.DataFrame()
    summary = summarize_qb_opponent_offense_signal(all_rows, seed=0)

    cases = pl.DataFrame()
    if latest_season is not None and not all_rows.is_empty():
        cases = (
            all_rows.filter(
                (pl.col("season") == int(latest_season)) & pl.col("qb_name").is_in(list(qb_names))
            )
            .group_by(["season", "qb_name"])
            .agg(
                (
                    (pl.col("opponent_offense_coefficient") * pl.col(dropback_col)).sum()
                    / pl.col(dropback_col).sum()
                ).alias("faced_opponent_offense"),
                (
                    (pl.col("adjusted_residual") * pl.col(dropback_col)).sum()
                    / pl.col(dropback_col).sum()
                ).alias("mean_adjusted_residual"),
                pl.col(dropback_col).sum().alias("total_dropbacks"),
            )
            .sort("qb_name")
        )

    decision: dict[str, object] = {
        "decision": "not_supported",
        "consistent_sign_count": 0,
        "consistent_sign_total": 0,
        "consistent_sign_p_value": 1.0,
    }
    pooled = (
        summary.filter(pl.col("scope") == "pooled") if not summary.is_empty() else pl.DataFrame()
    )
    if not pooled.is_empty():
        pooled_row = pooled.row(0, named=True)
        total = int(pooled_row["direction_total_count"] or 0)
        positive = int(pooled_row["direction_positive_count"] or 0)
        consistent = positive if float(pooled_row["slope"]) >= 0.0 else total - positive
        consistent_p = _one_sided_binomial_tail(consistent, total)
        supported = (
            float(pooled_row["ci_lower"]) * float(pooled_row["ci_upper"]) > 0.0
            and consistent_p < 0.05
        )
        decision = {
            "decision": "supported" if supported else "not_supported",
            "consistent_sign_count": consistent,
            "consistent_sign_total": total,
            "consistent_sign_p_value": consistent_p,
        }

    return summary, cases, decision


def compute_qb_leverage_diagnostics(
    data_dir: Path,
    seasons: Sequence[int],
    *,
    qb_names: Sequence[str] = ("Drake Maye", "Matthew Stafford"),
    moderate_wp_low: float = 0.05,
    moderate_wp_high: float = 0.95,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Return leverage summaries, named cases, filtered-history rows, and decision inputs."""
    profile_rows: list[pl.DataFrame] = []
    variant_rows: list[pl.DataFrame] = []
    latest_season = max(seasons) if seasons else None

    for season in sorted(seasons):
        pbp = load_pbp_data(int(season))
        if pbp.is_empty():
            continue
        qb_identity = load_qb_identity_crosswalk(int(season))
        regular_metrics = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        if "season" not in regular_metrics.columns:
            regular_metrics = regular_metrics.with_columns(
                pl.lit(int(season)).cast(pl.Int64).alias("season")
            )

        dropback_plays = pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_id").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
        ).select(
            pl.lit(int(season)).cast(pl.Int64).alias("season"),
            pl.col("passer_player_id").cast(pl.String).alias("qb_id"),
            pl.col("passer_player_name").cast(pl.String).alias("qb_name"),
            pl.col("posteam").cast(pl.String).alias("team"),
            pl.col("wp").cast(pl.Float64).alias("wp"),
        )
        if not qb_identity.is_empty():
            dropback_plays = (
                dropback_plays.join(
                    qb_identity.select(["qb_id", "qb_name"]).rename(
                        {"qb_name": "canonical_qb_name"}
                    ),
                    on="qb_id",
                    how="left",
                )
                .with_columns(
                    pl.coalesce([pl.col("canonical_qb_name"), pl.col("qb_name")]).alias("qb_name")
                )
                .drop("canonical_qb_name")
            )

        if "adj_def_qb_epa_per_dropback_faced" in regular_metrics.columns:
            schedule_frame = regular_metrics.with_columns(
                (-pl.col("adj_def_qb_epa_per_dropback_faced").cast(pl.Float64)).alias(
                    "schedule_softness"
                )
            ).select(
                [
                    column
                    for column in (
                        "season",
                        "qb_id",
                        "qb_name",
                        "team",
                        "qb_is_eligible",
                        "schedule_softness",
                    )
                    if column in regular_metrics.columns or column == "schedule_softness"
                ]
            )
        else:
            schedule_frame = regular_metrics.select(
                [
                    column
                    for column in ("season", "qb_id", "qb_name", "team", "qb_is_eligible")
                    if column in regular_metrics.columns
                ]
            ).with_columns(pl.lit(0.0).alias("schedule_softness"))

        profile = build_qb_leverage_profile_frame(
            dropback_plays,
            schedule_frame,
            moderate_wp_low=moderate_wp_low,
            moderate_wp_high=moderate_wp_high,
        )
        if "qb_is_eligible" in profile.columns:
            profile = profile.filter(pl.col("qb_is_eligible"))
        if not profile.is_empty():
            profile_rows.append(profile)

        filtered_pbp = pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_id").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
            & (pl.col("wp").cast(pl.Float64) > moderate_wp_low)
            & (pl.col("wp").cast(pl.Float64) < moderate_wp_high)
        )
        filtered_games = _build_qb_game_frame_from_pbp(
            filtered_pbp,
            season=int(season),
            qb_identity_df=qb_identity,
        )
        if filtered_games.is_empty():
            continue

        filtered_qb_ratings, _ = solve_qb_stat_ridge(
            filtered_games,
            response_col="qb_epa_per_dropback",
        )
        history_frame = (
            filtered_qb_ratings.rename(
                {"offense_rating": "moderate_leverage_adjusted_epa_per_dropback"}
            )
            .with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
            .join(
                regular_metrics.select(
                    [
                        column
                        for column in ("season", "qb_id", "qb_name", "team", "qb_is_eligible")
                        if column in regular_metrics.columns
                    ]
                ),
                on=[column for column in ("season", "qb_id") if column in regular_metrics.columns],
                how="left",
            )
        )
        if "qb_is_eligible" in history_frame.columns:
            history_frame = history_frame.filter(pl.col("qb_is_eligible"))
        if not history_frame.is_empty():
            variant_rows.append(
                history_frame.select(
                    [
                        column
                        for column in (
                            "season",
                            "qb_id",
                            "qb_name",
                            "team",
                            "moderate_leverage_adjusted_epa_per_dropback",
                        )
                        if column in history_frame.columns
                    ]
                )
            )

    profiles = pl.concat(profile_rows, how="vertical") if profile_rows else pl.DataFrame()
    summary = summarize_qb_leverage_signal(profiles, seed=0)
    variant_history = pl.concat(variant_rows, how="vertical") if variant_rows else pl.DataFrame()

    cases = pl.DataFrame()
    if latest_season is not None and not profiles.is_empty():
        cases = profiles.filter(
            (pl.col("season") == int(latest_season)) & pl.col("qb_name").is_in(list(qb_names))
        ).select(
            [
                column
                for column in (
                    "season",
                    "qb_name",
                    "schedule_softness",
                    "low_leverage_share",
                    "moderate_leverage_share",
                )
                if column in profiles.columns
            ]
        )

    decision = {
        "decision": "not_supported",
        "moderate_wp_band": f"{moderate_wp_low:.2f}-{moderate_wp_high:.2f}",
        "share_signal_supported": False,
    }
    pooled = (
        summary.filter(pl.col("scope") == "pooled") if not summary.is_empty() else pl.DataFrame()
    )
    if not pooled.is_empty():
        pooled_row = pooled.row(0, named=True)
        share_signal_supported = bool(
            float(pooled_row["ci_lower"]) > 0.0 and float(pooled_row["direction_p_value"]) < 0.05
        )
        decision["share_signal_supported"] = share_signal_supported

    return summary, cases, variant_history, decision


def compute_weekly_mae_curves(predictions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate weekly MAE and RMSE curves by baseline across all evaluated seasons."""
    if predictions.is_empty():
        return pl.DataFrame(
            schema={
                "baseline": pl.String,
                "week": pl.Int64,
                "games": pl.Int64,
                "mae": pl.Float64,
                "rmse": pl.Float64,
            }
        )

    scored_predictions = _ensure_error_column(predictions)
    return (
        scored_predictions.group_by(["baseline", "week"])
        .agg(
            pl.len().alias("games"),
            pl.col("error").abs().mean().alias("mae"),
            (pl.col("error") * pl.col("error")).mean().sqrt().alias("rmse"),
        )
        .sort(["baseline", "week"])
    )


def compute_season_mae_deltas(
    metrics: pl.DataFrame,
    *,
    baseline_a: str,
    baseline_b: str,
) -> pl.DataFrame:
    """Return per-season error deltas between two baselines."""
    season_metrics = metrics.filter(pl.col("split") == "season")
    frame_a = season_metrics.filter(pl.col("baseline") == baseline_a).select(
        "season",
        pl.col("games").alias("games_a"),
        pl.col("mae").alias("mae_a"),
        pl.col("rmse").alias("rmse_a"),
    )
    frame_b = season_metrics.filter(pl.col("baseline") == baseline_b).select(
        "season",
        pl.col("games").alias("games_b"),
        pl.col("mae").alias("mae_b"),
        pl.col("rmse").alias("rmse_b"),
    )

    if frame_a.is_empty() or frame_b.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "games": pl.Int64,
                "baseline_a": pl.String,
                "baseline_b": pl.String,
                "mae_a": pl.Float64,
                "mae_b": pl.Float64,
                "mae_delta": pl.Float64,
                "rmse_a": pl.Float64,
                "rmse_b": pl.Float64,
                "rmse_delta": pl.Float64,
            }
        )

    return (
        frame_a.join(frame_b, on="season", how="inner")
        .with_columns(
            pl.lit(baseline_a).alias("baseline_a"),
            pl.lit(baseline_b).alias("baseline_b"),
            pl.min_horizontal("games_a", "games_b").alias("games"),
            (pl.col("mae_a") - pl.col("mae_b")).alias("mae_delta"),
            (pl.col("rmse_a") - pl.col("rmse_b")).alias("rmse_delta"),
        )
        .select(
            "season",
            "games",
            "baseline_a",
            "baseline_b",
            "mae_a",
            "mae_b",
            "mae_delta",
            "rmse_a",
            "rmse_b",
            "rmse_delta",
        )
        .sort("season")
    )


def build_qb_adjustment_audit_frame(
    qb_games: pl.DataFrame,
    *,
    response_col: str,
    qb_col: str = "qb_id",
    qb_name_col: str = "qb_name",
    team_col: str = "team",
    defense_col: str = "opponent_team",
    dropback_col: str = "qb_dropbacks",
    ridge_lambda: float | None = None,
) -> pl.DataFrame:
    """Reconstruct raw, adjusted, and faced-defense QB values in common EPA units."""
    filtered_games = qb_games.drop_nulls([qb_col, defense_col, response_col])
    if filtered_games.is_empty():
        return pl.DataFrame(
            schema={
                qb_col: pl.String,
                qb_name_col: pl.String,
                team_col: pl.String,
                "raw_value": pl.Float64,
                "adjusted_value": pl.Float64,
                "weighted_faced_defense": pl.Float64,
                "adjustment_delta": pl.Float64,
                "identity_residual": pl.Float64,
                "total_dropbacks": pl.Float64,
            }
        )

    qb_ratings, defense_ratings = solve_qb_stat_ridge(
        filtered_games,
        response_col=response_col,
        qb_col=qb_col,
        defense_col=defense_col,
        dropback_col=dropback_col,
        ridge_lambda=ridge_lambda,
    )

    group_keys = [
        column for column in (qb_col, qb_name_col, team_col) if column in filtered_games.columns
    ]
    weight_col = dropback_col if dropback_col in filtered_games.columns else response_col

    raw_frame = _weighted_mean_frame(
        filtered_games,
        group_keys=group_keys,
        value_col=response_col,
        weight_col=weight_col,
        alias="raw_value",
    )
    joined_defenses = filtered_games.join(
        defense_ratings.rename({"team": defense_col, "defense_rating": "weighted_faced_defense"}),
        on=defense_col,
        how="left",
    )
    faced_defense = _weighted_mean_frame(
        joined_defenses.drop_nulls(["weighted_faced_defense"]),
        group_keys=group_keys,
        value_col="weighted_faced_defense",
        weight_col=weight_col,
        alias="weighted_faced_defense",
    ).select([*group_keys, "weighted_faced_defense"])

    adjusted_frame = qb_ratings.rename({"offense_rating": "adjusted_value"})
    if qb_name_col in filtered_games.columns:
        adjusted_frame = adjusted_frame.join(
            filtered_games.select(group_keys).unique(),
            on=qb_col,
            how="left",
        )

    return (
        raw_frame.join(adjusted_frame, on=group_keys, how="left")
        .join(faced_defense, on=group_keys, how="left")
        .with_columns(
            (pl.col("adjusted_value") - pl.col("raw_value")).alias("adjustment_delta"),
            (
                pl.col("adjusted_value") - pl.col("raw_value") - pl.col("weighted_faced_defense")
            ).alias("identity_residual"),
        )
        .sort(group_keys)
    )


def summarize_qb_adjustment_slopes(audit_frame: pl.DataFrame) -> pl.DataFrame:
    """Summarize season-level QB adjustment slope, correlation, and residual size."""
    if audit_frame.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "rows": pl.Int64,
                "slope": pl.Float64,
                "correlation": pl.Float64,
                "mean_abs_identity_residual": pl.Float64,
            }
        )

    rows: list[dict[str, object]] = []

    grouped_frames: list[tuple[int | None, pl.DataFrame]]
    if "season" in audit_frame.columns:
        grouped_frames = [
            (int(keys[0]), frame)
            for keys, frame in audit_frame.group_by(["season"], maintain_order=True)
        ]
    else:
        grouped_frames = [(None, audit_frame)]

    for season_value, frame in grouped_frames:
        paired = frame.drop_nulls(
            ["weighted_faced_defense", "adjustment_delta", "identity_residual"]
        )
        if paired.is_empty():
            continue

        x_values = np.asarray(
            paired.select("weighted_faced_defense").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            paired.select("adjustment_delta").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        x_centered = x_values - float(x_values.mean())
        y_centered = y_values - float(y_values.mean())
        denominator = float((x_centered * x_centered).sum())
        slope = float((x_centered * y_centered).sum() / denominator) if denominator > 0.0 else 0.0
        x_std = float(x_values.std(ddof=1)) if x_values.size > 1 else 0.0
        y_std = float(y_values.std(ddof=1)) if y_values.size > 1 else 0.0
        correlation = (
            float(np.corrcoef(x_values, y_values)[0, 1])
            if x_values.size > 1 and x_std > 0.0 and y_std > 0.0
            else 0.0
        )
        rows.append(
            {
                "season": season_value,
                "rows": paired.height,
                "slope": slope,
                "correlation": correlation,
                "mean_abs_identity_residual": float(
                    paired.select(pl.col("identity_residual").abs().mean()).item()
                ),
            }
        )

    return pl.DataFrame(rows).sort("season") if rows else pl.DataFrame()


def summarize_defense_spread(
    team_defense_ratings: pl.DataFrame,
    qb_defense_ratings: pl.DataFrame,
) -> pl.DataFrame:
    """Report the team and QB defense-coefficient spread plus their ratio."""
    team_values = np.asarray(
        team_defense_ratings.select("defense_rating").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    qb_values = np.asarray(
        qb_defense_ratings.select("defense_rating").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    team_sd = float(team_values.std(ddof=1)) if team_values.size > 1 else 0.0
    qb_sd = float(qb_values.std(ddof=1)) if qb_values.size > 1 else 0.0
    ratio = float(qb_sd / team_sd) if team_sd > 0.0 else 0.0
    return pl.DataFrame(
        {
            "team_defense_sd": [team_sd],
            "qb_defense_sd": [qb_sd],
            "qb_to_team_spread_ratio": [ratio],
        }
    )


def _weighted_qb_raw_means(
    qb_games: pl.DataFrame,
    *,
    qb_col: str,
    response_col: str,
    dropback_col: str,
) -> pl.DataFrame:
    """Return one weighted mean raw QB response per QB."""
    return qb_games.group_by(qb_col).agg(
        ((pl.col(response_col) * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()).alias(
            "raw_weighted"
        )
    )


def compute_qb_season_audit_summary(
    data_dir: Path,
    seasons: Sequence[int],
    *,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
) -> pl.DataFrame:
    """Return season-level eligible-QB schedule-adjustment slopes and residual summaries."""
    rows: list[pl.DataFrame] = []
    for season in sorted(seasons):
        qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet")
        qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        qb_meta = qb_combined.select(
            [column for column in ("qb_id", "qb_is_eligible") if column in qb_combined.columns]
        )
        audit = build_qb_adjustment_audit_frame(
            qb_games,
            response_col=response_col,
            dropback_col=dropback_col,
        ).join(qb_meta, on="qb_id", how="left")
        if "qb_is_eligible" in audit.columns:
            audit = audit.filter(pl.col("qb_is_eligible"))
        if audit.is_empty():
            continue
        season_summary = summarize_qb_adjustment_slopes(
            audit.with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        )
        if not season_summary.is_empty():
            rows.append(season_summary)

    return pl.concat(rows, how="vertical") if rows else pl.DataFrame()


def compute_qb_defense_spread_summary(
    data_dir: Path,
    seasons: Sequence[int],
) -> pl.DataFrame:
    """Return season-level QB-vs-team defense spread comparisons."""
    rows: list[dict[str, float | int]] = []
    for season in sorted(seasons):
        team_games = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet")
        team_defense, _ = solve_team_stat_ridge(
            team_games,
            response_col="passing_epa_per_offensive_snap",
        )
        _, qb_defense = solve_qb_stat_ridge(
            qb_games,
            response_col="qb_epa_per_dropback",
        )
        spread = summarize_defense_spread(team_defense, qb_defense).row(0, named=True)
        rows.append({"season": int(season), **spread})

    return pl.DataFrame(rows).sort("season") if rows else pl.DataFrame()


def compute_qb_experiment_sweep(
    data_dir: Path,
    season: int,
    *,
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
    defense_penalty_multipliers: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> pl.DataFrame:
    """Return a 2025-style QB revision comparison table for the published eligible-QB slice."""
    qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet").drop_nulls(
        ["qb_id", "opponent_team", response_col]
    )
    qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
    qb_meta = qb_combined.select(
        [
            column
            for column in ("qb_id", "qb_name", "qb_is_eligible")
            if column in qb_combined.columns
        ]
    )
    raw_means = _weighted_qb_raw_means(
        qb_games,
        qb_col="qb_id",
        response_col=response_col,
        dropback_col=dropback_col,
    )

    qbs = sorted(qb_games["qb_id"].unique().to_list())
    defenses = sorted(qb_games["opponent_team"].unique().to_list())
    qb_index = {qb: index for index, qb in enumerate(qbs)}
    defense_index = {team: index for index, team in enumerate(defenses)}
    design = np.zeros((qb_games.height, len(qbs) + len(defenses)), dtype=np.float64)
    for row_index, (quarterback, defense) in enumerate(
        qb_games.select(["qb_id", "opponent_team"]).iter_rows()
    ):
        design[row_index, qb_index[str(quarterback)]] = 1.0
        design[row_index, len(qbs) + defense_index[str(defense)]] = -1.0
    response = np.asarray(
        qb_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )
    sample_weights = np.asarray(
        qb_games.select(pl.col(dropback_col).cast(pl.Float64).fill_null(0.0)).to_series().to_list(),
        dtype=np.float64,
    )
    base_lambda = tune_ridge_lambda(design, response, sample_weights=sample_weights)

    def _summarize_variant(
        label: str,
        qb_ratings: pl.DataFrame,
        faced_schedule: pl.DataFrame,
    ) -> dict[str, object]:
        frame = (
            qb_ratings.join(raw_means, on="qb_id", how="left")
            .join(faced_schedule, on="qb_id", how="left")
            .join(qb_meta, on="qb_id", how="left")
        )
        if "qb_is_eligible" in frame.columns:
            frame = frame.filter(pl.col("qb_is_eligible"))
        frame = frame.drop_nulls(["raw_weighted", "faced"])
        x_values = frame["faced"].to_numpy()
        y_values = (frame["offense_rating"] - frame["raw_weighted"]).to_numpy()
        x_centered = x_values - float(x_values.mean())
        y_centered = y_values - float(y_values.mean())
        denominator = float((x_centered * x_centered).sum())
        slope = float((x_centered * y_centered).sum() / denominator) if denominator > 0.0 else 0.0
        correlation = (
            float(np.corrcoef(x_values, y_values)[0, 1])
            if len(x_values) > 1
            and float(np.std(x_values, ddof=1)) > 0.0
            and float(np.std(y_values, ddof=1)) > 0.0
            else 0.0
        )
        return {
            "variant": label,
            "season": int(season),
            "eligible_rows": int(frame.height),
            "slope": slope,
            "correlation": correlation,
        }

    rows: list[dict[str, object]] = []

    current_qb, current_defense = solve_qb_stat_ridge(qb_games, response_col=response_col)
    current_schedule = (
        qb_games.join(
            current_defense.rename({"team": "opponent_team", "defense_rating": "faced"}),
            on="opponent_team",
            how="left",
        )
        .group_by("qb_id")
        .agg(
            ((pl.col("faced") * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()).alias(
                "faced"
            )
        )
    )
    rows.append(_summarize_variant("current", current_qb, current_schedule))

    team_games = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
    team_defense, _ = solve_team_stat_ridge(
        team_games,
        response_col="passing_epa_per_offensive_snap",
    )
    fixed_qb = solve_qb_stat_with_fixed_defense_offsets(
        qb_games,
        response_col=response_col,
        fixed_defense_ratings=team_defense.select(["team", "defense_rating"]),
        ridge_lambda=base_lambda,
    )
    fixed_schedule = (
        qb_games.join(
            team_defense.rename({"team": "opponent_team", "defense_rating": "faced"}),
            on="opponent_team",
            how="left",
        )
        .group_by("qb_id")
        .agg(
            ((pl.col("faced") * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()).alias(
                "faced"
            )
        )
    )
    rows.append(_summarize_variant("fixed_team_defense", fixed_qb, fixed_schedule))

    for multiplier in defense_penalty_multipliers:
        qb_ratings, defense_ratings = solve_qb_stat_ridge(
            qb_games,
            response_col=response_col,
            ridge_lambda=base_lambda,
            defense_ridge_lambda=base_lambda * float(multiplier),
        )
        schedule = (
            qb_games.join(
                defense_ratings.rename({"team": "opponent_team", "defense_rating": "faced"}),
                on="opponent_team",
                how="left",
            )
            .group_by("qb_id")
            .agg(
                ((pl.col("faced") * pl.col(dropback_col)).sum() / pl.col(dropback_col).sum()).alias(
                    "faced"
                )
            )
        )
        rows.append(
            {
                **_summarize_variant(
                    f"lighter_defense_penalty_x{multiplier:g}",
                    qb_ratings,
                    schedule,
                ),
                "defense_penalty_multiplier": float(multiplier),
            }
        )

    return pl.DataFrame(rows)


def compute_qb_case_study(
    data_dir: Path,
    season: int,
    *,
    qb_names: Sequence[str] = ("Drake Maye", "Matthew Stafford"),
    response_col: str = "qb_epa_per_dropback",
    dropback_col: str = "qb_dropbacks",
) -> pl.DataFrame:
    """Return the Maye/Stafford comparison for the current and comparison QB variants."""
    qb_games = pl.read_parquet(data_dir / f"{season}_qb_game_logs.parquet").drop_nulls(
        ["qb_id", "opponent_team", response_col]
    )
    qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
    qb_meta = qb_combined.select(
        [column for column in ("qb_id", "qb_name") if column in qb_combined.columns]
    )
    raw_means = _weighted_qb_raw_means(
        qb_games,
        qb_col="qb_id",
        response_col=response_col,
        dropback_col=dropback_col,
    )

    qbs = sorted(qb_games["qb_id"].unique().to_list())
    defenses = sorted(qb_games["opponent_team"].unique().to_list())
    qb_index = {qb: index for index, qb in enumerate(qbs)}
    defense_index = {team: index for index, team in enumerate(defenses)}
    design = np.zeros((qb_games.height, len(qbs) + len(defenses)), dtype=np.float64)
    for row_index, (quarterback, defense) in enumerate(
        qb_games.select(["qb_id", "opponent_team"]).iter_rows()
    ):
        design[row_index, qb_index[str(quarterback)]] = 1.0
        design[row_index, len(qbs) + defense_index[str(defense)]] = -1.0
    response = np.asarray(
        qb_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )
    sample_weights = np.asarray(
        qb_games.select(pl.col(dropback_col).cast(pl.Float64).fill_null(0.0)).to_series().to_list(),
        dtype=np.float64,
    )
    base_lambda = tune_ridge_lambda(design, response, sample_weights=sample_weights)

    def _variant_frame(
        label: str, qb_ratings: pl.DataFrame, defense_ratings: pl.DataFrame
    ) -> pl.DataFrame:
        schedule = (
            qb_games.join(
                defense_ratings.rename(
                    {"team": "opponent_team", "defense_rating": "faced_difficulty"}
                ),
                on="opponent_team",
                how="left",
            )
            .group_by("qb_id")
            .agg(
                (
                    (pl.col("faced_difficulty") * pl.col(dropback_col)).sum()
                    / pl.col(dropback_col).sum()
                ).alias("faced_difficulty")
            )
        )
        return (
            qb_ratings.join(raw_means, on="qb_id", how="left")
            .join(schedule, on="qb_id", how="left")
            .join(qb_meta, on="qb_id", how="left")
            .with_columns(
                pl.lit(label).alias("variant"),
                (pl.col("offense_rating") - pl.col("raw_weighted")).alias("adjustment_delta"),
            )
            .filter(pl.col("qb_name").is_in(list(qb_names)))
            .select(
                "variant",
                "qb_name",
                "raw_weighted",
                pl.col("offense_rating").alias("adjusted_value"),
                "faced_difficulty",
                "adjustment_delta",
            )
        )

    current_qb, current_defense = solve_qb_stat_ridge(qb_games, response_col=response_col)
    team_games = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
    team_defense, _ = solve_team_stat_ridge(
        team_games,
        response_col="passing_epa_per_offensive_snap",
    )
    fixed_qb = solve_qb_stat_with_fixed_defense_offsets(
        qb_games,
        response_col=response_col,
        fixed_defense_ratings=team_defense.select(["team", "defense_rating"]),
        ridge_lambda=base_lambda,
    )
    q2_qb, q2_defense = solve_qb_stat_ridge(
        qb_games,
        response_col=response_col,
        ridge_lambda=base_lambda,
        defense_ridge_lambda=0.0,
    )

    current_case = _variant_frame("current", current_qb, current_defense)
    fixed_case = _variant_frame(
        "fixed_team_defense",
        fixed_qb,
        team_defense.select(["team", "defense_rating"]),
    )
    q2_case = _variant_frame("lighter_defense_penalty_x0", q2_qb, q2_defense)
    return pl.concat([current_case, fixed_case, q2_case], how="vertical")


__all__ = [
    "build_qb_split_half_frame",
    "build_qb_adjustment_audit_frame",
    "build_qb_leverage_profile_frame",
    "build_qb_opponent_offense_frame",
    "compute_qb_case_study",
    "compute_qb_defense_spread_summary",
    "compute_qb_leverage_diagnostics",
    "compute_qb_metric_stability_from_history",
    "compute_qb_opponent_offense_diagnostics",
    "compute_qb_experiment_sweep",
    "compute_qb_playoff_season_summary",
    "compute_qb_playoff_validation_frame",
    "compute_qb_season_audit_summary",
    "compute_qb_split_half_diagnostics",
    "compute_season_mae_deltas",
    "compute_weekly_mae_curves",
    "evaluate_qb_split_half_decision",
    "summarize_qb_split_half_signal",
    "summarize_defense_spread",
    "summarize_qb_adjustment_slopes",
    "summarize_qb_leverage_signal",
    "summarize_qb_opponent_offense_signal",
]
