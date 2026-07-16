"""Stage 3b diagnostics helpers for team and quarterback validation analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

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
    rows.append(_summarize_variant("q1_fixed_team_defense", fixed_qb, fixed_schedule))

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
                **_summarize_variant(f"q2_defense_penalty_x{multiplier:g}", qb_ratings, schedule),
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
    """Return the Maye/Stafford-style comparison for current, Q1, and the strongest Q2 trial."""
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
        "q1_fixed_team_defense",
        fixed_qb,
        team_defense.select(["team", "defense_rating"]),
    )
    q2_case = _variant_frame("q2_defense_penalty_x0", q2_qb, q2_defense)
    return pl.concat([current_case, fixed_case, q2_case], how="vertical")


__all__ = [
    "build_qb_adjustment_audit_frame",
    "compute_qb_case_study",
    "compute_qb_defense_spread_summary",
    "compute_qb_experiment_sweep",
    "compute_qb_season_audit_summary",
    "compute_season_mae_deltas",
    "compute_weekly_mae_curves",
    "summarize_defense_spread",
    "summarize_qb_adjustment_slopes",
]
