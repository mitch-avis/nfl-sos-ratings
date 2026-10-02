"""Walk-forward validation helpers for team margin prediction."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import load_espn_qbr, load_pbp_data
from nfl_sos_ratings.simultaneous_adjustment import solve_srs
from nfl_sos_ratings.validation import history_strings
from nfl_sos_ratings.validation.baselines import (
    PLAY_LEVEL_EPA_ST_BASELINE,
    ROLLING_EPA_BASELINE,
    ROLLING_EPA_ST_BASELINE,
)
from nfl_sos_ratings.validation.diagnostics import (
    DEFAULT_BOOTSTRAP,
    BootstrapSettings,
    compute_qb_case_study,
    compute_qb_defense_spread_summary,
    compute_qb_designed_rush_preview,
    compute_qb_experiment_sweep,
    compute_qb_leverage_diagnostics,
    compute_qb_metric_stability_from_history,
    compute_qb_opponent_offense_diagnostics,
    compute_qb_playoff_validation_frame,
    compute_qb_schedule_lens_anchor,
    compute_qb_schedule_lens_divergence,
    compute_qb_schedule_lens_trace,
    compute_qb_season_audit_summary,
    compute_qb_split_half_diagnostics,
    compute_season_mae_deltas,
    compute_weekly_mae_curves,
    evaluate_qb_split_half_decision,
    summarize_qb_split_half_signal,
)
from nfl_sos_ratings.validation.report import (
    ValidationReportInputs,
    write_validation_report,
)
from nfl_sos_ratings.validation.snapshots import (
    build_play_level_team_adjusted_snapshot,
    build_play_level_team_frame_from_pbp,
    build_special_teams_game_frame_from_pbp,
    build_special_teams_rating_snapshot,
    build_team_adjusted_snapshot,
    build_team_rating_snapshot,
    build_team_weighted_rating_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_NORMALIZED_NAME_RE = re.compile(r"[^a-z0-9]+")
_PLAY_LEVEL_EPA_ST_COLUMN = "play_level_weighted_team_rating"
_TEAM_T1_FEATURE_COLUMNS = [
    "adj_off_passing_epa_per_offensive_snap",
    "adj_off_rushing_epa_per_offensive_snap",
    "adj_def_passing_epa_per_offensive_snap",
    "adj_def_rushing_epa_per_offensive_snap",
]
_TEAM_T2_FEATURE_COLUMNS = [*_TEAM_T1_FEATURE_COLUMNS, "st_rating"]
# A correlation needs at least two paired observations.
_MIN_CORRELATION_SAMPLES = 2
# ESPN QBR (and the CPOE-bearing QSaCR it is compared against) starts in 2006.
_QBR_FIRST_SEASON = 2006
# Weeks before this one form the "early" evaluation split; the rest form "late".
_LATE_SEASON_FIRST_WEEK = 8


@dataclass(frozen=True, slots=True)
class EloConfig:
    """Fixed constants for the simple team Elo baseline."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_field_elo: float = 55.0
    regression_to_mean: float = 2.0 / 3.0
    use_margin_multiplier: bool = True


def _normalize_person_name(name: str) -> str:
    """Normalize a player name for fuzzy season-end joins across sources."""
    return _NORMALIZED_NAME_RE.sub("", name.lower())


def _pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson correlation, or 0 when either series is constant."""
    if min(x_values.size, y_values.size) < _MIN_CORRELATION_SAMPLES:
        return 0.0
    x_centered = x_values - float(x_values.mean())
    y_centered = y_values - float(y_values.mean())
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
        return 0.0
    return float((x_centered @ y_centered) / (x_norm * y_norm))


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Spearman rank correlation using average ranks for ties."""
    x_ranks = np.asarray(pl.Series(x_values).rank(method="average").to_list(), dtype=np.float64)
    y_ranks = np.asarray(pl.Series(y_values).rank(method="average").to_list(), dtype=np.float64)
    return _pearson(x_ranks, y_ranks)


def _weighted_pearson(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted Pearson correlation, or 0.0 when undefined."""
    if min(x_values.size, y_values.size, weights.size) < _MIN_CORRELATION_SAMPLES:
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


def _weighted_spearman(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted Spearman correlation using average ranks for ties."""
    x_ranks = np.asarray(pl.Series(x_values).rank(method="average").to_list(), dtype=np.float64)
    y_ranks = np.asarray(pl.Series(y_values).rank(method="average").to_list(), dtype=np.float64)
    return _weighted_pearson(x_ranks, y_ranks, weights)


def _bootstrap_weighted_correlation_interval(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
    *,
    correlation_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    bootstrap: BootstrapSettings,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for a weighted correlation."""
    if x_values.size <= 1 or y_values.size <= 1 or weights.size <= 1:
        point_estimate = correlation_fn(x_values, y_values, weights)
        return point_estimate, point_estimate

    rng = np.random.default_rng(bootstrap.seed)
    bootstrap_values = np.empty(bootstrap.resamples, dtype=np.float64)
    for sample_index in range(bootstrap.resamples):
        indices = rng.integers(0, x_values.size, size=x_values.size)
        bootstrap_values[sample_index] = correlation_fn(
            x_values[indices],
            y_values[indices],
            weights[indices],
        )
    return float(np.quantile(bootstrap_values, 0.025)), float(np.quantile(bootstrap_values, 0.975))


def compute_playoff_metric_correlations(
    playoff_summary: pl.DataFrame,
    regular_metrics: pl.DataFrame,
    *,
    metric_columns: Sequence[str],
    bootstrap: BootstrapSettings = DEFAULT_BOOTSTRAP,
) -> pl.DataFrame:
    """Return per-season and pooled weighted playoff validation correlations by metric.

    Each metric is correlated with playoff ridge-adjusted EPA per dropback, weighted by playoff
    dropbacks.
    """
    target_col, weight_col = "playoff_adjusted_epa_per_dropback", "playoff_dropbacks"
    join_keys = [
        column
        for column in ("season", "qb_id", "qb_name", "team")
        if column in playoff_summary.columns and column in regular_metrics.columns
    ]
    if not join_keys:
        return pl.DataFrame()

    joined = playoff_summary.join(
        regular_metrics.select(
            [
                *join_keys,
                *[column for column in metric_columns if column in regular_metrics.columns],
            ]
        ),
        on=join_keys,
        how="left",
    )
    rows: list[dict[str, object]] = []

    def summarize_frame(frame: pl.DataFrame, *, season_label: str) -> None:
        for metric in metric_columns:
            if metric not in frame.columns:
                continue
            metric_frame = frame.drop_nulls([metric, target_col, weight_col])
            if metric_frame.is_empty():
                continue
            x_values = np.asarray(
                metric_frame.select(metric).to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            y_values = np.asarray(
                metric_frame.select(target_col).to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            weights = np.asarray(
                metric_frame.select(pl.col(weight_col).cast(pl.Float64)).to_series().to_list(),
                dtype=np.float64,
            )
            spearman = _weighted_spearman(x_values, y_values, weights)
            pearson = _weighted_pearson(x_values, y_values, weights)
            row_seed = bootstrap.seed + len(rows)
            spearman_ci_lower, spearman_ci_upper = _bootstrap_weighted_correlation_interval(
                x_values,
                y_values,
                weights,
                correlation_fn=_weighted_spearman,
                bootstrap=BootstrapSettings(bootstrap.resamples, row_seed),
            )
            pearson_ci_lower, pearson_ci_upper = _bootstrap_weighted_correlation_interval(
                x_values,
                y_values,
                weights,
                correlation_fn=_weighted_pearson,
                bootstrap=BootstrapSettings(bootstrap.resamples, row_seed + 10_000),
            )
            rows.append(
                {
                    "season_label": season_label,
                    "metric": metric,
                    "qb_seasons": int(metric_frame.height),
                    "playoff_dropbacks": float(weights.sum()),
                    "spearman": spearman,
                    "spearman_ci_lower": spearman_ci_lower,
                    "spearman_ci_upper": spearman_ci_upper,
                    "pearson": pearson,
                    "pearson_ci_lower": pearson_ci_lower,
                    "pearson_ci_upper": pearson_ci_upper,
                }
            )

    if "season" in joined.columns:
        for season_key, frame in joined.group_by("season", maintain_order=True):
            season_value = season_key[0]
            summarize_frame(frame, season_label=str(season_value))
    summarize_frame(joined, season_label="pooled")
    return pl.DataFrame(rows).sort(["season_label", "metric"])


def _build_home_game_frame(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one row per home game with the realized home margin."""
    required_columns = {"game_id", "week", "team", "opponent_team", "is_home", "point_margin"}
    missing = sorted(required_columns - set(weekly_team_rows.columns))
    if missing:
        detail = ", ".join(missing)
        msg = f"weekly_team_rows is missing required columns: {detail}"
        raise ValueError(msg)

    return (
        weekly_team_rows.filter(pl.col("is_home"))
        .select(
            pl.lit(season).cast(pl.Int64).alias("season"),
            "game_id",
            pl.col("week").cast(pl.Int64).alias("week"),
            pl.col("team").alias("home_team"),
            pl.col("opponent_team").alias("away_team"),
            pl.col("point_margin").cast(pl.Float64).alias("home_margin"),
        )
        .sort(["week", "game_id"])
    )


def build_snapshot_feature_rows(
    weekly_team_rows: pl.DataFrame,
    season: int,
    baseline_name: str,
    snapshot_builder: Callable[[pl.DataFrame, int], pl.DataFrame],
    rating_column: str,
) -> pl.DataFrame:
    """Build week-specific home-game feature rows from a pregame team snapshot."""
    home_games = _build_home_game_frame(weekly_team_rows, season)
    if home_games.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "baseline": pl.String,
                "game_id": pl.String,
                "home_team": pl.String,
                "away_team": pl.String,
                "rating_diff": pl.Float64,
                "home_margin": pl.Float64,
            }
        )

    feature_frames: list[pl.DataFrame] = []
    weeks = sorted(home_games.select("week").to_series().unique().to_list())
    for week in weeks:
        snapshot = snapshot_builder(weekly_team_rows, int(week))
        ratings = snapshot.select(["team", rating_column]).rename({rating_column: "rating"})
        week_games = home_games.filter(pl.col("week") == int(week))
        week_features = (
            week_games.join(
                ratings.rename({"team": "home_team", "rating": "home_rating"}),
                on="home_team",
                how="left",
            )
            .join(
                ratings.rename({"team": "away_team", "rating": "away_rating"}),
                on="away_team",
                how="left",
            )
            .with_columns(
                pl.lit(baseline_name).alias("baseline"),
                (pl.col("home_rating").fill_null(0.0) - pl.col("away_rating").fill_null(0.0)).alias(
                    "rating_diff"
                ),
            )
            .select(
                "season",
                "week",
                "baseline",
                "game_id",
                "home_team",
                "away_team",
                "rating_diff",
                "home_margin",
            )
        )
        feature_frames.append(week_features)

    return pl.concat(feature_frames, how="vertical") if feature_frames else home_games.clear()


def _empty_team_value_snapshot(
    weekly_team_rows: pl.DataFrame,
    rating_column: str,
) -> pl.DataFrame:
    """Return a zeroed snapshot for every team present in the weekly rows."""
    teams = sorted(
        weekly_team_rows.select("team").drop_nulls().to_series().cast(pl.String).unique().to_list()
    )
    return pl.DataFrame({"team": teams, rating_column: [0.0] * len(teams)})


def _build_srs_snapshot(weekly_team_rows: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
    """Return an SRS snapshot built only from games before the cutoff week."""
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        return _empty_team_value_snapshot(weekly_team_rows, "SRS")
    return solve_srs(filtered_rows, response_col="point_margin").rename({"srs_rating": "SRS"})


def build_srs_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from week-specific SRS snapshots."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="SRS",
        snapshot_builder=_build_srs_snapshot,
        rating_column="SRS",
    )


def _build_raw_epa_snapshot(weekly_team_rows: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
    """Return pre-cutoff mean raw EPA margin per play for every team."""
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        return _empty_team_value_snapshot(weekly_team_rows, "raw_epa_margin")
    return (
        filtered_rows.group_by("team")
        .agg(pl.col("epa_margin_per_play").mean().alias("raw_epa_margin"))
        .sort("team")
    )


def build_raw_epa_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from raw pregame EPA margin means."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="RawEPA",
        snapshot_builder=_build_raw_epa_snapshot,
        rating_column="raw_epa_margin",
    )


def build_saovr_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from week-specific SaOvR snapshots."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="SaOvR",
        snapshot_builder=build_team_rating_snapshot,
        rating_column="SaOvR",
    )


def build_weighted_team_feature_rows(
    weekly_team_rows: pl.DataFrame,
    season: int,
    weight_map: dict[str, float],
    baseline_name: str = ROLLING_EPA_BASELINE,
) -> pl.DataFrame:
    """Build walk-forward rows from a weighted pregame team component snapshot."""
    weighted_column = "weighted_team_rating"
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name=baseline_name,
        snapshot_builder=lambda frame, cutoff_week: build_team_weighted_rating_snapshot(
            frame,
            cutoff_week=cutoff_week,
            weight_map=weight_map,
            output_col=weighted_column,
        ),
        rating_column=weighted_column,
    )


def _zscore_values(values: np.ndarray) -> np.ndarray:
    """Return sample-standardized values for walk-forward component weighting."""
    if values.size == 0:
        return values
    centered = values - float(values.mean())
    if values.size == 1:
        return centered
    std = float(values.std(ddof=1))
    return centered / std if std > 0.0 else centered


def _build_weighted_rating_from_frame(
    component_frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    weight_map: dict[str, float],
    output_col: str,
) -> pl.DataFrame:
    """Return one weighted, z-scored team rating from standardized component columns."""
    weighted_values = np.zeros(component_frame.height, dtype=np.float64)
    for column in feature_columns:
        if column not in component_frame.columns:
            continue
        component_values = np.asarray(
            component_frame.select(column).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        weighted_values += _zscore_values(component_values) * float(weight_map.get(column, 0.0))

    return pl.DataFrame(
        {
            "team": component_frame.select("team").to_series().cast(pl.String).to_list(),
            output_col: np.round(_zscore_values(weighted_values), 6).tolist(),
        }
    ).sort("team")


def _standardize_feature_columns(
    component_frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    """Return a frame with the requested feature columns standardized within the frame."""
    result = component_frame
    for column in feature_columns:
        if column not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(column))
        values = np.asarray(
            result.select(column).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
            dtype=np.float64,
        )
        result = result.with_columns(pl.Series(column, _zscore_values(values)))
    return result


def build_weighted_team_special_teams_feature_rows(
    weekly_team_rows: pl.DataFrame,
    st_game_rows: pl.DataFrame,
    season: int,
    weight_map: dict[str, float],
    baseline_name: str = ROLLING_EPA_ST_BASELINE,
) -> pl.DataFrame:
    """Build walk-forward rows from a weighted team-plus-special-teams snapshot."""
    weighted_column = "weighted_team_rating"

    def snapshot_builder(frame: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
        adjusted_snapshot = build_team_adjusted_snapshot(frame, cutoff_week=cutoff_week)
        st_snapshot = build_special_teams_rating_snapshot(st_game_rows, cutoff_week=cutoff_week)
        merged = adjusted_snapshot.join(st_snapshot, on="team", how="left").with_columns(
            pl.col("st_rating").fill_null(0.0)
        )
        return _build_weighted_rating_from_frame(
            merged,
            feature_columns=_TEAM_T2_FEATURE_COLUMNS,
            weight_map=weight_map,
            output_col=weighted_column,
        )

    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name=baseline_name,
        snapshot_builder=snapshot_builder,
        rating_column=weighted_column,
    )


def build_play_level_weighted_team_special_teams_feature_rows(
    weekly_team_rows: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    season: int,
    weight_map: dict[str, float],
    baseline_name: str = PLAY_LEVEL_EPA_ST_BASELINE,
) -> pl.DataFrame:
    """Build walk-forward rows from a play-level weighted team-plus-special-teams snapshot."""
    weighted_column = "weighted_team_rating"
    play_rows = build_play_level_team_frame_from_pbp(pbp)
    st_game_rows = build_special_teams_game_frame_from_pbp(pbp)

    def snapshot_builder(frame: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
        del frame
        adjusted_snapshot = build_play_level_team_adjusted_snapshot(
            play_rows,
            cutoff_week=cutoff_week,
        )
        st_snapshot = build_special_teams_rating_snapshot(st_game_rows, cutoff_week=cutoff_week)
        merged = adjusted_snapshot.join(st_snapshot, on="team", how="left").with_columns(
            pl.col("st_rating").fill_null(0.0)
        )
        return _build_weighted_rating_from_frame(
            merged,
            feature_columns=_TEAM_T2_FEATURE_COLUMNS,
            weight_map=weight_map,
            output_col=weighted_column,
        )

    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name=baseline_name,
        snapshot_builder=snapshot_builder,
        rating_column=weighted_column,
    )


def _equal_weight_map(feature_columns: Sequence[str]) -> dict[str, float]:
    """Return equal weights over an arbitrary feature-column list."""
    equal_weight = 1.0 / len(feature_columns)
    return dict.fromkeys(feature_columns, equal_weight)


def _normalize_feature_weight_map(
    weight_map: dict[str, float],
    feature_columns: Sequence[str],
) -> dict[str, float]:
    """Normalize a fitted feature weight map by absolute weight while preserving signs."""
    total_abs_weight = float(sum(abs(weight_map.get(column, 0.0)) for column in feature_columns))
    if total_abs_weight <= 0.0:
        return _equal_weight_map(feature_columns)
    return {
        column: float(weight_map.get(column, 0.0) / total_abs_weight) for column in feature_columns
    }


def _build_rolling_feature_weight_maps(
    training_rows: pl.DataFrame,
    seasons: Sequence[int],
    feature_columns: Sequence[str],
) -> dict[int, dict[str, float]]:
    """Fit one rolling weight map per season for the requested feature set."""
    weight_maps: dict[int, dict[str, float]] = {}
    equal_weight_map = _equal_weight_map(feature_columns)

    for season in sorted(seasons):
        prior_rows = training_rows.filter(pl.col("next_season") < int(season))
        if prior_rows.is_empty():
            weight_maps[int(season)] = dict(equal_weight_map)
            continue

        fitted_weights = composite_weights.fit_linear_weights(
            prior_rows,
            feature_columns=list(feature_columns),
            target_column="target",
        )
        weight_maps[int(season)] = _normalize_feature_weight_map(
            fitted_weights,
            feature_columns,
        )

    return weight_maps


def build_rolling_team_weight_maps(
    training_rows: pl.DataFrame,
    seasons: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Fit one rolling team component weight map per season using only prior season pairs."""
    return _build_rolling_feature_weight_maps(training_rows, seasons, _TEAM_T1_FEATURE_COLUMNS)


def _elo_margin_multiplier(rating_gap: float, home_margin: float) -> float:
    """Return a standard logarithmic margin-of-victory Elo multiplier."""
    return float(np.log(abs(home_margin) + 1.0) * (2.2 / ((abs(rating_gap) * 0.001) + 2.2)))


def build_elo_feature_rows(
    home_games: pl.DataFrame,
    config: EloConfig | None = None,
) -> pl.DataFrame:
    """Build one-row-per-game pregame Elo features from chronological home games."""
    resolved_config = config or EloConfig()
    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows: list[dict[str, object]] = []

    sorted_games = home_games.sort(["season", "week", "game_id"])
    for row in sorted_games.iter_rows(named=True):
        season = int(row["season"])
        if current_season is not None and season != current_season:
            for team, rating in list(ratings.items()):
                ratings[team] = (
                    resolved_config.initial_rating
                    + (rating - resolved_config.initial_rating) * resolved_config.regression_to_mean
                )
        current_season = season

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_rating = ratings.get(home_team, resolved_config.initial_rating)
        away_rating = ratings.get(away_team, resolved_config.initial_rating)
        rating_diff = home_rating - away_rating
        home_margin = float(row["home_margin"])

        rows.append(
            {
                "season": season,
                "week": int(row["week"]),
                "baseline": "Elo",
                "game_id": str(row["game_id"]),
                "home_team": home_team,
                "away_team": away_team,
                "rating_diff": rating_diff,
                "home_margin": home_margin,
            }
        )

        expected_home = 1.0 / (
            1.0 + 10.0 ** (-(rating_diff + resolved_config.home_field_elo) / 400.0)
        )
        actual_home = 1.0 if home_margin > 0.0 else 0.0 if home_margin < 0.0 else 0.5
        multiplier = (
            _elo_margin_multiplier(rating_diff, home_margin)
            if resolved_config.use_margin_multiplier and home_margin != 0.0
            else 1.0
        )
        delta = resolved_config.k_factor * multiplier * (actual_home - expected_home)
        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    return pl.DataFrame(rows).sort(["season", "week", "game_id"])


def run_walk_forward_backtest(
    data_dir: Path,
    seasons: list[int],
    start_week: int = 5,
    elo_config: EloConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full team walk-forward backtest across all requested seasons."""
    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        home_games = _build_home_game_frame(weekly_team_rows, season)
        feature_frames.extend(
            [
                build_saovr_feature_rows(weekly_team_rows, season),
                build_srs_feature_rows(weekly_team_rows, season),
                build_raw_epa_feature_rows(weekly_team_rows, season),
                build_elo_feature_rows(home_games, config=elo_config),
            ]
        )

    if not feature_frames:
        empty_predictions = evaluate_feature_rows(pl.DataFrame(), start_week=start_week)
        return empty_predictions, score_prediction_rows(empty_predictions)

    all_features = pl.concat(feature_frames, how="vertical")
    predictions = evaluate_feature_rows(all_features, start_week=start_week)
    return predictions, score_prediction_rows(predictions)


def run_weighted_team_backtest(
    data_dir: Path,
    seasons: list[int],
    start_week: int = 5,
    baseline_name: str = ROLLING_EPA_BASELINE,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the rolling weighted-team walk-forward backtest with prior-season weights."""
    training_rows = composite_weights.build_team_training_rows(data_dir, seasons)
    weight_maps = build_rolling_team_weight_maps(training_rows, seasons)

    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        feature_frames.append(
            build_weighted_team_feature_rows(
                weekly_team_rows,
                season=season,
                weight_map=weight_maps[int(season)],
                baseline_name=baseline_name,
            )
        )

    if not feature_frames:
        empty_predictions = evaluate_feature_rows(pl.DataFrame(), start_week=start_week)
        return empty_predictions, score_prediction_rows(empty_predictions), weight_maps

    predictions = evaluate_feature_rows(
        pl.concat(feature_frames, how="vertical"), start_week=start_week
    )
    return predictions, score_prediction_rows(predictions), weight_maps


def build_team_training_rows_with_special_teams(
    data_dir: Path,
    seasons: Sequence[int],
) -> pl.DataFrame:
    """Build rolling team training rows augmented with a full-season special-teams rating."""
    base_rows = composite_weights.build_team_training_rows(data_dir, seasons).select(
        "season",
        "next_season",
        "team",
        *_TEAM_T1_FEATURE_COLUMNS,
        "target",
    )
    st_rows: list[pl.DataFrame] = []
    for season in sorted(seasons):
        pbp = load_pbp_data(int(season))
        st_game_rows = build_special_teams_game_frame_from_pbp(pbp)
        st_snapshot = build_special_teams_rating_snapshot(st_game_rows, cutoff_week=100)
        if st_snapshot.is_empty():
            continue
        st_values = np.asarray(
            st_snapshot.select("st_rating").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        st_rows.append(
            st_snapshot.with_columns(
                pl.lit(int(season)).cast(pl.Int64).alias("season"),
                pl.Series("st_rating", _zscore_values(st_values)),
            )
        )

    if not st_rows:
        return base_rows.with_columns(pl.lit(0.0).alias("st_rating"))

    st_frame = pl.concat(st_rows, how="diagonal_relaxed")
    return base_rows.join(
        st_frame.select("season", "team", "st_rating"),
        on=["season", "team"],
        how="left",
    ).with_columns(pl.col("st_rating").fill_null(0.0))


def build_play_level_team_training_rows_with_special_teams(
    data_dir: Path,
    seasons: Sequence[int],
) -> pl.DataFrame:
    """Build rolling play-level training rows from EPA components plus special teams."""
    season_list = sorted(seasons)
    rows: list[pl.DataFrame] = []

    for season, next_season in pairwise(season_list):
        pbp = load_pbp_data(int(season))
        play_rows = build_play_level_team_frame_from_pbp(pbp)
        if play_rows.is_empty():
            continue

        cutoff_week = int(play_rows.select(pl.col("week").max()).item()) + 1
        adjusted_snapshot = build_play_level_team_adjusted_snapshot(
            play_rows,
            cutoff_week=cutoff_week,
        )
        st_game_rows = build_special_teams_game_frame_from_pbp(pbp)
        st_snapshot = build_special_teams_rating_snapshot(st_game_rows, cutoff_week=cutoff_week)
        features = adjusted_snapshot.join(st_snapshot, on="team", how="left").with_columns(
            pl.col("st_rating").fill_null(0.0)
        )
        standardized_features = _standardize_feature_columns(
            features,
            feature_columns=_TEAM_T2_FEATURE_COLUMNS,
        )
        targets = pl.read_parquet(data_dir / f"{next_season}_combined.parquet").select(
            pl.col("team").cast(pl.String),
            pl.col("SaOvR").cast(pl.Float64).fill_null(0.0).alias("target"),
        )
        rows.append(
            standardized_features.join(targets, on="team", how="inner")
            .with_columns(
                pl.lit(int(season)).cast(pl.Int64).alias("season"),
                pl.lit(int(next_season)).cast(pl.Int64).alias("next_season"),
            )
            .select("season", "next_season", "team", *_TEAM_T2_FEATURE_COLUMNS, "target")
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "next_season": pl.Int64,
                "team": pl.String,
                **dict.fromkeys(_TEAM_T2_FEATURE_COLUMNS, pl.Float64),
                "target": pl.Float64,
            }
        )

    return pl.concat(rows, how="vertical_relaxed")


def run_weighted_team_special_teams_backtest(
    data_dir: Path,
    seasons: list[int],
    start_week: int = 5,
    baseline_name: str = ROLLING_EPA_ST_BASELINE,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the weighted-team walk-forward backtest with a special-teams component."""
    training_rows = build_team_training_rows_with_special_teams(data_dir, seasons)
    weight_maps = _build_rolling_feature_weight_maps(
        training_rows,
        seasons,
        _TEAM_T2_FEATURE_COLUMNS,
    )

    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        st_game_rows = build_special_teams_game_frame_from_pbp(load_pbp_data(int(season)))
        feature_frames.append(
            build_weighted_team_special_teams_feature_rows(
                weekly_team_rows,
                st_game_rows=st_game_rows,
                season=season,
                weight_map=weight_maps[int(season)],
                baseline_name=baseline_name,
            )
        )

    if not feature_frames:
        empty_predictions = evaluate_feature_rows(pl.DataFrame(), start_week=start_week)
        return empty_predictions, score_prediction_rows(empty_predictions), weight_maps

    predictions = evaluate_feature_rows(
        pl.concat(feature_frames, how="vertical"), start_week=start_week
    )
    return predictions, score_prediction_rows(predictions), weight_maps


def run_play_level_team_special_teams_backtest(
    data_dir: Path,
    seasons: list[int],
    start_week: int = 5,
    baseline_name: str = PLAY_LEVEL_EPA_ST_BASELINE,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the play-level weighted-team backtest with the same ST component as the rolling blend."""
    training_rows = build_play_level_team_training_rows_with_special_teams(data_dir, seasons)
    weight_maps = _build_rolling_feature_weight_maps(
        training_rows,
        seasons,
        _TEAM_T2_FEATURE_COLUMNS,
    )

    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        feature_frames.append(
            build_play_level_weighted_team_special_teams_feature_rows(
                weekly_team_rows,
                load_pbp_data(int(season)),
                season=season,
                weight_map=weight_maps[int(season)],
                baseline_name=baseline_name,
            )
        )

    if not feature_frames:
        empty_predictions = evaluate_feature_rows(pl.DataFrame(), start_week=start_week)
        return empty_predictions, score_prediction_rows(empty_predictions), weight_maps

    predictions = evaluate_feature_rows(
        pl.concat(feature_frames, how="vertical"), start_week=start_week
    )
    return predictions, score_prediction_rows(predictions), weight_maps


def build_play_level_team_season_ratings(
    seasons: Sequence[int],
    weight_maps: dict[int, dict[str, float]],
    output_col: str = _PLAY_LEVEL_EPA_ST_COLUMN,
) -> pl.DataFrame:
    """Build one final full-season play-level weighted rating row per team and season."""
    season_frames: list[pl.DataFrame] = []

    for season in sorted(seasons):
        if int(season) not in weight_maps:
            continue
        pbp = load_pbp_data(int(season))
        play_rows = build_play_level_team_frame_from_pbp(pbp)
        if play_rows.is_empty():
            continue
        st_rows = build_special_teams_game_frame_from_pbp(pbp)
        cutoff_week = int(play_rows.select(pl.col("week").max()).item()) + 1
        adjusted_snapshot = build_play_level_team_adjusted_snapshot(
            play_rows,
            cutoff_week=cutoff_week,
        )
        st_snapshot = build_special_teams_rating_snapshot(st_rows, cutoff_week=cutoff_week)
        merged = adjusted_snapshot.join(st_snapshot, on="team", how="left").with_columns(
            pl.col("st_rating").fill_null(0.0)
        )
        weighted = _build_weighted_rating_from_frame(
            merged,
            feature_columns=_TEAM_T2_FEATURE_COLUMNS,
            weight_map=weight_maps[int(season)],
            output_col=output_col,
        ).with_columns(pl.lit(int(season)).cast(pl.Int64).alias("season"))
        season_frames.append(weighted.select("season", "team", output_col))

    if not season_frames:
        return pl.DataFrame(schema={"season": pl.Int64, "team": pl.String, output_col: pl.Float64})

    return pl.concat(season_frames, how="vertical_relaxed").sort(["season", "team"])


def compute_team_rating_stability_from_history(
    rating_history: pl.DataFrame,
    rating_column: str,
) -> dict[str, float | int] | None:
    """Compute adjacent-season team stability for an arbitrary season-by-team rating history."""
    if rating_history.is_empty() or rating_column not in rating_history.columns:
        return None

    seasons = sorted(rating_history.select("season").to_series().cast(pl.Int64).unique().to_list())
    pair_frames: list[pl.DataFrame] = []
    for season, next_season in pairwise(seasons):
        current = rating_history.filter(pl.col("season") == int(season)).select(
            "team", pl.col(rating_column).alias("rating_t")
        )
        nxt = rating_history.filter(pl.col("season") == int(next_season)).select(
            "team", pl.col(rating_column).alias("rating_t1")
        )
        joined = current.join(nxt, on="team", how="inner")
        if not joined.is_empty():
            pair_frames.append(joined)

    if not pair_frames:
        return None

    paired = pl.concat(pair_frames, how="vertical_relaxed")
    x_values = np.asarray(
        paired.select("rating_t").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    y_values = np.asarray(
        paired.select("rating_t1").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    return {
        "paired_rows": len(x_values),
        "pearson": _pearson(x_values, y_values),
        "spearman": _spearman(x_values, y_values),
    }


def build_team_decision_lines(
    metrics: pl.DataFrame,
    mae_deltas: pl.DataFrame,
    *,
    base_team_stability: dict[str, object] | None,
    t4_team_stability: dict[str, float | int] | None,
) -> list[str]:
    """Build the team decision-rule and outcome narrative for the validation report."""
    return history_strings.build_team_report_decision_lines(
        metrics,
        mae_deltas,
        base_team_stability=base_team_stability,
        t4_team_stability=t4_team_stability,
        baselines=(ROLLING_EPA_ST_BASELINE, PLAY_LEVEL_EPA_ST_BASELINE),
    )


def build_qb_status_lines() -> list[str]:
    """Return the current quarterback-methodology status note for the report."""
    return history_strings.build_qb_report_status_lines()


def compute_stability_metrics(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute adjacent-season Pearson and Spearman stability for teams and QBs."""
    sorted_seasons = sorted(seasons)
    qb_metric_columns = ("QSaCR", "qb_passer_rating", "qb_any_a")
    qb_pairs: list[pl.DataFrame] = []
    team_pairs: list[pl.DataFrame] = []

    for season, next_season in pairwise(sorted_seasons):
        current_qb = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        next_qb = pl.read_parquet(data_dir / f"{next_season}_qb_combined.parquet")
        if "qb_is_eligible" in current_qb.columns:
            current_qb = current_qb.filter(pl.col("qb_is_eligible"))
        if "qb_is_eligible" in next_qb.columns:
            next_qb = next_qb.filter(pl.col("qb_is_eligible"))

        current_qb = current_qb.select(["qb_id", *qb_metric_columns]).rename(
            {column: f"{column}_t" for column in qb_metric_columns}
        )
        next_qb = next_qb.select(["qb_id", *qb_metric_columns]).rename(
            {column: f"{column}_t1" for column in qb_metric_columns}
        )
        qb_pairs.append(current_qb.join(next_qb, on="qb_id", how="inner"))

        current_team = pl.read_parquet(data_dir / f"{season}_combined.parquet").select(
            pl.col("team"), pl.col("SaOvR").alias("SaOvR_t")
        )
        next_team = pl.read_parquet(data_dir / f"{next_season}_combined.parquet").select(
            pl.col("team"), pl.col("SaOvR").alias("SaOvR_t1")
        )
        team_pairs.append(current_team.join(next_team, on="team", how="inner"))

    rows: list[dict[str, object]] = []
    if qb_pairs:
        qb_pair_frame = pl.concat(qb_pairs, how="diagonal_relaxed").drop_nulls(
            [
                "QSaCR_t",
                "QSaCR_t1",
                "qb_passer_rating_t",
                "qb_passer_rating_t1",
                "qb_any_a_t",
                "qb_any_a_t1",
            ]
        )
        for metric in qb_metric_columns:
            x_values = np.asarray(
                qb_pair_frame.select(f"{metric}_t").to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            y_values = np.asarray(
                qb_pair_frame.select(f"{metric}_t1").to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            rows.append(
                {
                    "entity": "qb",
                    "metric": metric,
                    "paired_rows": len(x_values),
                    "pearson": _pearson(x_values, y_values),
                    "spearman": _spearman(x_values, y_values),
                }
            )

    if team_pairs:
        team_pair_frame = pl.concat(team_pairs, how="diagonal_relaxed").drop_nulls(
            ["SaOvR_t", "SaOvR_t1"]
        )
        x_values = np.asarray(
            team_pair_frame.select("SaOvR_t").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            team_pair_frame.select("SaOvR_t1").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        rows.append(
            {
                "entity": "team",
                "metric": "SaOvR",
                "paired_rows": len(x_values),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort(["entity", "metric"])


def compute_qbr_correlations(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute per-season QSaCR versus ESPN QBR correlations on matched QBs."""
    eligible_seasons = sorted(season for season in seasons if season >= _QBR_FIRST_SEASON)
    if not eligible_seasons:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "joined_rows": pl.Int64,
                "pearson": pl.Float64,
                "spearman": pl.Float64,
            }
        )

    qbr_df = load_espn_qbr(level="season", seasons=eligible_seasons)
    if "game_week" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("game_week") == "Season Total")

    qbr_df = qbr_df.with_columns(
        pl.col("team_abb").cast(pl.String).alias("team"),
        pl.col("name_display")
        .cast(pl.String)
        .map_elements(_normalize_person_name, return_dtype=pl.String)
        .alias("normalized_name"),
    )

    rows: list[dict[str, object]] = []
    for season in eligible_seasons:
        qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        if "qb_is_eligible" in qb_combined.columns:
            qb_combined = qb_combined.filter(pl.col("qb_is_eligible"))
        qb_join_frame = qb_combined.with_columns(
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.col("qb_name")
            .cast(pl.String)
            .map_elements(_normalize_person_name, return_dtype=pl.String)
            .alias("normalized_name"),
        )
        joined = qb_join_frame.join(
            qbr_df.filter(pl.col("season") == season).select(
                "season",
                "team",
                "normalized_name",
                "qbr_total",
            ),
            on=["season", "team", "normalized_name"],
            how="inner",
        ).drop_nulls(["QSaCR", "qbr_total"])

        x_values = np.asarray(
            joined.select("QSaCR").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            joined.select("qbr_total").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        rows.append(
            {
                "season": season,
                "joined_rows": len(x_values),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort("season")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the walk-forward validation command."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings validate", description="Run the walk-forward validation suite."
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory holding historical Parquet artifacts.",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=START_YEAR,
        help="First season to evaluate.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=END_YEAR,
        help="Last season to evaluate.",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=5,
        help="First week to score in each season.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/validation-report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args(argv)


def _qb_playoff_correlations(
    qb_playoff_validation: pl.DataFrame, qb_leverage_history: pl.DataFrame
) -> pl.DataFrame:
    """Return playoff validation correlations for every regular-season QB metric available."""
    if not qb_leverage_history.is_empty() and not qb_playoff_validation.is_empty():
        join_keys = [
            column
            for column in ("season", "qb_id", "qb_name", "team")
            if column in qb_leverage_history.columns and column in qb_playoff_validation.columns
        ]
        qb_playoff_validation = qb_playoff_validation.join(
            qb_leverage_history.select([*join_keys, "moderate_leverage_adjusted_epa_per_dropback"]),
            on=join_keys,
            how="left",
        )
    return compute_playoff_metric_correlations(
        qb_playoff_validation.select(
            [
                column
                for column in (
                    "season",
                    "qb_id",
                    "qb_name",
                    "team",
                    "playoff_adjusted_epa_per_dropback",
                    "playoff_dropbacks",
                )
                if column in qb_playoff_validation.columns
            ]
        )
        if not qb_playoff_validation.is_empty()
        else pl.DataFrame(),
        qb_playoff_validation,
        metric_columns=[
            column
            for column in (
                "QSaCR",
                "QSaOR",
                "QRaw",
                "qb_passer_rating",
                "qb_any_a",
                "moderate_leverage_adjusted_epa_per_dropback",
                "vs_top_half_adjusted_epa_per_dropback",
            )
            if column in qb_playoff_validation.columns
        ],
    )


def _apply_leverage_companion_gate(
    qb_leverage_decision: dict[str, object],
    stability: pl.DataFrame,
    qb_leverage_history: pl.DataFrame,
    qb_playoff_correlations: pl.DataFrame,
) -> None:
    """Record whether the moderate-leverage companion clears its stability and playoff gates."""
    qb_variant_stability = compute_qb_metric_stability_from_history(
        qb_leverage_history,
        "moderate_leverage_adjusted_epa_per_dropback",
    )
    qscr_row = stability.filter((pl.col("entity") == "qb") & (pl.col("metric") == "QSaCR"))
    leverage_playoff_row = qb_playoff_correlations.filter(
        (pl.col("season_label") == "pooled")
        & (pl.col("metric") == "moderate_leverage_adjusted_epa_per_dropback")
    )
    qsaor_playoff_row = qb_playoff_correlations.filter(
        (pl.col("season_label") == "pooled") & (pl.col("metric") == "QSaOR")
    )
    stability_pass = bool(
        qb_variant_stability is not None
        and not qscr_row.is_empty()
        and float(qb_variant_stability["pearson"]) >= float(qscr_row.select("pearson").item())
        and float(qb_variant_stability["spearman"]) >= float(qscr_row.select("spearman").item())
    )
    playoff_pass = bool(
        not leverage_playoff_row.is_empty()
        and not qsaor_playoff_row.is_empty()
        and float(leverage_playoff_row.select("spearman").item())
        >= float(qsaor_playoff_row.select("spearman").item())
    )
    qb_leverage_decision.update(
        {
            "stability_pass": stability_pass,
            "playoff_pass": playoff_pass,
            "variant_stability": qb_variant_stability,
            "decision": "publish_companion" if stability_pass and playoff_pass else "not_supported",
        }
    )


def main(argv: list[str] | None = None) -> None:
    """Run the walk-forward validation suite and write the Markdown report."""
    args = _parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    data_dir = Path(args.data_dir)
    report_path = Path(args.report_path)
    command = (
        "uv run python -m nfl_sos_ratings.validation.walk_forward "
        f"--data-dir {args.data_dir} --start-season {args.start_season} "
        f"--end-season {args.end_season} --start-week {args.start_week} "
        f"--report-path {args.report_path}"
    )

    base_predictions, metrics = run_walk_forward_backtest(
        data_dir,
        seasons=seasons,
        start_week=args.start_week,
    )
    t1_predictions, t1_metrics, _t1_weights = run_weighted_team_backtest(
        data_dir,
        seasons=seasons,
        start_week=args.start_week,
    )
    t2_predictions, t2_metrics, _t2_weights = run_weighted_team_special_teams_backtest(
        data_dir,
        seasons=seasons,
        start_week=args.start_week,
    )
    t4_predictions, t4_metrics, t4_weights = run_play_level_team_special_teams_backtest(
        data_dir,
        seasons=seasons,
        start_week=args.start_week,
    )
    combined_predictions = pl.concat(
        [base_predictions, t1_predictions, t2_predictions, t4_predictions], how="vertical"
    )
    combined_metrics = pl.concat([metrics, t1_metrics, t2_metrics, t4_metrics], how="vertical")
    mae_deltas = compute_pairwise_mae_bootstrap(
        combined_predictions,
        baselines=[
            PLAY_LEVEL_EPA_ST_BASELINE,
            ROLLING_EPA_ST_BASELINE,
            ROLLING_EPA_BASELINE,
            "SRS",
            "RawEPA",
            "SaOvR",
        ],
    )
    weekly_curves = compute_weekly_mae_curves(combined_predictions).filter(
        pl.col("baseline").is_in(
            [
                "Elo",
                "SRS",
                "SaOvR",
                ROLLING_EPA_ST_BASELINE,
                PLAY_LEVEL_EPA_ST_BASELINE,
            ]
        )
    )
    saovr_vs_srs = compute_season_mae_deltas(combined_metrics, baseline_a="SaOvR", baseline_b="SRS")
    t2_vs_srs = compute_season_mae_deltas(
        combined_metrics, baseline_a=ROLLING_EPA_ST_BASELINE, baseline_b="SRS"
    )
    qb_split_half = compute_qb_split_half_diagnostics(data_dir, seasons)
    qb_split_half_primary = summarize_qb_split_half_signal(
        qb_split_half,
        residual_col="vs_top_half_residual",
        weight_col="vs_top_half_dropbacks",
    )
    qb_split_half_placebo = summarize_qb_split_half_signal(
        qb_split_half,
        residual_col="vs_bottom_half_residual",
        weight_col="vs_bottom_half_dropbacks",
    )
    qb_split_half_decision = evaluate_qb_split_half_decision(
        qb_split_half_primary,
        qb_split_half_placebo,
    )
    qb_split_half_cases = qb_split_half.filter(
        (pl.col("season") == seasons[-1])
        & pl.col("qb_name").is_in(["Drake Maye", "Matthew Stafford"])
    )
    qb_playoff_validation = compute_qb_playoff_validation_frame(
        data_dir,
        seasons,
        qb_split_half=qb_split_half,
    )
    (
        qb_opponent_offense_summary,
        qb_opponent_offense_cases,
        qb_opponent_offense_decision,
    ) = compute_qb_opponent_offense_diagnostics(data_dir, seasons)
    (
        qb_leverage_summary,
        qb_leverage_cases,
        qb_leverage_history,
        qb_leverage_decision,
    ) = compute_qb_leverage_diagnostics(data_dir, seasons)
    qb_playoff_correlations = _qb_playoff_correlations(qb_playoff_validation, qb_leverage_history)
    qb_season_audit = compute_qb_season_audit_summary(data_dir, seasons)
    qb_defense_spread = compute_qb_defense_spread_summary(data_dir, seasons)
    qb_experiment_sweep = compute_qb_experiment_sweep(data_dir, seasons[-1])
    qb_case_study = compute_qb_case_study(data_dir, seasons[-1])
    qb_schedule_anchor = compute_qb_schedule_lens_anchor(
        data_dir,
        seasons[-1],
        qb_names=["Drake Maye", "Tyler Shough", "Joe Flacco", "J.J. McCarthy"],
    )
    qb_schedule_trace = compute_qb_schedule_lens_trace(
        data_dir,
        seasons[-1],
        qb_names=["Drake Maye", "Tyler Shough", "Joe Flacco", "J.J. McCarthy"],
    )
    qb_lens_divergence = compute_qb_schedule_lens_divergence(data_dir, seasons[-1])
    qb_designed_rush_preview = compute_qb_designed_rush_preview(
        data_dir,
        seasons[-1],
        qb_names=["Drake Maye", "Lamar Jackson", "Josh Allen", "Matthew Stafford"],
    )
    stability = compute_stability_metrics(data_dir, seasons=seasons)
    _apply_leverage_companion_gate(
        qb_leverage_decision, stability, qb_leverage_history, qb_playoff_correlations
    )
    base_team_stability = stability.filter(
        (pl.col("entity") == "team") & (pl.col("metric") == "SaOvR")
    )
    t4_history = build_play_level_team_season_ratings(seasons, t4_weights)
    t4_team_stability = compute_team_rating_stability_from_history(
        t4_history,
        _PLAY_LEVEL_EPA_ST_COLUMN,
    )
    team_decision_lines = build_team_decision_lines(
        combined_metrics,
        mae_deltas,
        base_team_stability=base_team_stability.row(0, named=True)
        if not base_team_stability.is_empty()
        else None,
        t4_team_stability=t4_team_stability,
    )
    qb_open_status_lines = build_qb_status_lines()
    regression_note_lines = history_strings.report_regression_note_lines()
    qbr_correlations = compute_qbr_correlations(data_dir, seasons=seasons)
    write_validation_report(
        report_path,
        ValidationReportInputs(
            metrics=metrics,
            stability=stability,
            qbr_correlations=qbr_correlations,
            mae_deltas=mae_deltas,
            seasons=seasons,
            start_week=args.start_week,
            command=command,
            comparison_metrics=combined_metrics,
            weekly_curves=weekly_curves,
            saovr_vs_srs=saovr_vs_srs,
            t2_vs_srs=t2_vs_srs,
            qb_season_audit=qb_season_audit,
            qb_defense_spread=qb_defense_spread,
            qb_experiment_sweep=qb_experiment_sweep,
            qb_case_study=qb_case_study,
            qb_schedule_anchor=qb_schedule_anchor,
            qb_schedule_trace=qb_schedule_trace,
            qb_lens_divergence=qb_lens_divergence,
            qb_designed_rush_preview=qb_designed_rush_preview,
            team_decision_lines=team_decision_lines,
            qb_open_status_lines=qb_open_status_lines,
            regression_note_lines=regression_note_lines,
            qb_opponent_offense_summary=qb_opponent_offense_summary,
            qb_opponent_offense_cases=qb_opponent_offense_cases,
            qb_opponent_offense_decision=qb_opponent_offense_decision,
            qb_leverage_summary=qb_leverage_summary,
            qb_leverage_cases=qb_leverage_cases,
            qb_leverage_decision=qb_leverage_decision,
            qb_split_half_primary=qb_split_half_primary,
            qb_split_half_placebo=qb_split_half_placebo,
            qb_split_half_cases=qb_split_half_cases,
            qb_split_half_decision=qb_split_half_decision,
            qb_playoff_correlations=qb_playoff_correlations,
        ),
    )

    print(f"Wrote validation report to {report_path}")


def _fit_margin_projection(training_rows: pl.DataFrame) -> tuple[float, float]:
    """Fit ``home_margin = k * rating_diff + hfa_points`` on past games only."""
    if training_rows.is_empty():
        return 0.0, 0.0

    rating_diff = np.asarray(
        training_rows.select("rating_diff").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    home_margin = np.asarray(
        training_rows.select("home_margin").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    if len(training_rows) == 1 or np.isclose(rating_diff.std(ddof=0), 0.0):
        return 0.0, float(home_margin.mean())

    design = np.column_stack((rating_diff, np.ones(len(rating_diff), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, home_margin, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def evaluate_feature_rows(feature_rows: pl.DataFrame, start_week: int) -> pl.DataFrame:
    """Fit prior-only margin models and predict every game from ``start_week`` onward."""
    if feature_rows.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "baseline": pl.String,
                "game_id": pl.String,
                "home_team": pl.String,
                "away_team": pl.String,
                "rating_diff": pl.Float64,
                "home_margin": pl.Float64,
                "predicted_margin": pl.Float64,
                "error": pl.Float64,
                "training_row_count": pl.Int64,
                "fitted_k": pl.Float64,
                "fitted_hfa_points": pl.Float64,
            }
        )

    prediction_rows: list[dict[str, object]] = []
    eval_rows = feature_rows.filter(pl.col("week") >= start_week).sort(
        ["baseline", "season", "week", "game_id"]
    )
    grouped_eval_rows = eval_rows.group_by(["baseline", "season", "week"], maintain_order=True)

    for keys, week_rows in grouped_eval_rows:
        baseline, season, week = keys
        baseline_name = str(baseline)
        season_value = int(season)
        week_value = int(week)
        training_rows = feature_rows.filter(
            (pl.col("baseline") == baseline_name)
            & (
                (pl.col("season") < season_value)
                | ((pl.col("season") == season_value) & (pl.col("week") < week_value))
            )
        )
        fitted_k, fitted_hfa_points = _fit_margin_projection(training_rows)

        for row in week_rows.iter_rows(named=True):
            predicted_margin = fitted_k * float(row["rating_diff"]) + fitted_hfa_points
            actual_margin = float(row["home_margin"])
            prediction_rows.append(
                {
                    "season": season_value,
                    "week": week_value,
                    "baseline": baseline_name,
                    "game_id": str(row["game_id"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "rating_diff": float(row["rating_diff"]),
                    "home_margin": actual_margin,
                    "predicted_margin": predicted_margin,
                    "error": predicted_margin - actual_margin,
                    "training_row_count": training_rows.height,
                    "fitted_k": fitted_k,
                    "fitted_hfa_points": fitted_hfa_points,
                }
            )

    return pl.DataFrame(prediction_rows).sort(["baseline", "season", "week", "game_id"])


def _metric_row(
    frame: pl.DataFrame,
    *,
    baseline: str,
    split: str,
    season: int | None,
) -> dict[str, object]:
    """Summarize one slice of prediction rows as MAE and RMSE."""
    if frame.is_empty():
        return {
            "baseline": baseline,
            "season": season,
            "split": split,
            "games": 0,
            "mae": 0.0,
            "rmse": 0.0,
        }

    errors = np.asarray(
        frame.select("error").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    return {
        "baseline": baseline,
        "season": season,
        "split": split,
        "games": frame.height,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def score_prediction_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate overall, per-season, and early/late walk-forward error metrics."""
    if predictions.is_empty():
        return pl.DataFrame(
            schema={
                "baseline": pl.String,
                "season": pl.Int64,
                "split": pl.String,
                "games": pl.Int64,
                "mae": pl.Float64,
                "rmse": pl.Float64,
            }
        )

    scored_predictions = predictions
    if "error" not in scored_predictions.columns:
        scored_predictions = scored_predictions.with_columns(
            (pl.col("predicted_margin") - pl.col("home_margin")).alias("error")
        )

    metric_rows: list[dict[str, object]] = []
    for baseline in scored_predictions.select("baseline").to_series().unique().to_list():
        baseline_frame = scored_predictions.filter(pl.col("baseline") == str(baseline))
        metric_rows.append(
            _metric_row(baseline_frame, baseline=str(baseline), split="overall", season=None)
        )
        metric_rows.append(
            _metric_row(
                baseline_frame.filter(pl.col("week") < _LATE_SEASON_FIRST_WEEK),
                baseline=str(baseline),
                split="early",
                season=None,
            )
        )
        metric_rows.append(
            _metric_row(
                baseline_frame.filter(pl.col("week") >= _LATE_SEASON_FIRST_WEEK),
                baseline=str(baseline),
                split="late",
                season=None,
            )
        )

        seasons = baseline_frame.select("season").to_series().unique().sort().to_list()
        for season in seasons:
            season_frame = baseline_frame.filter(pl.col("season") == int(season))
            metric_rows.append(
                _metric_row(
                    season_frame,
                    baseline=str(baseline),
                    split="season",
                    season=int(season),
                )
            )

    return pl.DataFrame(metric_rows).sort(["baseline", "split", "season"])


def _split_prediction_rows(predictions: pl.DataFrame, split: str) -> pl.DataFrame:
    """Return the prediction rows for one named evaluation split."""
    if split == "overall":
        return predictions
    if split == "early":
        return predictions.filter(pl.col("week") < _LATE_SEASON_FIRST_WEEK)
    if split == "late":
        return predictions.filter(pl.col("week") >= _LATE_SEASON_FIRST_WEEK)
    detail = f"unsupported split: {split}"
    raise ValueError(detail)


def compute_pairwise_mae_bootstrap(
    predictions: pl.DataFrame,
    *,
    baselines: Sequence[str] | None = None,
    splits: Sequence[str] = ("overall", "early", "late"),
    resamples: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Compute paired-bootstrap MAE deltas for every requested baseline pair.

    The delta is ``MAE(baseline_a) - MAE(baseline_b)``, so negative values favor
    ``baseline_a``.
    """
    if predictions.is_empty():
        return pl.DataFrame(
            schema={
                "baseline_a": pl.String,
                "baseline_b": pl.String,
                "split": pl.String,
                "games": pl.Int64,
                "mae_delta": pl.Float64,
                "ci_lower": pl.Float64,
                "ci_upper": pl.Float64,
                "probability_baseline_a_not_worse": pl.Float64,
                "distinguishable_from_zero": pl.Boolean,
            }
        )

    scored_predictions = predictions
    if "error" not in scored_predictions.columns:
        scored_predictions = scored_predictions.with_columns(
            (pl.col("predicted_margin") - pl.col("home_margin")).alias("error")
        )

    selected_baselines = list(
        baselines
        if baselines is not None
        else scored_predictions.select("baseline")
        .to_series()
        .cast(pl.String)
        .unique()
        .sort()
        .to_list()
    )
    rng = np.random.default_rng(seed)
    row_id_columns = ["season", "week", "game_id", "home_team", "away_team"]
    bootstrap_rows: list[dict[str, object]] = []

    for split in splits:
        split_predictions = _split_prediction_rows(scored_predictions, str(split))
        if split_predictions.is_empty():
            continue

        error_frame = (
            split_predictions.with_columns(pl.col("error").abs().alias("abs_error"))
            .select([*row_id_columns, "baseline", "abs_error"])
            .pivot(
                on="baseline", index=row_id_columns, values="abs_error", aggregate_function="first"
            )
            .sort(row_id_columns)
        )

        for baseline_a, baseline_b in combinations(selected_baselines, 2):
            if baseline_a not in error_frame.columns or baseline_b not in error_frame.columns:
                continue

            paired_errors = error_frame.select([baseline_a, baseline_b]).drop_nulls()
            if paired_errors.is_empty():
                continue

            diffs = np.asarray(
                paired_errors.select(pl.col(baseline_a) - pl.col(baseline_b))
                .to_series()
                .cast(pl.Float64)
                .to_list(),
                dtype=np.float64,
            )
            observed_delta = float(diffs.mean())
            sampled_means = np.empty(resamples, dtype=np.float64)
            sample_size = diffs.size

            for sample_index in range(resamples):
                sampled_indices = rng.integers(0, sample_size, size=sample_size)
                sampled_means[sample_index] = float(diffs[sampled_indices].mean())

            ci_lower, ci_upper = np.quantile(sampled_means, [0.025, 0.975])
            bootstrap_rows.append(
                {
                    "baseline_a": baseline_a,
                    "baseline_b": baseline_b,
                    "split": str(split),
                    "games": int(sample_size),
                    "mae_delta": observed_delta,
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "probability_baseline_a_not_worse": float(np.mean(sampled_means <= 0.0)),
                    "distinguishable_from_zero": bool(ci_upper < 0.0 or ci_lower > 0.0),
                }
            )

    return pl.DataFrame(bootstrap_rows).sort(["split", "baseline_a", "baseline_b"])


__all__ = [
    "EloConfig",
    "build_elo_feature_rows",
    "build_play_level_team_training_rows_with_special_teams",
    "build_raw_epa_feature_rows",
    "build_rolling_team_weight_maps",
    "build_saovr_feature_rows",
    "build_snapshot_feature_rows",
    "build_srs_feature_rows",
    "build_team_training_rows_with_special_teams",
    "build_weighted_team_feature_rows",
    "build_weighted_team_special_teams_feature_rows",
    "compute_pairwise_mae_bootstrap",
    "compute_playoff_metric_correlations",
    "compute_qbr_correlations",
    "compute_stability_metrics",
    "evaluate_feature_rows",
    "main",
    "run_play_level_team_special_teams_backtest",
    "run_walk_forward_backtest",
    "run_weighted_team_backtest",
    "run_weighted_team_special_teams_backtest",
    "score_prediction_rows",
]


if __name__ == "__main__":
    main()
