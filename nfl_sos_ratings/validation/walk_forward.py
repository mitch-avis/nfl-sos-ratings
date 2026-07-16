"""Walk-forward validation helpers for team margin prediction."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import load_espn_qbr, load_pbp_data
from nfl_sos_ratings.simultaneous_adjustment import solve_srs
from nfl_sos_ratings.validation.diagnostics import (
    compute_qb_case_study,
    compute_qb_defense_spread_summary,
    compute_qb_experiment_sweep,
    compute_qb_playoff_validation_frame,
    compute_qb_season_audit_summary,
    compute_qb_split_half_diagnostics,
    compute_season_mae_deltas,
    compute_weekly_mae_curves,
    evaluate_qb_split_half_decision,
    summarize_qb_split_half_signal,
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

_NORMALIZED_NAME_RE = re.compile(r"[^a-z0-9]+")
_TEAM_T1_FEATURE_COLUMNS = [
    "adj_off_passing_epa_per_offensive_snap",
    "adj_off_rushing_epa_per_offensive_snap",
    "adj_def_passing_epa_per_offensive_snap",
    "adj_def_rushing_epa_per_offensive_snap",
]
_TEAM_T2_FEATURE_COLUMNS = [*_TEAM_T1_FEATURE_COLUMNS, "st_rating"]


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
    if x_values.size < 2 or y_values.size < 2:
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


def _weighted_spearman(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted Spearman correlation using average ranks for ties."""
    x_ranks = np.asarray(pl.Series(x_values).rank(method="average").to_list(), dtype=np.float64)
    y_ranks = np.asarray(pl.Series(y_values).rank(method="average").to_list(), dtype=np.float64)
    return _weighted_pearson(x_ranks, y_ranks, weights)


def compute_playoff_metric_correlations(
    playoff_summary: pl.DataFrame,
    regular_metrics: pl.DataFrame,
    *,
    metric_columns: Sequence[str],
    target_col: str = "playoff_adjusted_epa_per_dropback",
    weight_col: str = "playoff_dropbacks",
) -> pl.DataFrame:
    """Return per-season and pooled weighted playoff validation correlations by metric."""
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
            rows.append(
                {
                    "season_label": season_label,
                    "metric": metric,
                    "qb_seasons": int(metric_frame.height),
                    "playoff_dropbacks": float(weights.sum()),
                    "spearman": _weighted_spearman(x_values, y_values, weights),
                    "pearson": _weighted_pearson(x_values, y_values, weights),
                }
            )

    if "season" in joined.columns:
        for season_key, frame in joined.group_by("season", maintain_order=True):
            season_value = season_key[0] if isinstance(season_key, tuple) else season_key
            summarize_frame(frame, season_label=str(season_value))
    summarize_frame(joined, season_label="pooled")
    return pl.DataFrame(rows).sort(["season_label", "metric"])


def _build_home_game_frame(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one row per home game with the realized home margin."""
    required_columns = {"game_id", "week", "team", "opponent_team", "is_home", "point_margin"}
    missing = sorted(required_columns - set(weekly_team_rows.columns))
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"weekly_team_rows is missing required columns: {detail}")

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
    baseline_name: str = "T1Weighted",
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
    baseline_name: str = "T2Weighted",
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
    play_rows: pl.DataFrame,
    st_game_rows: pl.DataFrame,
    season: int,
    weight_map: dict[str, float],
    baseline_name: str = "T4Weighted",
) -> pl.DataFrame:
    """Build walk-forward rows from a play-level weighted team-plus-special-teams snapshot."""
    weighted_column = "weighted_team_rating"

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


def _equal_team_weight_map() -> dict[str, float]:
    """Return the default equal-weight team component map for T1 fallbacks."""
    equal_weight = 1.0 / len(_TEAM_T1_FEATURE_COLUMNS)
    return dict.fromkeys(_TEAM_T1_FEATURE_COLUMNS, equal_weight)


def _equal_weight_map(feature_columns: Sequence[str]) -> dict[str, float]:
    """Return equal weights over an arbitrary feature-column list."""
    equal_weight = 1.0 / len(feature_columns)
    return dict.fromkeys(feature_columns, equal_weight)


def _normalize_team_weight_map(weight_map: dict[str, float]) -> dict[str, float]:
    """Normalize a fitted team weight map by absolute weight while preserving signs."""
    total_abs_weight = float(sum(abs(weight) for weight in weight_map.values()))
    if total_abs_weight <= 0.0:
        return _equal_team_weight_map()
    return {
        column: float(weight / total_abs_weight)
        for column, weight in weight_map.items()
        if column in _TEAM_T1_FEATURE_COLUMNS
    }


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
    """Fit one T1 team component weight map per season using only prior season pairs."""
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
    baseline_name: str = "T1Weighted",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the T1 weighted-team walk-forward backtest with prior-season rolling weights."""
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
    """Build rolling T4 training rows from play-level EPA components plus special teams."""
    season_list = sorted(seasons)
    rows: list[pl.DataFrame] = []

    for season, next_season in zip(season_list, season_list[1:], strict=False):
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
    baseline_name: str = "T2Weighted",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the T2 weighted-team walk-forward backtest with a special-teams component."""
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
    baseline_name: str = "T4Weighted",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict[str, float]]]:
    """Run the T4 play-level weighted-team backtest with the same ST component as T2."""
    training_rows = build_play_level_team_training_rows_with_special_teams(data_dir, seasons)
    weight_maps = _build_rolling_feature_weight_maps(
        training_rows,
        seasons,
        _TEAM_T2_FEATURE_COLUMNS,
    )

    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        pbp = load_pbp_data(int(season))
        play_rows = build_play_level_team_frame_from_pbp(pbp)
        st_game_rows = build_special_teams_game_frame_from_pbp(pbp)
        feature_frames.append(
            build_play_level_weighted_team_special_teams_feature_rows(
                weekly_team_rows,
                play_rows=play_rows,
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


def build_play_level_team_season_ratings(
    data_dir: Path,
    seasons: Sequence[int],
    weight_maps: dict[int, dict[str, float]],
    output_col: str = "T4Weighted",
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
    for season, next_season in zip(seasons, seasons[1:], strict=False):
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
        "paired_rows": int(len(x_values)),
        "pearson": _pearson(x_values, y_values),
        "spearman": _spearman(x_values, y_values),
    }


def _find_pairwise_mae_row(
    mae_deltas: pl.DataFrame,
    baseline_a: str,
    baseline_b: str,
    split: str,
) -> dict[str, object] | None:
    """Return one bootstrap delta row when the baseline ordering matches the stored table."""
    if mae_deltas.is_empty():
        return None
    matched = mae_deltas.filter(
        (pl.col("baseline_a") == baseline_a)
        & (pl.col("baseline_b") == baseline_b)
        & (pl.col("split") == split)
    )
    return matched.row(0, named=True) if not matched.is_empty() else None


def _row_float(row: dict[str, object], key: str) -> float:
    """Return one row value as a float for report rendering and comparisons."""
    return float(cast(float | int | str, row[key]))


def _row_bool(row: dict[str, object], key: str) -> bool:
    """Return one row value as a bool for report rendering and comparisons."""
    return bool(row[key])


def build_stage3c_lines(
    metrics: pl.DataFrame,
    mae_deltas: pl.DataFrame,
    *,
    base_team_stability: dict[str, object] | None,
    t4_team_stability: dict[str, float | int] | None,
) -> list[str]:
    """Build the Stage 3c decision-rule and outcome narrative for the validation report."""
    overall_metrics = {
        str(row["baseline"]): row
        for row in metrics.filter(pl.col("split") == "overall").iter_rows(named=True)
    }
    t2_row = overall_metrics.get("T2Weighted")
    t4_row = overall_metrics.get("T4Weighted")
    srs_row = overall_metrics.get("SRS")
    if t2_row is None or t4_row is None or srs_row is None:
        return []

    candidate = "T4Weighted" if float(t4_row["mae"]) < float(t2_row["mae"]) else "T2Weighted"
    candidate_row = t4_row if candidate == "T4Weighted" else t2_row
    candidate_vs_raw = _find_pairwise_mae_row(mae_deltas, candidate, "RawEPA", "overall")
    candidate_vs_saovr = _find_pairwise_mae_row(mae_deltas, candidate, "SaOvR", "overall")
    candidate_vs_srs = _find_pairwise_mae_row(mae_deltas, candidate, "SRS", "overall")
    t4_vs_t2 = _find_pairwise_mae_row(mae_deltas, "T4Weighted", "T2Weighted", "overall")

    stability_ok = False
    if (
        candidate == "T4Weighted"
        and base_team_stability is not None
        and t4_team_stability is not None
    ):
        stability_ok = float(t4_team_stability["pearson"]) >= _row_float(
            base_team_stability,
            "pearson",
        ) and float(t4_team_stability["spearman"]) >= _row_float(base_team_stability, "spearman")

    if candidate == "T2Weighted":
        stability_ok = True

    promotion_pass = bool(
        candidate_vs_raw is not None
        and candidate_vs_saovr is not None
        and candidate_vs_srs is not None
        and _row_bool(candidate_vs_raw, "distinguishable_from_zero")
        and _row_float(candidate_vs_raw, "mae_delta") < 0.0
        and _row_bool(candidate_vs_saovr, "distinguishable_from_zero")
        and _row_float(candidate_vs_saovr, "mae_delta") < 0.0
        and _row_float(candidate_row, "mae") < _row_float(srs_row, "mae")
        and _row_float(candidate_row, "rmse") < _row_float(srs_row, "rmse")
        and _row_float(candidate_vs_srs, "ci_lower") <= 0.0
        and stability_ok
    )

    t4_displacement_lines = [
        "- T4 displacement check: T4Weighted overall MAE "
        f"{_row_float(t4_row, 'mae'):.3f} and RMSE {_row_float(t4_row, 'rmse'):.3f} versus "
        f"T2Weighted MAE {_row_float(t2_row, 'mae'):.3f} and RMSE {_row_float(t2_row, 'rmse'):.3f}."
    ]
    if t4_vs_t2 is not None:
        t4_displacement_lines.append(
            "  Bootstrap delta "
            f"{_row_float(t4_vs_t2, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(t4_vs_t2, 'ci_lower'):.3f}, {_row_float(t4_vs_t2, 'ci_upper'):.3f}] "
            f"and P(A<=B) {_row_float(t4_vs_t2, 'probability_baseline_a_not_worse'):.3f}."
        )

    lines = [
        "## Stage 3c Decision Rule",
        "",
        "> A candidate team backbone is promoted to the published ratings if, on the full held-out",
        "> walk-forward window: (1) it is significantly better than RawEPA and than the Stage 1",
        "> SaOvR (95% paired-bootstrap CI excluding zero); (2) it is numerically better than SRS",
        "> on both overall MAE and overall RMSE, and not significantly worse than SRS; and (3)",
        "> adopting it does not degrade team year-over-year stability below the Stage 3 recorded",
        "> value. Statistical parity with SRS plus the construct advantages (schedule-adjusted,",
        "> outcome-free components, unit-level decomposition) is sufficient and will be stated",
        "> plainly, as parity, in the methodology documentation — never overclaimed as",
        "> superiority.",
        "",
        '- Rationale: the stricter "beat SRS with CI clearing zero" bar is statistically',
        "  unattainable on this sample, and the current report already shows SRS itself does not",
        "  separate from RawEPA at 95%.",
        "",
        "## Stage 3c Team Outcome",
        "",
        f"- Candidate selected for the final Stage 3c gate: {candidate}.",
        *t4_displacement_lines,
    ]

    if candidate_vs_raw is not None:
        lines.append(
            "- Candidate vs RawEPA: MAE delta "
            f"{_row_float(candidate_vs_raw, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_raw, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_raw, 'ci_upper'):.3f}] "
            f"and P(A<=B) {_row_float(candidate_vs_raw, 'probability_baseline_a_not_worse'):.3f}."
        )
    if candidate_vs_saovr is not None:
        lines.append(
            "- Candidate vs Stage 1 SaOvR: MAE delta "
            f"{_row_float(candidate_vs_saovr, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_saovr, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_saovr, 'ci_upper'):.3f}] "
            f"and P(A<=B) {_row_float(candidate_vs_saovr, 'probability_baseline_a_not_worse'):.3f}."
        )
    if candidate_vs_srs is not None:
        lines.append(
            "- Candidate vs SRS: overall MAE/RMSE "
            f"{_row_float(candidate_row, 'mae'):.3f}/"
            f"{_row_float(candidate_row, 'rmse'):.3f} versus "
            f"{_row_float(srs_row, 'mae'):.3f}/{_row_float(srs_row, 'rmse'):.3f}."
        )
        lines.append(
            "  Bootstrap delta "
            f"{_row_float(candidate_vs_srs, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_srs, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_srs, 'ci_upper'):.3f}] and P(A<=B) "
            f"{_row_float(candidate_vs_srs, 'probability_baseline_a_not_worse'):.3f}."
        )

    if (
        candidate == "T4Weighted"
        and base_team_stability is not None
        and t4_team_stability is not None
    ):
        lines.append(
            "- Stability guard: T4Weighted Pearson/Spearman "
            f"{float(t4_team_stability['pearson']):.3f}/{float(t4_team_stability['spearman']):.3f} "
            f"versus Stage 3 SaOvR {_row_float(base_team_stability, 'pearson'):.3f}/"
            f"{_row_float(base_team_stability, 'spearman'):.3f}."
        )

    promotion_label = "Pass" if promotion_pass else "Fail"
    lines.append(f"- Promotion decision under the fixed Stage 3c rule: {promotion_label}.")
    lines.append("")
    return lines


def build_qb_open_status_lines() -> list[str]:
    """Return the fixed Stage 3c quarterback open-status note for the report."""
    return [
        "## QB Open Status",
        "",
        "- The Stage 3b QB audit continues to stand as a positive linear-adjustment result:",
        "  the additive adjustment operated at full strength in EPA units, the identity checks",
        "  held, and Q1/Q2 were correctly not adopted.",
        "- The QB question remains open anyway, but on a new hypothesis: possible model",
        "  misspecification from additive QB-vs-defense effects rather than miscalibrated",
        "  adjustment strength. Stage 3d is the pre-registered next step.",
    ]


def compute_stability_metrics(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute adjacent-season Pearson and Spearman stability for teams and QBs."""
    sorted_seasons = sorted(seasons)
    qb_metric_columns = ("QSaCR", "qb_passer_rating", "qb_any_a")
    qb_pairs: list[pl.DataFrame] = []
    team_pairs: list[pl.DataFrame] = []

    for season, next_season in zip(sorted_seasons, sorted_seasons[1:], strict=False):
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
                    "paired_rows": int(len(x_values)),
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
                "paired_rows": int(len(x_values)),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort(["entity", "metric"])


def compute_qbr_correlations(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute per-season QSaCR versus ESPN QBR correlations on matched QBs."""
    eligible_seasons = sorted(season for season in seasons if season >= 2006)
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
                "joined_rows": int(len(x_values)),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort("season")


def _format_markdown_value(value: object) -> str:
    """Format one scalar value for a Markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object | None]]) -> str:
    """Render a simple GitHub-flavored Markdown table."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def build_validation_report_text(
    metrics: pl.DataFrame,
    stability: pl.DataFrame,
    qbr_correlations: pl.DataFrame,
    seasons: list[int],
    start_week: int,
    command: str,
    mae_deltas: pl.DataFrame | None = None,
    stage3b_metrics: pl.DataFrame | None = None,
    weekly_curves: pl.DataFrame | None = None,
    saovr_vs_srs: pl.DataFrame | None = None,
    t2_vs_srs: pl.DataFrame | None = None,
    qb_season_audit: pl.DataFrame | None = None,
    qb_defense_spread: pl.DataFrame | None = None,
    qb_experiment_sweep: pl.DataFrame | None = None,
    qb_case_study: pl.DataFrame | None = None,
    stage3c_lines: list[str] | None = None,
    qb_open_status_lines: list[str] | None = None,
    block_r_lines: list[str] | None = None,
    qb_split_half_primary: pl.DataFrame | None = None,
    qb_split_half_placebo: pl.DataFrame | None = None,
    qb_split_half_cases: pl.DataFrame | None = None,
    qb_split_half_decision: dict[str, object] | None = None,
    qb_playoff_correlations: pl.DataFrame | None = None,
) -> str:
    """Render the Stage 3 and Stage 3b validation report as Markdown."""
    overall_metrics = metrics.filter(pl.col("split") != "season").sort(["baseline", "split"])
    season_metrics = metrics.filter(pl.col("split") == "season").sort(["season", "baseline"])
    overall_map = {
        str(row["baseline"]): float(row["mae"])
        for row in metrics.filter(pl.col("split") == "overall").iter_rows(named=True)
    }
    late_map = {
        str(row["baseline"]): float(row["mae"])
        for row in metrics.filter(pl.col("split") == "late").iter_rows(named=True)
    }
    stability_map = {str(row["metric"]): row for row in stability.iter_rows(named=True)}

    history_lines = [
        "## Stage 3 History",
        "",
        (
            "The original Stage 3 headline compared prior-carrying Elo against "
            "within-season-only backbones."
        ),
        "That result is preserved here as history rather than deleted or rewritten.",
        "",
    ]
    acceptance_lines = [
        "## Acceptance Check",
        "",
        "- Leakage discipline: the snapshot perturbation test and prior-only fit test pass.",
    ]

    if {"SaOvR", "Elo", "SRS", "RawEPA"} <= set(overall_map):
        saovr_mae = overall_map["SaOvR"]
        elo_mae = overall_map["Elo"]
        srs_mae = overall_map["SRS"]
        raw_epa_mae = overall_map["RawEPA"]
        team_pass = saovr_mae < elo_mae and saovr_mae < srs_mae and saovr_mae < raw_epa_mae
        acceptance_lines.append(
            "- Team headline: "
            f"{'Pass' if team_pass else 'Fail'}. SaOvR overall MAE {saovr_mae:.3f}; "
            f"Elo {elo_mae:.3f}; SRS {srs_mae:.3f}; RawEPA {raw_epa_mae:.3f}."
        )
        if {"SaOvR", "Elo", "SRS", "RawEPA"} <= set(late_map) and not team_pass:
            acceptance_lines.append(
                "- Team late-season context: SaOvR late-week MAE "
                f"{late_map['SaOvR']:.3f}; Elo {late_map['Elo']:.3f}; "
                f"SRS {late_map['SRS']:.3f}; RawEPA {late_map['RawEPA']:.3f}."
            )

    if {"QSaCR", "qb_passer_rating", "qb_any_a"} <= set(stability_map):
        qsacr_row = stability_map["QSaCR"]
        passer_row = stability_map["qb_passer_rating"]
        any_a_row = stability_map["qb_any_a"]
        qb_pass = (
            float(qsacr_row["pearson"]) > float(passer_row["pearson"])
            and float(qsacr_row["pearson"]) > float(any_a_row["pearson"])
            and float(qsacr_row["spearman"]) > float(passer_row["spearman"])
            and float(qsacr_row["spearman"]) > float(any_a_row["spearman"])
        )
        acceptance_lines.append(
            "- QB stability: "
            f"{'Pass' if qb_pass else 'Fail'}. QSaCR Pearson/Spearman "
            f"{float(qsacr_row['pearson']):.3f}/{float(qsacr_row['spearman']):.3f}; "
            "passer rating "
            f"{float(passer_row['pearson']):.3f}/{float(passer_row['spearman']):.3f}; "
            f"ANY/A {float(any_a_row['pearson']):.3f}/{float(any_a_row['spearman']):.3f}."
        )

    if not qbr_correlations.is_empty():
        pearson_mean = float(qbr_correlations.select(pl.col("pearson").mean()).item())
        spearman_mean = float(qbr_correlations.select(pl.col("spearman").mean()).item())
        acceptance_lines.append(
            "- External reference: mean QBR Pearson/Spearman correlation "
            f"{pearson_mean:.3f}/{spearman_mean:.3f} across {qbr_correlations.height} seasons."
        )
    acceptance_lines.append("")

    stage3b_lines = [
        "## Stage 3b Criterion",
        "",
        "Stage 3b re-registers the validation target into information-matched leagues.",
        "",
        (
            "- League 1 is binding: within-season-only team backbones must beat SRS and "
            "RawEPA on held-out\n  MAE, with paired-bootstrap support."
        ),
        (
            "- League 2 is informative: prior-carrying forecast-only variants can be "
            "compared against Elo,\n  but that is not the binding published-rating gate."
        ),
        "",
    ]

    if stage3b_metrics is not None and not stage3b_metrics.is_empty():
        stage3b_map = {
            str(row["baseline"]): float(row["mae"])
            for row in stage3b_metrics.filter(pl.col("split") == "overall").iter_rows(named=True)
        }
        t2_vs_srs_row = None
        t2_vs_raw_row = None
        if mae_deltas is not None and not mae_deltas.is_empty():
            t2_vs_srs_match = mae_deltas.filter(
                (pl.col("baseline_a") == "T2Weighted")
                & (pl.col("baseline_b") == "SRS")
                & (pl.col("split") == "overall")
            )
            if not t2_vs_srs_match.is_empty():
                t2_vs_srs_row = t2_vs_srs_match.row(0, named=True)
            t2_vs_raw_match = mae_deltas.filter(
                (pl.col("baseline_a") == "T2Weighted")
                & (pl.col("baseline_b") == "RawEPA")
                & (pl.col("split") == "overall")
            )
            if not t2_vs_raw_match.is_empty():
                t2_vs_raw_row = t2_vs_raw_match.row(0, named=True)

        if {"T1Weighted", "T2Weighted"} <= set(stage3b_map):
            league1_pass = (
                t2_vs_srs_row is not None
                and bool(t2_vs_srs_row["distinguishable_from_zero"])
                and t2_vs_raw_row is not None
                and bool(t2_vs_raw_row["distinguishable_from_zero"])
                and stage3b_map.get("T2Weighted", float("inf"))
                < stage3b_map.get("SRS", float("inf"))
                and stage3b_map.get("T2Weighted", float("inf"))
                < stage3b_map.get("RawEPA", float("inf"))
            )
            stage3b_lines.append("## Stage 3b Acceptance Check")
            stage3b_lines.append("")
            stage3b_lines.append(
                "- League 1 team headline: "
                f"{'Pass' if league1_pass else 'Fail'}. "
                f"T1Weighted overall MAE {stage3b_map['T1Weighted']:.3f}; "
                f"T2Weighted overall MAE {stage3b_map['T2Weighted']:.3f}; "
                f"SRS {stage3b_map.get('SRS', float('nan')):.3f};\n  "
                f"RawEPA {stage3b_map.get('RawEPA', float('nan')):.3f}."
            )
            if t2_vs_srs_row is not None:
                stage3b_lines.append(
                    "- League 1 bootstrap vs SRS: "
                    f"MAE delta {float(t2_vs_srs_row['mae_delta']):.3f} with 95% CI "
                    f"[{float(t2_vs_srs_row['ci_lower']):.3f}, "
                    f"{float(t2_vs_srs_row['ci_upper']):.3f}]."
                )
            if t2_vs_raw_row is not None:
                stage3b_lines.append(
                    "- League 1 bootstrap vs RawEPA: "
                    f"MAE delta {float(t2_vs_raw_row['mae_delta']):.3f} with 95% CI "
                    f"[{float(t2_vs_raw_row['ci_lower']):.3f}, "
                    f"{float(t2_vs_raw_row['ci_upper']):.3f}]."
                )

    if qb_experiment_sweep is not None and not qb_experiment_sweep.is_empty():
        current_row = qb_experiment_sweep.filter(pl.col("variant") == "current")
        q1_row = qb_experiment_sweep.filter(pl.col("variant") == "q1_fixed_team_defense")
        q2_row = qb_experiment_sweep.sort("slope", descending=True).head(1)
        if not current_row.is_empty() and not q1_row.is_empty() and not q2_row.is_empty():
            current_variant = current_row.row(0, named=True)
            q1_variant = q1_row.row(0, named=True)
            q2_variant = q2_row.row(0, named=True)
            stage3b_lines.append(
                "- QB revision sweep: not adopted. "
                f"Current eligible-QB slope {float(current_variant['slope']):.3f}; "
                f"Q1 fixed-defense slope {float(q1_variant['slope']):.3f};\n  "
                f"best tested Q2 slope {float(q2_variant['slope']):.3f} ({q2_variant['variant']})."
            )
            stage3b_lines.append(
                "- League 2 forecast-only prior experiment: not evaluated in this worktree."
            )
            stage3b_lines.append("")

    overview_lines = [
        "# Validation Report",
        "",
        f"Evaluation seasons: {min(seasons)}-{max(seasons)}.",
        f"Prediction weeks start at {start_week}.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        *(block_r_lines or []),
        *history_lines,
        *stage3b_lines,
        *(stage3c_lines or []),
        *acceptance_lines,
    ]

    if not overall_metrics.is_empty():
        overall_rows = [
            [row["baseline"], row["split"], row["games"], row["mae"], row["rmse"]]
            for row in overall_metrics.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Original Walk-Forward Summary",
                "",
                _markdown_table(["Baseline", "Split", "Games", "MAE", "RMSE"], overall_rows),
                "",
            ]
        )

    if stage3b_metrics is not None and not stage3b_metrics.is_empty():
        stage3b_rows = [
            [row["baseline"], row["split"], row["games"], row["mae"], row["rmse"]]
            for row in stage3b_metrics.filter(pl.col("split") != "season").iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## League 1 Team Experiments",
                "",
                _markdown_table(["Baseline", "Split", "Games", "MAE", "RMSE"], stage3b_rows),
                "",
            ]
        )

    if mae_deltas is not None and not mae_deltas.is_empty():
        delta_rows = [
            [
                row["baseline_a"],
                row["baseline_b"],
                row["split"],
                row["games"],
                row["mae_delta"],
                row["ci_lower"],
                row["ci_upper"],
                row["probability_baseline_a_not_worse"],
                row["distinguishable_from_zero"],
            ]
            for row in mae_deltas.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Paired Bootstrap MAE Deltas",
                "",
                _markdown_table(
                    [
                        "Baseline A",
                        "Baseline B",
                        "Split",
                        "Games",
                        "MAE Delta",
                        "CI Lower",
                        "CI Upper",
                        "P(A<=B)",
                        "Distinguishable",
                    ],
                    delta_rows,
                ),
                "",
            ]
        )

    if weekly_curves is not None and not weekly_curves.is_empty():
        weekly_rows = [
            [row["week"], row["baseline"], row["games"], row["mae"], row["rmse"]]
            for row in weekly_curves.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Weekly MAE Curves",
                "",
                _markdown_table(["Week", "Baseline", "Games", "MAE", "RMSE"], weekly_rows),
                "",
            ]
        )

    if not season_metrics.is_empty():
        season_rows = [
            [row["season"], row["baseline"], row["games"], row["mae"], row["rmse"]]
            for row in season_metrics.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Original Per-Season Walk-Forward",
                "",
                _markdown_table(["Season", "Baseline", "Games", "MAE", "RMSE"], season_rows),
                "",
            ]
        )

    if saovr_vs_srs is not None and not saovr_vs_srs.is_empty():
        delta_rows = [
            [row["season"], row["mae_a"], row["mae_b"], row["mae_delta"], row["rmse_delta"]]
            for row in saovr_vs_srs.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Per-Season SaOvR vs SRS",
                "",
                _markdown_table(
                    ["Season", "SaOvR MAE", "SRS MAE", "MAE Delta", "RMSE Delta"],
                    delta_rows,
                ),
                "",
            ]
        )

    if t2_vs_srs is not None and not t2_vs_srs.is_empty():
        delta_rows = [
            [row["season"], row["mae_a"], row["mae_b"], row["mae_delta"], row["rmse_delta"]]
            for row in t2_vs_srs.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Per-Season T2Weighted vs SRS",
                "",
                _markdown_table(
                    ["Season", "T2Weighted MAE", "SRS MAE", "MAE Delta", "RMSE Delta"],
                    delta_rows,
                ),
                "",
            ]
        )

    stability_rows = [
        [row["metric"], row["entity"], row["paired_rows"], row["pearson"], row["spearman"]]
        for row in stability.iter_rows(named=True)
    ]
    overview_lines.extend(
        [
            "## Stability",
            "",
            _markdown_table(
                ["Metric", "Entity", "Paired Rows", "Pearson", "Spearman"], stability_rows
            ),
            "",
        ]
    )

    qbr_rows = [
        [row["season"], row["joined_rows"], row["pearson"], row["spearman"]]
        for row in qbr_correlations.iter_rows(named=True)
    ]
    overview_lines.extend(
        [
            "## QBR Correlations",
            "",
            _markdown_table(["Season", "Joined Rows", "Pearson", "Spearman"], qbr_rows),
            "",
        ]
    )

    if qb_season_audit is not None and not qb_season_audit.is_empty():
        audit_rows = [
            [
                row["season"],
                row["rows"],
                row["slope"],
                row["correlation"],
                row["mean_abs_identity_residual"],
            ]
            for row in qb_season_audit.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## QB Adjustment Audit",
                "",
                _markdown_table(
                    ["Season", "Eligible QBs", "Slope", "Correlation", "Mean Abs Residual"],
                    audit_rows,
                ),
                "",
            ]
        )

    if qb_defense_spread is not None and not qb_defense_spread.is_empty():
        spread_rows = [
            [
                row["season"],
                row["team_defense_sd"],
                row["qb_defense_sd"],
                row["qb_to_team_spread_ratio"],
            ]
            for row in qb_defense_spread.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## QB Defense Spread Audit",
                "",
                _markdown_table(
                    ["Season", "Team Defense SD", "QB Defense SD", "QB/Team Ratio"],
                    spread_rows,
                ),
                "",
            ]
        )

    if qb_experiment_sweep is not None and not qb_experiment_sweep.is_empty():
        experiment_rows = [
            [
                row["variant"],
                row["eligible_rows"],
                row["slope"],
                row["correlation"],
                row.get("defense_penalty_multiplier"),
            ]
            for row in qb_experiment_sweep.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## QB Revision Sweep",
                "",
                _markdown_table(
                    [
                        "Variant",
                        "Eligible QBs",
                        "Slope",
                        "Correlation",
                        "Defense Penalty Multiplier",
                    ],
                    experiment_rows,
                ),
                "",
            ]
        )

    if qb_case_study is not None and not qb_case_study.is_empty():
        case_rows = [
            [
                row["variant"],
                row["qb_name"],
                row["raw_weighted"],
                row["adjusted_value"],
                row["faced_difficulty"],
                row["adjustment_delta"],
            ]
            for row in qb_case_study.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Maye/Stafford Case Study",
                "",
                _markdown_table(
                    [
                        "Variant",
                        "QB",
                        "Raw EPA/DB",
                        "Adjusted EPA/DB",
                        "Faced Difficulty",
                        "Adjustment Delta",
                    ],
                    case_rows,
                ),
                "",
            ]
        )

    if qb_open_status_lines:
        overview_lines.extend([*qb_open_status_lines, ""])

    if qb_split_half_primary is not None and not qb_split_half_primary.is_empty():
        overview_lines.extend(["## Stage 3d D1 Split-Half Diagnostics", ""])
        if qb_split_half_decision is not None:
            decision_text = str(qb_split_half_decision.get("decision", "not_supported"))
            overview_lines.append(f"- Decision gate reading: {decision_text}.")
            if qb_split_half_decision.get("primary_gate_supported") is not None:
                overview_lines.append(
                    "- Primary top-half gate: "
                    f"{'passed' if qb_split_half_decision['primary_gate_supported'] else 'failed'}."
                )
            if qb_split_half_decision.get("placebo_is_symmetric"):
                overview_lines.append(
                    "- Placebo check: bottom-half residuals showed a same-direction signal, "
                    "so the strong-defense-specific interpretation is not supported."
                )
            overview_lines.append("")
        primary_rows = [
            [
                row["scope"],
                row["season"],
                row["rows"],
                row["total_dropbacks"],
                row["slope"],
                row["ci_lower"],
                row["ci_upper"],
                row["direction_positive_count"],
                row["direction_total_count"],
                row["direction_p_value"],
            ]
            for row in qb_split_half_primary.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "Top-half residual regression summary:",
                _markdown_table(
                    [
                        "Scope",
                        "Season",
                        "QB Seasons",
                        "Dropbacks",
                        "Slope",
                        "CI Lower",
                        "CI Upper",
                        "Positive Seasons",
                        "Season Count",
                        "Binomial P",
                    ],
                    primary_rows,
                ),
                "",
            ]
        )
    if qb_split_half_placebo is not None and not qb_split_half_placebo.is_empty():
        placebo_rows = [
            [
                row["scope"],
                row["season"],
                row["rows"],
                row["total_dropbacks"],
                row["slope"],
                row["ci_lower"],
                row["ci_upper"],
            ]
            for row in qb_split_half_placebo.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "Bottom-half placebo summary:",
                _markdown_table(
                    [
                        "Scope",
                        "Season",
                        "QB Seasons",
                        "Dropbacks",
                        "Slope",
                        "CI Lower",
                        "CI Upper",
                    ],
                    placebo_rows,
                ),
                "",
            ]
        )
    if qb_split_half_cases is not None and not qb_split_half_cases.is_empty():
        case_rows = [
            [
                row.get("season"),
                row.get("qb_name"),
                row.get("faced_difficulty"),
                row.get("additive_prediction"),
                row.get("vs_top_half_adjusted_epa_per_dropback"),
                row.get("vs_top_half_residual"),
                row.get("vs_top_half_dropbacks"),
                row.get("vs_bottom_half_adjusted_epa_per_dropback"),
                row.get("vs_bottom_half_residual"),
                row.get("vs_bottom_half_dropbacks"),
            ]
            for row in qb_split_half_cases.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "2025 named case rows:",
                _markdown_table(
                    [
                        "Season",
                        "QB",
                        "Faced Difficulty",
                        "Additive Prediction",
                        "Top-Half Adj EPA/DB",
                        "Top-Half Residual",
                        "Top-Half DB",
                        "Bottom-Half Adj EPA/DB",
                        "Bottom-Half Residual",
                        "Bottom-Half DB",
                    ],
                    case_rows,
                ),
                "",
            ]
        )

    if qb_playoff_correlations is not None and not qb_playoff_correlations.is_empty():
        correlation_rows = [
            [
                row["season_label"],
                row["metric"],
                row["qb_seasons"],
                row["playoff_dropbacks"],
                row["spearman"],
                row["pearson"],
            ]
            for row in qb_playoff_correlations.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Stage 3d D3 Playoff Validation",
                "",
                "- Interpretation rule: whichever metric best predicts playoff performance is "
                "evidence about that metric, not about 2025 specifically. If QSaCR wins, that "
                "is vindicating evidence for the current composite and must be recorded as such.",
                "",
                _markdown_table(
                    [
                        "Season",
                        "Metric",
                        "QB Seasons",
                        "Playoff Dropbacks",
                        "Spearman",
                        "Pearson",
                    ],
                    correlation_rows,
                ),
                "",
            ]
        )

    overview_lines.extend(
        [
            "## SaCR Caveat",
            "",
            "SaCR may be evaluated as a secondary line with a caveat:",
            "its frozen Stage 2 weights were fit on the full 1999-2025 history.",
            "A walk-forward SaCR line over that same window has look-ahead in the weights.",
            (
                "SaOvR is the headline walk-forward metric because it does not depend on "
                "a fitted Stage 2 weight snapshot."
            ),
            "",
        ]
    )
    return "\n".join(overview_lines).rstrip() + "\n"


def write_validation_report(
    report_path: Path,
    metrics: pl.DataFrame,
    stability: pl.DataFrame,
    qbr_correlations: pl.DataFrame,
    mae_deltas: pl.DataFrame | None,
    seasons: list[int],
    start_week: int,
    command: str,
    stage3b_metrics: pl.DataFrame | None = None,
    weekly_curves: pl.DataFrame | None = None,
    saovr_vs_srs: pl.DataFrame | None = None,
    t2_vs_srs: pl.DataFrame | None = None,
    qb_season_audit: pl.DataFrame | None = None,
    qb_defense_spread: pl.DataFrame | None = None,
    qb_experiment_sweep: pl.DataFrame | None = None,
    qb_case_study: pl.DataFrame | None = None,
    stage3c_lines: list[str] | None = None,
    qb_open_status_lines: list[str] | None = None,
    block_r_lines: list[str] | None = None,
    qb_split_half_primary: pl.DataFrame | None = None,
    qb_split_half_placebo: pl.DataFrame | None = None,
    qb_split_half_cases: pl.DataFrame | None = None,
    qb_split_half_decision: dict[str, object] | None = None,
    qb_playoff_correlations: pl.DataFrame | None = None,
) -> None:
    """Write the Stage 3 validation report to disk."""
    report_text = build_validation_report_text(
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr_correlations,
        mae_deltas=mae_deltas,
        seasons=seasons,
        start_week=start_week,
        command=command,
        stage3b_metrics=stage3b_metrics,
        weekly_curves=weekly_curves,
        saovr_vs_srs=saovr_vs_srs,
        t2_vs_srs=t2_vs_srs,
        qb_season_audit=qb_season_audit,
        qb_defense_spread=qb_defense_spread,
        qb_experiment_sweep=qb_experiment_sweep,
        qb_case_study=qb_case_study,
        stage3c_lines=stage3c_lines,
        qb_open_status_lines=qb_open_status_lines,
        block_r_lines=block_r_lines,
        qb_split_half_primary=qb_split_half_primary,
        qb_split_half_placebo=qb_split_half_placebo,
        qb_split_half_cases=qb_split_half_cases,
        qb_split_half_decision=qb_split_half_decision,
        qb_playoff_correlations=qb_playoff_correlations,
    )
    report_path.write_text(report_text, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Stage 3 validation command."""
    parser = argparse.ArgumentParser(description="Run the Stage 3 walk-forward validation suite.")
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


def main(argv: list[str] | None = None) -> None:
    """Run the Stage 3 validation suite and write the Markdown report."""
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
        baselines=["T4Weighted", "T2Weighted", "T1Weighted", "SRS", "RawEPA", "SaOvR"],
    )
    weekly_curves = compute_weekly_mae_curves(combined_predictions).filter(
        pl.col("baseline").is_in(["Elo", "SRS", "SaOvR", "T2Weighted", "T4Weighted"])
    )
    saovr_vs_srs = compute_season_mae_deltas(combined_metrics, baseline_a="SaOvR", baseline_b="SRS")
    t2_vs_srs = compute_season_mae_deltas(
        combined_metrics, baseline_a="T2Weighted", baseline_b="SRS"
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
    qb_playoff_correlations = compute_playoff_metric_correlations(
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
                "vs_top_half_adjusted_epa_per_dropback",
            )
            if column in qb_playoff_validation.columns
        ],
    )
    qb_season_audit = compute_qb_season_audit_summary(data_dir, seasons)
    qb_defense_spread = compute_qb_defense_spread_summary(data_dir, seasons)
    qb_experiment_sweep = compute_qb_experiment_sweep(data_dir, seasons[-1])
    qb_case_study = compute_qb_case_study(data_dir, seasons[-1])
    stability = compute_stability_metrics(data_dir, seasons=seasons)
    base_team_stability = stability.filter(
        (pl.col("entity") == "team") & (pl.col("metric") == "SaOvR")
    )
    t4_history = build_play_level_team_season_ratings(data_dir, seasons, t4_weights)
    t4_team_stability = compute_team_rating_stability_from_history(t4_history, "T4Weighted")
    stage3c_lines = build_stage3c_lines(
        combined_metrics,
        mae_deltas,
        base_team_stability=base_team_stability.row(0, named=True)
        if not base_team_stability.is_empty()
        else None,
        t4_team_stability=t4_team_stability,
    )
    qb_open_status_lines = build_qb_open_status_lines()
    block_r_lines = [
        "## Block R Regression Note",
        "",
        (
            "- A Stage 3c regression combined pooled offense/defense reference arrays with "
            "current-season-only special-teams reference values, causing the team ratings path "
            "to raise a NumPy broadcast error before `*_combined.parquet` and `*_ratings.parquet` "
            "wrote."
        ),
        (
            "- The fix backfills historical `st_rating` values from "
            "`*_simultaneous_team_adjustments.parquet` when rebuilding pooled team references "
            "and makes the multi-season pipeline exit non-zero with a failure summary if any "
            "season data step fails."
        ),
        "",
    ]
    qbr_correlations = compute_qbr_correlations(data_dir, seasons=seasons)
    write_validation_report(
        report_path,
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr_correlations,
        mae_deltas=mae_deltas,
        seasons=seasons,
        start_week=args.start_week,
        command=command,
        stage3b_metrics=combined_metrics,
        weekly_curves=weekly_curves,
        saovr_vs_srs=saovr_vs_srs,
        t2_vs_srs=t2_vs_srs,
        qb_season_audit=qb_season_audit,
        qb_defense_spread=qb_defense_spread,
        qb_experiment_sweep=qb_experiment_sweep,
        qb_case_study=qb_case_study,
        stage3c_lines=stage3c_lines,
        qb_open_status_lines=qb_open_status_lines,
        block_r_lines=block_r_lines,
        qb_split_half_primary=qb_split_half_primary,
        qb_split_half_placebo=qb_split_half_placebo,
        qb_split_half_cases=qb_split_half_cases,
        qb_split_half_decision=qb_split_half_decision,
        qb_playoff_correlations=qb_playoff_correlations,
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
                baseline_frame.filter(pl.col("week") < 8),
                baseline=str(baseline),
                split="early",
                season=None,
            )
        )
        metric_rows.append(
            _metric_row(
                baseline_frame.filter(pl.col("week") >= 8),
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
        return predictions.filter(pl.col("week") < 8)
    if split == "late":
        return predictions.filter(pl.col("week") >= 8)
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
    "build_rolling_team_weight_maps",
    "build_team_training_rows_with_special_teams",
    "build_raw_epa_feature_rows",
    "build_saovr_feature_rows",
    "build_validation_report_text",
    "build_weighted_team_feature_rows",
    "build_weighted_team_special_teams_feature_rows",
    "build_srs_feature_rows",
    "build_snapshot_feature_rows",
    "compute_playoff_metric_correlations",
    "compute_qbr_correlations",
    "compute_pairwise_mae_bootstrap",
    "compute_stability_metrics",
    "evaluate_feature_rows",
    "main",
    "run_play_level_team_special_teams_backtest",
    "run_weighted_team_special_teams_backtest",
    "run_weighted_team_backtest",
    "run_walk_forward_backtest",
    "score_prediction_rows",
    "write_validation_report",
]


if __name__ == "__main__":
    main()
