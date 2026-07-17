"""Composite-weight fitting helpers.

These helpers build season-pair training rows from the on-disk Parquet back-catalog and fit
predictive composite weights for the published team and QB ratings.

QB season pairs are intentionally restricted to the CPOE era because
``adj_qb_completion_percentage_above_expectation`` is not available before 2006.
That means the QB fit learns from consecutive-season quarterbacks who both stayed in the league
and cleared the eligibility gate in adjacent seasons, which introduces survivorship bias toward
established starters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR, TEAM_ABBR_ALIASES
from nfl_sos_ratings.data_loader import load_pbp_data
from nfl_sos_ratings.simultaneous_adjustment import solve_srs

QB_COMPOSITE_START_SEASON = 2006


@dataclass(frozen=True, slots=True)
class CompositeComponent:
    """One composite component, optionally derived from more than one source column."""

    name: str
    source_columns: tuple[str, ...]
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class FrozenCompositeSpec:
    """A frozen published composite definition plus its fitting provenance."""

    name: str
    components: tuple[CompositeComponent, ...]
    weights: tuple[tuple[str, float], ...]
    target_column: str
    fit_window: tuple[int, int]
    fitting_command: str
    refit_policy: str
    sample_weight_column: str | None = None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the ordered component names that carry nonzero frozen weights."""
        return tuple(name for name, _ in self.weights)

    def weight_map(self) -> dict[str, float]:
        """Return the frozen coefficient mapping keyed by component name."""
        return dict(self.weights)


TEAM_SACR_COMPONENTS: tuple[CompositeComponent, ...] = (
    CompositeComponent(
        name="adj_off_passing_epa_per_offensive_snap",
        source_columns=("adj_off_passing_epa_per_offensive_snap",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_off_rushing_epa_per_offensive_snap",
        source_columns=("adj_off_rushing_epa_per_offensive_snap",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_def_passing_epa_per_offensive_snap",
        source_columns=("adj_def_passing_epa_per_offensive_snap",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_def_rushing_epa_per_offensive_snap",
        source_columns=("adj_def_rushing_epa_per_offensive_snap",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_def_takeaway_creation_rate_per_defensive_snap",
        source_columns=(
            "adj_def_def_interceptions_per_defensive_snap",
            "adj_def_def_fumbles_forced_per_defensive_snap",
        ),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="st_rating",
        source_columns=("st_rating",),
        higher_is_better=True,
    ),
)

QB_QSACR_COMPONENTS: tuple[CompositeComponent, ...] = (
    CompositeComponent(
        name="adj_qb_epa_per_dropback",
        source_columns=("adj_qb_epa_per_dropback",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_qb_completion_percentage_above_expectation",
        source_columns=("adj_qb_completion_percentage_above_expectation",),
        higher_is_better=True,
    ),
    CompositeComponent(
        name="adj_qb_sack_rate",
        source_columns=("adj_qb_sack_rate",),
        higher_is_better=False,
    ),
    CompositeComponent(
        name="adj_qb_td_int_margin_rate",
        source_columns=("adj_qb_td_int_margin_rate",),
        higher_is_better=True,
    ),
)

COMPOSITE_WEIGHT_FITTING_COMMAND = "uv run python -m nfl_sos_ratings.composite_weights"

COMPOSITE_WEIGHT_REFIT_POLICY = (
    "Refit only when a maintainer explicitly reruns the composite-weight workflow, reviews the "
    "held-out diagnostics, and updates the published snapshot in the same change set."
)

TEAM_TAKEAWAY_CREATION_CANDIDATE_WEIGHT = -0.04081182634425329

TEAM_SACR_FROZEN_SPEC = FrozenCompositeSpec(
    name="SaCR",
    components=tuple(
        component
        for component in TEAM_SACR_COMPONENTS
        if component.name
        in {
            "adj_off_passing_epa_per_offensive_snap",
            "adj_off_rushing_epa_per_offensive_snap",
            "adj_def_passing_epa_per_offensive_snap",
            "adj_def_rushing_epa_per_offensive_snap",
            "st_rating",
        }
    ),
    weights=(
        ("adj_off_passing_epa_per_offensive_snap", 0.3828739475913225),
        ("adj_off_rushing_epa_per_offensive_snap", 0.19062479977967036),
        ("adj_def_passing_epa_per_offensive_snap", 0.27163464954613765),
        ("adj_def_rushing_epa_per_offensive_snap", 0.0973640754908631),
        ("st_rating", 0.0575025275920063),
    ),
    target_column="SaOvR",
    fit_window=(1999, 2025),
    fitting_command=COMPOSITE_WEIGHT_FITTING_COMMAND,
    refit_policy=COMPOSITE_WEIGHT_REFIT_POLICY,
)

QB_QSACR_FROZEN_SPEC = FrozenCompositeSpec(
    name="QSaCR",
    components=QB_QSACR_COMPONENTS,
    weights=(
        ("adj_qb_epa_per_dropback", 0.6687790473858877),
        ("adj_qb_completion_percentage_above_expectation", 0.21464381898367774),
        ("adj_qb_sack_rate", 0.06725872314827445),
        ("adj_qb_td_int_margin_rate", 0.04931841048216012),
    ),
    target_column="adj_qb_epa_per_dropback",
    fit_window=(2006, 2025),
    fitting_command=COMPOSITE_WEIGHT_FITTING_COMMAND,
    refit_policy=COMPOSITE_WEIGHT_REFIT_POLICY,
    sample_weight_column="qb_dropbacks",
)


def _read_back_catalog_frame(data_dir: Path, season: int, suffix: str) -> pl.DataFrame:
    """Read one season artifact from the Parquet back-catalog."""
    return pl.read_parquet(data_dir / f"{season}_{suffix}.parquet")


def _component_expr(component: CompositeComponent) -> pl.Expr:
    """Return a Polars expression that derives one component from its source columns."""
    parts = [
        pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(0.0)
        for column in component.source_columns
    ]
    if len(parts) == 1:
        return parts[0].alias(component.name)
    return pl.sum_horizontal(parts).alias(component.name)


def _validate_component_columns(
    df: pl.DataFrame,
    components: Sequence[CompositeComponent],
    *,
    frame_name: str,
) -> None:
    """Raise when a consumed back-catalog frame is missing a required composite column."""
    missing = sorted(
        {
            source_column
            for component in components
            for source_column in component.source_columns
            if source_column not in df.columns
        }
    )
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"{frame_name} is missing required composite columns: {detail}")


def _zscore(values: np.ndarray) -> np.ndarray:
    """Return a sample-standardized array, or a centered array when spread is zero."""
    if len(values) == 0:
        return values
    centered = values - float(values.mean())
    if len(values) == 1:
        return centered
    std = float(values.std(ddof=1))
    return centered / std if std > 0.0 else centered


def _standardize_component_columns(
    df: pl.DataFrame,
    components: Sequence[CompositeComponent],
) -> pl.DataFrame:
    """Z-score component columns and orient every component so higher is better."""
    result = df
    for component in components:
        values = np.asarray(
            result.select(component.name)
            .to_series()
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(0.0)
            .to_list(),
            dtype=np.float64,
        )
        oriented = _zscore(values)
        if not component.higher_is_better:
            oriented *= -1.0
        result = result.with_columns(pl.Series(component.name, oriented))
    return result


def _canonicalize_team_codes(df: pl.DataFrame, column: str = "team") -> pl.DataFrame:
    """Return a frame with one team column normalized through the shared alias table."""
    return df.with_columns(pl.col(column).cast(pl.String).replace(TEAM_ABBR_ALIASES).alias(column))


def _empty_training_frame(schema: dict[str, type | pl.DataType | None]) -> pl.DataFrame:
    """Return an empty frame with the requested schema."""
    return pl.DataFrame(schema=schema)


def _build_special_teams_game_frame_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Convert special-play PBP into one net special-teams margin row per team-game."""
    special_flag = "special" if "special" in pbp_df.columns else "special_teams_play"
    required_columns = {"game_id", "week", "posteam", "defteam", special_flag, "epa"}
    if not required_columns.issubset(set(pbp_df.columns)):
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "st_epa_margin_per_play": pl.Float64,
            }
        )

    special_plays = pbp_df.filter(pl.col(special_flag).cast(pl.Int64) == 1).drop_nulls(
        ["posteam", "defteam"]
    )
    if special_plays.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "st_epa_margin_per_play": pl.Float64,
            }
        )

    offense_perspective = special_plays.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent_team"),
        pl.col("epa").cast(pl.Float64).alias("st_epa_for"),
        pl.lit(0.0).alias("st_epa_against"),
        pl.lit(1.0).alias("special_play_count"),
    )
    defense_perspective = special_plays.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("defteam").cast(pl.String).alias("team"),
        pl.col("posteam").cast(pl.String).alias("opponent_team"),
        pl.lit(0.0).alias("st_epa_for"),
        pl.col("epa").cast(pl.Float64).alias("st_epa_against"),
        pl.lit(1.0).alias("special_play_count"),
    )

    return (
        pl.concat([offense_perspective, defense_perspective], how="vertical")
        .group_by(["game_id", "week", "team", "opponent_team"])
        .agg(
            pl.col("st_epa_for").sum().alias("st_epa_for"),
            pl.col("st_epa_against").sum().alias("st_epa_against"),
            pl.col("special_play_count").sum().alias("special_play_count"),
        )
        .with_columns(
            (
                (pl.col("st_epa_for") - pl.col("st_epa_against")) / pl.col("special_play_count")
            ).alias("st_epa_margin_per_play")
        )
        .select(["game_id", "week", "team", "opponent_team", "st_epa_margin_per_play"])
    )


def _build_special_teams_rating_frame(season: int, teams: list[str]) -> pl.DataFrame:
    """Return one season's full-year special-teams ratings, or zeros when unavailable."""
    zero_frame = pl.DataFrame({"team": teams, "st_rating": [0.0] * len(teams)})
    try:
        pbp_df = load_pbp_data(season)
    except Exception:
        return zero_frame

    st_game_rows = _build_special_teams_game_frame_from_pbp(pbp_df)
    if st_game_rows.is_empty():
        return zero_frame

    return solve_srs(st_game_rows, response_col="st_epa_margin_per_play").rename(
        {"srs_rating": "st_rating"}
    )


def _attach_special_teams_rating(current: pl.DataFrame, season: int) -> pl.DataFrame:
    """Join a season's special-teams rating onto the current training frame when needed."""
    if "st_rating" in current.columns:
        return current

    teams = current.select("team").to_series().cast(pl.String).to_list()
    st_ratings = _build_special_teams_rating_frame(season, teams)
    return current.join(st_ratings, on="team", how="left").with_columns(
        pl.col("st_rating").fill_null(0.0)
    )


def build_team_training_rows(data_dir: Path, seasons: Iterable[int]) -> pl.DataFrame:
    """Build team season-pair rows with season-t standardized predictors and season-t+1 targets."""
    season_list = sorted(seasons)
    feature_components = TEAM_SACR_FROZEN_SPEC.components
    feature_names = [component.name for component in feature_components]
    rows: list[pl.DataFrame] = []

    for season, next_season in zip(season_list, season_list[1:], strict=False):
        current = _canonicalize_team_codes(_read_back_catalog_frame(data_dir, season, "combined"))
        current = _attach_special_teams_rating(current, season)
        upcoming = _canonicalize_team_codes(
            _read_back_catalog_frame(data_dir, next_season, "combined")
        )
        _validate_component_columns(current, feature_components, frame_name=f"{season}_combined")
        if "SaOvR" not in upcoming.columns:
            raise ValueError(f"{next_season}_combined is missing required composite columns: SaOvR")

        features = current.select(
            pl.col("team"), *[_component_expr(component) for component in feature_components]
        )
        features = _standardize_component_columns(features, feature_components)
        targets = upcoming.select(
            pl.col("team"),
            pl.col("SaOvR").cast(pl.Float64).fill_nan(None).fill_null(0.0).alias("target"),
        )
        rows.append(
            features.join(targets, on="team", how="inner")
            .with_columns(
                pl.lit(season).cast(pl.Int64).alias("season"),
                pl.lit(next_season).cast(pl.Int64).alias("next_season"),
            )
            .select("season", "next_season", "team", *feature_names, "target")
        )

    if not rows:
        schema: dict[str, type | pl.DataType | None] = {
            "season": pl.Int64,
            "next_season": pl.Int64,
            "team": pl.String,
            **dict.fromkeys(feature_names, pl.Float64),
            "target": pl.Float64,
        }
        return _empty_training_frame(schema)
    return pl.concat(rows, how="vertical_relaxed")


def _resolve_qb_weight_column(df: pl.DataFrame) -> str:
    """Return the QB season-volume column used as the WLS sample weight."""
    for column in ("qb_dropbacks", "qb_dropbacks_total"):
        if column in df.columns:
            return column
    raise ValueError("QB training rows require qb_dropbacks or qb_dropbacks_total")


def build_qb_training_rows(data_dir: Path, seasons: Iterable[int]) -> pl.DataFrame:
    """Build QB season-pair rows with season-t standardized predictors and season-t+1 targets."""
    season_list = [season for season in sorted(seasons) if season >= QB_COMPOSITE_START_SEASON]
    feature_names = [component.name for component in QB_QSACR_COMPONENTS]
    rows: list[pl.DataFrame] = []

    for season, next_season in zip(season_list, season_list[1:], strict=False):
        current = _read_back_catalog_frame(data_dir, season, "qb_combined")
        upcoming = _read_back_catalog_frame(data_dir, next_season, "qb_combined")
        _validate_component_columns(
            current, QB_QSACR_COMPONENTS, frame_name=f"{season}_qb_combined"
        )
        if "adj_qb_epa_per_dropback" not in upcoming.columns:
            raise ValueError(
                f"{next_season}_qb_combined is missing required composite columns: "
                "adj_qb_epa_per_dropback"
            )

        if "qb_is_eligible" in current.columns:
            current = current.filter(pl.col("qb_is_eligible"))

        weight_column = _resolve_qb_weight_column(current)
        features = current.select(
            pl.col("qb_id").cast(pl.String),
            pl.col(weight_column)
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(0.0)
            .alias("qb_dropbacks"),
            *[_component_expr(component) for component in QB_QSACR_COMPONENTS],
        )
        features = _standardize_component_columns(features, QB_QSACR_COMPONENTS)
        targets = upcoming.select(
            pl.col("qb_id").cast(pl.String),
            pl.col("adj_qb_epa_per_dropback")
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(0.0)
            .alias("target"),
        )
        rows.append(
            features.join(targets, on="qb_id", how="inner")
            .with_columns(
                pl.lit(season).cast(pl.Int64).alias("season"),
                pl.lit(next_season).cast(pl.Int64).alias("next_season"),
            )
            .select("season", "next_season", "qb_id", "qb_dropbacks", *feature_names, "target")
        )

    if not rows:
        schema: dict[str, type | pl.DataType | None] = {
            "season": pl.Int64,
            "next_season": pl.Int64,
            "qb_id": pl.String,
            "qb_dropbacks": pl.Float64,
            **dict.fromkeys(feature_names, pl.Float64),
            "target": pl.Float64,
        }
        return _empty_training_frame(schema)
    return pl.concat(rows, how="vertical_relaxed")


def _matrix(df: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return a float64 design matrix for the requested columns."""
    return np.asarray(
        df.select(
            [pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(0.0) for column in columns]
        ).to_numpy(),
        dtype=np.float64,
    )


def _vector(df: pl.DataFrame, column: str) -> np.ndarray:
    """Return a float64 vector for one DataFrame column."""
    return np.asarray(
        df.select(pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(0.0))
        .to_series()
        .to_list(),
        dtype=np.float64,
    )


def _zscore_against(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """Return sample z-scores against a reference distribution."""
    if len(reference_values) == 0:
        return values
    mean = float(reference_values.mean())
    centered = values - mean
    if len(reference_values) <= 1:
        return centered
    std = float(reference_values.std(ddof=1))
    return centered / std if std > 0.0 else centered


def _component_values(df: pl.DataFrame, component: CompositeComponent) -> np.ndarray | None:
    """Return one component's raw values, or None when its source columns are unavailable."""
    missing = [column for column in component.source_columns if column not in df.columns]
    if missing:
        return None
    values = np.zeros(df.height, dtype=np.float64)
    for column in component.source_columns:
        values += _vector(df, column)
    return values


def build_weighted_composite(
    df: pl.DataFrame,
    spec: FrozenCompositeSpec,
    reference_df: pl.DataFrame | None = None,
) -> np.ndarray:
    """Build one frozen composite from standardized, oriented component inputs."""
    resolved_reference_df = df if reference_df is None or reference_df.is_empty() else reference_df
    weights = spec.weight_map()
    composite = np.zeros(df.height, dtype=np.float64)
    present_weight_total = 0.0

    for component in spec.components:
        values = _component_values(df, component)
        if values is None:
            continue
        reference_values = _component_values(resolved_reference_df, component)
        reference_source = reference_values if reference_values is not None else values
        zscore = _zscore_against(values, reference_source)
        oriented = zscore if component.higher_is_better else -zscore
        component_weight = weights[component.name]
        composite += oriented * component_weight
        present_weight_total += component_weight

    if present_weight_total <= 0.0:
        return composite
    return composite / present_weight_total


def _fit_linear_model(
    df: pl.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    sample_weight_column: str | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a linear model with an intercept, optionally using sample weights."""
    features = _matrix(df, feature_columns)
    target = _vector(df, target_column)
    design = np.column_stack([np.ones(df.height, dtype=np.float64), features])

    if sample_weight_column is not None:
        sample_weights = np.sqrt(_vector(df, sample_weight_column))
        design = design * sample_weights[:, None]
        target = target * sample_weights

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients[1:], float(coefficients[0])


def fit_linear_weights(
    df: pl.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    sample_weight_column: str | None = None,
) -> dict[str, float]:
    """Fit OLS or WLS coefficients for the supplied training frame."""
    weights, _ = _fit_linear_model(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_weight_column=sample_weight_column,
    )
    return {column: float(weight) for column, weight in zip(feature_columns, weights, strict=True)}


def _normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
    """Return a positive weight map normalized to sum to 1.0 when possible."""
    total = float(sum(weights.values()))
    if total == 0.0:
        return dict.fromkeys(weights, 0.0)
    return {column: float(weight / total) for column, weight in weights.items()}


def _weighted_mae(errors: np.ndarray, sample_weights: np.ndarray | None) -> float:
    """Return MAE with optional non-negative sample weights."""
    absolute_errors = np.abs(errors)
    if sample_weights is None:
        return float(absolute_errors.mean())
    weight_total = float(sample_weights.sum())
    return (
        float(np.dot(absolute_errors, sample_weights) / weight_total) if weight_total > 0 else 0.0
    )


def _weighted_rmse(errors: np.ndarray, sample_weights: np.ndarray | None) -> float:
    """Return RMSE with optional non-negative sample weights."""
    squared_errors = errors * errors
    if sample_weights is None:
        return float(np.sqrt(squared_errors.mean()))
    weight_total = float(sample_weights.sum())
    if weight_total <= 0:
        return 0.0
    return float(np.sqrt(np.dot(squared_errors, sample_weights) / weight_total))


def evaluate_leave_one_season_out(
    df: pl.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    holdout_column: str,
    sample_weight_column: str | None = None,
) -> dict[str, float]:
    """Compare the fitted blend against a calibrated equal-weight blend by held-out season."""
    weighted_residuals: list[np.ndarray] = []
    equal_residuals: list[np.ndarray] = []
    evaluation_weights: list[np.ndarray] = []

    for holdout_value in df.select(holdout_column).to_series().unique().to_list():
        training = df.filter(pl.col(holdout_column) != holdout_value)
        holdout = df.filter(pl.col(holdout_column) == holdout_value)
        if training.is_empty() or holdout.is_empty():
            continue

        fitted_weights, fitted_intercept = _fit_linear_model(
            training,
            feature_columns=feature_columns,
            target_column=target_column,
            sample_weight_column=sample_weight_column,
        )
        holdout_features = _matrix(holdout, feature_columns)
        holdout_target = _vector(holdout, target_column)
        weighted_prediction = fitted_intercept + (holdout_features @ fitted_weights)
        weighted_residuals.append(holdout_target - weighted_prediction)

        equal_training = training.with_columns(
            pl.mean_horizontal(*[pl.col(column) for column in feature_columns]).alias("equal_blend")
        )
        equal_holdout = holdout.with_columns(
            pl.mean_horizontal(*[pl.col(column) for column in feature_columns]).alias("equal_blend")
        )
        equal_slope, equal_intercept = _fit_linear_model(
            equal_training,
            feature_columns=("equal_blend",),
            target_column=target_column,
            sample_weight_column=sample_weight_column,
        )
        equal_prediction = equal_intercept + (
            _matrix(equal_holdout, ("equal_blend",)) @ equal_slope
        )
        equal_residuals.append(holdout_target - equal_prediction)

        if sample_weight_column is not None:
            evaluation_weights.append(_vector(holdout, sample_weight_column))

    if not weighted_residuals:
        return {
            "weighted_mae": 0.0,
            "weighted_rmse": 0.0,
            "equal_weight_mae": 0.0,
            "equal_weight_rmse": 0.0,
        }

    weighted_error_vector = np.concatenate(weighted_residuals)
    equal_error_vector = np.concatenate(equal_residuals)
    sample_weights = np.concatenate(evaluation_weights) if evaluation_weights else None
    return {
        "weighted_mae": _weighted_mae(weighted_error_vector, sample_weights),
        "weighted_rmse": _weighted_rmse(weighted_error_vector, sample_weights),
        "equal_weight_mae": _weighted_mae(equal_error_vector, sample_weights),
        "equal_weight_rmse": _weighted_rmse(equal_error_vector, sample_weights),
    }


def main() -> None:
    """Print the reproducible composite-weight fit and held-out diagnostics."""
    data_dir = Path(DATA_DIR)
    seasons = range(START_YEAR, END_YEAR + 1)

    team_rows = build_team_training_rows(data_dir, seasons)
    team_candidate_weights = fit_linear_weights(
        team_rows,
        feature_columns=[component.name for component in TEAM_SACR_COMPONENTS],
        target_column="target",
    )
    team_candidate_diag = evaluate_leave_one_season_out(
        team_rows,
        feature_columns=[component.name for component in TEAM_SACR_COMPONENTS],
        target_column="target",
        holdout_column="season",
    )
    team_frozen_weights = _normalize_weight_map(
        fit_linear_weights(
            team_rows,
            feature_columns=TEAM_SACR_FROZEN_SPEC.feature_columns,
            target_column="target",
        )
    )
    team_frozen_diag = evaluate_leave_one_season_out(
        team_rows.select(
            ["season", "next_season", "team", *TEAM_SACR_FROZEN_SPEC.feature_columns, "target"]
        ),
        feature_columns=TEAM_SACR_FROZEN_SPEC.feature_columns,
        target_column="target",
        holdout_column="season",
    )

    qb_rows = build_qb_training_rows(data_dir, seasons)
    qb_frozen_weights = _normalize_weight_map(
        fit_linear_weights(
            qb_rows,
            feature_columns=QB_QSACR_FROZEN_SPEC.feature_columns,
            target_column="target",
            sample_weight_column=QB_QSACR_FROZEN_SPEC.sample_weight_column,
        )
    )
    qb_frozen_diag = evaluate_leave_one_season_out(
        qb_rows,
        feature_columns=QB_QSACR_FROZEN_SPEC.feature_columns,
        target_column="target",
        holdout_column="season",
        sample_weight_column=QB_QSACR_FROZEN_SPEC.sample_weight_column,
    )

    print("Composite-weight fit summary")
    print(f"Command: {COMPOSITE_WEIGHT_FITTING_COMMAND}")
    print()
    print("Team candidate fit (includes turnover-creation test component):")
    for name, weight in team_candidate_weights.items():
        print(f"  {name}: {weight:.12f}")
    print(
        "  holdout: "
        f"weighted_rmse={team_candidate_diag['weighted_rmse']:.6f}, "
        f"equal_weight_rmse={team_candidate_diag['equal_weight_rmse']:.6f}, "
        f"weighted_mae={team_candidate_diag['weighted_mae']:.6f}, "
        f"equal_weight_mae={team_candidate_diag['equal_weight_mae']:.6f}"
    )
    print()
    print("Frozen SaCR weights:")
    for name, weight in team_frozen_weights.items():
        print(f"  {name}: {weight:.12f}")
    print(
        "  holdout: "
        f"weighted_rmse={team_frozen_diag['weighted_rmse']:.6f}, "
        f"equal_weight_rmse={team_frozen_diag['equal_weight_rmse']:.6f}, "
        f"weighted_mae={team_frozen_diag['weighted_mae']:.6f}, "
        f"equal_weight_mae={team_frozen_diag['equal_weight_mae']:.6f}"
    )
    print(
        "  excluded candidate: "
        f"adj_def_takeaway_creation_rate_per_defensive_snap={TEAM_TAKEAWAY_CREATION_CANDIDATE_WEIGHT:.12f}"
    )
    print()
    print("Frozen QSaCR weights:")
    for name, weight in qb_frozen_weights.items():
        print(f"  {name}: {weight:.12f}")
    print(
        "  holdout: "
        f"weighted_rmse={qb_frozen_diag['weighted_rmse']:.6f}, "
        f"equal_weight_rmse={qb_frozen_diag['equal_weight_rmse']:.6f}, "
        f"weighted_mae={qb_frozen_diag['weighted_mae']:.6f}, "
        f"equal_weight_mae={qb_frozen_diag['equal_weight_mae']:.6f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CompositeComponent",
    "FrozenCompositeSpec",
    "COMPOSITE_WEIGHT_FITTING_COMMAND",
    "COMPOSITE_WEIGHT_REFIT_POLICY",
    "QB_COMPOSITE_START_SEASON",
    "QB_QSACR_COMPONENTS",
    "QB_QSACR_FROZEN_SPEC",
    "TEAM_SACR_COMPONENTS",
    "TEAM_SACR_FROZEN_SPEC",
    "TEAM_TAKEAWAY_CREATION_CANDIDATE_WEIGHT",
    "build_weighted_composite",
    "build_qb_training_rows",
    "build_team_training_rows",
    "evaluate_leave_one_season_out",
    "fit_linear_weights",
    "main",
]
