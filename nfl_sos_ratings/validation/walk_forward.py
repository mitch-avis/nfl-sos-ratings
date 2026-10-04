"""Walk-forward validation of the published team rating, plus the QB stability and QBR checks.

Team check: for every week from ``--start-week`` on, each baseline is rebuilt from that season's
games played before the week, a margin model ``home_margin = k * rating_gap + home_edge`` is fit
on earlier predictions only, and the week's home margins are predicted. The baselines are:

- ``TeamRating``: the published points-based team rating from ``team_rating.fit_team_ratings``,
  called on the same team-game rows the pipeline publishes, so the validated estimator is the
  published one.
- ``SRS``: the simple rating system on point margin, from the same games.
- ``RawEPA``: each team's mean raw EPA margin per play, from the same games.
- ``Elo``: a fixed-constant Elo that carries ratings across seasons. It sees more information than
  the others, so it is a reference, never part of the decision rule.

The decision rule, written before the first run (``.agents/ratings-simplification-plan.md``):
``TeamRating`` stays the published headline if its overall MAE is not significantly worse than
``RawEPA`` or ``SRS`` in a paired bootstrap (the 95% interval of the MAE difference does not lie
entirely above zero).

QB checks: year-over-year stability of adjusted EPA per dropback beside passer rating and ANY/A,
and the per-season correlation of adjusted EPA per dropback with ESPN QBR.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import load_espn_qbr
from nfl_sos_ratings.simultaneous_adjustment import solve_srs
from nfl_sos_ratings.team_rating import fit_team_ratings
from nfl_sos_ratings.validation.report import ValidationReportInputs, write_validation_report

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TEAM_RATING_BASELINE = "TeamRating"
GATED_COMPARATORS: tuple[str, ...] = ("RawEPA", "SRS")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
TEAM_STABILITY_METRICS: tuple[str, ...] = ("team_rating", "SRS")
QB_STABILITY_METRICS: tuple[str, ...] = (
    "adj_qb_epa_per_dropback",
    "qb_passer_rating",
    "qb_any_a",
)
QB_REFERENCE_METRIC = "adj_qb_epa_per_dropback"

_NORMALIZED_NAME_RE = re.compile(r"[^a-z0-9]+")
# A correlation needs at least two paired observations.
_MIN_CORRELATION_SAMPLES = 2
# ESPN QBR starts in 2006.
_QBR_FIRST_SEASON = 2006
# Weeks before this one form the "early" evaluation split; the rest form "late".
_LATE_SEASON_FIRST_WEEK = 8
_FEATURE_SCHEMA: dict[str, type[pl.DataType]] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "baseline": pl.String,
    "game_id": pl.String,
    "home_team": pl.String,
    "away_team": pl.String,
    "rating_diff": pl.Float64,
    "home_margin": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class EloConfig:
    """Fixed constants for the simple team Elo baseline."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_field_elo: float = 55.0
    regression_to_mean: float = 2.0 / 3.0
    use_margin_multiplier: bool = True


@dataclass(frozen=True, slots=True)
class ComparatorResult:
    """The candidate's overall MAE difference against one comparator (negative favors it)."""

    comparator: str
    mae_delta: float
    ci_lower: float
    ci_upper: float

    @property
    def significantly_worse(self) -> bool:
        """Return whether the whole 95% interval says the candidate has the larger error."""
        return self.ci_lower > 0.0


@dataclass(frozen=True, slots=True)
class TeamDecision:
    """The outcome of the pre-registered team decision rule."""

    adopted: bool
    comparisons: tuple[ComparatorResult, ...]


def _normalize_person_name(name: str) -> str:
    """Normalize a player name for joins across sources."""
    return _NORMALIZED_NAME_RE.sub("", name.lower())


def _pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN when either side has no spread."""
    if len(x_values) < _MIN_CORRELATION_SAMPLES or np.std(x_values) == 0 or np.std(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return the Spearman rank correlation."""
    x_ranks = pl.Series(x_values).rank("average").to_numpy()
    y_ranks = pl.Series(y_values).rank("average").to_numpy()
    return _pearson(np.asarray(x_ranks, dtype=np.float64), np.asarray(y_ranks, dtype=np.float64))


def _build_home_game_frame(game_logs: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one row per home game with the realized home margin."""
    required = {"game_id", "week", "team", "opponent_team", "is_home", "point_margin"}
    missing = sorted(required - set(game_logs.columns))
    if missing:
        msg = f"team game logs are missing required columns: {', '.join(missing)}"
        raise ValueError(msg)
    return (
        game_logs.filter(pl.col("is_home"))
        .select(
            pl.lit(season).cast(pl.Int64).alias("season"),
            "game_id",
            pl.col("week").cast(pl.Int64),
            pl.col("team").alias("home_team"),
            pl.col("opponent_team").alias("away_team"),
            pl.col("point_margin").cast(pl.Float64).alias("home_margin"),
        )
        .sort(["week", "game_id"])
    )


def build_snapshot_feature_rows(
    game_logs: pl.DataFrame,
    season: int,
    baseline_name: str,
    snapshot_builder: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    """Build home-game feature rows from a rating snapshot of the games before each week.

    Args:
        game_logs: One season of team-game rows.
        season: The season label stamped on every row.
        baseline_name: The baseline label stamped on every row.
        snapshot_builder: Maps the pre-week game rows to a ``team``/``rating`` frame.

    Returns:
        One row per home game with the pregame rating gap and the realized home margin.

    """
    home_games = _build_home_game_frame(game_logs, season)
    frames: list[pl.DataFrame] = []
    for week in sorted(home_games.get_column("week").unique().to_list()):
        prior_games = game_logs.filter(pl.col("week") < week)
        teams = sorted(game_logs.get_column("team").unique().to_list())
        ratings = (
            pl.DataFrame({"team": teams, "rating": [0.0] * len(teams)})
            if prior_games.is_empty()
            else snapshot_builder(prior_games)
        )
        frames.append(
            home_games.filter(pl.col("week") == week)
            .join(
                ratings.select(pl.col("team").alias("home_team"), pl.col("rating").alias("home")),
                on="home_team",
                how="left",
            )
            .join(
                ratings.select(pl.col("team").alias("away_team"), pl.col("rating").alias("away")),
                on="away_team",
                how="left",
            )
            .with_columns(
                pl.lit(baseline_name).alias("baseline"),
                (pl.col("home").fill_null(0.0) - pl.col("away").fill_null(0.0)).alias(
                    "rating_diff"
                ),
            )
            .select(list(_FEATURE_SCHEMA))
        )
    return pl.concat(frames) if frames else pl.DataFrame(schema=_FEATURE_SCHEMA)


def _team_rating_snapshot(prior_games: pl.DataFrame) -> pl.DataFrame:
    """Return the published team rating fit on the given games."""
    return fit_team_ratings(prior_games).ratings.select(
        "team", pl.col("team_rating").alias("rating")
    )


def _srs_snapshot(prior_games: pl.DataFrame) -> pl.DataFrame:
    """Return SRS on point margin fit on the given games."""
    return solve_srs(prior_games, response_col="point_margin").select(
        "team", pl.col("srs_rating").alias("rating")
    )


def _raw_epa_snapshot(prior_games: pl.DataFrame) -> pl.DataFrame:
    """Return each team's mean raw EPA margin per play over the given games."""
    return prior_games.group_by("team").agg(pl.col("epa_margin_per_play").mean().alias("rating"))


def build_team_rating_feature_rows(game_logs: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward rows from week-by-week snapshots of the published team rating."""
    return build_snapshot_feature_rows(
        game_logs, season, TEAM_RATING_BASELINE, _team_rating_snapshot
    )


def build_srs_feature_rows(game_logs: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward rows from week-by-week SRS snapshots."""
    return build_snapshot_feature_rows(game_logs, season, "SRS", _srs_snapshot)


def build_raw_epa_feature_rows(game_logs: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward rows from week-by-week raw EPA margin means."""
    return build_snapshot_feature_rows(game_logs, season, "RawEPA", _raw_epa_snapshot)


def _elo_margin_multiplier(rating_gap: float, home_margin: float) -> float:
    """Return a standard logarithmic margin-of-victory Elo multiplier."""
    return float(np.log(abs(home_margin) + 1.0) * (2.2 / ((abs(rating_gap) * 0.001) + 2.2)))


def build_elo_feature_rows(
    home_games: pl.DataFrame,
    config: EloConfig | None = None,
) -> pl.DataFrame:
    """Build one-row-per-game pregame Elo features from chronological home games."""
    resolved = config or EloConfig()
    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows: list[dict[str, object]] = []

    for row in home_games.sort(["season", "week", "game_id"]).iter_rows(named=True):
        season = int(row["season"])
        if current_season is not None and season != current_season:
            for team, rating in list(ratings.items()):
                ratings[team] = (
                    resolved.initial_rating
                    + (rating - resolved.initial_rating) * resolved.regression_to_mean
                )
        current_season = season

        home_team, away_team = str(row["home_team"]), str(row["away_team"])
        home_rating = ratings.get(home_team, resolved.initial_rating)
        away_rating = ratings.get(away_team, resolved.initial_rating)
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

        expected_home = 1.0 / (1.0 + 10.0 ** (-(rating_diff + resolved.home_field_elo) / 400.0))
        actual_home = 1.0 if home_margin > 0.0 else 0.0 if home_margin < 0.0 else 0.5
        multiplier = (
            _elo_margin_multiplier(rating_diff, home_margin)
            if resolved.use_margin_multiplier and home_margin != 0.0
            else 1.0
        )
        delta = resolved.k_factor * multiplier * (actual_home - expected_home)
        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    return pl.DataFrame(rows, schema=_FEATURE_SCHEMA).sort(["season", "week", "game_id"])


def _fit_margin_projection(training_rows: pl.DataFrame) -> tuple[float, float]:
    """Fit ``home_margin = k * rating_diff + home_edge`` on past games only."""
    if training_rows.is_empty():
        return 0.0, 0.0
    rating_diff = training_rows.get_column("rating_diff").cast(pl.Float64).to_numpy()
    home_margin = training_rows.get_column("home_margin").cast(pl.Float64).to_numpy()
    if training_rows.height == 1 or np.isclose(rating_diff.std(), 0.0):
        return 0.0, float(home_margin.mean())
    design = np.column_stack((rating_diff, np.ones(len(rating_diff), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, home_margin, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def evaluate_feature_rows(feature_rows: pl.DataFrame, start_week: int) -> pl.DataFrame:
    """Fit prior-only margin models and predict every game from ``start_week`` onward."""
    prediction_rows: list[dict[str, object]] = []
    eval_rows = feature_rows.filter(pl.col("week") >= start_week).sort(
        ["baseline", "season", "week", "game_id"]
    )
    for (baseline, season, week), week_rows in eval_rows.group_by(
        ["baseline", "season", "week"], maintain_order=True
    ):
        training_rows = feature_rows.filter(
            (pl.col("baseline") == baseline)
            & (
                (pl.col("season") < season)
                | ((pl.col("season") == season) & (pl.col("week") < week))
            )
        )
        fitted_k, fitted_home_edge = _fit_margin_projection(training_rows)
        for row in week_rows.iter_rows(named=True):
            predicted = fitted_k * float(row["rating_diff"]) + fitted_home_edge
            actual = float(row["home_margin"])
            prediction_rows.append(
                {
                    **{column: row[column] for column in _FEATURE_SCHEMA},
                    "predicted_margin": predicted,
                    "error": predicted - actual,
                    "training_row_count": training_rows.height,
                    "fitted_k": fitted_k,
                    "fitted_hfa_points": fitted_home_edge,
                }
            )
    if not prediction_rows:
        return pl.DataFrame(
            schema={
                **_FEATURE_SCHEMA,
                "predicted_margin": pl.Float64,
                "error": pl.Float64,
                "training_row_count": pl.Int64,
                "fitted_k": pl.Float64,
                "fitted_hfa_points": pl.Float64,
            }
        )
    return pl.DataFrame(prediction_rows).sort(["baseline", "season", "week", "game_id"])


def _with_error(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return predictions with an ``error`` column, deriving it when absent."""
    if "error" in predictions.columns:
        return predictions
    return predictions.with_columns(
        (pl.col("predicted_margin") - pl.col("home_margin")).alias("error")
    )


def _split_prediction_rows(predictions: pl.DataFrame, split: str) -> pl.DataFrame:
    """Return the prediction rows for one named evaluation split."""
    if split == "overall":
        return predictions
    if split == "early":
        return predictions.filter(pl.col("week") < _LATE_SEASON_FIRST_WEEK)
    if split == "late":
        return predictions.filter(pl.col("week") >= _LATE_SEASON_FIRST_WEEK)
    msg = f"unsupported split: {split}"
    raise ValueError(msg)


def score_prediction_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return MAE and RMSE per baseline for the overall, early, and late splits."""
    scored = _with_error(predictions)
    rows: list[dict[str, object]] = []
    for baseline in sorted(scored.get_column("baseline").unique().to_list()):
        baseline_rows = scored.filter(pl.col("baseline") == baseline)
        for split in ("overall", "early", "late"):
            errors = _split_prediction_rows(baseline_rows, split).get_column("error").to_numpy()
            rows.append(
                {
                    "baseline": baseline,
                    "season": None,
                    "split": split,
                    "games": len(errors),
                    "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
                    "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "baseline": pl.String,
            "season": pl.Int64,
            "split": pl.String,
            "games": pl.Int64,
            "mae": pl.Float64,
            "rmse": pl.Float64,
        },
    )


def compute_weekly_mae_curves(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return MAE and RMSE by baseline and week across every evaluated season."""
    return (
        _with_error(predictions)
        .group_by(["baseline", "week"])
        .agg(
            pl.len().alias("games"),
            pl.col("error").abs().mean().alias("mae"),
            (pl.col("error") ** 2).mean().sqrt().alias("rmse"),
        )
        .sort(["baseline", "week"])
    )


def compute_pairwise_mae_bootstrap(
    predictions: pl.DataFrame,
    *,
    baselines: Sequence[str],
    splits: Sequence[str] = ("overall", "early", "late"),
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pl.DataFrame:
    """Return paired-bootstrap MAE differences for every pair of ``baselines``.

    The difference is ``MAE(baseline_a) - MAE(baseline_b)`` over the games both predicted, so a
    negative value favors ``baseline_a``. Games are resampled with replacement.
    """
    scored = _with_error(predictions)
    rng = np.random.default_rng(seed)
    id_columns = ["season", "week", "game_id", "home_team", "away_team"]
    rows: list[dict[str, object]] = []
    for split in splits:
        split_rows = _split_prediction_rows(scored, split)
        if split_rows.is_empty():
            continue
        errors = (
            split_rows.with_columns(pl.col("error").abs())
            .pivot(on="baseline", index=id_columns, values="error", aggregate_function="first")
            .sort(id_columns)
        )
        for baseline_a, baseline_b in combinations(baselines, 2):
            if baseline_a not in errors.columns or baseline_b not in errors.columns:
                continue
            diffs = (
                errors.select(baseline_a, baseline_b)
                .drop_nulls()
                .select(pl.col(baseline_a) - pl.col(baseline_b))
                .to_series()
                .to_numpy()
            )
            if diffs.size == 0:
                continue
            sampled = np.array(
                [diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(resamples)]
            )
            ci_lower, ci_upper = np.quantile(sampled, [0.025, 0.975])
            rows.append(
                {
                    "baseline_a": baseline_a,
                    "baseline_b": baseline_b,
                    "split": split,
                    "games": int(diffs.size),
                    "mae_delta": float(diffs.mean()),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "probability_baseline_a_not_worse": float(np.mean(sampled <= 0.0)),
                    "distinguishable_from_zero": bool(ci_upper < 0.0 or ci_lower > 0.0),
                }
            )
    return pl.DataFrame(rows).sort(["split", "baseline_a", "baseline_b"])


def evaluate_team_decision(
    mae_deltas: pl.DataFrame,
    *,
    candidate: str = TEAM_RATING_BASELINE,
    comparators: Sequence[str] = GATED_COMPARATORS,
) -> TeamDecision:
    """Apply the pre-registered rule: adopt unless significantly worse than a comparator.

    Raises:
        ValueError: If the overall bootstrap rows lack a candidate-versus-comparator pair.

    """
    overall = mae_deltas.filter(pl.col("split") == "overall")
    results: list[ComparatorResult] = []
    for comparator in comparators:
        forward = overall.filter(
            (pl.col("baseline_a") == candidate) & (pl.col("baseline_b") == comparator)
        )
        reverse = overall.filter(
            (pl.col("baseline_a") == comparator) & (pl.col("baseline_b") == candidate)
        )
        if not forward.is_empty():
            row = forward.row(0, named=True)
            results.append(
                ComparatorResult(comparator, row["mae_delta"], row["ci_lower"], row["ci_upper"])
            )
        elif not reverse.is_empty():
            row = reverse.row(0, named=True)
            results.append(
                ComparatorResult(comparator, -row["mae_delta"], -row["ci_upper"], -row["ci_lower"])
            )
        else:
            msg = f"no overall bootstrap row compares {candidate} with {comparator}"
            raise ValueError(msg)
    return TeamDecision(
        adopted=not any(result.significantly_worse for result in results),
        comparisons=tuple(results),
    )


def run_walk_forward_backtest(
    data_dir: Path,
    seasons: Sequence[int],
    start_week: int = 5,
    elo_config: EloConfig | None = None,
) -> pl.DataFrame:
    """Return walk-forward predictions for every baseline across the requested seasons."""
    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        game_logs = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        feature_frames.extend(
            [
                build_team_rating_feature_rows(game_logs, season),
                build_srs_feature_rows(game_logs, season),
                build_raw_epa_feature_rows(game_logs, season),
                build_elo_feature_rows(
                    _build_home_game_frame(game_logs, season), config=elo_config
                ),
            ]
        )
    features = pl.concat(feature_frames) if feature_frames else pl.DataFrame(schema=_FEATURE_SCHEMA)
    return evaluate_feature_rows(features, start_week=start_week)


def _paired_correlation(
    entity: str, metric: str, current: pl.Series, upcoming: pl.Series
) -> dict[str, object]:
    """Return one stability row for adjacent-season values of one metric."""
    x_values = current.cast(pl.Float64).to_numpy()
    y_values = upcoming.cast(pl.Float64).to_numpy()
    return {
        "entity": entity,
        "metric": metric,
        "paired_rows": len(x_values),
        "pearson": _pearson(x_values, y_values),
        "spearman": _spearman(x_values, y_values),
    }


def _eligible_qbs(path: Path) -> pl.DataFrame:
    """Read one season's QB table, keeping qualified passers when eligibility is present."""
    frame = pl.read_parquet(path)
    return frame.filter(pl.col("qb_is_eligible")) if "qb_is_eligible" in frame.columns else frame


def compute_stability_metrics(data_dir: Path, seasons: Sequence[int]) -> pl.DataFrame:
    """Return adjacent-season Pearson and Spearman stability for team and QB metrics."""
    team_pairs: list[pl.DataFrame] = []
    qb_pairs: list[pl.DataFrame] = []
    for season, next_season in pairwise(sorted(seasons)):
        team_pairs.append(
            pl.read_parquet(data_dir / f"{season}_ratings.parquet")
            .select("team", *TEAM_STABILITY_METRICS)
            .join(
                pl.read_parquet(data_dir / f"{next_season}_ratings.parquet").select(
                    "team", *TEAM_STABILITY_METRICS
                ),
                on="team",
                suffix="_next",
            )
        )
        qb_pairs.append(
            _eligible_qbs(data_dir / f"{season}_qb_combined.parquet")
            .select("qb_id", *QB_STABILITY_METRICS)
            .join(
                _eligible_qbs(data_dir / f"{next_season}_qb_combined.parquet").select(
                    "qb_id", *QB_STABILITY_METRICS
                ),
                on="qb_id",
                suffix="_next",
            )
        )

    rows: list[dict[str, object]] = []
    for entity, pairs, metrics in (
        ("team", team_pairs, TEAM_STABILITY_METRICS),
        ("qb", qb_pairs, QB_STABILITY_METRICS),
    ):
        if not pairs:
            continue
        stacked = pl.concat(pairs, how="diagonal_relaxed")
        for metric in metrics:
            paired = stacked.select(metric, f"{metric}_next").drop_nulls()
            rows.append(
                _paired_correlation(
                    entity, metric, paired.get_column(metric), paired.get_column(f"{metric}_next")
                )
            )
    return pl.DataFrame(rows).sort(["entity", "metric"])


def compute_qbr_correlations(data_dir: Path, seasons: Sequence[int]) -> pl.DataFrame:
    """Return per-season correlations of adjusted EPA per dropback with ESPN QBR."""
    eligible_seasons = sorted(season for season in seasons if season >= _QBR_FIRST_SEASON)
    schema = {
        "season": pl.Int64,
        "joined_rows": pl.Int64,
        "pearson": pl.Float64,
        "spearman": pl.Float64,
    }
    if not eligible_seasons:
        return pl.DataFrame(schema=schema)

    qbr = load_espn_qbr(level="season", seasons=eligible_seasons)
    if "game_week" in qbr.columns:
        qbr = qbr.filter(pl.col("game_week") == "Season Total")
    qbr = qbr.select(
        pl.col("season").cast(pl.Int64),
        pl.col("team_abb").cast(pl.String).alias("team"),
        pl.col("name_display")
        .cast(pl.String)
        .map_elements(_normalize_person_name, return_dtype=pl.String)
        .alias("normalized_name"),
        "qbr_total",
    )

    rows: list[dict[str, object]] = []
    for season in eligible_seasons:
        joined = (
            _eligible_qbs(data_dir / f"{season}_qb_combined.parquet")
            .with_columns(
                pl.lit(season).cast(pl.Int64).alias("season"),
                pl.col("qb_name")
                .cast(pl.String)
                .map_elements(_normalize_person_name, return_dtype=pl.String)
                .alias("normalized_name"),
            )
            .join(qbr, on=["season", "team", "normalized_name"], how="inner")
            .drop_nulls([QB_REFERENCE_METRIC, "qbr_total"])
        )
        x_values = joined.get_column(QB_REFERENCE_METRIC).cast(pl.Float64).to_numpy()
        y_values = joined.get_column("qbr_total").cast(pl.Float64).to_numpy()
        rows.append(
            {
                "season": season,
                "joined_rows": joined.height,
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )
    return pl.DataFrame(rows, schema=schema)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``validate`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings validate",
        description=(
            "Run the walk-forward team validation and the QB checks, then write the Markdown "
            "report."
        ),
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory of Parquet outputs.")
    parser.add_argument("--start-season", type=int, default=START_YEAR, help="First season.")
    parser.add_argument("--end-season", type=int, default=END_YEAR, help="Last season.")
    parser.add_argument("--start-week", type=int, default=5, help="First week to score.")
    parser.add_argument(
        "--report-path", default="docs/validation-report.md", help="Markdown report path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the validation suite and write the Markdown report."""
    args = _parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    data_dir = Path(args.data_dir)
    report_path = Path(args.report_path)

    predictions = run_walk_forward_backtest(data_dir, seasons, start_week=args.start_week)
    mae_deltas = compute_pairwise_mae_bootstrap(
        predictions, baselines=[TEAM_RATING_BASELINE, *GATED_COMPARATORS, "Elo"]
    )
    write_validation_report(
        report_path,
        ValidationReportInputs(
            command=(
                f"nfl-sos-ratings validate --data-dir {args.data_dir} "
                f"--start-season {args.start_season} --end-season {args.end_season} "
                f"--start-week {args.start_week} --report-path {args.report_path}"
            ),
            seasons=seasons,
            start_week=args.start_week,
            metrics=score_prediction_rows(predictions),
            mae_deltas=mae_deltas,
            weekly_curves=compute_weekly_mae_curves(predictions),
            decision=evaluate_team_decision(mae_deltas),
            stability=compute_stability_metrics(data_dir, seasons),
            qbr_correlations=compute_qbr_correlations(data_dir, seasons),
        ),
    )
    print(f"Wrote validation report to {report_path}")


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "GATED_COMPARATORS",
    "TEAM_RATING_BASELINE",
    "ComparatorResult",
    "EloConfig",
    "TeamDecision",
    "build_elo_feature_rows",
    "build_raw_epa_feature_rows",
    "build_snapshot_feature_rows",
    "build_srs_feature_rows",
    "build_team_rating_feature_rows",
    "compute_pairwise_mae_bootstrap",
    "compute_qbr_correlations",
    "compute_stability_metrics",
    "compute_weekly_mae_curves",
    "evaluate_feature_rows",
    "evaluate_team_decision",
    "main",
    "run_walk_forward_backtest",
    "score_prediction_rows",
]


if __name__ == "__main__":
    main()
