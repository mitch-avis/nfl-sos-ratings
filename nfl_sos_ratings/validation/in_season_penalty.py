"""In-season penalty test: previous-season ridge penalties against per-fit cross-validation.

Early in a season the published team fit cross-validates its ridge penalty on a few weeks of
games, and that choice is unstable: sometimes it lands on the grid's largest penalty and every
team rates near zero. This test compares two walk-forward team ratings built from the same
pre-week games:

- ``TeamRating`` (incumbent): ``fit_team_ratings`` cross-validating every snapshot, as published.
- ``TeamRatingPriorPenalty`` (candidate): ``fit_team_ratings`` with the previous season's
  full-season scrimmage and special-teams penalties.

Both go through the walk-forward harness's prior-only margin projection from prediction week 2.
The primary window is prediction weeks 2-5 and the guard is weeks 6 and later; each gets a paired
game bootstrap of the candidate-minus-incumbent MAE. The decision rule is pre-registered in
``.agents/ratings-simplification-plan.md``.

Run ``nfl-sos-ratings check-in-season-penalty``; it only reads ``data/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR
from nfl_sos_ratings.ridge import DEFAULT_RIDGE_LAMBDAS
from nfl_sos_ratings.team_rating import TeamRatingFit, fit_team_ratings
from nfl_sos_ratings.validation.walk_forward import (
    TEAM_RATING_BASELINE,
    build_snapshot_feature_rows,
    build_team_rating_feature_rows,
    compute_pairwise_mae_bootstrap,
    evaluate_feature_rows,
    score_prediction_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CANDIDATE_BASELINE = "TeamRatingPriorPenalty"
FIRST_SCORED_WEEK = 2
GUARD_FIRST_WEEK = 6
DEFAULT_START_SEASON = 2000
PENALTY_TABLE_WEEKS = (2, 3, 4, 5)

type Interval = tuple[float, float]


def build_prior_penalty_feature_rows(
    game_logs: pl.DataFrame, season: int, prior_fit: TeamRatingFit
) -> pl.DataFrame:
    """Build walk-forward rows from team ratings fit with the previous season's penalties."""

    def snapshot(prior_games: pl.DataFrame) -> pl.DataFrame:
        """Rate the pre-week games with the previous season's fixed penalties."""
        return fit_team_ratings(
            prior_games,
            scrimmage_lambda=prior_fit.scrimmage_lambda,
            special_teams_lambda=prior_fit.special_teams_lambda,
        ).ratings.select("team", pl.col("team_rating").alias("rating"))

    return build_snapshot_feature_rows(game_logs, season, CANDIDATE_BASELINE, snapshot)


def _window(predictions: pl.DataFrame, window: str) -> pl.DataFrame:
    """Return the prediction rows of the primary (weeks 2-5) or guard (weeks 6+) window."""
    week = pl.col("week")
    if window == "primary":
        return predictions.filter((week >= FIRST_SCORED_WEEK) & (week < GUARD_FIRST_WEEK))
    return predictions.filter(week >= GUARD_FIRST_WEEK)


def compare_windows(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return the candidate-minus-incumbent MAE bootstrap for the primary and guard windows."""
    return pl.concat(
        [
            compute_pairwise_mae_bootstrap(
                _window(predictions, window),
                baselines=[CANDIDATE_BASELINE, TEAM_RATING_BASELINE],
                splits=("overall",),
            ).with_columns(pl.lit(window).alias("window"))
            for window in ("primary", "guard")
        ]
    )


def reading(primary: Interval, guard: Interval) -> str:
    """Return the pre-registered reading of the primary and guard 95% intervals."""
    if primary[0] > 0.0 or guard[0] > 0.0:
        return "keep cross-validation: the candidate is significantly worse in a window"
    if primary[1] < 0.0:
        return "recommend the candidate: significantly better in weeks 2-5, not worse later"
    return "a tie, so the simpler candidate is the recommendation (no in-season tuning)"


def scrimmage_penalty_table(data_dir: Path, seasons: Sequence[int]) -> pl.DataFrame:
    """Return each season's cross-validated scrimmage penalty through weeks 2-5 and in full."""
    rows: list[dict[str, float | int]] = []
    for season in seasons:
        game_logs = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        row: dict[str, float | int] = {"season": season}
        for week in PENALTY_TABLE_WEEKS:
            row[f"week_{week}"] = fit_team_ratings(
                game_logs.filter(pl.col("week") <= week)
            ).scrimmage_lambda
        row["full"] = fit_team_ratings(game_logs).scrimmage_lambda
        rows.append(row)
    return pl.DataFrame(rows)


def _say(text: str) -> None:
    """Write one line of the report to standard output."""
    sys.stdout.write(f"{text}\n")


def _report_window(title: str, scores: pl.DataFrame, comparison: pl.DataFrame) -> Interval:
    """Print one window's MAE per baseline and the paired difference; return its interval."""
    _say(f"\n{title}")
    for row in scores.iter_rows(named=True):
        _say(f"  {row['baseline']:<24} MAE {row['mae']:.3f} over {row['games']} games")
    delta = comparison.row(0, named=True)
    _say(
        f"  candidate minus incumbent: {delta['mae_delta']:+.3f} "
        f"(95% interval {delta['ci_lower']:+.3f} to {delta['ci_upper']:+.3f})"
    )
    return float(delta["ci_lower"]), float(delta["ci_upper"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``check-in-season-penalty`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings check-in-season-penalty",
        description=(
            "Compare previous-season ridge penalties with per-fit cross-validation in the "
            "walk-forward team check (reads data/, writes nothing)."
        ),
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs (default: {DATA_DIR})."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=DEFAULT_START_SEASON,
        help=f"First scored season; its previous season must be in --data-dir "
        f"(default: {DEFAULT_START_SEASON}).",
    )
    parser.add_argument("--end-season", type=int, default=END_YEAR, help="Last scored season.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the in-season penalty test and print both windows, the reading, and the penalties."""
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    seasons = list(range(args.start_season, args.end_season + 1))
    features: list[pl.DataFrame] = []
    for season in seasons:
        game_logs = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        prior_fit = fit_team_ratings(
            pl.read_parquet(data_dir / f"{season - 1}_team_game_logs.parquet")
        )
        features.extend(
            [
                build_team_rating_feature_rows(game_logs, season),
                build_prior_penalty_feature_rows(game_logs, season, prior_fit),
            ]
        )
    predictions = evaluate_feature_rows(pl.concat(features), start_week=FIRST_SCORED_WEEK)
    comparisons = compare_windows(predictions)

    _say(
        f"In-season penalty test, {args.start_season}-{args.end_season}: "
        f"{CANDIDATE_BASELINE} (candidate) against {TEAM_RATING_BASELINE} (incumbent)"
    )
    intervals: dict[str, Interval] = {}
    for window, title in (
        ("primary", f"Primary (prediction weeks {FIRST_SCORED_WEEK}-{GUARD_FIRST_WEEK - 1})"),
        ("guard", f"Guard (prediction weeks {GUARD_FIRST_WEEK} and later)"),
    ):
        scores = score_prediction_rows(_window(predictions, window)).filter(
            pl.col("split") == "overall"
        )
        intervals[window] = _report_window(
            title, scores, comparisons.filter(pl.col("window") == window)
        )
    _say(f"\nReading: {reading(intervals['primary'], intervals['guard'])}")

    table = scrimmage_penalty_table(data_dir, seasons)
    _say("\nCross-validated scrimmage penalty by season (games through each week, and full):")
    with pl.Config(tbl_rows=-1, float_precision=0):
        _say(str(table))
    top = float(DEFAULT_RIDGE_LAMBDAS.max())
    counts = ", ".join(
        f"week {week}: {int((table.get_column(f'week_{week}') == top).sum())}"
        for week in PENALTY_TABLE_WEEKS
    )
    _say(f"Seasons at the grid's largest penalty ({top:g}) of {table.height}: {counts}")


__all__ = [
    "CANDIDATE_BASELINE",
    "build_prior_penalty_feature_rows",
    "compare_windows",
    "main",
    "reading",
    "scrimmage_penalty_table",
]
