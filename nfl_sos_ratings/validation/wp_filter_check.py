"""Walk-forward test of the garbage-time filter: do ratings on the kept plays predict better?

A filter at X% keeps the plays whose offense's win probability before the snap was between X% and
100% minus X% (``team_wp_bins``), plus the plays without one. For each candidate threshold (5%,
10%, and 20%) and for 0%, the team fit runs unchanged on the kept plays: the kept plays
and EPA replace the four game-log columns the fit reads
(:func:`nfl_sos_ratings.wp_filter.team_game_logs_at_threshold`), and the penalties are the
previous season's, cross-validated on that season's kept plays at the same threshold. Each
threshold then goes through the walk-forward harness's prior-only margin model, so its own fitted
slope absorbs any difference in rating scale.

The team fit is the published one without its preseason prior, as published when this test ran
(``walk_forward.build_team_rating_feature_rows`` without a season prior). Before reading any result,
the test checks that 0% is that rating: the 0% kept columns equal the game logs in every season, and
the 0% walk-forward rows equal its rows. Each candidate is compared with 0% by a paired game
bootstrap of the absolute errors (10,000 resamples, one seed for all three), with 98.33% intervals:
95% after a Bonferroni adjustment for three comparisons. The decision rule, written in
``.agents/roadmap.md`` before the first run: a threshold qualifies only if its overall interval lies
entirely below zero; the qualifying threshold with the lowest MAE is the recommendation, and with
none, no filter is.

Descriptive extras, never decision inputs: the kept share of plays, team and QB year-over-year
stability, the QB correlation with ESPN QBR, and one team's and one passer's ratings by threshold.
The QB fit is likewise the published one on kept dropbacks and their play-level EPA, its penalty
cross-validated per season.

Run ``nfl-sos-ratings check-wp-filter``; it only reads ``data/`` (and downloads ESPN QBR).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import PBP_START_SEASON
from nfl_sos_ratings.qb_rating import fit_qb_ratings
from nfl_sos_ratings.team_rating import (
    SCRIMMAGE_EPA_COLUMN,
    SCRIMMAGE_PLAYS_COLUMN,
    SPECIAL_TEAMS_EPA_COLUMN,
    SPECIAL_TEAMS_PLAYS_COLUMN,
    TeamRatingFit,
    fit_team_ratings,
    fit_team_ratings_with_previous_penalties,
)
from nfl_sos_ratings.validation.walk_forward import (
    TEAM_RATING_BASELINE,
    build_snapshot_feature_rows,
    build_team_rating_feature_rows,
    compute_pairwise_mae_bootstrap,
    evaluate_feature_rows,
    load_season_qbr,
    previous_season_fit,
    qbr_correlation,
    score_prediction_rows,
)
from nfl_sos_ratings.wp_filter import qb_games_at_threshold, team_game_logs_at_threshold

if TYPE_CHECKING:
    from collections.abc import Sequence

BASELINE_THRESHOLD = 0
CANDIDATE_THRESHOLDS: tuple[int, ...] = (5, 10, 20)
THRESHOLDS: tuple[int, ...] = (BASELINE_THRESHOLD, *CANDIDATE_THRESHOLDS)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FAMILY_CONFIDENCE = 0.95
# Bonferroni over the candidate comparisons: 98.33% intervals for three candidates.
COMPARISON_CONFIDENCE = 1.0 - (1.0 - FAMILY_CONFIDENCE) / len(CANDIDATE_THRESHOLDS)
DEFAULT_START_WEEK = 5
# How far a 0% value may sit from the published one before the test refuses to run.
ZERO_TOLERANCE = 1e-9
SPLITS: tuple[str, ...] = ("overall", "early", "late")
DEFAULT_SPOTLIGHT_SEASON = 2025
DEFAULT_SPOTLIGHT_TEAM = "NE"
DEFAULT_SPOTLIGHT_QB = "Drake Maye"
QB_RATING_COLUMN = "adj_qb_epa_per_dropback"
_TEAM_PLAY_COLUMNS = (
    SCRIMMAGE_PLAYS_COLUMN,
    SCRIMMAGE_EPA_COLUMN,
    SPECIAL_TEAMS_PLAYS_COLUMN,
    SPECIAL_TEAMS_EPA_COLUMN,
)
_ROW_KEYS = ["season", "week", "game_id"]
# A correlation needs at least two pairs.
_MIN_PAIRS = 2


def baseline_name(threshold: int) -> str:
    """Return the walk-forward baseline label of one threshold, such as ``Filter10``."""
    return f"Filter{threshold}"


def previous_threshold_fit(
    game_logs: pl.DataFrame, bins: pl.DataFrame, threshold: int
) -> TeamRatingFit:
    """Return one season's full team fit on the plays kept at ``threshold``.

    Its penalties are cross-validated on those kept plays; the next season's snapshots reuse them,
    as the published fit reuses the previous season's.
    """
    return fit_team_ratings(team_game_logs_at_threshold(game_logs, bins, threshold))


def build_threshold_feature_rows(
    game_logs: pl.DataFrame,
    bins: pl.DataFrame,
    season: int,
    threshold: int,
    previous: TeamRatingFit | None,
) -> pl.DataFrame:
    """Build walk-forward rows from the published team fit on the plays kept at ``threshold``.

    Args:
        game_logs: One season of team game logs.
        bins: That season's ``team_wp_bins``.
        season: The season label stamped on every row.
        threshold: The filter threshold, 0-20.
        previous: The previous season's fit at the same threshold
            (:func:`previous_threshold_fit`), or ``None`` for the first play-by-play season, whose
            snapshots cross-validate their own penalties.

    Returns:
        One row per home game, labeled :func:`baseline_name`.

    """
    kept = team_game_logs_at_threshold(game_logs, bins, threshold)

    def snapshot(prior_games: pl.DataFrame) -> pl.DataFrame:
        """Rate the pre-week kept plays with the previous season's penalties at this threshold."""
        return fit_team_ratings_with_previous_penalties(prior_games, previous).ratings.select(
            "team", pl.col("team_rating").alias("rating")
        )

    return build_snapshot_feature_rows(kept, season, baseline_name(threshold), snapshot)


def _largest_gap(left: pl.Series, right: pl.Series) -> float:
    """Return the largest absolute difference of two aligned columns; a null counts as infinite."""
    gaps = np.abs(left.cast(pl.Float64).to_numpy() - right.cast(pl.Float64).to_numpy())
    if np.isnan(gaps).any():
        return math.inf
    return float(gaps.max()) if gaps.size else 0.0


def check_zero_kept_columns(game_logs: pl.DataFrame, bins: pl.DataFrame, season: int) -> float:
    """Return the largest gap between the 0% kept columns and the game logs.

    Raises:
        ValueError: If a gap exceeds ``ZERO_TOLERANCE``, which means the bins do not cover the game
            logs and 0% would not be the published rating.

    """
    kept = team_game_logs_at_threshold(game_logs, bins, BASELINE_THRESHOLD)
    largest = max(
        _largest_gap(kept.get_column(column), game_logs.get_column(column))
        for column in _TEAM_PLAY_COLUMNS
    )
    if largest > ZERO_TOLERANCE:
        msg = (
            f"{season}: at 0% the kept plays differ from the team game logs by up to "
            f"{largest:.3g}; check {season}_team_wp_bins"
        )
        raise ValueError(msg)
    return largest


def check_zero_threshold_rows(features: pl.DataFrame) -> float:
    """Return the largest gap between the 0% walk-forward rows and the published rating's rows.

    Raises:
        ValueError: If the two do not cover the same games or a rating gap differs by more than
            ``ZERO_TOLERANCE``.

    """
    published = features.filter(pl.col("baseline") == TEAM_RATING_BASELINE).select(
        *_ROW_KEYS, "rating_diff"
    )
    zero = features.filter(pl.col("baseline") == baseline_name(BASELINE_THRESHOLD)).select(
        *_ROW_KEYS, pl.col("rating_diff").alias("zero_rating_diff")
    )
    joined = published.join(zero, on=_ROW_KEYS, how="inner")
    if not joined.height == published.height == zero.height:
        msg = "the 0% rows and the published team rating's rows do not cover the same games"
        raise ValueError(msg)
    largest = _largest_gap(joined.get_column("rating_diff"), joined.get_column("zero_rating_diff"))
    if largest > ZERO_TOLERANCE:
        msg = f"the 0% rows differ from the published team rating's rows by up to {largest:.3g}"
        raise ValueError(msg)
    return largest


def compare_thresholds(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return each candidate's paired MAE difference from 0% per split, at 98.33% confidence.

    ``mae_delta`` is the candidate's MAE minus the 0% MAE, so a negative value favors the filter.
    Every comparison reuses the same seed, so all three resample the same games.
    """
    return pl.concat(
        [
            compute_pairwise_mae_bootstrap(
                predictions,
                baselines=[baseline_name(threshold), baseline_name(BASELINE_THRESHOLD)],
                splits=SPLITS,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
                confidence=COMPARISON_CONFIDENCE,
            ).with_columns(pl.lit(threshold).alias("threshold"))
            for threshold in CANDIDATE_THRESHOLDS
        ]
    )


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """The outcome of the decision rule written before the first run."""

    recommended_threshold: int
    qualifying: tuple[int, ...]
    excluding_zero: tuple[int, ...]


def decide(comparisons: pl.DataFrame, scores: pl.DataFrame) -> FilterDecision:
    """Apply the decision rule to the overall comparisons and MAEs.

    A threshold qualifies when its whole interval lies below zero; the qualifying threshold with
    the lowest overall MAE is the recommendation, and with none qualifying it is 0% (no filter).
    ``excluding_zero`` lists every threshold whose interval excludes zero, in either direction.
    """
    overall = comparisons.filter(pl.col("split") == "overall").sort("threshold")
    qualifying = tuple(
        int(row["threshold"]) for row in overall.iter_rows(named=True) if row["ci_upper"] < 0.0
    )
    excluding_zero = tuple(
        int(row["threshold"])
        for row in overall.iter_rows(named=True)
        if row["ci_upper"] < 0.0 or row["ci_lower"] > 0.0
    )
    if not qualifying:
        return FilterDecision(BASELINE_THRESHOLD, (), excluding_zero)
    mae: dict[str, float] = dict(
        scores.filter(pl.col("split") == "overall").select("baseline", "mae").iter_rows()
    )
    best = min(qualifying, key=lambda threshold: mae[baseline_name(threshold)])
    return FilterDecision(best, qualifying, excluding_zero)


def kept_play_share(
    game_logs: pl.DataFrame, bins: pl.DataFrame, threshold: int
) -> tuple[float, float]:
    """Return a season's kept and total scrimmage plus special-teams plays at ``threshold``."""
    plays = pl.col(SCRIMMAGE_PLAYS_COLUMN) + pl.col(SPECIAL_TEAMS_PLAYS_COLUMN)
    kept = team_game_logs_at_threshold(game_logs, bins, threshold).select(plays.sum()).item()
    total = game_logs.select(plays.sum()).item()
    return float(kept), float(total)


def year_over_year_pearson(ratings: pl.DataFrame, key: str, value: str) -> tuple[int, float]:
    """Return the pair count and Pearson correlation of ``value`` across consecutive seasons.

    ``ratings`` holds ``season``, ``key``, and ``value``; a row pairs with the same ``key`` in the
    next season. Fewer than two pairs give NaN.
    """
    following = ratings.select(
        (pl.col("season") - 1).alias("season"), key, pl.col(value).alias("next_value")
    )
    pairs = ratings.join(following, on=["season", key], how="inner").drop_nulls(
        [value, "next_value"]
    )
    if pairs.height < _MIN_PAIRS:
        return pairs.height, math.nan
    current = pairs.get_column(value).cast(pl.Float64).to_numpy()
    upcoming = pairs.get_column("next_value").cast(pl.Float64).to_numpy()
    return pairs.height, float(np.corrcoef(current, upcoming)[0, 1])


def _read(data_dir: Path, season: int, suffix: str) -> pl.DataFrame:
    """Read one season's Parquet output."""
    return pl.read_parquet(data_dir / f"{season}_{suffix}.parquet")


@dataclass(frozen=True, slots=True)
class _TeamResults:
    """The team side: walk-forward rows, full-season ratings, and kept plays by threshold."""

    features: pl.DataFrame
    ratings: pl.DataFrame
    kept_plays: dict[int, tuple[float, float]]


def _run_team_side(data_dir: Path, seasons: Sequence[int]) -> _TeamResults:
    """Build every season's walk-forward rows and full-season ratings at every threshold."""
    features: list[pl.DataFrame] = []
    ratings: list[pl.DataFrame] = []
    kept_plays = dict.fromkeys(THRESHOLDS, (0.0, 0.0))
    for season in seasons:
        game_logs = _read(data_dir, season, "team_game_logs")
        bins = _read(data_dir, season, "team_wp_bins")
        check_zero_kept_columns(game_logs, bins, season)
        features.append(
            build_team_rating_feature_rows(game_logs, season, previous_season_fit(data_dir, season))
        )
        previous_inputs = (
            None
            if season <= PBP_START_SEASON
            else (
                _read(data_dir, season - 1, "team_game_logs"),
                _read(data_dir, season - 1, "team_wp_bins"),
            )
        )
        for threshold in THRESHOLDS:
            previous = (
                None
                if previous_inputs is None
                else previous_threshold_fit(*previous_inputs, threshold)
            )
            features.append(
                build_threshold_feature_rows(game_logs, bins, season, threshold, previous)
            )
            fit = fit_team_ratings_with_previous_penalties(
                team_game_logs_at_threshold(game_logs, bins, threshold), previous
            )
            ratings.append(
                fit.ratings.select(
                    pl.lit(season).cast(pl.Int64).alias("season"),
                    pl.lit(threshold).cast(pl.Int64).alias("threshold"),
                    "team",
                    "team_rating",
                )
            )
            kept, total = kept_play_share(game_logs, bins, threshold)
            kept_plays[threshold] = (
                kept_plays[threshold][0] + kept,
                kept_plays[threshold][1] + total,
            )
    return _TeamResults(pl.concat(features), pl.concat(ratings), kept_plays)


def _qb_ratings(data_dir: Path, seasons: Sequence[int]) -> pl.DataFrame:
    """Return every qualifying passer's full-season adjusted EPA per dropback at every threshold.

    Qualifying passers are the published ``qb_is_eligible`` ones, held fixed across thresholds.
    """
    frames: list[pl.DataFrame] = []
    for season in seasons:
        qb_games = _read(data_dir, season, "qb_game_logs")
        bins = _read(data_dir, season, "qb_wp_bins")
        qualifying = (
            _read(data_dir, season, "qb_combined")
            .filter(pl.col("qb_is_eligible"))
            .select("qb_id", "qb_name", "team")
        )
        for threshold in THRESHOLDS:
            fit = fit_qb_ratings(qb_games_at_threshold(qb_games, bins, threshold))
            frames.append(
                qualifying.join(
                    fit.ratings.select("qb_id", QB_RATING_COLUMN), on="qb_id", how="inner"
                ).with_columns(
                    pl.lit(season).cast(pl.Int64).alias("season"),
                    pl.lit(threshold).cast(pl.Int64).alias("threshold"),
                )
            )
    return pl.concat(frames)


def _say(text: str) -> None:
    """Write one line of the report to standard output."""
    sys.stdout.write(f"{text}\n")


def _interval(row: dict[str, object]) -> str:
    """Format one comparison row as its difference and interval."""
    return f"{row['mae_delta']:+.3f} ({row['ci_lower']:+.3f} to {row['ci_upper']:+.3f})"


def _report_accuracy(scores: pl.DataFrame, comparisons: pl.DataFrame) -> None:
    """Print each threshold's MAE and RMSE by split and its paired difference from 0%."""
    _say("\nMAE (RMSE) of predicted home margins by threshold; early is weeks before 8:")
    for threshold in THRESHOLDS:
        rows = {
            row["split"]: row
            for row in scores.filter(pl.col("baseline") == baseline_name(threshold)).iter_rows(
                named=True
            )
        }
        cells = "; ".join(
            f"{split} {rows[split]['mae']:.3f} ({rows[split]['rmse']:.3f})" for split in SPLITS
        )
        _say(f"  {threshold:>2}%: {cells}; {rows['overall']['games']} games")
    _say(
        f"\nPaired MAE difference from 0% (negative favors the filter), "
        f"{COMPARISON_CONFIDENCE:.2%} intervals, {BOOTSTRAP_RESAMPLES:,} resamples:"
    )
    for threshold in CANDIDATE_THRESHOLDS:
        rows = {
            row["split"]: row
            for row in comparisons.filter(pl.col("threshold") == threshold).iter_rows(named=True)
        }
        _say(
            f"  {threshold:>2}%: "
            + "; ".join(f"{split} {_interval(rows[split])}" for split in SPLITS)
        )


def report_decision(decision: FilterDecision) -> None:
    """Print the decision rule's outcome and every interval that excludes zero."""
    if decision.qualifying:
        qualifying = ", ".join(f"{threshold}%" for threshold in decision.qualifying)
        _say(
            f"\nDecision: {qualifying} qualified; the recommendation is "
            f"{decision.recommended_threshold}% (the lowest MAE among them)."
        )
    else:
        _say("\nDecision: no threshold qualified, so the recommendation is 0% (no filter).")
    excluding = ", ".join(
        f"{threshold}% ({'better' if threshold in decision.qualifying else 'worse'})"
        for threshold in decision.excluding_zero
    )
    _say(f"Overall intervals excluding zero: {excluding or 'none'}.")
    _say("The decision goes to the maintainer either way.")


def _by_threshold(values: dict[int, str]) -> str:
    """Join one formatted value per threshold, skipping thresholds without one."""
    return ", ".join(
        f"{threshold}% {values[threshold]}" for threshold in THRESHOLDS if threshold in values
    )


def _report_extras(team: _TeamResults, qbs: pl.DataFrame, qbr: pl.DataFrame) -> None:
    """Print the descriptive extras: kept share, stability, and the QBR correlation."""
    _say("\nDescriptive only, not decision inputs:")
    shares = {
        threshold: f"{kept / total:.3f}" if total else "n/a"
        for threshold, (kept, total) in team.kept_plays.items()
    }
    _say(f"  Kept play share (scrimmage and special teams): {_by_threshold(shares)}")
    team_pearson = {
        threshold: year_over_year_pearson(
            team.ratings.filter(pl.col("threshold") == threshold), "team", "team_rating"
        )
        for threshold in THRESHOLDS
    }
    pairs = team_pearson[BASELINE_THRESHOLD][0]
    _say(
        f"  Team year-over-year Pearson of team_rating ({pairs} pairs): "
        + _by_threshold({key: f"{value:.3f}" for key, (_, value) in team_pearson.items()})
    )
    qb_pearson = {
        threshold: year_over_year_pearson(
            qbs.filter(pl.col("threshold") == threshold), "qb_id", QB_RATING_COLUMN
        )
        for threshold in THRESHOLDS
    }
    pairs = qb_pearson[BASELINE_THRESHOLD][0]
    _say(
        f"  QB year-over-year Pearson of adjusted EPA per dropback, qualifying both seasons "
        f"({pairs} pairs): "
        + _by_threshold({key: f"{value:.3f}" for key, (_, value) in qb_pearson.items()})
    )
    qbr_seasons = sorted(qbr.get_column("season").unique().to_list())
    correlations: dict[int, str] = {}
    for threshold in THRESHOLDS:
        values = [
            qbr_correlation(
                qbr,
                season,
                qbs.filter((pl.col("threshold") == threshold) & (pl.col("season") == season)),
            )["pearson"]
            for season in qbr_seasons
        ]
        correlations[threshold] = f"{float(np.nanmean(values)):.3f}" if values else "n/a"
    _say(
        f"  ESPN QBR mean per-season Pearson ({len(qbr_seasons)} seasons): "
        + _by_threshold(correlations)
    )


def _rank_rows(frame: pl.DataFrame, value: str) -> pl.DataFrame:
    """Add each row's rank by ``value`` (1 is best) within its threshold."""
    return frame.with_columns(
        pl.col(value).rank("min", descending=True).over("threshold").cast(pl.Int64).alias("rank")
    )


def report_spotlight(
    team_ratings: pl.DataFrame, qb_ratings: pl.DataFrame, season: int, team: str, qb_name: str
) -> None:
    """Print one team's and one passer's ratings and ranks by threshold in one season.

    ``team_ratings`` holds ``season``, ``threshold``, ``team``, and ``team_rating``;
    ``qb_ratings`` holds ``season``, ``threshold``, ``qb_name``, and the adjusted EPA per dropback
    of the qualifying passers. Ranks are within each threshold, 1 the best.
    """
    for label, frame, key, value, decimals in (
        (team, team_ratings, "team", "team_rating", 2),
        (qb_name, qb_ratings, "qb_name", QB_RATING_COLUMN, 3),
    ):
        rows = _rank_rows(frame.filter(pl.col("season") == season), value).filter(
            pl.col(key) == label
        )
        if rows.is_empty():
            _say(f"  {label} in {season}: not in the data")
            continue
        values = {
            int(row["threshold"]): f"{row[value]:.{decimals}f} (rank {row['rank']})"
            for row in rows.iter_rows(named=True)
        }
        _say(f"  {label} in {season}: {_by_threshold(values)}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``check-wp-filter`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings check-wp-filter",
        description=(
            "Test whether rating teams without garbage-time plays predicts game margins better "
            "in the walk-forward check (reads data/, downloads ESPN QBR, writes nothing)."
        ),
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs (default: {DATA_DIR})."
    )
    parser.add_argument("--start-season", type=int, default=START_YEAR, help="First scored season.")
    parser.add_argument("--end-season", type=int, default=END_YEAR, help="Last scored season.")
    parser.add_argument(
        "--start-week", type=int, default=DEFAULT_START_WEEK, help="First scored week."
    )
    parser.add_argument(
        "--spotlight-season",
        type=int,
        default=DEFAULT_SPOTLIGHT_SEASON,
        help="Season of the team and passer shown by threshold.",
    )
    parser.add_argument(
        "--spotlight-team", default=DEFAULT_SPOTLIGHT_TEAM, help="Team shown by threshold."
    )
    parser.add_argument(
        "--spotlight-qb", default=DEFAULT_SPOTLIGHT_QB, help="Passer shown by threshold."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the garbage-time filter test and print the integrity check, results, and extras.

    Raises:
        ValueError: If 0% does not reproduce the published team rating.

    """
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    seasons = list(range(args.start_season, args.end_season + 1))
    team = _run_team_side(data_dir, seasons)
    largest = check_zero_threshold_rows(team.features)
    predictions = evaluate_feature_rows(
        team.features.filter(pl.col("baseline") != TEAM_RATING_BASELINE),
        start_week=args.start_week,
    )
    scores = score_prediction_rows(predictions)
    comparisons = compare_thresholds(predictions)
    decision = decide(comparisons, scores)

    _say(
        f"Garbage-time filter test, {args.start_season}-{args.end_season}, prediction weeks "
        f"{args.start_week} and later: thresholds "
        + ", ".join(f"{threshold}%" for threshold in CANDIDATE_THRESHOLDS)
        + " against 0% (every play)"
    )
    _say(
        f"Integrity check passed: at 0% the kept plays equal the game logs in every season, and "
        f"the 0% rows match the published team rating's (largest gap {largest:.1e})."
    )
    _report_accuracy(scores, comparisons)
    report_decision(decision)
    qbs = _qb_ratings(data_dir, seasons)
    _report_extras(team, qbs, load_season_qbr(seasons))
    report_spotlight(
        team.ratings, qbs, args.spotlight_season, args.spotlight_team, args.spotlight_qb
    )


__all__ = [
    "CANDIDATE_THRESHOLDS",
    "COMPARISON_CONFIDENCE",
    "FilterDecision",
    "baseline_name",
    "build_threshold_feature_rows",
    "check_zero_kept_columns",
    "check_zero_threshold_rows",
    "compare_thresholds",
    "decide",
    "kept_play_share",
    "main",
    "previous_threshold_fit",
    "report_decision",
    "report_spotlight",
    "year_over_year_pearson",
]
