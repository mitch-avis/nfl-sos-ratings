"""Tests for the walk-forward test of the garbage-time filter."""

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.team_rating import fit_team_ratings, fit_team_ratings_with_previous_penalties
from nfl_sos_ratings.validation import walk_forward
from nfl_sos_ratings.validation.walk_forward import (
    TEAM_RATING_BASELINE,
    build_snapshot_feature_rows,
    build_team_rating_feature_rows,
)
from nfl_sos_ratings.validation.wp_filter_check import (
    FilterDecision,
    baseline_name,
    build_threshold_feature_rows,
    check_zero_kept_columns,
    check_zero_threshold_rows,
    compare_thresholds,
    decide,
    kept_play_share,
    main,
    previous_threshold_fit,
    report_decision,
    report_spotlight,
    year_over_year_pearson,
)
from tests.stubs import stub
from tests.wp_league import (
    CLOSE_PLAYS,
    LOPSIDED_PLAYS,
    ST_CLOSE_PLAYS,
    ST_UNBINNED_PLAYS,
    qb_bins,
    qb_games,
    team_bins,
    team_game_logs,
)

if TYPE_CHECKING:
    from pathlib import Path

_SEASONS = (2023, 2024, 2025)


def _with_margin(logs: pl.DataFrame) -> pl.DataFrame:
    """Add a point margin that follows each side's EPA per play, as the harness needs."""
    rate = logs.select(
        "game_id", "team", (pl.col("offensive_epa") / pl.col("offensive_snaps")).alias("_rate")
    )
    return (
        logs.join(rate, on=["game_id", "team"])
        .join(
            rate.rename({"team": "opponent_team", "_rate": "_opponent_rate"}),
            on=["game_id", "opponent_team"],
        )
        .with_columns((60 * (pl.col("_rate") - pl.col("_opponent_rate"))).alias("point_margin"))
        .drop("_rate", "_opponent_rate")
        .sort("game_id", "team")
    )


def _league() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the league's team game logs (with margins) and its bins as a season writes them."""
    bins = team_bins()
    return _with_margin(team_game_logs(bins)), bins.drop("is_home")


def _rating_diffs(rows: pl.DataFrame) -> list[float]:
    """Return the rows' rating gaps in game order."""
    return rows.sort("season", "week", "game_id").get_column("rating_diff").to_list()


def test_threshold_rows_at_zero_equal_the_published_rating_rows() -> None:
    # Arrange
    logs, bins = _league()
    previous = previous_threshold_fit(logs, bins, 0)

    # Act
    rows = build_threshold_feature_rows(logs, bins, 2025, 0, previous)

    # Assert
    published = build_team_rating_feature_rows(logs, 2025, fit_team_ratings(logs))
    assert _rating_diffs(rows) == pytest.approx(_rating_diffs(published))
    assert rows.get_column("baseline").unique().to_list() == [baseline_name(0)]


def test_threshold_rows_fit_the_kept_plays_with_penalties_tuned_on_them() -> None:
    # Arrange
    logs, bins = _league()
    previous = previous_threshold_fit(logs, bins, 10)
    kept = team_game_logs(team_bins(), 10).join(
        logs.select("game_id", "team", "point_margin"), on=["game_id", "team"]
    )
    kept_previous = fit_team_ratings(kept)

    def snapshot(prior: pl.DataFrame) -> pl.DataFrame:
        """Rate the pre-week kept plays with penalties tuned on last season's kept plays."""
        return fit_team_ratings_with_previous_penalties(prior, kept_previous).ratings.select(
            "team", pl.col("team_rating").alias("rating")
        )

    # Act
    rows = build_threshold_feature_rows(logs, bins, 2025, 10, previous)

    # Assert
    expected = build_snapshot_feature_rows(kept, 2025, baseline_name(10), snapshot)
    assert _rating_diffs(rows) == pytest.approx(_rating_diffs(expected))


def test_zero_kept_columns_check_accepts_complete_bins() -> None:
    # Arrange
    logs, bins = _league()

    # Act
    largest = check_zero_kept_columns(logs, bins, 2025)

    # Assert
    assert largest == pytest.approx(0.0)


def test_zero_kept_columns_check_rejects_bins_missing_a_game() -> None:
    # Arrange
    logs, bins = _league()
    missing = bins.filter(pl.col("game_id") != "g00")

    # Act & Assert
    with pytest.raises(ValueError, match="2025"):
        check_zero_kept_columns(logs, missing, 2025)


def _feature_rows(baseline: str, diffs: list[float | None]) -> pl.DataFrame:
    """Return walk-forward rows for one baseline with the given rating gaps."""
    count = len(diffs)
    return pl.DataFrame(
        {
            "season": [2025] * count,
            "week": list(range(5, 5 + count)),
            "baseline": [baseline] * count,
            "game_id": [f"g{number}" for number in range(count)],
            "home_team": ["A"] * count,
            "away_team": ["B"] * count,
            "rating_diff": diffs,
            "home_margin": [3.0] * count,
        }
    )


def test_zero_threshold_rows_check_accepts_identical_rows() -> None:
    # Arrange
    features = pl.concat(
        [
            _feature_rows(TEAM_RATING_BASELINE, [1.0, -2.0, 0.5]),
            _feature_rows(baseline_name(0), [1.0, -2.0, 0.5]),
        ]
    )

    # Act
    largest = check_zero_threshold_rows(features)

    # Assert
    assert largest == pytest.approx(0.0)


def test_zero_threshold_rows_check_rejects_a_different_rating_gap() -> None:
    # Arrange
    features = pl.concat(
        [
            _feature_rows(TEAM_RATING_BASELINE, [1.0, -2.0, 0.5]),
            _feature_rows(baseline_name(0), [1.0, -2.0, 0.6]),
        ]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="0%"):
        check_zero_threshold_rows(features)


def test_zero_threshold_rows_check_rejects_missing_games() -> None:
    # Arrange
    features = pl.concat(
        [
            _feature_rows(TEAM_RATING_BASELINE, [1.0, -2.0, 0.5]),
            _feature_rows(baseline_name(0), [1.0, -2.0]),
        ]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="same games"):
        check_zero_threshold_rows(features)


def test_zero_threshold_rows_check_rejects_a_missing_rating_gap() -> None:
    # Arrange
    features = pl.concat(
        [
            _feature_rows(TEAM_RATING_BASELINE, [1.0, -2.0, 0.5]),
            _feature_rows(baseline_name(0), [1.0, -2.0, None]),
        ]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="0%"):
        check_zero_threshold_rows(features)


def test_compare_thresholds_subtracts_the_zero_threshold_error() -> None:
    # Arrange
    games = 6
    errors = {0: 2.0, 5: 1.0, 10: 2.5, 20: 2.0}
    predictions = pl.DataFrame(
        {
            "season": [2025] * games * len(errors),
            "week": [5, 6, 7, 8, 9, 10] * len(errors),
            "baseline": [baseline_name(threshold) for threshold in errors for _ in range(games)],
            "game_id": [f"g{number}" for number in range(games)] * len(errors),
            "home_team": ["A"] * games * len(errors),
            "away_team": ["B"] * games * len(errors),
            "error": [error for error in errors.values() for _ in range(games)],
        }
    )

    # Act
    comparisons = compare_thresholds(predictions)

    # Assert
    overall = comparisons.filter(pl.col("split") == "overall").sort("threshold")
    assert overall.get_column("threshold").to_list() == [5, 10, 20]
    assert overall.get_column("mae_delta").to_list() == pytest.approx([-1.0, 0.5, 0.0])


def _decision_inputs(
    intervals: dict[int, tuple[float, float]], mae: dict[int, float]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return overall comparison rows and scores for the given intervals and MAEs."""
    comparisons = pl.DataFrame(
        {
            "threshold": list(intervals),
            "split": ["overall"] * len(intervals),
            "mae_delta": [(low + high) / 2 for low, high in intervals.values()],
            "ci_lower": [low for low, _ in intervals.values()],
            "ci_upper": [high for _, high in intervals.values()],
        }
    )
    scores = pl.DataFrame(
        {
            "baseline": [baseline_name(threshold) for threshold in mae],
            "split": ["overall"] * len(mae),
            "mae": list(mae.values()),
        }
    )
    return comparisons, scores


def test_decide_without_a_qualifying_threshold_recommends_no_filter() -> None:
    # Arrange
    comparisons, scores = _decision_inputs(
        {5: (-0.05, 0.02), 10: (-0.04, 0.03), 20: (-0.02, 0.06)},
        {0: 10.6, 5: 10.59, 10: 10.6, 20: 10.62},
    )

    # Act
    decision = decide(comparisons, scores)

    # Assert
    assert (decision.recommended_threshold, decision.qualifying) == (0, ())
    assert decision.excluding_zero == ()


def test_decide_reports_a_threshold_that_is_significantly_worse() -> None:
    # Arrange
    comparisons, scores = _decision_inputs(
        {5: (-0.05, 0.02), 10: (-0.04, 0.03), 20: (0.01, 0.09)},
        {0: 10.6, 5: 10.59, 10: 10.6, 20: 10.65},
    )

    # Act
    decision = decide(comparisons, scores)

    # Assert
    assert (decision.recommended_threshold, decision.qualifying) == (0, ())
    assert decision.excluding_zero == (20,)


def test_decide_recommends_the_qualifying_threshold_with_the_lowest_mae() -> None:
    # Arrange
    comparisons, scores = _decision_inputs(
        {5: (-0.09, -0.01), 10: (-0.12, -0.02), 20: (-0.05, 0.04)},
        {0: 10.6, 5: 10.55, 10: 10.53, 20: 10.6},
    )

    # Act
    decision = decide(comparisons, scores)

    # Assert
    assert decision.recommended_threshold == 10
    assert decision.qualifying == (5, 10)
    assert decision.excluding_zero == (5, 10)


def test_report_decision_names_the_qualifying_thresholds_and_their_direction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    decision = FilterDecision(
        recommended_threshold=10, qualifying=(5, 10), excluding_zero=(5, 10, 20)
    )

    # Act
    report_decision(decision)

    # Assert
    output = capsys.readouterr().out
    assert "5%, 10% qualified; the recommendation is 10%" in output
    assert "5% (better), 10% (better), 20% (worse)" in output


def test_report_spotlight_says_when_the_team_or_passer_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    team_ratings = pl.DataFrame(
        {"season": [2025, 2025], "threshold": [0, 0], "team": ["A", "B"], "team_rating": [2.0, 1.0]}
    )
    qb_ratings = pl.DataFrame(
        {
            "season": [2025],
            "threshold": [0],
            "qb_name": ["One"],
            "adj_qb_epa_per_dropback": [0.1],
        }
    )

    # Act
    report_spotlight(team_ratings, qb_ratings, 2025, "B", "Nobody")

    # Assert
    output = capsys.readouterr().out
    assert "B in 2025: 0% 1.00 (rank 2)" in output
    assert "Nobody in 2025: not in the data" in output


def test_kept_play_share_counts_scrimmage_and_special_teams_plays() -> None:
    # Arrange
    logs, bins = _league()

    # Act
    kept, total = kept_play_share(logs, bins, 10)

    # Assert
    per_game_kept = CLOSE_PLAYS + ST_CLOSE_PLAYS + ST_UNBINNED_PLAYS
    per_game_total = per_game_kept + LOPSIDED_PLAYS
    assert kept / total == pytest.approx(per_game_kept / per_game_total)


def test_year_over_year_pearson_pairs_consecutive_seasons_by_key() -> None:
    # Arrange
    ratings = pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2024, 2024, 2024, 2026],
            "team": ["A", "B", "C", "A", "B", "D", "A"],
            "team_rating": [1.0, 2.0, 3.0, 2.0, 4.0, 9.0, 5.0],
        }
    )

    # Act
    pairs, pearson = year_over_year_pearson(ratings, "team", "team_rating")

    # Assert
    assert pairs == 2
    assert pearson == pytest.approx(1.0)


def test_year_over_year_pearson_without_two_pairs_is_undefined() -> None:
    # Arrange
    ratings = pl.DataFrame({"season": [2023, 2024], "team": ["A", "A"], "team_rating": [1.0, 2.0]})

    # Act
    pairs, pearson = year_over_year_pearson(ratings, "team", "team_rating")

    # Assert
    assert pairs == 1
    assert math.isnan(pearson)


def _write_seasons(data_dir: Path) -> None:
    """Write three identical synthetic seasons with every file the check reads."""
    logs, bins = _league()
    passer_bins = qb_bins()
    official = qb_games(passer_bins).with_columns(pl.col("qb_epa_per_dropback") + 0.01)
    combined = (
        official.select("qb_id")
        .unique()
        .sort("qb_id")
        .with_columns(
            pl.col("qb_id").str.replace("qb-", "Passer ").alias("qb_name"),
            pl.col("qb_id").str.replace("qb-", "").alias("team"),
            pl.lit(value=True).alias("qb_is_eligible"),
        )
    )
    for season in _SEASONS:
        logs.write_parquet(data_dir / f"{season}_team_game_logs.parquet")
        bins.write_parquet(data_dir / f"{season}_team_wp_bins.parquet")
        official.write_parquet(data_dir / f"{season}_qb_game_logs.parquet")
        passer_bins.drop("opponent_team").write_parquet(data_dir / f"{season}_qb_wp_bins.parquet")
        combined.write_parquet(data_dir / f"{season}_qb_combined.parquet")


def test_main_prints_the_integrity_check_decision_and_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_seasons(tmp_path)
    qbr = pl.DataFrame(
        {
            "season": [season for season in _SEASONS[1:] for _ in range(4)],
            "game_week": ["Season Total"] * 8,
            "team_abb": ["AAA", "BBB", "CCC", "DDD"] * 2,
            "name_display": ["Passer AAA", "Passer BBB", "Passer CCC", "Passer DDD"] * 2,
            "qbr_total": [80.0, 60.0, 40.0, 20.0] * 2,
        }
    )
    monkeypatch.setattr(walk_forward, "load_espn_qbr", stub(lambda: qbr))

    # Act
    main(
        [
            "--data-dir",
            str(tmp_path),
            "--start-season",
            "2024",
            "--end-season",
            "2025",
            "--spotlight-team",
            "AAA",
            "--spotlight-qb",
            "Passer DDD",
        ]
    )

    # Assert
    output = capsys.readouterr().out
    assert "Integrity check passed" in output
    assert "Decision:" in output
    assert "Kept play share" in output
    assert "Team year-over-year Pearson" in output
    assert "QB year-over-year Pearson" in output
    assert "ESPN QBR" in output
    assert "AAA in 2025" in output
    assert "Passer DDD in 2025" in output
