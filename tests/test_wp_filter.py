"""Tests for team and QB ratings refit with garbage-time plays filtered out."""

import polars as pl
import pytest

from nfl_sos_ratings.qb_rating import (
    QbRatingFit,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
)
from nfl_sos_ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    TeamRatingFit,
    compute_team_schedule_strength,
    fit_team_ratings,
)
from nfl_sos_ratings.wp_filter import (
    MAX_WP_THRESHOLD,
    QbWpFilter,
    TeamWpFilter,
    qb_games_at_threshold,
    team_game_logs_at_threshold,
)
from tests.wp_league import (
    CLOSE_PLAYS,
    LAMBDA,
    LOPSIDED_BIN,
    LOPSIDED_PLAYS,
    OFFENSE,
    PASSER,
    ST_CLOSE_PLAYS,
    ST_UNBINNED_PLAYS,
    qb_bins,
    qb_games,
    team_bins,
    team_game_logs,
)


def _season_fit(logs: pl.DataFrame) -> TeamRatingFit:
    """Return the season fit with fixed penalties, as the published ratings use."""
    return fit_team_ratings(logs, scrimmage_lambda=LAMBDA, special_teams_lambda=LAMBDA)


def _at_season_scale(kept_logs: pl.DataFrame, season: TeamRatingFit) -> pl.DataFrame:
    """Fit only the kept plays with the season's penalties, at the season's plays per game."""
    kept = fit_team_ratings(
        kept_logs,
        scrimmage_lambda=season.scrimmage_lambda,
        special_teams_lambda=season.special_teams_lambda,
    )
    scrimmage = season.scrimmage_plays_per_game / kept.scrimmage_plays_per_game
    special = season.special_teams_plays_per_game / kept.special_teams_plays_per_game
    return (
        kept.ratings.with_columns(
            pl.col("offense_rating") * scrimmage,
            pl.col("defense_rating") * scrimmage,
            pl.col("special_teams_rating") * special,
        )
        .with_columns(
            (
                pl.col("offense_rating") + pl.col("defense_rating") + pl.col("special_teams_rating")
            ).alias("team_rating")
        )
        .sort("team")
    )


def _column(frame: pl.DataFrame, column: str) -> list[float]:
    """Return one column sorted by team."""
    return frame.sort("team").get_column(column).to_list()


def test_team_ratings_at_zero_reproduce_the_season_fit() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    season = _season_fit(logs)

    # Act
    filtered = TeamWpFilter(logs, bins, season).ratings(0)

    # Assert
    for column in TEAM_RATING_COLUMNS:
        assert _column(filtered, column) == pytest.approx(_column(season.ratings, column))


def test_team_schedule_strength_at_zero_reproduces_the_published_sos() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    season = _season_fit(logs)

    # Act
    filtered = TeamWpFilter(logs, bins, season).ratings(0)

    # Assert
    published = compute_team_schedule_strength(logs, season)
    assert _column(filtered, "sos") == pytest.approx(_column(published, "sos"))


def test_team_ratings_at_a_threshold_match_a_fit_on_the_kept_plays() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    season = _season_fit(logs)
    expected = _at_season_scale(team_game_logs(bins, LOPSIDED_BIN + 1), season)

    # Act
    filtered = TeamWpFilter(logs, bins, season).ratings(LOPSIDED_BIN + 1)

    # Assert
    for column in TEAM_RATING_COLUMNS:
        assert _column(filtered, column) == pytest.approx(_column(expected, column))


def test_team_filter_pulls_down_the_offense_padded_in_lopsided_games() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    season = _season_fit(logs)
    model = TeamWpFilter(logs, bins, season)

    # Act
    change = model.ratings(LOPSIDED_BIN + 1).sort("team").get_column(
        "offense_rating"
    ) - model.ratings(0).sort("team").get_column("offense_rating")

    # Assert
    by_team = dict(zip(sorted(OFFENSE), change.to_list(), strict=True))
    assert by_team["DDD"] < -1.0
    assert all(value > 0.0 for team, value in by_team.items() if team != "DDD")


def test_team_schedule_strength_at_a_threshold_matches_head_to_head_refits() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    season = _season_fit(logs)
    kept_logs = team_game_logs(bins, LOPSIDED_BIN + 1)

    # Act
    filtered = TeamWpFilter(logs, bins, season).ratings(LOPSIDED_BIN + 1)

    # Assert
    expected = compute_team_schedule_strength(kept_logs, season)
    assert _column(filtered, "sos") == pytest.approx(_column(expected, "sos"))


def test_team_kept_play_share_keeps_plays_without_a_bin() -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)

    # Act
    filtered = TeamWpFilter(logs, bins, _season_fit(logs)).ratings(MAX_WP_THRESHOLD)

    # Assert
    kept = CLOSE_PLAYS + ST_CLOSE_PLAYS + ST_UNBINNED_PLAYS
    total = CLOSE_PLAYS + LOPSIDED_PLAYS + ST_CLOSE_PLAYS + ST_UNBINNED_PLAYS
    assert filtered.get_column("wp_kept_play_share").to_list() == pytest.approx(
        [kept / total] * len(OFFENSE)
    )


@pytest.mark.parametrize("threshold", [-1, MAX_WP_THRESHOLD + 1])
def test_team_filter_rejects_a_threshold_outside_the_slider(threshold: int) -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)
    model = TeamWpFilter(logs, bins, _season_fit(logs))

    # Act & Assert
    with pytest.raises(ValueError, match="threshold"):
        model.ratings(threshold)


def _official(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return the rows with an official EPA that differs from the play-level sum."""
    return qb_games.with_columns(pl.col("qb_epa_per_dropback") + 0.01)


def _qb_column(frame: pl.DataFrame, column: str) -> list[float]:
    """Return one column sorted by passer."""
    return frame.sort("qb_id").get_column(column).to_list()


def test_qb_ratings_at_zero_use_play_level_epa_with_the_season_penalty() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))
    season = fit_qb_ratings(official, ridge_lambda=LAMBDA)

    # Act
    filtered = QbWpFilter(official, bins, season).ratings(0)

    # Assert
    expected = fit_qb_ratings(qb_games(bins), ridge_lambda=LAMBDA).ratings
    assert _qb_column(filtered, "adj_qb_epa_per_dropback") == pytest.approx(
        _qb_column(expected, "adj_qb_epa_per_dropback")
    )


def test_qb_ratings_at_a_threshold_match_a_fit_on_the_kept_dropbacks() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))
    season: QbRatingFit = fit_qb_ratings(official, ridge_lambda=LAMBDA)
    kept = qb_games(bins, 2)

    # Act
    filtered = QbWpFilter(official, bins, season).ratings(2)

    # Assert
    expected = fit_qb_ratings(kept, ridge_lambda=LAMBDA).ratings
    assert _qb_column(filtered, "adj_qb_epa_per_dropback") == pytest.approx(
        _qb_column(expected, "adj_qb_epa_per_dropback")
    )


def test_qb_faced_pass_defense_at_a_threshold_matches_head_to_head_refits() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))
    season = fit_qb_ratings(official, ridge_lambda=LAMBDA)
    kept = qb_games(bins, 2)

    # Act
    filtered = QbWpFilter(official, bins, season).ratings(2)

    # Assert
    expected = compute_qb_faced_pass_defense(kept, fit_qb_ratings(kept, ridge_lambda=LAMBDA))
    assert _qb_column(filtered, "qb_faced_pass_defense") == pytest.approx(
        _qb_column(expected, "qb_faced_pass_defense")
    )


def test_qb_filter_reports_kept_dropbacks_and_their_play_level_epa() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))
    model = QbWpFilter(official, bins, fit_qb_ratings(official, ridge_lambda=LAMBDA))
    kept = bins.filter(pl.col("wp_bin") >= 2)
    expected = (
        kept.group_by("qb_id")
        .agg(
            pl.col("qb_wp_bin_dropbacks").sum().alias("qb_dropbacks"),
            (pl.col("qb_wp_bin_epa").sum() / pl.col("qb_wp_bin_dropbacks").sum()).alias(
                "qb_epa_per_dropback"
            ),
        )
        .sort("qb_id")
    )

    # Act
    filtered = model.ratings(2)

    # Assert
    assert _qb_column(filtered, "qb_dropbacks") == expected.get_column("qb_dropbacks").to_list()
    assert _qb_column(filtered, "qb_epa_per_dropback") == pytest.approx(
        expected.get_column("qb_epa_per_dropback").to_list()
    )
    assert filtered.get_column("wp_kept_dropback_share").to_list() == pytest.approx(
        [30 / 38] * len(PASSER)
    )


_TEAM_PLAY_COLUMNS = ("offensive_snaps", "offensive_epa", "st_plays", "st_epa")


def _with_margin(logs: pl.DataFrame) -> pl.DataFrame:
    """Add a column the filter must carry through untouched."""
    return logs.with_columns(pl.col("week").cast(pl.Float64).alias("point_margin"))


def test_team_game_logs_at_threshold_hold_the_kept_plays_and_epa() -> None:
    # Arrange
    bins = team_bins()
    logs = _with_margin(team_game_logs(bins))

    # Act
    kept = team_game_logs_at_threshold(logs, bins.drop("is_home"), 10)

    # Assert
    expected = team_game_logs(bins, 10)
    for column in _TEAM_PLAY_COLUMNS:
        assert kept.sort("game_id", "team").get_column(column).to_list() == pytest.approx(
            expected.get_column(column).to_list()
        )


def test_team_game_logs_at_threshold_keep_the_other_columns() -> None:
    # Arrange
    bins = team_bins()
    logs = _with_margin(team_game_logs(bins))

    # Act
    kept = team_game_logs_at_threshold(logs, bins.drop("is_home"), 10)

    # Assert
    assert kept.columns == logs.columns
    assert kept.schema == logs.schema
    assert kept.drop(_TEAM_PLAY_COLUMNS).equals(logs.drop(_TEAM_PLAY_COLUMNS))


def test_team_game_logs_at_zero_equal_the_game_logs() -> None:
    # Arrange
    bins = team_bins()
    logs = _with_margin(team_game_logs(bins))

    # Act
    kept = team_game_logs_at_threshold(logs, bins.drop("is_home"), 0)

    # Assert
    for column in _TEAM_PLAY_COLUMNS:
        assert kept.get_column(column).to_list() == pytest.approx(logs.get_column(column).to_list())


def test_team_game_logs_at_threshold_give_a_game_without_bins_no_plays() -> None:
    # Arrange
    bins = team_bins()
    logs = _with_margin(team_game_logs(bins))
    without_first_game = bins.drop("is_home").filter(pl.col("game_id") != "g00")

    # Act
    kept = team_game_logs_at_threshold(logs, without_first_game, 0)

    # Assert
    first_game = kept.filter(pl.col("game_id") == "g00")
    for column in _TEAM_PLAY_COLUMNS:
        assert first_game.get_column(column).to_list() == [0, 0]


def test_qb_games_at_threshold_hold_kept_dropbacks_and_their_play_level_epa() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))

    # Act
    kept = qb_games_at_threshold(official, bins.drop("opponent_team"), 10)

    # Assert
    expected = qb_games(bins, 10)
    ordered = kept.sort("game_id", "qb_id")
    assert (
        ordered.get_column("qb_dropbacks").to_list()
        == expected.get_column("qb_dropbacks").to_list()
    )
    assert ordered.get_column("qb_epa_per_dropback").to_list() == pytest.approx(
        expected.get_column("qb_epa_per_dropback").to_list()
    )
    assert kept.drop("qb_dropbacks", "qb_epa_per_dropback").equals(
        official.drop("qb_dropbacks", "qb_epa_per_dropback")
    )


def test_qb_games_at_threshold_give_a_game_without_bins_no_dropbacks() -> None:
    # Arrange
    bins = qb_bins()
    official = _official(qb_games(bins))
    without_first_game = bins.drop("opponent_team").filter(pl.col("game_id") != "q00")

    # Act
    kept = qb_games_at_threshold(official, without_first_game, 0)

    # Assert
    first_game = kept.filter(pl.col("game_id") == "q00")
    assert first_game.get_column("qb_dropbacks").to_list() == [0, 0]
    assert first_game.get_column("qb_epa_per_dropback").null_count() == 2


@pytest.mark.parametrize("threshold", [-1, MAX_WP_THRESHOLD + 1])
def test_threshold_game_rows_reject_a_threshold_outside_the_slider(threshold: int) -> None:
    # Arrange
    bins = team_bins()
    logs = team_game_logs(bins)

    # Act & Assert
    with pytest.raises(ValueError, match="threshold"):
        team_game_logs_at_threshold(logs, bins.drop("is_home"), threshold)
