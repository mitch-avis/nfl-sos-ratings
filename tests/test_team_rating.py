"""Tests for the points-based team rating and its head-to-head-excluded schedule strength."""

import itertools

import polars as pl
import pytest

from nfl_sos_ratings.team_rating import compute_team_schedule_strength, fit_team_ratings

# The first four teams have mean-zero effects, so a round-robin among them recovers them exactly.
_OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": -0.10, "EEE": 0.03, "FFF": -0.02}
_DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.00, "DDD": -0.04, "EEE": 0.02, "FFF": 0.06}
_ST_POSSESSION = {"AAA": 0.20, "BBB": -0.10, "CCC": 0.00, "DDD": -0.10, "EEE": 0.0, "FFF": 0.1}
_ST_COVERAGE = {"AAA": 0.05, "BBB": 0.05, "CCC": -0.05, "DDD": -0.05, "EEE": 0.0, "FFF": 0.0}
_ROUND_ROBIN = tuple(itertools.permutations(("AAA", "BBB", "CCC", "DDD"), 2))
# Six teams that each play only three opponents (home and away), so opponent sets differ.
_PARTIAL_PAIRS = (
    ("AAA", "BBB"),
    ("BBB", "CCC"),
    ("CCC", "DDD"),
    ("DDD", "EEE"),
    ("EEE", "FFF"),
    ("FFF", "AAA"),
    ("AAA", "CCC"),
    ("DDD", "FFF"),
    ("BBB", "EEE"),
)
_PARTIAL = tuple(pair for home, away in _PARTIAL_PAIRS for pair in ((home, away), (away, home)))
_SCRIMMAGE_PLAYS = 60
_ST_PLAYS = 12
_TINY_LAMBDA = 1e-6


def _game_logs(games: tuple[tuple[str, str], ...] = _ROUND_ROBIN) -> pl.DataFrame:
    """Return noiseless team-game rows for ``(home, away)`` games built from known effects."""
    rows: list[dict[str, object]] = []
    for game_number, (home, away) in enumerate(games):
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            sign = 1.0 if is_home else -1.0
            epa_per_play = 0.02 + _OFFENSE[team] - _DEFENSE[opponent] + 0.01 * sign
            st_per_play = _ST_POSSESSION[team] - _ST_COVERAGE[opponent]
            rows.append(
                {
                    "game_id": f"g{game_number:02d}",
                    "week": game_number + 1,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "offensive_snaps": _SCRIMMAGE_PLAYS,
                    "offensive_epa": epa_per_play * _SCRIMMAGE_PLAYS,
                    "st_plays": _ST_PLAYS,
                    "st_epa": st_per_play * _ST_PLAYS,
                }
            )
    return pl.DataFrame(rows)


def _rating(ratings: pl.DataFrame, team: str, column: str) -> float:
    """Return one team's value for one rating column."""
    return float(ratings.filter(pl.col("team") == team).get_column(column).item())


def test_fit_team_ratings_team_rating_is_sum_of_unit_ratings() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(game_logs).ratings

    # Assert
    summed = ratings.select(
        pl.col("offense_rating") + pl.col("defense_rating") + pl.col("special_teams_rating")
    ).to_series()
    assert ratings.get_column("team_rating").to_list() == pytest.approx(summed.to_list())


def test_fit_team_ratings_offense_rating_is_points_per_game() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    assert _rating(ratings, "AAA", "offense_rating") == pytest.approx(0.10 * 60, abs=1e-4)


def test_fit_team_ratings_better_defense_has_positive_defense_rating() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    assert _rating(ratings, "BBB", "defense_rating") == pytest.approx(0.08 * 60, abs=1e-4)


def test_fit_team_ratings_special_teams_rating_adds_both_special_teams_units() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    expected = (0.20 + 0.05) * 12
    assert _rating(ratings, "AAA", "special_teams_rating") == pytest.approx(expected, abs=1e-4)


def test_fit_team_ratings_missing_special_teams_columns_raises_value_error() -> None:
    # Arrange
    game_logs = _game_logs().drop("st_plays")

    # Act & Assert
    with pytest.raises(ValueError, match="st_plays"):
        fit_team_ratings(game_logs)


def test_compute_team_schedule_strength_ignores_the_subject_teams_own_games() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    blowouts = game_logs.with_columns(
        pl.when((pl.col("team") == "AAA") | (pl.col("opponent_team") == "AAA"))
        .then(pl.col("offensive_epa") * 3.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    lambdas = {"scrimmage_lambda": 10.0, "special_teams_lambda": 10.0}
    baseline_fit = fit_team_ratings(game_logs, **lambdas)
    blowout_fit = fit_team_ratings(blowouts, **lambdas)

    # Act
    baseline_sos = compute_team_schedule_strength(game_logs, baseline_fit)
    blowout_sos = compute_team_schedule_strength(blowouts, blowout_fit)

    # Assert
    assert _rating(blowout_sos, "AAA", "sos") == pytest.approx(_rating(baseline_sos, "AAA", "sos"))


def test_compute_team_schedule_strength_other_teams_see_the_changed_games() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    blowouts = game_logs.with_columns(
        pl.when((pl.col("team") == "AAA") | (pl.col("opponent_team") == "AAA"))
        .then(pl.col("offensive_epa") * 3.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    lambdas = {"scrimmage_lambda": 10.0, "special_teams_lambda": 10.0}
    baseline_fit = fit_team_ratings(game_logs, **lambdas)
    blowout_fit = fit_team_ratings(blowouts, **lambdas)

    # Act
    baseline_sos = compute_team_schedule_strength(game_logs, baseline_fit)
    blowout_sos = compute_team_schedule_strength(blowouts, blowout_fit)

    # Assert
    assert _rating(blowout_sos, "BBB", "sos") != pytest.approx(_rating(baseline_sos, "BBB", "sos"))


def test_fit_team_ratings_ignores_game_outcomes() -> None:
    # Arrange
    game_logs = _game_logs().with_columns(
        pl.lit(0).alias("point_margin"), pl.lit(0.0).alias("win_value")
    )
    flipped = game_logs.with_columns(
        pl.lit(30).alias("point_margin"), pl.lit(1.0).alias("win_value")
    )
    baseline = fit_team_ratings(game_logs).ratings

    # Act
    result = fit_team_ratings(flipped).ratings

    # Assert
    assert result.equals(baseline)


def test_compute_team_schedule_strength_opponent_without_other_games_raises_value_error() -> None:
    # Arrange
    game_logs = _game_logs((("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC")))
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Act & Assert
    with pytest.raises(ValueError, match="no head-to-head-excluded rating"):
        compute_team_schedule_strength(game_logs, fit)
