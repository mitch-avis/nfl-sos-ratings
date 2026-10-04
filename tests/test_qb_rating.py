"""Tests for the adjusted-EPA-per-dropback QB rating and the QB's faced pass defense."""

import itertools

import polars as pl
import pytest

from nfl_sos_ratings.qb_rating import compute_qb_faced_pass_defense, fit_qb_ratings

# Each team has one passer, named after it. The effects average to zero across teams.
_PASSER = {"AAA": 0.15, "BBB": 0.05, "CCC": -0.05, "DDD": -0.15}
_DEFENSE = {"AAA": 0.04, "BBB": -0.02, "CCC": 0.06, "DDD": -0.08}
_LEAGUE = 0.03
_TINY_LAMBDA = 1e-6


def _qb_games(dropbacks: dict[str, int] | None = None) -> pl.DataFrame:
    """Return noiseless QB-game rows for a double round-robin built from known effects."""
    volume = dropbacks or {}
    rows: list[dict[str, object]] = []
    for game_number, (home, away) in enumerate(itertools.permutations(_PASSER, 2)):
        for team, opponent in ((home, away), (away, home)):
            rows.append(
                {
                    "game_id": f"g{game_number:02d}",
                    "qb_id": f"qb_{team}",
                    "team": team,
                    "opponent_team": opponent,
                    "qb_dropbacks": volume.get(opponent, 35),
                    "qb_epa_per_dropback": _LEAGUE + _PASSER[team] - _DEFENSE[opponent],
                }
            )
    return pl.DataFrame(rows)


def _value(frame: pl.DataFrame, qb_id: str, column: str) -> float:
    """Return one QB's value for one column."""
    return float(frame.filter(pl.col("qb_id") == qb_id).get_column(column).item())


def test_fit_qb_ratings_adjusted_epa_is_league_average_plus_passer_effect() -> None:
    # Arrange
    qb_games = _qb_games()

    # Act
    ratings = fit_qb_ratings(qb_games, ridge_lambda=_TINY_LAMBDA).ratings

    # Assert
    assert _value(ratings, "qb_AAA", "adj_qb_epa_per_dropback") == pytest.approx(
        _LEAGUE + 0.15, abs=1e-6
    )


def test_compute_qb_faced_pass_defense_weights_defenses_by_dropbacks() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})
    fit = fit_qb_ratings(qb_games, ridge_lambda=_TINY_LAMBDA)

    # Act
    faced = compute_qb_faced_pass_defense(qb_games, fit)

    # Assert
    # qb_AAA faces BBB, CCC, and DDD twice each; CCC games carry 70 dropbacks, the others 35.
    expected = (35 * _DEFENSE["BBB"] + 70 * _DEFENSE["CCC"] + 35 * _DEFENSE["DDD"]) / 140
    assert _value(faced, "qb_AAA", "qb_faced_pass_defense") == pytest.approx(expected, abs=1e-6)


def test_compute_qb_faced_pass_defense_ignores_the_passers_own_games() -> None:
    # Arrange
    qb_games = _qb_games()
    shredded = qb_games.with_columns(
        pl.when(pl.col("qb_id") == "qb_AAA")
        .then(pl.col("qb_epa_per_dropback") + 1.0)
        .otherwise(pl.col("qb_epa_per_dropback"))
        .alias("qb_epa_per_dropback")
    )
    baseline = compute_qb_faced_pass_defense(qb_games, fit_qb_ratings(qb_games, ridge_lambda=10.0))

    # Act
    result = compute_qb_faced_pass_defense(shredded, fit_qb_ratings(shredded, ridge_lambda=10.0))

    # Assert
    assert _value(result, "qb_AAA", "qb_faced_pass_defense") == pytest.approx(
        _value(baseline, "qb_AAA", "qb_faced_pass_defense")
    )


def test_fit_qb_ratings_missing_dropbacks_raises_value_error() -> None:
    # Arrange
    qb_games = _qb_games().drop("qb_dropbacks")

    # Act & Assert
    with pytest.raises(ValueError, match="qb_dropbacks"):
        fit_qb_ratings(qb_games)


def test_fit_qb_ratings_ignores_game_outcomes() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(pl.lit(0).alias("qb_wins"))
    flipped = qb_games.with_columns(pl.lit(1).alias("qb_wins"))
    baseline = fit_qb_ratings(qb_games).ratings

    # Act
    result = fit_qb_ratings(flipped).ratings

    # Assert
    assert result.equals(baseline)
