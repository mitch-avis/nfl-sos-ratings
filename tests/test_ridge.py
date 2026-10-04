"""Tests for the shared offense-versus-defense ridge solver."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings.ridge import UnitColumns, fit_unit_ridge, predict_unit

_TEAMS = ("AAA", "BBB", "CCC", "DDD")
_OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": -0.10}
_DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.00, "DDD": -0.04}
_INTERCEPT = 0.02
_HOME = 0.01
_COLUMNS = UnitColumns(response="epa_per_play", weight="plays")


def _schedule_rows(*, rounds: int = 1) -> pl.DataFrame:
    """Return ``rounds`` double round-robins of offense rows built from the known effects."""
    rows: list[dict[str, object]] = []
    matchups = itertools.product(range(rounds), itertools.permutations(_TEAMS, 2))
    for game_number, (_, (home, away)) in enumerate(matchups):
        game_id = f"g{game_number:03d}"
        for offense, defense, is_home in ((home, away, True), (away, home, False)):
            value = (
                _INTERCEPT + _OFFENSE[offense] - _DEFENSE[defense] + (_HOME if is_home else -_HOME)
            )
            rows.append(
                {
                    "game_id": game_id,
                    "team": offense,
                    "opponent_team": defense,
                    "is_home": is_home,
                    "plays": 60,
                    "epa_per_play": value,
                }
            )
    return pl.DataFrame(rows)


def test_fit_unit_ridge_noiseless_schedule_recovers_offense_effects() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.offense == pytest.approx(_OFFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_recovers_defense_effects() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.defense == pytest.approx(_DEFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_recovers_intercept_and_home_field() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert (fit.intercept, fit.home_field) == pytest.approx((_INTERCEPT, _HOME), abs=1e-6)


def test_fit_unit_ridge_heavy_penalty_shrinks_effects_but_keeps_league_mean() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e9)

    # Assert
    assert max(abs(value) for value in fit.offense.values()) < 1e-4
    assert fit.intercept == pytest.approx(_INTERCEPT, abs=1e-6)


def test_fit_unit_ridge_zero_weight_rows_do_not_move_the_fit() -> None:
    # Arrange
    rows = _schedule_rows()
    outlier = rows.head(1).with_columns(
        pl.lit(0, dtype=pl.Int64).alias("plays"), pl.lit(99.0).alias("epa_per_play")
    )
    padded = pl.concat([rows, outlier])

    # Act
    fit = fit_unit_ridge(padded, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.offense == pytest.approx(_OFFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_cross_validates_to_smallest_penalty() -> None:
    # Arrange
    rows = _schedule_rows()
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_fit_unit_ridge_pure_noise_cross_validates_to_largest_penalty() -> None:
    # Arrange
    rows = _schedule_rows(rounds=5)
    noise_only = rows.with_columns(
        pl.Series("epa_per_play", np.random.default_rng(7).normal(0.0, 0.5, rows.height))
    )
    grid = np.array([1e-3, 1.0, 1e6])

    # Act
    fit = fit_unit_ridge(noise_only, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e6)


def test_fit_unit_ridge_empty_rows_raises_value_error() -> None:
    # Arrange
    rows = _schedule_rows().clear()

    # Act & Assert
    with pytest.raises(ValueError, match="no rows"):
        fit_unit_ridge(rows, _COLUMNS)


def test_fit_unit_ridge_without_penalty_falls_back_to_least_squares() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=0.0)

    # Assert
    assert np.isfinite(list(fit.offense.values())).all()


def test_fit_unit_ridge_single_game_uses_the_smallest_candidate_penalty() -> None:
    # Arrange
    rows = _schedule_rows().filter(pl.col("game_id") == "g000")
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_fit_unit_ridge_all_zero_weights_uses_the_smallest_candidate_penalty() -> None:
    # Arrange
    rows = _schedule_rows().with_columns(pl.lit(0, dtype=pl.Int64).alias("plays"))
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_predict_unit_adds_intercept_offense_defense_and_home_field() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(rounds=2), _COLUMNS, ridge_lambda=1e-6)
    rows = pl.DataFrame({"team": ["AAA"], "opponent_team": ["BBB"], "is_home": [True]})

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    expected = _INTERCEPT + _OFFENSE["AAA"] - _DEFENSE["BBB"] + _HOME
    assert predicted.tolist() == pytest.approx([expected], abs=1e-6)


def test_predict_unit_treats_a_neutral_site_as_no_home_field() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(rounds=2), _COLUMNS, ridge_lambda=1e-6)
    rows = pl.DataFrame({"team": ["AAA"], "opponent_team": ["BBB"], "is_home": [None]})

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    expected = _INTERCEPT + _OFFENSE["AAA"] - _DEFENSE["BBB"]
    assert predicted.tolist() == pytest.approx([expected], abs=1e-6)


def test_predict_unit_unknown_unit_is_nan() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(), _COLUMNS, ridge_lambda=1.0)
    rows = pl.DataFrame({"team": ["ZZZ", "AAA"], "opponent_team": ["AAA", "YYY"]})

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    assert np.isnan(predicted).all()
