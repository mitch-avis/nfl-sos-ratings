"""Tests for the point-margin simple rating system."""

import polars as pl
import pytest

from nfl_sos_ratings.srs import solve_srs


def test_solve_srs_centers_team_point_margin_ratings() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A"],
            "point_margin": [10.0, -10.0, 6.0, -6.0],
        }
    )

    # Act
    result = solve_srs(games, response_col="point_margin")

    # Assert
    assert result.sort("team").to_dicts() == [
        {"team": "A", "srs_rating": 4.0},
        {"team": "B", "srs_rating": -4.0},
    ]


def test_solve_srs_credits_a_win_over_a_strong_opponent() -> None:
    # Arrange
    # A beats B by 7 and C beats B by 7, but A also beats C by 3, so A rates above C.
    games = pl.DataFrame(
        {
            "team": ["A", "B", "C", "B", "A", "C"],
            "opponent_team": ["B", "A", "B", "C", "C", "A"],
            "point_margin": [7.0, -7.0, 7.0, -7.0, 3.0, -3.0],
        }
    )

    # Act
    ratings = dict(solve_srs(games, response_col="point_margin").iter_rows())

    # Assert
    assert ratings["A"] - ratings["C"] == pytest.approx(2.0)
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-5)


def test_solve_srs_empty_input_returns_a_typed_empty_frame() -> None:
    # Arrange
    games = pl.DataFrame(
        schema={"team": pl.String, "opponent_team": pl.String, "point_margin": pl.Float64}
    )

    # Act
    result = solve_srs(games, response_col="point_margin")

    # Assert
    assert result.is_empty()
    assert result.columns == ["team", "srs_rating"]
