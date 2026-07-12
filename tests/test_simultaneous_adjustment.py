"""Tests for nfl_sos_ratings.simultaneous_adjustment."""

import polars as pl
import pytest

from nfl_sos_ratings import simultaneous_adjustment


def test_solve_srs_centers_team_point_margin_ratings() -> None:
    """Verify SRS solves centered team ratings from team-game point margins."""
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A"],
            "point_margin": [10.0, -10.0, 6.0, -6.0],
        }
    )

    result = simultaneous_adjustment.solve_srs(games, response_col="point_margin")

    assert result.sort("team").to_dicts() == [
        {"team": "A", "srs_rating": 4.0},
        {"team": "B", "srs_rating": -4.0},
    ]


def test_solve_team_stat_ridge_matches_srs_on_point_margin_fixture() -> None:
    """Verify ridge offense and defense solutions align with standalone SRS on margins."""
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A"],
            "point_margin": [10.0, -10.0, 6.0, -6.0],
        }
    )

    srs = simultaneous_adjustment.solve_srs(games, response_col="point_margin")
    ridge = simultaneous_adjustment.solve_team_stat_ridge(
        games,
        response_col="point_margin",
        ridge_lambda=1e-6,
    )

    comparison = ridge.join(srs, on="team", how="inner").with_columns(
        ((pl.col("offense_rating") + pl.col("defense_rating")) / 2.0).alias("net_rating")
    )

    assert comparison.filter(pl.col("team") == "A").select("net_rating").item() == pytest.approx(
        comparison.filter(pl.col("team") == "A").select("srs_rating").item(), abs=1e-3
    )
    assert comparison.filter(pl.col("team") == "B").select("net_rating").item() == pytest.approx(
        comparison.filter(pl.col("team") == "B").select("srs_rating").item(), abs=1e-3
    )


def test_solve_qb_stat_ridge_ranks_qbs_and_defenses() -> None:
    """Verify the QB ridge variant separates stronger QBs from tougher defenses."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B"],
            "opponent_team": ["DEF_1", "DEF_2", "DEF_1", "DEF_2"],
            "epa_per_dropback": [0.20, 0.30, -0.10, 0.05],
        }
    )

    qb_ratings, defense_ratings = simultaneous_adjustment.solve_qb_stat_ridge(
        qb_games,
        response_col="epa_per_dropback",
        ridge_lambda=1e-6,
    )

    assert (
        qb_ratings.filter(pl.col("qb_id") == "QB_A").select("offense_rating").item()
        > qb_ratings.filter(pl.col("qb_id") == "QB_B").select("offense_rating").item()
    )
    assert (
        defense_ratings.filter(pl.col("team") == "DEF_1").select("defense_rating").item()
        > defense_ratings.filter(pl.col("team") == "DEF_2").select("defense_rating").item()
    )


def test_compute_team_adjusted_stats_emits_prefixed_columns() -> None:
    """Verify the team wrapper returns prefixed offense and defense columns."""
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A"],
            "stat_one": [10.0, -10.0, 6.0, -6.0],
            "stat_two": [3.0, 1.0, 2.0, 0.0],
        }
    )

    result = simultaneous_adjustment.compute_team_adjusted_stats(
        games,
        response_cols=["stat_one", "stat_two"],
        ridge_lambda=1e-6,
    )

    assert result.columns == [
        "team",
        "adj_off_stat_one",
        "adj_def_stat_one",
        "adj_off_stat_two",
        "adj_def_stat_two",
    ]
    assert result.height == 2


def test_compute_qb_adjusted_stats_emits_prefixed_columns() -> None:
    """Verify the QB wrapper returns prefixed QB and defense adjustment columns."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B"],
            "opponent_team": ["DEF_1", "DEF_2", "DEF_1", "DEF_2"],
            "stat_one": [0.20, 0.30, -0.10, 0.05],
        }
    )

    qb_ratings, defense_ratings = simultaneous_adjustment.compute_qb_adjusted_stats(
        qb_games,
        response_cols=["stat_one"],
        ridge_lambda=1e-6,
    )

    assert qb_ratings.columns == ["qb_id", "adj_stat_one"]
    assert defense_ratings.columns == ["team", "adj_def_stat_one"]


def test_compute_qb_adjusted_stats_skips_rows_missing_opponent_team() -> None:
    """Verify missing opponent-team rows do not crash the QB simultaneous-adjustment path."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB_A", "QB_A", "QB_B"],
            "opponent_team": ["DEF_1", None, "DEF_2"],
            "stat_one": [0.2, 0.1, -0.1],
        }
    )

    qb_ratings, defense_ratings = simultaneous_adjustment.compute_qb_adjusted_stats(
        qb_games,
        response_cols=["stat_one"],
        ridge_lambda=1e-6,
    )

    assert qb_ratings.columns == ["qb_id", "adj_stat_one"]
    assert defense_ratings.columns == ["team", "adj_def_stat_one"]
    assert qb_ratings.select("qb_id").to_series().to_list() == ["QB_A", "QB_B"]
    assert defense_ratings.select("team").to_series().to_list() == ["DEF_1", "DEF_2"]
