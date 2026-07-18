"""Methodology contract tests for published rating inputs."""

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings import composite_weights, data_loader, qb_ratings, ratings


def _pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return a simple Pearson correlation for two equal-length vectors."""
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return a simple Spearman correlation using average ranks."""
    rank_a = np.asarray(pl.Series(values_a).rank(method="average").to_list(), dtype=np.float64)
    rank_b = np.asarray(pl.Series(values_b).rank(method="average").to_list(), dtype=np.float64)
    return _pearson(rank_a, rank_b)


def test_team_quality_ratings_ignore_win_based_outcome_columns() -> None:
    """Verify changing win-based team outcomes alone does not change published quality ratings."""
    base_df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "points_per_offensive_snap": [0.22, 0.31, 0.40],
            "total_yards_per_offensive_snap": [4.2, 4.9, 5.4],
            "passing_yards_per_offensive_snap": [2.5, 3.0, 3.4],
            "rushing_yards_per_offensive_snap": [1.7, 1.9, 2.0],
            "passing_epa_per_offensive_snap": [-0.02, 0.10, 0.18],
            "rushing_epa_per_offensive_snap": [-0.01, 0.04, 0.09],
            "passing_tds_per_offensive_snap": [0.01, 0.02, 0.03],
            "rushing_tds_per_offensive_snap": [0.0, 0.01, 0.02],
            "passing_first_downs_per_offensive_snap": [0.12, 0.16, 0.20],
            "rushing_first_downs_per_offensive_snap": [0.06, 0.07, 0.08],
            "passing_cpoe": [-1.0, 0.5, 2.0],
            "sacks_suffered_per_offensive_snap": [0.08, 0.05, 0.03],
            "passing_interceptions_per_offensive_snap": [0.04, 0.02, 0.01],
            "sack_fumbles_lost_per_offensive_snap": [0.02, 0.01, 0.0],
            "rushing_fumbles_lost_per_offensive_snap": [0.01, 0.0, 0.0],
            "points_allowed_per_defensive_snap": [0.42, 0.32, 0.24],
            "total_yards_allowed_per_defensive_snap": [5.3, 4.6, 4.1],
            "passing_yards_allowed_per_defensive_snap": [3.3, 2.9, 2.6],
            "rushing_yards_allowed_per_defensive_snap": [2.0, 1.7, 1.5],
            "passing_epa_allowed_per_defensive_snap": [0.22, 0.09, 0.01],
            "rushing_epa_allowed_per_defensive_snap": [0.10, 0.05, 0.01],
            "passing_tds_allowed_per_defensive_snap": [0.03, 0.02, 0.01],
            "rushing_tds_allowed_per_defensive_snap": [0.02, 0.01, 0.0],
            "passing_first_downs_allowed_per_defensive_snap": [0.20, 0.16, 0.13],
            "rushing_first_downs_allowed_per_defensive_snap": [0.08, 0.07, 0.06],
            "passing_cpoe_allowed": [1.5, 0.0, -1.0],
            "def_sacks_per_defensive_snap": [0.03, 0.05, 0.07],
            "def_interceptions_per_defensive_snap": [0.01, 0.02, 0.03],
            "def_pass_defended_per_defensive_snap": [0.07, 0.09, 0.12],
            "def_tackles_for_loss_per_defensive_snap": [0.09, 0.11, 0.13],
            "def_qb_hits_per_defensive_snap": [0.10, 0.13, 0.15],
            "def_fumbles_forced_per_defensive_snap": [0.01, 0.02, 0.03],
            "def_safeties_per_defensive_snap": [0.0, 0.0, 0.01],
            "st_rating": [0.20, 0.00, -0.10],
            "win_pct": [0.5, 0.5, 0.5],
            "win_value": [0.5, 0.5, 0.5],
        }
    )
    varied_df = base_df.with_columns(
        win_pct=pl.Series("win_pct", [0.95, 0.50, 0.05]),
        win_value=pl.Series("win_value", [1.0, 0.5, 0.0]),
    )

    base_quality = ratings.compute_ratings(base_df).sort("team")
    varied_quality = ratings.compute_ratings(varied_df).sort("team")

    assert base_quality.select(["SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]).to_dict(
        as_series=False
    ) == (
        varied_quality.select(["SaOR", "SaDR", "SaSTR", "SaOvR", "SaCR"]).to_dict(as_series=False)
    )


def test_qb_quality_ratings_ignore_win_and_clutch_outcome_columns() -> None:
    """Verify changing QB outcome-only fields alone does not change published quality ratings."""
    base_df = pl.DataFrame(
        {
            "qb_id": ["A", "B", "C"],
            "qb_name": ["QB A", "QB B", "QB C"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "qb_epa_per_dropback": [0.18, 0.08, -0.02],
            "qb_any_a": [7.8, 6.9, 6.1],
            "qb_completion_percentage_above_expectation": [4.0, 1.5, -0.5],
            "qb_td_int_margin_rate": [0.07, 0.04, 0.01],
            "qb_sack_rate": [0.03, 0.05, 0.07],
            "qb_pass_yards_per_dropback": [7.6, 6.8, 6.0],
            "qb_passer_rating": [108.0, 96.0, 86.0],
            "qopp_qb_epa_per_dropback": [0.02, 0.01, 0.00],
            "qopp_qb_any_a": [6.3, 6.3, 6.3],
            "qopp_qb_completion_percentage_above_expectation": [0.5, 0.5, 0.5],
            "qopp_qb_td_int_margin_rate": [0.02, 0.02, 0.02],
            "qopp_qb_sack_rate": [0.05, 0.05, 0.05],
            "qopp_qb_pass_yards_per_dropback": [6.4, 6.4, 6.4],
            "qopp_qb_passer_rating": [92.0, 92.0, 92.0],
            "qb_win_pct": [0.5, 0.5, 0.5],
            "qb_wins": [8, 8, 8],
            "qb_fourth_quarter_comebacks": [1, 1, 1],
            "qb_game_winning_drives": [2, 2, 2],
        }
    )
    varied_df = base_df.with_columns(
        qb_win_pct=pl.Series("qb_win_pct", [0.9, 0.5, 0.1]),
        qb_wins=pl.Series("qb_wins", [14, 8, 2]),
        qb_fourth_quarter_comebacks=pl.Series("qb_fourth_quarter_comebacks", [5, 1, 0]),
        qb_game_winning_drives=pl.Series("qb_game_winning_drives", [6, 2, 0]),
    )

    base_quality = qb_ratings.compute_qb_ratings(base_df).sort("qb_id")
    varied_quality = qb_ratings.compute_qb_ratings(varied_df).sort("qb_id")

    assert base_quality.select(["QRaw", "QSaOR", "QSaCR"]).to_dict(as_series=False) == (
        varied_quality.select(["QRaw", "QSaOR", "QSaCR"]).to_dict(as_series=False)
    )


def test_published_rating_inputs_remain_regular_season_only_when_playoff_loader_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published rating inputs stay REG-only while the playoff loader remains validation-only."""
    pbp = pl.DataFrame(
        {
            "season_type": ["REG", "POST"],
            "week": [1, 20],
            "posteam": ["KC", "KC"],
            "defteam": ["DEN", "BUF"],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "BUF"],
            "epa": [0.1, 0.9],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", lambda seasons: pbp)

    regular = data_loader.load_pbp_data(2025)
    postseason = data_loader.load_playoff_pbp_data(2025)

    assert regular.height == 1
    assert regular.select("season_type").item() == "REG"
    assert postseason.height == 1
    assert postseason.select("season_type").item() == "POST"


def test_team_within_season_standardization_preserves_composite_ranking_and_stability() -> None:
    """Season-local team z-scoring should preserve ranking and year-over-year stability."""
    season_2024 = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "adj_off_passing_epa_per_offensive_snap": [0.30, 0.16, 0.05, -0.08],
            "adj_off_rushing_epa_per_offensive_snap": [0.14, 0.09, 0.03, -0.02],
            "adj_def_passing_epa_per_offensive_snap": [0.20, 0.10, 0.02, -0.06],
            "adj_def_rushing_epa_per_offensive_snap": [0.11, 0.06, 0.01, -0.03],
            "st_rating": [0.07, 0.02, -0.01, -0.05],
        }
    )
    season_2025 = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "adj_off_passing_epa_per_offensive_snap": [0.26, 0.18, 0.01, -0.10],
            "adj_off_rushing_epa_per_offensive_snap": [0.12, 0.08, 0.02, -0.03],
            "adj_def_passing_epa_per_offensive_snap": [0.18, 0.12, 0.00, -0.04],
            "adj_def_rushing_epa_per_offensive_snap": [0.10, 0.07, 0.00, -0.02],
            "st_rating": [0.05, 0.03, -0.02, -0.04],
        }
    )

    raw_2024 = composite_weights.build_weighted_composite(
        season_2024,
        composite_weights.TEAM_SACR_FROZEN_SPEC,
        season_2024,
    )
    raw_2025 = composite_weights.build_weighted_composite(
        season_2025,
        composite_weights.TEAM_SACR_FROZEN_SPEC,
        season_2025,
    )
    ratings_2024 = ratings.compute_ratings(season_2024).sort("team")
    ratings_2025 = ratings.compute_ratings(season_2025).sort("team")

    expected_ranking = (
        season_2024.with_columns(pl.Series("_raw_sacr", raw_2024.tolist()))
        .sort("_raw_sacr", descending=True)
        .select("team")
        .to_series()
        .to_list()
    )

    assert (
        ratings_2024.sort("SaCR", descending=True).select("team").to_series().to_list()
        == expected_ranking
    )
    assert _pearson(raw_2024, raw_2025) == pytest.approx(
        _pearson(
            np.asarray(ratings_2024.select("SaCR").to_series().to_list(), dtype=np.float64),
            np.asarray(ratings_2025.select("SaCR").to_series().to_list(), dtype=np.float64),
        ),
        abs=2e-4,
    )
    assert _spearman(raw_2024, raw_2025) == pytest.approx(
        _spearman(
            np.asarray(ratings_2024.select("SaCR").to_series().to_list(), dtype=np.float64),
            np.asarray(ratings_2025.select("SaCR").to_series().to_list(), dtype=np.float64),
        ),
        abs=2e-4,
    )


def test_qb_within_season_standardization_preserves_composite_ranking_and_stability() -> None:
    """Season-local QB z-scoring should preserve ranking and year-over-year stability."""
    season_2024 = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "qb_id": ["A", "B", "C"],
            "qb_name": ["A QB", "B QB", "C QB"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "adj_qb_epa_per_dropback": [0.18, 0.08, -0.01],
            "adj_qb_completion_percentage_above_expectation": [3.0, 1.0, -0.5],
            "adj_qb_sack_rate": [0.03, 0.05, 0.07],
            "adj_qb_td_int_margin_rate": [0.07, 0.04, 0.01],
            "adj_def_qb_epa_per_dropback_faced": [0.04, 0.00, -0.03],
            "qb_epa_per_dropback": [0.20, 0.10, 0.00],
            "qb_any_a": [7.8, 6.9, 6.1],
            "qb_completion_percentage_above_expectation": [4.0, 1.5, -1.0],
            "qb_td_int_margin_rate": [0.08, 0.04, 0.01],
            "qb_sack_rate": [0.03, 0.05, 0.07],
            "qb_pass_yards_per_dropback": [7.5, 6.8, 6.0],
            "qb_passer_rating": [106.0, 96.0, 86.0],
        }
    )
    season_2025 = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "qb_id": ["A", "B", "C"],
            "qb_name": ["A QB", "B QB", "C QB"],
            "team": ["A", "B", "C"],
            "qb_is_eligible": [True, True, True],
            "adj_qb_epa_per_dropback": [0.15, 0.11, 0.01],
            "adj_qb_completion_percentage_above_expectation": [2.2, 1.8, 0.0],
            "adj_qb_sack_rate": [0.03, 0.04, 0.06],
            "adj_qb_td_int_margin_rate": [0.06, 0.05, 0.02],
            "adj_def_qb_epa_per_dropback_faced": [0.03, 0.01, -0.02],
            "qb_epa_per_dropback": [0.17, 0.12, 0.02],
            "qb_any_a": [7.4, 7.0, 6.2],
            "qb_completion_percentage_above_expectation": [3.0, 2.0, 0.0],
            "qb_td_int_margin_rate": [0.07, 0.05, 0.02],
            "qb_sack_rate": [0.03, 0.04, 0.06],
            "qb_pass_yards_per_dropback": [7.2, 6.9, 6.2],
            "qb_passer_rating": [103.0, 99.0, 88.0],
        }
    )

    raw_2024 = composite_weights.build_weighted_composite(
        season_2024,
        composite_weights.QB_QSACR_FROZEN_SPEC,
        season_2024,
    )
    raw_2025 = composite_weights.build_weighted_composite(
        season_2025,
        composite_weights.QB_QSACR_FROZEN_SPEC,
        season_2025,
    )
    ratings_2024 = qb_ratings.compute_qb_ratings(season_2024).sort("qb_id")
    ratings_2025 = qb_ratings.compute_qb_ratings(season_2025).sort("qb_id")

    expected_ranking = (
        season_2024.with_columns(pl.Series("_raw_qsacr", raw_2024.tolist()))
        .sort("_raw_qsacr", descending=True)
        .select("qb_id")
        .to_series()
        .to_list()
    )

    assert (
        ratings_2024.sort("QSaCR", descending=True).select("qb_id").to_series().to_list()
        == expected_ranking
    )
    assert _pearson(raw_2024, raw_2025) == pytest.approx(
        _pearson(
            np.asarray(ratings_2024.select("QSaCR").to_series().to_list(), dtype=np.float64),
            np.asarray(ratings_2025.select("QSaCR").to_series().to_list(), dtype=np.float64),
        ),
        abs=2e-4,
    )
    assert _spearman(raw_2024, raw_2025) == pytest.approx(
        _spearman(
            np.asarray(ratings_2024.select("QSaCR").to_series().to_list(), dtype=np.float64),
            np.asarray(ratings_2025.select("QSaCR").to_series().to_list(), dtype=np.float64),
        ),
        abs=2e-4,
    )
