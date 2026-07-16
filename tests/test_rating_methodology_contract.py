"""Methodology contract tests for published rating inputs."""

import polars as pl
import pytest

from nfl_sos_ratings import data_loader, qb_ratings, ratings


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
