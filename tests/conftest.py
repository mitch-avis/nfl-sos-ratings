import importlib
import sys
from pathlib import Path

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _alias_module(alias: str, target: str) -> None:
    module = importlib.import_module(target)
    sys.modules.setdefault(alias, module)


_alias_module("config", "nfl_sos_ratings.config")
_alias_module("team_stats", "nfl_sos_ratings.team_stats")
_alias_module("data_loader", "nfl_sos_ratings.data_loader")
_alias_module("opponent_stats", "nfl_sos_ratings.opponent_stats")
_alias_module("ratings", "nfl_sos_ratings.ratings")
_alias_module("visualize", "nfl_sos_ratings.visualize")
_alias_module("main", "nfl_sos_ratings.main")


@pytest.fixture
def visualize_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN", "KC", "LAC"],
            "points_for": [27.0, 21.0, 23.0],
            "points_allowed": [19.0, 24.0, 22.0],
            "total_yards": [380.0, 340.0, 350.0],
            "passing_yards": [255.0, 225.0, 235.0],
            "rushing_yards": [125.0, 115.0, 115.0],
            "passing_epa": [0.18, 0.06, 0.11],
            "rushing_epa": [0.07, 0.01, 0.03],
            "def_sacks": [2.9, 2.1, 2.5],
            "def_interceptions": [1.2, 0.8, 1.0],
            "def_pass_defended": [6.1, 5.0, 5.5],
            "def_tackles_for_loss": [6.8, 5.9, 6.2],
            "def_qb_hits": [7.6, 6.8, 7.1],
            "qb_passer_rating": [103.0, 95.0, 98.0],
            "qb_completion_percentage_above_expectation": [2.2, 0.9, 1.4],
            "qb_aggressiveness": [11.3, 10.0, 10.7],
            "qb_avg_intended_air_yards": [8.4, 7.8, 8.0],
            "qb_avg_time_to_throw": [2.62, 2.83, 2.74],
            "diff_points_for": [3.0, -1.0, 0.5],
            "diff_total_yards": [20.0, -10.0, 5.0],
            "diff_passing_yards": [10.0, -8.0, 4.0],
            "diff_rushing_yards": [12.0, -2.0, 1.0],
            "diff_passing_epa": [0.2, -0.1, 0.05],
            "diff_rushing_epa": [0.1, -0.2, 0.03],
            "diff_points_allowed": [-2.0, 1.0, -0.5],
            "diff_sacks_suffered": [-1.0, 0.5, -0.2],
            "diff_passing_interceptions": [-0.3, 0.2, -0.1],
            "diff_def_sacks": [0.8, -0.5, 0.1],
            "diff_def_interceptions": [0.4, -0.2, 0.1],
            "diff_def_pass_defended": [1.2, -0.8, 0.2],
            "diff_qb_passer_rating": [8.0, -5.0, 2.0],
            "diff_qb_completion_percentage_above_expectation": [2.5, -1.4, 0.6],
            "diff_qb_aggressiveness": [1.1, -0.7, 0.3],
            "diff_qb_avg_intended_air_yards": [0.6, -0.4, 0.1],
            "diff_qb_avg_time_to_throw": [-0.2, 0.1, -0.05],
            "diff_qb_avg_air_yards_to_sticks": [0.7, -0.3, 0.2],
            "opp_points_for": [24.0, 20.0, 22.0],
            "opp_points_allowed": [18.0, 22.0, 20.0],
            "opp_total_yards": [360.0, 330.0, 340.0],
            "opp_passing_yards": [245.0, 220.0, 230.0],
            "opp_rushing_yards": [115.0, 110.0, 108.0],
            "opp_passing_epa": [0.12, 0.08, 0.1],
            "opp_rushing_epa": [0.05, 0.02, 0.04],
            "opp_def_sacks": [2.4, 3.0, 2.6],
            "opp_def_interceptions": [0.9, 1.1, 1.0],
            "opp_def_pass_defended": [5.5, 6.2, 5.8],
            "opp_def_tackles_for_loss": [6.4, 7.0, 6.7],
            "opp_def_qb_hits": [7.3, 8.1, 7.8],
            "opp_qb_passer_rating": [98.0, 93.0, 95.0],
            "opp_qb_completion_percentage_above_expectation": [1.8, 0.6, 1.2],
            "opp_qb_aggressiveness": [11.0, 9.8, 10.4],
            "opp_qb_avg_intended_air_yards": [8.2, 7.6, 7.9],
            "opp_qb_avg_time_to_throw": [2.7, 2.9, 2.8],
            "SaCR": [1.0, -0.8, 0.1],
            "SaOR": [0.9, -0.6, 0.2],
            "SaDR": [0.7, -0.5, 0.0],
        }
    )
