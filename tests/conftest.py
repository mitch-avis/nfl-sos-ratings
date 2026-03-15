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
            "opp_total_yards": [360.0, 330.0, 340.0],
            "opp_passing_yards": [245.0, 220.0, 230.0],
            "opp_rushing_yards": [115.0, 110.0, 108.0],
            "opp_passing_epa": [0.12, 0.08, 0.1],
            "opp_rushing_epa": [0.05, 0.02, 0.04],
            "SaCR": [1.0, -0.8, 0.1],
            "SaOR": [0.9, -0.6, 0.2],
            "SaDR": [0.7, -0.5, 0.0],
        }
    )
