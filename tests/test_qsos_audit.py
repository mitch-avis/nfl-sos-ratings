"""Regression anchors for the QSoS schedule-strength audit."""

from pathlib import Path

import polars as pl
import pytest

_DATA_DIR = Path("data")

_QB_OPPONENT_QUALITY_ANCHOR_2025: dict[str, dict[str, float | int]] = {
    "Drake Maye": {
        "games": 17,
        "avg_opp_SaCR": -0.7008823529411764,
        "avg_opp_SaDR": -0.466,
        "avg_opp_SRS": -4.2849441176470595,
    },
    "Tyler Shough": {
        "games": 11,
        "avg_opp_SaCR": -0.3277272727272727,
        "avg_opp_SaDR": -0.20163636363636356,
        "avg_opp_SRS": -1.5668308181818185,
    },
    "Joe Flacco": {
        "games": 13,
        "avg_opp_SaCR": -0.15538461538461543,
        "avg_opp_SaDR": -0.3677692307692307,
        "avg_opp_SRS": -1.695016076923077,
    },
    "J.J. McCarthy": {
        "games": 10,
        "avg_opp_SaCR": 0.12709999999999996,
        "avg_opp_SaDR": -0.5338,
        "avg_opp_SRS": -0.9946244999999999,
    },
}


def test_2025_qsos_anchor_table_matches_published_opponent_quality() -> None:
    """The four-QB 2025 audit anchor should reproduce from published Parquet outputs."""
    qb_logs = pl.read_parquet(_DATA_DIR / "2025_qb_game_logs.parquet")
    team_ratings = pl.read_parquet(_DATA_DIR / "2025_ratings.parquet").select(
        ["team", "SaCR", "SaDR", "SRS"]
    )

    for qb_name, expected in _QB_OPPONENT_QUALITY_ANCHOR_2025.items():
        joined = (
            qb_logs.filter(pl.col("qb_name") == qb_name)
            .select(["week", "opponent_team"])
            .sort("week")
            .join(team_ratings, left_on="opponent_team", right_on="team", how="left")
        )

        assert joined.height == expected["games"]
        assert joined["SaCR"].mean() == pytest.approx(expected["avg_opp_SaCR"])
        assert joined["SaDR"].mean() == pytest.approx(expected["avg_opp_SaDR"])
        assert joined["SRS"].mean() == pytest.approx(expected["avg_opp_SRS"])


def test_2025_qsos_anchor_keeps_joe_flacco_all_played_games() -> None:
    """Joe Flacco's 2025 audit row should span all played games, including both team stints."""
    qb_logs = pl.read_parquet(_DATA_DIR / "2025_qb_game_logs.parquet")

    flacco_logs = qb_logs.filter(pl.col("qb_name") == "Joe Flacco")

    assert flacco_logs.height == 13
    assert flacco_logs.select("team").unique().sort("team").to_series().to_list() == [
        "CIN",
        "CLE",
    ]
