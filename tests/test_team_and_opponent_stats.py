import polars as pl
import pytest

from nfl_sos_ratings import opponent_stats, team_stats


def _weekly_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "KC", "KC", "LAC", "LAC"],
            "opponent_team": ["KC", "LAC", "DEN", "BUF", "LAC", "DEN", "KC"],
            "week": [1, 2, 1, 2, 3, 2, 3],
            "season": [2025] * 7,
            "season_type": ["REG"] * 7,
            "games": [1] * 7,
            "passing_yards": [200, 210, 190, 250, 260, 180, 205],
            "rushing_yards": [100, 110, 95, 120, 115, 90, 98],
            "points_for": [24, 21, 17, 31, 27, 20, 23],
            "points_allowed": [17, 20, 24, 14, 21, 21, 27],
            "passing_epa": [0.2, 0.1, -0.1, 0.35, 0.3, 0.05, 0.08],
            "rushing_epa": [0.1, 0.12, 0.02, 0.15, 0.11, 0.03, 0.04],
            "passing_tds": [2, 2, 1, 3, 3, 2, 2],
            "rushing_tds": [1, 1, 1, 1, 1, 1, 1],
            "passing_first_downs": [10, 11, 9, 13, 14, 8, 10],
            "rushing_first_downs": [6, 6, 5, 7, 7, 5, 5],
            "passing_cpoe": [2.1, 1.8, -1.2, 3.0, 2.7, 0.4, 0.9],
            "sacks_suffered": [2, 2, 3, 1, 1, 2, 2],
            "passing_interceptions": [1, 0, 2, 0, 1, 1, 1],
            "sack_fumbles_lost": [0, 0, 1, 0, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0, 0, 0, 0],
            "def_sacks": [3, 2, 2, 4, 3, 2, 2],
            "def_interceptions": [1, 1, 0, 1, 1, 1, 0],
            "def_pass_defended": [5, 4, 4, 6, 5, 3, 4],
            "def_tackles_for_loss": [6, 5, 5, 7, 6, 4, 4],
            "def_qb_hits": [7, 6, 5, 8, 7, 4, 5],
            "def_fumbles_forced": [1, 0, 1, 1, 1, 0, 0],
            "def_safeties": [0, 0, 0, 0, 0, 0, 0],
        }
    )


def _qb_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC", "KC", "LAC", "LAC"],
            "week": [1, 2, 1, 2, 3, 2, 3],
            "qb_passer_rating": [100.0, 97.0, 88.0, 111.0, 109.0, 94.0, 96.0],
            "qb_aggressiveness": [12.0, 11.0, 10.0, 13.5, 13.0, 9.5, 10.5],
        }
    )


def _schedule_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "home_team": ["DEN", "DEN", "KC", "LAC"],
            "away_team": ["KC", "LAC", "BUF", "KC"],
        }
    )


def test_team_stats_aggregations_and_win_totals() -> None:
    weekly = _weekly_df()
    qb = _qb_df()

    numeric_cols = team_stats._get_numeric_stat_cols(weekly)
    per_game = team_stats.compute_all_teams_per_game(weekly)
    qb_per_game = team_stats.compute_all_teams_qb_per_game(qb)
    win_totals = team_stats.compute_win_totals(weekly)

    assert "passing_yards" in numeric_cols
    assert "season" not in numeric_cols
    assert per_game.height == 3
    assert qb_per_game.height == 3
    assert win_totals.filter(pl.col("team") == "DEN").select("wins").item() == 2


def test_compute_stats_excluding_opponent() -> None:
    weekly = _weekly_df()
    qb = _qb_df()

    team_result = team_stats.compute_team_stats_excluding_opponent(weekly, "KC", "DEN")
    qb_result = team_stats.compute_qb_stats_excluding_opponent(qb, weekly, "KC", "DEN")

    assert team_result is not None
    assert team_result.select("games_included").item() == 2
    assert team_result.select("passing_yards").item() == 255.0
    assert qb_result is not None
    assert qb_result.select("qb_passer_rating").item() == 110.0


def test_compute_stats_excluding_opponent_returns_none() -> None:
    weekly = _weekly_df().filter((pl.col("team") == "DEN") & (pl.col("opponent_team") == "KC"))
    qb = _qb_df().filter((pl.col("team_abbr") == "DEN") & (pl.col("week") == 1))

    assert team_stats.compute_team_stats_excluding_opponent(weekly, "DEN", "KC") is None
    assert team_stats.compute_qb_stats_excluding_opponent(qb, weekly, "DEN", "KC") is None


def test_opponent_profile_and_all_profiles() -> None:
    weekly = _weekly_df()
    qb = _qb_df()
    schedule = _schedule_df()

    opponents = opponent_stats.get_opponents(schedule, "DEN")
    profile = opponent_stats.compute_opponent_profile(weekly, qb, "DEN", schedule)
    all_team, all_qb, details = opponent_stats.compute_all_opponent_profiles(weekly, qb, schedule)

    assert opponents == ["KC", "LAC"]
    assert opponent_stats.is_division_opponent("DEN", "KC") is True
    assert profile["team_stats"] is not None
    assert profile["qb_stats"] is not None
    assert profile["team_stats"].select("team").item() == "DEN"
    assert len(profile["opponent_details"]) == 2
    assert all_team is not None
    assert all_qb is not None
    assert sorted(details) == ["DEN", "KC", "LAC"]


def test_opponent_profile_handles_missing_opponent_stats() -> None:
    weekly = pl.DataFrame(
        {
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "week": [1],
            "season": [2025],
            "season_type": ["REG"],
            "games": [1],
            "passing_yards": [200],
            "rushing_yards": [100],
        }
    )
    qb = pl.DataFrame({"team_abbr": [], "week": [], "qb_passer_rating": []}, strict=False)
    schedule = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})

    profile = opponent_stats.compute_opponent_profile(weekly, qb, "DEN", schedule)

    assert profile["team_stats"] is None
    assert profile["qb_stats"] is None
    assert profile["opponent_details"] == [
        {"opponent": "KC", "division": True, "games_included": 0}
    ]


def test_compute_all_opponent_profiles_handles_missing_qb_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weekly = pl.DataFrame({"team": ["DEN", "KC"]})
    qb = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_passer_rating": [100.0]})
    schedule = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})

    monkeypatch.setattr(
        opponent_stats,
        "compute_opponent_profile",
        lambda weekly_df, qb_df, team, schedule_df: {
            "team_stats": pl.DataFrame({"team": [team], "points_for": [20.0]}),
            "qb_stats": None,
            "opponents": ["KC"],
            "opponent_details": [],
        },
    )

    all_team, all_qb, details = opponent_stats.compute_all_opponent_profiles(weekly, qb, schedule)

    assert all_team is not None
    assert all_qb is None
    assert sorted(details) == ["DEN", "KC"]


def test_compute_all_opponent_profiles_handles_missing_team_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weekly = pl.DataFrame({"team": ["DEN", "KC"]})
    qb = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_passer_rating": [100.0]})
    schedule = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})

    monkeypatch.setattr(
        opponent_stats,
        "compute_opponent_profile",
        lambda weekly_df, qb_df, team, schedule_df: {
            "team_stats": None,
            "qb_stats": pl.DataFrame({"team": [team], "qb_passer_rating": [95.0]}),
            "opponents": ["KC"],
            "opponent_details": [],
        },
    )

    all_team, all_qb, details = opponent_stats.compute_all_opponent_profiles(weekly, qb, schedule)

    assert all_team is None
    assert all_qb is not None
    assert sorted(details) == ["DEN", "KC"]
