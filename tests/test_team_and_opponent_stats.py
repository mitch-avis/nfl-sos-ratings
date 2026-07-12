"""Tests for team and opponent stats computations."""

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
    """Verify team/QB per-game aggregations and win totals are computed correctly."""
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


def test_compute_all_teams_qb_per_game_prefers_majority_snaps_then_dropbacks() -> None:
    """Verify primary team QB selection uses snaps and dropbacks before attempts."""
    qb = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "DEN"],
            "week": [1, 1, 1],
            "qb_name": ["Snap Leader", "Attempt Leader", "Dropback Leader"],
            "qb_attempts": [20, 30, 18],
            "qb_dropbacks": [22, 21, 24],
            "qb_offense_snaps": [45, 40, 40],
            "qb_passer_rating": [101.0, 95.0, 98.0],
        }
    )

    result = team_stats.compute_all_teams_qb_per_game(qb)

    assert result.select("team").item() == "DEN"
    assert result.select("qb_offense_snaps").item() == 45.0
    assert result.select("qb_dropbacks").item() == 22.0
    assert result.select("qb_passer_rating").item() == 101.0


def test_compute_team_snap_counts_from_pbp_counts_scrimmage_snaps() -> None:
    """Verify PBP-derived team snap counts include scrimmage snaps and exclude noise rows."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 6,
            "week": [1] * 6,
            "posteam": ["DEN", "DEN", "KC", "KC", "DEN", "DEN"],
            "defteam": ["KC", "KC", "DEN", "DEN", "KC", "KC"],
            "qb_dropback": [1, 0, 1, 1, 0, 0],
            "rush": [0, 1, 0, 0, 0, 0],
            "qb_kneel": [0, 0, 0, 0, 1, 0],
            "qb_spike": [0, 0, 0, 0, 0, 0],
            "play_type": ["pass", "run", "pass", "pass", "qb_kneel", "no_play"],
        }
    )

    result = team_stats.compute_team_snap_counts_from_pbp(pbp).sort("team")

    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team": "DEN",
            "offensive_snaps": 3,
            "defensive_snaps": 2,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team": "KC",
            "offensive_snaps": 2,
            "defensive_snaps": 3,
        },
    ]


def test_compute_team_game_stats_from_pbp_aggregates_core_team_metrics() -> None:
    """Verify team game stats combine PBP offense with player-stats defense add-ons."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 5,
            "season": [2025] * 5,
            "season_type": ["REG"] * 5,
            "week": [1] * 5,
            "posteam": ["DEN", "DEN", "DEN", "KC", "KC"],
            "defteam": ["KC", "KC", "KC", "DEN", "DEN"],
            "pass": [1, 0, 0, 1, 0],
            "rush": [0, 1, 0, 0, 1],
            "qb_dropback": [1, 0, 1, 1, 0],
            "qb_kneel": [0, 0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0, 0],
            "passing_yards": [20.0, 0.0, 0.0, 15.0, 0.0],
            "rushing_yards": [0.0, 10.0, 0.0, 0.0, 5.0],
            "epa": [3.0, 0.5, -1.0, -0.5, 0.2],
            "pass_touchdown": [1, 0, 0, 0, 0],
            "rush_touchdown": [0, 0, 0, 0, 0],
            "first_down": [1, 1, 0, 1, 0],
            "cpoe": [5.0, None, None, -4.0, None],
            "sack": [0, 0, 1, 0, 0],
            "interception": [0, 0, 0, 1, 0],
            "fumble_lost": [0, 0, 1, 0, 0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "def_tackles_for_loss": [6, 3],
            "def_fumbles_forced": [1, 0],
            "def_sacks": [1, 1],
            "def_qb_hits": [4, 2],
            "def_interceptions": [1, 0],
            "def_pass_defended": [5, 2],
            "def_safeties": [0, 0],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "home_score": [24],
            "away_score": [17],
        }
    )

    result = team_stats.compute_team_game_stats_from_pbp(pbp, player_stats, schedule).sort("team")

    expected_columns = [
        "game_id",
        "season",
        "season_type",
        "week",
        "team",
        "opponent_team",
        "games",
        "offensive_snaps",
        "defensive_snaps",
        "passing_yards",
        "rushing_yards",
        "total_yards",
        "passing_epa",
        "rushing_epa",
        "passing_tds",
        "rushing_tds",
        "passing_first_downs",
        "rushing_first_downs",
        "passing_cpoe",
        "sacks_suffered",
        "passing_interceptions",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "points_for",
        "points_allowed",
        "point_margin",
        "win_value",
        "turnover_margin",
        "passing_yards_allowed",
        "rushing_yards_allowed",
        "total_yards_allowed",
        "passing_epa_allowed",
        "rushing_epa_allowed",
        "passing_tds_allowed",
        "rushing_tds_allowed",
        "passing_first_downs_allowed",
        "rushing_first_downs_allowed",
        "passing_cpoe_allowed",
        "def_tackles_for_loss",
        "def_fumbles_forced",
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_safeties",
    ]

    assert result.select(expected_columns).to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "season": 2025,
            "season_type": "REG",
            "week": 1,
            "team": "DEN",
            "opponent_team": "KC",
            "games": 1,
            "offensive_snaps": 3,
            "defensive_snaps": 2,
            "passing_yards": 20.0,
            "rushing_yards": 10.0,
            "total_yards": 30.0,
            "passing_epa": 2.0,
            "rushing_epa": 0.5,
            "passing_tds": 1,
            "rushing_tds": 0,
            "passing_first_downs": 1,
            "rushing_first_downs": 1,
            "passing_cpoe": 5.0,
            "sacks_suffered": 1,
            "passing_interceptions": 0,
            "sack_fumbles_lost": 1,
            "rushing_fumbles_lost": 0,
            "points_for": 24,
            "points_allowed": 17,
            "point_margin": 7,
            "win_value": 1.0,
            "turnover_margin": 1,
            "passing_yards_allowed": 15.0,
            "rushing_yards_allowed": 5.0,
            "total_yards_allowed": 20.0,
            "passing_epa_allowed": -0.5,
            "rushing_epa_allowed": 0.2,
            "passing_tds_allowed": 0,
            "rushing_tds_allowed": 0,
            "passing_first_downs_allowed": 1,
            "rushing_first_downs_allowed": 0,
            "passing_cpoe_allowed": -4.0,
            "def_tackles_for_loss": 6,
            "def_fumbles_forced": 1,
            "def_sacks": 1,
            "def_qb_hits": 4,
            "def_interceptions": 1,
            "def_pass_defended": 5,
            "def_safeties": 0,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "season": 2025,
            "season_type": "REG",
            "week": 1,
            "team": "KC",
            "opponent_team": "DEN",
            "games": 1,
            "offensive_snaps": 2,
            "defensive_snaps": 3,
            "passing_yards": 15.0,
            "rushing_yards": 5.0,
            "total_yards": 20.0,
            "passing_epa": -0.5,
            "rushing_epa": 0.2,
            "passing_tds": 0,
            "rushing_tds": 0,
            "passing_first_downs": 1,
            "rushing_first_downs": 0,
            "passing_cpoe": -4.0,
            "sacks_suffered": 0,
            "passing_interceptions": 1,
            "sack_fumbles_lost": 0,
            "rushing_fumbles_lost": 0,
            "points_for": 17,
            "points_allowed": 24,
            "point_margin": -7,
            "win_value": 0.0,
            "turnover_margin": -1,
            "passing_yards_allowed": 20.0,
            "rushing_yards_allowed": 10.0,
            "total_yards_allowed": 30.0,
            "passing_epa_allowed": 2.0,
            "rushing_epa_allowed": 0.5,
            "passing_tds_allowed": 1,
            "rushing_tds_allowed": 0,
            "passing_first_downs_allowed": 1,
            "rushing_first_downs_allowed": 1,
            "passing_cpoe_allowed": 5.0,
            "def_tackles_for_loss": 3,
            "def_fumbles_forced": 0,
            "def_sacks": 1,
            "def_qb_hits": 2,
            "def_interceptions": 0,
            "def_pass_defended": 2,
            "def_safeties": 0,
        },
    ]


def test_compute_team_game_stats_from_pbp_derives_per_snap_rates() -> None:
    """Verify team game rows expose offensive and defensive per-snap rate fields."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 5,
            "season": [2025] * 5,
            "season_type": ["REG"] * 5,
            "week": [1] * 5,
            "posteam": ["DEN", "DEN", "DEN", "KC", "KC"],
            "defteam": ["KC", "KC", "KC", "DEN", "DEN"],
            "pass": [1, 0, 0, 1, 0],
            "rush": [0, 1, 0, 0, 1],
            "qb_dropback": [1, 0, 1, 1, 0],
            "qb_kneel": [0, 0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0, 0],
            "passing_yards": [20.0, 0.0, 0.0, 15.0, 0.0],
            "rushing_yards": [0.0, 10.0, 0.0, 0.0, 5.0],
            "epa": [3.0, 0.5, -1.0, -0.5, 0.2],
            "pass_touchdown": [1, 0, 0, 0, 0],
            "rush_touchdown": [0, 0, 0, 0, 0],
            "first_down": [1, 1, 0, 1, 0],
            "cpoe": [5.0, None, None, -4.0, None],
            "sack": [0, 0, 1, 0, 0],
            "interception": [0, 0, 0, 1, 0],
            "fumble_lost": [0, 0, 1, 0, 0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "def_sacks": [1, 1],
            "def_fumbles_forced": [1, 0],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "home_score": [24],
            "away_score": [17],
        }
    )

    result = team_stats.compute_team_game_stats_from_pbp(pbp, player_stats, schedule).sort("team")

    den = result.filter(pl.col("team") == "DEN")
    assert den.select("points_per_offensive_snap").item() == 8.0
    assert den.select("total_yards_per_offensive_snap").item() == 10.0
    assert den.select("passing_epa_per_offensive_snap").item() == pytest.approx(2.0 / 3.0)
    assert den.select("sacks_suffered_per_offensive_snap").item() == pytest.approx(1.0 / 3.0)
    assert den.select("points_allowed_per_defensive_snap").item() == 8.5
    assert den.select("total_yards_allowed_per_defensive_snap").item() == 10.0
    assert den.select("passing_epa_allowed_per_defensive_snap").item() == -0.25
    assert den.select("def_sacks_per_defensive_snap").item() == 0.5
    assert den.select("def_fumbles_forced_per_defensive_snap").item() == 0.5


def test_compute_stats_excluding_opponent() -> None:
    """Verify exclusion helpers remove targeted opponent games for team and QB stats."""
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
    """Verify exclusion helpers return None when no games remain after filtering."""
    weekly = _weekly_df().filter((pl.col("team") == "DEN") & (pl.col("opponent_team") == "KC"))
    qb = _qb_df().filter((pl.col("team_abbr") == "DEN") & (pl.col("week") == 1))

    assert team_stats.compute_team_stats_excluding_opponent(weekly, "DEN", "KC") is None
    assert team_stats.compute_qb_stats_excluding_opponent(qb, weekly, "DEN", "KC") is None


def test_opponent_profile_and_all_profiles() -> None:
    """Verify single-team and all-team opponent profile computations return expected structures."""
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
    """Verify opponent profile gracefully handles opponents without remaining data."""
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
    """Verify all-team opponent profile assembly works when QB rows are missing."""
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
    """Verify all-team opponent profile assembly works when team rows are missing."""
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
