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


def _schedule_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "home_team": ["DEN", "DEN", "KC", "LAC"],
            "away_team": ["KC", "LAC", "BUF", "KC"],
        }
    )


def test_get_numeric_stat_cols_keeps_stats_and_drops_identity_columns() -> None:
    """Verify the numeric stat list keeps stat columns and skips identity columns."""
    # Arrange
    weekly = _weekly_df()

    # Act
    numeric_cols = team_stats._get_numeric_stat_cols(weekly)

    # Assert
    assert "passing_yards" in numeric_cols
    assert "season" not in numeric_cols


def test_compute_all_teams_per_game_returns_one_row_per_team() -> None:
    """Verify team per-game aggregation returns one row per team."""
    # Arrange
    weekly = _weekly_df()

    # Act
    per_game = team_stats.compute_all_teams_per_game(weekly)

    # Assert
    assert per_game.height == 3


def test_compute_win_totals_counts_wins() -> None:
    """Verify win totals count each team's wins."""
    # Arrange
    weekly = _weekly_df()

    # Act
    win_totals = team_stats.compute_win_totals(weekly)

    # Assert
    assert win_totals.filter(pl.col("team") == "DEN").select("wins").item() == 2


def test_compute_team_snap_counts_from_pbp_counts_scrimmage_snaps() -> None:
    """Verify PBP-derived team snap counts include scrimmage snaps and exclude noise rows."""
    # Arrange
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

    # Act
    result = team_stats.compute_team_snap_counts_from_pbp(pbp).sort("team")

    # Assert
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
    # Arrange
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

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(pbp, player_stats, schedule).sort("team")

    # Assert
    expected_columns = [
        "game_id",
        "season",
        "season_type",
        "week",
        "team",
        "opponent_team",
        "is_home",
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
            "is_home": True,
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
            "turnover_margin": 0,  # one giveaway each: a strip-sack and an interception
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
            "is_home": False,
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
            "turnover_margin": 0,
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
    # Arrange
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

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(pbp, player_stats, schedule).sort("team")

    # Assert
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


def test_turnover_margin_is_takeaways_minus_giveaways() -> None:
    """Verify turnover margin counts recovered giveaways only, so the two teams sum to zero.

    Fixture: DEN throws an interception and loses a strip-sack fumble; KC loses a fumble after
    a catch and keeps a fumble DEN forced on a run. Player stats credit DEN with both forced
    fumbles and KC with the interception and the strip, as nflverse does.
    """
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 4,
            "season": [2025] * 4,
            "season_type": ["REG"] * 4,
            "week": [1] * 4,
            "posteam": ["DEN", "DEN", "KC", "KC"],
            "defteam": ["KC", "KC", "DEN", "DEN"],
            "pass": [1, 1, 1, 0],
            "rush": [0, 0, 0, 1],
            "qb_dropback": [1, 1, 1, 0],
            "pass_attempt": [1, 1, 1, 0],
            "rush_attempt": [0, 0, 0, 1],
            "complete_pass": [0, 0, 1, 0],
            "sack": [0, 1, 0, 0],
            "interception": [1, 0, 0, 0],
            "fumble": [0, 1, 1, 1],
            "fumble_forced": [0, 1, 1, 1],
            "fumble_lost": [0, 1, 1, 0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "def_interceptions": [0, 1],
            "def_fumbles_forced": [2, 1],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "home_score": [10],
            "away_score": [13],
        }
    )

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(pbp, player_stats, schedule).sort("team")

    # Assert
    assert result.select("team", "takeaways", "giveaways", "turnover_margin").rows() == [
        ("DEN", 1, 2, -1),
        ("KC", 2, 1, 1),
    ]


def _den_kc_schedule() -> pl.DataFrame:
    """Return a one-game DEN-KC schedule with final scores."""
    return pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "home_score": [10],
            "away_score": [13],
        }
    )


def _pass_play(posteam: str, **flags: object) -> dict[str, object]:
    """Return one nflverse-shaped play row for the DEN-KC game with quiet pass flags."""
    play: dict[str, object] = {
        "game_id": "2025_01_DEN_KC",
        "season": 2025,
        "season_type": "REG",
        "week": 1,
        "posteam": posteam,
        "defteam": "KC" if posteam == "DEN" else "DEN",
        "pass": 1,
        "rush": 0,
        "qb_dropback": 1,
        "pass_attempt": 1,
        "rush_attempt": 0,
        "complete_pass": 0,
        "incomplete_pass": 0,
        "sack": 0,
        "interception": 0,
        "two_point_attempt": 0,
        "receiver_player_id": None,
    }
    play.update(flags)
    return play


def test_targets_count_attempts_with_an_intended_receiver() -> None:
    """Verify targets leave out throwaways, sacks, and two-point tries, and catch rate uses them.

    Fixture: DEN throws a completion, two incompletions, and an interception to named receivers,
    plus a throwaway with no receiver; it is also sacked and completes a two-point pass. That is
    5 official attempts, 4 targets, and 1 reception.
    """
    # Arrange
    receiver = "00-0000001"
    plays = [
        _pass_play("DEN", complete_pass=1, receiver_player_id=receiver),
        _pass_play("DEN", incomplete_pass=1, receiver_player_id=receiver),
        _pass_play("DEN", incomplete_pass=1, receiver_player_id=receiver),
        _pass_play("DEN", interception=1, receiver_player_id=receiver),
        _pass_play("DEN", incomplete_pass=1),
        _pass_play("DEN", sack=1),
        _pass_play("DEN", complete_pass=1, two_point_attempt=1, receiver_player_id=receiver),
        _pass_play("KC", incomplete_pass=1, receiver_player_id=receiver),
    ]

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(
        pl.DataFrame(plays), pl.DataFrame(), _den_kc_schedule()
    ).sort("team")

    # Assert
    den = result.row(0, named=True)
    kc = result.row(1, named=True)
    assert (den["attempts"], den["targets"], den["receptions"]) == (5, 4, 1)
    assert den["catch_rate"] == 0.25
    assert den["completion_pct"] == 0.2
    assert (kc["targets_faced"], kc["catch_rate_allowed"]) == (4, 0.25)


def test_targets_are_unknown_when_incompletions_name_no_receiver() -> None:
    """Verify targets and catch rates are null, not undercounts, without receiver data.

    nflverse play-by-play names no receiver on incomplete passes in 2003-2008, so those seasons
    cannot count targets; completions still name theirs.
    """
    # Arrange
    receiver = "00-0000001"
    plays = [
        _pass_play("DEN", complete_pass=1, receiver_player_id=receiver),
        _pass_play("DEN", incomplete_pass=1),
        _pass_play("DEN", incomplete_pass=1),
        _pass_play("KC", complete_pass=1, receiver_player_id=receiver),
    ]

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(
        pl.DataFrame(plays), pl.DataFrame(), _den_kc_schedule()
    )

    # Assert
    assert result.select(
        pl.col("targets", "catch_rate", "targets_faced", "catch_rate_allowed").null_count()
    ).row(0) == (2, 2, 2, 2)


def test_compute_team_stats_excluding_opponent_removes_head_to_head_games() -> None:
    """Verify the team exclusion helper removes games against the excluded opponent."""
    # Arrange
    weekly = _weekly_df()

    # Act
    team_result = team_stats.compute_team_stats_excluding_opponent(weekly, "KC", "DEN")

    # Assert
    assert team_result is not None
    assert team_result.select("games_included").item() == 2
    assert team_result.select("passing_yards").item() == 255.0


def test_compute_team_stats_excluding_opponent_returns_none_without_games() -> None:
    """Verify the team exclusion helper returns None when no games remain."""
    # Arrange
    weekly = _weekly_df().filter((pl.col("team") == "DEN") & (pl.col("opponent_team") == "KC"))

    # Act
    result = team_stats.compute_team_stats_excluding_opponent(weekly, "DEN", "KC")

    # Assert
    assert result is None


def test_get_opponents_returns_unique_scheduled_opponents() -> None:
    """Verify the opponent list is the sorted set of scheduled opponents."""
    # Arrange
    schedule = _schedule_df()

    # Act
    opponents = opponent_stats.get_opponents(schedule, "DEN")

    # Assert
    assert opponents == ["KC", "LAC"]


def test_is_division_opponent_recognizes_division_rivals() -> None:
    """Verify division membership comes from the configured divisions."""
    # Act
    result = opponent_stats.is_division_opponent("DEN", "KC")

    # Assert
    assert result is True


def test_compute_opponent_profile_returns_the_team_profile_and_details() -> None:
    """Verify a single-team opponent profile carries the profile and per-opponent details."""
    # Arrange
    weekly = _weekly_df()
    schedule = _schedule_df()

    # Act
    profile = opponent_stats.compute_opponent_profile(weekly, "DEN", schedule)

    # Assert
    assert profile["team_stats"] is not None
    assert profile["team_stats"].select("team").item() == "DEN"
    assert len(profile["opponent_details"]) == 2


def test_compute_all_opponent_profiles_returns_profiles_for_every_team() -> None:
    """Verify all-team opponent profiles cover every team with details."""
    # Arrange
    weekly = _weekly_df()
    schedule = _schedule_df()

    # Act
    all_team, details = opponent_stats.compute_all_opponent_profiles(weekly, schedule)

    # Assert
    assert all_team is not None
    assert sorted(details) == ["DEN", "KC", "LAC"]


def test_opponent_profile_handles_missing_opponent_stats() -> None:
    """Verify opponent profile gracefully handles opponents without remaining data."""
    # Arrange
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
    schedule = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})

    # Act
    profile = opponent_stats.compute_opponent_profile(weekly, "DEN", schedule)

    # Assert
    assert profile["team_stats"] is None
    assert profile["opponent_details"] == [
        {"opponent": "KC", "division": True, "games_included": 0}
    ]


def test_compute_win_totals_counts_a_shutout_loss() -> None:
    """Verify a game in which a team scored zero points still counts as a loss."""
    # Arrange
    weekly = pl.DataFrame(
        {
            "team": ["DEN", "KC", "DEN", "KC"],
            "opponent_team": ["KC", "DEN", "KC", "DEN"],
            "week": [1, 1, 2, 2],
            "points_for": [38, 0, 17, 20],
            "points_allowed": [0, 38, 20, 17],
        }
    )

    # Act
    totals = team_stats.compute_win_totals(weekly)

    # Assert
    kc = totals.filter(pl.col("team") == "KC").row(0, named=True)
    assert (kc["wins"], kc["losses"], kc["win_pct"]) == (1, 1, 0.5)


def test_compute_opponent_profile_averages_each_opponent_once_without_head_to_head() -> None:
    """Verify each opponent is profiled from its other games and weighted equally."""
    # Arrange
    weekly = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "KC", "KC", "LAC", "LAC"],
            "opponent_team": ["KC", "KC", "DEN", "DEN", "BUF", "MIA", "NYJ"],
            "week": [1, 2, 1, 2, 3, 1, 2],
            "points_for": [24, 21, 17, 20, 30, 10, 14],
        }
    )
    schedule = pl.DataFrame({"home_team": ["DEN", "KC", "LAC"], "away_team": ["KC", "DEN", "DEN"]})

    # Act
    profile = opponent_stats.compute_opponent_profile(weekly, "DEN", schedule)

    # Assert
    # KC without DEN games scored 30; LAC scored 10 and 14 (mean 12); equal weight -> 21.
    assert profile["team_stats"] is not None
    assert profile["team_stats"].get_column("points_for").item() == 21.0


def test_aggregate_defense_only_player_stats_empty_input_is_typed_empty() -> None:
    # Act
    result = team_stats._aggregate_defense_only_player_stats(pl.DataFrame())

    # Assert
    assert result.is_empty()
    assert result.columns == ["team", "opponent_team", "week"]


def test_aggregate_defense_only_player_stats_without_defense_columns_is_typed_empty() -> None:
    # Arrange
    player_stats = pl.DataFrame({"team": ["DEN"], "opponent_team": ["KC"], "week": [1]})

    # Act
    result = team_stats._aggregate_defense_only_player_stats(player_stats)

    # Assert
    assert result.is_empty()


def test_compute_team_game_stats_from_pbp_empty_input_is_typed_empty() -> None:
    # Act
    result = team_stats.compute_team_game_stats_from_pbp(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    )

    # Assert
    assert result.is_empty()
    assert "is_home" in result.columns


def test_compute_team_snap_counts_from_pbp_empty_input_is_typed_empty() -> None:
    # Act
    result = team_stats.compute_team_snap_counts_from_pbp(pl.DataFrame())

    # Assert
    assert result.is_empty()
    assert {"offensive_snaps", "defensive_snaps"} <= set(result.columns)


def test_compute_team_game_stats_from_pbp_without_possession_rows_is_empty() -> None:
    # Arrange
    pbp = pl.DataFrame(
        {"game_id": ["g1"], "week": [1], "posteam": [None], "defteam": [None], "epa": [0.0]},
        schema_overrides={"posteam": pl.String, "defteam": pl.String},
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "home_score": [24],
            "away_score": [17],
        }
    )

    # Act
    result = team_stats.compute_team_game_stats_from_pbp(pbp, pl.DataFrame(), schedule)

    # Assert
    assert result.is_empty()


def test_compute_all_opponent_profiles_without_any_other_games_is_none() -> None:
    # Arrange
    weekly = pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "week": [1, 1],
            "points_for": [20, 17],
        }
    )
    schedule = pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})

    # Act
    profiles, details = opponent_stats.compute_all_opponent_profiles(weekly, schedule)

    # Assert
    assert profiles is None
    assert sorted(details) == ["DEN", "KC"]
