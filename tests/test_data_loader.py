"""Tests for nfl_sos_ratings.data_loader."""

import polars as pl
import pytest

from nfl_sos_ratings import data_loader
from nfl_sos_ratings.config import TEAM_ABBR_ALIASES
from tests.stubs import stub


def test_load_weekly_team_stats_enriches_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify weekly team stats are built from REG PBP with player-stat defense add-ons."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "season": [2025] * 4,
            "season_type": ["REG", "REG", "REG", "POST"],
            "week": [1, 1, 1, 2],
            "posteam": ["DEN", "DEN", "KC", "DEN"],
            "defteam": ["KC", "KC", "DEN", "BUF"],
            "pass": [1, 0, 1, 1],
            "rush": [0, 1, 0, 0],
            "qb_dropback": [1, 0, 1, 1],
            "qb_kneel": [0, 0, 0, 0],
            "qb_spike": [0, 0, 0, 0],
            "passing_yards": [20.0, 0.0, 15.0, 99.0],
            "rushing_yards": [0.0, 10.0, 0.0, 0.0],
            "epa": [1.0, 0.5, -0.5, 4.0],
            "pass_touchdown": [1, 0, 0, 1],
            "rush_touchdown": [0, 0, 0, 0],
            "first_down": [1, 1, 1, 1],
            "cpoe": [4.0, None, -2.0, 6.0],
            "sack": [0, 0, 0, 0],
            "interception": [0, 0, 1, 0],
            "fumble_lost": [0, 0, 0, 0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "POST"],
            "week": [1, 2],
            "team": ["DEN", "DEN"],
            "opponent_team": ["KC", "BUF"],
            "def_sacks": [2, 9],
            "def_pass_defended": [4, 12],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_02_DEN_BUF"],
            "game_type": ["REG", "POST"],
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
            "week": [1, 2],
            "home_score": [24, 17],
            "away_score": [17, 21],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )
    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_weekly_team_stats(2025)

    # Assert
    assert result.height == 2
    assert result.filter(pl.col("team") == "DEN").select("total_yards").item() == 30.0
    assert result.filter(pl.col("team") == "DEN").select("points_for").item() == 24
    assert result.filter(pl.col("team") == "DEN").select("points_allowed").item() == 17
    assert result.filter(pl.col("team") == "DEN").select("def_sacks").item() == 2


def test_load_weekly_team_stats_normalizes_rams_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PBP-backed weekly team data uses canonical Rams abbreviations."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_LA_SEA"],
            "season": [2025],
            "week": [1],
            "season_type": ["REG"],
            "posteam": ["LA"],
            "defteam": ["SEA"],
            "pass": [1],
            "rush": [0],
            "qb_dropback": [1],
            "qb_kneel": [0],
            "qb_spike": [0],
            "passing_yards": [280.0],
            "rushing_yards": [0.0],
            "epa": [1.5],
            "pass_touchdown": [0],
            "rush_touchdown": [0],
            "first_down": [1],
            "cpoe": [3.0],
            "sack": [0],
            "interception": [0],
            "fumble_lost": [0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025],
            "season_type": ["REG"],
            "week": [1],
            "team": ["LA"],
            "opponent_team": ["SEA"],
            "def_sacks": [1],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_LA_SEA"],
            "game_type": ["REG"],
            "home_team": ["LA"],
            "away_team": ["SEA"],
            "week": [1],
            "home_score": [24],
            "away_score": [17],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )
    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_weekly_team_stats(2025)

    # Assert
    assert result.select("team").item() == "LAR"
    assert result.select("opponent_team").item() == "SEA"


def test_load_weekly_team_stats_prefers_official_team_stats_for_published_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify official weekly team stats replace published split columns and mirrors."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "season": [2025, 2025, 2025],
            "season_type": ["REG", "REG", "REG"],
            "week": [1, 1, 1],
            "posteam": ["DEN", "DEN", "KC"],
            "defteam": ["KC", "KC", "DEN"],
            "pass": [1, 0, 1],
            "rush": [0, 1, 0],
            "qb_dropback": [1, 0, 1],
            "qb_kneel": [0, 0, 0],
            "qb_spike": [0, 0, 0],
            "passing_yards": [20.0, 0.0, 15.0],
            "rushing_yards": [0.0, 10.0, 0.0],
            "epa": [1.0, 0.5, -0.5],
            "pass_touchdown": [1, 0, 0],
            "rush_touchdown": [0, 0, 0],
            "first_down": [1, 1, 1],
            "cpoe": [4.0, None, -2.0],
            "sack": [0, 0, 0],
            "interception": [0, 0, 1],
            "fumble_lost": [0, 0, 0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025],
            "season_type": ["REG"],
            "week": [1],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "def_sacks": [2],
            "def_pass_defended": [4],
        }
    )
    team_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC"],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "passing_yards": [250.0, 180.0],
            "rushing_yards": [110.0, 90.0],
            "passing_tds": [2, 1],
            "rushing_tds": [1, 0],
            "passing_first_downs": [12, 9],
            "rushing_first_downs": [6, 4],
            "passing_epa": [5.5, 1.7],
            "rushing_epa": [1.2, 0.4],
            "passing_cpoe": [3.5, -1.5],
            "sacks_suffered": [2, 3],
            "passing_interceptions": [1, 2],
            "sack_fumbles_lost": [1, 0],
            "rushing_fumbles_lost": [0, 1],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "game_type": ["REG"],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "week": [1],
            "home_score": [24],
            "away_score": [17],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )
    monkeypatch.setattr(data_loader.nfl, "load_team_stats", stub(lambda: team_stats))
    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_weekly_team_stats(2025).sort("team")

    # Assert
    den = result.filter(pl.col("team") == "DEN")

    assert den.select("passing_yards").item() == 250.0
    assert den.select("rushing_yards").item() == 110.0
    assert den.select("total_yards").item() == 360.0
    assert den.select("passing_tds").item() == 2
    assert den.select("rushing_tds").item() == 1
    assert den.select("passing_first_downs").item() == 12
    assert den.select("rushing_first_downs").item() == 6
    assert den.select("passing_epa").item() == 5.5
    assert den.select("rushing_epa").item() == 1.2
    assert den.select("passing_cpoe").item() == 3.5
    assert den.select("sacks_suffered").item() == 2
    assert den.select("passing_interceptions").item() == 1
    assert den.select("sack_fumbles_lost").item() == 1
    assert den.select("rushing_fumbles_lost").item() == 0
    assert den.select("passing_epa_allowed").item() == 1.7
    assert den.select("rushing_epa_allowed").item() == 0.4
    assert den.select("passing_first_downs_allowed").item() == 9
    assert den.select("rushing_first_downs_allowed").item() == 4
    assert den.select("passing_cpoe_allowed").item() == -1.5
    assert den.select("passing_epa_per_offensive_snap").item() == pytest.approx(5.5 / 2.0)
    assert den.select("passing_epa_allowed_per_defensive_snap").item() == pytest.approx(1.7)


def test_load_schedule_filters_regular_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify schedule loading keeps only regular-season games."""
    # Arrange
    schedule = pl.DataFrame(
        {
            "game_type": ["REG", "POST"],
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_schedule(2025)

    # Assert
    assert result.height == 1
    assert result.select("away_team").item() == "KC"


def test_load_schedule_normalizes_rams_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify schedule team columns use the canonical Rams abbreviation."""
    # Arrange
    schedule = pl.DataFrame(
        {
            "game_type": ["REG"],
            "home_team": ["LA"],
            "away_team": ["SEA"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_schedule(2025)

    # Assert
    assert result.select("home_team").item() == "LAR"


def test_load_snap_counts_data_returns_typed_empty_before_source_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pre-2012 snap-count loads short-circuit to an empty typed frame."""

    # Arrange
    def _unexpected_snap_counts_call(seasons: int) -> pl.DataFrame:
        msg = f"snap counts loader should not run for season {seasons}"
        raise AssertionError(msg)

    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", _unexpected_snap_counts_call)

    # Act
    result = data_loader.load_snap_counts_data(2000)

    # Assert
    assert result.is_empty()
    assert result.schema == {
        "game_id": pl.String,
        "week": pl.Int64,
        "team": pl.String,
        "player": pl.String,
        "pfr_player_id": pl.String,
        "position": pl.String,
        "offense_snaps": pl.Float64,
    }


def test_load_qb_identity_crosswalk_skips_weekly_rosters_before_source_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pre-2002 QB identity loading falls back to players data only."""
    # Arrange
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": ["QB"],
            "pfr_id": ["DoeJo00"],
        }
    )

    def _unexpected_rosters_weekly_call(seasons: int) -> pl.DataFrame:
        msg = f"weekly rosters loader should not run for season {seasons}"
        raise AssertionError(msg)

    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        _unexpected_rosters_weekly_call,
    )

    # Act
    result = data_loader.load_qb_identity_crosswalk(2001)

    # Assert
    assert result.to_dicts() == [
        {
            "qb_id": "00-0031234",
            "snap_player_id": "DoeJo00",
            "qb_name": "John Doe",
            "qb_position": "QB",
        }
    ]


def test_load_qb_stats_merges_pbp_and_snap_counts_by_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify QB loading uses canonical identity instead of qb_name to merge sources."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC"],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "posteam": ["DEN", "DEN"],
            "passer_player_id": ["00-0031234", "00-0031234"],
            "passer_player_name": ["J.Doe", "J.Doe"],
            "qb_dropback": [1, 1],
            "pass": [1, 0],
            "complete_pass": [1, 0],
            "passing_yards": [18.0, 0.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [0, 1],
            "fumble_lost": [0, 0],
            "qb_epa": [1.2, -0.4],
            "cpoe": [4.0, None],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "team": ["DEN"],
            "player": ["John Doe"],
            "pfr_player_id": ["DoeJo00"],
            "position": ["QB"],
            "offense_snaps": [60.0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": ["QB"],
            "pfr_id": ["DoeJo00"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["DEN"],
            "full_name": ["John Doe"],
            "gsis_id": ["00-0031234"],
            "pfr_id": ["DoeJo00"],
            "position": ["QB"],
            "game_type": ["REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025)

    # Assert
    assert result.height == 1
    assert result.select("game_id").item() == "2025_01_DEN_KC"
    assert result.select("week").item() == 1
    assert result.select("team_abbr").item() == "DEN"
    assert result.select("qb_name").item() == "John Doe"
    assert result.select("qb_id").item() == "00-0031234"
    assert result.select("snap_player_id").item() == "DoeJo00"
    assert result.select("qb_dropbacks").item() == 2
    assert result.select("qb_offense_snaps").item() == 60
    assert result.select("qb_attempts").item() == 1
    assert result.select("qb_completions").item() == 1
    assert result.select("qb_pass_yards").item() == 18.0
    assert result.select("qb_pass_touchdowns").item() == 0
    assert result.select("qb_interceptions").item() == 0
    assert result.select("qb_sacks").item() == 1
    assert result.select("qb_sack_yards_lost").item() == 0.0
    assert result.select("qb_sack_fumbles_lost").item() == 0
    assert result.select("qb_passing_epa").item() == pytest.approx(0.8)
    assert result.select("qb_epa_per_dropback").item() == pytest.approx(0.4)
    assert result.select("qb_pass_yards_per_dropback").item() == 9.0
    assert result.select("qb_td_int_margin_rate").item() == 0.0
    assert result.select("qb_sack_rate").item() == 0.5
    assert result.select("qb_any_a").item() == 9.0
    assert result.select("qb_fourth_quarter_comeback").item() == 0
    assert result.select("qb_game_winning_drive").item() == 0
    assert result.select("qb_completion_percentage_above_expectation").item() == 4.0
    assert result.select("qb_passer_rating").item() == 118.8


def test_load_qb_stats_excludes_non_qb_trick_passers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify QB loading drops passers whose authoritative position is not QB."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC"],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "posteam": ["DEN", "DEN"],
            "passer_player_id": ["00-0031234", "00-0099999"],
            "passer_player_name": ["John Doe", "Gadget Guy"],
            "qb_dropback": [1, 1],
            "pass": [1, 1],
            "complete_pass": [1, 1],
            "passing_yards": [18.0, 12.0],
            "pass_touchdown": [0, 1],
            "interception": [0, 0],
            "sack": [0, 0],
            "fumble_lost": [0, 0],
            "qb_epa": [1.2, 1.5],
            "cpoe": [4.0, 8.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "team": ["DEN"],
            "player": ["John Doe"],
            "pfr_player_id": ["DoeJo00"],
            "position": ["QB"],
            "offense_snaps": [60.0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234", "00-0099999"],
            "display_name": ["John Doe", "Gadget Guy"],
            "position": ["QB", "WR"],
            "pfr_id": ["DoeJo00", "GadgGu00"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["DEN", "DEN"],
            "full_name": ["John Doe", "Gadget Guy"],
            "gsis_id": ["00-0031234", "00-0099999"],
            "pfr_id": ["DoeJo00", "GadgGu00"],
            "position": ["QB", "WR"],
            "game_type": ["REG", "REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025)

    # Assert
    assert result.height == 1
    assert result.select("qb_id").item() == "00-0031234"
    assert result.select("qb_name").item() == "John Doe"


def test_load_qb_stats_prefers_official_weekly_player_stats_for_attempt_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify official weekly player stats replace attempt-based QB game fields."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC"],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "posteam": ["DEN", "DEN"],
            "passer_player_id": ["00-0031234", "00-0031234"],
            "passer_player_name": ["John Doe", "John Doe"],
            "qb_dropback": [1, 1],
            "pass": [1, 0],
            "complete_pass": [1, 0],
            "passing_yards": [18.0, 0.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [0, 1],
            "fumble_lost": [0, 0],
            "qb_epa": [1.2, -0.4],
            "cpoe": [4.0, None],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "team": ["DEN"],
            "player": ["John Doe"],
            "pfr_player_id": ["DoeJo00"],
            "position": ["QB"],
            "offense_snaps": [60.0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "season_type": ["REG"],
            "game_id": ["2025_01_DEN_KC"],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "player_id": ["00-0031234"],
            "player_display_name": ["John Doe"],
            "position": ["QB"],
            "attempts": [5],
            "completions": [4],
            "passing_yards": [50.0],
            "passing_tds": [1],
            "passing_interceptions": [1],
            "sacks_suffered": [1],
            # nflverse stores the yards lost on sacks as a negative number.
            "sack_yards_lost": [-8.0],
            "passing_epa": [3.0],
            "passing_cpoe": [2.5],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": ["QB"],
            "pfr_id": ["DoeJo00"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["DEN"],
            "full_name": ["John Doe"],
            "gsis_id": ["00-0031234"],
            "pfr_id": ["DoeJo00"],
            "position": ["QB"],
            "game_type": ["REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025)

    # Assert
    assert result.height == 1
    assert result.select("qb_dropbacks").item() == 2
    assert result.select("qb_attempts").item() == 5
    assert result.select("qb_completions").item() == 4
    assert result.select("qb_pass_yards").item() == 50.0
    assert result.select("qb_pass_touchdowns").item() == 1
    assert result.select("qb_interceptions").item() == 1
    assert result.select("qb_sacks").item() == 1
    assert result.select("qb_sack_yards_lost").item() == 8.0
    assert result.select("qb_passing_epa").item() == 3.0
    assert result.select("qb_completion_percentage_above_expectation").item() == 2.5
    assert result.select("qb_epa_per_dropback").item() == 1.5
    assert result.select("qb_pass_yards_per_dropback").item() == 25.0
    assert result.select("qb_td_int_margin_rate").item() == 0.0
    assert result.select("qb_sack_rate").item() == 0.5
    # ANY/A = (50 yards + 20 * 1 TD - 45 * 1 INT - 8 sack yards) / (5 attempts + 1 sack).
    assert result.select("qb_any_a").item() == pytest.approx(17.0 / 6.0)
    assert result.select("qb_passer_rating").item() == 108.3


def test_load_qb_stats_keeps_individual_qbs_and_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify QB loading keeps individual game rows from PBP plus snap counts."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 5,
            "season_type": ["REG", "REG", "REG", "REG", "POST"],
            "week": [1, 1, 1, 1, 2],
            "posteam": ["DEN", "DEN", "DEN", "KC", "DEN"],
            "passer_player_id": ["GSIS_A", "GSIS_A", "GSIS_B", "GSIS_C", "GSIS_A"],
            "passer_player_name": ["Starter QB", "Starter QB", "Backup QB", "KC QB", "Starter QB"],
            "qb_dropback": [1, 1, 1, 1, 1],
            "pass": [1, 1, 1, 1, 1],
            "complete_pass": [1, 0, 0, 1, 1],
            "passing_yards": [20.0, 0.0, 0.0, 15.0, 99.0],
            "pass_touchdown": [1, 0, 0, 0, 1],
            "interception": [0, 0, 1, 0, 0],
            "sack": [0, 0, 0, 0, 0],
            "fumble_lost": [0, 0, 0, 0, 0],
            "qb_epa": [2.0, -0.5, -2.0, 1.0, 4.0],
            "cpoe": [5.0, -3.0, -6.0, 1.5, 9.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": [
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
            ],
            "week": [1, 1, 1, 1],
            "team": ["DEN", "DEN", "DEN", "KC"],
            "player": ["Starter QB", "Backup QB", "Runner QB", "KC QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_D", "PFR_C"],
            "position": ["QB", "QB", "QB", "QB"],
            "offense_snaps": [42.0, 16.0, 3.0, 58.0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["GSIS_A", "GSIS_B", "GSIS_C", "GSIS_D"],
            "display_name": ["Starter QB", "Backup QB", "KC QB", "Runner QB"],
            "position": ["QB", "QB", "QB", "QB"],
            "pfr_id": ["PFR_A", "PFR_B", "PFR_C", "PFR_D"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "team": ["DEN", "DEN", "KC", "DEN"],
            "full_name": ["Starter QB", "Backup QB", "KC QB", "Runner QB"],
            "gsis_id": ["GSIS_A", "GSIS_B", "GSIS_C", "GSIS_D"],
            "pfr_id": ["PFR_A", "PFR_B", "PFR_C", "PFR_D"],
            "position": ["QB", "QB", "QB", "QB"],
            "game_type": ["REG", "REG", "REG", "REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025).sort(["team_abbr", "week"])

    # Assert
    assert result.columns == [
        "game_id",
        "week",
        "team_abbr",
        "qb_name",
        "qb_id",
        "snap_player_id",
        "qb_dropbacks",
        "qb_offense_snaps",
        "qb_attempts",
        "qb_completions",
        "qb_pass_yards",
        "qb_pass_touchdowns",
        "qb_interceptions",
        "qb_sacks",
        "qb_sack_yards_lost",
        "qb_sack_fumbles_lost",
        "qb_passing_epa",
        "qb_epa_per_dropback",
        "qb_pass_yards_per_dropback",
        "qb_td_int_margin_rate",
        "qb_sack_rate",
        "qb_any_a",
        "qb_fourth_quarter_comeback",
        "qb_game_winning_drive",
        "qb_completion_percentage_above_expectation",
        "qb_passer_rating",
        "qb_completion_pct",
        "qb_carries",
        "qb_rushing_yards",
        "qb_rushing_tds",
        "qb_rushing_first_downs",
        "qb_rushing_epa",
        "qb_rushing_fumbles",
        "qb_rushing_fumbles_lost",
        "qb_rushing_2pt_conversions",
        "qb_designed_carries",
        "qb_designed_rush_yards",
        "qb_designed_rush_epa",
        "qb_scrambles",
        "qb_scramble_yards",
        "qb_kneels",
        "qb_yards_per_carry",
        "qb_epa_per_carry",
        "qb_scramble_rate",
        "qb_yards_per_scramble",
        "qb_designed_yards_per_carry",
        "qb_designed_epa_per_carry",
    ]
    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Backup QB",
            "qb_id": "GSIS_B",
            "snap_player_id": "PFR_B",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 16,
            "qb_attempts": 1,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 1,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": -2.0,
            "qb_epa_per_dropback": -2.0,
            "qb_pass_yards_per_dropback": 0.0,
            "qb_td_int_margin_rate": -1.0,
            "qb_sack_rate": 0.0,
            "qb_any_a": -45.0,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": -6.0,
            "qb_passer_rating": 0.0,
            "qb_completion_pct": 0.0,
            "qb_carries": 0,
            "qb_rushing_yards": 0.0,
            "qb_rushing_tds": 0,
            "qb_rushing_first_downs": 0,
            "qb_rushing_epa": 0.0,
            "qb_rushing_fumbles": 0,
            "qb_rushing_fumbles_lost": 0,
            "qb_rushing_2pt_conversions": 0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_yards_per_carry": None,
            "qb_epa_per_carry": None,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Runner QB",
            "qb_id": "GSIS_D",
            "snap_player_id": "PFR_D",
            "qb_dropbacks": 0,
            "qb_offense_snaps": 3,
            "qb_attempts": 0,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 0,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": 0.0,
            "qb_epa_per_dropback": None,
            "qb_pass_yards_per_dropback": None,
            "qb_td_int_margin_rate": None,
            "qb_sack_rate": None,
            "qb_any_a": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": None,
            "qb_passer_rating": None,
            "qb_completion_pct": None,
            "qb_carries": 0,
            "qb_rushing_yards": 0.0,
            "qb_rushing_tds": 0,
            "qb_rushing_first_downs": 0,
            "qb_rushing_epa": 0.0,
            "qb_rushing_fumbles": 0,
            "qb_rushing_fumbles_lost": 0,
            "qb_rushing_2pt_conversions": 0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_yards_per_carry": None,
            "qb_epa_per_carry": None,
            "qb_scramble_rate": None,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Starter QB",
            "qb_id": "GSIS_A",
            "snap_player_id": "PFR_A",
            "qb_dropbacks": 2,
            "qb_offense_snaps": 42,
            "qb_attempts": 2,
            "qb_completions": 1,
            "qb_pass_yards": 20.0,
            "qb_pass_touchdowns": 1,
            "qb_interceptions": 0,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": 1.5,
            "qb_epa_per_dropback": 0.75,
            "qb_pass_yards_per_dropback": 10.0,
            "qb_td_int_margin_rate": 0.5,
            "qb_sack_rate": 0.0,
            "qb_any_a": 20.0,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": 1.0,
            "qb_passer_rating": 125.0,
            "qb_completion_pct": 0.5,
            "qb_carries": 0,
            "qb_rushing_yards": 0.0,
            "qb_rushing_tds": 0,
            "qb_rushing_first_downs": 0,
            "qb_rushing_epa": 0.0,
            "qb_rushing_fumbles": 0,
            "qb_rushing_fumbles_lost": 0,
            "qb_rushing_2pt_conversions": 0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_yards_per_carry": None,
            "qb_epa_per_carry": None,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "KC",
            "qb_name": "KC QB",
            "qb_id": "GSIS_C",
            "snap_player_id": "PFR_C",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 58,
            "qb_attempts": 1,
            "qb_completions": 1,
            "qb_pass_yards": 15.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 0,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": 1.0,
            "qb_epa_per_dropback": 1.0,
            "qb_pass_yards_per_dropback": 15.0,
            "qb_td_int_margin_rate": 0.0,
            "qb_sack_rate": 0.0,
            "qb_any_a": 15.0,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": 1.5,
            "qb_passer_rating": 118.8,
            "qb_completion_pct": 1.0,
            "qb_carries": 0,
            "qb_rushing_yards": 0.0,
            "qb_rushing_tds": 0,
            "qb_rushing_first_downs": 0,
            "qb_rushing_epa": 0.0,
            "qb_rushing_fumbles": 0,
            "qb_rushing_fumbles_lost": 0,
            "qb_rushing_2pt_conversions": 0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_yards_per_carry": None,
            "qb_epa_per_carry": None,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
        },
    ]


def test_load_qb_stats_normalizes_rams_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify QB game rows use the canonical Rams abbreviation."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_LA_SEA"],
            "season_type": ["REG"],
            "week": [1],
            "posteam": ["LA"],
            "passer_player_id": ["GSIS_QB"],
            "passer_player_name": ["Rams QB"],
            "qb_dropback": [1],
            "pass": [1],
            "complete_pass": [1],
            "passing_yards": [12.0],
            "pass_touchdown": [0],
            "interception": [0],
            "sack": [0],
            "fumble_lost": [0],
            "qb_epa": [0.5],
            "cpoe": [2.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_LA_SEA"],
            "week": [1],
            "team": ["LA"],
            "player": ["Rams QB"],
            "pfr_player_id": ["PFR_QB"],
            "position": ["QB"],
            "offense_snaps": [58.0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["GSIS_QB"],
            "display_name": ["Rams QB"],
            "position": ["QB"],
            "pfr_id": ["PFR_QB"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["LA"],
            "full_name": ["Rams QB"],
            "gsis_id": ["GSIS_QB"],
            "pfr_id": ["PFR_QB"],
            "position": ["QB"],
            "game_type": ["REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025)

    # Assert
    assert result.select("team_abbr").item() == "LAR"


def test_load_pbp_data_filters_regular_season_and_normalizes_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PBP loading keeps regular season rows and normalizes team columns."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "season_type": ["REG", "POST"],
            "week": [1, 2],
            "posteam": ["LA", "DEN"],
            "defteam": ["SEA", "KC"],
            "home_team": ["LA", "DEN"],
            "away_team": ["SEA", "KC"],
            "epa": [0.15, -0.2],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_pbp_data(2025)

    # Assert
    assert result.height == 1
    assert result.select("posteam").item() == "LAR"
    assert result.select("defteam").item() == "SEA"
    assert result.select("home_team").item() == "LAR"
    assert result.select("away_team").item() == "SEA"


def test_load_playoff_pbp_data_filters_postseason_and_normalizes_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the validation-only playoff PBP loader keeps postseason rows only."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "season_type": ["REG", "POST"],
            "week": [1, 20],
            "posteam": ["LA", "DEN"],
            "defteam": ["SEA", "KC"],
            "home_team": ["LA", "DEN"],
            "away_team": ["SEA", "KC"],
            "epa": [0.15, -0.2],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_playoff_pbp_data(2025)

    # Assert
    assert result.height == 1
    assert result.select("week").item() == 20
    assert result.select("posteam").item() == "DEN"
    assert result.select("defteam").item() == "KC"


# Every play-by-play column that holds a team code in the nflverse files for 1999-2026: plain
# codes, then yard lines written "<team> <yards>". nflverse writes the Rams as LA in each.
_PBP_TEAM_CODE_COLUMNS = (
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "side_of_field",
    "timeout_team",
    "td_team",
    "return_team",
    "penalty_team",
    "fumbled_1_team",
    "fumbled_2_team",
    "fumble_recovery_1_team",
    "fumble_recovery_2_team",
    "forced_fumble_player_1_team",
    "forced_fumble_player_2_team",
    "solo_tackle_1_team",
    "solo_tackle_2_team",
    "assist_tackle_1_team",
    "assist_tackle_2_team",
    "assist_tackle_3_team",
    "assist_tackle_4_team",
    "tackle_with_assist_1_team",
    "tackle_with_assist_2_team",
)
_PBP_YARD_LINE_COLUMNS = ("yrdln", "drive_start_yard_line", "drive_end_yard_line", "end_yard_line")


def _play_with_team_codes(season_type: str, team: str) -> dict[str, str]:
    """Return one play whose every team-code column names ``team``, yard lines at its 25."""
    return {
        "season_type": season_type,
        **dict.fromkeys(_PBP_TEAM_CODE_COLUMNS, team),
        **dict.fromkeys(_PBP_YARD_LINE_COLUMNS, f"{team} 25"),
    }


@pytest.mark.parametrize(("alias", "canonical"), sorted(TEAM_ABBR_ALIASES.items()))
def test_load_pbp_data_normalizes_every_team_code_column(
    monkeypatch: pytest.MonkeyPatch, alias: str, canonical: str
) -> None:
    """Verify every column holding a team code is normalized, yard-line text included.

    A raw ``LA`` left in ``penalty_team``, ``td_team``, or ``drive_start_yard_line`` never matches
    the normalized ``posteam`` (``LAR``), which zeroed the Rams' penalty and touchdown counts.
    """
    # Arrange
    pbp = pl.DataFrame([_play_with_team_codes("REG", alias)])
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_pbp_data(2025)

    # Assert
    assert result.to_dicts() == [_play_with_team_codes("REG", canonical)]


def test_load_playoff_pbp_data_normalizes_every_team_code_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the playoff loader normalizes the same team-code columns as the season loader."""
    # Arrange
    pbp = pl.DataFrame([_play_with_team_codes("POST", "LA")])
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_playoff_pbp_data(2025)

    # Assert
    assert result.to_dicts() == [_play_with_team_codes("POST", "LAR")]


def test_load_pbp_data_rewrites_only_the_team_code_in_yard_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify yard lines keep their yards and leave midfield, nulls, and other codes alone.

    Each shape occurs in nflverse yard lines: ``50`` and ``MID 50`` at midfield, ``LAC 30`` (a
    code that starts like the alias ``LA``), and 2002's ``LA -10`` end yard line.
    """
    # Arrange
    pbp = pl.DataFrame(
        {
            "season_type": ["REG"] * 6,
            "yrdln": ["LA 25", "LAC 30", "50", "MID 50", "LA -10", None],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_pbp_data(2025)

    # Assert
    assert result.get_column("yrdln").to_list() == [
        "LAR 25",
        "LAC 30",
        "50",
        "MID 50",
        "LAR -10",
        None,
    ]


def test_load_weekly_team_stats_credits_rams_penalties_touchdowns_and_drive_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Rams team-games count their penalties, touchdowns, and drive starts.

    nflverse writes the Rams as ``LA`` in ``penalty_team``, ``td_team``, and yard lines, not only
    in ``posteam``. Fixture, SEA at LA: the Rams false-start, run 5 yards, and score on a 70-yard
    run, a drive from their own 25. Seattle's drive from its own 30 draws Rams pass interference
    (20 yards), then ends in a fumble the Rams return for a touchdown.
    """
    # Arrange
    game = {"game_id": "2025_01_SEA_LA", "season": 2025, "season_type": "REG", "week": 1}
    rams_drive = {
        **game,
        "posteam": "LA",
        "defteam": "SEA",
        "fixed_drive": 1,
        "fixed_drive_result": "Touchdown",
        "drive_start_yard_line": "LA 25",
    }
    seattle_drive = {
        **game,
        "posteam": "SEA",
        "defteam": "LA",
        "fixed_drive": 2,
        "fixed_drive_result": "Opp touchdown",
        "drive_start_yard_line": "SEA 30",
    }
    quiet = {
        "pass": 0,
        "rush": 0,
        "qb_dropback": 0,
        "rushing_yards": 0.0,
        "fumble": 0,
        "fumble_lost": 0,
        "touchdown": 0,
        "td_team": None,
        "penalty": 0,
        "penalty_team": None,
        "penalty_type": None,
        "penalty_yards": None,
    }
    pbp = pl.DataFrame(
        [
            {
                **rams_drive,
                **quiet,
                "penalty": 1,
                "penalty_team": "LA",
                "penalty_type": "False Start",
                "penalty_yards": 5.0,
            },
            {**rams_drive, **quiet, "rush": 1, "rushing_yards": 5.0},
            {
                **rams_drive,
                **quiet,
                "rush": 1,
                "rushing_yards": 70.0,
                "touchdown": 1,
                "td_team": "LA",
            },
            {
                **seattle_drive,
                **quiet,
                "pass": 1,
                "qb_dropback": 1,
                "penalty": 1,
                "penalty_team": "LA",
                "penalty_type": "Defensive Pass Interference",
                "penalty_yards": 20.0,
            },
            {
                **seattle_drive,
                **quiet,
                "rush": 1,
                "fumble": 1,
                "fumble_lost": 1,
                "touchdown": 1,
                "td_team": "LA",
            },
        ]
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_SEA_LA"],
            "game_type": ["REG"],
            "week": [1],
            "home_team": ["LA"],
            "away_team": ["SEA"],
            "home_score": [14],
            "away_score": [0],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_player_stats", stub(pl.DataFrame))
    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))
    monkeypatch.setattr(data_loader.nfl, "load_team_stats", stub(pl.DataFrame))
    rams_columns = {
        "penalties": 2,
        "penalty_yards": 25.0,
        "offensive_penalties": 1,
        "offensive_penalty_yards": 5.0,
        "defensive_penalties": 1,
        "defensive_penalty_yards": 20.0,
        "defensive_pass_interference": 1,
        # One false start and one penalty over the Rams' two snaps.
        "presnap_penalty_rate": 0.5,
        "penalty_rate_per_offensive_snap": 0.5,
        "penalty_differential": -2,
        "penalty_yards_differential": -25.0,
        "def_tds": 1,
        "fumble_recovery_tds": 1,
        "total_tds": 2,
        "avg_starting_field_position": 25.0,
        "long_field_score_pct": 1.0,
        "avg_starting_field_position_allowed": 30.0,
    }
    seattle_columns = {
        "penalty_differential": 2,
        "avg_starting_field_position": 30.0,
        "avg_starting_field_position_allowed": 25.0,
    }

    # Act
    result = data_loader.load_weekly_team_stats(2025)

    # Assert
    rams = result.filter(pl.col("team") == "LAR").select(list(rams_columns)).to_dicts()
    seattle = result.filter(pl.col("team") == "SEA").select(list(seattle_columns)).to_dicts()
    assert rams == [rams_columns]
    assert seattle == [seattle_columns]


def test_load_weekly_player_stats_filters_regular_season_and_normalizes_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify weekly player stats are REG-filtered and team columns are normalized."""
    # Arrange
    player_stats = pl.DataFrame(
        {
            "season_type": ["REG", "POST"],
            "week": [1, 2],
            "player_id": ["P1", "P2"],
            "player_name": ["Player One", "Player Two"],
            "team": ["LA", "DEN"],
            "opponent_team": ["SEA", "KC"],
            "passing_epa": [4.2, 1.1],
        }
    )

    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )

    # Act
    result = data_loader.load_weekly_player_stats(2025)

    # Assert
    assert result.height == 1
    assert result.select("team").item() == "LAR"
    assert result.select("opponent_team").item() == "SEA"
    assert result.select("passing_epa").item() == 4.2


def test_load_snap_counts_data_normalizes_team_when_season_type_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify snap-count loading normalizes team abbreviations without assuming season_type."""
    # Arrange
    snap_counts = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["LA"],
            "player": ["Rams QB"],
            "position": ["QB"],
            "offense_snaps": [58.0],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))

    # Act
    result = data_loader.load_snap_counts_data(2025)

    # Assert
    assert result.height == 1
    assert result.select("team").item() == "LAR"
    assert result.select("offense_snaps").item() == 58.0


def test_load_espn_qbr_filters_regular_season_and_normalizes_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify QBR loading keeps regular season only and canonicalizes team codes."""
    # Arrange
    qbr = pl.DataFrame(
        {
            "season": [2025, 2025, 2024],
            "season_type": ["Regular", "Playoffs", "Regular"],
            "team_abb": ["WSH", "KC", "LA"],
            "player_id": ["10", "11", "12"],
            "qbr_total": [55.0, 60.0, 70.0],
            "qbr_raw": [54.0, 61.0, 69.0],
        }
    )
    requested: list[str] = []

    def fake_fetch(url: str) -> pl.DataFrame:
        requested.append(url)
        return qbr

    monkeypatch.setattr(data_loader, "_fetch_release_parquet", fake_fetch)

    # Act
    result = data_loader.load_espn_qbr("season", seasons=[2025])

    # Assert
    assert requested == [data_loader.ESPN_QBR_RELEASE_URLS["season"]]
    assert result.height == 1
    assert result.select("team_abb").item() == "WAS"
    assert result.select("qbr_raw").item() == 54.0


def test_load_espn_qbr_week_level_keeps_all_seasons_and_normalizes_opponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify week-level QBR uses the week asset and normalizes opponent codes too."""
    # Arrange
    qbr = pl.DataFrame(
        {
            "season": [2024, 2025],
            "season_type": ["Regular", "Regular"],
            "team_abb": ["KC", "DET"],
            "opp_abb": ["OAK", "WSH"],
            "qbr_total": [61.0, 66.0],
        }
    )
    requested: list[str] = []

    def fake_fetch(url: str) -> pl.DataFrame:
        requested.append(url)
        return qbr

    monkeypatch.setattr(data_loader, "_fetch_release_parquet", fake_fetch)

    # Act
    result = data_loader.load_espn_qbr("week")

    # Assert
    assert requested == [data_loader.ESPN_QBR_RELEASE_URLS["week"]]
    assert result.height == 2
    assert result.select("opp_abb").to_series().to_list() == ["LV", "WAS"]


def test_load_espn_qbr_rejects_unknown_level() -> None:
    """Verify an unknown QBR level fails fast with a clear error."""
    # Arrange
    from typing import Literal, cast

    bad_level = cast("Literal['season', 'week']", "quarter")

    # Act & Assert
    with pytest.raises(ValueError, match="season"):
        data_loader.load_espn_qbr(bad_level)


def test_fetch_release_parquet_reads_downloaded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the release download helper parses Parquet bytes from the response."""
    # Arrange
    import io
    import urllib.request
    from contextlib import contextmanager

    buffer = io.BytesIO()
    pl.DataFrame({"season": [2025]}).write_parquet(buffer)
    payload = buffer.getvalue()

    timeouts: list[float | None] = []

    @contextmanager
    def fake_urlopen(url: str, timeout: float | None = None):
        timeouts.append(timeout)
        yield io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # Act
    result = data_loader._fetch_release_parquet("https://example.invalid/x.parquet")

    # Assert
    assert result.select("season").item() == 2025
    assert timeouts == [60]


def test_load_qb_stats_adds_official_rushing_and_completion_percentage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify official rushing fields and derived QB rates flow into game rows."""
    # Arrange
    # Rows: a completion, three designed runs, a scramble, and a kneel. As in nflverse, the
    # scramble and the kneel have rush = 0 and a rusher but no passer, and the scramble is a pass.
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 6,
            "season_type": ["REG"] * 6,
            "week": [1] * 6,
            "posteam": ["DEN"] * 6,
            "passer_player_id": ["00-0031234", None, None, None, None, None],
            "passer_player_name": ["John Doe", None, None, None, None, None],
            "rusher_player_id": [
                None,
                "00-0031234",
                "00-0031234",
                "00-0031234",
                "00-0031234",
                "00-0031234",
            ],
            "rusher_player_name": [
                None,
                "John Doe",
                "John Doe",
                "John Doe",
                "John Doe",
                "John Doe",
            ],
            "qb_dropback": [1, 0, 0, 0, 1, 0],
            "pass": [1, 0, 0, 0, 1, 0],
            "complete_pass": [1, 0, 0, 0, 0, 0],
            "passing_yards": [18.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "yards_gained": [18.0, 10.0, 14.0, 0.0, 12.0, -1.0],
            "pass_touchdown": [0, 0, 0, 0, 0, 0],
            "interception": [0, 0, 0, 0, 0, 0],
            "sack": [0, 0, 0, 0, 0, 0],
            "fumble_lost": [0, 0, 0, 0, 0, 0],
            "qb_epa": [1.2, 0.0, 0.0, 0.0, 0.7, 0.0],
            "epa": [1.2, 0.6, 0.9, 0.0, 0.7, -0.2],
            "cpoe": [4.0, None, None, None, None, None],
            "rush": [0, 1, 1, 1, 0, 0],
            "qb_scramble": [0, 0, 0, 0, 1, 0],
            "qb_kneel": [0, 0, 0, 0, 0, 1],
            "qb_spike": [0, 0, 0, 0, 0, 0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"],
            "week": [1],
            "team": ["DEN"],
            "player": ["John Doe"],
            "pfr_player_id": ["DoeJo00"],
            "position": ["QB"],
            "offense_snaps": [60.0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "season_type": ["REG"],
            "game_id": ["2025_01_DEN_KC"],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "player_id": ["00-0031234"],
            "player_display_name": ["John Doe"],
            "position": ["QB"],
            "attempts": [8],
            "completions": [6],
            "passing_yards": [80.0],
            "passing_tds": [1],
            "passing_interceptions": [0],
            "sacks_suffered": [1],
            "sack_yards_lost": [-6.0],
            "passing_epa": [3.0],
            "passing_cpoe": [2.5],
            "carries": [5],
            "rushing_yards": [35.0],
            "rushing_tds": [1],
            "rushing_first_downs": [2],
            "rushing_epa": [1.5],
            "rushing_fumbles": [1],
            "rushing_fumbles_lost": [0],
            "rushing_2pt_conversions": [0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": ["QB"],
            "pfr_id": ["DoeJo00"],
        }
    )
    rosters_weekly = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["DEN"],
            "full_name": ["John Doe"],
            "gsis_id": ["00-0031234"],
            "pfr_id": ["DoeJo00"],
            "position": ["QB"],
            "game_type": ["REG"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        stub(lambda: player_stats),
    )
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(
        data_loader.nfl,
        "load_rosters_weekly",
        stub(lambda: rosters_weekly),
    )

    # Act
    result = data_loader.load_qb_stats(2025)

    # Assert
    row = result.to_dicts()[0]
    assert row["qb_carries"] == 5
    assert row["qb_rushing_yards"] == 35.0
    assert row["qb_rushing_tds"] == 1
    assert row["qb_rushing_first_downs"] == 2
    assert row["qb_rushing_epa"] == 1.5
    assert row["qb_rushing_fumbles"] == 1
    assert row["qb_rushing_fumbles_lost"] == 0
    assert row["qb_designed_carries"] == 3
    assert row["qb_designed_rush_yards"] == 24.0
    assert row["qb_designed_rush_epa"] == pytest.approx(1.5)
    assert row["qb_scrambles"] == 1
    assert row["qb_scramble_yards"] == 12.0
    assert row["qb_kneels"] == 1
    # The play-by-play split reconciles to the official carries.
    assert row["qb_designed_carries"] + row["qb_scrambles"] + row["qb_kneels"] == row["qb_carries"]
    assert row["qb_completion_pct"] == 0.75
    assert row["qb_yards_per_carry"] == 7.0
    assert abs(row["qb_epa_per_carry"] - 0.3) < 1e-9
    # Dropbacks count only plays with a passer, so the scramble is not one of them.
    assert row["qb_dropbacks"] == 1
    assert row["qb_scramble_rate"] == pytest.approx(1.0)
    assert row["qb_yards_per_scramble"] == pytest.approx(12.0)
    assert row["qb_designed_yards_per_carry"] == pytest.approx(8.0)
    assert row["qb_designed_epa_per_carry"] == pytest.approx(0.5)


def test_load_qb_identity_crosswalk_without_any_sources_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(data_loader.nfl, "load_players", stub(pl.DataFrame))
    monkeypatch.setattr(data_loader.nfl, "load_rosters_weekly", stub(pl.DataFrame))

    # Act
    crosswalk = data_loader.load_qb_identity_crosswalk(2025)

    # Assert
    assert crosswalk.is_empty()
    assert crosswalk.columns == ["qb_id", "snap_player_id", "qb_name", "qb_position"]


def test_load_official_weekly_qb_stats_without_player_ids_is_empty() -> None:
    # Act
    official = data_loader._load_official_weekly_qb_stats(
        pl.DataFrame({"week": [1]}), pl.DataFrame()
    )

    # Assert
    assert official.is_empty()


def test_override_qb_game_stats_without_official_rows_keeps_the_input() -> None:
    # Arrange
    qb_df = pl.DataFrame({"qb_id": ["q1"], "week": [1], "qb_attempts": [30]})

    # Act
    result = data_loader._override_qb_game_stats_with_official_weekly(qb_df, pl.DataFrame())

    # Assert
    assert result.equals(qb_df)


def test_load_official_weekly_team_surface_without_rows_is_typed_empty() -> None:
    # Act
    surface = data_loader._load_official_weekly_team_surface(pl.DataFrame())

    # Assert
    assert surface.is_empty()
    assert "official_passing_yards" in surface.columns


def test_override_team_game_stats_without_official_rows_keeps_the_input() -> None:
    # Arrange
    team_df = pl.DataFrame({"team": ["DEN"], "week": [1], "passing_yards": [250.0]})

    # Act
    result = data_loader._override_team_game_stats_with_official_weekly(team_df, pl.DataFrame())

    # Assert
    assert result.equals(team_df)


def test_load_playoff_pbp_data_without_a_season_type_column_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    pbp = pl.DataFrame({"game_id": ["g1"], "posteam": ["DEN"]})
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    playoff = data_loader.load_playoff_pbp_data(2025)

    # Assert
    assert playoff.is_empty()
    assert playoff.columns == ["game_id", "posteam"]


def test_load_espn_qbr_without_a_season_type_column_keeps_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    qbr = pl.DataFrame({"season": [2025], "team_abb": ["WSH"], "qbr_total": [55.0]})
    monkeypatch.setattr(data_loader, "_fetch_release_parquet", stub(lambda: qbr))

    # Act
    result = data_loader.load_espn_qbr("season")

    # Assert
    assert result.get_column("team_abb").to_list() == ["WAS"]


def test_load_official_weekly_qb_stats_without_an_identity_crosswalk_keeps_player_ids() -> None:
    # Arrange
    player_stats = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "team": ["DEN"],
            "player_id": ["00-0039732"],
            "player_display_name": ["Bo Nix"],
            "position": ["QB"],
            "attempts": [30],
        }
    )

    # Act
    official = data_loader._load_official_weekly_qb_stats(
        player_stats, data_loader._empty_qb_identity_crosswalk()
    )

    # Assert
    assert official.height == 1
    assert "qb_name" not in official.columns


def test_load_playoff_qb_stats_keeps_postseason_games_with_official_passing_epa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    games = ["2025_18_DEN_KC", "2025_19_DEN_LAC"]
    pbp = pl.DataFrame(
        {
            "game_id": games,
            "season_type": ["REG", "POST"],
            "week": [18, 19],
            "posteam": ["DEN", "DEN"],
            "passer_player_id": ["00-0031234", "00-0031234"],
            "passer_player_name": ["John Doe", "John Doe"],
            "qb_dropback": [1, 1],
            "pass": [1, 1],
            "complete_pass": [1, 1],
            "passing_yards": [18.0, 12.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [0, 0],
            "fumble_lost": [0, 0],
            "qb_epa": [1.2, 0.4],
            "cpoe": [4.0, 3.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": games,
            "game_type": ["REG", "WC"],
            "week": [18, 19],
            "team": ["DEN", "DEN"],
            "player": ["John Doe", "John Doe"],
            "pfr_player_id": ["DoeJo00", "DoeJo00"],
            "position": ["QB", "QB"],
            "offense_snaps": [60.0, 58.0],
        }
    )
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [18, 19],
            "season_type": ["REG", "POST"],
            "game_id": games,
            "team": ["DEN", "DEN"],
            "opponent_team": ["KC", "LAC"],
            "player_id": ["00-0031234", "00-0031234"],
            "player_display_name": ["John Doe", "John Doe"],
            "position": ["QB", "QB"],
            "attempts": [1, 1],
            "completions": [1, 1],
            "passing_yards": [18.0, 12.0],
            "passing_tds": [0, 0],
            "passing_interceptions": [0, 0],
            "sacks_suffered": [0, 0],
            "sack_yards_lost": [0.0, 0.0],
            "passing_epa": [1.2, 0.5],
            "passing_cpoe": [4.0, 3.0],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": ["QB"],
            "pfr_id": ["DoeJo00"],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(lambda: snap_counts))
    monkeypatch.setattr(data_loader.nfl, "load_player_stats", stub(lambda: player_stats))
    monkeypatch.setattr(data_loader.nfl, "load_players", lambda: players)
    monkeypatch.setattr(data_loader.nfl, "load_rosters_weekly", stub(pl.DataFrame))

    # Act
    result = data_loader.load_playoff_qb_stats(2025)

    # Assert
    assert result.select("game_id", "qb_dropbacks", "qb_epa_per_dropback").rows() == [
        ("2025_19_DEN_LAC", 1, 0.5)
    ]


def test_load_playoff_schedule_keeps_postseason_games(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_18_LA_SF", "2025_22_LA_NE"],
            "game_type": ["REG", "SB"],
            "home_team": ["SF", "NE"],
            "away_team": ["LA", "LA"],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_schedules", stub(lambda: schedule))

    # Act
    result = data_loader.load_playoff_schedule(2025)

    # Assert
    assert result.select("game_type", "home_team", "away_team").rows() == [("SB", "NE", "LAR")]


def test_load_playoff_qb_stats_skips_snap_counts_before_their_first_season(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    def _unexpected_snap_counts_call(seasons: int) -> pl.DataFrame:
        msg = f"snap counts loader should not run for season {seasons}"
        raise AssertionError(msg)

    pbp = pl.DataFrame(
        {
            "game_id": ["2005_18_DEN_NE"],
            "season_type": ["POST"],
            "week": [18],
            "posteam": ["DEN"],
            "passer_player_id": ["00-0031234"],
            "passer_player_name": ["John Doe"],
            "qb_dropback": [1],
            "pass": [1],
            "complete_pass": [1],
            "passing_yards": [10.0],
            "pass_touchdown": [0],
            "interception": [0],
            "sack": [0],
            "fumble_lost": [0],
            "qb_epa": [0.7],
            "cpoe": [1.5],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", _unexpected_snap_counts_call)
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))
    monkeypatch.setattr(data_loader.nfl, "load_player_stats", stub(pl.DataFrame))
    monkeypatch.setattr(data_loader.nfl, "load_players", pl.DataFrame)

    # Act
    result = data_loader.load_playoff_qb_stats(2005)

    # Assert
    assert result.select("qb_dropbacks", "qb_epa_per_dropback").rows() == [(1, 0.7)]


def test_filter_postseason_without_a_season_type_column_keeps_no_rows() -> None:
    # Arrange
    frame = pl.DataFrame({"game_id": ["g1"], "team": ["NE"]})

    # Act
    result = data_loader._filter_postseason(frame)

    # Assert
    assert result.is_empty()
    assert result.columns == ["game_id", "team"]


def test_load_playoff_qb_stats_without_postseason_plays_is_typed_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    regular_season_only = pl.DataFrame(
        {"game_id": ["2025_01_DEN_KC"], "season_type": ["REG"], "posteam": ["DEN"]}
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: regular_season_only))
    monkeypatch.setattr(data_loader.nfl, "load_snap_counts", stub(pl.DataFrame))
    monkeypatch.setattr(data_loader.nfl, "load_player_stats", stub(pl.DataFrame))
    monkeypatch.setattr(data_loader.nfl, "load_players", pl.DataFrame)
    monkeypatch.setattr(data_loader.nfl, "load_rosters_weekly", stub(pl.DataFrame))

    # Act
    result = data_loader.load_playoff_qb_stats(2025)

    # Assert
    assert result.is_empty()
    assert result.schema["qb_epa_per_dropback"] == pl.Float64


def test_load_pbp_data_treats_an_empty_team_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    pbp = pl.DataFrame(
        {
            "season_type": ["REG", "REG"],
            "week": [2, 2],
            "posteam": ["ATL", ""],
            "defteam": ["DAL", ""],
            "home_team": ["ATL", "ATL"],
            "away_team": ["DAL", "DAL"],
        }
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    result = data_loader.load_pbp_data(1999)

    # Assert
    assert result.get_column("posteam").to_list() == ["ATL", None]
    assert result.get_column("defteam").to_list() == ["DAL", None]


def test_load_wp_bins_bins_regular_season_plays_with_normalized_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    play = {
        "game_id": "2025_01_LA_SF",
        "week": 1,
        "posteam": "LA",
        "defteam": "SF",
        "qb_dropback": 1,
        "rush": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "special": 0,
        "epa": 0.4,
        "qb_epa": 0.4,
        "wp": 0.62,
        "passer_player_id": "00-1",
        "passer_player_name": "M.Stafford",
    }
    pbp = pl.DataFrame(
        [{**play, "season_type": "REG"}, {**play, "season_type": "POST", "game_id": "2025_19"}]
    )
    monkeypatch.setattr(data_loader.nfl, "load_pbp", stub(lambda: pbp))

    # Act
    team_bins, qb_bins = data_loader.load_wp_bins(2025)

    # Assert
    assert team_bins.select("game_id", "team", "wp_bin", "wp_bin_plays").rows() == [
        ("2025_01_LA_SF", "LAR", 38, 1)
    ]
    assert qb_bins.select("game_id", "qb_id", "qb_wp_bin_dropbacks").rows() == [
        ("2025_01_LA_SF", "00-1", 1)
    ]
