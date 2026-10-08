"""Tests for repairing player team codes in games nflverse credits to one team."""

import polars as pl

from nfl_sos_ratings.player_team_repair import (
    game_player_teams,
    one_team_games,
    repair_pbp_player_teams,
    repair_player_stats_teams,
    two_team_games_only,
)

# A game nflverse credits wholly to PIT (jax_1 plays for JAX, pit_1 for PIT), and a normal one.
_BROKEN = "2001_01_PIT_JAX"
_NORMAL = "2001_01_DEN_KC"


def _tackle_rows(game: str, teams: list[str]) -> pl.DataFrame:
    """Return one play per team code, each crediting a solo tackle to that team."""
    return pl.DataFrame(
        {
            "game_id": [game] * len(teams),
            "solo_tackle_1_team": teams,
            "solo_tackle_1_player_id": ["X"] * len(teams),
        }
    )


def test_one_team_games_finds_games_whose_player_teams_name_one_team() -> None:
    # Arrange
    plays = pl.concat(
        [_tackle_rows(_BROKEN, ["PIT"] * 20), _tackle_rows(_NORMAL, ["DEN", "KC"] * 10)]
    )

    # Act
    games = one_team_games(plays, ["solo_tackle_1_team"])

    # Assert
    assert games == [_BROKEN]


def test_one_team_games_ignores_games_with_too_few_team_values_to_tell() -> None:
    # Arrange
    plays = _tackle_rows(_BROKEN, ["PIT"] * 19)

    # Act
    games = one_team_games(plays, ["solo_tackle_1_team", "missing_team_column"])

    # Assert
    assert games == []


def test_game_player_teams_keeps_the_roster_team_that_plays_in_the_game() -> None:
    # Arrange
    rosters = pl.DataFrame(
        {
            "player_id": ["jax_1", "pit_1", "ten_1", "ten_1", "both_1", "both_1"],
            "team": ["JAX", "PIT", "TEN", "JAX", "JAX", "PIT"],
        }
    )

    # Act
    teams = game_player_teams([_BROKEN], rosters)

    # Assert
    assert teams.sort("player_id").rows() == [
        (_BROKEN, "jax_1", "JAX"),
        (_BROKEN, "pit_1", "PIT"),
        (_BROKEN, "ten_1", "JAX"),
    ]


def test_repair_pbp_player_teams_rebuilds_team_codes_from_player_ids() -> None:
    """Each team column takes its player's team; a penalty without a player reads the play text.

    Columns with no player behind them (``return_team``) become null in a repaired game, and other
    games keep their values.
    """
    # Arrange
    plays = pl.DataFrame(
        {
            "game_id": [_BROKEN, _BROKEN, _BROKEN, _NORMAL],
            "fumbled_1_team": ["PIT", None, None, "KC"],
            "fumbled_1_player_id": ["jax_1", None, None, "kc_1"],
            "td_team": [None, "PIT", None, "DEN"],
            "td_player_id": [None, "jax_1", None, "den_1"],
            "penalty_team": [None, "PIT", "PIT", "KC"],
            "penalty_player_id": [None, "pit_1", None, None],
            "return_team": ["PIT", None, None, "KC"],
            "desc": ["muff", "TD", "PENALTY on JAX, Delay of Game, 5 yards", "x"],
        }
    )
    player_teams = pl.DataFrame(
        {"game_id": [_BROKEN, _BROKEN], "player_id": ["jax_1", "pit_1"], "team": ["JAX", "PIT"]}
    )

    # Act
    repaired = repair_pbp_player_teams(plays, [_BROKEN], player_teams)

    # Assert
    assert repaired.select("fumbled_1_team", "td_team", "penalty_team", "return_team").rows() == [
        ("JAX", None, None, None),
        (None, "JAX", "PIT", None),
        (None, None, "JAX", None),
        ("KC", "DEN", "KC", "KC"),
    ]


def test_repair_player_stats_teams_rebuilds_team_and_opponent() -> None:
    # Arrange
    stats = pl.DataFrame(
        {
            "game_id": [_BROKEN, _BROKEN, _BROKEN, _NORMAL],
            "player_id": ["jax_1", "pit_1", "unknown_1", "den_1"],
            "team": ["PIT", "PIT", "PIT", "DEN"],
            "opponent_team": ["JAX", "JAX", "JAX", "KC"],
        }
    )
    player_teams = pl.DataFrame(
        {"game_id": [_BROKEN, _BROKEN], "player_id": ["jax_1", "pit_1"], "team": ["JAX", "PIT"]}
    )

    # Act
    repaired = repair_player_stats_teams(stats, [_BROKEN], player_teams)

    # Assert
    assert repaired.select("player_id", "team", "opponent_team").rows() == [
        ("jax_1", "JAX", "PIT"),
        ("pit_1", "PIT", "JAX"),
        ("unknown_1", None, None),
        ("den_1", "DEN", "KC"),
    ]


def test_two_team_games_only_drops_games_credited_to_one_team() -> None:
    # Arrange
    team_stats = pl.DataFrame(
        {
            "game_id": [_BROKEN, _NORMAL, _NORMAL],
            "team": ["PIT", "DEN", "KC"],
            "passing_yards": [379, 200, 190],
        }
    )

    # Act
    kept = two_team_games_only(team_stats)

    # Assert
    assert kept.get_column("game_id").to_list() == [_NORMAL, _NORMAL]
