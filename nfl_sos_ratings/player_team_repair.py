"""Repair player team codes in the games nflverse credits wholly to one team.

In Jacksonville's 2001-2002 home games, nflverse names the visiting team for every player: the
play-by-play team columns other than ``posteam`` and ``defteam`` (who fumbled, recovered, tackled,
scored, or drew a penalty), the weekly player stats, and the weekly team stats, which therefore
have one row holding both teams' totals (nflverse-pbp issue 92). These helpers find such games and
rebuild each player's team from the season roster, restricted to the two teams in the game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

# A game needs at least this many player team values before naming one team marks it as broken:
# every real game credits plays to both teams, but a short fixture or a stub may not.
MIN_PLAYER_TEAM_VALUES = 20
# Play-by-play team columns and the player id each one belongs to.
PBP_PLAYER_TEAM_IDS: dict[str, str] = {
    "td_team": "td_player_id",
    "penalty_team": "penalty_player_id",
    "fumbled_1_team": "fumbled_1_player_id",
    "fumbled_2_team": "fumbled_2_player_id",
    "fumble_recovery_1_team": "fumble_recovery_1_player_id",
    "fumble_recovery_2_team": "fumble_recovery_2_player_id",
    "forced_fumble_player_1_team": "forced_fumble_player_1_player_id",
    "forced_fumble_player_2_team": "forced_fumble_player_2_player_id",
    "solo_tackle_1_team": "solo_tackle_1_player_id",
    "solo_tackle_2_team": "solo_tackle_2_player_id",
    "assist_tackle_1_team": "assist_tackle_1_player_id",
    "assist_tackle_2_team": "assist_tackle_2_player_id",
    "assist_tackle_3_team": "assist_tackle_3_player_id",
    "assist_tackle_4_team": "assist_tackle_4_player_id",
    "tackle_with_assist_1_team": "tackle_with_assist_1_player_id",
    "tackle_with_assist_2_team": "tackle_with_assist_2_player_id",
}
TEAMS_PER_GAME = 2
# Play-by-play team columns with no player behind them: unknown in a repaired game.
PBP_TEAM_ONLY_COLUMNS = ("return_team", "timeout_team")
# A penalty on the team as a whole names no player; the play text names the team.
_PENALTY_TEAM_PATTERN = r"PENALTY on ([A-Z]{2,3})\b"


def one_team_games(frame: pl.DataFrame, team_columns: Sequence[str]) -> list[str]:
    """Return the games whose player team values, across ``team_columns``, name a single team."""
    present = [column for column in team_columns if column in frame.columns]
    if not present or "game_id" not in frame.columns:
        return []
    values = (
        frame.select("game_id", *(pl.col(column).cast(pl.String) for column in present))
        .unpivot(index="game_id", value_name="_team_value")
        .drop_nulls("_team_value")
    )
    return (
        values.group_by("game_id")
        .agg(pl.col("_team_value").n_unique().alias("teams"), pl.len().alias("values"))
        .filter((pl.col("teams") == 1) & (pl.col("values") >= MIN_PLAYER_TEAM_VALUES))
        .get_column("game_id")
        .sort()
        .to_list()
    )


def _game_sides(games: Sequence[str]) -> pl.DataFrame:
    """Return each game's visiting and home team, read from its id (``2001_01_PIT_JAX``)."""
    parts = [game.split("_") for game in games]
    return pl.DataFrame(
        {
            "game_id": list(games),
            "away": [part[-2] for part in parts],
            "home": [part[-1] for part in parts],
        },
        schema={"game_id": pl.String, "away": pl.String, "home": pl.String},
    )


def game_player_teams(games: Sequence[str], rosters: pl.DataFrame) -> pl.DataFrame:
    """Return ``game_id``, ``player_id``, and ``team`` for every rostered player in ``games``.

    ``rosters`` holds the season's ``player_id`` and (normalized) ``team`` pairs, one per team a
    player was on. A player on the roster of exactly one of the game's two teams gets that team; a
    player on both or neither is left out, so his team stays unknown.
    """
    candidates = (
        _game_sides(games)
        .join(rosters.select("player_id", "team").unique(), how="cross")
        .filter((pl.col("team") == pl.col("away")) | (pl.col("team") == pl.col("home")))
    )
    return (
        candidates.group_by("game_id", "player_id")
        .agg(pl.col("team").unique())
        .filter(pl.col("team").list.len() == 1)
        .select("game_id", "player_id", pl.col("team").list.first())
    )


def _player_team(player_teams: pl.DataFrame, id_column: str, alias: str) -> pl.DataFrame:
    """Return ``player_teams`` keyed for a join on ``id_column``, its team named ``alias``."""
    return player_teams.rename({"player_id": id_column, "team": alias})


def repair_pbp_player_teams(
    pbp: pl.DataFrame, games: Sequence[str], player_teams: pl.DataFrame
) -> pl.DataFrame:
    """Return ``pbp`` with every player team column in ``games`` rebuilt from ``player_teams``.

    A column takes the team of the player in its id column (``PBP_PLAYER_TEAM_IDS``), or null
    when the player is unknown; a penalty with no player takes the team its play text names, when
    that team plays in the game. Columns with no player behind them become null. Other games keep
    their values.
    """
    broken = pl.col("game_id").is_in(list(games))
    repaired = pbp.with_row_index("_row")
    sides = _game_sides(games)
    for team_column, id_column in PBP_PLAYER_TEAM_IDS.items():
        if team_column not in pbp.columns or id_column not in pbp.columns:
            continue
        mapped = f"_{team_column}"
        repaired = repaired.join(
            _player_team(player_teams, id_column, mapped), on=["game_id", id_column], how="left"
        )
        replacement = pl.col(mapped)
        if team_column == "penalty_team" and "desc" in pbp.columns:
            replacement = pl.coalesce(replacement, pl.col("_penalty_text_team"))
            repaired = repaired.join(sides, on="game_id", how="left").with_columns(
                pl.col("desc").str.extract(_PENALTY_TEAM_PATTERN, 1).alias("_penalty_text_team")
            )
            repaired = repaired.with_columns(
                pl.when(
                    (pl.col("_penalty_text_team") == pl.col("away"))
                    | (pl.col("_penalty_text_team") == pl.col("home"))
                )
                .then(pl.col("_penalty_text_team"))
                .alias("_penalty_text_team")
            )
        repaired = repaired.with_columns(
            pl.when(broken).then(replacement).otherwise(pl.col(team_column)).alias(team_column)
        )
    repaired = repaired.with_columns(
        pl.when(broken).then(None).otherwise(pl.col(column)).alias(column)
        for column in PBP_TEAM_ONLY_COLUMNS
        if column in pbp.columns
    )
    return repaired.sort("_row").select(pbp.columns)


def repair_player_stats_teams(
    stats: pl.DataFrame, games: Sequence[str], player_teams: pl.DataFrame
) -> pl.DataFrame:
    """Return weekly player stats with ``team`` and ``opponent_team`` rebuilt in ``games``.

    A player the roster places in the game gets his team and the other team as his opponent; an
    unknown player gets null for both. Other games keep their values.
    """
    broken = pl.col("game_id").is_in(list(games))
    repaired = (
        stats.with_row_index("_row")
        .join(
            _player_team(player_teams, "player_id", "_team"),
            on=["game_id", "player_id"],
            how="left",
        )
        .join(_game_sides(games), on="game_id", how="left")
        .with_columns(
            pl.when(broken).then(pl.col("_team")).otherwise(pl.col("team")).alias("team"),
            pl.when(broken)
            .then(
                pl.when(pl.col("_team") == pl.col("away"))
                .then(pl.col("home"))
                .when(pl.col("_team") == pl.col("home"))
                .then(pl.col("away"))
            )
            .otherwise(pl.col("opponent_team"))
            .alias("opponent_team"),
        )
    )
    return repaired.sort("_row").select(stats.columns)


def two_team_games_only(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Return weekly team stats without the games that have a row for only one team."""
    complete = (
        team_stats.group_by("game_id")
        .agg(pl.col("team").n_unique().alias("teams"))
        .filter(pl.col("teams") == TEAMS_PER_GAME)
        .get_column("game_id")
    )
    return team_stats.filter(pl.col("game_id").is_in(complete.to_list()))
