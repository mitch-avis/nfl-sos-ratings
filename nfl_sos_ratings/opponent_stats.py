"""Head-to-head-excluded opponent profiles: what a team's opponents did against everyone else.

For each team, every unique opponent is profiled from its per-game averages in games that did
not involve that team, and the opponent profiles are averaged with equal weight per opponent. A
division rival played twice is profiled once, because removing both head-to-head games makes the
two profiles identical. The averaged profile is descriptive context for the analyst UI; the
published ratings come from the simultaneous solve in ``team_rating``.
"""

from typing import TypedDict

import polars as pl

from nfl_sos_ratings.config import TEAM_TO_DIVISION
from nfl_sos_ratings.team_stats import compute_team_stats_excluding_opponent

type OpponentDetail = dict[str, str | bool | int]


class OpponentProfile(TypedDict):
    """One team's averaged opponent profile and the per-opponent game counts behind it."""

    team_stats: pl.DataFrame | None
    opponents: list[str]
    opponent_details: list[OpponentDetail]


def get_opponents(schedule_df: pl.DataFrame, team: str) -> list[str]:
    """Return the sorted unique regular-season opponents of ``team``."""
    home_opps = schedule_df.filter(pl.col("home_team") == team).get_column("away_team").to_list()
    away_opps = schedule_df.filter(pl.col("away_team") == team).get_column("home_team").to_list()
    return sorted(set(home_opps + away_opps))


def is_division_opponent(team: str, opponent: str) -> bool:
    """Return whether two teams share a division."""
    return TEAM_TO_DIVISION.get(team) == TEAM_TO_DIVISION.get(opponent)


def compute_opponent_profile(
    weekly_df: pl.DataFrame, team: str, schedule_df: pl.DataFrame
) -> OpponentProfile:
    """Return ``team``'s averaged opponent profile, built without head-to-head games.

    Args:
        weekly_df: One row per team-game.
        team: The team whose opponents are profiled.
        schedule_df: The regular-season schedule with ``home_team`` and ``away_team``.

    Returns:
        The averaged profile (``None`` when no opponent has other games), the opponent list,
        and per-opponent game counts.

    """
    opponents = get_opponents(schedule_df, team)
    rows: list[pl.DataFrame] = []
    details: list[OpponentDetail] = []
    for opponent in opponents:
        profile = compute_team_stats_excluding_opponent(weekly_df, opponent, exclude_opponent=team)
        games = 0
        if profile is not None:
            rows.append(profile)
            games = int(profile.get_column("games_included").item())
        details.append(
            {
                "opponent": opponent,
                "division": is_division_opponent(team, opponent),
                "games_included": games,
            }
        )

    averaged = None
    if rows:
        combined = pl.concat(rows)
        numeric = [
            column
            for column, dtype in combined.schema.items()
            if dtype.is_numeric() and column != "games_included"
        ]
        averaged = combined.select(
            pl.lit(team).alias("team"), *[pl.col(column).mean() for column in numeric]
        )
    return {"team_stats": averaged, "opponents": opponents, "opponent_details": details}


def compute_all_opponent_profiles(
    weekly_df: pl.DataFrame, schedule_df: pl.DataFrame
) -> tuple[pl.DataFrame | None, dict[str, list[OpponentDetail]]]:
    """Return every team's averaged opponent profile plus per-team opponent details."""
    rows: list[pl.DataFrame] = []
    details: dict[str, list[OpponentDetail]] = {}
    for team in sorted(weekly_df.get_column("team").unique().cast(pl.String).to_list()):
        profile = compute_opponent_profile(weekly_df, team, schedule_df)
        if profile["team_stats"] is not None:
            rows.append(profile["team_stats"])
        details[team] = profile["opponent_details"]
    return (pl.concat(rows).sort("team") if rows else None), details
