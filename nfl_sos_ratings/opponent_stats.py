"""Core strength-of-schedule logic: compute opponent stat profiles."""

import polars as pl
from config import TEAM_TO_DIVISION
from team_stats import (
    compute_qb_stats_excluding_opponent,
    compute_team_stats_excluding_opponent,
)


def get_opponents(schedule_df: pl.DataFrame, team: str) -> list[str]:
    """Return list of unique regular-season opponents for a team."""
    home_opps = (
        schedule_df.filter(pl.col("home_team") == team).select("away_team").to_series().to_list()
    )
    away_opps = (
        schedule_df.filter(pl.col("away_team") == team).select("home_team").to_series().to_list()
    )
    return sorted(set(home_opps + away_opps))


def is_division_opponent(team: str, opponent: str) -> bool:
    """Check if two teams are in the same division."""
    return TEAM_TO_DIVISION.get(team) == TEAM_TO_DIVISION.get(opponent)


def compute_opponent_profile(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    team: str,
    schedule_df: pl.DataFrame,
) -> dict:
    """Compute the averaged opponent stat profile for a given team.

    For each of the team's 14 unique opponents, compute that opponent's per-game
    stats from all their games EXCLUDING matchups against `team`. Then average
    the 14 opponent profiles together (equal weight per opponent).

    Returns a dict with:
      - "team_stats": single-row DataFrame of averaged opponent team stats
      - "qb_stats": single-row DataFrame of averaged opponent QB stats
      - "opponents": list of opponent abbreviations
      - "opponent_details": list of dicts with per-opponent game counts
    """
    opponents = get_opponents(schedule_df, team)

    team_stat_rows = []
    qb_stat_rows = []
    opponent_details = []

    for opp in opponents:
        # Team stats for this opponent, excluding games against `team`
        opp_team_stats = compute_team_stats_excluding_opponent(
            weekly_df, opp, exclude_opponent=team
        )
        if opp_team_stats is not None:
            team_stat_rows.append(opp_team_stats)
            games = opp_team_stats.select("games_included").item()
        else:
            games = 0

        # QB stats for this opponent, excluding games against `team`
        opp_qb_stats = compute_qb_stats_excluding_opponent(
            qb_df, weekly_df, opp, exclude_opponent=team
        )
        if opp_qb_stats is not None:
            qb_stat_rows.append(opp_qb_stats)

        division = is_division_opponent(team, opp)
        opponent_details.append({"opponent": opp, "division": division, "games_included": games})

    # Average across all opponents (simple mean, equal weight per opponent)
    if team_stat_rows:
        combined_team = pl.concat(team_stat_rows)
        numeric_cols = [
            c
            for c, d in zip(combined_team.columns, combined_team.dtypes, strict=True)
            if d.is_numeric() and c not in {"games_included"}
        ]
        avg_team = combined_team.select(
            [pl.lit(team).alias("team")] + [pl.col(c).mean().alias(c) for c in numeric_cols]
        )
    else:
        avg_team = None

    if qb_stat_rows:
        combined_qb = pl.concat(qb_stat_rows)
        qb_numeric_cols = [
            c
            for c, d in zip(combined_qb.columns, combined_qb.dtypes, strict=True)
            if d.is_numeric()
        ]
        avg_qb = combined_qb.select(
            [pl.lit(team).alias("team")] + [pl.col(c).mean().alias(c) for c in qb_numeric_cols]
        )
    else:
        avg_qb = None

    return {
        "team_stats": avg_team,
        "qb_stats": avg_qb,
        "opponents": opponents,
        "opponent_details": opponent_details,
    }


def compute_all_opponent_profiles(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, dict]:
    """Compute opponent profiles for all 32 teams.

    Returns:
      - opp_team_stats_df: DataFrame with each team's averaged opponent team stats
      - opp_qb_stats_df: DataFrame with each team's averaged opponent QB stats
      - details: dict mapping team -> opponent_details list

    """
    team_rows = []
    qb_rows = []
    details = {}

    teams = sorted(weekly_df.select("team").unique().to_series().to_list())

    for team in teams:
        print(f"  Computing opponent profile for {team}...")
        profile = compute_opponent_profile(weekly_df, qb_df, team, schedule_df)

        if profile["team_stats"] is not None:
            team_rows.append(profile["team_stats"])
        if profile["qb_stats"] is not None:
            qb_rows.append(profile["qb_stats"])
        details[team] = profile["opponent_details"]

    opp_team_stats_df = pl.concat(team_rows).sort("team") if team_rows else None
    opp_qb_stats_df = pl.concat(qb_rows).sort("team") if qb_rows else None

    return opp_team_stats_df, opp_qb_stats_df, details
