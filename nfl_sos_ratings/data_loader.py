"""Data loading functions wrapping nflreadpy."""

import nflreadpy as nfl
import polars as pl
from config import QB_NGS_COLS


def _extract_points_per_team_week(schedule: pl.DataFrame) -> pl.DataFrame:
    """Pivot schedule scores into one row per team per week with points_for/points_allowed."""
    home = schedule.select(
        pl.col("home_team").alias("team"),
        pl.col("week"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_allowed"),
    )
    away = schedule.select(
        pl.col("away_team").alias("team"),
        pl.col("week"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_allowed"),
    )
    return pl.concat([home, away])


def load_weekly_team_stats(season: int) -> pl.DataFrame:
    """Load game-by-game team stats for a season, regular season only.

    Enriches the data with:
      - total_yards (passing_yards + rushing_yards)
      - points_for / points_allowed (from the schedule's score columns)
    """
    df = nfl.load_team_stats(seasons=season, summary_level="week")
    df = df.filter(pl.col("season_type") == "REG")

    # Add total yards
    df = df.with_columns((pl.col("passing_yards") + pl.col("rushing_yards")).alias("total_yards"))

    # Add points from schedule
    schedule = nfl.load_schedules(seasons=season)
    schedule = schedule.filter(pl.col("game_type") == "REG")
    points = _extract_points_per_team_week(schedule)
    df = df.join(points, on=["team", "week"], how="left")

    return df


def load_schedule(season: int) -> pl.DataFrame:
    """Load the regular season schedule for a given season."""
    df = nfl.load_schedules(seasons=season)
    df = df.filter(pl.col("game_type") == "REG")
    return df


def load_qb_stats(season: int) -> pl.DataFrame:
    """Load Next Gen Stats passing data and reduce to primary QB per team per week.

    The primary QB is defined as the player with the most attempts for a given
    team in a given week. Returns a DataFrame with one row per team per week
    containing the QB's NGS metrics.
    """
    df = nfl.load_nextgen_stats(seasons=season, stat_type="passing")

    # Filter to regular season (week > 0; week 0 is the season summary row)
    df = df.filter((pl.col("season_type") == "REG") & (pl.col("week") > 0))

    # Keep only the QB with the most attempts per team per week
    df = df.sort("attempts", descending=True).group_by(["team_abbr", "week"]).first()

    # Select relevant columns, renaming to prefix with qb_ for clarity
    keep_cols = ["team_abbr", "week"]
    rename_map = {}
    for col in QB_NGS_COLS:
        if col in df.columns:
            keep_cols.append(col)
            rename_map[col] = f"qb_{col}"

    df = df.select(keep_cols).rename(rename_map)
    return df
