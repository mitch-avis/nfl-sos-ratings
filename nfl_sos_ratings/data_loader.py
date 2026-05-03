"""Data loading functions wrapping nflreadpy."""

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.config import QB_NGS_COLS

_QB_ID_CANDIDATES = ["player_gsis_id", "player_id", "gsis_id", "player"]
_QB_NAME_CANDIDATES = ["player_display_name", "player_name", "player"]


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate present in `columns`, else None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


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
    """Load Next Gen Stats passing data as one row per quarterback game.

    Returns regular-season QB rows with identifier columns (`qb_id`, `qb_name`
    when available), team/week keys, and selected QB metrics prefixed `qb_`.
    """
    df = nfl.load_nextgen_stats(seasons=season, stat_type="passing")

    # Filter to regular season (week > 0; week 0 is the season summary row)
    df = df.filter((pl.col("season_type") == "REG") & (pl.col("week") > 0))

    if "attempts" in df.columns:
        df = df.filter(pl.col("attempts") > 0)

    # Select relevant columns, renaming to prefix with qb_ for clarity.
    keep_cols = ["team_abbr", "week"]
    rename_map = {}

    qb_id_col = _first_existing_column(df.columns, _QB_ID_CANDIDATES)
    qb_name_col = _first_existing_column(df.columns, _QB_NAME_CANDIDATES)
    if qb_id_col is not None:
        keep_cols.append(qb_id_col)
        rename_map[qb_id_col] = "qb_id"
    if qb_name_col is not None and qb_name_col not in keep_cols:
        keep_cols.append(qb_name_col)
        rename_map[qb_name_col] = "qb_name"

    for col in QB_NGS_COLS:
        if col in df.columns:
            keep_cols.append(col)
            rename_map[col] = f"qb_{col}"

    df = df.select(keep_cols).rename(rename_map)

    if "qb_id" not in df.columns:
        df = df.with_columns(pl.col("team_abbr").alias("qb_id"))
    if "qb_name" not in df.columns:
        df = df.with_columns(pl.col("team_abbr").alias("qb_name"))

    return df
