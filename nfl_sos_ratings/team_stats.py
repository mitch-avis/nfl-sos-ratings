"""Compute per-game team statistics (team-level and QB-level)."""

import polars as pl


def _get_numeric_stat_cols(df: pl.DataFrame) -> list[str]:
    """Return numeric column names excluding identifiers."""
    exclude = {"season", "week", "season_type", "games"}
    return [
        col
        for col, dtype in zip(df.columns, df.dtypes, strict=True)
        if dtype.is_numeric() and col not in exclude
    ]


def compute_all_teams_per_game(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-game averages for all teams from weekly team stats.

    Returns a DataFrame with one row per team and per-game averages for every
    numeric stat column.
    """
    stat_cols = _get_numeric_stat_cols(weekly_df)

    per_game = (
        weekly_df.group_by("team")
        .agg(
            [pl.col(c).mean().alias(c) for c in stat_cols]
            + [pl.col("team").count().alias("games_played")]
        )
        .sort("team")
    )
    return per_game


def compute_all_teams_qb_per_game(qb_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-game QB averages for all teams from NGS passing data.

    Returns a DataFrame with one row per team and per-game averages for every
    QB stat column.
    """
    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric() and col not in {"week"}
    ]

    per_game = (
        qb_df.group_by("team_abbr")
        .agg([pl.col(c).mean().alias(c) for c in qb_stat_cols])
        .rename({"team_abbr": "team"})
        .sort("team")
    )
    return per_game


def compute_win_totals(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute wins, losses, ties, and win_pct per team from weekly game results.

    A game counts only when both points_for and points_allowed are non-null
    and points_for > 0 (filters out unplayed weeks).
    """
    valid = weekly_df.filter(
        pl.col("points_for").is_not_null()
        & pl.col("points_allowed").is_not_null()
        & (pl.col("points_for") > 0)
    )
    return (
        valid.with_columns(
            [
                (pl.col("points_for") > pl.col("points_allowed")).cast(pl.Int32).alias("win"),
                (pl.col("points_for") < pl.col("points_allowed")).cast(pl.Int32).alias("loss"),
                (pl.col("points_for") == pl.col("points_allowed")).cast(pl.Int32).alias("tie"),
            ]
        )
        .group_by("team")
        .agg(
            [
                pl.col("win").sum().alias("wins"),
                pl.col("loss").sum().alias("losses"),
                pl.col("tie").sum().alias("ties"),
            ]
        )
        .with_columns(
            (
                (pl.col("wins") + 0.5 * pl.col("ties"))
                / (pl.col("wins") + pl.col("losses") + pl.col("ties"))
            ).alias("win_pct")
        )
        .sort("team")
    )


def compute_team_stats_excluding_opponent(
    weekly_df: pl.DataFrame, team: str, exclude_opponent: str
) -> pl.DataFrame | None:
    """Compute per-game averages for `team`, excluding games against `exclude_opponent`.

    Returns a single-row DataFrame with per-game stat averages, or None if no games remain.
    """
    stat_cols = _get_numeric_stat_cols(weekly_df)

    filtered = weekly_df.filter(
        (pl.col("team") == team) & (pl.col("opponent_team") != exclude_opponent)
    )
    games = filtered.height
    if games == 0:
        return None

    result = filtered.select(
        [pl.lit(team).alias("team")]
        + [pl.col(c).mean().alias(c) for c in stat_cols]
        + [pl.lit(games).alias("games_included")]
    )
    return result


def compute_qb_stats_excluding_opponent(
    qb_df: pl.DataFrame,
    weekly_df: pl.DataFrame,
    team: str,
    exclude_opponent: str,
) -> pl.DataFrame | None:
    """Compute per-game QB averages for `team`.

    Exclude weeks where they played `exclude_opponent`.
    Uses weekly_df to identify which weeks to exclude.
    Returns None if no games remain.
    """
    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric() and col not in {"week"}
    ]

    # Find weeks where team played the exclude_opponent
    exclude_weeks = (
        weekly_df.filter((pl.col("team") == team) & (pl.col("opponent_team") == exclude_opponent))
        .select("week")
        .to_series()
        .to_list()
    )

    filtered = qb_df.filter((pl.col("team_abbr") == team) & (~pl.col("week").is_in(exclude_weeks)))

    if filtered.height == 0:
        return None

    result = filtered.select(
        [pl.lit(team).alias("team")] + [pl.col(c).mean().alias(c) for c in qb_stat_cols]
    )
    return result
