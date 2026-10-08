"""Compute per-game team statistics (team-level and QB-level)."""

import polars as pl

from nfl_sos_ratings.pbp_expressions import rate_expr, scrimmage_snap_expr, value_expr
from nfl_sos_ratings.team_stats_expanded import compute_expanded_team_game_stats

_DEFENSE_ONLY_PLAYER_STATS = [
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_safeties",
]


def _get_numeric_stat_cols(df: pl.DataFrame) -> list[str]:
    """Return numeric column names excluding identifiers."""
    exclude = {"season", "week", "season_type", "games"}
    return [
        col
        for col, dtype in zip(df.columns, df.dtypes, strict=True)
        if dtype.is_numeric() and col not in exclude
    ]


def _extract_points_per_team_game(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Pivot schedule scores into one row per team-game with points for and allowed."""
    select_keys = [key for key in ("game_id", "week") if key in schedule_df.columns]
    home = schedule_df.select(
        [
            *select_keys,
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent_team"),
            pl.lit(True).alias("is_home"),
            pl.col("home_score").alias("points_for"),
            pl.col("away_score").alias("points_allowed"),
        ]
    )
    away = schedule_df.select(
        [
            *select_keys,
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent_team"),
            pl.lit(False).alias("is_home"),
            pl.col("away_score").alias("points_for"),
            pl.col("home_score").alias("points_allowed"),
        ]
    )
    return pl.concat([home, away])


def _aggregate_defense_only_player_stats(player_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate defense-only player stats to one row per team-week-opponent."""
    if player_stats_df.is_empty():
        return pl.DataFrame(
            schema={"team": pl.String, "opponent_team": pl.String, "week": pl.Int64}
        )

    defense_cols = [
        column for column in _DEFENSE_ONLY_PLAYER_STATS if column in player_stats_df.columns
    ]
    if not defense_cols:
        return pl.DataFrame(
            schema={"team": pl.String, "opponent_team": pl.String, "week": pl.Int64}
        )

    group_keys = [
        key
        for key in ("season", "season_type", "week", "team", "opponent_team")
        if key in player_stats_df.columns
    ]
    return player_stats_df.group_by(group_keys).agg(
        [pl.col(column).fill_null(0).sum().alias(column) for column in defense_cols]
    )


def compute_team_game_stats_from_pbp(
    pbp_df: pl.DataFrame,
    player_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """Derive one row per team-game from PBP, plus defense-only player-stat add-ons.

    Per-snap rate fields are computed as the relevant game total divided by
    offensive or defensive snaps for that team-game.
    """
    if pbp_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "season": pl.Int64,
                "season_type": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "is_home": pl.Boolean,
                "games": pl.Int64,
            }
        )

    group_keys = [
        key for key in ("game_id", "season", "season_type", "week") if key in pbp_df.columns
    ]
    offense_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & scrimmage_snap_expr(pbp_df.columns)
        )
        .group_by([*group_keys, "posteam", "defteam"])
        .agg(
            [
                value_expr(pbp_df.columns, "passing_yards", 0.0).sum().alias("passing_yards"),
                value_expr(pbp_df.columns, "rushing_yards", 0.0).sum().alias("rushing_yards"),
                pl.when(value_expr(pbp_df.columns, "qb_dropback") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("passing_epa"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("rushing_epa"),
                value_expr(pbp_df.columns, "pass_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("passing_tds"),
                value_expr(pbp_df.columns, "rush_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("rushing_tds"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("passing_first_downs"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_first_downs"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(
                    pl.col("cpoe") if "cpoe" in pbp_df.columns else pl.lit(None, dtype=pl.Float64)
                )
                .otherwise(None)
                .mean()
                .alias("passing_cpoe"),
                value_expr(pbp_df.columns, "sack").sum().cast(pl.Int64).alias("sacks_suffered"),
                value_expr(pbp_df.columns, "interception")
                .sum()
                .cast(pl.Int64)
                .alias("passing_interceptions"),
                pl.when(value_expr(pbp_df.columns, "sack") > 0)
                .then(value_expr(pbp_df.columns, "fumble_lost"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("sack_fumbles_lost"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "fumble_lost"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_fumbles_lost"),
            ]
        )
        .rename({"posteam": "team", "defteam": "opponent_team"})
        .with_columns((pl.col("passing_yards") + pl.col("rushing_yards")).alias("total_yards"))
    )

    allowed_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & scrimmage_snap_expr(pbp_df.columns)
        )
        .group_by([*group_keys, "defteam", "posteam"])
        .agg(
            [
                value_expr(pbp_df.columns, "passing_yards", 0.0)
                .sum()
                .alias("passing_yards_allowed"),
                value_expr(pbp_df.columns, "rushing_yards", 0.0)
                .sum()
                .alias("rushing_yards_allowed"),
                pl.when(value_expr(pbp_df.columns, "qb_dropback") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("passing_epa_allowed"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("rushing_epa_allowed"),
                value_expr(pbp_df.columns, "pass_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("passing_tds_allowed"),
                value_expr(pbp_df.columns, "rush_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("rushing_tds_allowed"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("passing_first_downs_allowed"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_first_downs_allowed"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(
                    pl.col("cpoe") if "cpoe" in pbp_df.columns else pl.lit(None, dtype=pl.Float64)
                )
                .otherwise(None)
                .mean()
                .alias("passing_cpoe_allowed"),
            ]
        )
        .rename({"defteam": "team", "posteam": "opponent_team"})
        .with_columns(
            (pl.col("passing_yards_allowed") + pl.col("rushing_yards_allowed")).alias(
                "total_yards_allowed"
            )
        )
    )

    snap_counts = compute_team_snap_counts_from_pbp(pbp_df)
    points = _extract_points_per_team_game(schedule_df)
    defense_only = _aggregate_defense_only_player_stats(player_stats_df)

    join_keys = [key for key in group_keys if key in points.columns]
    join_keys.extend(["team", "opponent_team"])

    result = (
        offense_stats.join(allowed_stats, on=[*group_keys, "team", "opponent_team"], how="left")
        .join(
            snap_counts,
            on=[key for key in ("game_id", "week", "team") if key in offense_stats.columns],
            how="left",
        )
        .join(points, on=join_keys, how="left")
        .join(
            defense_only,
            on=[
                key for key in [*group_keys, "team", "opponent_team"] if key in defense_only.columns
            ],
            how="left",
        )
        .with_columns(pl.lit(1).cast(pl.Int64).alias("games"))
    )

    fill_zero_exprs = [
        pl.col(column).fill_null(0)
        for column in [
            "offensive_snaps",
            "defensive_snaps",
            "passing_yards_allowed",
            "rushing_yards_allowed",
            "total_yards_allowed",
            "passing_epa_allowed",
            "rushing_epa_allowed",
            "passing_tds_allowed",
            "rushing_tds_allowed",
            "passing_first_downs_allowed",
            "rushing_first_downs_allowed",
            *_DEFENSE_ONLY_PLAYER_STATS,
        ]
        if column in result.columns
    ]
    result = result.with_columns(fill_zero_exprs).with_columns(
        (pl.col("points_for") - pl.col("points_allowed")).alias("point_margin"),
        pl.when(pl.col("points_for") > pl.col("points_allowed"))
        .then(1.0)
        .when(pl.col("points_for") < pl.col("points_allowed"))
        .then(0.0)
        .otherwise(0.5)
        .alias("win_value"),
    )

    rate_specs = [
        ("points_for", "offensive_snaps", "points_per_offensive_snap"),
        ("total_yards", "offensive_snaps", "total_yards_per_offensive_snap"),
        ("passing_yards", "offensive_snaps", "passing_yards_per_offensive_snap"),
        ("rushing_yards", "offensive_snaps", "rushing_yards_per_offensive_snap"),
        ("passing_epa", "offensive_snaps", "passing_epa_per_offensive_snap"),
        ("rushing_epa", "offensive_snaps", "rushing_epa_per_offensive_snap"),
        ("passing_tds", "offensive_snaps", "passing_tds_per_offensive_snap"),
        ("rushing_tds", "offensive_snaps", "rushing_tds_per_offensive_snap"),
        ("sacks_suffered", "offensive_snaps", "sacks_suffered_per_offensive_snap"),
        (
            "passing_interceptions",
            "offensive_snaps",
            "passing_interceptions_per_offensive_snap",
        ),
        ("sack_fumbles_lost", "offensive_snaps", "sack_fumbles_lost_per_offensive_snap"),
        (
            "rushing_fumbles_lost",
            "offensive_snaps",
            "rushing_fumbles_lost_per_offensive_snap",
        ),
        (
            "passing_first_downs",
            "offensive_snaps",
            "passing_first_downs_per_offensive_snap",
        ),
        (
            "rushing_first_downs",
            "offensive_snaps",
            "rushing_first_downs_per_offensive_snap",
        ),
        ("points_allowed", "defensive_snaps", "points_allowed_per_defensive_snap"),
        (
            "total_yards_allowed",
            "defensive_snaps",
            "total_yards_allowed_per_defensive_snap",
        ),
        (
            "passing_yards_allowed",
            "defensive_snaps",
            "passing_yards_allowed_per_defensive_snap",
        ),
        (
            "rushing_yards_allowed",
            "defensive_snaps",
            "rushing_yards_allowed_per_defensive_snap",
        ),
        (
            "passing_epa_allowed",
            "defensive_snaps",
            "passing_epa_allowed_per_defensive_snap",
        ),
        (
            "rushing_epa_allowed",
            "defensive_snaps",
            "rushing_epa_allowed_per_defensive_snap",
        ),
        (
            "passing_tds_allowed",
            "defensive_snaps",
            "passing_tds_allowed_per_defensive_snap",
        ),
        (
            "rushing_tds_allowed",
            "defensive_snaps",
            "rushing_tds_allowed_per_defensive_snap",
        ),
        (
            "passing_first_downs_allowed",
            "defensive_snaps",
            "passing_first_downs_allowed_per_defensive_snap",
        ),
        (
            "rushing_first_downs_allowed",
            "defensive_snaps",
            "rushing_first_downs_allowed_per_defensive_snap",
        ),
        ("def_sacks", "defensive_snaps", "def_sacks_per_defensive_snap"),
        (
            "def_interceptions",
            "defensive_snaps",
            "def_interceptions_per_defensive_snap",
        ),
        (
            "def_pass_defended",
            "defensive_snaps",
            "def_pass_defended_per_defensive_snap",
        ),
        (
            "def_tackles_for_loss",
            "defensive_snaps",
            "def_tackles_for_loss_per_defensive_snap",
        ),
        ("def_qb_hits", "defensive_snaps", "def_qb_hits_per_defensive_snap"),
        (
            "def_fumbles_forced",
            "defensive_snaps",
            "def_fumbles_forced_per_defensive_snap",
        ),
        ("def_safeties", "defensive_snaps", "def_safeties_per_defensive_snap"),
    ]
    result = result.with_columns(
        [
            rate_expr(numerator, denominator, output)
            for numerator, denominator, output in rate_specs
            if {numerator, denominator}.issubset(set(result.columns))
        ]
    )

    expanded = compute_expanded_team_game_stats(pbp_df)
    if "team" in expanded.columns and not expanded.is_empty():
        expansion_keys = [
            key
            for key in [*group_keys, "team", "opponent_team"]
            if key in expanded.columns and key in result.columns
        ]
        result = result.join(expanded, on=expansion_keys, how="left")

    result = _add_receiving_display_mirrors(result)

    return result.sort([key for key in ("team", "week", "game_id") if key in result.columns])


# Receiving display mirrors: at team level the receiving surface restates the
# passing surface exactly (verified: team receiving yards equal gross passing
# yards). Kept for display only; every alias is duplicate_of its source in the
# metric registry and is never ratings-eligible.
_RECEIVING_ALIAS_SOURCES = {
    "targets": "attempts",
    "receptions": "completions",
    "receiving_yards": "passing_yards",
    "receiving_tds": "passing_tds",
    "receiving_air_yards": "passing_air_yards",
    "receiving_yards_after_catch": "passing_yards_after_catch",
    "receiving_first_downs": "passing_first_downs",
    "catch_rate": "completion_pct",
    "targets_faced": "attempts_faced",
    "receptions_allowed": "completions_allowed",
    "receiving_yards_allowed": "passing_yards_allowed",
    "catch_rate_allowed": "completion_pct_allowed",
}


def _add_receiving_display_mirrors(result: pl.DataFrame) -> pl.DataFrame:
    """Add display-only receiving aliases of the passing surface."""
    aliases = [
        pl.col(source).alias(alias)
        for alias, source in _RECEIVING_ALIAS_SOURCES.items()
        if source in result.columns
    ]
    return result.with_columns(aliases) if aliases else result


def compute_team_snap_counts_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Compute team offensive and defensive snap counts from play-by-play data."""
    if pbp_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "offensive_snaps": pl.Int64,
                "defensive_snaps": pl.Int64,
            }
        )

    scrimmage_plays = pbp_df.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & scrimmage_snap_expr(pbp_df.columns)
    )

    offense = scrimmage_plays.select(
        "game_id",
        "week",
        pl.col("posteam").alias("team"),
        pl.lit(1).alias("offensive_snaps"),
        pl.lit(0).alias("defensive_snaps"),
    )
    defense = scrimmage_plays.select(
        "game_id",
        "week",
        pl.col("defteam").alias("team"),
        pl.lit(0).alias("offensive_snaps"),
        pl.lit(1).alias("defensive_snaps"),
    )

    return (
        pl.concat([offense, defense])
        .group_by(["game_id", "week", "team"])
        .agg(
            [
                pl.col("offensive_snaps").sum().cast(pl.Int64).alias("offensive_snaps"),
                pl.col("defensive_snaps").sum().cast(pl.Int64).alias("defensive_snaps"),
            ]
        )
        .sort(["team", "week", "game_id"])
    )


def compute_all_teams_per_game(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-game averages for all teams from weekly team stats.

    Returns a DataFrame with one row per team and per-game averages for every
    numeric stat column.
    """
    stat_cols = _get_numeric_stat_cols(weekly_df)

    return (
        weekly_df.group_by("team")
        .agg(
            [
                (pl.col(c).max() if c.startswith("longest_") else pl.col(c).mean()).alias(c)
                for c in stat_cols
            ]
            + [pl.col("team").count().alias("games_played")]
        )
        .sort("team")
    )


def compute_win_totals(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute wins, losses, ties, and win_pct per team from weekly game results.

    A game counts when both scores are present; a team held scoreless still records the loss.
    """
    valid = weekly_df.filter(
        pl.col("points_for").is_not_null() & pl.col("points_allowed").is_not_null()
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

    return filtered.select(
        [pl.lit(team).alias("team")]
        + [pl.col(c).mean().alias(c) for c in stat_cols]
        + [pl.lit(games).alias("games_included")]
    )
