"""Data loading functions wrapping nflreadpy plus direct nflverse release assets."""

import io
import urllib.request
from typing import Literal

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.config import TEAM_ABBR_ALIASES
from nfl_sos_ratings.qb_stats import compute_qb_game_stats_from_pbp
from nfl_sos_ratings.team_stats import compute_team_game_stats_from_pbp

# ESPN QBR has no nflreadpy load function yet; these are the official nflverse
# release assets (Parquet, the smallest published format).
ESPN_QBR_RELEASE_URLS: dict[str, str] = {
    "season": (
        "https://github.com/nflverse/nflverse-data/releases/download/espn_data/"
        "qbr_season_level.parquet"
    ),
    "week": (
        "https://github.com/nflverse/nflverse-data/releases/download/espn_data/"
        "qbr_week_level.parquet"
    ),
}

SNAP_COUNTS_START_SEASON = 2012
ROSTERS_WEEKLY_START_SEASON = 2002


def _season_is_before_source_floor(season: int, *, start_season: int) -> bool:
    """Return whether a season predates an upstream dataset's first available year."""
    return season < start_season


def _empty_snap_counts_data() -> pl.DataFrame:
    """Return the standard empty snap-count schema used by QB loading."""
    return pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team": pl.String,
            "player": pl.String,
            "pfr_player_id": pl.String,
            "position": pl.String,
            "offense_snaps": pl.Float64,
        }
    )


def _empty_qb_identity_crosswalk() -> pl.DataFrame:
    """Return the standard empty QB identity crosswalk schema."""
    return pl.DataFrame(
        schema={
            "qb_id": pl.String,
            "snap_player_id": pl.String,
            "qb_name": pl.String,
            "qb_position": pl.String,
        }
    )


def _standardize_qb_identity_source(
    df: pl.DataFrame,
    *,
    name_column: str,
    source_priority: int,
) -> pl.DataFrame:
    """Return one player-identity source in the shared crosswalk schema."""
    if df.is_empty():
        return _empty_qb_identity_crosswalk().with_columns(
            pl.lit(source_priority).cast(pl.Int64).alias("source_priority")
        )

    name_expr = (
        pl.col(name_column).cast(pl.String)
        if name_column in df.columns
        else pl.lit(None, dtype=pl.String)
    )
    qb_id_expr = (
        pl.col("gsis_id").cast(pl.String)
        if "gsis_id" in df.columns
        else pl.lit(None, dtype=pl.String)
    )
    snap_id_expr = (
        pl.col("pfr_id").cast(pl.String)
        if "pfr_id" in df.columns
        else pl.lit(None, dtype=pl.String)
    )
    position_expr = (
        pl.col("position").cast(pl.String)
        if "position" in df.columns
        else pl.lit(None, dtype=pl.String)
    )

    return (
        df.select(
            qb_id_expr.alias("qb_id"),
            snap_id_expr.alias("snap_player_id"),
            name_expr.alias("qb_name"),
            position_expr.alias("qb_position"),
        )
        .filter(pl.col("qb_id").is_not_null() | pl.col("snap_player_id").is_not_null())
        .with_columns(pl.lit(source_priority).cast(pl.Int64).alias("source_priority"))
    )


def load_qb_identity_crosswalk(season: int) -> pl.DataFrame:
    """Load canonical player identities used to normalize QB-source rows."""
    players = _standardize_qb_identity_source(
        nfl.load_players(),
        name_column="display_name",
        source_priority=0,
    )
    rosters_weekly_source = _empty_qb_identity_crosswalk()
    if not _season_is_before_source_floor(season, start_season=ROSTERS_WEEKLY_START_SEASON):
        rosters_weekly_source = _filter_regular_season(nfl.load_rosters_weekly(seasons=season))

    rosters_weekly = _standardize_qb_identity_source(
        rosters_weekly_source,
        name_column="full_name",
        source_priority=1,
    )

    identity_sources = pl.concat([players, rosters_weekly], how="diagonal_relaxed")
    if identity_sources.is_empty():
        return _empty_qb_identity_crosswalk()

    return (
        identity_sources.filter(pl.col("qb_id").is_not_null())
        .sort("source_priority")
        .group_by("qb_id")
        .agg(
            pl.col("snap_player_id").drop_nulls().first().alias("snap_player_id"),
            pl.col("qb_name").drop_nulls().first().alias("qb_name"),
            pl.col("qb_position").drop_nulls().first().alias("qb_position"),
        )
    )


_OFFICIAL_QB_RUSHING_FIELDS: dict[str, tuple[str, type[pl.Int64] | type[pl.Float64]]] = {
    "carries": ("official_qb_carries", pl.Int64),
    "rushing_yards": ("official_qb_rushing_yards", pl.Float64),
    "rushing_tds": ("official_qb_rushing_tds", pl.Int64),
    "rushing_first_downs": ("official_qb_rushing_first_downs", pl.Int64),
    "rushing_epa": ("official_qb_rushing_epa", pl.Float64),
    "rushing_fumbles": ("official_qb_rushing_fumbles", pl.Int64),
    "rushing_fumbles_lost": ("official_qb_rushing_fumbles_lost", pl.Int64),
    "rushing_2pt_conversions": ("official_qb_rushing_2pt_conversions", pl.Int64),
}


def _official_rushing_selection(columns: list[str]) -> list[pl.Expr]:
    """Return the official QB rushing selection, tolerating absent columns."""
    return [
        (pl.col(source).cast(dtype) if source in columns else pl.lit(None, dtype=dtype)).alias(
            target
        )
        for source, (target, dtype) in _OFFICIAL_QB_RUSHING_FIELDS.items()
    ]


def _load_official_weekly_qb_stats(
    weekly_player_stats_df: pl.DataFrame,
    qb_identity_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return authoritative weekly QB passing stats keyed to canonical QB IDs."""
    if weekly_player_stats_df.is_empty() or "player_id" not in weekly_player_stats_df.columns:
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team_abbr": pl.String,
                "qb_id": pl.String,
                "official_qb_attempts": pl.Int64,
                "official_qb_completions": pl.Int64,
                "official_qb_pass_yards": pl.Float64,
                "official_qb_pass_touchdowns": pl.Int64,
                "official_qb_interceptions": pl.Int64,
                "official_qb_sacks": pl.Int64,
                "official_qb_sack_yards_lost": pl.Float64,
                "official_qb_passing_epa": pl.Float64,
                "official_qb_completion_percentage_above_expectation": pl.Float64,
                "official_qb_carries": pl.Int64,
                "official_qb_rushing_yards": pl.Float64,
                "official_qb_rushing_tds": pl.Int64,
                "official_qb_rushing_first_downs": pl.Int64,
                "official_qb_rushing_epa": pl.Float64,
                "official_qb_rushing_fumbles": pl.Int64,
                "official_qb_rushing_fumbles_lost": pl.Int64,
                "official_qb_rushing_2pt_conversions": pl.Int64,
            }
        )

    qb_name_expr = (
        pl.col("player_display_name").cast(pl.String)
        if "player_display_name" in weekly_player_stats_df.columns
        else (
            pl.col("player_name").cast(pl.String)
            if "player_name" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.String)
        )
    )

    official_qb_stats = weekly_player_stats_df.filter(
        ((pl.col("position") == "QB") if "position" in weekly_player_stats_df.columns else True)
        & pl.col("team").is_not_null()
        & pl.col("week").is_not_null()
        & pl.col("player_id").is_not_null()
    ).select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String).alias("team_abbr"),
        pl.col("player_id").cast(pl.String).alias("qb_id"),
        qb_name_expr.alias("qb_name"),
        (
            pl.col("attempts").cast(pl.Int64)
            if "attempts" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_attempts"),
        (
            pl.col("completions").cast(pl.Int64)
            if "completions" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_completions"),
        (
            pl.col("passing_yards").cast(pl.Float64)
            if "passing_yards" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_pass_yards"),
        (
            pl.col("passing_tds").cast(pl.Int64)
            if "passing_tds" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_pass_touchdowns"),
        (
            pl.col("passing_interceptions").cast(pl.Int64)
            if "passing_interceptions" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_interceptions"),
        (
            pl.col("sacks_suffered").cast(pl.Int64)
            if "sacks_suffered" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_sacks"),
        (
            pl.col("sack_yards_lost").cast(pl.Float64)
            if "sack_yards_lost" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_sack_yards_lost"),
        (
            pl.col("passing_epa").cast(pl.Float64)
            if "passing_epa" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_passing_epa"),
        (
            pl.col("passing_cpoe").cast(pl.Float64)
            if "passing_cpoe" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_completion_percentage_above_expectation"),
        *_official_rushing_selection(weekly_player_stats_df.columns),
    )
    if official_qb_stats.is_empty() or qb_identity_df.is_empty():
        return official_qb_stats.drop("qb_name")

    return (
        official_qb_stats.join(
            qb_identity_df.select(["qb_id", "qb_name"])
            .unique(subset=["qb_id"], keep="first")
            .rename({"qb_name": "canonical_qb_name"}),
            on="qb_id",
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("canonical_qb_name"), pl.col("qb_name")]).alias("qb_name")
        )
        .drop(["canonical_qb_name", "qb_name"])
    )


def _override_qb_game_stats_with_official_weekly(
    qb_df: pl.DataFrame,
    official_qb_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace attempt-based QB game fields with authoritative weekly player stats."""
    if qb_df.is_empty() or official_qb_stats_df.is_empty():
        return qb_df

    return (
        qb_df.join(
            official_qb_stats_df,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("official_qb_attempts"), pl.col("qb_attempts")])
            .cast(pl.Int64)
            .alias("qb_attempts"),
            pl.coalesce([pl.col("official_qb_completions"), pl.col("qb_completions")])
            .cast(pl.Int64)
            .alias("qb_completions"),
            pl.coalesce([pl.col("official_qb_pass_yards"), pl.col("qb_pass_yards")]).alias(
                "qb_pass_yards"
            ),
            pl.coalesce([pl.col("official_qb_pass_touchdowns"), pl.col("qb_pass_touchdowns")])
            .cast(pl.Int64)
            .alias("qb_pass_touchdowns"),
            pl.coalesce([pl.col("official_qb_interceptions"), pl.col("qb_interceptions")])
            .cast(pl.Int64)
            .alias("qb_interceptions"),
            pl.coalesce([pl.col("official_qb_sacks"), pl.col("qb_sacks")])
            .cast(pl.Int64)
            .alias("qb_sacks"),
            pl.coalesce(
                [pl.col("official_qb_sack_yards_lost"), pl.col("qb_sack_yards_lost")]
            ).alias("qb_sack_yards_lost"),
            pl.coalesce([pl.col("official_qb_passing_epa"), pl.col("qb_passing_epa")]).alias(
                "qb_passing_epa"
            ),
            pl.coalesce(
                [
                    pl.col("official_qb_completion_percentage_above_expectation"),
                    pl.col("qb_completion_percentage_above_expectation"),
                ]
            ).alias("qb_completion_percentage_above_expectation"),
            pl.col("official_qb_carries").fill_null(0).cast(pl.Int64).alias("qb_carries"),
            pl.col("official_qb_rushing_yards").fill_null(0.0).alias("qb_rushing_yards"),
            pl.col("official_qb_rushing_tds").fill_null(0).cast(pl.Int64).alias("qb_rushing_tds"),
            pl.col("official_qb_rushing_first_downs")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_first_downs"),
            pl.col("official_qb_rushing_epa").fill_null(0.0).alias("qb_rushing_epa"),
            pl.col("official_qb_rushing_fumbles")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_fumbles"),
            pl.col("official_qb_rushing_fumbles_lost")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_fumbles_lost"),
            pl.col("official_qb_rushing_2pt_conversions")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_2pt_conversions"),
        )
        .with_columns(
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_passing_epa") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_epa_per_dropback"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_pass_yards") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_pass_yards_per_dropback"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(
                (pl.col("qb_pass_touchdowns") - pl.col("qb_interceptions")) / pl.col("qb_dropbacks")
            )
            .otherwise(None)
            .alias("qb_td_int_margin_rate"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_sacks") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_sack_rate"),
            pl.when((pl.col("qb_attempts") + pl.col("qb_sacks")) > 0)
            .then(
                (
                    pl.col("qb_pass_yards")
                    + (20.0 * pl.col("qb_pass_touchdowns"))
                    - (45.0 * pl.col("qb_interceptions"))
                    - pl.col("qb_sack_yards_lost")
                )
                / (pl.col("qb_attempts") + pl.col("qb_sacks"))
            )
            .otherwise(None)
            .alias("qb_any_a"),
        )
        .with_columns(
            pl.when(pl.col("qb_attempts") > 0)
            .then(pl.col("qb_completions") / pl.col("qb_attempts"))
            .otherwise(None)
            .alias("qb_completion_pct"),
            pl.when(pl.col("qb_carries") > 0)
            .then(pl.col("qb_rushing_yards") / pl.col("qb_carries"))
            .otherwise(None)
            .alias("qb_yards_per_carry"),
            pl.when(pl.col("qb_carries") > 0)
            .then(pl.col("qb_rushing_epa") / pl.col("qb_carries"))
            .otherwise(None)
            .alias("qb_epa_per_carry"),
        )
        .drop(
            [
                "official_qb_attempts",
                "official_qb_completions",
                "official_qb_pass_yards",
                "official_qb_pass_touchdowns",
                "official_qb_interceptions",
                "official_qb_sacks",
                "official_qb_sack_yards_lost",
                "official_qb_passing_epa",
                "official_qb_completion_percentage_above_expectation",
                *[target for target, _ in _OFFICIAL_QB_RUSHING_FIELDS.values()],
            ]
        )
    )


def _normalize_team_abbreviations(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize known source-specific team abbreviations in selected columns."""
    exprs = [
        pl.col(column).replace(TEAM_ABBR_ALIASES).alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _filter_regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """Filter a frame to regular-season rows when a season-type column is present."""
    for column in ("season_type", "game_type"):
        if column in df.columns:
            return df.filter(pl.col(column) == "REG")
    return df


def _extract_points_per_team_week(schedule: pl.DataFrame) -> pl.DataFrame:
    """Pivot schedule scores into one row per team per week with points_for/points_allowed."""
    schedule = _normalize_team_abbreviations(schedule, ["home_team", "away_team"])
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


def _fetch_release_parquet(url: str) -> pl.DataFrame:
    """Download one nflverse release Parquet asset into a dataframe."""
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URLs above
        return pl.read_parquet(io.BytesIO(response.read()))


def load_espn_qbr(
    level: Literal["season", "week"] = "season",
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Load ESPN QBR from the nflverse release assets.

    nflreadpy has no QBR load function yet, so this downloads the published
    Parquet directly. Rows are filtered to the regular season, and team codes
    (ESPN's WSH/LA plus historical OAK/SD/STL) are normalized to this
    project's abbreviations.
    """
    if level not in ESPN_QBR_RELEASE_URLS:
        valid_levels = ", ".join(sorted(ESPN_QBR_RELEASE_URLS))
        raise ValueError(f"Unknown QBR level {level!r}; expected one of: {valid_levels}")

    qbr_df = _fetch_release_parquet(ESPN_QBR_RELEASE_URLS[level])
    if "season_type" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("season_type") == "Regular")
    if seasons is not None and "season" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("season").is_in(seasons))

    team_columns = [column for column in ("team_abb", "opp_abb") if column in qbr_df.columns]
    return _normalize_team_abbreviations(qbr_df, team_columns)


def load_pbp_data(season: int) -> pl.DataFrame:
    """Load regular-season play-by-play data with normalized team abbreviations."""
    df = nfl.load_pbp(seasons=season)
    df = _filter_regular_season(df)
    return _normalize_team_abbreviations(df, ["posteam", "defteam", "home_team", "away_team"])


def load_weekly_player_stats(season: int) -> pl.DataFrame:
    """Load regular-season weekly player stats with normalized team abbreviations."""
    df = nfl.load_player_stats(seasons=season, summary_level="week")
    df = _filter_regular_season(df)
    return _normalize_team_abbreviations(df, ["team", "opponent_team"])


def load_snap_counts_data(season: int) -> pl.DataFrame:
    """Load snap-count data with normalized team abbreviations."""
    if _season_is_before_source_floor(season, start_season=SNAP_COUNTS_START_SEASON):
        return _empty_snap_counts_data()

    df = nfl.load_snap_counts(seasons=season)
    df = _filter_regular_season(df)
    return _normalize_team_abbreviations(df, ["team"])


def load_official_weekly_team_stats(season: int) -> pl.DataFrame:
    """Load official weekly team stats with normalized team abbreviations."""
    df = nfl.load_team_stats(seasons=season, summary_level="week")
    df = _filter_regular_season(df)
    return _normalize_team_abbreviations(df, ["team", "opponent_team"])


def _load_official_weekly_team_surface(official_team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Return authoritative weekly team offense stats for published columns."""
    if official_team_stats_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "official_passing_yards": pl.Float64,
                "official_rushing_yards": pl.Float64,
                "official_total_yards": pl.Float64,
                "official_passing_tds": pl.Int64,
                "official_rushing_tds": pl.Int64,
                "official_passing_first_downs": pl.Int64,
                "official_rushing_first_downs": pl.Int64,
                "official_passing_epa": pl.Float64,
                "official_rushing_epa": pl.Float64,
                "official_passing_cpoe": pl.Float64,
                "official_sacks_suffered": pl.Int64,
                "official_passing_interceptions": pl.Int64,
                "official_sack_fumbles_lost": pl.Int64,
                "official_rushing_fumbles_lost": pl.Int64,
            }
        )

    passing_yards_expr = (
        pl.col("passing_yards").cast(pl.Float64)
        if "passing_yards" in official_team_stats_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    rushing_yards_expr = (
        pl.col("rushing_yards").cast(pl.Float64)
        if "rushing_yards" in official_team_stats_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )

    return official_team_stats_df.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("opponent_team").cast(pl.String),
        passing_yards_expr.alias("official_passing_yards"),
        rushing_yards_expr.alias("official_rushing_yards"),
        (passing_yards_expr + rushing_yards_expr).alias("official_total_yards"),
        (
            pl.col("passing_tds").cast(pl.Int64)
            if "passing_tds" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_tds"),
        (
            pl.col("rushing_tds").cast(pl.Int64)
            if "rushing_tds" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_tds"),
        (
            pl.col("passing_first_downs").cast(pl.Int64)
            if "passing_first_downs" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_first_downs"),
        (
            pl.col("rushing_first_downs").cast(pl.Int64)
            if "rushing_first_downs" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_first_downs"),
        (
            pl.col("passing_epa").cast(pl.Float64)
            if "passing_epa" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_passing_epa"),
        (
            pl.col("rushing_epa").cast(pl.Float64)
            if "rushing_epa" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_rushing_epa"),
        (
            pl.col("passing_cpoe").cast(pl.Float64)
            if "passing_cpoe" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_passing_cpoe"),
        (
            pl.col("sacks_suffered").cast(pl.Int64)
            if "sacks_suffered" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_sacks_suffered"),
        (
            pl.col("passing_interceptions").cast(pl.Int64)
            if "passing_interceptions" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_interceptions"),
        (
            pl.col("sack_fumbles_lost").cast(pl.Int64)
            if "sack_fumbles_lost" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_sack_fumbles_lost"),
        (
            pl.col("rushing_fumbles_lost").cast(pl.Int64)
            if "rushing_fumbles_lost" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_fumbles_lost"),
    )


def _override_team_game_stats_with_official_weekly(
    team_df: pl.DataFrame,
    official_team_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace published team stat columns with authoritative weekly team stats."""
    if team_df.is_empty() or official_team_stats_df.is_empty():
        return team_df

    official_offense = _load_official_weekly_team_surface(official_team_stats_df)
    official_allowed = official_offense.select(
        pl.col("game_id"),
        pl.col("week"),
        pl.col("opponent_team").alias("team"),
        pl.col("team").alias("opponent_team"),
        pl.col("official_passing_yards").alias("official_passing_yards_allowed"),
        pl.col("official_rushing_yards").alias("official_rushing_yards_allowed"),
        pl.col("official_total_yards").alias("official_total_yards_allowed"),
        pl.col("official_passing_tds").alias("official_passing_tds_allowed"),
        pl.col("official_rushing_tds").alias("official_rushing_tds_allowed"),
        pl.col("official_passing_first_downs").alias("official_passing_first_downs_allowed"),
        pl.col("official_rushing_first_downs").alias("official_rushing_first_downs_allowed"),
        pl.col("official_passing_epa").alias("official_passing_epa_allowed"),
        pl.col("official_rushing_epa").alias("official_rushing_epa_allowed"),
        pl.col("official_passing_cpoe").alias("official_passing_cpoe_allowed"),
    )

    result = (
        team_df.join(official_offense, on=["game_id", "week", "team", "opponent_team"], how="left")
        .join(official_allowed, on=["game_id", "week", "team", "opponent_team"], how="left")
        .with_columns(
            pl.coalesce([pl.col("official_passing_yards"), pl.col("passing_yards")]).alias(
                "passing_yards"
            ),
            pl.coalesce([pl.col("official_rushing_yards"), pl.col("rushing_yards")]).alias(
                "rushing_yards"
            ),
            pl.coalesce([pl.col("official_total_yards"), pl.col("total_yards")]).alias(
                "total_yards"
            ),
            pl.coalesce([pl.col("official_passing_tds"), pl.col("passing_tds")])
            .cast(pl.Int64)
            .alias("passing_tds"),
            pl.coalesce([pl.col("official_rushing_tds"), pl.col("rushing_tds")])
            .cast(pl.Int64)
            .alias("rushing_tds"),
            pl.coalesce([pl.col("official_passing_first_downs"), pl.col("passing_first_downs")])
            .cast(pl.Int64)
            .alias("passing_first_downs"),
            pl.coalesce([pl.col("official_rushing_first_downs"), pl.col("rushing_first_downs")])
            .cast(pl.Int64)
            .alias("rushing_first_downs"),
            pl.coalesce([pl.col("official_passing_epa"), pl.col("passing_epa")]).alias(
                "passing_epa"
            ),
            pl.coalesce([pl.col("official_rushing_epa"), pl.col("rushing_epa")]).alias(
                "rushing_epa"
            ),
            pl.coalesce([pl.col("official_passing_cpoe"), pl.col("passing_cpoe")]).alias(
                "passing_cpoe"
            ),
            pl.coalesce([pl.col("official_sacks_suffered"), pl.col("sacks_suffered")])
            .cast(pl.Int64)
            .alias("sacks_suffered"),
            pl.coalesce([pl.col("official_passing_interceptions"), pl.col("passing_interceptions")])
            .cast(pl.Int64)
            .alias("passing_interceptions"),
            pl.coalesce([pl.col("official_sack_fumbles_lost"), pl.col("sack_fumbles_lost")])
            .cast(pl.Int64)
            .alias("sack_fumbles_lost"),
            pl.coalesce([pl.col("official_rushing_fumbles_lost"), pl.col("rushing_fumbles_lost")])
            .cast(pl.Int64)
            .alias("rushing_fumbles_lost"),
            pl.coalesce(
                [pl.col("official_passing_yards_allowed"), pl.col("passing_yards_allowed")]
            ).alias("passing_yards_allowed"),
            pl.coalesce(
                [pl.col("official_rushing_yards_allowed"), pl.col("rushing_yards_allowed")]
            ).alias("rushing_yards_allowed"),
            pl.coalesce(
                [pl.col("official_total_yards_allowed"), pl.col("total_yards_allowed")]
            ).alias("total_yards_allowed"),
            pl.coalesce([pl.col("official_passing_tds_allowed"), pl.col("passing_tds_allowed")])
            .cast(pl.Int64)
            .alias("passing_tds_allowed"),
            pl.coalesce([pl.col("official_rushing_tds_allowed"), pl.col("rushing_tds_allowed")])
            .cast(pl.Int64)
            .alias("rushing_tds_allowed"),
            pl.coalesce(
                [
                    pl.col("official_passing_first_downs_allowed"),
                    pl.col("passing_first_downs_allowed"),
                ]
            )
            .cast(pl.Int64)
            .alias("passing_first_downs_allowed"),
            pl.coalesce(
                [
                    pl.col("official_rushing_first_downs_allowed"),
                    pl.col("rushing_first_downs_allowed"),
                ]
            )
            .cast(pl.Int64)
            .alias("rushing_first_downs_allowed"),
            pl.coalesce(
                [pl.col("official_passing_epa_allowed"), pl.col("passing_epa_allowed")]
            ).alias("passing_epa_allowed"),
            pl.coalesce(
                [pl.col("official_rushing_epa_allowed"), pl.col("rushing_epa_allowed")]
            ).alias("rushing_epa_allowed"),
            pl.coalesce(
                [pl.col("official_passing_cpoe_allowed"), pl.col("passing_cpoe_allowed")]
            ).alias("passing_cpoe_allowed"),
        )
        .with_columns(
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("total_yards") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("total_yards_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("passing_yards") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("passing_yards_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("rushing_yards") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("rushing_yards_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("passing_epa") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("passing_epa_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("rushing_epa") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("rushing_epa_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("passing_tds") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("passing_tds_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("rushing_tds") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("rushing_tds_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("sacks_suffered") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("sacks_suffered_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("passing_interceptions") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("passing_interceptions_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("sack_fumbles_lost") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("sack_fumbles_lost_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("rushing_fumbles_lost") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("rushing_fumbles_lost_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("passing_first_downs") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("passing_first_downs_per_offensive_snap"),
            pl.when(pl.col("offensive_snaps") > 0)
            .then(pl.col("rushing_first_downs") / pl.col("offensive_snaps"))
            .otherwise(None)
            .alias("rushing_first_downs_per_offensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("total_yards_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("total_yards_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("passing_yards_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("passing_yards_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("rushing_yards_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("rushing_yards_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("passing_epa_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("passing_epa_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("rushing_epa_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("rushing_epa_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("passing_tds_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("passing_tds_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("rushing_tds_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("rushing_tds_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("passing_first_downs_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("passing_first_downs_allowed_per_defensive_snap"),
            pl.when(pl.col("defensive_snaps") > 0)
            .then(pl.col("rushing_first_downs_allowed") / pl.col("defensive_snaps"))
            .otherwise(None)
            .alias("rushing_first_downs_allowed_per_defensive_snap"),
        )
        .drop(
            [
                "official_passing_yards",
                "official_rushing_yards",
                "official_total_yards",
                "official_passing_tds",
                "official_rushing_tds",
                "official_passing_first_downs",
                "official_rushing_first_downs",
                "official_passing_epa",
                "official_rushing_epa",
                "official_passing_cpoe",
                "official_sacks_suffered",
                "official_passing_interceptions",
                "official_sack_fumbles_lost",
                "official_rushing_fumbles_lost",
                "official_passing_yards_allowed",
                "official_rushing_yards_allowed",
                "official_total_yards_allowed",
                "official_passing_tds_allowed",
                "official_rushing_tds_allowed",
                "official_passing_first_downs_allowed",
                "official_rushing_first_downs_allowed",
                "official_passing_epa_allowed",
                "official_rushing_epa_allowed",
                "official_passing_cpoe_allowed",
            ]
        )
    )

    return result


def load_weekly_team_stats(season: int) -> pl.DataFrame:
    """Load PBP-derived game-by-game team stats for a regular season."""
    pbp_df = load_pbp_data(season)
    player_stats_df = load_weekly_player_stats(season)
    schedule_df = load_schedule(season)
    weekly_team_stats_df = compute_team_game_stats_from_pbp(pbp_df, player_stats_df, schedule_df)
    official_team_stats_df = load_official_weekly_team_stats(season)
    return _override_team_game_stats_with_official_weekly(
        weekly_team_stats_df,
        official_team_stats_df,
    )


def load_schedule(season: int) -> pl.DataFrame:
    """Load the regular season schedule for a given season."""
    df = nfl.load_schedules(seasons=season)
    df = df.filter(pl.col("game_type") == "REG")
    return _normalize_team_abbreviations(df, ["home_team", "away_team"])


def load_qb_stats(season: int) -> pl.DataFrame:
    """Load PBP-derived quarterback game stats with snap-count support."""
    pbp_df = load_pbp_data(season)
    snap_counts_df = load_snap_counts_data(season)
    qb_identity_df = load_qb_identity_crosswalk(season)
    qb_df = compute_qb_game_stats_from_pbp(pbp_df, snap_counts_df, qb_identity_df)
    official_qb_stats_df = _load_official_weekly_qb_stats(
        load_weekly_player_stats(season), qb_identity_df
    )
    qb_df = _override_qb_game_stats_with_official_weekly(qb_df, official_qb_stats_df)

    attempts = pl.col("qb_attempts").cast(pl.Float64)
    completions = pl.col("qb_completions").cast(pl.Float64)
    passing_yards = pl.col("qb_pass_yards").cast(pl.Float64)
    touchdowns = pl.col("qb_pass_touchdowns").cast(pl.Float64)
    interceptions = pl.col("qb_interceptions").cast(pl.Float64)

    passer_rating = (
        pl.when(attempts > 0)
        .then(
            (
                (
                    (((completions / attempts) - 0.3) * 5).clip(0.0, 2.375)
                    + (((passing_yards / attempts) - 3.0) * 0.25).clip(0.0, 2.375)
                    + (((touchdowns / attempts) * 20.0).clip(0.0, 2.375))
                    + ((2.375 - ((interceptions / attempts) * 25.0)).clip(0.0, 2.375))
                )
                / 6.0
                * 100.0
            ).round(1)
        )
        .otherwise(None)
    )

    return qb_df.with_columns(
        pl.coalesce([pl.col("qb_id"), pl.col("snap_player_id"), pl.col("qb_name")]).alias("qb_id"),
        passer_rating.alias("qb_passer_rating"),
    ).select(
        [
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
        ]
        + [
            column
            for column in (
                "qb_completion_pct",
                "qb_carries",
                "qb_rushing_yards",
                "qb_rushing_tds",
                "qb_rushing_first_downs",
                "qb_rushing_epa",
                "qb_rushing_fumbles",
                "qb_rushing_fumbles_lost",
                "qb_rushing_2pt_conversions",
                "qb_yards_per_carry",
                "qb_epa_per_carry",
            )
            if column in qb_df.columns
        ]
    )
