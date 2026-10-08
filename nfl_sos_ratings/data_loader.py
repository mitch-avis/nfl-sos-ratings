"""Data loading functions wrapping nflreadpy plus direct nflverse release assets."""

import io
import os
import urllib.request
from typing import TYPE_CHECKING, Literal

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import CacheMode, update_config

from nfl_sos_ratings.config import TEAM_ABBR_ALIASES
from nfl_sos_ratings.qb_stats import (
    CPOE_PARTS,
    QB_PASSING_TOTALS,
    compute_qb_game_stats_from_pbp,
    qb_passer_rating_expr,
)
from nfl_sos_ratings.team_stats import add_per_snap_rates, compute_team_game_stats_from_pbp
from nfl_sos_ratings.wp_bins import compute_qb_wp_bins, compute_team_wp_bins

if TYPE_CHECKING:
    from collections.abc import Mapping

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

# nflverse play-by-play starts in 1999, so that season has no previous season in the data.
PBP_START_SEASON = 1999
# nflverse publishes snap counts from 2012, but its 2012 file has no rows, so QB snaps start in
# 2013; a season without snap counts leaves them null (``compute_qb_game_volumes_from_pbp``).
SNAP_COUNTS_START_SEASON = 2012
_CACHE_MODE_VARIABLE = "NFLREADPY_CACHE"
# Seconds to wait on an nflverse release download before failing instead of hanging.
_RELEASE_DOWNLOAD_TIMEOUT_SECONDS = 60
ROSTERS_WEEKLY_START_SEASON = 2002
# Play-by-play columns that hold a team code. nflverse writes the Rams as LA in every one of them,
# not only in posteam and defteam, so the loaders normalize them all before any code compares one
# team column with another.
_PBP_TEAM_COLUMNS = (
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
# Play-by-play yard lines, written "<team> <yards>" (LA 25), or 50 at midfield.
_PBP_YARD_LINE_COLUMNS = ("yrdln", "drive_start_yard_line", "drive_end_yard_line", "end_yard_line")
# The first season nflverse charts pass depth, air yards, and yards after catch, with their EPA
# splits and expected yards after catch: before it they are null, apart from 1999's yards after
# catch on 337 of 9,522 completions and pass depth on 344 of 16,651 passes.
PASS_DEPTH_START_SEASON = 2006
_BEFORE_PASS_DEPTH = range(PBP_START_SEASON, PASS_DEPTH_START_SEASON)
# Seasons in which nflverse has no value for a play-by-play field, so every stat built on it is
# unknown there: the loaders write nulls for them, including where nflverse writes 0. Play-by-play
# records a QB hit on no play in 2003-2005 (before 2003 only on sacks) and has no drive penalty
# yards before 2001.
_PBP_FIELD_GAPS: dict[str, range] = {
    "air_yards": _BEFORE_PASS_DEPTH,
    "air_epa": _BEFORE_PASS_DEPTH,
    "pass_length": _BEFORE_PASS_DEPTH,
    "yards_after_catch": _BEFORE_PASS_DEPTH,
    "yac_epa": _BEFORE_PASS_DEPTH,
    "xyac_mean_yardage": _BEFORE_PASS_DEPTH,
    "qb_hit": range(2003, 2006),
    "drive_yards_penalized": range(PBP_START_SEASON, 2001),
}
# The same for weekly player stats, which credit every player with 0 tackles for loss in 2003-2011
# (one in all of 2006) and 0 QB hits in 2003-2005.
_PLAYER_STAT_GAPS: dict[str, range] = {
    "def_tackles_for_loss": range(2003, 2012),
    "def_qb_hits": range(2003, 2006),
}


def use_disk_cache_unless_configured(environ: Mapping[str, str] = os.environ) -> None:
    """Cache nflverse downloads on disk unless ``NFLREADPY_CACHE`` already picks a mode.

    nflreadpy defaults to an in-memory cache, so every pipeline run downloads every season again.
    The on-disk cache (nflreadpy's default location and one-day lifetime) lets a rerun the same
    day reuse those files. Setting ``NFLREADPY_CACHE`` (``memory``, ``filesystem``, or ``off``)
    overrides this default.
    """
    if _CACHE_MODE_VARIABLE not in environ:
        update_config(cache_mode=CacheMode.FILESYSTEM)


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


_OFFICIAL_QB_RUSHING_FIELDS: dict[str, tuple[str, type[pl.Int64 | pl.Float64]]] = {
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
        # nflverse stores the yards lost on sacks as a negative number; the magnitude keeps the
        # column in yards lost, as the play-by-play fallback counts them and ANY/A subtracts them.
        (
            pl.col("sack_yards_lost").cast(pl.Float64).abs()
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
    """Normalize known source-specific team abbreviations in selected columns.

    An empty abbreviation becomes null: old play-by-play leaves ``posteam`` empty, not null, on
    non-plays such as timeouts, and code that drops null teams would otherwise treat ``""`` as a
    team (a phantom opponent that once doubled late-game flags and QB game rows).
    """
    exprs = [
        pl.when(pl.col(column) == "")
        .then(None)
        .otherwise(pl.col(column).replace(TEAM_ABBR_ALIASES))
        .alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _normalize_yard_line_teams(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize the team code that opens yard-line text such as ``LA 25``.

    Code that reads a yard line compares its team with ``posteam`` to tell the offense's own half
    from the opponent's, so the code has to match the normalized ``posteam``. Whatever follows the
    code is kept as written (one 2002 end yard line reads ``LA -10``); midfield (``50`` or
    ``MID 50``), other codes, and nulls pass through unchanged.
    """
    exprs: list[pl.Expr] = []
    for column in columns:
        if column not in df.columns:
            continue
        parts = pl.col(column).str.extract_groups(r"^(?<team>[A-Z]+) (?<rest>.+)$")
        normalized = pl.concat_str(
            [parts.struct.field("team").replace(TEAM_ABBR_ALIASES), parts.struct.field("rest")],
            separator=" ",
        )
        exprs.append(pl.coalesce(normalized, pl.col(column)).alias(column))
    return df.with_columns(exprs) if exprs else df


def _normalize_pbp_teams(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize every team code in play-by-play: the team columns and the yard-line text."""
    df = _normalize_team_abbreviations(df, list(_PBP_TEAM_COLUMNS))
    return _normalize_yard_line_teams(df, list(_PBP_YARD_LINE_COLUMNS))


def _blank_fields_missing_in_season(
    df: pl.DataFrame, season: int, gaps: Mapping[str, range]
) -> pl.DataFrame:
    """Return ``df`` with every field the season's source lacks (per ``gaps``) set to null."""
    exprs = [
        pl.lit(None, dtype=df.schema[column]).alias(column)
        for column, seasons in gaps.items()
        if column in df.columns and season in seasons
    ]
    return df.with_columns(exprs) if exprs else df


def _filter_regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """Filter a frame to regular-season rows when a season-type column is present."""
    for column in ("season_type", "game_type"):
        if column in df.columns:
            return df.filter(pl.col(column) == "REG")
    return df


def _filter_postseason(df: pl.DataFrame) -> pl.DataFrame:
    """Filter a frame to postseason rows; a frame without a season-type column has none."""
    if "season_type" in df.columns:
        return df.filter(pl.col("season_type") == "POST")
    if "game_type" in df.columns:
        return df.filter(pl.col("game_type") != "REG")
    return df.clear()


def _fetch_release_parquet(url: str) -> pl.DataFrame:
    """Download one nflverse release Parquet asset into a dataframe."""
    with urllib.request.urlopen(url, timeout=_RELEASE_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed https URLs above
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
        msg = f"Unknown QBR level {level!r}; expected one of: {valid_levels}"
        raise ValueError(msg)

    qbr_df = _fetch_release_parquet(ESPN_QBR_RELEASE_URLS[level])
    if "season_type" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("season_type") == "Regular")
    if seasons is not None and "season" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("season").is_in(seasons))

    team_columns = [column for column in ("team_abb", "opp_abb") if column in qbr_df.columns]
    return _normalize_team_abbreviations(qbr_df, team_columns)


def load_pbp_data(season: int) -> pl.DataFrame:
    """Load regular-season play-by-play data with normalized team abbreviations.

    Fields nflverse lacks in the season are null (``_PBP_FIELD_GAPS``).
    """
    df = nfl.load_pbp(seasons=season)
    df = _filter_regular_season(df)
    df = _blank_fields_missing_in_season(df, season, _PBP_FIELD_GAPS)
    return _normalize_pbp_teams(df)


def load_wp_bins(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one regular season's team and QB win-probability bins (see ``wp_bins``).

    The play-by-play is loaded once for both. A season whose play-by-play lacks ``wp`` gives
    typed empty frames.
    """
    pbp_df = load_pbp_data(season)
    return compute_team_wp_bins(pbp_df), compute_qb_wp_bins(pbp_df)


def load_playoff_pbp_data(season: int) -> pl.DataFrame:
    """Load postseason play-by-play data for validation-only analyses.

    This path exists for playoff validation work only. Published rating computation remains
    regular-season-only and must continue to use :func:`load_pbp_data`.
    """
    df = nfl.load_pbp(seasons=season)
    for column in ("season_type", "game_type"):
        if column in df.columns:
            df = df.filter(pl.col(column) == "POST")
            break
    else:
        return pl.DataFrame(schema=df.schema)
    return _normalize_pbp_teams(df)


def load_weekly_player_stats(season: int) -> pl.DataFrame:
    """Load regular-season weekly player stats with normalized team abbreviations.

    Stats nflverse credits to no one in the season are null (``_PLAYER_STAT_GAPS``).
    """
    df = nfl.load_player_stats(seasons=season, summary_level="week")
    df = _filter_regular_season(df)
    df = _blank_fields_missing_in_season(df, season, _PLAYER_STAT_GAPS)
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

    overridden = (
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
    # The per-snap rates divide the official totals now, with their parts beside them.
    return add_per_snap_rates(overridden)


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
    return _build_qb_stats(
        load_pbp_data(season),
        load_snap_counts_data(season),
        load_qb_identity_crosswalk(season),
        load_weekly_player_stats(season),
    )


def load_playoff_qb_stats(season: int) -> pl.DataFrame:
    """Load postseason quarterback game stats, built exactly as :func:`load_qb_stats` builds them.

    This path exists for validation checks only; published ratings never use postseason games.
    """
    snap_counts_df = (
        _empty_snap_counts_data()
        if _season_is_before_source_floor(season, start_season=SNAP_COUNTS_START_SEASON)
        else _normalize_team_abbreviations(
            _filter_postseason(nfl.load_snap_counts(seasons=season)), ["team"]
        )
    )
    weekly_player_stats_df = _normalize_team_abbreviations(
        _filter_postseason(nfl.load_player_stats(seasons=season, summary_level="week")),
        ["team", "opponent_team"],
    )
    return _build_qb_stats(
        load_playoff_pbp_data(season),
        snap_counts_df,
        load_qb_identity_crosswalk(season),
        weekly_player_stats_df,
    )


def load_playoff_schedule(season: int) -> pl.DataFrame:
    """Load the postseason schedule (``game_type`` WC, DIV, CON, or SB) for validation checks."""
    df = nfl.load_schedules(seasons=season)
    df = df.filter(pl.col("game_type") != "REG")
    return _normalize_team_abbreviations(df, ["home_team", "away_team"])


def _build_qb_stats(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame,
    qb_identity_df: pl.DataFrame,
    weekly_player_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build QB game stats from play-by-play, overriding passing fields with official stats."""
    qb_df = compute_qb_game_stats_from_pbp(pbp_df, snap_counts_df, qb_identity_df)
    official_qb_stats_df = _load_official_weekly_qb_stats(weekly_player_stats_df, qb_identity_df)
    qb_df = _override_qb_game_stats_with_official_weekly(qb_df, official_qb_stats_df)

    passer_rating = qb_passer_rating_expr(
        *(pl.col(column).cast(pl.Float64) for column in QB_PASSING_TOTALS)
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
            *CPOE_PARTS,
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
            )
            if column in qb_df.columns
        ]
    )
