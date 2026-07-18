"""Partial-season rating snapshot helpers."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from nfl_sos_ratings.pbp_expressions import scrimmage_snap_expr, value_expr
from nfl_sos_ratings.ratings import compute_ratings
from nfl_sos_ratings.simultaneous_adjustment import compute_team_adjusted_stats, solve_srs

_DEFAULT_TEAM_RESPONSE_COLS: tuple[str, str] = (
    "passing_epa_per_offensive_snap",
    "rushing_epa_per_offensive_snap",
)

_DEFAULT_TEAM_ADJUSTED_COLS: tuple[str, str, str, str] = (
    "adj_off_passing_epa_per_offensive_snap",
    "adj_off_rushing_epa_per_offensive_snap",
    "adj_def_passing_epa_per_offensive_snap",
    "adj_def_rushing_epa_per_offensive_snap",
)


def _empty_team_snapshot(teams: list[str]) -> pl.DataFrame:
    """Return a zeroed team snapshot when no games qualify before the cutoff."""
    return pl.DataFrame(
        {
            "team": teams,
            "SaOR": [0.0] * len(teams),
            "SaDR": [0.0] * len(teams),
            "SaOvR": [0.0] * len(teams),
            "SaCR": [0.0] * len(teams),
        }
    ).sort("team")


def _empty_team_adjusted_snapshot(teams: list[str]) -> pl.DataFrame:
    """Return a zeroed adjusted-component snapshot when no games qualify."""
    return pl.DataFrame(
        {
            "team": teams,
            **{column: [0.0] * len(teams) for column in _DEFAULT_TEAM_ADJUSTED_COLS},
        }
    ).sort("team")


def _empty_special_teams_game_frame() -> pl.DataFrame:
    """Return the standard empty special-teams game-frame schema."""
    return pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team": pl.String,
            "opponent_team": pl.String,
            "st_epa_margin_per_play": pl.Float64,
        }
    )


def _empty_special_teams_snapshot(teams: list[str]) -> pl.DataFrame:
    """Return a zeroed special-teams snapshot for the requested teams."""
    return pl.DataFrame({"team": teams, "st_rating": [0.0] * len(teams)}).sort("team")


def _empty_play_level_team_frame() -> pl.DataFrame:
    """Return the standard empty play-level team-frame schema."""
    return pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team": pl.String,
            "opponent_team": pl.String,
            "is_home": pl.Boolean,
            "passing_epa_per_offensive_snap": pl.Float64,
            "rushing_epa_per_offensive_snap": pl.Float64,
        }
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    """Return a sample-standardized array, or centered zeros when spread is absent."""
    if len(values) == 0:
        return values
    centered = values - float(values.mean())
    if len(values) == 1:
        return centered
    std = float(values.std(ddof=1))
    return centered / std if std > 0.0 else centered


def build_special_teams_game_frame_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Convert raw special-play PBP into one net special-teams margin row per team-game."""
    special_flag = "special" if "special" in pbp_df.columns else "special_teams_play"
    required_columns = {"game_id", "week", "posteam", "defteam", special_flag, "epa"}
    missing = sorted(required_columns - set(pbp_df.columns))
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"pbp_df is missing required special-teams columns: {detail}")

    special_plays = pbp_df.filter(pl.col(special_flag).cast(pl.Int64) == 1).drop_nulls(
        ["posteam", "defteam"]
    )
    if special_plays.is_empty():
        return _empty_special_teams_game_frame()

    offense_perspective = special_plays.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent_team"),
        pl.col("epa").cast(pl.Float64).alias("st_epa_for"),
        pl.lit(0.0).alias("st_epa_against"),
        pl.lit(1.0).alias("special_play_count"),
    )
    defense_perspective = special_plays.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("defteam").cast(pl.String).alias("team"),
        pl.col("posteam").cast(pl.String).alias("opponent_team"),
        pl.lit(0.0).alias("st_epa_for"),
        pl.col("epa").cast(pl.Float64).alias("st_epa_against"),
        pl.lit(1.0).alias("special_play_count"),
    )

    return (
        pl.concat([offense_perspective, defense_perspective], how="vertical")
        .group_by(["game_id", "week", "team", "opponent_team"])
        .agg(
            pl.col("st_epa_for").sum().alias("st_epa_for"),
            pl.col("st_epa_against").sum().alias("st_epa_against"),
            pl.col("special_play_count").sum().alias("special_play_count"),
        )
        .with_columns(
            (
                (pl.col("st_epa_for") - pl.col("st_epa_against")) / pl.col("special_play_count")
            ).alias("st_epa_margin_per_play")
        )
        .select(["game_id", "week", "team", "opponent_team", "st_epa_margin_per_play"])
        .sort(["week", "game_id", "team"])
    )


def build_play_level_team_frame_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Convert scrimmage-snap PBP into one play-level row per team offensive snap."""
    required_columns = {"game_id", "week", "posteam", "defteam", "epa"}
    missing = sorted(required_columns - set(pbp_df.columns))
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"pbp_df is missing required play-level team columns: {detail}")

    if "posteam_type" in pbp_df.columns:
        is_home_expr = pl.col("posteam_type").cast(pl.String) == "home"
    elif "home_team" in pbp_df.columns:
        is_home_expr = pl.col("posteam").cast(pl.String) == pl.col("home_team").cast(pl.String)
    else:
        raise ValueError("pbp_df must include either posteam_type or home_team for is_home")

    scrimmage_rows = pbp_df.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & scrimmage_snap_expr(pbp_df.columns)
    )
    if scrimmage_rows.is_empty():
        return _empty_play_level_team_frame()

    return scrimmage_rows.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent_team"),
        is_home_expr.alias("is_home"),
        pl.when(value_expr(pbp_df.columns, "qb_dropback") > 0)
        .then(value_expr(pbp_df.columns, "epa", 0.0).cast(pl.Float64))
        .otherwise(0.0)
        .alias("passing_epa_per_offensive_snap"),
        pl.when(value_expr(pbp_df.columns, "rush") > 0)
        .then(value_expr(pbp_df.columns, "epa", 0.0).cast(pl.Float64))
        .otherwise(0.0)
        .alias("rushing_epa_per_offensive_snap"),
    ).sort(["week", "game_id", "team"])


def build_special_teams_rating_snapshot(
    st_game_rows: pl.DataFrame,
    cutoff_week: int,
) -> pl.DataFrame:
    """Build a pre-cutoff special-teams rating snapshot from team-game ST margins."""
    if "week" not in st_game_rows.columns:
        raise ValueError("st_game_rows must include a week column")

    filtered_rows = st_game_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        teams = sorted(
            st_game_rows.select("team").drop_nulls().to_series().cast(pl.String).unique().to_list()
        )
        return _empty_special_teams_snapshot(teams)

    return solve_srs(filtered_rows, response_col="st_epa_margin_per_play").rename(
        {"srs_rating": "st_rating"}
    )


def build_team_adjusted_snapshot(
    weekly_team_rows: pl.DataFrame,
    cutoff_week: int,
    response_cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build the pre-cutoff adjusted team component frame used by the team ridge backbone."""
    if "week" not in weekly_team_rows.columns:
        raise ValueError("weekly_team_rows must include a week column")

    selected_response_cols = list(response_cols or _DEFAULT_TEAM_RESPONSE_COLS)
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        teams = sorted(
            weekly_team_rows.select("team")
            .drop_nulls()
            .to_series()
            .cast(pl.String)
            .unique()
            .to_list()
        )
        return _empty_team_adjusted_snapshot(teams)

    adjusted = compute_team_adjusted_stats(
        filtered_rows,
        response_cols=selected_response_cols,
    )
    for column in _DEFAULT_TEAM_ADJUSTED_COLS:
        if column not in adjusted.columns:
            adjusted = adjusted.with_columns(pl.lit(0.0).alias(column))
    return adjusted.select(["team", *_DEFAULT_TEAM_ADJUSTED_COLS]).sort("team")


def build_play_level_team_adjusted_snapshot(
    play_rows: pl.DataFrame,
    cutoff_week: int,
    response_cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build a pre-cutoff adjusted team component frame from play-level offensive snaps."""
    if "week" not in play_rows.columns:
        raise ValueError("play_rows must include a week column")

    selected_response_cols = list(response_cols or _DEFAULT_TEAM_RESPONSE_COLS)
    filtered_rows = play_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        teams = sorted(
            play_rows.select("team").drop_nulls().to_series().cast(pl.String).unique().to_list()
        )
        return _empty_team_adjusted_snapshot(teams)

    adjusted = compute_team_adjusted_stats(
        filtered_rows,
        response_cols=selected_response_cols,
    )
    for column in _DEFAULT_TEAM_ADJUSTED_COLS:
        if column not in adjusted.columns:
            adjusted = adjusted.with_columns(pl.lit(0.0).alias(column))
    return adjusted.select(["team", *_DEFAULT_TEAM_ADJUSTED_COLS]).sort("team")


def build_team_weighted_rating_snapshot(
    weekly_team_rows: pl.DataFrame,
    cutoff_week: int,
    weight_map: dict[str, float],
    response_cols: Sequence[str] | None = None,
    output_col: str = "T1SaOvR",
) -> pl.DataFrame:
    """Build a pre-cutoff team snapshot from weighted ridge-adjusted components."""
    adjusted = build_team_adjusted_snapshot(
        weekly_team_rows,
        cutoff_week=cutoff_week,
        response_cols=response_cols,
    )
    if adjusted.is_empty():
        return pl.DataFrame(schema={"team": pl.String, output_col: pl.Float64})

    weighted_values = np.zeros(adjusted.height, dtype=np.float64)
    for column in _DEFAULT_TEAM_ADJUSTED_COLS:
        if column not in adjusted.columns:
            continue
        component_values = np.asarray(
            adjusted.select(column).to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        weighted_values += _zscore(component_values) * float(weight_map.get(column, 0.0))

    return pl.DataFrame(
        {
            "team": adjusted.select("team").to_series().cast(pl.String).to_list(),
            output_col: np.round(_zscore(weighted_values), 6).tolist(),
        }
    ).sort("team")


def build_team_rating_snapshot(
    weekly_team_rows: pl.DataFrame,
    cutoff_week: int,
    response_cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build a team rating snapshot using only games before ``cutoff_week``.

    Args:
        weekly_team_rows: Team-game rows with ``team``, ``opponent_team``, ``week``,
            ``is_home``, and the selected response columns.
        cutoff_week: Snapshot cutoff. Only rows with ``week < cutoff_week`` are used.
        response_cols: Optional override for the ridge response columns.

    Returns:
        A ``compute_ratings``-compatible team ratings table for the pre-cutoff games.

    """
    if "week" not in weekly_team_rows.columns:
        raise ValueError("weekly_team_rows must include a week column")

    adjusted = build_team_adjusted_snapshot(
        weekly_team_rows,
        cutoff_week=cutoff_week,
        response_cols=response_cols,
    )
    if adjusted.is_empty():
        teams = sorted(
            weekly_team_rows.select("team")
            .drop_nulls()
            .to_series()
            .cast(pl.String)
            .unique()
            .to_list()
        )
        return _empty_team_snapshot(teams)
    return compute_ratings(adjusted)


__all__ = [
    "build_play_level_team_adjusted_snapshot",
    "build_play_level_team_frame_from_pbp",
    "build_special_teams_game_frame_from_pbp",
    "build_special_teams_rating_snapshot",
    "build_team_adjusted_snapshot",
    "build_team_rating_snapshot",
    "build_team_weighted_rating_snapshot",
]
