"""Quarterback season-level aggregation helpers."""

import polars as pl

type PolarsCastType = type[pl.Int64] | type[pl.Float64]

_QB_TOTAL_COLUMNS: dict[str, tuple[str, PolarsCastType]] = {
    "qb_attempts": ("qb_attempts_total", pl.Int64),
    "qb_completions": ("qb_completions_total", pl.Int64),
    "qb_dropbacks": ("qb_dropbacks_total", pl.Int64),
    "qb_offense_snaps": ("qb_offense_snaps_total", pl.Int64),
    "qb_fourth_quarter_comeback": ("qb_fourth_quarter_comebacks", pl.Int64),
    "qb_game_winning_drive": ("qb_game_winning_drives", pl.Int64),
    "qb_pass_yards": ("qb_pass_yards_total", pl.Float64),
    "qb_pass_touchdowns": ("qb_pass_touchdowns_total", pl.Float64),
    "qb_interceptions": ("qb_interceptions_total", pl.Float64),
    "qb_sacks": ("qb_sacks_total", pl.Float64),
    "qb_sack_yards_lost": ("qb_sack_yards_lost_total", pl.Float64),
    "qb_sack_fumbles_lost": ("qb_sack_fumbles_lost_total", pl.Float64),
    "qb_passing_epa": ("qb_passing_epa_total", pl.Float64),
    "qb_carries": ("qb_carries_total", pl.Int64),
    "qb_rushing_yards": ("qb_rushing_yards_total", pl.Float64),
    "qb_rushing_tds": ("qb_rushing_tds_total", pl.Int64),
    "qb_rushing_first_downs": ("qb_rushing_first_downs_total", pl.Int64),
    "qb_rushing_epa": ("qb_rushing_epa_total", pl.Float64),
    "qb_rushing_fumbles": ("qb_rushing_fumbles_total", pl.Int64),
    "qb_rushing_fumbles_lost": ("qb_rushing_fumbles_lost_total", pl.Int64),
    "qb_rushing_2pt_conversions": ("qb_rushing_2pt_conversions_total", pl.Int64),
    "qb_designed_carries": ("qb_designed_carries_total", pl.Int64),
    "qb_designed_rush_yards": ("qb_designed_rush_yards_total", pl.Float64),
    "qb_designed_rush_epa": ("qb_designed_rush_epa_total", pl.Float64),
    "qb_scrambles": ("qb_scrambles_total", pl.Int64),
    "qb_scramble_yards": ("qb_scramble_yards_total", pl.Float64),
    "qb_kneels": ("qb_kneels_total", pl.Int64),
}

_QB_PER_GAME_COLUMNS: dict[str, str] = {
    "qb_attempts": "qb_attempts_per_game",
    "qb_completions": "qb_completions_per_game",
    "qb_dropbacks": "qb_dropbacks_per_game",
    "qb_offense_snaps": "qb_offense_snaps_per_game",
    "qb_fourth_quarter_comeback": "qb_fourth_quarter_comebacks_per_game",
    "qb_game_winning_drive": "qb_game_winning_drives_per_game",
    "qb_pass_yards": "qb_pass_yards_per_game",
    "qb_pass_touchdowns": "qb_pass_touchdowns_per_game",
    "qb_interceptions": "qb_interceptions_per_game",
    "qb_sacks": "qb_sacks_per_game",
    "qb_sack_yards_lost": "qb_sack_yards_lost_per_game",
    "qb_sack_fumbles_lost": "qb_sack_fumbles_lost_per_game",
    "qb_passing_epa": "qb_passing_epa_per_game",
    "qb_carries": "qb_carries_per_game",
    "qb_rushing_yards": "qb_rushing_yards_per_game",
    "qb_rushing_tds": "qb_rushing_tds_per_game",
    "qb_rushing_first_downs": "qb_rushing_first_downs_per_game",
    "qb_rushing_epa": "qb_rushing_epa_per_game",
    "qb_rushing_fumbles": "qb_rushing_fumbles_per_game",
    "qb_rushing_fumbles_lost": "qb_rushing_fumbles_lost_per_game",
    "qb_rushing_2pt_conversions": "qb_rushing_2pt_conversions_per_game",
    "qb_designed_carries": "qb_designed_carries_per_game",
    "qb_designed_rush_yards": "qb_designed_rush_yards_per_game",
    "qb_designed_rush_epa": "qb_designed_rush_epa_per_game",
    "qb_scrambles": "qb_scrambles_per_game",
    "qb_scramble_yards": "qb_scramble_yards_per_game",
    "qb_kneels": "qb_kneels_per_game",
}


def _resolve_qb_keys(qb_df: pl.DataFrame) -> list[str]:
    """Return the available QB identifier keys for grouping."""
    keys: list[str] = []
    if "qb_id" in qb_df.columns:
        keys.append("qb_id")
    if "qb_name" in qb_df.columns:
        keys.append("qb_name")
    if not keys:
        keys.append("team_abbr")
    return keys


def _select_primary_qb_rows(qb_df: pl.DataFrame) -> pl.DataFrame:
    """Return one primary QB row per team-week using snaps, then dropbacks, then attempts."""
    if not {"team_abbr", "week"}.issubset(set(qb_df.columns)):
        return qb_df

    sort_keys = [
        column
        for column in ("qb_offense_snaps", "qb_dropbacks", "qb_attempts")
        if column in qb_df.columns
    ]
    if not sort_keys:
        return qb_df

    return (
        qb_df.sort(sort_keys, descending=[True] * len(sort_keys))
        .group_by(["team_abbr", "week"])
        .first()
    )


def _compute_team_late_game_flags_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return team-game 4QC and GWD flags from late-game PBP score states.

    Fourth-quarter comeback = eventual winner with a quarter-4-or-later
    offensive snap where `score_differential < 0`.
    Game-winning drive = eventual winner with a quarter-4-or-later scoring play
    where `score_differential <= 0` before the play and
    `score_differential_post > 0` after the play.
    """
    required_cols = {
        "game_id",
        "posteam",
        "qtr",
        "score_differential",
        "score_differential_post",
        "posteam_score",
        "posteam_score_post",
    }
    if pbp_df.is_empty() or not required_cols.issubset(set(pbp_df.columns)):
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "team_abbr": pl.String,
                "qb_fourth_quarter_comeback": pl.Int64,
                "qb_game_winning_drive": pl.Int64,
            }
        )

    offense = (
        pbp_df.filter(pl.col("posteam").is_not_null())
        .with_row_index("play_order")
        .sort("play_order")
    )

    team_scores = offense.group_by(["game_id", "posteam"]).agg(
        pl.col("posteam_score_post").drop_nulls().max().alias("final_points_for")
    )
    final_state = (
        team_scores.join(
            team_scores.rename(
                {
                    "posteam": "opponent_team_abbr",
                    "final_points_for": "final_points_allowed",
                }
            ),
            on="game_id",
            how="inner",
        )
        .filter(pl.col("posteam") != pl.col("opponent_team_abbr"))
        .rename({"posteam": "team_abbr"})
        .select(["game_id", "team_abbr", "final_points_for", "final_points_allowed"])
    )
    trailing_late = (
        offense.filter((pl.col("qtr") >= 4) & (pl.col("score_differential") < 0))
        .group_by(["game_id", "posteam"])
        .agg(pl.lit(1).alias("had_fourth_quarter_deficit"))
        .rename({"posteam": "team_abbr"})
    )
    lead_taking = (
        offense.filter(
            (pl.col("qtr") >= 4)
            & (pl.col("posteam_score_post") > pl.col("posteam_score"))
            & (pl.col("score_differential") <= 0)
            & (pl.col("score_differential_post") > 0)
        )
        .group_by(["game_id", "posteam"])
        .agg(pl.lit(1).alias("had_game_winning_drive"))
        .rename({"posteam": "team_abbr"})
    )

    return (
        final_state.join(trailing_late, on=["game_id", "team_abbr"], how="left")
        .join(lead_taking, on=["game_id", "team_abbr"], how="left")
        .with_columns(
            (pl.col("final_points_for") > pl.col("final_points_allowed")).alias("team_won"),
            pl.col("had_fourth_quarter_deficit").fill_null(0),
            pl.col("had_game_winning_drive").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("team_won"))
            .then(pl.col("had_fourth_quarter_deficit"))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_fourth_quarter_comeback"),
            pl.when(pl.col("team_won"))
            .then(pl.col("had_game_winning_drive"))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_game_winning_drive"),
        )
        .select(["game_id", "team_abbr", "qb_fourth_quarter_comeback", "qb_game_winning_drive"])
    )


def _canonicalize_qb_rows(
    qb_rows: pl.DataFrame,
    qb_identity_df: pl.DataFrame | None,
    *,
    join_key: str,
) -> pl.DataFrame:
    """Attach canonical QB identifiers and names from the GSIS/PFR crosswalk."""
    if qb_rows.is_empty():
        return qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_position"))

    if "qb_id" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_id"))
    if "snap_player_id" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("snap_player_id"))
    if "qb_name" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_name"))

    if qb_identity_df is None or qb_identity_df.is_empty() or join_key not in qb_rows.columns:
        return qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_position"))

    if join_key == "qb_id":
        lookup = (
            qb_identity_df.filter(pl.col("qb_id").is_not_null())
            .select(["qb_id", "snap_player_id", "qb_name", "qb_position"])
            .unique(subset=["qb_id"], keep="first")
            .rename(
                {
                    "snap_player_id": "identity_snap_player_id",
                    "qb_name": "identity_qb_name",
                    "qb_position": "identity_qb_position",
                }
            )
        )
        return (
            qb_rows.join(lookup, on="qb_id", how="left")
            .with_columns(
                pl.coalesce([pl.col("snap_player_id"), pl.col("identity_snap_player_id")]).alias(
                    "snap_player_id"
                ),
                pl.coalesce([pl.col("identity_qb_name"), pl.col("qb_name")]).alias("qb_name"),
                pl.col("identity_qb_position").alias("qb_position"),
            )
            .drop(["identity_snap_player_id", "identity_qb_name", "identity_qb_position"])
        )

    lookup = (
        qb_identity_df.filter(pl.col("snap_player_id").is_not_null())
        .select(["snap_player_id", "qb_id", "qb_name", "qb_position"])
        .unique(subset=["snap_player_id"], keep="first")
        .rename(
            {
                "qb_id": "identity_qb_id",
                "qb_name": "identity_qb_name",
                "qb_position": "identity_qb_position",
            }
        )
    )
    return (
        qb_rows.join(lookup, on="snap_player_id", how="left")
        .with_columns(
            pl.coalesce([pl.col("identity_qb_id"), pl.col("qb_id")]).alias("qb_id"),
            pl.coalesce([pl.col("identity_qb_name"), pl.col("qb_name")]).alias("qb_name"),
            pl.col("identity_qb_position").alias("qb_position"),
        )
        .drop(["identity_qb_id", "identity_qb_name", "identity_qb_position"])
    )


def compute_qb_game_volumes_from_pbp(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame | None = None,
    qb_identity_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Combine PBP dropbacks and snap counts into one row per quarterback game."""
    schema = {
        "game_id": pl.String,
        "week": pl.Int64,
        "team_abbr": pl.String,
        "qb_name": pl.String,
        "qb_id": pl.String,
        "snap_player_id": pl.String,
        "qb_dropbacks": pl.Int64,
        "qb_offense_snaps": pl.Int64,
    }
    parts: list[pl.DataFrame] = []

    if not pbp_df.is_empty():
        dropbacks = (
            pbp_df.filter(
                pl.col("posteam").is_not_null()
                & pl.col("passer_player_name").is_not_null()
                & (pl.col("qb_dropback").fill_null(0) > 0)
            )
            .group_by(["game_id", "week", "posteam", "passer_player_id", "passer_player_name"])
            .agg(pl.col("qb_dropback").sum().cast(pl.Int64).alias("qb_dropbacks"))
            .rename(
                {
                    "posteam": "team_abbr",
                    "passer_player_id": "qb_id",
                    "passer_player_name": "qb_name",
                }
            )
            .with_columns(
                pl.lit(None, dtype=pl.String).alias("snap_player_id"),
                pl.lit(0).cast(pl.Int64).alias("qb_offense_snaps"),
            )
            .select(schema.keys())
        )
        dropbacks = _canonicalize_qb_rows(dropbacks, qb_identity_df, join_key="qb_id")
        if not dropbacks.is_empty():
            parts.append(dropbacks)

    if snap_counts_df is not None and not snap_counts_df.is_empty():
        snap_counts = (
            snap_counts_df.filter(
                (pl.col("position") == "QB") & (pl.col("offense_snaps").fill_null(0) > 0)
            )
            .group_by(["game_id", "week", "team", "player", "pfr_player_id"])
            .agg(
                pl.col("offense_snaps")
                .fill_null(0)
                .sum()
                .round(0)
                .cast(pl.Int64)
                .alias("qb_offense_snaps")
            )
            .rename(
                {
                    "team": "team_abbr",
                    "player": "qb_name",
                    "pfr_player_id": "snap_player_id",
                }
            )
            .with_columns(
                pl.lit(None, dtype=pl.String).alias("qb_id"),
                pl.lit(0).cast(pl.Int64).alias("qb_dropbacks"),
            )
            .select(schema.keys())
        )
        snap_counts = _canonicalize_qb_rows(snap_counts, qb_identity_df, join_key="snap_player_id")
        if not snap_counts.is_empty():
            parts.append(snap_counts)

    if not parts:
        return pl.DataFrame(schema=schema)

    combined = pl.concat(parts, how="diagonal_relaxed")
    if qb_identity_df is not None and not qb_identity_df.is_empty():
        combined = combined.with_columns(
            pl.coalesce([pl.col("qb_id"), pl.col("snap_player_id"), pl.col("qb_name")]).alias(
                "_qb_identity_key"
            )
        )
        group_keys = ["game_id", "week", "team_abbr", "_qb_identity_key"]
        agg_exprs: list[pl.Expr] = [pl.col("qb_name").drop_nulls().first().alias("qb_name")]
    else:
        group_keys = ["game_id", "week", "team_abbr", "qb_name"]
        agg_exprs = []

    agg_exprs.extend(
        [
            pl.col("qb_id").drop_nulls().first().alias("qb_id"),
            pl.col("snap_player_id").drop_nulls().first().alias("snap_player_id"),
            pl.col("qb_position").drop_nulls().first().alias("qb_position"),
            pl.col("qb_dropbacks").sum().cast(pl.Int64).alias("qb_dropbacks"),
            pl.col("qb_offense_snaps").sum().cast(pl.Int64).alias("qb_offense_snaps"),
        ]
    )

    result = combined.group_by(group_keys).agg(agg_exprs)
    if "_qb_identity_key" in result.columns:
        result = result.drop("_qb_identity_key")

    return (
        result.filter(pl.col("qb_position").is_null() | (pl.col("qb_position") == "QB"))
        .drop("qb_position")
        .sort(["team_abbr", "week", "game_id", "qb_name"])
    )


def compute_qb_game_stats_from_pbp(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame | None = None,
    qb_identity_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Derive per-game quarterback stats from PBP, supplemented with snap counts.

    Derived rate fields use dropbacks as the denominator.
    `qb_any_a` uses the standard formula:
    `(pass_yards + 20 * pass_tds - 45 * interceptions - sack_yards_lost) / (attempts + sacks)`.
    """
    volumes = compute_qb_game_volumes_from_pbp(pbp_df, snap_counts_df, qb_identity_df)
    if volumes.is_empty():
        return volumes.with_columns(
            pl.lit(0).cast(pl.Int64).alias("qb_attempts"),
            pl.lit(0).cast(pl.Int64).alias("qb_completions"),
            pl.lit(0.0).alias("qb_pass_yards"),
            pl.lit(0).cast(pl.Int64).alias("qb_pass_touchdowns"),
            pl.lit(0).cast(pl.Int64).alias("qb_interceptions"),
            pl.lit(0).cast(pl.Int64).alias("qb_sacks"),
            pl.lit(0.0).alias("qb_sack_yards_lost"),
            pl.lit(0).cast(pl.Int64).alias("qb_sack_fumbles_lost"),
            pl.lit(0.0).alias("qb_passing_epa"),
            pl.lit(0).cast(pl.Int64).alias("qb_designed_carries"),
            pl.lit(0.0).alias("qb_designed_rush_yards"),
            pl.lit(0.0).alias("qb_designed_rush_epa"),
            pl.lit(0).cast(pl.Int64).alias("qb_scrambles"),
            pl.lit(0.0).alias("qb_scramble_yards"),
            pl.lit(0).cast(pl.Int64).alias("qb_kneels"),
            pl.lit(None, dtype=pl.Float64).alias("qb_completion_percentage_above_expectation"),
            pl.lit(None, dtype=pl.Float64).alias("qb_scramble_rate"),
            pl.lit(None, dtype=pl.Float64).alias("qb_yards_per_scramble"),
            pl.lit(None, dtype=pl.Float64).alias("qb_designed_yards_per_carry"),
            pl.lit(None, dtype=pl.Float64).alias("qb_designed_epa_per_carry"),
        )

    pbp_columns = set(pbp_df.columns)
    sack_yards = (
        pl.col("yards_gained").fill_null(0.0) if "yards_gained" in pbp_columns else pl.lit(0.0)
    )

    pbp_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
        )
        .group_by(["game_id", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg(
            [
                pl.col("pass").fill_null(0).sum().cast(pl.Int64).alias("qb_attempts"),
                pl.col("complete_pass").fill_null(0).sum().cast(pl.Int64).alias("qb_completions"),
                pl.col("passing_yards").fill_null(0.0).sum().alias("qb_pass_yards"),
                pl.col("pass_touchdown")
                .fill_null(0)
                .sum()
                .cast(pl.Int64)
                .alias("qb_pass_touchdowns"),
                pl.col("interception").fill_null(0).sum().cast(pl.Int64).alias("qb_interceptions"),
                pl.col("sack").fill_null(0).sum().cast(pl.Int64).alias("qb_sacks"),
                pl.when(pl.col("sack").fill_null(0) > 0)
                .then((-sack_yards).clip(0.0, None))
                .otherwise(0.0)
                .sum()
                .alias("qb_sack_yards_lost"),
                pl.when(pl.col("sack").fill_null(0) > 0)
                .then(pl.col("fumble_lost").fill_null(0))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("qb_sack_fumbles_lost"),
                pl.col("qb_epa").fill_null(0.0).sum().alias("qb_passing_epa"),
                pl.col("cpoe").mean().alias("qb_completion_percentage_above_expectation"),
            ]
        )
        .rename(
            {
                "posteam": "team_abbr",
                "passer_player_id": "qb_id",
                "passer_player_name": "qb_name",
            }
        )
    )
    pbp_stats = _canonicalize_qb_rows(pbp_stats, qb_identity_df, join_key="qb_id")
    pbp_stats = pbp_stats.select(
        [
            "game_id",
            "week",
            "team_abbr",
            "qb_id",
            "qb_attempts",
            "qb_completions",
            "qb_pass_yards",
            "qb_pass_touchdowns",
            "qb_interceptions",
            "qb_sacks",
            "qb_sack_yards_lost",
            "qb_sack_fumbles_lost",
            "qb_passing_epa",
            "qb_completion_percentage_above_expectation",
        ]
    )

    rushing_stats = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team_abbr": pl.String,
            "qb_id": pl.String,
            "qb_name": pl.String,
            "snap_player_id": pl.String,
            "qb_designed_carries": pl.Int64,
            "qb_designed_rush_yards": pl.Float64,
            "qb_designed_rush_epa": pl.Float64,
            "qb_scrambles": pl.Int64,
            "qb_scramble_yards": pl.Float64,
            "qb_kneels": pl.Int64,
        }
    )
    if {
        "game_id",
        "week",
        "posteam",
        "rusher_player_id",
        "rush",
    }.issubset(pbp_columns):
        rush_flag = pl.col("rush").fill_null(0) > 0
        scramble_flag = (
            pl.col("qb_scramble").fill_null(0) > 0
            if "qb_scramble" in pbp_columns
            else pl.lit(False)
        )
        kneel_flag = (
            pl.col("qb_kneel").fill_null(0) > 0 if "qb_kneel" in pbp_columns else pl.lit(False)
        )
        two_point_flag = (
            pl.col("two_point_attempt").fill_null(0) > 0
            if "two_point_attempt" in pbp_columns
            else pl.lit(False)
        )
        designed_rush_flag = rush_flag & ~scramble_flag & ~kneel_flag & ~two_point_flag
        rush_yards = (
            pl.col("rushing_yards").fill_null(0.0)
            if "rushing_yards" in pbp_columns
            else (
                pl.col("yards_gained").fill_null(0.0)
                if "yards_gained" in pbp_columns
                else pl.lit(0.0)
            )
        )
        rush_epa = (
            pl.col("epa").fill_null(0.0)
            if "epa" in pbp_columns
            else (pl.col("qb_epa").fill_null(0.0) if "qb_epa" in pbp_columns else pl.lit(0.0))
        )
        rusher_name = pl.coalesce(
            [
                pl.col("rusher_player_name").cast(pl.String)
                if "rusher_player_name" in pbp_columns
                else pl.lit(None, dtype=pl.String),
                pl.col("passer_player_name").cast(pl.String)
                if "passer_player_name" in pbp_columns
                else pl.lit(None, dtype=pl.String),
            ]
        )
        rushing_stats = (
            pbp_df.filter(
                pl.col("posteam").is_not_null()
                & pl.col("rusher_player_id").is_not_null()
                & rush_flag
            )
            .group_by(["game_id", "week", "posteam", "rusher_player_id"])
            .agg(
                rusher_name.drop_nulls().first().alias("qb_name"),
                designed_rush_flag.cast(pl.Int64).sum().alias("qb_designed_carries"),
                rush_yards.filter(designed_rush_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_designed_rush_yards"),
                rush_epa.filter(designed_rush_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_designed_rush_epa"),
                (scramble_flag & ~two_point_flag).cast(pl.Int64).sum().alias("qb_scrambles"),
                rush_yards.filter(scramble_flag & ~two_point_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_scramble_yards"),
                (kneel_flag & ~two_point_flag).cast(pl.Int64).sum().alias("qb_kneels"),
            )
            .rename(
                {
                    "posteam": "team_abbr",
                    "rusher_player_id": "qb_id",
                }
            )
        )
        rushing_stats = _canonicalize_qb_rows(rushing_stats, qb_identity_df, join_key="qb_id")
        rushing_stats = rushing_stats.select(
            [
                "game_id",
                "week",
                "team_abbr",
                "qb_id",
                "qb_designed_carries",
                "qb_designed_rush_yards",
                "qb_designed_rush_epa",
                "qb_scrambles",
                "qb_scramble_yards",
                "qb_kneels",
            ]
        )

    return (
        volumes.join(
            pbp_stats,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .join(
            rushing_stats,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .with_columns(
            pl.col("qb_attempts").fill_null(0).cast(pl.Int64),
            pl.col("qb_completions").fill_null(0).cast(pl.Int64),
            pl.col("qb_pass_yards").fill_null(0.0),
            pl.col("qb_pass_touchdowns").fill_null(0).cast(pl.Int64),
            pl.col("qb_interceptions").fill_null(0).cast(pl.Int64),
            pl.col("qb_sacks").fill_null(0).cast(pl.Int64),
            pl.col("qb_sack_yards_lost").fill_null(0.0),
            pl.col("qb_sack_fumbles_lost").fill_null(0).cast(pl.Int64),
            pl.col("qb_passing_epa").fill_null(0.0),
            pl.col("qb_designed_carries").fill_null(0).cast(pl.Int64),
            pl.col("qb_designed_rush_yards").fill_null(0.0),
            pl.col("qb_designed_rush_epa").fill_null(0.0),
            pl.col("qb_scrambles").fill_null(0).cast(pl.Int64),
            pl.col("qb_scramble_yards").fill_null(0.0),
            pl.col("qb_kneels").fill_null(0).cast(pl.Int64),
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
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_scrambles") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_scramble_rate"),
            pl.when(pl.col("qb_scrambles") > 0)
            .then(pl.col("qb_scramble_yards") / pl.col("qb_scrambles"))
            .otherwise(None)
            .alias("qb_yards_per_scramble"),
            pl.when(pl.col("qb_designed_carries") > 0)
            .then(pl.col("qb_designed_rush_yards") / pl.col("qb_designed_carries"))
            .otherwise(None)
            .alias("qb_designed_yards_per_carry"),
            pl.when(pl.col("qb_designed_carries") > 0)
            .then(pl.col("qb_designed_rush_epa") / pl.col("qb_designed_carries"))
            .otherwise(None)
            .alias("qb_designed_epa_per_carry"),
        )
        .join(
            _compute_team_late_game_flags_from_pbp(pbp_df),
            on=["game_id", "team_abbr"],
            how="left",
        )
        .join(
            _select_primary_qb_rows(volumes)
            .select(["game_id", "team_abbr", "qb_name"])
            .with_columns(pl.lit(1).alias("_is_primary_qb")),
            on=["game_id", "team_abbr", "qb_name"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("_is_primary_qb").fill_null(0) > 0)
            .then(pl.col("qb_fourth_quarter_comeback").fill_null(0))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_fourth_quarter_comeback"),
            pl.when(pl.col("_is_primary_qb").fill_null(0) > 0)
            .then(pl.col("qb_game_winning_drive").fill_null(0))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_game_winning_drive"),
        )
        .select(
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
                "qb_designed_carries",
                "qb_designed_rush_yards",
                "qb_designed_rush_epa",
                "qb_scrambles",
                "qb_scramble_yards",
                "qb_kneels",
                "qb_epa_per_dropback",
                "qb_pass_yards_per_dropback",
                "qb_td_int_margin_rate",
                "qb_sack_rate",
                "qb_any_a",
                "qb_scramble_rate",
                "qb_yards_per_scramble",
                "qb_designed_yards_per_carry",
                "qb_designed_epa_per_carry",
                "qb_fourth_quarter_comeback",
                "qb_game_winning_drive",
                "qb_completion_percentage_above_expectation",
            ]
        )
        .sort(["team_abbr", "week", "game_id", "qb_name"])
    )


def compute_qb_season_stats(
    qb_df: pl.DataFrame,
    weekly_df: pl.DataFrame | None = None,
    min_games: int = 0,
    min_attempts: int | None = None,
) -> pl.DataFrame:
    """Aggregate per-quarterback season stats with volume and eligibility fields."""
    if qb_df.is_empty():
        return pl.DataFrame(
            schema={
                "qb_id": pl.String,
                "qb_name": pl.String,
                "team": pl.String,
                "qb_games_played": pl.Int64,
                "qb_attempts_total": pl.Int64,
                "qb_win_pct": pl.Float64,
                "qb_is_eligible": pl.Boolean,
            }
        )

    if "qb_id" not in qb_df.columns:
        qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_id"))
    if "qb_name" not in qb_df.columns:
        qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_name"))

    effective_min_attempts = (
        min_attempts
        if min_attempts is not None
        else _compute_default_qb_attempt_qualifier(weekly_df)
    )

    qb_keys = _resolve_qb_keys(qb_df)

    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric() and col not in {"week", *set(_QB_PER_GAME_COLUMNS)}
    ]

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("qb_games_played"),
    ]
    agg_exprs.extend(pl.col(col).mean().alias(col) for col in qb_stat_cols)
    for source_col, (total_col, total_dtype) in _QB_TOTAL_COLUMNS.items():
        if source_col in qb_df.columns:
            agg_exprs.append(pl.col(source_col).sum().cast(total_dtype).alias(total_col))
        elif total_col == "qb_attempts_total":
            agg_exprs.append(pl.lit(0).cast(pl.Int64).alias(total_col))

    season_stats = qb_df.group_by(qb_keys).agg(agg_exprs)

    # Keep the player's most frequent team for schedule/label context.
    team_map = (
        qb_df.group_by(qb_keys + ["team_abbr"])
        .len()
        .sort("len", descending=True)
        .group_by(qb_keys)
        .first()
        .select(qb_keys + [pl.col("team_abbr").alias("team")])
    )
    season_stats = season_stats.join(team_map, on=qb_keys, how="left")

    required_weekly_cols = {"team", "week", "points_for", "points_allowed"}
    if weekly_df is not None and required_weekly_cols.issubset(set(weekly_df.columns)):
        primary_qb_df = _select_primary_qb_rows(qb_df)
        qb_results = (
            primary_qb_df.join(
                weekly_df.select(["team", "week", "points_for", "points_allowed"]),
                left_on=["team_abbr", "week"],
                right_on=["team", "week"],
                how="left",
            )
            .with_columns(
                [
                    (pl.col("points_for") > pl.col("points_allowed"))
                    .cast(pl.Int64)
                    .alias("qb_win"),
                    (pl.col("points_for") < pl.col("points_allowed"))
                    .cast(pl.Int64)
                    .alias("qb_loss"),
                    (pl.col("points_for") == pl.col("points_allowed"))
                    .cast(pl.Int64)
                    .alias("qb_tie"),
                ]
            )
            .group_by(qb_keys)
            .agg(
                [
                    pl.col("qb_win").sum().alias("qb_wins"),
                    pl.col("qb_loss").sum().alias("qb_losses"),
                    pl.col("qb_tie").sum().alias("qb_ties"),
                ]
            )
            .with_columns(
                pl.when((pl.col("qb_wins") + pl.col("qb_losses") + pl.col("qb_ties")) > 0)
                .then(
                    (pl.col("qb_wins") + 0.5 * pl.col("qb_ties"))
                    / (pl.col("qb_wins") + pl.col("qb_losses") + pl.col("qb_ties"))
                )
                .otherwise(0.5)
                .alias("qb_win_pct")
            )
        )
        season_stats = season_stats.join(qb_results, on=qb_keys, how="left").with_columns(
            pl.col("qb_wins").fill_null(0).cast(pl.Int64),
            pl.col("qb_losses").fill_null(0).cast(pl.Int64),
            pl.col("qb_ties").fill_null(0).cast(pl.Int64),
            pl.col("qb_win_pct").fill_null(0.5),
        )
    else:
        season_stats = season_stats.with_columns(pl.lit(0.5).alias("qb_win_pct"))

    per_game_exprs = [
        pl.when(pl.col("qb_games_played") > 0)
        .then(pl.col(total_col) / pl.col("qb_games_played"))
        .otherwise(None)
        .alias(per_game_col)
        for source_col, per_game_col in _QB_PER_GAME_COLUMNS.items()
        if (total_col := _QB_TOTAL_COLUMNS[source_col][0]) in season_stats.columns
    ]
    if per_game_exprs:
        season_stats = season_stats.with_columns(per_game_exprs)

    rate_exprs: list[pl.Expr] = []
    rate_inputs = [
        ("qb_pass_yards_total", "qb_yards_per_attempt"),
        ("qb_pass_touchdowns_total", "qb_touchdown_rate"),
        ("qb_interceptions_total", "qb_interception_rate"),
        ("qb_completions_total", "qb_completion_pct"),
    ]
    for numerator_col, output_col in rate_inputs:
        if numerator_col in season_stats.columns:
            rate_exprs.append(
                pl.when(pl.col("qb_attempts_total") > 0)
                .then(pl.col(numerator_col) / pl.col("qb_attempts_total"))
                .otherwise(None)
                .alias(output_col)
            )
    carry_rate_inputs = [
        ("qb_rushing_yards_total", "qb_yards_per_carry"),
        ("qb_rushing_epa_total", "qb_epa_per_carry"),
    ]
    for numerator_col, output_col in carry_rate_inputs:
        if {numerator_col, "qb_carries_total"}.issubset(set(season_stats.columns)):
            rate_exprs.append(
                pl.when(pl.col("qb_carries_total") > 0)
                .then(pl.col(numerator_col) / pl.col("qb_carries_total"))
                .otherwise(None)
                .alias(output_col)
            )
    designed_carry_rate_inputs = [
        ("qb_designed_rush_yards_total", "qb_designed_yards_per_carry"),
        ("qb_designed_rush_epa_total", "qb_designed_epa_per_carry"),
    ]
    for numerator_col, output_col in designed_carry_rate_inputs:
        if {numerator_col, "qb_designed_carries_total"}.issubset(set(season_stats.columns)):
            rate_exprs.append(
                pl.when(pl.col("qb_designed_carries_total") > 0)
                .then(pl.col(numerator_col) / pl.col("qb_designed_carries_total"))
                .otherwise(None)
                .alias(output_col)
            )
    if {"qb_scramble_yards_total", "qb_scrambles_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when(pl.col("qb_scrambles_total") > 0)
            .then(pl.col("qb_scramble_yards_total") / pl.col("qb_scrambles_total"))
            .otherwise(None)
            .alias("qb_yards_per_scramble")
        )
    if {"qb_scrambles_total", "qb_dropbacks_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(pl.col("qb_scrambles_total") / pl.col("qb_dropbacks_total"))
            .otherwise(None)
            .alias("qb_scramble_rate")
        )
    if {"qb_passing_epa_total", "qb_dropbacks_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(pl.col("qb_passing_epa_total") / pl.col("qb_dropbacks_total"))
            .otherwise(None)
            .alias("qb_epa_per_dropback")
        )
    if {"qb_pass_yards_total", "qb_dropbacks_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(pl.col("qb_pass_yards_total") / pl.col("qb_dropbacks_total"))
            .otherwise(None)
            .alias("qb_pass_yards_per_dropback")
        )
    if {"qb_pass_touchdowns_total", "qb_interceptions_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            (pl.col("qb_pass_touchdowns_total") - pl.col("qb_interceptions_total")).alias(
                "qb_td_int_differential"
            )
        )
    if {"qb_pass_touchdowns_total", "qb_interceptions_total", "qb_dropbacks_total"}.issubset(
        set(season_stats.columns)
    ):
        rate_exprs.append(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(
                (pl.col("qb_pass_touchdowns_total") - pl.col("qb_interceptions_total"))
                / pl.col("qb_dropbacks_total")
            )
            .otherwise(None)
            .alias("qb_td_int_margin_rate")
        )
    if {"qb_sacks_total", "qb_dropbacks_total"}.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(pl.col("qb_sacks_total") / pl.col("qb_dropbacks_total"))
            .otherwise(None)
            .alias("qb_sack_rate")
        )
    if {
        "qb_pass_yards_total",
        "qb_pass_touchdowns_total",
        "qb_interceptions_total",
        "qb_sack_yards_lost_total",
        "qb_sacks_total",
    }.issubset(set(season_stats.columns)):
        rate_exprs.append(
            pl.when((pl.col("qb_attempts_total") + pl.col("qb_sacks_total")) > 0)
            .then(
                (
                    pl.col("qb_pass_yards_total")
                    + (20.0 * pl.col("qb_pass_touchdowns_total"))
                    - (45.0 * pl.col("qb_interceptions_total"))
                    - pl.col("qb_sack_yards_lost_total")
                )
                / (pl.col("qb_attempts_total") + pl.col("qb_sacks_total"))
            )
            .otherwise(None)
            .alias("qb_any_a")
        )
    if rate_exprs:
        season_stats = season_stats.with_columns(rate_exprs)

    return season_stats.with_columns(
        (
            (pl.col("qb_games_played") >= min_games)
            & (pl.col("qb_attempts_total") >= effective_min_attempts)
        ).alias("qb_is_eligible")
    ).sort("team")


def _compute_default_qb_attempt_qualifier(weekly_df: pl.DataFrame | None) -> int:
    """Return the season-appropriate QB attempt qualifier.

    When weekly team data is available, follow the standard 14 attempts per team game rule using
    the maximum regular-season team game count in that season. Fall back to the 17-game threshold
    when weekly data is unavailable.
    """
    if weekly_df is None or weekly_df.is_empty() or "team" not in weekly_df.columns:
        return 238

    team_game_counts = weekly_df.group_by("team").len()
    if team_game_counts.is_empty():
        return 238

    return int(team_game_counts.select(pl.col("len").max()).item() * 14)
