"""Expanded Tier 1 team metrics derived from play-by-play data.

Every published column here matches a metric-registry entry; formulas follow
docs/stats-catalog.md. The frame is keyed by (game keys, team, opponent_team)
and is joined onto the core weekly team frame in ``team_stats``.

Defensive mirrors are built by mirroring each offense row onto its opponent
(what the offense produced is exactly what the defense allowed), so both
sides always agree by construction.
"""

from __future__ import annotations

import polars as pl

from nfl_sos_ratings.pbp_expressions import rate_expr, scrimmage_snap_expr, value_expr

_GROUP_KEY_CANDIDATES = ("game_id", "season", "season_type", "week")

_PRESNAP_PENALTY_TYPES = ("False Start", "Delay of Game")

# Offense-row column -> opponent's defense-row column.
_DEFENSE_MIRROR_RENAMES = {
    "attempts": "attempts_faced",
    "completions": "completions_allowed",
    "completion_pct": "completion_pct_allowed",
    "net_passing_yards": "net_passing_yards_allowed",
    "passing_air_yards": "air_yards_allowed",
    "passing_yards_after_catch": "yac_allowed",
    "epa_per_dropback": "epa_per_dropback_allowed",
    "adjusted_net_yards_per_attempt": "any_a_allowed",
    "explosive_pass_rate": "explosive_pass_rate_allowed",
    "team_passer_rating": "team_passer_rating_allowed",
    "deep_attempt_rate": "deep_attempt_rate_faced",
    "sack_yards_lost": "def_sack_yards",
    "sack_rate_per_dropback": "def_sack_rate_per_dropback",
    "aux_pressure_rate": "qb_pressure_events_rate",
    "carries": "carries_faced",
    "yards_per_carry": "yards_per_carry_allowed",
    "rush_success_rate": "rush_success_rate_allowed",
    "explosive_rush_rate": "explosive_rush_rate_allowed",
    "stuffed_run_rate": "stuff_rate",
    "success_rate": "success_rate_allowed",
    "explosive_play_rate": "explosive_play_rate_allowed",
    "epa_per_offensive_snap": "epa_per_defensive_snap_allowed",
    "yards_per_offensive_snap": "yards_per_defensive_snap_allowed",
    "first_downs": "first_downs_allowed",
    "first_downs_penalty": "penalty_first_downs_allowed",
    "third_down_pct": "third_down_pct_allowed",
    "fourth_down_pct": "fourth_down_pct_allowed",
    "series_conversion_rate": "series_conversion_rate_allowed",
    "three_and_out_rate": "three_and_outs_forced_rate",
    "score_pct_per_drive": "score_pct_per_drive_allowed",
    "punt_pct_per_drive": "punts_forced_pct",
    "points_per_drive": "points_per_drive_allowed",
    "red_zone_td_pct": "red_zone_td_pct_allowed",
    "goal_to_go_td_pct": "goal_to_go_td_pct_allowed",
    "avg_starting_field_position": "avg_starting_field_position_allowed",
    "two_pt_conversion_rate": "two_pt_conversion_rate_allowed",
    "giveaways": "takeaways",
    "fumbles_lost": "fumble_recovery_opp",
    "giveaway_rate_per_offensive_snap": "takeaway_rate_per_defensive_snap",
    "giveaways_per_drive": "takeaways_per_drive",
    "aux_int_return_yards": "def_interception_yards",
    "aux_def_tds": "def_tds",
    "aux_fumble_recovery_tds": "fumble_recovery_tds",
    "aux_def_2pt": "defensive_2pt_conversions",
    "aux_havoc_rate": "havoc_rate",
    "aux_def_pen_count": "defensive_penalties",
    "aux_def_pen_yards": "defensive_penalty_yards",
    "aux_dpi": "defensive_pass_interference",
    "aux_total_yards": "aux_total_yards_allowed",
}


def compute_expanded_team_game_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Derive the expanded Tier 1 metric surface, one row per team-game."""
    if pbp_df.is_empty() or not {"posteam", "defteam"}.issubset(pbp_df.columns):
        return pl.DataFrame(schema={"team": pl.String, "opponent_team": pl.String})

    keys = [key for key in _GROUP_KEY_CANDIDATES if key in pbp_df.columns]
    plays = pbp_df.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    if plays.is_empty():
        return pl.DataFrame(schema={"team": pl.String, "opponent_team": pl.String})

    frame = _aggregate_play_stats(plays, keys)
    for extra in (
        _aggregate_series_stats(plays, keys),
        _aggregate_drive_stats(plays, keys),
    ):
        if extra is not None:
            frame = frame.join(extra, on=[*keys, "team"], how="left")

    frame = _add_offense_ratios(frame)
    frame = _join_defense_mirrors(frame, keys)
    frame = _join_committed_penalties(frame, plays, keys)
    frame = _join_touchdown_totals(frame, plays, keys)
    frame = _add_cross_side_margins(frame)

    aux_columns = [column for column in frame.columns if column.startswith("aux_")]
    sort_keys = [key for key in ("team", "week", "game_id") if key in frame.columns]
    return frame.drop(aux_columns).sort(sort_keys)


def _aggregate_play_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Aggregate play-level counts per team-game (offense perspective)."""
    columns = plays.columns
    scrimmage = scrimmage_snap_expr(columns)
    is_pass_attempt = value_expr(columns, "pass_attempt") > 0
    is_two_point = value_expr(columns, "two_point_attempt") > 0
    is_sack = value_expr(columns, "sack") > 0
    is_complete = value_expr(columns, "complete_pass") > 0
    is_dropback = value_expr(columns, "qb_dropback") > 0
    is_rush_attempt = value_expr(columns, "rush_attempt") > 0
    is_interception = value_expr(columns, "interception") > 0
    is_fumble_lost = value_expr(columns, "fumble_lost") > 0
    yards = value_expr(columns, "yards_gained", 0.0)
    two_pt_success = (
        pl.col("two_point_conv_result") == "success"
        if "two_point_conv_result" in columns
        else pl.lit(False)  # noqa: FBT003
    )
    td_team = pl.col("td_team") if "td_team" in columns else pl.lit(None, dtype=pl.String)
    is_touchdown = value_expr(columns, "touchdown") > 0
    penalty_team = (
        pl.col("penalty_team") if "penalty_team" in columns else pl.lit(None, dtype=pl.String)
    )
    penalty_type = (
        pl.col("penalty_type") if "penalty_type" in columns else pl.lit(None, dtype=pl.String)
    )
    is_penalty = value_expr(columns, "penalty") > 0
    down = value_expr(columns, "down", 0)
    ydstogo = value_expr(columns, "ydstogo", 0)
    is_go_try = (
        value_expr(columns, "fourth_down_converted") + value_expr(columns, "fourth_down_failed")
    ) > 0
    fourth_down_faced = (down == 4) & (
        scrimmage
        | (value_expr(columns, "punt_attempt") > 0)
        | (value_expr(columns, "field_goal_attempt") > 0)
    )
    xyac = pl.col("xyac_mean_yardage") if "xyac_mean_yardage" in columns else pl.lit(None)
    yac = pl.col("yards_after_catch") if "yards_after_catch" in columns else pl.lit(None)

    def _count(condition: pl.Expr, name: str) -> pl.Expr:
        return condition.cast(pl.Int64).sum().alias(name)

    return (
        plays.group_by([*keys, "posteam", "defteam"])
        .agg(
            # Passing volume (official attempts exclude sacks and two-point tries).
            _count(is_pass_attempt & ~is_sack & ~is_two_point, "attempts"),
            _count(is_complete & ~is_two_point, "completions"),
            _count(is_dropback, "dropbacks"),
            (-yards).filter(is_sack).sum().fill_null(0.0).alias("sack_yards_lost"),
            _count(value_expr(columns, "qb_scramble") > 0, "scrambles"),
            value_expr(columns, "rushing_yards", 0.0)
            .filter(value_expr(columns, "qb_scramble") > 0)
            .sum()
            .fill_null(0.0)
            .alias("scramble_yards"),
            value_expr(columns, "air_yards", 0.0)
            .filter(is_pass_attempt & ~is_two_point)
            .sum()
            .fill_null(0.0)
            .alias("passing_air_yards"),
            value_expr(columns, "yards_after_catch", 0.0)
            .filter(is_complete)
            .sum()
            .fill_null(0.0)
            .alias("passing_yards_after_catch"),
            value_expr(columns, "passing_yards", 0.0)
            .filter(is_complete)
            .max()
            .alias("longest_pass"),
            _count((value_expr(columns, "fumble") > 0) & is_sack, "sack_fumbles"),
            _count(is_pass_attempt & two_pt_success & is_two_point, "passing_2pt_conversions"),
            value_expr(columns, "air_epa", 0.0)
            .filter(is_pass_attempt)
            .sum()
            .fill_null(0.0)
            .alias("air_epa_total"),
            value_expr(columns, "yac_epa", 0.0)
            .filter(is_complete)
            .sum()
            .fill_null(0.0)
            .alias("yac_epa_total"),
            xyac.filter(is_complete).mean().alias("xyac_per_completion"),
            (yac - xyac).filter(is_complete).mean().alias("yac_over_expected_per_completion"),
            # Rushing (official carries exclude two-point tries).
            _count(is_rush_attempt & ~is_two_point, "carries"),
            _count(
                (value_expr(columns, "rush") > 0)
                & (value_expr(columns, "qb_kneel") == 0)
                & ~is_two_point,
                "designed_carries",
            ),
            value_expr(columns, "rushing_yards", 0.0)
            .filter(is_rush_attempt)
            .max()
            .alias("longest_rush"),
            _count((value_expr(columns, "fumble") > 0) & is_rush_attempt, "rushing_fumbles"),
            _count(is_rush_attempt & two_pt_success & is_two_point, "rushing_2pt_conversions"),
            # Overall offense (scrimmage snaps only).
            value_expr(columns, "epa", 0.0)
            .filter(scrimmage)
            .sum()
            .fill_null(0.0)
            .alias("offensive_epa"),
            value_expr(columns, "wpa", 0.0)
            .filter(scrimmage)
            .sum()
            .fill_null(0.0)
            .alias("offensive_wpa"),
            value_expr(columns, "success", 0).filter(scrimmage).mean().alias("success_rate"),
            value_expr(columns, "success", 0).filter(is_dropback).mean().alias("pass_success_rate"),
            value_expr(columns, "success", 0)
            .filter(
                (value_expr(columns, "rush") > 0)
                & (value_expr(columns, "qb_kneel") == 0)
                & ~is_two_point
            )
            .mean()
            .alias("rush_success_rate"),
            value_expr(columns, "shotgun", 0).filter(scrimmage).mean().alias("shotgun_rate"),
            value_expr(columns, "no_huddle", 0).filter(scrimmage).mean().alias("no_huddle_rate"),
            (pl.col("pass_oe") if "pass_oe" in columns else pl.lit(None))
            .filter(scrimmage)
            .mean()
            .alias("pass_rate_over_expected"),
            _count(scrimmage, "aux_off_snaps"),
            _count(scrimmage & (down <= 2), "aux_early_snaps"),
            _count(is_dropback & (down <= 2), "aux_early_dropbacks"),
            _count(is_complete & (yards >= 20), "aux_explosive_passes"),
            _count(is_rush_attempt & ~is_two_point & (yards >= 10), "aux_explosive_rushes"),
            _count(is_rush_attempt & ~is_two_point & (yards <= 0), "aux_stuffed_rushes"),
            _count(
                is_pass_attempt & ~is_two_point & (pl.col("pass_length") == "deep")
                if "pass_length" in columns
                else pl.lit(False),  # noqa: FBT003
                "aux_deep_attempts",
            ),
            # Turnovers.
            _count((value_expr(columns, "fumble") > 0) & scrimmage, "fumbles"),
            _count((value_expr(columns, "fumble") > 0) & is_complete, "receiving_fumbles"),
            _count(is_fumble_lost & is_complete, "receiving_fumbles_lost"),
            _count(is_fumble_lost & scrimmage, "fumbles_lost"),
            _count(is_interception, "aux_interceptions"),
            value_expr(columns, "epa", 0.0)
            .filter(is_interception | is_fumble_lost)
            .sum()
            .fill_null(0.0)
            .alias("turnover_epa"),
            value_expr(columns, "return_yards", 0.0)
            .filter(is_interception)
            .sum()
            .fill_null(0.0)
            .alias("aux_int_return_yards"),
            # Downs and conversions.
            _count(value_expr(columns, "first_down") > 0, "first_downs"),
            _count(value_expr(columns, "first_down_penalty") > 0, "first_downs_penalty"),
            _count(value_expr(columns, "third_down_converted") > 0, "third_down_conversions"),
            _count(
                (
                    value_expr(columns, "third_down_converted")
                    + value_expr(columns, "third_down_failed")
                )
                > 0,
                "third_down_attempts",
            ),
            ydstogo.filter(scrimmage & (down == 3)).mean().alias("third_down_avg_distance"),
            _count(value_expr(columns, "fourth_down_converted") > 0, "fourth_down_conversions"),
            _count(is_go_try, "fourth_down_attempts"),
            _count(fourth_down_faced, "aux_fourth_downs_faced"),
            _count(is_go_try & (ydstogo <= 2), "aux_fourth_short_go"),
            _count(fourth_down_faced & (ydstogo <= 2), "aux_fourth_short_faced"),
            _count(value_expr(columns, "fourth_down_failed") > 0, "turnovers_on_downs"),
            # Scoring extras.
            _count(is_two_point, "two_pt_attempts"),
            _count(is_two_point & two_pt_success, "two_pt_conversions"),
            _count(value_expr(columns, "pass_touchdown") > 0, "aux_pass_tds"),
            _count(value_expr(columns, "rush_touchdown") > 0, "aux_rush_tds"),
            # Aux inputs for ratios and defensive mirrors.
            value_expr(columns, "passing_yards", 0.0).sum().fill_null(0.0).alias("aux_pass_yards"),
            value_expr(columns, "rushing_yards", 0.0).sum().fill_null(0.0).alias("aux_rush_yards"),
            value_expr(columns, "epa", 0.0)
            .filter(is_dropback)
            .sum()
            .fill_null(0.0)
            .alias("aux_pass_epa"),
            value_expr(columns, "epa", 0.0)
            .filter(value_expr(columns, "rush") > 0)
            .sum()
            .fill_null(0.0)
            .alias("aux_rush_epa"),
            _count(is_sack, "aux_sacks"),
            _count(value_expr(columns, "qb_hit") > 0, "aux_qb_hits"),
            (
                _count(
                    (value_expr(columns, "tackled_for_loss") > 0)
                    | (value_expr(columns, "fumble_forced") > 0)
                    | is_interception
                    | (
                        pl.col("pass_defense_1_player_id").is_not_null()
                        if "pass_defense_1_player_id" in columns
                        else pl.lit(False)  # noqa: FBT003
                    ),
                    "aux_havoc_events",
                )
            ),
            _count(is_touchdown & (td_team == pl.col("defteam")), "aux_def_tds"),
            _count(
                is_touchdown & (td_team == pl.col("defteam")) & (value_expr(columns, "fumble") > 0),
                "aux_fumble_recovery_tds",
            ),
            value_expr(columns, "defensive_two_point_conv", 0)
            .sum()
            .cast(pl.Int64)
            .alias("aux_def_2pt"),
            # Penalties observed from this offense's plays.
            _count(is_penalty & (penalty_team == pl.col("posteam")), "offensive_penalties"),
            value_expr(columns, "penalty_yards", 0.0)
            .filter(is_penalty & (penalty_team == pl.col("posteam")))
            .sum()
            .fill_null(0.0)
            .alias("offensive_penalty_yards"),
            _count(
                is_penalty
                & (penalty_team == pl.col("posteam"))
                & penalty_type.is_in(list(_PRESNAP_PENALTY_TYPES)),
                "aux_presnap_penalties",
            ),
            _count(is_penalty & (penalty_team == pl.col("defteam")), "aux_def_pen_count"),
            value_expr(columns, "penalty_yards", 0.0)
            .filter(is_penalty & (penalty_team == pl.col("defteam")))
            .sum()
            .fill_null(0.0)
            .alias("aux_def_pen_yards"),
            _count(
                is_penalty
                & (penalty_team == pl.col("defteam"))
                & (penalty_type == "Defensive Pass Interference"),
                "aux_dpi",
            ),
        )
        .rename({"posteam": "team", "defteam": "opponent_team"})
    )


def _aggregate_series_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame | None:
    """Aggregate first-down series outcomes per team-game."""
    if not {"series", "series_success"}.issubset(plays.columns):
        return None

    columns = plays.columns
    per_series = plays.group_by([*keys, "posteam", "series"]).agg(
        value_expr(columns, "series_success", 0).max().alias("converted"),
        value_expr(columns, "goal_to_go", 0).max().alias("goal_to_go"),
        (
            (pl.col("series_result") == "Touchdown").max()
            if "series_result" in columns
            else pl.lit(False).max()  # noqa: FBT003
        ).alias("touchdown"),
    )
    return (
        per_series.group_by([*keys, "posteam"])
        .agg(
            pl.len().cast(pl.Int64).alias("series"),
            pl.col("converted").mean().alias("series_conversion_rate"),
            pl.col("touchdown")
            .filter(pl.col("goal_to_go") > 0)
            .cast(pl.Int64)
            .mean()
            .alias("goal_to_go_td_pct"),
        )
        .rename({"posteam": "team"})
    )


def _aggregate_drive_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame | None:
    """Aggregate drive-level outcomes and field position per team-game."""
    if not {"fixed_drive", "fixed_drive_result"}.issubset(plays.columns):
        return None

    columns = plays.columns
    per_drive = plays.group_by([*keys, "posteam", "fixed_drive"]).agg(
        pl.col("fixed_drive_result").first().alias("result"),
        value_expr(columns, "drive_play_count", 0).first().alias("play_count"),
        value_expr(columns, "drive_first_downs", 0).first().alias("first_downs"),
        value_expr(columns, "drive_inside20", 0).max().alias("inside_20"),
        value_expr(columns, "ydsnet", 0).first().alias("net_yards"),
        value_expr(columns, "drive_yards_penalized", 0).first().alias("yards_penalized"),
        (
            pl.col("drive_time_of_possession").first()
            if "drive_time_of_possession" in columns
            else pl.lit(None, dtype=pl.String).first()
        ).alias("possession_clock"),
        (
            pl.col("drive_start_yard_line").first()
            if "drive_start_yard_line" in columns
            else pl.lit(None, dtype=pl.String).first()
        ).alias("start_yard_line"),
        (
            value_expr(columns, "posteam_score_post", 0).last()
            - value_expr(columns, "posteam_score", 0).first()
        ).alias("points"),
    )

    start_side = pl.col("start_yard_line").str.extract(r"^([A-Z]{2,3})", 1)
    start_number = pl.col("start_yard_line").str.extract(r"(\d+)$", 1).cast(pl.Int64)
    per_drive = per_drive.with_columns(
        pl.when(pl.col("start_yard_line") == "50")
        .then(50)
        .when(start_side == pl.col("posteam"))
        .then(start_number)
        .otherwise(100 - start_number)
        .alias("start_from_own_goal"),
        (
            pl.col("possession_clock").str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60
            + pl.col("possession_clock").str.extract(r":(\d+)$", 1).cast(pl.Int64)
        ).alias("possession_seconds"),
        pl.col("result").is_in(["Touchdown", "Field goal"]).alias("scored"),
        (pl.col("result") == "Punt").alias("punted"),
        pl.col("result").is_in(["Interception", "Fumble", "Opp touchdown"]).alias("turned_over"),
    )

    return (
        per_drive.group_by([*keys, "posteam"])
        .agg(
            pl.len().cast(pl.Int64).alias("drives"),
            pl.col("net_yards").mean().alias("yards_per_drive"),
            pl.col("play_count").mean().alias("plays_per_drive"),
            pl.col("possession_seconds").mean().alias("time_per_drive"),
            pl.col("first_downs").mean().alias("first_downs_per_drive"),
            pl.col("scored").mean().alias("score_pct_per_drive"),
            pl.col("punted").mean().alias("punt_pct_per_drive"),
            pl.col("turned_over").mean().alias("turnover_pct_per_drive"),
            (pl.col("punted") & (pl.col("first_downs") == 0)).mean().alias("three_and_out_rate"),
            pl.col("points").mean().alias("points_per_drive"),
            (pl.col("inside_20") > 0).cast(pl.Int64).sum().alias("red_zone_trips"),
            (pl.col("result") == "Touchdown")
            .filter(pl.col("inside_20") > 0)
            .cast(pl.Int64)
            .mean()
            .alias("red_zone_td_pct"),
            pl.col("points")
            .filter(pl.col("inside_20") > 0)
            .mean()
            .alias("points_per_red_zone_trip"),
            pl.col("start_from_own_goal").mean().alias("avg_starting_field_position"),
            pl.col("scored")
            .filter(pl.col("start_from_own_goal") <= 25)
            .mean()
            .alias("long_field_score_pct"),
            pl.col("yards_penalized").sum().alias("drive_penalty_yards"),
        )
        .rename({"posteam": "team"})
    )


def _join_committed_penalties(
    frame: pl.DataFrame, plays: pl.DataFrame, keys: list[str]
) -> pl.DataFrame:
    """Join all-unit committed penalties and the game penalty differentials."""
    if "penalty_team" not in plays.columns or "penalty" not in plays.columns:
        return frame

    columns = plays.columns
    committed = (
        plays.filter((value_expr(columns, "penalty") > 0) & pl.col("penalty_team").is_not_null())
        .group_by([*keys, "penalty_team"])
        .agg(
            pl.len().cast(pl.Int64).alias("penalties"),
            value_expr(columns, "penalty_yards", 0.0).sum().fill_null(0.0).alias("penalty_yards"),
        )
        .rename({"penalty_team": "team"})
        .with_columns(pl.col("team").cast(pl.String))
    )

    frame = frame.join(committed, on=[*keys, "team"], how="left").with_columns(
        pl.col("penalties").fill_null(0),
        pl.col("penalty_yards").fill_null(0.0),
    )
    opponent_committed = committed.rename(
        {
            "team": "opponent_team",
            "penalties": "aux_opp_penalties",
            "penalty_yards": "aux_opp_penalty_yards",
        }
    )
    return frame.join(opponent_committed, on=[*keys, "opponent_team"], how="left").with_columns(
        (pl.col("aux_opp_penalties").fill_null(0) - pl.col("penalties")).alias(
            "penalty_differential"
        ),
        (pl.col("aux_opp_penalty_yards").fill_null(0.0) - pl.col("penalty_yards")).alias(
            "penalty_yards_differential"
        ),
    )


def _join_touchdown_totals(
    frame: pl.DataFrame, plays: pl.DataFrame, keys: list[str]
) -> pl.DataFrame:
    """Join total touchdowns credited to each team via td_team."""
    if "td_team" not in plays.columns or "touchdown" not in plays.columns:
        return frame

    columns = plays.columns
    touchdowns = (
        plays.filter((value_expr(columns, "touchdown") > 0) & pl.col("td_team").is_not_null())
        .group_by([*keys, "td_team"])
        .agg(pl.len().cast(pl.Int64).alias("total_tds"))
        .rename({"td_team": "team"})
        .with_columns(pl.col("team").cast(pl.String))
    )
    return frame.join(touchdowns, on=[*keys, "team"], how="left").with_columns(
        pl.col("total_tds").fill_null(0)
    )


def _passer_rating_expr(
    completions: str, attempts: str, yards: str, touchdowns: str, interceptions: str
) -> pl.Expr:
    """Return the official NFL passer-rating formula as a Polars expression."""

    def _clamp(component: pl.Expr) -> pl.Expr:
        return component.clip(0.0, 2.375)

    attempts_col = pl.col(attempts)
    a = _clamp(((pl.col(completions) / attempts_col) - 0.3) * 5.0)
    b = _clamp(((pl.col(yards) / attempts_col) - 3.0) * 0.25)
    c = _clamp((pl.col(touchdowns) / attempts_col) * 20.0)
    d = _clamp(2.375 - ((pl.col(interceptions) / attempts_col) * 25.0))
    return pl.when(attempts_col > 0).then((a + b + c + d) / 6.0 * 100.0).otherwise(None)


def _add_offense_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive offense-side ratio columns from the aggregated counts."""
    frame = frame.with_columns(
        (pl.col("aux_pass_yards") + pl.col("aux_rush_yards")).alias("aux_total_yards"),
        (pl.col("aux_interceptions") + pl.col("fumbles_lost")).alias("giveaways"),
        (pl.col("aux_pass_tds") + pl.col("aux_rush_tds")).alias("scrimmage_tds"),
        (pl.col("aux_pass_yards") - pl.col("sack_yards_lost")).alias("net_passing_yards"),
    )
    frame = frame.with_columns(pl.col("scrimmage_tds").alias("offensive_tds"))

    ratio_specs = [
        ("completions", "attempts", "completion_pct"),
        ("aux_sacks", "dropbacks", "sack_rate_per_dropback"),
        ("passing_air_yards", "attempts", "air_yards_per_attempt"),
        ("passing_yards_after_catch", "completions", "yac_per_completion"),
        ("aux_pass_yards", "attempts", "yards_per_attempt"),
        ("aux_pass_yards", "dropbacks", "yards_per_dropback"),
        ("aux_pass_epa", "dropbacks", "epa_per_dropback"),
        ("aux_pass_tds", "attempts", "passing_td_rate_per_attempt"),
        ("aux_interceptions", "attempts", "int_rate_per_attempt"),
        ("aux_explosive_passes", "dropbacks", "explosive_pass_rate"),
        ("aux_deep_attempts", "attempts", "deep_attempt_rate"),
        ("aux_rush_yards", "carries", "yards_per_carry"),
        ("aux_rush_epa", "carries", "epa_per_carry"),
        ("aux_explosive_rushes", "carries", "explosive_rush_rate"),
        ("aux_stuffed_rushes", "carries", "stuffed_run_rate"),
        ("offensive_epa", "aux_off_snaps", "epa_per_offensive_snap"),
        ("aux_total_yards", "aux_off_snaps", "yards_per_offensive_snap"),
        ("dropbacks", "aux_off_snaps", "pass_rate"),
        ("aux_early_dropbacks", "aux_early_snaps", "early_down_pass_rate"),
        ("third_down_conversions", "third_down_attempts", "third_down_pct"),
        ("fourth_down_conversions", "fourth_down_attempts", "fourth_down_pct"),
        ("fourth_down_attempts", "aux_fourth_downs_faced", "fourth_down_go_rate"),
        ("aux_fourth_short_go", "aux_fourth_short_faced", "fourth_down_aggressiveness"),
        ("two_pt_conversions", "two_pt_attempts", "two_pt_conversion_rate"),
        ("giveaways", "aux_off_snaps", "giveaway_rate_per_offensive_snap"),
        ("offensive_penalties", "aux_off_snaps", "penalty_rate_per_offensive_snap"),
        ("aux_presnap_penalties", "aux_off_snaps", "presnap_penalty_rate"),
        ("aux_havoc_events", "aux_off_snaps", "aux_havoc_rate"),
    ]
    frame = frame.with_columns(
        [
            rate_expr(numerator, denominator, output)
            for numerator, denominator, output in ratio_specs
            if {numerator, denominator}.issubset(frame.columns)
        ]
    )

    derived = [
        pl.when((pl.col("attempts") + pl.col("aux_sacks")) > 0)
        .then(
            (pl.col("aux_pass_yards") - pl.col("sack_yards_lost"))
            / (pl.col("attempts") + pl.col("aux_sacks"))
        )
        .otherwise(None)
        .alias("net_yards_per_attempt"),
        pl.when((pl.col("attempts") + pl.col("aux_sacks")) > 0)
        .then(
            (
                pl.col("aux_pass_yards")
                + 20.0 * pl.col("aux_pass_tds")
                - 45.0 * pl.col("aux_interceptions")
                - pl.col("sack_yards_lost")
            )
            / (pl.col("attempts") + pl.col("aux_sacks"))
        )
        .otherwise(None)
        .alias("adjusted_net_yards_per_attempt"),
        _passer_rating_expr(
            "completions", "attempts", "aux_pass_yards", "aux_pass_tds", "aux_interceptions"
        ).alias("team_passer_rating"),
        pl.when(pl.col("aux_off_snaps") > 0)
        .then(
            (pl.col("aux_explosive_passes") + pl.col("aux_explosive_rushes"))
            / pl.col("aux_off_snaps")
        )
        .otherwise(None)
        .alias("explosive_play_rate"),
        pl.when(pl.col("dropbacks") > 0)
        .then((pl.col("aux_sacks") + pl.col("aux_qb_hits")) / pl.col("dropbacks"))
        .otherwise(None)
        .alias("aux_pressure_rate"),
    ]
    frame = frame.with_columns(derived)

    if "drives" in frame.columns:
        frame = frame.with_columns(rate_expr("giveaways", "drives", "giveaways_per_drive"))
    return frame


def _join_defense_mirrors(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Mirror each offense row onto the opponent as its defensive surface.

    A full join keeps teams that only appear on defense in a fixture; in real
    games both teams run offense, so this is a no-op there.
    """
    available = {
        source: target
        for source, target in _DEFENSE_MIRROR_RENAMES.items()
        if source in frame.columns
    }
    mirror_columns = [pl.col(source).alias(target) for source, target in available.items()]
    if "turnover_epa" in frame.columns:
        mirror_columns.append((-pl.col("turnover_epa")).alias("takeaway_epa"))
    mirror = frame.select(
        [
            *keys,
            pl.col("opponent_team").alias("team"),
            pl.col("team").alias("opponent_team"),
            *mirror_columns,
        ]
    )
    return frame.join(mirror, on=[*keys, "team", "opponent_team"], how="full", coalesce=True)


def _add_cross_side_margins(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive whole-team margins that need both offense and defense values."""
    margin_specs = [
        ("aux_total_yards", "aux_total_yards_allowed", "total_yards_differential"),
        ("epa_per_offensive_snap", "epa_per_defensive_snap_allowed", "epa_margin_per_play"),
        ("success_rate", "success_rate_allowed", "success_rate_margin"),
    ]
    margins = [
        (pl.col(own) - pl.col(allowed)).alias(output)
        for own, allowed, output in margin_specs
        if {own, allowed}.issubset(frame.columns)
    ]
    return frame.with_columns(margins) if margins else frame
