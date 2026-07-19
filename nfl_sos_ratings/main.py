"""Single-season PBP-first pipeline for team and quarterback ratings.

The pipeline loads PBP-derived team and QB game data, builds one-hop opponent
profiles, computes published ridge-backed ratings, and writes simultaneous-
adjustment outputs for auditability and UI detail surfaces.
"""

import io
import os
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from nfl_sos_ratings.config import DATA_DIR, SEASON, TEAM_ABBR_ALIASES
from nfl_sos_ratings.data_loader import (
    load_pbp_data,
    load_qb_stats,
    load_schedule,
    load_weekly_team_stats,
)
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.opponent_stats import compute_all_opponent_profiles
from nfl_sos_ratings.qb_opponent_stats import compute_qb_opponent_profiles
from nfl_sos_ratings.qb_ratings import compute_qb_ratings
from nfl_sos_ratings.qb_stats import compute_qb_season_stats
from nfl_sos_ratings.ratings import compute_ratings
from nfl_sos_ratings.simultaneous_adjustment import (
    compute_qb_adjusted_stats,
    compute_team_adjusted_stats,
    solve_srs,
)
from nfl_sos_ratings.team_stats import (
    compute_all_teams_per_game,
    compute_all_teams_qb_per_game,
    compute_win_totals,
)
from nfl_sos_ratings.validation.snapshots import (
    build_play_level_team_adjusted_snapshot,
    build_play_level_team_frame_from_pbp,
    build_special_teams_game_frame_from_pbp,
    build_special_teams_rating_snapshot,
)

# Response-column membership lives in the metric registry (the SSOT).
_TEAM_SIMULTANEOUS_COLS = get_registry().pool_columns("team_simultaneous")

_QB_SIMULTANEOUS_COLS = get_registry().pool_columns("qb_simultaneous")

_QB_FACED_DEFENSE_COLUMN = "adj_def_qb_epa_per_dropback_faced"
_QB_FACED_RUSH_DEFENSE_COLUMN = "adj_def_rushing_epa_per_offensive_snap_faced"
_QB_ADJUSTED_DESIGNED_RUSH_COLUMN = "adj_qb_designed_rush_epa_per_carry"
_TEAM_SOS_COLUMN = "sos"
_QB_FACED_OVERALL_QUALITY_COLUMN = "faced_opp_SaCR"


def _matching_qb_join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    """Return the most specific QB identity keys shared by two frames."""
    keys = [
        key for key in ("qb_id", "qb_name", "team") if key in left.columns and key in right.columns
    ]
    if keys:
        return keys
    return ["team"] if "team" in left.columns and "team" in right.columns else []


def _add_qb_differentials(qb_combined: pl.DataFrame) -> pl.DataFrame:
    """Add paired QB minus opponent-allowed QB differential columns."""
    qb_diff_exprs = [
        (pl.col(col) - pl.col(f"qopp_{col}")).alias(f"diff_{col}")
        for col in qb_combined.columns
        if col.startswith("qb_") and f"qopp_{col}" in qb_combined.columns
    ]
    return qb_combined.with_columns(qb_diff_exprs) if qb_diff_exprs else qb_combined


def _normalize_team_abbreviations(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize team abbreviations in the selected columns using the shared alias table."""
    exprs = [
        pl.col(column).cast(pl.String).replace(TEAM_ABBR_ALIASES).alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _qb_schedule_group_keys(df: pl.DataFrame) -> list[str]:
    """Return the grouping keys that match the published one-row-per-QB season surface."""
    keys = [key for key in ("qb_id", "qb_name") if key in df.columns]
    if keys:
        return keys
    if "team_abbr" in df.columns:
        return ["team_abbr"]
    if "team" in df.columns:
        return ["team"]
    return []


def _raise_on_unmatched_opponents(frame: pl.DataFrame, value_column: str, label: str) -> None:
    """Fail loudly when an opponent-context join is missing a required value."""
    unmatched = frame.filter(pl.col(value_column).is_null())
    if unmatched.is_empty():
        return

    opponents = unmatched.select("opponent_team").unique().sort("opponent_team")
    raise ValueError(
        f"Missing {label} for opponent_team values: " + ", ".join(opponents.to_series().to_list())
    )


def _build_qb_combined(
    qb_season_stats: pl.DataFrame,
    qb_opp_profiles: pl.DataFrame | None,
) -> pl.DataFrame:
    """Join QB season rows to opponent context and derive differential columns."""
    if qb_opp_profiles is None:
        return qb_season_stats

    join_keys = _matching_qb_join_keys(qb_season_stats, qb_opp_profiles)
    qb_combined = qb_season_stats.join(qb_opp_profiles, on=join_keys, how="left")
    return _add_qb_differentials(qb_combined)


def _build_qb_faced_defense_adjustments(
    qb_games: pl.DataFrame,
    defense_adjustments: pl.DataFrame,
) -> pl.DataFrame:
    """Average faced-defense coefficients, weighted by the QB's dropback volume when available."""
    defense_column = "adj_def_qb_epa_per_dropback"
    if (
        qb_games.is_empty()
        or defense_adjustments.is_empty()
        or "opponent_team" not in qb_games.columns
        or defense_column not in defense_adjustments.columns
    ):
        return pl.DataFrame(schema={"team": pl.String, _QB_FACED_DEFENSE_COLUMN: pl.Float64})

    normalized_qb_games = _normalize_team_abbreviations(qb_games, ["opponent_team"])
    normalized_defense_adjustments = _normalize_team_abbreviations(defense_adjustments, ["team"])

    faced_defenses = normalized_qb_games.join(
        normalized_defense_adjustments.rename(
            {"team": "opponent_team", defense_column: _QB_FACED_DEFENSE_COLUMN}
        ),
        on="opponent_team",
        how="left",
    )
    _raise_on_unmatched_opponents(
        faced_defenses,
        _QB_FACED_DEFENSE_COLUMN,
        "QB defense adjustments",
    )

    group_keys = _qb_schedule_group_keys(faced_defenses)
    if not group_keys:
        return pl.DataFrame(schema={"team": pl.String, _QB_FACED_DEFENSE_COLUMN: pl.Float64})

    if "qb_dropbacks" in faced_defenses.columns:
        schedule_strength = (
            faced_defenses.with_columns(pl.col("qb_dropbacks").cast(pl.Float64).fill_null(0.0))
            .group_by(group_keys)
            .agg(
                (pl.col(_QB_FACED_DEFENSE_COLUMN) * pl.col("qb_dropbacks"))
                .sum()
                .alias("_weighted_defense"),
                pl.col("qb_dropbacks").sum().alias("_dropback_total"),
                pl.col(_QB_FACED_DEFENSE_COLUMN).mean().alias("_fallback_mean"),
            )
            .with_columns(
                pl.when(pl.col("_dropback_total") > 0.0)
                .then(pl.col("_weighted_defense") / pl.col("_dropback_total"))
                .otherwise(pl.col("_fallback_mean"))
                .alias(_QB_FACED_DEFENSE_COLUMN)
            )
            .select([*group_keys, _QB_FACED_DEFENSE_COLUMN])
        )
    else:
        schedule_strength = faced_defenses.group_by(group_keys).agg(
            pl.col(_QB_FACED_DEFENSE_COLUMN).mean().alias(_QB_FACED_DEFENSE_COLUMN)
        )
    if "team_abbr" in schedule_strength.columns and "team" not in schedule_strength.columns:
        schedule_strength = schedule_strength.rename({"team_abbr": "team"})
    return schedule_strength


def _build_qb_faced_rush_defense_adjustments(
    qb_games: pl.DataFrame,
    team_adjustments: pl.DataFrame,
) -> pl.DataFrame:
    """Average faced rush-defense coefficients, weighted by designed carries when available."""
    defense_column = "adj_def_rushing_epa_per_offensive_snap"
    if (
        qb_games.is_empty()
        or team_adjustments.is_empty()
        or "opponent_team" not in qb_games.columns
        or defense_column not in team_adjustments.columns
    ):
        return pl.DataFrame(schema={"team": pl.String, _QB_FACED_RUSH_DEFENSE_COLUMN: pl.Float64})

    normalized_qb_games = _normalize_team_abbreviations(qb_games, ["opponent_team"])
    normalized_team_adjustments = _normalize_team_abbreviations(team_adjustments, ["team"])

    faced_rush_defenses = normalized_qb_games.join(
        normalized_team_adjustments.rename(
            {"team": "opponent_team", defense_column: _QB_FACED_RUSH_DEFENSE_COLUMN}
        ),
        on="opponent_team",
        how="left",
    )
    _raise_on_unmatched_opponents(
        faced_rush_defenses,
        _QB_FACED_RUSH_DEFENSE_COLUMN,
        "QB rush-defense adjustments",
    )

    group_keys = _qb_schedule_group_keys(faced_rush_defenses)
    if not group_keys:
        return pl.DataFrame(schema={"team": pl.String, _QB_FACED_RUSH_DEFENSE_COLUMN: pl.Float64})

    if "qb_designed_carries" in faced_rush_defenses.columns:
        schedule_strength = (
            faced_rush_defenses.with_columns(
                pl.col("qb_designed_carries").cast(pl.Float64).fill_null(0.0)
            )
            .group_by(group_keys)
            .agg(
                (pl.col(_QB_FACED_RUSH_DEFENSE_COLUMN) * pl.col("qb_designed_carries"))
                .sum()
                .alias("_weighted_defense"),
                pl.col("qb_designed_carries").sum().alias("_carry_total"),
                pl.col(_QB_FACED_RUSH_DEFENSE_COLUMN).mean().alias("_fallback_mean"),
            )
            .with_columns(
                pl.when(pl.col("_carry_total") > 0.0)
                .then(pl.col("_weighted_defense") / pl.col("_carry_total"))
                .otherwise(pl.col("_fallback_mean"))
                .alias(_QB_FACED_RUSH_DEFENSE_COLUMN)
            )
            .select([*group_keys, _QB_FACED_RUSH_DEFENSE_COLUMN])
        )
    else:
        schedule_strength = faced_rush_defenses.group_by(group_keys).agg(
            pl.col(_QB_FACED_RUSH_DEFENSE_COLUMN).mean().alias(_QB_FACED_RUSH_DEFENSE_COLUMN)
        )
    if "team_abbr" in schedule_strength.columns and "team" not in schedule_strength.columns:
        schedule_strength = schedule_strength.rename({"team_abbr": "team"})
    return schedule_strength


def _build_qb_adjusted_designed_rush_surface(
    qb_combined: pl.DataFrame,
    faced_rush_defense: pl.DataFrame,
) -> pl.DataFrame:
    """Adjust designed-rush EPA per carry by the faced rush-defense coefficient."""
    if qb_combined.is_empty() or "qb_designed_epa_per_carry" not in qb_combined.columns:
        return pl.DataFrame(schema={_QB_ADJUSTED_DESIGNED_RUSH_COLUMN: pl.Float64})

    join_keys = _matching_qb_join_keys(qb_combined, faced_rush_defense)
    if not join_keys:
        return pl.DataFrame(schema={_QB_ADJUSTED_DESIGNED_RUSH_COLUMN: pl.Float64})

    return (
        qb_combined.select([*join_keys, "qb_designed_epa_per_carry"])
        .join(faced_rush_defense, on=join_keys, how="left")
        .with_columns(
            pl.when(
                pl.col("qb_designed_epa_per_carry").is_not_null()
                & pl.col(_QB_FACED_RUSH_DEFENSE_COLUMN).is_not_null()
            )
            .then(pl.col("qb_designed_epa_per_carry") + pl.col(_QB_FACED_RUSH_DEFENSE_COLUMN))
            .otherwise(pl.col("qb_designed_epa_per_carry"))
            .alias(_QB_ADJUSTED_DESIGNED_RUSH_COLUMN)
        )
        .select([*join_keys, _QB_ADJUSTED_DESIGNED_RUSH_COLUMN])
    )


def _build_team_schedule_strength(
    team_games: pl.DataFrame,
    team_ratings: pl.DataFrame,
) -> pl.DataFrame:
    """Return each team's played-game mean opponent SaCR as a published schedule context."""
    if (
        team_games.is_empty()
        or team_ratings.is_empty()
        or "opponent_team" not in team_games.columns
    ):
        return pl.DataFrame(schema={"team": pl.String, _TEAM_SOS_COLUMN: pl.Float64})

    joined = _normalize_team_abbreviations(team_games, ["opponent_team"]).join(
        _normalize_team_abbreviations(team_ratings, ["team"])
        .select(["team", "SaCR"])
        .rename({"team": "opponent_team", "SaCR": _TEAM_SOS_COLUMN}),
        on="opponent_team",
        how="left",
    )
    _raise_on_unmatched_opponents(joined, _TEAM_SOS_COLUMN, "team strength-of-schedule values")
    return joined.group_by("team").agg(pl.col(_TEAM_SOS_COLUMN).mean().alias(_TEAM_SOS_COLUMN))


def _build_qb_faced_overall_quality(
    qb_games: pl.DataFrame,
    team_ratings: pl.DataFrame,
) -> pl.DataFrame:
    """Return each QB's equal-game mean opponent SaCR over the games he played."""
    if qb_games.is_empty() or team_ratings.is_empty() or "opponent_team" not in qb_games.columns:
        return pl.DataFrame(schema={_QB_FACED_OVERALL_QUALITY_COLUMN: pl.Float64})

    joined = _normalize_team_abbreviations(qb_games, ["opponent_team"]).join(
        _normalize_team_abbreviations(team_ratings, ["team"])
        .select(["team", "SaCR"])
        .rename({"team": "opponent_team", "SaCR": _QB_FACED_OVERALL_QUALITY_COLUMN}),
        on="opponent_team",
        how="left",
    )
    _raise_on_unmatched_opponents(
        joined,
        _QB_FACED_OVERALL_QUALITY_COLUMN,
        "QB overall-opponent schedule values",
    )
    group_keys = _qb_schedule_group_keys(joined)
    if not group_keys:
        return pl.DataFrame(schema={_QB_FACED_OVERALL_QUALITY_COLUMN: pl.Float64})
    return joined.group_by(group_keys).agg(
        pl.col(_QB_FACED_OVERALL_QUALITY_COLUMN).mean().alias(_QB_FACED_OVERALL_QUALITY_COLUMN)
    )


def _build_team_game_logs(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return additive team game logs for the UI contract."""
    ordered_columns = [
        column
        for column in ("game_id", "week", "team", "opponent_team")
        if column in weekly_df.columns
    ]
    remaining_columns = [
        column
        for column in weekly_df.columns
        if column not in set(ordered_columns) and column not in {"season", "season_type"}
    ]
    sort_columns = [column for column in ("team", "week", "game_id") if column in weekly_df.columns]
    return weekly_df.select(ordered_columns + remaining_columns).sort(sort_columns)


def _build_qb_game_logs(qb_df: pl.DataFrame, weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return additive QB game logs enriched with opponent and game-result context."""
    context_columns = [
        column
        for column in (
            "team",
            "week",
            "opponent_team",
            "points_for",
            "points_allowed",
            "point_margin",
            "win_value",
            "turnover_margin",
        )
        if column in weekly_df.columns
    ]
    weekly_context = weekly_df.select(context_columns)
    if "team" in weekly_context.columns:
        weekly_context = weekly_context.rename({"team": "team_abbr"})

    join_keys = [
        column
        for column in ("team_abbr", "week")
        if column in qb_df.columns and column in weekly_context.columns
    ]
    qb_game_logs = qb_df.join(weekly_context, on=join_keys, how="left") if join_keys else qb_df
    if "team_abbr" in qb_game_logs.columns:
        qb_game_logs = qb_game_logs.rename({"team_abbr": "team"})

    ordered_columns = [
        column
        for column in ("game_id", "week", "team", "opponent_team", "qb_id", "qb_name")
        if column in qb_game_logs.columns
    ]
    remaining_columns = [
        column
        for column in qb_game_logs.columns
        if column not in set(ordered_columns)
        and column not in {"season", "season_type", "snap_player_id"}
    ]
    sort_columns = [
        column
        for column in ("team", "week", "game_id", "qb_name")
        if column in qb_game_logs.columns
    ]
    return qb_game_logs.select(ordered_columns + remaining_columns).sort(sort_columns)


def _write_data_file(frame: pl.DataFrame, season: int, suffix: str) -> Path:
    """Validate columns against the metric registry and write one Parquet data file."""
    unknown = get_registry().validate_columns(frame.columns)
    if unknown:
        raise ValueError(
            f"Output {season}_{suffix} contains columns missing from the metric registry: "
            + ", ".join(unknown)
        )
    data_path = Path(DATA_DIR) / f"{season}_{suffix}.parquet"
    frame.write_parquet(data_path)
    print(f"Saved {suffix} to {data_path}")
    return data_path


def _write_qb_output_files(
    qb_combined: pl.DataFrame,
    qb_ratings_df: pl.DataFrame,
    season: int,
) -> None:
    """Write the QB season table and compact ratings summary outputs."""
    _write_data_file(qb_combined, season, "qb_combined")

    qb_summary_cols = [
        col
        for col in [
            "qb_id",
            "qb_name",
            "team",
            "qb_games_played",
            "qb_attempts_total",
            "qb_is_eligible",
            "qb_win_pct",
            _QB_FACED_OVERALL_QUALITY_COLUMN,
            _QB_FACED_RUSH_DEFENSE_COLUMN,
            _QB_ADJUSTED_DESIGNED_RUSH_COLUMN,
        ]
        if col in qb_combined.columns
    ]
    qb_summary_join_keys = [
        key
        for key in ("qb_id", "qb_name", "team")
        if key in qb_ratings_df.columns and key in qb_summary_cols
    ]
    if not qb_summary_join_keys and "team" in qb_ratings_df.columns and "team" in qb_summary_cols:
        qb_summary_join_keys = ["team"]

    qb_ratings_summary = qb_ratings_df.join(
        qb_combined.select(qb_summary_cols),
        on=qb_summary_join_keys,
        how="left",
    )
    _write_data_file(qb_ratings_summary, season, "qb_ratings")


def run_season(season: int) -> None:
    """Run the full NFL strength-of-schedule analysis pipeline for one season."""
    print(f"=== NFL Strength of Schedule -- {season} Season ===\n")

    # --- Load data ---
    print("Loading weekly team stats...")
    weekly_df = load_weekly_team_stats(season)
    print(f"  {weekly_df.height} team-game rows loaded.")

    print("Loading schedule...")
    schedule_df = load_schedule(season)
    print(f"  {schedule_df.height} games loaded.")

    print("Loading QB game stats...")
    qb_df = load_qb_stats(season)
    print(f"  {qb_df.height} QB-game rows loaded.\n")

    # --- Compute team per-game stats ---
    print("Computing per-game team stats...")
    team_per_game = compute_all_teams_per_game(weekly_df)
    print(f"  {team_per_game.height} teams computed.")

    print("Computing per-game QB stats...")
    qb_per_game = compute_all_teams_qb_per_game(qb_df)
    print(f"  {qb_per_game.height} teams computed.\n")

    print("Computing QB season summary stats...")
    qb_season_stats = compute_qb_season_stats(qb_df, weekly_df=weekly_df)
    print(f"  {qb_season_stats.height} QBs computed.\n")

    win_totals = compute_win_totals(weekly_df)
    qb_season_stats = qb_season_stats.join(
        win_totals.select(["team", "wins", "losses", "ties", "win_pct"]),
        on="team",
        how="left",
    )

    print("Computing QB opponent profiles...")
    qb_opp_profiles, qb_opp_details = compute_qb_opponent_profiles(
        weekly_df, qb_df, schedule_df, qb_season_stats
    )
    if qb_opp_profiles is None:
        print("  WARNING: No QB opponent profiles were computed.")
    else:
        print(f"  {qb_opp_profiles.height} QB profiles computed.")
    print()

    # --- Compute opponent profiles ---
    print("Computing opponent profiles (this may take a moment)...")
    opp_team_df, opp_qb_df, opp_details = compute_all_opponent_profiles(
        weekly_df, qb_df, schedule_df
    )
    print()

    # --- Merge and save ---
    os.makedirs(DATA_DIR, exist_ok=True)

    team_game_logs = _build_team_game_logs(weekly_df)
    _write_data_file(team_game_logs, season, "team_game_logs")

    qb_game_logs = _build_qb_game_logs(qb_df, weekly_df)
    _write_data_file(qb_game_logs, season, "qb_game_logs")

    # Team per-game stats (team + QB combined), with win totals
    team_combined = team_per_game.join(qb_per_game, on="team", how="left").join(
        win_totals, on="team", how="left"
    )
    _write_data_file(team_combined, season, "team_per_game_stats")

    _write_data_file(qb_season_stats, season, "qb_per_game_stats")

    if qb_opp_profiles is not None:
        _write_data_file(qb_opp_profiles, season, "qb_opponent_profiles")

    qb_combined = _build_qb_combined(qb_season_stats, qb_opp_profiles)

    qb_adjustment_games = qb_df.join(
        weekly_df.select(["team", "week", "opponent_team"]),
        left_on=["team_abbr", "week"],
        right_on=["team", "week"],
        how="left",
    )
    qb_adjusted_df, qb_defense_adjustments = compute_qb_adjusted_stats(
        qb_adjustment_games,
        response_cols=_QB_SIMULTANEOUS_COLS,
    )
    qb_identity = qb_season_stats.select(
        [col for col in ("qb_id", "qb_name", "team") if col in qb_season_stats.columns]
    )
    qb_adjusted_output = qb_adjusted_df.join(qb_identity, on="qb_id", how="left")
    qb_faced_defense = _build_qb_faced_defense_adjustments(
        qb_adjustment_games,
        qb_defense_adjustments,
    )
    qb_schedule_join_keys = _matching_qb_join_keys(qb_adjusted_output, qb_faced_defense)
    if qb_schedule_join_keys:
        qb_adjusted_output = qb_adjusted_output.join(
            qb_faced_defense,
            on=qb_schedule_join_keys,
            how="left",
        )
    _write_data_file(qb_adjusted_output, season, "simultaneous_qb_adjustments")
    qb_combined = qb_combined.join(
        qb_adjusted_output,
        on=[
            key
            for key in ("qb_id", "qb_name", "team")
            if key in qb_adjusted_output.columns and key in qb_combined.columns
        ],
        how="left",
    )

    qb_ratings_df = compute_qb_ratings(qb_combined)

    qb_ratings_join_keys = [
        key
        for key in ("qb_id", "qb_name", "team")
        if key in qb_ratings_df.columns and key in qb_combined.columns
    ]
    if (
        not qb_ratings_join_keys
        and "team" in qb_ratings_df.columns
        and "team" in qb_combined.columns
    ):
        qb_ratings_join_keys = ["team"]
    qb_combined = qb_combined.join(qb_ratings_df, on=qb_ratings_join_keys, how="left")
    _write_qb_output_files(qb_combined, qb_ratings_df, season)

    # Opponent profiles (team + QB combined)
    if opp_team_df is None and opp_qb_df is None:
        print("WARNING: No opponent profile data was computed.")
        return

    if opp_team_df is not None and opp_qb_df is not None:
        opp_combined = opp_team_df.join(opp_qb_df, on="team", how="left")
    elif opp_team_df is not None:
        opp_combined = opp_team_df
    else:
        if opp_qb_df is None:
            print("WARNING: No opponent profile data was computed.")
            return
        opp_combined = opp_qb_df

    _write_data_file(opp_combined, season, "opponent_profiles")

    # Combined: team stats + opponent stats side by side
    opp_renamed = opp_combined.rename({c: f"opp_{c}" for c in opp_combined.columns if c != "team"})
    combined = team_combined.join(opp_renamed, on="team", how="left")

    pbp_df = load_pbp_data(season)
    play_rows = build_play_level_team_frame_from_pbp(pbp_df)
    if play_rows.is_empty():
        team_adjusted_df = compute_team_adjusted_stats(
            weekly_df,
            response_cols=_TEAM_SIMULTANEOUS_COLS,
        )
    else:
        play_cutoff_week = int(play_rows.select(pl.col("week").max()).item()) + 1
        team_adjusted_df = build_play_level_team_adjusted_snapshot(
            play_rows,
            cutoff_week=play_cutoff_week,
        )
        st_game_rows = build_special_teams_game_frame_from_pbp(pbp_df)
        st_rating_df = build_special_teams_rating_snapshot(
            st_game_rows,
            cutoff_week=play_cutoff_week,
        )
        if "st_rating" in team_adjusted_df.columns:
            team_adjusted_df = team_adjusted_df.drop("st_rating")
        team_adjusted_df = team_adjusted_df.join(st_rating_df, on="team", how="left").with_columns(
            pl.col("st_rating").fill_null(0.0)
        )
    _write_data_file(team_adjusted_df, season, "simultaneous_team_adjustments")
    combined = combined.join(team_adjusted_df, on="team", how="left")

    # Add diff columns: for every paired (stat, opp_stat), compute diff = stat - opp_stat
    diff_exprs = [
        (pl.col(col) - pl.col(f"opp_{col}")).alias(f"diff_{col}")
        for col in combined.columns
        if f"opp_{col}" in combined.columns
    ]
    if diff_exprs:
        combined = combined.with_columns(diff_exprs)

    # Schedule-adjusted ratings (SaOR, SaDR, SaCR)
    ratings_df = compute_ratings(combined)
    combined = combined.join(ratings_df, on="team", how="left")

    team_sos = _build_team_schedule_strength(weekly_df, ratings_df)
    combined = combined.join(team_sos, on="team", how="left")

    qb_faced_rush_defense = _build_qb_faced_rush_defense_adjustments(qb_game_logs, team_adjusted_df)
    qb_rush_join_keys = _matching_qb_join_keys(qb_combined, qb_faced_rush_defense)
    if qb_rush_join_keys:
        qb_adjusted_designed_rush = _build_qb_adjusted_designed_rush_surface(
            qb_combined,
            qb_faced_rush_defense,
        )
        qb_combined = qb_combined.join(
            qb_faced_rush_defense,
            on=qb_rush_join_keys,
            how="left",
        )
        qb_adjusted_rush_join_keys = _matching_qb_join_keys(qb_combined, qb_adjusted_designed_rush)
        if qb_adjusted_rush_join_keys:
            qb_combined = qb_combined.join(
                qb_adjusted_designed_rush,
                on=qb_adjusted_rush_join_keys,
                how="left",
            )

    qb_faced_overall_quality = _build_qb_faced_overall_quality(qb_game_logs, ratings_df)
    qb_overall_join_keys = _matching_qb_join_keys(qb_combined, qb_faced_overall_quality)
    if qb_overall_join_keys:
        qb_combined = qb_combined.join(
            qb_faced_overall_quality,
            on=qb_overall_join_keys,
            how="left",
        )

    srs_df = solve_srs(weekly_df, response_col="point_margin").rename({"srs_rating": "SRS"})
    combined = combined.join(srs_df, on="team", how="left")

    if "st_rating" in combined.columns:
        combined = combined.drop("st_rating")

    _write_data_file(combined, season, "combined")

    # Standalone ratings summary
    ratings_summary = ratings_df.join(
        combined.select(["team", "games_played", "SRS", _TEAM_SOS_COLUMN]), on="team", how="left"
    ).select(
        ["team", "games_played", "SaCR", _TEAM_SOS_COLUMN, "SaOR", "SaDR", "SaSTR", "SaOvR", "SRS"]
    )
    _write_data_file(ratings_summary, season, "ratings")

    _write_qb_output_files(qb_combined, qb_ratings_df, season)

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY -- {season} NFL Strength of Schedule")
    print(f"{'=' * 70}\n")

    # Show key comparison columns: team offense vs opponent offense
    summary_cols = ["team", "games_played"]
    for prefix in ["", "opp_"]:
        for stat in [
            "points_for",
            "points_allowed",
            "total_yards",
            "passing_yards",
            "rushing_yards",
            "passing_epa",
            "rushing_epa",
        ]:
            col = f"{prefix}{stat}"
            if col in combined.columns:
                summary_cols.append(col)

    available_cols = [c for c in summary_cols if c in combined.columns]
    with pl.Config(tbl_cols=-1, tbl_rows=32, fmt_float="mixed", float_precision=2):
        print(combined.select(available_cols).sort("team"))

    # Schedule-adjusted ratings table (sorted by SaCR descending)
    print(f"\n{'=' * 50}")
    print("SCHEDULE-ADJUSTED RATINGS (SaCR rank)")
    print(f"{'=' * 50}")
    print(
        "  SaCR = Composite  |  SaOR = Offense  |  SaDR = Defense  |  "
        "SaSTR = Special Teams  |  SaOvR = Overall"
    )
    print("  (z-scores: 0 = league avg, +1 = 1 SD above avg)\n")
    with pl.Config(tbl_cols=-1, tbl_rows=32, fmt_float="mixed", float_precision=3):
        print(ratings_summary.sort("SaCR", descending=True))

    print("\nOpponent detail sample (DEN):")
    if "DEN" in opp_details:
        for d in opp_details["DEN"]:
            div_marker = " (DIV)" if d["division"] else ""
            print(f"  {d['opponent']}{div_marker}: {d['games_included']} games")

    print("\nQB opponent detail sample (DEN):")
    qb_sample_key = None
    if not qb_season_stats.is_empty() and "team" in qb_season_stats.columns:
        den_qbs = qb_season_stats.filter(pl.col("team") == "DEN")
        if not den_qbs.is_empty():
            for qb_row in den_qbs.to_dicts():
                for key in ("qb_id", "qb_name"):
                    candidate = qb_row.get(key)
                    if candidate in qb_opp_details:
                        qb_sample_key = str(candidate)
                        break
                if qb_sample_key is not None:
                    break
    if qb_sample_key is not None:
        for d in qb_opp_details[qb_sample_key]:
            div_marker = " (DIV)" if d["division"] else ""
            print(f"  {d['opponent']}{div_marker}: {d['games_included']} games")

    print(f"\nDone! Parquet files saved to {DATA_DIR}/")


def main() -> None:
    """Run the full NFL strength-of-schedule analysis pipeline for the configured season."""
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    run_season(SEASON)


if __name__ == "__main__":
    main()
