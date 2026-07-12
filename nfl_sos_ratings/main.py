"""Single-season PBP-first pipeline for team and quarterback ratings.

The pipeline loads PBP-derived team and QB game data, builds one-hop opponent
profiles, computes equal-weight diff-based ratings, and writes simultaneous-
adjustment outputs for side-by-side comparison.
"""

import io
import os
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from nfl_sos_ratings.config import OUTPUT_DIR, SEASON
from nfl_sos_ratings.data_loader import load_qb_stats, load_schedule, load_weekly_team_stats
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

_TEAM_SIMULTANEOUS_COLS = [
    "points_per_offensive_snap",
    "total_yards_per_offensive_snap",
    "passing_yards_per_offensive_snap",
    "rushing_yards_per_offensive_snap",
    "passing_epa_per_offensive_snap",
    "rushing_epa_per_offensive_snap",
    "passing_tds_per_offensive_snap",
    "rushing_tds_per_offensive_snap",
    "passing_first_downs_per_offensive_snap",
    "rushing_first_downs_per_offensive_snap",
    "passing_cpoe",
    "sacks_suffered_per_offensive_snap",
    "passing_interceptions_per_offensive_snap",
    "sack_fumbles_lost_per_offensive_snap",
    "rushing_fumbles_lost_per_offensive_snap",
    "points_allowed_per_defensive_snap",
    "total_yards_allowed_per_defensive_snap",
    "passing_yards_allowed_per_defensive_snap",
    "rushing_yards_allowed_per_defensive_snap",
    "passing_epa_allowed_per_defensive_snap",
    "rushing_epa_allowed_per_defensive_snap",
    "passing_tds_allowed_per_defensive_snap",
    "rushing_tds_allowed_per_defensive_snap",
    "passing_first_downs_allowed_per_defensive_snap",
    "rushing_first_downs_allowed_per_defensive_snap",
    "passing_cpoe_allowed",
    "def_sacks_per_defensive_snap",
    "def_interceptions_per_defensive_snap",
    "def_pass_defended_per_defensive_snap",
    "def_tackles_for_loss_per_defensive_snap",
    "def_qb_hits_per_defensive_snap",
    "def_fumbles_forced_per_defensive_snap",
    "def_safeties_per_defensive_snap",
]

_QB_SIMULTANEOUS_COLS = [
    "qb_epa_per_dropback",
    "qb_any_a",
    "qb_completion_percentage_above_expectation",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
    "qb_pass_yards_per_dropback",
    "qb_passer_rating",
]


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    team_game_logs = _build_team_game_logs(weekly_df)
    team_game_logs_csv = os.path.join(OUTPUT_DIR, f"{season}_team_game_logs.csv")
    team_game_logs.write_csv(team_game_logs_csv)
    print(f"Saved team game logs to {team_game_logs_csv}")

    qb_game_logs = _build_qb_game_logs(qb_df, weekly_df)
    qb_game_logs_csv = os.path.join(OUTPUT_DIR, f"{season}_qb_game_logs.csv")
    qb_game_logs.write_csv(qb_game_logs_csv)
    print(f"Saved QB game logs to {qb_game_logs_csv}")

    # Team per-game stats (team + QB combined), with win totals
    team_combined = team_per_game.join(qb_per_game, on="team", how="left").join(
        win_totals, on="team", how="left"
    )
    team_csv = os.path.join(OUTPUT_DIR, f"{season}_team_per_game_stats.csv")
    team_combined.write_csv(team_csv)
    print(f"Saved team per-game stats to {team_csv}")

    qb_csv = os.path.join(OUTPUT_DIR, f"{season}_qb_per_game_stats.csv")
    qb_season_stats.write_csv(qb_csv)
    print(f"Saved QB per-game stats to {qb_csv}")

    if qb_opp_profiles is not None:
        qb_opp_csv = os.path.join(OUTPUT_DIR, f"{season}_qb_opponent_profiles.csv")
        qb_opp_profiles.write_csv(qb_opp_csv)
        print(f"Saved QB opponent profiles to {qb_opp_csv}")

    qb_combined = _build_qb_combined(qb_season_stats, qb_opp_profiles)

    qb_adjustment_games = qb_df.join(
        weekly_df.select(["team", "week", "opponent_team"]),
        left_on=["team_abbr", "week"],
        right_on=["team", "week"],
        how="left",
    )
    qb_adjusted_df, _ = compute_qb_adjusted_stats(
        qb_adjustment_games,
        response_cols=_QB_SIMULTANEOUS_COLS,
    )
    qb_identity = qb_season_stats.select(
        [col for col in ("qb_id", "qb_name", "team") if col in qb_season_stats.columns]
    )
    qb_adjusted_output = qb_adjusted_df.join(qb_identity, on="qb_id", how="left")
    qb_adjusted_csv = os.path.join(OUTPUT_DIR, f"{season}_simultaneous_qb_adjustments.csv")
    qb_adjusted_output.write_csv(qb_adjusted_csv)
    print(f"Saved simultaneous QB adjustments to {qb_adjusted_csv}")
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

    qb_combined_csv = os.path.join(OUTPUT_DIR, f"{season}_qb_combined.csv")
    qb_combined.write_csv(qb_combined_csv)
    print(f"Saved QB combined stats to {qb_combined_csv}")

    qb_ratings_csv = os.path.join(OUTPUT_DIR, f"{season}_qb_ratings.csv")
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
    qb_ratings_summary.write_csv(qb_ratings_csv)
    print(f"Saved QB schedule-adjusted ratings to {qb_ratings_csv}")

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

    opp_csv = os.path.join(OUTPUT_DIR, f"{season}_opponent_profiles.csv")
    opp_combined.write_csv(opp_csv)
    print(f"Saved opponent profiles to {opp_csv}")

    # Combined: team stats + opponent stats side by side
    opp_renamed = opp_combined.rename({c: f"opp_{c}" for c in opp_combined.columns if c != "team"})
    combined = team_combined.join(opp_renamed, on="team", how="left")

    team_adjusted_df = compute_team_adjusted_stats(
        weekly_df,
        response_cols=_TEAM_SIMULTANEOUS_COLS,
    )
    team_adjusted_csv = os.path.join(OUTPUT_DIR, f"{season}_simultaneous_team_adjustments.csv")
    team_adjusted_df.write_csv(team_adjusted_csv)
    print(f"Saved simultaneous team adjustments to {team_adjusted_csv}")
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

    srs_df = solve_srs(weekly_df, response_col="point_margin").rename({"srs_rating": "SRS"})
    combined = combined.join(srs_df, on="team", how="left")

    combined_csv = os.path.join(OUTPUT_DIR, f"{season}_combined.csv")
    combined.write_csv(combined_csv)
    print(f"Saved combined stats to {combined_csv}")

    # Standalone ratings summary
    ratings_summary = ratings_df.join(
        combined.select(["team", "games_played", "SRS"]), on="team", how="left"
    ).select(["team", "games_played", "SaCR", "SaOR", "SaDR", "SaOvR", "SRS"])
    ratings_csv = os.path.join(OUTPUT_DIR, f"{season}_ratings.csv")
    ratings_summary.write_csv(ratings_csv)
    print(f"Saved schedule-adjusted ratings to {ratings_csv}")

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
    print("  SaCR = Composite  |  SaOR = Offense  |  SaDR = Defense  |  SaOvR = Overall")
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

    print(f"\nDone! CSV files saved to {OUTPUT_DIR}/")


def main() -> None:
    """Run the full NFL strength-of-schedule analysis pipeline for the configured season."""
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    run_season(SEASON)


if __name__ == "__main__":
    main()
