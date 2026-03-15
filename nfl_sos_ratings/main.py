"""NFL Strength of Schedule -- Main Pipeline.

Loads team and QB data for a given season, computes per-game stats for each
team, then builds opponent strength profiles by averaging each team's opponents'
stats (excluding head-to-head matchups). Outputs CSVs for further analysis.
"""

import io
import os
import sys

import polars as pl
from config import OUTPUT_DIR, SEASON
from data_loader import load_qb_stats, load_schedule, load_weekly_team_stats
from opponent_stats import compute_all_opponent_profiles
from ratings import compute_ratings
from team_stats import (
    compute_all_teams_per_game,
    compute_all_teams_qb_per_game,
    compute_win_totals,
)


def main() -> None:
    """Run the full NFL strength-of-schedule analysis pipeline."""
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(f"=== NFL Strength of Schedule -- {SEASON} Season ===\n")

    # --- Load data ---
    print("Loading weekly team stats...")
    weekly_df = load_weekly_team_stats(SEASON)
    print(f"  {weekly_df.height} team-game rows loaded.")

    print("Loading schedule...")
    schedule_df = load_schedule(SEASON)
    print(f"  {schedule_df.height} games loaded.")

    print("Loading QB Next Gen Stats...")
    qb_df = load_qb_stats(SEASON)
    print(f"  {qb_df.height} QB-game rows loaded.\n")

    # --- Compute team per-game stats ---
    print("Computing per-game team stats...")
    team_per_game = compute_all_teams_per_game(weekly_df)
    print(f"  {team_per_game.height} teams computed.")

    print("Computing per-game QB stats...")
    qb_per_game = compute_all_teams_qb_per_game(qb_df)
    print(f"  {qb_per_game.height} teams computed.\n")

    # --- Compute opponent profiles ---
    print("Computing opponent profiles (this may take a moment)...")
    opp_team_df, opp_qb_df, opp_details = compute_all_opponent_profiles(
        weekly_df, qb_df, schedule_df
    )
    print()

    # --- Merge and save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Team per-game stats (team + QB combined), with win totals
    win_totals = compute_win_totals(weekly_df)
    team_combined = team_per_game.join(qb_per_game, on="team", how="left").join(
        win_totals, on="team", how="left"
    )
    team_csv = os.path.join(OUTPUT_DIR, "team_per_game_stats.csv")
    team_combined.write_csv(team_csv)
    print(f"Saved team per-game stats to {team_csv}")

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

    opp_csv = os.path.join(OUTPUT_DIR, "opponent_profiles.csv")
    opp_combined.write_csv(opp_csv)
    print(f"Saved opponent profiles to {opp_csv}")

    # Combined: team stats + opponent stats side by side
    opp_renamed = opp_combined.rename({c: f"opp_{c}" for c in opp_combined.columns if c != "team"})
    combined = team_combined.join(opp_renamed, on="team", how="left")

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

    combined_csv = os.path.join(OUTPUT_DIR, "combined.csv")
    combined.write_csv(combined_csv)
    print(f"Saved combined stats to {combined_csv}")

    # Standalone ratings summary
    ratings_summary = ratings_df.join(
        combined.select(["team", "games_played"]), on="team", how="left"
    ).select(["team", "games_played", "SaCR", "SaOR", "SaDR"])
    ratings_csv = os.path.join(OUTPUT_DIR, "ratings.csv")
    ratings_summary.write_csv(ratings_csv)
    print(f"Saved schedule-adjusted ratings to {ratings_csv}")

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY -- {SEASON} NFL Strength of Schedule")
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
    print("  SaCR = Composite  |  SaOR = Offense  |  SaDR = Defense")
    print("  (z-scores: 0 = league avg, +1 = 1 SD above avg)\n")
    with pl.Config(tbl_cols=-1, tbl_rows=32, fmt_float="mixed", float_precision=3):
        print(ratings_summary.sort("SaCR", descending=True))

    print("\nOpponent detail sample (DEN):")
    if "DEN" in opp_details:
        for d in opp_details["DEN"]:
            div_marker = " (DIV)" if d["division"] else ""
            print(f"  {d['opponent']}{div_marker}: {d['games_included']} games")

    print(f"\nDone! CSV files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
