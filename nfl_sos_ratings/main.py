"""NFL Strength of Schedule -- Main Pipeline.

Loads team and QB data for a given season, computes per-game stats for each
team, then builds opponent strength profiles by averaging each team's opponents'
stats (excluding head-to-head matchups). Outputs CSVs for further analysis.
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
from nfl_sos_ratings.qb_ratings import calibrate_qb_model, compute_qb_ratings
from nfl_sos_ratings.qb_stats import compute_qb_season_stats
from nfl_sos_ratings.ratings import compute_ratings
from nfl_sos_ratings.team_stats import (
    compute_all_teams_per_game,
    compute_all_teams_qb_per_game,
    compute_win_totals,
)


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

    print("Loading QB Next Gen Stats...")
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
        print(f"  {qb_opp_profiles.height} teams computed.")
    print()

    # --- Compute opponent profiles ---
    print("Computing opponent profiles (this may take a moment)...")
    opp_team_df, opp_qb_df, opp_details = compute_all_opponent_profiles(
        weekly_df, qb_df, schedule_df
    )
    print()

    # --- Merge and save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    calibration_years = [year for year in range(season - 5, season) if year >= 2006]
    historical_qb_frames: list[pl.DataFrame] = []
    for year in calibration_years:
        try:
            historical_weekly = load_weekly_team_stats(year)
            historical_qb_raw = load_qb_stats(year)
            historical_qb = compute_qb_season_stats(
                historical_qb_raw,
                weekly_df=historical_weekly,
            )
            historical_qb = historical_qb.join(
                compute_win_totals(historical_weekly).select(
                    ["team", "wins", "losses", "ties", "win_pct"]
                ),
                on="team",
                how="left",
            )
            historical_opp_profiles, _ = compute_qb_opponent_profiles(
                historical_weekly,
                historical_qb_raw,
                load_schedule(year),
                historical_qb,
            )
            historical_qb_frames.append(_build_qb_combined(historical_qb, historical_opp_profiles))
        except Exception as exc:
            print(f"  WARNING: Skipping QB calibration season {year}: {exc}")

    if historical_qb_frames:
        min_corr, sos_weight, outcome_weight = calibrate_qb_model(
            pl.concat(historical_qb_frames, how="diagonal_relaxed")
        )
        print(
            "Calibrated QB model constants: "
            f"min_correlation={min_corr:.2f}, sos_weight={sos_weight:.2f}, "
            f"outcome_weight={outcome_weight:.2f}"
        )
    else:
        min_corr, sos_weight, outcome_weight = 0.1, 2.0, 0.75
        print("  WARNING: No historical QB seasons available; using default QB constants.")

    try:
        qb_ratings_df = compute_qb_ratings(
            qb_combined,
            min_correlation=min_corr,
            sos_weight=sos_weight,
            outcome_weight=outcome_weight,
        )
    except TypeError:
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

    combined_csv = os.path.join(OUTPUT_DIR, f"{season}_combined.csv")
    combined.write_csv(combined_csv)
    print(f"Saved combined stats to {combined_csv}")

    # Standalone ratings summary
    ratings_summary = ratings_df.join(
        combined.select(["team", "games_played"]), on="team", how="left"
    ).select(["team", "games_played", "SaCR", "SaOR", "SaDR"])
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
    print("  SaCR = Composite  |  SaOR = Offense  |  SaDR = Defense")
    print("  (z-scores: 0 = league avg, +1 = 1 SD above avg)\n")
    with pl.Config(tbl_cols=-1, tbl_rows=32, fmt_float="mixed", float_precision=3):
        print(ratings_summary.sort("SaCR", descending=True))

    print("\nOpponent detail sample (DEN):")
    if "DEN" in opp_details:
        for d in opp_details["DEN"]:
            div_marker = " (DIV)" if d["division"] else ""
            print(f"  {d['opponent']}{div_marker}: {d['games_included']} games")

    print("\nQB opponent detail sample (DEN):")
    if "DEN" in qb_opp_details:
        for d in qb_opp_details["DEN"]:
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
