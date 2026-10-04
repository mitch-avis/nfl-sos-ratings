"""Single-season pipeline: load one season, compute team and QB ratings, write Parquet outputs.

Team ratings come from ``team_rating`` (points per game, opponent-adjusted EPA) and QB ratings
from ``qb_rating`` (adjusted EPA per dropback). Head-to-head-excluded opponent profiles and the
``diff_*`` columns are written beside them for the analyst UI's descriptive views.
"""

import argparse
import io
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from nfl_sos_ratings.config import DATA_DIR, SEASON
from nfl_sos_ratings.data_loader import load_qb_stats, load_schedule, load_weekly_team_stats
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.opponent_stats import compute_all_opponent_profiles
from nfl_sos_ratings.qb_opponent_stats import compute_qb_opponent_profiles
from nfl_sos_ratings.qb_rating import compute_qb_faced_pass_defense, fit_qb_ratings
from nfl_sos_ratings.qb_stats import compute_qb_season_stats
from nfl_sos_ratings.srs import solve_srs
from nfl_sos_ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    compute_team_schedule_strength,
    fit_team_ratings,
)
from nfl_sos_ratings.team_stats import (
    compute_all_teams_per_game,
    compute_all_teams_qb_per_game,
    compute_win_totals,
)

TEAM_RATINGS_ORDER = ("team", "games_played", *TEAM_RATING_COLUMNS, "sos", "SRS")
QB_RATINGS_ORDER = (
    "qb_id",
    "qb_name",
    "team",
    "qb_games_played",
    "qb_dropbacks_total",
    "qb_attempts_total",
    "qb_epa_per_dropback",
    "qb_faced_pass_defense",
    "adj_qb_epa_per_dropback",
)
_QB_IDENTITY_KEYS = ("qb_id", "qb_name", "team")


def _matching_qb_join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    """Return the QB identity keys both frames share."""
    return [key for key in _QB_IDENTITY_KEYS if key in left.columns and key in right.columns]


def _with_qb_differentials(qb_combined: pl.DataFrame) -> pl.DataFrame:
    """Add ``diff_<stat> = <stat> - qopp_<stat>`` for every paired QB stat."""
    exprs = [
        (pl.col(column) - pl.col(f"qopp_{column}")).alias(f"diff_{column}")
        for column in qb_combined.columns
        if column.startswith("qb_") and f"qopp_{column}" in qb_combined.columns
    ]
    return qb_combined.with_columns(exprs) if exprs else qb_combined


def _with_diff_columns(combined: pl.DataFrame) -> pl.DataFrame:
    """Add ``diff_<stat> = <stat> - opp_<stat>`` for every paired team stat."""
    exprs = [
        (pl.col(column) - pl.col(f"opp_{column}")).alias(f"diff_{column}")
        for column in combined.columns
        if f"opp_{column}" in combined.columns
    ]
    return combined.with_columns(exprs) if exprs else combined


def _build_team_game_logs(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return team game logs with identity columns first, sorted by team and week."""
    leading = [c for c in ("game_id", "week", "team", "opponent_team") if c in weekly_df.columns]
    rest = [c for c in weekly_df.columns if c not in leading and c not in {"season", "season_type"}]
    return weekly_df.select(leading + rest).sort(
        [c for c in ("team", "week", "game_id") if c in weekly_df.columns]
    )


def _build_qb_game_logs(qb_df: pl.DataFrame, weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return QB game logs enriched with opponent, venue, and game-result context."""
    context = weekly_df.select(
        [
            column
            for column in (
                "team",
                "week",
                "opponent_team",
                "is_home",
                "points_for",
                "points_allowed",
                "point_margin",
                "win_value",
                "turnover_margin",
            )
            if column in weekly_df.columns
        ]
    ).rename({"team": "team_abbr"})
    qb_game_logs = qb_df.join(context, on=["team_abbr", "week"], how="left").rename(
        {"team_abbr": "team"}
    )
    leading = [
        c
        for c in ("game_id", "week", "team", "opponent_team", "qb_id", "qb_name")
        if c in qb_game_logs.columns
    ]
    rest = [
        c
        for c in qb_game_logs.columns
        if c not in leading and c not in {"season", "season_type", "snap_player_id"}
    ]
    return qb_game_logs.select(leading + rest).sort(["team", "week", "game_id", "qb_name"])


def _write_data_file(frame: pl.DataFrame, season: int, suffix: str) -> Path:
    """Validate columns against the metric registry and write one Parquet data file."""
    unknown = get_registry().validate_columns(frame.columns)
    if unknown:
        msg = (
            f"Output {season}_{suffix} contains columns missing from the metric registry: "
            + ", ".join(unknown)
        )
        raise ValueError(msg)
    data_path = Path(DATA_DIR) / f"{season}_{suffix}.parquet"
    frame.write_parquet(data_path)
    print(f"Saved {suffix} to {data_path}")
    return data_path


def build_team_ratings(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per team with the published ratings, schedule strength, and SRS."""
    fit = fit_team_ratings(weekly_df)
    games_played = weekly_df.group_by("team").len("games_played")
    return (
        fit.ratings.join(compute_team_schedule_strength(weekly_df, fit), on="team")
        .join(solve_srs(weekly_df, response_col="point_margin"), on="team")
        .rename({"srs_rating": "SRS"})
        .join(games_played, on="team")
        .select(TEAM_RATINGS_ORDER)
        .sort("team_rating", descending=True)
    )


def build_qb_ratings(qb_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return one row per passer with adjusted EPA per dropback and faced pass defense."""
    fit = fit_qb_ratings(qb_game_logs)
    return fit.ratings.join(compute_qb_faced_pass_defense(qb_game_logs, fit), on="qb_id")


def _load_season_frames(season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load one season's team game rows, schedule, and QB game rows."""
    print("Loading weekly team stats...")
    weekly_df = load_weekly_team_stats(season)
    print(f"  {weekly_df.height} team-game rows loaded.")
    print("Loading schedule...")
    schedule_df = load_schedule(season)
    print(f"  {schedule_df.height} games loaded.")
    print("Loading QB game stats...")
    qb_df = load_qb_stats(season)
    print(f"  {qb_df.height} QB-game rows loaded.\n")
    return weekly_df, schedule_df, qb_df


def run_season(season: int) -> None:
    """Build and write every output for one regular season."""
    print(f"=== NFL Strength of Schedule -- {season} Season ===\n")
    weekly_df, schedule_df, qb_df = _load_season_frames(season)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    team_game_logs = _build_team_game_logs(weekly_df)
    qb_game_logs = _build_qb_game_logs(qb_df, weekly_df)
    _write_data_file(team_game_logs, season, "team_game_logs")
    _write_data_file(qb_game_logs, season, "qb_game_logs")

    win_totals = compute_win_totals(weekly_df)
    team_combined = (
        compute_all_teams_per_game(weekly_df)
        .join(compute_all_teams_qb_per_game(qb_df), on="team", how="left")
        .join(win_totals, on="team", how="left")
    )
    _write_data_file(team_combined, season, "team_per_game_stats")

    qb_season_stats = compute_qb_season_stats(qb_df, weekly_df=weekly_df).join(
        win_totals.select(["team", "wins", "losses", "ties", "win_pct"]), on="team", how="left"
    )
    _write_data_file(qb_season_stats, season, "qb_per_game_stats")

    print("Computing QB opponent profiles...")
    qb_opp_profiles, _ = compute_qb_opponent_profiles(
        weekly_df, qb_df, schedule_df, qb_season_stats
    )
    if qb_opp_profiles is not None:
        _write_data_file(qb_opp_profiles, season, "qb_opponent_profiles")

    print("Computing team opponent profiles...")
    opp_team_df, opp_qb_df, _ = compute_all_opponent_profiles(weekly_df, qb_df, schedule_df)
    opp_combined = (
        opp_team_df.join(opp_qb_df, on="team", how="left")
        if opp_team_df is not None and opp_qb_df is not None
        else opp_team_df
    )
    if opp_combined is not None:
        _write_data_file(opp_combined, season, "opponent_profiles")
        team_combined = _with_diff_columns(
            team_combined.join(
                opp_combined.rename({c: f"opp_{c}" for c in opp_combined.columns if c != "team"}),
                on="team",
                how="left",
            )
        )

    print("Fitting team ratings...")
    ratings = build_team_ratings(weekly_df)
    _write_data_file(ratings, season, "ratings")
    _write_data_file(
        team_combined.join(ratings.drop("games_played"), on="team", how="left"),
        season,
        "combined",
    )

    print("Fitting QB ratings...")
    qb_combined = qb_season_stats
    if qb_opp_profiles is not None:
        qb_combined = _with_qb_differentials(
            qb_combined.join(
                qb_opp_profiles,
                on=_matching_qb_join_keys(qb_combined, qb_opp_profiles),
                how="left",
            )
        )
    qb_combined = qb_combined.join(build_qb_ratings(qb_game_logs), on="qb_id", how="left")
    _write_data_file(qb_combined, season, "qb_combined")
    _write_data_file(
        qb_combined.filter(pl.col("qb_is_eligible"))
        .select([column for column in QB_RATINGS_ORDER if column in qb_combined.columns])
        .sort("adj_qb_epa_per_dropback", descending=True),
        season,
        "qb_ratings",
    )

    with pl.Config(tbl_cols=-1, tbl_rows=40, float_precision=2):
        print(f"\n{season} team ratings (points per game vs an average team):")
        print(ratings)
    print(f"\nDone! Parquet files saved to {DATA_DIR}/")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``season`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings season",
        description="Build one season's Parquet outputs in data/.",
    )
    parser.add_argument(
        "--season", type=int, default=SEASON, help=f"Season to build (default: {SEASON})."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the single-season pipeline."""
    args = _parse_args(argv)
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    run_season(args.season)


if __name__ == "__main__":
    main()
