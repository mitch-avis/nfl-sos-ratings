"""Single-season pipeline: load one season, compute team and QB ratings, write Parquet outputs.

Team ratings come from ``team_rating`` (points per game, opponent-adjusted EPA) and QB ratings
from ``qb_rating`` (adjusted EPA per dropback), each with a week-by-week rating history.
Head-to-head-excluded opponent profiles are written beside them for the analyst UI's descriptive
views, and each game's plays and EPA split into win-probability bins (``wp_bins``) for the
garbage-time filter.
"""

import argparse
import io
import sys
from pathlib import Path

import polars as pl

from nfl_sos_ratings.config import DATA_DIR, SEASON
from nfl_sos_ratings.data_loader import (
    PBP_START_SEASON,
    load_qb_stats,
    load_schedule,
    load_weekly_team_stats,
    load_wp_bins,
    use_disk_cache_unless_configured,
)
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.opponent_stats import compute_all_opponent_profiles
from nfl_sos_ratings.qb_opponent_stats import compute_qb_opponent_profiles
from nfl_sos_ratings.qb_rating import (
    QbRatingFit,
    bootstrap_qb_ratings,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
    fit_qb_ratings_by_week,
)
from nfl_sos_ratings.qb_stats import compute_qb_season_stats
from nfl_sos_ratings.rating_ranges import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    QB_PAIR_COLUMNS,
    QB_RANGE_COLUMNS,
    TEAM_PAIR_COLUMNS,
    TEAM_RANGE_COLUMNS,
    summarize_rank_pairs,
    summarize_rank_ranges,
    summarize_unit_rank_ranges,
)
from nfl_sos_ratings.row_order import data_file_row_order
from nfl_sos_ratings.srs import solve_srs
from nfl_sos_ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    TeamRatingFit,
    bootstrap_team_ratings,
    compute_team_schedule_strength,
    fit_team_ratings,
    fit_team_ratings_by_week,
    fit_team_ratings_with_previous_penalties,
)
from nfl_sos_ratings.team_stats import compute_all_teams_per_game, compute_win_totals

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
    """Validate columns against the metric registry and write one Parquet data file.

    Rows are written in the file's fixed order (``data_file_row_order``).
    """
    unknown = get_registry().validate_columns(frame.columns)
    if unknown:
        msg = (
            f"Output {season}_{suffix} contains columns missing from the metric registry: "
            + ", ".join(unknown)
        )
        raise ValueError(msg)
    order = data_file_row_order(suffix, frame.columns)
    data_path = Path(DATA_DIR) / f"{season}_{suffix}.parquet"
    frame.sort(
        [column for column, _ in order],
        descending=[descending for _, descending in order],
        nulls_last=True,
    ).write_parquet(data_path)
    print(f"Saved {suffix} to {data_path}")
    return data_path


def played_schedule(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Return only the games that have final scores, so in-progress seasons skip future games.

    A schedule without score columns is returned unchanged.
    """
    if not {"home_score", "away_score"} <= set(schedule_df.columns):
        return schedule_df
    return schedule_df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )


def build_team_ratings(weekly_df: pl.DataFrame, fit: TeamRatingFit | None = None) -> pl.DataFrame:
    """Return one row per team with the published ratings, schedule strength, and SRS.

    ``fit`` is the season fit of ``weekly_df`` when the caller already has it.
    """
    fit = fit if fit is not None else fit_team_ratings(weekly_df)
    games_played = weekly_df.group_by("team").len("games_played")
    return (
        fit.ratings.join(compute_team_schedule_strength(weekly_df, fit), on="team")
        .join(solve_srs(weekly_df, response_col="point_margin"), on="team")
        .rename({"srs_rating": "SRS"})
        .join(games_played, on="team")
        .select(TEAM_RATINGS_ORDER)
        .sort("team_rating", descending=True)
    )


def build_qb_ratings(qb_game_logs: pl.DataFrame, fit: QbRatingFit | None = None) -> pl.DataFrame:
    """Return one row per passer with adjusted EPA per dropback and faced pass defense.

    ``fit`` is the season fit of ``qb_game_logs`` when the caller already has it.
    """
    fit = fit if fit is not None else fit_qb_ratings(qb_game_logs)
    return fit.ratings.join(compute_qb_faced_pass_defense(qb_game_logs, fit), on="qb_id")


def build_team_rank_summaries(
    weekly_df: pl.DataFrame, fit: TeamRatingFit
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return every team's rank ranges and every ordered pair's head-to-head chances.

    Both come from one set of game-bootstrap resamples: each redraws the season's games with
    replacement and refits with ``fit``'s penalties; ranks are among all teams in the resample
    (see ``rating_ranges``). The ranges cover the team rating and each unit rating.
    """
    draws = bootstrap_team_ratings(
        weekly_df, fit, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    ranges = summarize_rank_ranges(draws, fit.ratings, TEAM_RANGE_COLUMNS).join(
        summarize_unit_rank_ranges(draws, fit.ratings), on="team", how="left", maintain_order="left"
    )
    return ranges, summarize_rank_pairs(draws, fit.ratings, TEAM_PAIR_COLUMNS)


def build_qb_rank_summaries(
    qb_game_logs: pl.DataFrame, fit: QbRatingFit, qb_combined: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the eligible passers' rank ranges and their head-to-head chances.

    Both come from one set of game-bootstrap resamples. Ranks and pairs are among the passers
    ``qb_combined`` flags ``qb_is_eligible`` for the full season, and each range row carries the
    passer's name and primary team for display.
    """
    eligible = qb_combined.filter(pl.col("qb_is_eligible"))
    eligible_ids = eligible.get_column("qb_id").to_list()
    draws = bootstrap_qb_ratings(
        qb_game_logs, fit, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    ranges = summarize_rank_ranges(draws, fit.ratings, QB_RANGE_COLUMNS, eligible=eligible_ids)
    identity = eligible.select(
        [key for key in _QB_IDENTITY_KEYS if key in eligible.columns]
    ).unique("qb_id", keep="first")
    ranges = ranges.join(identity, on="qb_id", how="left", maintain_order="left").select(
        *identity.columns, pl.exclude(identity.columns)
    )
    return ranges, summarize_rank_pairs(draws, fit.ratings, QB_PAIR_COLUMNS, eligible=eligible_ids)


def _previous_season_fit(season: int) -> TeamRatingFit | None:
    """Return the previous season's full-season team fit, whose penalties this season reuses.

    The previous season's game logs come from ``DATA_DIR`` when built, otherwise from nflverse.
    The first play-by-play season has no previous season and returns ``None``.
    """
    if season <= PBP_START_SEASON:
        return None
    path = Path(DATA_DIR) / f"{season - 1}_team_game_logs.parquet"
    if path.exists():
        previous_logs = pl.read_parquet(path)
    else:
        print(f"Loading {season - 1} team stats for the ridge penalties...")
        previous_logs = _build_team_game_logs(load_weekly_team_stats(season - 1))
    return fit_team_ratings(previous_logs)


def _load_season_frames(season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load one season's team game rows, schedule, and QB game rows."""
    print("Loading weekly team stats...")
    weekly_df = load_weekly_team_stats(season)
    print(f"  {weekly_df.height} team-game rows loaded.")
    print("Loading schedule...")
    schedule_df = played_schedule(load_schedule(season))
    print(f"  {schedule_df.height} played games loaded.")
    print("Loading QB game stats...")
    qb_df = load_qb_stats(season)
    print(f"  {qb_df.height} QB-game rows loaded.\n")
    return weekly_df, schedule_df, qb_df


def _write_team_rating_outputs(
    season: int, weekly_df: pl.DataFrame, team_combined: pl.DataFrame
) -> pl.DataFrame:
    """Fit the season's team ratings, write every team rating output, and return the ratings."""
    print("Fitting team ratings...")
    team_fit = fit_team_ratings_with_previous_penalties(weekly_df, _previous_season_fit(season))
    ratings = build_team_ratings(weekly_df, team_fit)
    _write_data_file(ratings, season, "ratings")
    _write_data_file(
        team_combined.join(ratings.drop("games_played"), on="team", how="left"),
        season,
        "combined",
    )
    _write_data_file(fit_team_ratings_by_week(weekly_df, team_fit), season, "ratings_by_week")
    print(f"Resampling games {BOOTSTRAP_RESAMPLES} times for team rank ranges and pairs...")
    team_ranges, team_pairs = build_team_rank_summaries(weekly_df, team_fit)
    _write_data_file(team_ranges, season, "rating_ranges")
    _write_data_file(team_pairs, season, "rating_pairs")
    return ratings


def _write_qb_rating_outputs(
    season: int,
    qb_game_logs: pl.DataFrame,
    qb_season_stats: pl.DataFrame,
    qb_opp_profiles: pl.DataFrame | None,
) -> None:
    """Fit the season's QB ratings and write every QB rating output."""
    print("Fitting QB ratings...")
    qb_combined = qb_season_stats
    if qb_opp_profiles is not None:
        qb_combined = qb_combined.join(
            qb_opp_profiles, on=_matching_qb_join_keys(qb_combined, qb_opp_profiles), how="left"
        )
    qb_fit = fit_qb_ratings(qb_game_logs)
    qb_combined = qb_combined.join(build_qb_ratings(qb_game_logs, qb_fit), on="qb_id", how="left")
    _write_data_file(qb_combined, season, "qb_combined")
    _write_data_file(
        qb_combined.filter(pl.col("qb_is_eligible"))
        .select([column for column in QB_RATINGS_ORDER if column in qb_combined.columns])
        .sort("adj_qb_epa_per_dropback", descending=True),
        season,
        "qb_ratings",
    )
    _write_data_file(fit_qb_ratings_by_week(qb_game_logs, qb_fit), season, "qb_ratings_by_week")
    print(f"Resampling games {BOOTSTRAP_RESAMPLES} times for QB rank ranges and pairs...")
    qb_ranges, qb_pairs = build_qb_rank_summaries(qb_game_logs, qb_fit, qb_combined)
    _write_data_file(qb_ranges, season, "qb_rating_ranges")
    _write_data_file(qb_pairs, season, "qb_rating_pairs")


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
    team_combined = compute_all_teams_per_game(weekly_df).join(win_totals, on="team", how="left")
    _write_data_file(team_combined, season, "team_per_game_stats")

    qb_season_stats = compute_qb_season_stats(qb_df, weekly_df=weekly_df).join(
        win_totals.select(["team", "wins", "losses", "ties", "win_pct"]), on="team", how="left"
    )
    _write_data_file(qb_season_stats, season, "qb_per_game_stats")

    print("Computing QB opponent profiles...")
    qb_opp_profiles, _ = compute_qb_opponent_profiles(weekly_df, qb_df, qb_season_stats)
    if qb_opp_profiles is not None:
        _write_data_file(qb_opp_profiles, season, "qb_opponent_profiles")

    print("Computing team opponent profiles...")
    opp_profiles, _ = compute_all_opponent_profiles(weekly_df, schedule_df)
    if opp_profiles is not None:
        _write_data_file(opp_profiles, season, "opponent_profiles")
        team_combined = team_combined.join(
            opp_profiles.rename({c: f"opp_{c}" for c in opp_profiles.columns if c != "team"}),
            on="team",
            how="left",
        )

    ratings = _write_team_rating_outputs(season, weekly_df, team_combined)
    _write_qb_rating_outputs(season, qb_game_logs, qb_season_stats, qb_opp_profiles)

    print("Binning plays by win probability...")
    team_wp_bins, qb_wp_bins = load_wp_bins(season)
    _write_data_file(team_wp_bins, season, "team_wp_bins")
    _write_data_file(qb_wp_bins, season, "qb_wp_bins")

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
    use_disk_cache_unless_configured()
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    run_season(args.season)


if __name__ == "__main__":
    main()
