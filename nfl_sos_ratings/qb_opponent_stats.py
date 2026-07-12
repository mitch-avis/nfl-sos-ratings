"""Quarterback-specific opponent profile helpers."""

import polars as pl

from nfl_sos_ratings.config import TEAM_ABBR_ALIASES
from nfl_sos_ratings.opponent_stats import is_division_opponent
from nfl_sos_ratings.team_stats import compute_team_stats_excluding_opponent

DEFENSIVE_CONTEXT_COLS: list[str] = [
    "points_allowed",
    "def_sacks",
    "def_interceptions",
    "def_pass_defended",
    "def_tackles_for_loss",
    "def_qb_hits",
]

_DERIVED_QB_RATE_COLS = {
    "qb_yards_per_attempt",
    "qb_touchdown_rate",
    "qb_interception_rate",
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
    "qb_any_a",
}


def _normalize_team_abbreviations(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize known source-specific team abbreviations in selected columns."""
    exprs = [
        pl.col(column).replace(TEAM_ABBR_ALIASES).alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _compute_qb_allowed_stats_excluding_team(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    defense_team: str,
    evaluated_team: str,
) -> pl.DataFrame | None:
    """Compute QB stats allowed by `defense_team`, excluding games versus `evaluated_team`."""
    defense_games = weekly_df.filter(
        (pl.col("opponent_team") == defense_team) & (pl.col("team") != evaluated_team)
    )
    if defense_games.is_empty():
        return None

    qb_allowed = defense_games.join(
        qb_df,
        left_on=["team", "week"],
        right_on=["team_abbr", "week"],
        how="inner",
    )
    if qb_allowed.is_empty():
        return None

    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric() and col != "week" and col not in _DERIVED_QB_RATE_COLS
    ]
    if not qb_stat_cols:
        return None

    agg_exprs = [pl.col(col).mean().alias(f"qopp_{col}") for col in qb_stat_cols]
    rate_inputs = [
        ("qb_pass_yards", "qopp_qb_yards_per_attempt"),
        ("qb_pass_touchdowns", "qopp_qb_touchdown_rate"),
        ("qb_interceptions", "qopp_qb_interception_rate"),
    ]
    if "qb_attempts" in qb_allowed.columns:
        attempts_sum = pl.col("qb_attempts").sum()
        for numerator_col, output_col in rate_inputs:
            if numerator_col in qb_allowed.columns:
                agg_exprs.append(
                    pl.when(attempts_sum > 0)
                    .then(pl.col(numerator_col).sum() / attempts_sum)
                    .otherwise(None)
                    .alias(output_col)
                )
    if "qb_dropbacks" in qb_allowed.columns:
        dropbacks_sum = pl.col("qb_dropbacks").sum()
        if "qb_passing_epa" in qb_allowed.columns:
            agg_exprs.append(
                pl.when(dropbacks_sum > 0)
                .then(pl.col("qb_passing_epa").sum() / dropbacks_sum)
                .otherwise(None)
                .alias("qopp_qb_epa_per_dropback")
            )
        if "qb_pass_yards" in qb_allowed.columns:
            agg_exprs.append(
                pl.when(dropbacks_sum > 0)
                .then(pl.col("qb_pass_yards").sum() / dropbacks_sum)
                .otherwise(None)
                .alias("qopp_qb_pass_yards_per_dropback")
            )
        if "qb_sacks" in qb_allowed.columns:
            agg_exprs.append(
                pl.when(dropbacks_sum > 0)
                .then(pl.col("qb_sacks").sum() / dropbacks_sum)
                .otherwise(None)
                .alias("qopp_qb_sack_rate")
            )
        if {"qb_pass_touchdowns", "qb_interceptions"}.issubset(set(qb_allowed.columns)):
            agg_exprs.append(
                pl.when(dropbacks_sum > 0)
                .then(
                    (pl.col("qb_pass_touchdowns").sum() - pl.col("qb_interceptions").sum())
                    / dropbacks_sum
                )
                .otherwise(None)
                .alias("qopp_qb_td_int_margin_rate")
            )
    if {
        "qb_pass_yards",
        "qb_pass_touchdowns",
        "qb_interceptions",
        "qb_sack_yards_lost",
        "qb_sacks",
    }.issubset(set(qb_allowed.columns)):
        denominator = pl.col("qb_attempts").sum() + pl.col("qb_sacks").sum()
        agg_exprs.append(
            pl.when(denominator > 0)
            .then(
                (
                    pl.col("qb_pass_yards").sum()
                    + (20.0 * pl.col("qb_pass_touchdowns").sum())
                    - (45.0 * pl.col("qb_interceptions").sum())
                    - pl.col("qb_sack_yards_lost").sum()
                )
                / denominator
            )
            .otherwise(None)
            .alias("qopp_qb_any_a")
        )

    return qb_allowed.select([pl.lit(defense_team).alias("opponent")] + agg_exprs)


def _select_primary_qb_games(qb_df: pl.DataFrame) -> pl.DataFrame:
    """Return one primary quarterback row per team-week when sufficient keys exist."""
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


def _get_faced_opponents(qb_games: pl.DataFrame) -> list[str]:
    """Return unique faced opponents in week order from a quarterback's actual game rows."""
    if qb_games.is_empty() or "opponent_team" not in qb_games.columns:
        return []
    sort_cols = [col for col in ("week", "opponent_team") if col in qb_games.columns]
    source = qb_games.sort(sort_cols) if sort_cols else qb_games
    opponents = source.select("opponent_team").to_series().to_list()
    return list(dict.fromkeys(str(opponent) for opponent in opponents))


def _details_key(qb_row: dict[str, object], qb_keys: list[str], team_label: str) -> str:
    """Return the details-map key for one quarterback season row."""
    for key in qb_keys:
        value = qb_row.get(key)
        if value is not None:
            return str(value)
    return team_label


def _qb_identity_filter(qb_row: dict[str, object], qb_keys: list[str]) -> pl.Expr:
    """Return the most specific QB identity filter available for one season row."""
    for key in qb_keys:
        value = qb_row.get(key)
        if value is not None:
            return pl.col(key) == pl.lit(value)
    return pl.lit(False)


def compute_qb_opponent_profiles(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    qb_season_df: pl.DataFrame,
    weighted: bool = False,
) -> tuple[pl.DataFrame | None, dict[str, list[dict[str, str | bool | int]]]]:
    """Compute QB opponent profiles for each individual quarterback season row."""
    weekly_df = _normalize_team_abbreviations(weekly_df, ["team", "opponent_team"])
    qb_df = _normalize_team_abbreviations(qb_df, ["team_abbr"])
    schedule_df = _normalize_team_abbreviations(schedule_df, ["home_team", "away_team"])
    qb_season_df = _normalize_team_abbreviations(qb_season_df, ["team"])

    qb_keys = [key for key in ("qb_id", "qb_name") if key in qb_season_df.columns]
    if not qb_keys:
        qb_keys = ["team"]

    details: dict[str, list[dict[str, str | bool | int]]] = {}
    profile_rows: list[pl.DataFrame] = []

    qb_rows = qb_season_df.select(qb_keys + ["team"]).to_dicts()
    primary_qb_df = _select_primary_qb_games(qb_df)

    qb_games_with_opponents = primary_qb_df.join(
        weekly_df.select(["team", "week", "opponent_team"]),
        left_on=["team_abbr", "week"],
        right_on=["team", "week"],
        how="inner",
    )

    for qb_row in qb_rows:
        team_label = str(qb_row.get("team", ""))
        qb_label = _details_key(qb_row, qb_keys, team_label)
        qb_filter = _qb_identity_filter(qb_row, qb_keys)

        qb_games = qb_games_with_opponents.filter(qb_filter)

        opponents = _get_faced_opponents(qb_games)
        if not opponents:
            details[qb_label] = []
            continue

        opp_rows: list[pl.DataFrame] = []
        team_details: list[dict[str, str | bool | int]] = []

        for opponent_value in opponents:
            evaluated_team = team_label
            opponent = str(opponent_value)
            opp_stats = compute_team_stats_excluding_opponent(
                weekly_df,
                opponent,
                evaluated_team,
            )
            games_included = (
                int(opp_stats.select("games_included").item()) if opp_stats is not None else 0
            )

            team_details.append(
                {
                    "opponent": opponent,
                    "division": is_division_opponent(evaluated_team, opponent),
                    "games_included": games_included,
                }
            )

            if opp_stats is None:
                continue

            opp_qb_allowed = _compute_qb_allowed_stats_excluding_team(
                weekly_df,
                primary_qb_df,
                opponent,
                evaluated_team,
            )

            opp_row = opp_stats.rename({"team": "opponent"}).with_columns(
                pl.lit(evaluated_team).alias("team"),
                pl.lit(opponent).alias("opponent"),
            )
            if opp_qb_allowed is not None:
                opp_row = opp_row.join(opp_qb_allowed, on="opponent", how="left")

            opp_rows.append(opp_row)

        details[qb_label] = team_details

        if not opp_rows:
            continue

        combined = pl.concat(opp_rows)
        available_cols = [col for col in DEFENSIVE_CONTEXT_COLS if col in combined.columns]
        available_cols.extend(
            col
            for col in combined.columns
            if col.startswith("qopp_qb_") and col not in available_cols
        )
        if not available_cols:
            continue

        if weighted:
            denominator = pl.col("games_included").sum()
            agg_exprs = [
                ((pl.col(col) * pl.col("games_included")).sum() / denominator).alias(
                    col if col.startswith("qopp_") else f"qopp_{col}"
                )
                for col in available_cols
            ]
        else:
            agg_exprs = [
                pl.col(col).mean().alias(col if col.startswith("qopp_") else f"qopp_{col}")
                for col in available_cols
            ]

        key_exprs = [pl.lit(qb_row[key]).alias(key) for key in qb_keys]
        profile_rows.append(
            combined.select(key_exprs + [pl.lit(team_label).alias("team")] + agg_exprs)
        )

    if not profile_rows:
        return None, details

    sort_keys = [key for key in ["team", *qb_keys] if key in profile_rows[0].columns]
    return pl.concat(profile_rows).sort(sort_keys), details
