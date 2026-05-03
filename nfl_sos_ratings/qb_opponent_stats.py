"""Quarterback-specific opponent profile helpers."""

import polars as pl

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
        if dtype.is_numeric() and col != "week"
    ]
    if not qb_stat_cols:
        return None

    return qb_allowed.select(
        [pl.lit(defense_team).alias("opponent")]
        + [pl.col(col).mean().alias(f"qopp_{col}") for col in qb_stat_cols]
    )


def compute_qb_opponent_profiles(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    qb_season_df: pl.DataFrame,
    weighted: bool = False,
) -> tuple[pl.DataFrame | None, dict[str, list[dict[str, str | bool | int]]]]:
    """Compute QB opponent profiles for each individual quarterback season row."""
    qb_keys = [key for key in ("qb_id", "qb_name") if key in qb_season_df.columns]
    if not qb_keys:
        qb_keys = ["team"]

    details: dict[str, list[dict[str, str | bool | int]]] = {}
    profile_rows: list[pl.DataFrame] = []

    qb_rows = qb_season_df.select(qb_keys + ["team"]).to_dicts()

    qb_games_with_opponents = qb_df.join(
        weekly_df.select(["team", "week", "opponent_team"]),
        left_on=["team_abbr", "week"],
        right_on=["team", "week"],
        how="inner",
    )

    for qb_row in qb_rows:
        team_label = str(qb_row.get("team", ""))
        qb_filter = pl.lit(True)
        for key in qb_keys:
            qb_filter = qb_filter & (pl.col(key) == pl.lit(qb_row[key]))

        qb_games = qb_games_with_opponents.filter(qb_filter)
        if qb_games.is_empty() and team_label:
            # Fallback for tests or sparse mocks missing QB identifiers.
            qb_games = qb_games_with_opponents.filter(pl.col("team_abbr") == team_label)

        opponents = (
            qb_games.select(["team_abbr", "opponent_team"]).to_dicts()
            if not qb_games.is_empty()
            else []
        )
        opp_rows: list[pl.DataFrame] = []
        team_details: list[dict[str, str | bool | int]] = []

        for game in opponents:
            evaluated_team = str(game["team_abbr"])
            opponent = str(game["opponent_team"])
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
                qb_df,
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

        details[team_label] = team_details

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
