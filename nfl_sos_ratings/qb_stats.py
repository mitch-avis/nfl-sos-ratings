"""Quarterback season-level aggregation helpers."""

import polars as pl


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


def compute_qb_season_stats(
    qb_df: pl.DataFrame,
    weekly_df: pl.DataFrame | None = None,
    min_games: int = 8,
    min_attempts: int = 200,
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

    qb_keys = _resolve_qb_keys(qb_df)

    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric() and col not in {"week"}
    ]

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("qb_games_played"),
    ]
    agg_exprs.extend(pl.col(col).mean().alias(col) for col in qb_stat_cols)

    if "qb_attempts" in qb_df.columns:
        agg_exprs.append(pl.col("qb_attempts").sum().cast(pl.Int64).alias("qb_attempts_total"))
    else:
        agg_exprs.append(pl.lit(0).cast(pl.Int64).alias("qb_attempts_total"))

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
        qb_results = (
            qb_df.join(
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
                (
                    (pl.col("qb_wins") + 0.5 * pl.col("qb_ties"))
                    / (pl.col("qb_wins") + pl.col("qb_losses") + pl.col("qb_ties"))
                ).alias("qb_win_pct")
            )
        )
        season_stats = season_stats.join(qb_results, on=qb_keys, how="left")
    else:
        season_stats = season_stats.with_columns(pl.lit(0.5).alias("qb_win_pct"))

    return season_stats.with_columns(
        (
            (pl.col("qb_games_played") >= min_games) & (pl.col("qb_attempts_total") >= min_attempts)
        ).alias("qb_is_eligible")
    ).sort("team")
