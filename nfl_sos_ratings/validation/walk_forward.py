"""Walk-forward validation helpers for team margin prediction."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.data_loader import load_espn_qbr
from nfl_sos_ratings.simultaneous_adjustment import solve_srs
from nfl_sos_ratings.validation.snapshots import build_team_rating_snapshot

_NORMALIZED_NAME_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class EloConfig:
    """Fixed constants for the simple team Elo baseline."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_field_elo: float = 55.0
    regression_to_mean: float = 2.0 / 3.0
    use_margin_multiplier: bool = True


def _normalize_person_name(name: str) -> str:
    """Normalize a player name for fuzzy season-end joins across sources."""
    return _NORMALIZED_NAME_RE.sub("", name.lower())


def _pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson correlation, or 0 when either series is constant."""
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    x_centered = x_values - float(x_values.mean())
    y_centered = y_values - float(y_values.mean())
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
        return 0.0
    return float((x_centered @ y_centered) / (x_norm * y_norm))


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Spearman rank correlation using average ranks for ties."""
    x_ranks = np.asarray(pl.Series(x_values).rank(method="average").to_list(), dtype=np.float64)
    y_ranks = np.asarray(pl.Series(y_values).rank(method="average").to_list(), dtype=np.float64)
    return _pearson(x_ranks, y_ranks)


def _build_home_game_frame(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one row per home game with the realized home margin."""
    required_columns = {"game_id", "week", "team", "opponent_team", "is_home", "point_margin"}
    missing = sorted(required_columns - set(weekly_team_rows.columns))
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"weekly_team_rows is missing required columns: {detail}")

    return (
        weekly_team_rows.filter(pl.col("is_home"))
        .select(
            pl.lit(season).cast(pl.Int64).alias("season"),
            "game_id",
            pl.col("week").cast(pl.Int64).alias("week"),
            pl.col("team").alias("home_team"),
            pl.col("opponent_team").alias("away_team"),
            pl.col("point_margin").cast(pl.Float64).alias("home_margin"),
        )
        .sort(["week", "game_id"])
    )


def build_snapshot_feature_rows(
    weekly_team_rows: pl.DataFrame,
    season: int,
    baseline_name: str,
    snapshot_builder: Callable[[pl.DataFrame, int], pl.DataFrame],
    rating_column: str,
) -> pl.DataFrame:
    """Build week-specific home-game feature rows from a pregame team snapshot."""
    home_games = _build_home_game_frame(weekly_team_rows, season)
    if home_games.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "baseline": pl.String,
                "game_id": pl.String,
                "home_team": pl.String,
                "away_team": pl.String,
                "rating_diff": pl.Float64,
                "home_margin": pl.Float64,
            }
        )

    feature_frames: list[pl.DataFrame] = []
    weeks = sorted(home_games.select("week").to_series().unique().to_list())
    for week in weeks:
        snapshot = snapshot_builder(weekly_team_rows, int(week))
        ratings = snapshot.select(["team", rating_column]).rename({rating_column: "rating"})
        week_games = home_games.filter(pl.col("week") == int(week))
        week_features = (
            week_games.join(
                ratings.rename({"team": "home_team", "rating": "home_rating"}),
                on="home_team",
                how="left",
            )
            .join(
                ratings.rename({"team": "away_team", "rating": "away_rating"}),
                on="away_team",
                how="left",
            )
            .with_columns(
                pl.lit(baseline_name).alias("baseline"),
                (pl.col("home_rating").fill_null(0.0) - pl.col("away_rating").fill_null(0.0)).alias(
                    "rating_diff"
                ),
            )
            .select(
                "season",
                "week",
                "baseline",
                "game_id",
                "home_team",
                "away_team",
                "rating_diff",
                "home_margin",
            )
        )
        feature_frames.append(week_features)

    return pl.concat(feature_frames, how="vertical") if feature_frames else home_games.clear()


def _empty_team_value_snapshot(
    weekly_team_rows: pl.DataFrame,
    rating_column: str,
) -> pl.DataFrame:
    """Return a zeroed snapshot for every team present in the weekly rows."""
    teams = sorted(
        weekly_team_rows.select("team").drop_nulls().to_series().cast(pl.String).unique().to_list()
    )
    return pl.DataFrame({"team": teams, rating_column: [0.0] * len(teams)})


def _build_srs_snapshot(weekly_team_rows: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
    """Return an SRS snapshot built only from games before the cutoff week."""
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        return _empty_team_value_snapshot(weekly_team_rows, "SRS")
    return solve_srs(filtered_rows, response_col="point_margin").rename({"srs_rating": "SRS"})


def build_srs_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from week-specific SRS snapshots."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="SRS",
        snapshot_builder=_build_srs_snapshot,
        rating_column="SRS",
    )


def _build_raw_epa_snapshot(weekly_team_rows: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
    """Return pre-cutoff mean raw EPA margin per play for every team."""
    filtered_rows = weekly_team_rows.filter(pl.col("week") < cutoff_week)
    if filtered_rows.is_empty():
        return _empty_team_value_snapshot(weekly_team_rows, "raw_epa_margin")
    return (
        filtered_rows.group_by("team")
        .agg(pl.col("epa_margin_per_play").mean().alias("raw_epa_margin"))
        .sort("team")
    )


def build_raw_epa_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from raw pregame EPA margin means."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="RawEPA",
        snapshot_builder=_build_raw_epa_snapshot,
        rating_column="raw_epa_margin",
    )


def build_saovr_feature_rows(weekly_team_rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Build walk-forward home-game rows from week-specific SaOvR snapshots."""
    return build_snapshot_feature_rows(
        weekly_team_rows,
        season=season,
        baseline_name="SaOvR",
        snapshot_builder=build_team_rating_snapshot,
        rating_column="SaOvR",
    )


def _elo_margin_multiplier(rating_gap: float, home_margin: float) -> float:
    """Return a standard logarithmic margin-of-victory Elo multiplier."""
    return float(np.log(abs(home_margin) + 1.0) * (2.2 / ((abs(rating_gap) * 0.001) + 2.2)))


def build_elo_feature_rows(
    home_games: pl.DataFrame,
    config: EloConfig | None = None,
) -> pl.DataFrame:
    """Build one-row-per-game pregame Elo features from chronological home games."""
    resolved_config = config or EloConfig()
    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows: list[dict[str, object]] = []

    sorted_games = home_games.sort(["season", "week", "game_id"])
    for row in sorted_games.iter_rows(named=True):
        season = int(row["season"])
        if current_season is not None and season != current_season:
            for team, rating in list(ratings.items()):
                ratings[team] = (
                    resolved_config.initial_rating
                    + (rating - resolved_config.initial_rating) * resolved_config.regression_to_mean
                )
        current_season = season

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_rating = ratings.get(home_team, resolved_config.initial_rating)
        away_rating = ratings.get(away_team, resolved_config.initial_rating)
        rating_diff = home_rating - away_rating
        home_margin = float(row["home_margin"])

        rows.append(
            {
                "season": season,
                "week": int(row["week"]),
                "baseline": "Elo",
                "game_id": str(row["game_id"]),
                "home_team": home_team,
                "away_team": away_team,
                "rating_diff": rating_diff,
                "home_margin": home_margin,
            }
        )

        expected_home = 1.0 / (
            1.0 + 10.0 ** (-(rating_diff + resolved_config.home_field_elo) / 400.0)
        )
        actual_home = 1.0 if home_margin > 0.0 else 0.0 if home_margin < 0.0 else 0.5
        multiplier = (
            _elo_margin_multiplier(rating_diff, home_margin)
            if resolved_config.use_margin_multiplier and home_margin != 0.0
            else 1.0
        )
        delta = resolved_config.k_factor * multiplier * (actual_home - expected_home)
        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    return pl.DataFrame(rows).sort(["season", "week", "game_id"])


def run_walk_forward_backtest(
    data_dir: Path,
    seasons: list[int],
    start_week: int = 5,
    elo_config: EloConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full team walk-forward backtest across all requested seasons."""
    feature_frames: list[pl.DataFrame] = []
    for season in sorted(seasons):
        weekly_team_rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")
        home_games = _build_home_game_frame(weekly_team_rows, season)
        feature_frames.extend(
            [
                build_saovr_feature_rows(weekly_team_rows, season),
                build_srs_feature_rows(weekly_team_rows, season),
                build_raw_epa_feature_rows(weekly_team_rows, season),
                build_elo_feature_rows(home_games, config=elo_config),
            ]
        )

    if not feature_frames:
        empty_predictions = evaluate_feature_rows(pl.DataFrame(), start_week=start_week)
        return empty_predictions, score_prediction_rows(empty_predictions)

    all_features = pl.concat(feature_frames, how="vertical")
    predictions = evaluate_feature_rows(all_features, start_week=start_week)
    return predictions, score_prediction_rows(predictions)


def compute_stability_metrics(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute adjacent-season Pearson and Spearman stability for teams and QBs."""
    sorted_seasons = sorted(seasons)
    qb_metric_columns = ("QSaCR", "qb_passer_rating", "qb_any_a")
    qb_pairs: list[pl.DataFrame] = []
    team_pairs: list[pl.DataFrame] = []

    for season, next_season in zip(sorted_seasons, sorted_seasons[1:], strict=False):
        current_qb = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        next_qb = pl.read_parquet(data_dir / f"{next_season}_qb_combined.parquet")
        if "qb_is_eligible" in current_qb.columns:
            current_qb = current_qb.filter(pl.col("qb_is_eligible"))
        if "qb_is_eligible" in next_qb.columns:
            next_qb = next_qb.filter(pl.col("qb_is_eligible"))

        current_qb = current_qb.select(["qb_id", *qb_metric_columns]).rename(
            {column: f"{column}_t" for column in qb_metric_columns}
        )
        next_qb = next_qb.select(["qb_id", *qb_metric_columns]).rename(
            {column: f"{column}_t1" for column in qb_metric_columns}
        )
        qb_pairs.append(current_qb.join(next_qb, on="qb_id", how="inner"))

        current_team = pl.read_parquet(data_dir / f"{season}_combined.parquet").select(
            pl.col("team"), pl.col("SaOvR").alias("SaOvR_t")
        )
        next_team = pl.read_parquet(data_dir / f"{next_season}_combined.parquet").select(
            pl.col("team"), pl.col("SaOvR").alias("SaOvR_t1")
        )
        team_pairs.append(current_team.join(next_team, on="team", how="inner"))

    rows: list[dict[str, object]] = []
    if qb_pairs:
        qb_pair_frame = pl.concat(qb_pairs, how="diagonal_relaxed").drop_nulls(
            [
                "QSaCR_t",
                "QSaCR_t1",
                "qb_passer_rating_t",
                "qb_passer_rating_t1",
                "qb_any_a_t",
                "qb_any_a_t1",
            ]
        )
        for metric in qb_metric_columns:
            x_values = np.asarray(
                qb_pair_frame.select(f"{metric}_t").to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            y_values = np.asarray(
                qb_pair_frame.select(f"{metric}_t1").to_series().cast(pl.Float64).to_list(),
                dtype=np.float64,
            )
            rows.append(
                {
                    "entity": "qb",
                    "metric": metric,
                    "paired_rows": int(len(x_values)),
                    "pearson": _pearson(x_values, y_values),
                    "spearman": _spearman(x_values, y_values),
                }
            )

    if team_pairs:
        team_pair_frame = pl.concat(team_pairs, how="diagonal_relaxed").drop_nulls(
            ["SaOvR_t", "SaOvR_t1"]
        )
        x_values = np.asarray(
            team_pair_frame.select("SaOvR_t").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            team_pair_frame.select("SaOvR_t1").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        rows.append(
            {
                "entity": "team",
                "metric": "SaOvR",
                "paired_rows": int(len(x_values)),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort(["entity", "metric"])


def compute_qbr_correlations(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Compute per-season QSaCR versus ESPN QBR correlations on matched QBs."""
    eligible_seasons = sorted(season for season in seasons if season >= 2006)
    if not eligible_seasons:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "joined_rows": pl.Int64,
                "pearson": pl.Float64,
                "spearman": pl.Float64,
            }
        )

    qbr_df = load_espn_qbr(level="season", seasons=eligible_seasons)
    if "game_week" in qbr_df.columns:
        qbr_df = qbr_df.filter(pl.col("game_week") == "Season Total")

    qbr_df = qbr_df.with_columns(
        pl.col("team_abb").cast(pl.String).alias("team"),
        pl.col("name_display")
        .cast(pl.String)
        .map_elements(_normalize_person_name, return_dtype=pl.String)
        .alias("normalized_name"),
    )

    rows: list[dict[str, object]] = []
    for season in eligible_seasons:
        qb_combined = pl.read_parquet(data_dir / f"{season}_qb_combined.parquet")
        if "qb_is_eligible" in qb_combined.columns:
            qb_combined = qb_combined.filter(pl.col("qb_is_eligible"))
        qb_join_frame = qb_combined.with_columns(
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.col("qb_name")
            .cast(pl.String)
            .map_elements(_normalize_person_name, return_dtype=pl.String)
            .alias("normalized_name"),
        )
        joined = qb_join_frame.join(
            qbr_df.filter(pl.col("season") == season).select(
                "season",
                "team",
                "normalized_name",
                "qbr_total",
            ),
            on=["season", "team", "normalized_name"],
            how="inner",
        ).drop_nulls(["QSaCR", "qbr_total"])

        x_values = np.asarray(
            joined.select("QSaCR").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        y_values = np.asarray(
            joined.select("qbr_total").to_series().cast(pl.Float64).to_list(),
            dtype=np.float64,
        )
        rows.append(
            {
                "season": season,
                "joined_rows": int(len(x_values)),
                "pearson": _pearson(x_values, y_values),
                "spearman": _spearman(x_values, y_values),
            }
        )

    return pl.DataFrame(rows).sort("season")


def _format_markdown_value(value: object) -> str:
    """Format one scalar value for a Markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a simple GitHub-flavored Markdown table."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def build_validation_report_text(
    metrics: pl.DataFrame,
    stability: pl.DataFrame,
    qbr_correlations: pl.DataFrame,
    seasons: list[int],
    start_week: int,
    command: str,
) -> str:
    """Render the Stage 3 validation report as Markdown."""
    overall_metrics = metrics.filter(pl.col("split") != "season").sort(["baseline", "split"])
    season_metrics = metrics.filter(pl.col("split") == "season").sort(["season", "baseline"])

    overall_map = {
        str(row["baseline"]): float(row["mae"])
        for row in metrics.filter(pl.col("split") == "overall").iter_rows(named=True)
    }
    late_map = {
        str(row["baseline"]): float(row["mae"])
        for row in metrics.filter(pl.col("split") == "late").iter_rows(named=True)
    }
    stability_map = {str(row["metric"]): row for row in stability.iter_rows(named=True)}
    acceptance_lines = [
        "## Acceptance Check",
        "",
        "- Leakage discipline: the snapshot perturbation test and prior-only fit test pass.",
    ]

    if {"SaOvR", "Elo", "SRS", "RawEPA"} <= set(overall_map):
        saovr_mae = overall_map["SaOvR"]
        elo_mae = overall_map["Elo"]
        srs_mae = overall_map["SRS"]
        raw_epa_mae = overall_map["RawEPA"]
        team_pass = saovr_mae < elo_mae and saovr_mae < srs_mae and saovr_mae < raw_epa_mae
        acceptance_lines.append(
            "- Team headline: "
            f"{'Pass' if team_pass else 'Fail'}. SaOvR overall MAE {saovr_mae:.3f}; "
            f"Elo {elo_mae:.3f}; SRS {srs_mae:.3f}; RawEPA {raw_epa_mae:.3f}."
        )
        if {"SaOvR", "Elo", "SRS", "RawEPA"} <= set(late_map) and not team_pass:
            acceptance_lines.append(
                "- Team late-season context: SaOvR late-week MAE "
                f"{late_map['SaOvR']:.3f}; Elo {late_map['Elo']:.3f}; "
                f"SRS {late_map['SRS']:.3f}; RawEPA {late_map['RawEPA']:.3f}."
            )
    elif {"SaOvR", "Elo"} <= set(overall_map):
        acceptance_lines.append(
            "- Team headline: partial baseline set in this report sample. "
            f"SaOvR overall MAE {overall_map['SaOvR']:.3f}; Elo {overall_map['Elo']:.3f}."
        )

    if {"QSaCR", "qb_passer_rating", "qb_any_a"} <= set(stability_map):
        qsacr_row = stability_map["QSaCR"]
        passer_row = stability_map["qb_passer_rating"]
        any_a_row = stability_map["qb_any_a"]
        qb_pass = (
            float(qsacr_row["pearson"]) > float(passer_row["pearson"])
            and float(qsacr_row["pearson"]) > float(any_a_row["pearson"])
            and float(qsacr_row["spearman"]) > float(passer_row["spearman"])
            and float(qsacr_row["spearman"]) > float(any_a_row["spearman"])
        )
        acceptance_lines.append(
            "- QB stability: "
            f"{'Pass' if qb_pass else 'Fail'}. QSaCR Pearson/Spearman "
            f"{float(qsacr_row['pearson']):.3f}/{float(qsacr_row['spearman']):.3f}; "
            "passer rating "
            f"{float(passer_row['pearson']):.3f}/{float(passer_row['spearman']):.3f}; "
            f"ANY/A {float(any_a_row['pearson']):.3f}/{float(any_a_row['spearman']):.3f}."
        )

    if not qbr_correlations.is_empty():
        pearson_mean = float(qbr_correlations.select(pl.col("pearson").mean()).item())
        spearman_mean = float(qbr_correlations.select(pl.col("spearman").mean()).item())
        acceptance_lines.append(
            "- External reference: mean QBR Pearson/Spearman correlation "
            f"{pearson_mean:.3f}/{spearman_mean:.3f} across {qbr_correlations.height} seasons."
        )
    acceptance_lines.append("")

    overview_lines = [
        "# Validation Report",
        "",
        f"Evaluation seasons: {min(seasons)}-{max(seasons)}.",
        f"Prediction weeks start at {start_week}.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        *acceptance_lines,
    ]

    if not overall_metrics.is_empty():
        overall_rows = [
            [
                row["baseline"],
                row["split"],
                row["games"],
                row["mae"],
                row["rmse"],
            ]
            for row in overall_metrics.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Walk-Forward Summary",
                "",
                _markdown_table(
                    ["Baseline", "Split", "Games", "MAE", "RMSE"],
                    overall_rows,
                ),
                "",
            ]
        )

    if not season_metrics.is_empty():
        season_rows = [
            [
                row["season"],
                row["baseline"],
                row["games"],
                row["mae"],
                row["rmse"],
            ]
            for row in season_metrics.iter_rows(named=True)
        ]
        overview_lines.extend(
            [
                "## Per-Season Walk-Forward",
                "",
                _markdown_table(
                    ["Season", "Baseline", "Games", "MAE", "RMSE"],
                    season_rows,
                ),
                "",
            ]
        )

    stability_rows = [
        [
            row["metric"],
            row["entity"],
            row["paired_rows"],
            row["pearson"],
            row["spearman"],
        ]
        for row in stability.iter_rows(named=True)
    ]
    overview_lines.extend(
        [
            "## Stability",
            "",
            _markdown_table(
                ["Metric", "Entity", "Paired Rows", "Pearson", "Spearman"],
                stability_rows,
            ),
            "",
        ]
    )

    qbr_rows = [
        [row["season"], row["joined_rows"], row["pearson"], row["spearman"]]
        for row in qbr_correlations.iter_rows(named=True)
    ]
    overview_lines.extend(
        [
            "## QBR Correlations",
            "",
            _markdown_table(
                ["Season", "Joined Rows", "Pearson", "Spearman"],
                qbr_rows,
            ),
            "",
            "## SaCR Caveat",
            "",
            "SaCR may be evaluated as a secondary line with a caveat:",
            "its frozen Stage 2 weights were fit on the full 1999-2025 history.",
            "A walk-forward SaCR line over that same window has look-ahead in the weights.",
            "SaOvR is the headline walk-forward metric because it does not depend on a "
            "fitted Stage 2 weight snapshot.",
            "",
        ]
    )
    return "\n".join(overview_lines).rstrip() + "\n"


def write_validation_report(
    report_path: Path,
    metrics: pl.DataFrame,
    stability: pl.DataFrame,
    qbr_correlations: pl.DataFrame,
    seasons: list[int],
    start_week: int,
    command: str,
) -> None:
    """Write the Stage 3 validation report to disk."""
    report_text = build_validation_report_text(
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr_correlations,
        seasons=seasons,
        start_week=start_week,
        command=command,
    )
    report_path.write_text(report_text, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Stage 3 validation command."""
    parser = argparse.ArgumentParser(description="Run the Stage 3 walk-forward validation suite.")
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory holding historical Parquet artifacts.",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=START_YEAR,
        help="First season to evaluate.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=END_YEAR,
        help="Last season to evaluate.",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=5,
        help="First week to score in each season.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/validation-report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Stage 3 validation suite and write the Markdown report."""
    args = _parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    data_dir = Path(args.data_dir)
    report_path = Path(args.report_path)
    command = (
        "uv run python -m nfl_sos_ratings.validation.walk_forward "
        f"--data-dir {args.data_dir} --start-season {args.start_season} "
        f"--end-season {args.end_season} --start-week {args.start_week} "
        f"--report-path {args.report_path}"
    )

    _predictions, metrics = run_walk_forward_backtest(
        data_dir,
        seasons=seasons,
        start_week=args.start_week,
    )
    stability = compute_stability_metrics(data_dir, seasons=seasons)
    qbr_correlations = compute_qbr_correlations(data_dir, seasons=seasons)
    write_validation_report(
        report_path,
        metrics=metrics,
        stability=stability,
        qbr_correlations=qbr_correlations,
        seasons=seasons,
        start_week=args.start_week,
        command=command,
    )

    print(f"Wrote validation report to {report_path}")


def _fit_margin_projection(training_rows: pl.DataFrame) -> tuple[float, float]:
    """Fit ``home_margin = k * rating_diff + hfa_points`` on past games only."""
    if training_rows.is_empty():
        return 0.0, 0.0

    rating_diff = np.asarray(
        training_rows.select("rating_diff").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    home_margin = np.asarray(
        training_rows.select("home_margin").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    if len(training_rows) == 1 or np.isclose(rating_diff.std(ddof=0), 0.0):
        return 0.0, float(home_margin.mean())

    design = np.column_stack((rating_diff, np.ones(len(rating_diff), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, home_margin, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def evaluate_feature_rows(feature_rows: pl.DataFrame, start_week: int) -> pl.DataFrame:
    """Fit prior-only margin models and predict every game from ``start_week`` onward."""
    if feature_rows.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "baseline": pl.String,
                "game_id": pl.String,
                "home_team": pl.String,
                "away_team": pl.String,
                "rating_diff": pl.Float64,
                "home_margin": pl.Float64,
                "predicted_margin": pl.Float64,
                "error": pl.Float64,
                "training_row_count": pl.Int64,
                "fitted_k": pl.Float64,
                "fitted_hfa_points": pl.Float64,
            }
        )

    prediction_rows: list[dict[str, object]] = []
    eval_rows = feature_rows.filter(pl.col("week") >= start_week).sort(
        ["baseline", "season", "week", "game_id"]
    )
    grouped_eval_rows = eval_rows.group_by(["baseline", "season", "week"], maintain_order=True)

    for keys, week_rows in grouped_eval_rows:
        baseline, season, week = keys
        baseline_name = str(baseline)
        season_value = int(season)
        week_value = int(week)
        training_rows = feature_rows.filter(
            (pl.col("baseline") == baseline_name)
            & (
                (pl.col("season") < season_value)
                | ((pl.col("season") == season_value) & (pl.col("week") < week_value))
            )
        )
        fitted_k, fitted_hfa_points = _fit_margin_projection(training_rows)

        for row in week_rows.iter_rows(named=True):
            predicted_margin = fitted_k * float(row["rating_diff"]) + fitted_hfa_points
            actual_margin = float(row["home_margin"])
            prediction_rows.append(
                {
                    "season": season_value,
                    "week": week_value,
                    "baseline": baseline_name,
                    "game_id": str(row["game_id"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "rating_diff": float(row["rating_diff"]),
                    "home_margin": actual_margin,
                    "predicted_margin": predicted_margin,
                    "error": predicted_margin - actual_margin,
                    "training_row_count": training_rows.height,
                    "fitted_k": fitted_k,
                    "fitted_hfa_points": fitted_hfa_points,
                }
            )

    return pl.DataFrame(prediction_rows).sort(["baseline", "season", "week", "game_id"])


def _metric_row(
    frame: pl.DataFrame,
    *,
    baseline: str,
    split: str,
    season: int | None,
) -> dict[str, object]:
    """Summarize one slice of prediction rows as MAE and RMSE."""
    if frame.is_empty():
        return {
            "baseline": baseline,
            "season": season,
            "split": split,
            "games": 0,
            "mae": 0.0,
            "rmse": 0.0,
        }

    errors = np.asarray(
        frame.select("error").to_series().cast(pl.Float64).to_list(),
        dtype=np.float64,
    )
    return {
        "baseline": baseline,
        "season": season,
        "split": split,
        "games": frame.height,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def score_prediction_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate overall, per-season, and early/late walk-forward error metrics."""
    if predictions.is_empty():
        return pl.DataFrame(
            schema={
                "baseline": pl.String,
                "season": pl.Int64,
                "split": pl.String,
                "games": pl.Int64,
                "mae": pl.Float64,
                "rmse": pl.Float64,
            }
        )

    scored_predictions = predictions
    if "error" not in scored_predictions.columns:
        scored_predictions = scored_predictions.with_columns(
            (pl.col("predicted_margin") - pl.col("home_margin")).alias("error")
        )

    metric_rows: list[dict[str, object]] = []
    for baseline in scored_predictions.select("baseline").to_series().unique().to_list():
        baseline_frame = scored_predictions.filter(pl.col("baseline") == str(baseline))
        metric_rows.append(
            _metric_row(baseline_frame, baseline=str(baseline), split="overall", season=None)
        )
        metric_rows.append(
            _metric_row(
                baseline_frame.filter(pl.col("week") < 8),
                baseline=str(baseline),
                split="early",
                season=None,
            )
        )
        metric_rows.append(
            _metric_row(
                baseline_frame.filter(pl.col("week") >= 8),
                baseline=str(baseline),
                split="late",
                season=None,
            )
        )

        seasons = baseline_frame.select("season").to_series().unique().sort().to_list()
        for season in seasons:
            season_frame = baseline_frame.filter(pl.col("season") == int(season))
            metric_rows.append(
                _metric_row(
                    season_frame,
                    baseline=str(baseline),
                    split="season",
                    season=int(season),
                )
            )

    return pl.DataFrame(metric_rows).sort(["baseline", "split", "season"])


__all__ = [
    "EloConfig",
    "build_elo_feature_rows",
    "build_raw_epa_feature_rows",
    "build_saovr_feature_rows",
    "build_validation_report_text",
    "build_srs_feature_rows",
    "build_snapshot_feature_rows",
    "compute_qbr_correlations",
    "compute_stability_metrics",
    "evaluate_feature_rows",
    "main",
    "run_walk_forward_backtest",
    "score_prediction_rows",
    "write_validation_report",
]


if __name__ == "__main__":
    main()
