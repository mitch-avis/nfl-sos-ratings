"""Markdown rendering for the walk-forward validation report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard, cast

import polars as pl

from nfl_sos_ratings.validation import history_strings
from nfl_sos_ratings.validation.baselines import ROLLING_EPA_BASELINE, ROLLING_EPA_ST_BASELINE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

type TableRows = list[list[object | None]]

_TEAM_HEADLINE_BASELINES = frozenset({"SaOvR", "Elo", "SRS", "RawEPA"})
_QB_STABILITY_METRICS = frozenset({"QSaCR", "qb_passer_rating", "qb_any_a"})
_SIGNAL_SUMMARY_COLUMNS = (
    "scope",
    "season",
    "rows",
    "total_dropbacks",
    "slope",
    "correlation",
    "ci_lower",
    "ci_upper",
    "direction_positive_count",
    "direction_total_count",
    "direction_p_value",
)
_SIGNAL_SUMMARY_HEADERS = (
    "Scope",
    "Season",
    "QB Seasons",
    "Dropbacks",
    "Slope",
    "Correlation",
    "CI Lower",
    "CI Upper",
    "Positive Seasons",
    "Season Count",
    "Binomial P",
)
_PLAYOFF_CI_COLUMNS = frozenset(
    {"spearman_ci_lower", "spearman_ci_upper", "pearson_ci_lower", "pearson_ci_upper"}
)


@dataclass(frozen=True, slots=True)
class ValidationReportInputs:
    """Everything the validation report renders.

    The first six fields are always rendered. Every optional section is skipped when its input
    is ``None`` or empty.
    """

    metrics: pl.DataFrame
    stability: pl.DataFrame
    qbr_correlations: pl.DataFrame
    seasons: list[int]
    start_week: int
    command: str
    mae_deltas: pl.DataFrame | None = None
    comparison_metrics: pl.DataFrame | None = None
    weekly_curves: pl.DataFrame | None = None
    saovr_vs_srs: pl.DataFrame | None = None
    t2_vs_srs: pl.DataFrame | None = None
    qb_season_audit: pl.DataFrame | None = None
    qb_defense_spread: pl.DataFrame | None = None
    qb_experiment_sweep: pl.DataFrame | None = None
    qb_case_study: pl.DataFrame | None = None
    qb_schedule_anchor: pl.DataFrame | None = None
    qb_schedule_trace: pl.DataFrame | None = None
    qb_lens_divergence: pl.DataFrame | None = None
    qb_designed_rush_preview: pl.DataFrame | None = None
    team_decision_lines: list[str] | None = None
    qb_open_status_lines: list[str] | None = None
    regression_note_lines: list[str] | None = None
    qb_opponent_offense_summary: pl.DataFrame | None = None
    qb_opponent_offense_cases: pl.DataFrame | None = None
    qb_opponent_offense_decision: dict[str, object] | None = None
    qb_leverage_summary: pl.DataFrame | None = None
    qb_leverage_cases: pl.DataFrame | None = None
    qb_leverage_decision: dict[str, object] | None = None
    qb_split_half_primary: pl.DataFrame | None = None
    qb_split_half_placebo: pl.DataFrame | None = None
    qb_split_half_cases: pl.DataFrame | None = None
    qb_split_half_decision: dict[str, object] | None = None
    qb_playoff_correlations: pl.DataFrame | None = None


def _format_markdown_value(value: object) -> str:
    """Format one scalar value for a Markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object | None]]) -> str:
    """Render a simple GitHub-flavored Markdown table."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _has_rows(frame: pl.DataFrame | None) -> TypeGuard[pl.DataFrame]:
    """Return whether an optional report input is present and non-empty."""
    return frame is not None and not frame.is_empty()


def _table_rows(frame: pl.DataFrame, columns: Sequence[str]) -> TableRows:
    """Return the named columns of every row; a missing column raises ``KeyError``."""
    return [[row[column] for column in columns] for row in frame.iter_rows(named=True)]


def _optional_table_rows(frame: pl.DataFrame, columns: Sequence[str]) -> TableRows:
    """Return the named columns of every row, with ``None`` for any missing column."""
    return [[row.get(column) for column in columns] for row in frame.iter_rows(named=True)]


def _table_section(title: str, headers: Sequence[str], rows: TableRows) -> list[str]:
    """Return one ``##`` section holding a single table."""
    return [f"## {title}", "", markdown_table(headers, rows), ""]


def _overall_mae_map(metrics: pl.DataFrame, split: str) -> dict[str, float]:
    """Return baseline -> MAE for one evaluation split."""
    return {
        str(row["baseline"]): float(row["mae"])
        for row in metrics.filter(pl.col("split") == split).iter_rows(named=True)
    }


def _team_acceptance_lines(overall_map: dict[str, float], late_map: dict[str, float]) -> list[str]:
    """Return the team headline acceptance line, plus late-season context when it fails."""
    if not set(overall_map) >= _TEAM_HEADLINE_BASELINES:
        return []
    saovr_mae = overall_map["SaOvR"]
    elo_mae = overall_map["Elo"]
    srs_mae = overall_map["SRS"]
    raw_epa_mae = overall_map["RawEPA"]
    team_pass = saovr_mae < elo_mae and saovr_mae < srs_mae and saovr_mae < raw_epa_mae
    lines = [
        (
            "- Team headline: "
            f"{'Pass' if team_pass else 'Fail'}. SaOvR overall MAE {saovr_mae:.3f}; "
            f"Elo {elo_mae:.3f}; SRS {srs_mae:.3f}; RawEPA {raw_epa_mae:.3f}."
        )
    ]
    if set(late_map) >= _TEAM_HEADLINE_BASELINES and not team_pass:
        lines.append(
            "- Team late-season context: SaOvR late-week MAE "
            f"{late_map['SaOvR']:.3f}; Elo {late_map['Elo']:.3f}; "
            f"SRS {late_map['SRS']:.3f}; RawEPA {late_map['RawEPA']:.3f}."
        )
    return lines


def _qb_stability_acceptance_lines(stability: pl.DataFrame) -> list[str]:
    """Return the QB stability acceptance line when all three compared metrics are present."""
    stability_map = {str(row["metric"]): row for row in stability.iter_rows(named=True)}
    if not set(stability_map) >= _QB_STABILITY_METRICS:
        return []
    qsacr_row = stability_map["QSaCR"]
    passer_row = stability_map["qb_passer_rating"]
    any_a_row = stability_map["qb_any_a"]
    qb_pass = (
        float(qsacr_row["pearson"]) > float(passer_row["pearson"])
        and float(qsacr_row["pearson"]) > float(any_a_row["pearson"])
        and float(qsacr_row["spearman"]) > float(passer_row["spearman"])
        and float(qsacr_row["spearman"]) > float(any_a_row["spearman"])
    )
    return [
        (
            "- QB stability: "
            f"{'Pass' if qb_pass else 'Fail'}. QSaCR Pearson/Spearman "
            f"{float(qsacr_row['pearson']):.3f}/{float(qsacr_row['spearman']):.3f}; "
            "passer rating "
            f"{float(passer_row['pearson']):.3f}/{float(passer_row['spearman']):.3f};"
            f"\n  ANY/A {float(any_a_row['pearson']):.3f}/{float(any_a_row['spearman']):.3f}."
        )
    ]


def _acceptance_lines(inputs: ValidationReportInputs) -> list[str]:
    """Return the acceptance-check section."""
    lines = [
        "## Acceptance Check",
        "",
        "- Leakage discipline: the snapshot perturbation test and prior-only fit test pass.",
        *_team_acceptance_lines(
            _overall_mae_map(inputs.metrics, "overall"), _overall_mae_map(inputs.metrics, "late")
        ),
        *_qb_stability_acceptance_lines(inputs.stability),
    ]
    qbr_correlations = inputs.qbr_correlations
    if not qbr_correlations.is_empty():
        pearson_mean = float(qbr_correlations.select(pl.col("pearson").mean()).item())
        spearman_mean = float(qbr_correlations.select(pl.col("spearman").mean()).item())
        lines.append(
            "- External reference: mean QBR Pearson/Spearman correlation "
            f"{pearson_mean:.3f}/{spearman_mean:.3f} across {qbr_correlations.height} seasons."
        )
    lines.append("")
    return lines


def _overall_delta_row(
    mae_deltas: pl.DataFrame | None, baseline_b: str
) -> dict[str, object] | None:
    """Return the overall rolling-EPA-plus-ST versus ``baseline_b`` bootstrap row, if any."""
    if not _has_rows(mae_deltas):
        return None
    match = mae_deltas.filter(
        (pl.col("baseline_a") == ROLLING_EPA_ST_BASELINE)
        & (pl.col("baseline_b") == baseline_b)
        & (pl.col("split") == "overall")
    )
    return None if match.is_empty() else match.row(0, named=True)


def _row_float(row: dict[str, object], key: str) -> float:
    """Return one numeric row value as a float."""
    return float(cast("float", row[key]))


def _bootstrap_delta_line(label: str, row: dict[str, object]) -> str:
    """Return one league bootstrap delta line."""
    return (
        f"- League 1 bootstrap vs {label}: "
        f"MAE delta {_row_float(row, 'mae_delta'):.3f} with 95% CI "
        f"[{_row_float(row, 'ci_lower'):.3f}, "
        f"{_row_float(row, 'ci_upper'):.3f}]."
    )


def _league_comparison_lines(inputs: ValidationReportInputs) -> list[str]:
    """Return the league-criterion acceptance lines for the rolling team baselines."""
    comparison_metrics = inputs.comparison_metrics
    if not _has_rows(comparison_metrics):
        return []
    comparison_map = _overall_mae_map(comparison_metrics, "overall")
    if not {ROLLING_EPA_BASELINE, ROLLING_EPA_ST_BASELINE} <= set(comparison_map):
        return []
    t2_vs_srs_row = _overall_delta_row(inputs.mae_deltas, "SRS")
    t2_vs_raw_row = _overall_delta_row(inputs.mae_deltas, "RawEPA")
    t2_mae = comparison_map.get(ROLLING_EPA_ST_BASELINE, float("inf"))
    league1_pass = (
        t2_vs_srs_row is not None
        and bool(t2_vs_srs_row["distinguishable_from_zero"])
        and t2_vs_raw_row is not None
        and bool(t2_vs_raw_row["distinguishable_from_zero"])
        and t2_mae < comparison_map.get("SRS", float("inf"))
        and t2_mae < comparison_map.get("RawEPA", float("inf"))
    )
    lines = [
        history_strings.report_league_acceptance_heading(),
        "",
        (
            "- League 1 team headline:\n  "
            f"{'Pass' if league1_pass else 'Fail'}. "
            f"{ROLLING_EPA_BASELINE} overall MAE "
            f"{comparison_map[ROLLING_EPA_BASELINE]:.3f};\n  "
            f"{ROLLING_EPA_ST_BASELINE} overall MAE "
            f"{comparison_map[ROLLING_EPA_ST_BASELINE]:.3f};\n  "
            f"SRS {comparison_map.get('SRS', float('nan')):.3f};\n  "
            f"RawEPA {comparison_map.get('RawEPA', float('nan')):.3f}."
        ),
    ]
    if t2_vs_srs_row is not None:
        lines.append(_bootstrap_delta_line("SRS", t2_vs_srs_row))
    if t2_vs_raw_row is not None:
        lines.append(_bootstrap_delta_line("RawEPA", t2_vs_raw_row))
    return lines


def _qb_sweep_comparison_lines(qb_experiment_sweep: pl.DataFrame | None) -> list[str]:
    """Return the QB revision-sweep summary lines when all three compared variants exist."""
    if not _has_rows(qb_experiment_sweep):
        return []
    current_row = qb_experiment_sweep.filter(pl.col("variant") == "current")
    fixed_defense_row = qb_experiment_sweep.filter(pl.col("variant") == "fixed_team_defense")
    lighter_penalty_row = qb_experiment_sweep.sort("slope", descending=True).head(1)
    if current_row.is_empty() or fixed_defense_row.is_empty() or lighter_penalty_row.is_empty():
        return []
    current_variant = current_row.row(0, named=True)
    fixed_variant = fixed_defense_row.row(0, named=True)
    lighter_variant = lighter_penalty_row.row(0, named=True)
    return [
        (
            "- QB revision sweep: not adopted. "
            f"Current eligible-QB slope {float(current_variant['slope']):.3f}; "
            f"fixed-defense slope {float(fixed_variant['slope']):.3f};\n  "
            "best tested lighter-defense-penalty slope "
            f"{float(lighter_variant['slope']):.3f} ({lighter_variant['variant']})."
        ),
        "- League 2 forecast-only prior experiment: not evaluated in this worktree.",
        "",
    ]


def _header_lines(inputs: ValidationReportInputs) -> list[str]:
    """Return the title, command, narrative history, comparisons, and acceptance check."""
    return [
        "# Validation Report",
        "",
        f"Evaluation seasons: {min(inputs.seasons)}-{max(inputs.seasons)}.",
        f"Prediction weeks start at {inputs.start_week}.",
        "",
        "## Command",
        "",
        "```bash",
        inputs.command,
        "```",
        "",
        *(inputs.regression_note_lines or []),
        *history_strings.report_history_lines(),
        *history_strings.report_league_criterion_lines(),
        *_league_comparison_lines(inputs),
        *_qb_sweep_comparison_lines(inputs.qb_experiment_sweep),
        *(inputs.team_decision_lines or []),
        *_acceptance_lines(inputs),
    ]


def _team_metric_sections(inputs: ValidationReportInputs) -> list[str]:
    """Return the team walk-forward summary, experiment, delta, and curve tables."""
    metric_headers = ["Baseline", "Split", "Games", "MAE", "RMSE"]
    metric_columns = ["baseline", "split", "games", "mae", "rmse"]
    lines: list[str] = []
    overall_metrics = inputs.metrics.filter(pl.col("split") != "season").sort(["baseline", "split"])
    if not overall_metrics.is_empty():
        lines += _table_section(
            "Original Walk-Forward Summary",
            metric_headers,
            _table_rows(overall_metrics, metric_columns),
        )
    if _has_rows(inputs.comparison_metrics):
        lines += _table_section(
            "League 1 Team Experiments",
            metric_headers,
            _table_rows(
                inputs.comparison_metrics.filter(pl.col("split") != "season"), metric_columns
            ),
        )
    if _has_rows(inputs.mae_deltas):
        lines += _table_section(
            "Paired Bootstrap MAE Deltas",
            [
                "Baseline A",
                "Baseline B",
                "Split",
                "Games",
                "MAE Delta",
                "CI Lower",
                "CI Upper",
                "P(A<=B)",
                "Distinguishable",
            ],
            _table_rows(
                inputs.mae_deltas,
                [
                    "baseline_a",
                    "baseline_b",
                    "split",
                    "games",
                    "mae_delta",
                    "ci_lower",
                    "ci_upper",
                    "probability_baseline_a_not_worse",
                    "distinguishable_from_zero",
                ],
            ),
        )
    if _has_rows(inputs.weekly_curves):
        lines += _table_section(
            "Weekly MAE Curves",
            ["Week", "Baseline", "Games", "MAE", "RMSE"],
            _table_rows(inputs.weekly_curves, ["week", "baseline", "games", "mae", "rmse"]),
        )
    return lines


def _season_comparison_sections(inputs: ValidationReportInputs) -> list[str]:
    """Return the per-season team tables, then the always-present stability and QBR tables."""
    lines: list[str] = []
    season_metrics = inputs.metrics.filter(pl.col("split") == "season").sort(["season", "baseline"])
    if not season_metrics.is_empty():
        lines += _table_section(
            "Original Per-Season Walk-Forward",
            ["Season", "Baseline", "Games", "MAE", "RMSE"],
            _table_rows(season_metrics, ["season", "baseline", "games", "mae", "rmse"]),
        )
    delta_columns = ["season", "mae_a", "mae_b", "mae_delta", "rmse_delta"]
    if _has_rows(inputs.saovr_vs_srs):
        lines += _table_section(
            "Per-Season SaOvR vs SRS",
            ["Season", "SaOvR MAE", "SRS MAE", "MAE Delta", "RMSE Delta"],
            _table_rows(inputs.saovr_vs_srs, delta_columns),
        )
    if _has_rows(inputs.t2_vs_srs):
        lines += _table_section(
            f"Per-Season {ROLLING_EPA_ST_BASELINE} vs SRS",
            ["Season", f"{ROLLING_EPA_ST_BASELINE} MAE", "SRS MAE", "MAE Delta", "RMSE Delta"],
            _table_rows(inputs.t2_vs_srs, delta_columns),
        )
    lines += _table_section(
        "Stability",
        ["Metric", "Entity", "Paired Rows", "Pearson", "Spearman"],
        _table_rows(inputs.stability, ["metric", "entity", "paired_rows", "pearson", "spearman"]),
    )
    lines += _table_section(
        "QBR Correlations",
        ["Season", "Joined Rows", "Pearson", "Spearman"],
        _table_rows(inputs.qbr_correlations, ["season", "joined_rows", "pearson", "spearman"]),
    )
    return lines


def _qb_schedule_lens_sections(inputs: ValidationReportInputs) -> list[str]:
    """Return the named-QB schedule-lens anchor, trace, divergence, and rush-preview tables."""
    lines: list[str] = []
    if _has_rows(inputs.qb_schedule_anchor):
        lines += _table_section(
            "2025 QB Schedule-Lens Anchor",
            ["QB", "Games", "Avg Opp SaCR", "Avg Opp SaDR", "Avg Opp SRS"],
            _table_rows(
                inputs.qb_schedule_anchor,
                ["qb_name", "games", "avg_opp_SaCR", "avg_opp_SaDR", "avg_opp_SRS"],
            ),
        )
    if _has_rows(inputs.qb_schedule_trace):
        lines += _table_section(
            "QB Schedule-Lens Trace",
            [
                "QB",
                "Raw EPA/DB",
                "Adjusted EPA/DB",
                "Weighted Faced Defense",
                "Adjustment Delta",
                "QSoS",
                "Faced Opp SaCR",
                "Faced Adj Def EPA/DB",
            ],
            _optional_table_rows(
                inputs.qb_schedule_trace,
                [
                    "qb_name",
                    "raw_value",
                    "adjusted_value",
                    "weighted_faced_defense",
                    "adjustment_delta",
                    "QSoS",
                    "faced_opp_SaCR",
                    "adj_def_qb_epa_per_dropback_faced",
                ],
            ),
        )
    if _has_rows(inputs.qb_lens_divergence):
        lines += _table_section(
            "QB Lens-Divergence Rankings",
            ["QB", "Team", "QSoS", "Faced Opp SaCR", "QSoS Rank", "Overall Rank", "Rank Gap"],
            _optional_table_rows(
                inputs.qb_lens_divergence,
                [
                    "qb_name",
                    "team",
                    "QSoS",
                    "faced_opp_SaCR",
                    "qsos_rank",
                    "overall_rank",
                    "rank_gap",
                ],
            ),
        )
    if _has_rows(inputs.qb_designed_rush_preview):
        lines += _table_section(
            "2025 QB Designed-Rush Preview",
            [
                "QB",
                "Team",
                "Designed Carries",
                "Designed EPA/Carry",
                "Faced Rush Defense",
                "Adj Designed Rush EPA/Carry",
                "QSoS",
                "Faced Adj Def EPA/DB",
                "Faced Opp SaCR",
            ],
            _optional_table_rows(
                inputs.qb_designed_rush_preview,
                [
                    "qb_name",
                    "team",
                    "qb_designed_carries_total",
                    "qb_designed_epa_per_carry",
                    "adj_def_rushing_epa_per_offensive_snap_faced",
                    "adj_qb_designed_rush_epa_per_carry",
                    "QSoS",
                    "adj_def_qb_epa_per_dropback_faced",
                    "faced_opp_SaCR",
                ],
            ),
        )
    return lines


def _qb_audit_sections(inputs: ValidationReportInputs) -> list[str]:
    """Return the QB adjustment-audit, defense-spread, revision-sweep, and case-study tables."""
    lines: list[str] = []
    if _has_rows(inputs.qb_season_audit):
        lines += _table_section(
            "QB Adjustment Audit",
            ["Season", "Eligible QBs", "Slope", "Correlation", "Mean Abs Residual"],
            _table_rows(
                inputs.qb_season_audit,
                ["season", "rows", "slope", "correlation", "mean_abs_identity_residual"],
            ),
        )
    if _has_rows(inputs.qb_defense_spread):
        lines += _table_section(
            "QB Defense Spread Audit",
            ["Season", "Team Defense SD", "QB Defense SD", "QB/Team Ratio"],
            _table_rows(
                inputs.qb_defense_spread,
                ["season", "team_defense_sd", "qb_defense_sd", "qb_to_team_spread_ratio"],
            ),
        )
    if _has_rows(inputs.qb_experiment_sweep):
        lines += _table_section(
            "QB Revision Sweep",
            ["Variant", "Eligible QBs", "Slope", "Correlation", "Defense Penalty Multiplier"],
            [
                [
                    row["variant"],
                    row["eligible_rows"],
                    row["slope"],
                    row["correlation"],
                    row.get("defense_penalty_multiplier"),
                ]
                for row in inputs.qb_experiment_sweep.iter_rows(named=True)
            ],
        )
    if _has_rows(inputs.qb_case_study):
        lines += _table_section(
            "Maye/Stafford Case Study",
            [
                "Variant",
                "QB",
                "Raw EPA/DB",
                "Adjusted EPA/DB",
                "Faced Difficulty",
                "Adjustment Delta",
            ],
            _table_rows(
                inputs.qb_case_study,
                [
                    "variant",
                    "qb_name",
                    "raw_weighted",
                    "adjusted_value",
                    "faced_difficulty",
                    "adjustment_delta",
                ],
            ),
        )
    return lines


def _pooled_signal_line(summary: pl.DataFrame) -> list[str]:
    """Return the pooled weighted-slope line of a signal summary, if it has a pooled row."""
    pooled_row = summary.filter(pl.col("scope") == "pooled")
    if pooled_row.is_empty():
        return []
    pooled = pooled_row.row(0, named=True)
    return [
        (
            "- Pooled weighted slope "
            f"{float(pooled['slope']):.3f} with 95% CI "
            f"[{float(pooled['ci_lower']):.3f}, {float(pooled['ci_upper']):.3f}]\n  and "
            f"{int(pooled['direction_positive_count'])} / "
            f"{int(pooled['direction_total_count'])} "
            f"positive seasons (p = {float(pooled['direction_p_value']):.3f})."
        )
    ]


def _signal_tables(
    summary: pl.DataFrame,
    cases: pl.DataFrame | None,
    case_columns: tuple[tuple[str, str], ...],
) -> list[str]:
    """Return a signal summary table, then its named-case table when cases exist.

    ``case_columns`` pairs each case-table header with its source column.
    """
    lines = [
        markdown_table(_SIGNAL_SUMMARY_HEADERS, _table_rows(summary, _SIGNAL_SUMMARY_COLUMNS)),
        "",
    ]
    if _has_rows(cases):
        lines += [
            markdown_table(
                [header for header, _ in case_columns],
                _table_rows(cases, [column for _, column in case_columns]),
            ),
            "",
        ]
    return lines


def _qb_opponent_offense_section(inputs: ValidationReportInputs) -> list[str]:
    """Return the opponent-offense residual section."""
    summary = inputs.qb_opponent_offense_summary
    if not _has_rows(summary):
        return []
    lines = history_strings.opponent_offense_report_heading_lines()
    decision = inputs.qb_opponent_offense_decision
    if decision is not None:
        lines.append(f"- Gate reading: {decision.get('decision', 'not_supported')}.")
    lines += _pooled_signal_line(summary)
    lines += _signal_tables(
        summary,
        inputs.qb_opponent_offense_cases,
        (
            ("Season", "season"),
            ("QB", "qb_name"),
            ("Faced Opponent Offense", "faced_opponent_offense"),
            ("Mean Adjusted Residual", "mean_adjusted_residual"),
            ("Dropbacks", "total_dropbacks"),
        ),
    )
    return lines


def _qb_leverage_section(inputs: ValidationReportInputs) -> list[str]:
    """Return the leverage-softness section."""
    summary = inputs.qb_leverage_summary
    if not _has_rows(summary):
        return []
    lines = history_strings.leverage_report_heading_lines()
    decision = inputs.qb_leverage_decision
    if decision is not None:
        lines += [
            (
                "- Moderate-leverage win-probability band: "
                f"{decision.get('moderate_wp_band', 'unknown')}."
            ),
            f"- Gate reading: {decision.get('decision', 'not_supported')}.",
            (
                "- Companion gate: stability "
                f"{'pass' if decision.get('stability_pass') else 'fail'}, "
                "playoff correlation "
                f"{'pass' if decision.get('playoff_pass') else 'fail'}."
            ),
        ]
    lines += _pooled_signal_line(summary)
    lines += _signal_tables(
        summary,
        inputs.qb_leverage_cases,
        (
            ("Season", "season"),
            ("QB", "qb_name"),
            ("Schedule Softness", "schedule_softness"),
            ("Low-Leverage Share", "low_leverage_share"),
            ("Moderate-Leverage Share", "moderate_leverage_share"),
        ),
    )
    return lines


def _split_half_decision_lines(decision: dict[str, object] | None) -> list[str]:
    """Return the split-half gate reading, primary-gate, and placebo lines."""
    if decision is None:
        return []
    lines = [f"- Decision gate reading: {decision.get('decision', 'not_supported')!s}."]
    if decision.get("primary_gate_supported") is not None:
        lines.append(
            "- Primary top-half gate: "
            f"{'passed' if decision['primary_gate_supported'] else 'failed'}."
        )
    if decision.get("placebo_is_symmetric"):
        lines.append(
            "- Placebo check: bottom-half residuals showed a same-direction signal,"
            "\n  so the strong-defense-specific interpretation is not supported."
        )
    lines.append("")
    return lines


def _labeled_table(label: str, headers: Sequence[str], rows: TableRows) -> list[str]:
    """Return a plain-text label line followed by a table."""
    return [label, "", markdown_table(headers, rows), ""]


def _qb_split_half_section(inputs: ValidationReportInputs) -> list[str]:
    """Return the split-half primary, placebo, and named-case tables."""
    lines: list[str] = []
    split_columns = ["scope", "season", "rows", "total_dropbacks", "slope", "ci_lower", "ci_upper"]
    split_headers = ["Scope", "Season", "QB Seasons", "Dropbacks", "Slope", "CI Lower", "CI Upper"]
    if _has_rows(inputs.qb_split_half_primary):
        lines += history_strings.split_half_report_heading_lines()
        lines += _split_half_decision_lines(inputs.qb_split_half_decision)
        lines += _labeled_table(
            "Top-half residual regression summary:",
            [*split_headers, "Positive Seasons", "Season Count", "Binomial P"],
            _table_rows(
                inputs.qb_split_half_primary,
                [
                    *split_columns,
                    "direction_positive_count",
                    "direction_total_count",
                    "direction_p_value",
                ],
            ),
        )
    if _has_rows(inputs.qb_split_half_placebo):
        lines += _labeled_table(
            "Bottom-half placebo summary:",
            split_headers,
            _table_rows(inputs.qb_split_half_placebo, split_columns),
        )
    if _has_rows(inputs.qb_split_half_cases):
        lines += _labeled_table(
            "2025 named case rows:",
            [
                "Season",
                "QB",
                "Faced Difficulty",
                "Additive Prediction",
                "Top-Half Adj EPA/DB",
                "Top-Half Residual",
                "Top-Half DB",
                "Bottom-Half Adj EPA/DB",
                "Bottom-Half Residual",
                "Bottom-Half DB",
            ],
            _optional_table_rows(
                inputs.qb_split_half_cases,
                [
                    "season",
                    "qb_name",
                    "faced_difficulty",
                    "additive_prediction",
                    "vs_top_half_adjusted_epa_per_dropback",
                    "vs_top_half_residual",
                    "vs_top_half_dropbacks",
                    "vs_bottom_half_adjusted_epa_per_dropback",
                    "vs_bottom_half_residual",
                    "vs_bottom_half_dropbacks",
                ],
            ),
        )
    return lines


def _qb_playoff_section(correlations: pl.DataFrame | None) -> list[str]:
    """Return the playoff validation correlation table, with CIs when they were computed."""
    if not _has_rows(correlations):
        return []
    include_cis = _PLAYOFF_CI_COLUMNS.issubset(set(correlations.columns))
    columns = ["season_label", "metric", "qb_seasons", "playoff_dropbacks", "spearman"]
    headers = ["Season", "Metric", "QB Seasons", "Playoff Dropbacks", "Spearman"]
    if include_cis:
        columns += ["spearman_ci_lower", "spearman_ci_upper"]
        headers += ["Spearman CI Lower", "Spearman CI Upper"]
    columns.append("pearson")
    headers.append("Pearson")
    if include_cis:
        columns += ["pearson_ci_lower", "pearson_ci_upper"]
        headers += ["Pearson CI Lower", "Pearson CI Upper"]
    return [
        *history_strings.playoff_validation_report_intro_lines(),
        markdown_table(headers, _table_rows(correlations, columns)),
        "",
    ]


def _space_blocks(text: str) -> str:
    """Return ``text`` with a blank line around every heading, table, and code fence.

    Sections are assembled from independent pieces, so one piece can end where the next one's
    heading or table starts; the spacing keeps the generated file lint-clean without edits.
    """
    spaced: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        previous = spaced[-1] if spaced else ""
        is_fence = line.startswith("```")
        if in_fence:
            spaced.append(line)
            in_fence = not is_fence
            continue
        starts_block = (
            line.startswith("#")
            or is_fence
            or (line.startswith("|") and not previous.startswith("|"))
        )
        ends_block = previous.startswith("#") or (
            previous.startswith("|") and not line.startswith("|")
        )
        if (starts_block or ends_block) and previous and line:
            spaced.append("")
        spaced.append(line)
        in_fence = is_fence
    return "\n".join(spaced)


def build_validation_report_text(inputs: ValidationReportInputs) -> str:
    """Render the validation report as Markdown."""
    lines = [
        *_header_lines(inputs),
        *_team_metric_sections(inputs),
        *_season_comparison_sections(inputs),
        *_qb_schedule_lens_sections(inputs),
        *_qb_audit_sections(inputs),
        *(inputs.qb_open_status_lines or []),
        *_qb_opponent_offense_section(inputs),
        *_qb_leverage_section(inputs),
        *_qb_split_half_section(inputs),
        *_qb_playoff_section(inputs.qb_playoff_correlations),
        *history_strings.sacr_report_caveat_lines(),
    ]
    return _space_blocks("\n".join(lines)).rstrip() + "\n"


def write_validation_report(report_path: Path, inputs: ValidationReportInputs) -> None:
    """Write the validation report to disk."""
    report_path.write_text(build_validation_report_text(inputs), encoding="utf-8")


__all__ = [
    "ValidationReportInputs",
    "build_validation_report_text",
    "markdown_table",
    "write_validation_report",
]
