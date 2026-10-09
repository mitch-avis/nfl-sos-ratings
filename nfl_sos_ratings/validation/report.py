"""Markdown rendering for the validation report."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from nfl_sos_ratings.validation.walk_forward import TeamDecision

_LINE_LIMIT = 100
_DECISION_RULE = (
    "The published team rating (TeamRating) is rebuilt each week from that season's earlier "
    "games, with its preseason prior from earlier seasons until a team has played 9 games, "
    "alongside SRS and raw EPA margin built from the same games alone. A margin model fit on "
    "earlier predictions turns each rating gap into a predicted home margin. TeamRating stays the "
    "published headline unless its overall mean absolute error is significantly worse than "
    "RawEPA's or SRS's: the 95% paired-bootstrap interval of the difference lies entirely above "
    "zero. Elo carries every rating across seasons and is shown as a reference only. This rule "
    "was written before the first run, and the prior joined TeamRating after its own "
    "pre-registered test."
)


@dataclass(frozen=True, slots=True)
class ValidationReportInputs:
    """Everything the validation report renders."""

    command: str
    seasons: Sequence[int]
    start_week: int
    metrics: pl.DataFrame
    mae_deltas: pl.DataFrame
    weekly_curves: pl.DataFrame
    decision: TeamDecision
    stability: pl.DataFrame
    qbr_correlations: pl.DataFrame


def _format_value(value: object) -> str:
    """Format one scalar for a Markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_format_value(value) for value in row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def _paragraph(text: str) -> str:
    """Wrap prose to the Markdown line limit."""
    return textwrap.fill(text, width=_LINE_LIMIT)


def _bullet(text: str) -> str:
    """Wrap one list item, indenting continuation lines under the bullet."""
    return textwrap.fill(text, width=_LINE_LIMIT, initial_indent="- ", subsequent_indent="  ")


def _table(frame: pl.DataFrame, columns: Sequence[str], headers: Sequence[str]) -> str:
    """Render the named frame columns as a Markdown table."""
    return markdown_table(headers, [list(row) for row in frame.select(columns).iter_rows()])


def _interval(delta: float, lower: float, upper: float) -> str:
    """Format an MAE difference with its 95% interval."""
    return f"{delta:+.3f} (95% CI {lower:+.3f} to {upper:+.3f})"


def _decision_section(inputs: ValidationReportInputs) -> list[str]:
    """Return the decision rule, its outcome, and every interval that excludes zero."""
    decision = inputs.decision
    verdict = (
        "Decision: adopt. TeamRating is not significantly worse than either comparator."
        if decision.adopted
        else "Decision: do not adopt. TeamRating is significantly worse than at least one "
        "comparator; the result goes to the maintainer without tuning."
    )
    lines = ["## Decision Rule", "", _paragraph(_DECISION_RULE), "", _paragraph(verdict), ""]
    lines.extend(
        _bullet(
            f"TeamRating vs {result.comparator}, overall MAE difference "
            f"{_interval(result.mae_delta, result.ci_lower, result.ci_upper)}"
            + (": significantly worse." if result.significantly_worse else ".")
        )
        for result in decision.comparisons
    )
    lines.extend(["", "## Intervals That Exclude Zero", ""])
    distinguishable = inputs.mae_deltas.filter("distinguishable_from_zero")
    if distinguishable.is_empty():
        lines.append("No paired-bootstrap interval excludes zero.")
    else:
        lines.append(
            _paragraph(
                "Every pair and split whose 95% interval excludes zero, in either direction. A "
                "negative difference favors the first baseline."
            )
        )
        lines.append("")
        lines.extend(
            _bullet(
                f"{row['split']}: {row['baseline_a']} vs {row['baseline_b']}, "
                f"{_interval(row['mae_delta'], row['ci_lower'], row['ci_upper'])}"
            )
            for row in distinguishable.iter_rows(named=True)
        )
    return [*lines, ""]


def _finite_mean(frame: pl.DataFrame, column: str) -> str:
    """Return the mean of a column's finite values to three places, or ``-`` when there are none."""
    values = frame.get_column(column).to_numpy()
    finite = values[np.isfinite(values)]
    return f"{float(finite.mean()):.3f}" if finite.size else "-"


def _qbr_section(qbr_correlations: pl.DataFrame) -> list[str]:
    """Return the ESPN QBR reference table and its mean correlations."""
    lines = ["## ESPN QBR Reference", ""]
    if qbr_correlations.is_empty():
        return [*lines, "No season in range has ESPN QBR.", ""]
    lines.append(
        _paragraph(
            "Per-season correlation between adjusted EPA per dropback and ESPN QBR for qualified "
            f"passers. Mean Pearson {_finite_mean(qbr_correlations, 'pearson')}, mean Spearman "
            f"{_finite_mean(qbr_correlations, 'spearman')}. QBR is a reference, not a fitting "
            "target."
        )
    )
    lines.extend(
        [
            "",
            _table(
                qbr_correlations,
                ["season", "joined_rows", "pearson", "spearman"],
                ["Season", "QBs", "Pearson", "Spearman"],
            ),
            "",
        ]
    )
    return lines


def build_validation_report_text(inputs: ValidationReportInputs) -> str:
    """Render the validation report as Markdown."""
    seasons = sorted(inputs.seasons)
    lines = [
        "# Validation Report",
        "",
        _paragraph(
            f"Evaluation seasons: {seasons[0]}-{seasons[-1]}. "
            f"Prediction weeks start at {inputs.start_week}."
        ),
        "",
        "## Command",
        "",
        "```bash",
        inputs.command,
        "```",
        "",
        *_decision_section(inputs),
        "## Walk-Forward Summary",
        "",
        _table(
            inputs.metrics.sort(["split", "mae"]),
            ["baseline", "split", "games", "mae", "rmse"],
            ["Baseline", "Split", "Games", "MAE", "RMSE"],
        ),
        "",
        "## Paired Bootstrap MAE Differences",
        "",
        _paragraph("A negative difference favors Baseline A."),
        "",
        _table(
            inputs.mae_deltas,
            ["baseline_a", "baseline_b", "split", "games", "mae_delta", "ci_lower", "ci_upper"],
            ["Baseline A", "Baseline B", "Split", "Games", "MAE Diff", "CI Lower", "CI Upper"],
        ),
        "",
        "## Weekly MAE",
        "",
        _table(
            inputs.weekly_curves.sort(["week", "baseline"]),
            ["week", "baseline", "games", "mae", "rmse"],
            ["Week", "Baseline", "Games", "MAE", "RMSE"],
        ),
        "",
        "## Year-Over-Year Stability",
        "",
        _paragraph(
            "Correlation of each metric with the same team's or qualified passer's value the "
            "next season. Informative only; it does not gate anything."
        ),
        "",
        _table(
            inputs.stability,
            ["entity", "metric", "paired_rows", "pearson", "spearman"],
            ["Entity", "Metric", "Pairs", "Pearson", "Spearman"],
        ),
        "",
        *_qbr_section(inputs.qbr_correlations),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_validation_report(report_path: Path, inputs: ValidationReportInputs) -> None:
    """Write the validation report to disk."""
    report_path.write_text(build_validation_report_text(inputs), encoding="utf-8")


__all__ = [
    "ValidationReportInputs",
    "build_validation_report_text",
    "markdown_table",
    "write_validation_report",
]
