"""Tests for the validation report's Markdown structure."""

import polars as pl

from nfl_sos_ratings.validation.report import ValidationReportInputs, build_validation_report_text

MARKDOWN_LINE_LIMIT = 100


def _signal_summary() -> pl.DataFrame:
    """Return a one-row pooled signal summary like the diagnostics produce."""
    return pl.DataFrame(
        {
            "scope": ["pooled"],
            "season": [None],
            "rows": [10],
            "total_dropbacks": [400.0],
            "slope": [-0.04],
            "correlation": [-0.01],
            "ci_lower": [-0.13],
            "ci_upper": [0.05],
            "direction_positive_count": [3],
            "direction_total_count": [7],
            "direction_p_value": [0.9],
        }
    )


def _report_lines() -> list[str]:
    """Render a report whose status list runs straight into the signal and split-half sections."""
    inputs = ValidationReportInputs(
        metrics=pl.DataFrame(
            {
                "baseline": ["SRS"],
                "split": ["overall"],
                "season": [None],
                "games": [10],
                "mae": [10.5],
                "rmse": [13.0],
            }
        ),
        stability=pl.DataFrame(
            {
                "metric": ["QSaCR", "qb_passer_rating", "qb_any_a"],
                "entity": ["qb", "qb", "qb"],
                "paired_rows": [5, 5, 5],
                "pearson": [0.479, 0.413, 0.338],
                "spearman": [0.466, 0.416, 0.330],
            }
        ),
        qbr_correlations=pl.DataFrame(
            {"season": [2025], "joined_rows": [30], "pearson": [0.9], "spearman": [0.88]}
        ),
        seasons=[2024, 2025],
        start_week=5,
        command="uv run nfl-sos-ratings validate",
        qb_open_status_lines=["## QB Open Status", "", "- Still open."],
        qb_opponent_offense_summary=_signal_summary(),
        qb_opponent_offense_decision={"decision": "not_supported"},
        qb_split_half_primary=_signal_summary(),
        qb_split_half_decision={
            "decision": "not_supported",
            "primary_gate_supported": False,
            "placebo_is_symmetric": True,
        },
    )
    return build_validation_report_text(inputs).split("\n")


def test_every_heading_has_blank_lines_around_it() -> None:
    lines = _report_lines()
    for index, line in enumerate(lines):
        if line.startswith("#") and index > 0:
            assert lines[index - 1] == "", f"no blank line before {line!r}"
            assert lines[index + 1] == "", f"no blank line after {line!r}"


def test_every_table_has_blank_lines_around_it() -> None:
    lines = _report_lines()
    for index, line in enumerate(lines):
        if line.startswith("|") and not lines[index - 1].startswith("|"):
            assert lines[index - 1] == "", f"no blank line before table row {line!r}"
        if line.startswith("|") and index + 1 < len(lines) and not lines[index + 1].startswith("|"):
            assert lines[index + 1] == "", f"no blank line after table row {line!r}"


def test_no_prose_line_exceeds_the_markdown_line_limit() -> None:
    lines = _report_lines()
    long_lines = [
        line for line in lines if len(line) > MARKDOWN_LINE_LIMIT and not line.startswith("|")
    ]
    assert long_lines == []
