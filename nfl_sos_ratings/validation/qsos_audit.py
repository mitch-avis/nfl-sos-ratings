"""Standalone QB schedule-strength audit report command."""

from __future__ import annotations

import argparse
from pathlib import Path

from nfl_sos_ratings.config import DATA_DIR, END_YEAR, START_YEAR
from nfl_sos_ratings.validation.diagnostics import (
    compute_qb_defense_spread_summary,
    compute_qb_designed_rush_preview,
    compute_qb_schedule_lens_anchor,
    compute_qb_schedule_lens_divergence,
    compute_qb_schedule_lens_trace,
    compute_qb_season_audit_summary,
)

_ANCHOR_QBS = ("Drake Maye", "Tyler Shough", "Joe Flacco", "J.J. McCarthy")
_RUSH_PREVIEW_QBS = ("Drake Maye", "Lamar Jackson", "Josh Allen", "Matthew Stafford")


def _markdown_table(headers: list[str], rows: list[list[object | None]]) -> str:
    """Return a GitHub-flavored Markdown table."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    def _format(value: object | None) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    body_rows = ["| " + " | ".join(_format(value) for value in row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def build_qsos_audit_markdown(data_dir: Path, seasons: list[int]) -> str:
    """Return the standalone QB schedule-strength audit markdown report."""
    latest_season = seasons[-1]
    qb_schedule_anchor = compute_qb_schedule_lens_anchor(
        data_dir,
        latest_season,
        qb_names=_ANCHOR_QBS,
    )
    qb_schedule_trace = compute_qb_schedule_lens_trace(
        data_dir,
        latest_season,
        qb_names=_ANCHOR_QBS,
    )
    qb_lens_divergence = compute_qb_schedule_lens_divergence(data_dir, latest_season)
    qb_designed_rush_preview = compute_qb_designed_rush_preview(
        data_dir,
        latest_season,
        qb_names=_RUSH_PREVIEW_QBS,
    )
    qb_season_audit = compute_qb_season_audit_summary(data_dir, seasons)
    qb_defense_spread = compute_qb_defense_spread_summary(data_dir, seasons)

    sections = [
        "# QB Schedule-Strength Audit",
        "",
        f"Evaluation seasons: {min(seasons)}-{max(seasons)}.",
        "",
    ]

    if not qb_schedule_anchor.is_empty():
        sections.extend(
            [
                f"## {latest_season} QB Schedule-Lens Anchor",
                "",
                _markdown_table(
                    ["QB", "Games", "Avg Opp SaCR", "Avg Opp SaDR", "Avg Opp SRS"],
                    [
                        [
                            row["qb_name"],
                            row["games"],
                            row["avg_opp_SaCR"],
                            row["avg_opp_SaDR"],
                            row["avg_opp_SRS"],
                        ]
                        for row in qb_schedule_anchor.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    if not qb_schedule_trace.is_empty():
        sections.extend(
            [
                "## QB Schedule-Lens Trace",
                "",
                _markdown_table(
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
                    [
                        [
                            row.get("qb_name"),
                            row.get("raw_value"),
                            row.get("adjusted_value"),
                            row.get("weighted_faced_defense"),
                            row.get("adjustment_delta"),
                            row.get("QSoS"),
                            row.get("faced_opp_SaCR"),
                            row.get("adj_def_qb_epa_per_dropback_faced"),
                        ]
                        for row in qb_schedule_trace.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    if not qb_lens_divergence.is_empty():
        sections.extend(
            [
                "## QB Lens-Divergence Rankings",
                "",
                _markdown_table(
                    [
                        "QB",
                        "Team",
                        "QSoS",
                        "Faced Opp SaCR",
                        "QSoS Rank",
                        "Overall Rank",
                        "Rank Gap",
                    ],
                    [
                        [
                            row.get("qb_name"),
                            row.get("team"),
                            row.get("QSoS"),
                            row.get("faced_opp_SaCR"),
                            row.get("qsos_rank"),
                            row.get("overall_rank"),
                            row.get("rank_gap"),
                        ]
                        for row in qb_lens_divergence.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    if not qb_designed_rush_preview.is_empty():
        sections.extend(
            [
                f"## {latest_season} QB Designed-Rush Preview",
                "",
                _markdown_table(
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
                    [
                        [
                            row.get("qb_name"),
                            row.get("team"),
                            row.get("qb_designed_carries_total"),
                            row.get("qb_designed_epa_per_carry"),
                            row.get("adj_def_rushing_epa_per_offensive_snap_faced"),
                            row.get("adj_qb_designed_rush_epa_per_carry"),
                            row.get("QSoS"),
                            row.get("adj_def_qb_epa_per_dropback_faced"),
                            row.get("faced_opp_SaCR"),
                        ]
                        for row in qb_designed_rush_preview.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    if not qb_season_audit.is_empty():
        sections.extend(
            [
                "## QB Adjustment Audit",
                "",
                _markdown_table(
                    ["Season", "Eligible QBs", "Slope", "Correlation", "Mean Abs Residual"],
                    [
                        [
                            row["season"],
                            row["rows"],
                            row["slope"],
                            row["correlation"],
                            row["mean_abs_identity_residual"],
                        ]
                        for row in qb_season_audit.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    if not qb_defense_spread.is_empty():
        sections.extend(
            [
                "## QB Defense Spread Audit",
                "",
                _markdown_table(
                    ["Season", "Team Defense SD", "QB Defense SD", "QB/Team Ratio"],
                    [
                        [
                            row["season"],
                            row["team_defense_sd"],
                            row["qb_defense_sd"],
                            row["qb_to_team_spread_ratio"],
                        ]
                        for row in qb_defense_spread.iter_rows(named=True)
                    ],
                ),
                "",
            ]
        )

    return "\n".join(sections).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone QB schedule-strength audit."""
    parser = argparse.ArgumentParser(description="Run the standalone QB schedule-strength audit.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory holding Parquet artifacts.")
    parser.add_argument(
        "--start-season", type=int, default=START_YEAR, help="First season to include."
    )
    parser.add_argument("--end-season", type=int, default=END_YEAR, help="Last season to include.")
    parser.add_argument("--output-path", default=None, help="Optional Markdown output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print or write the standalone QB schedule-strength audit markdown."""
    args = _parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    markdown = build_qsos_audit_markdown(Path(args.data_dir), seasons)
    if args.output_path:
        Path(args.output_path).write_text(markdown, encoding="utf-8")
        print(f"Wrote QB schedule-strength audit to {args.output_path}")
        return
    print(markdown, end="")


if __name__ == "__main__":
    main()
