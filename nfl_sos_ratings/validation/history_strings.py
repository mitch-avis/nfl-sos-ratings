"""Sanctioned historical validation-report language.

This module is the single allowed home for campaign-history vocabulary that must
remain in the generated validation report for archival continuity.
All other source files should use durable product language instead.
"""

from __future__ import annotations

from typing import cast

import polars as pl

TEAM_DECISION_RULE_HEADING = "## Stage 3c Decision Rule"
TEAM_OUTCOME_HEADING = "## Stage 3c Team Outcome"
QB_OPEN_STATUS_HEADING = "## QB Open Status"
REPORT_HISTORY_HEADING = "## Stage 3 History"
LEAGUE_CRITERION_HEADING = "## Stage 3b Criterion"
LEAGUE_ACCEPTANCE_HEADING = "## Stage 3b Acceptance Check"
OPPONENT_OFFENSE_SECTION_HEADING = "## D5 Opponent-Offense Effect"
LEVERAGE_SECTION_HEADING = "## D6 Leverage Profile and Filtered Variant"
SPLIT_HALF_DIAGNOSTICS_HEADING = "## Stage 3d D1 Split-Half Diagnostics"
PLAYOFF_VALIDATION_HEADING = "## Stage 3d D3 Playoff Validation"
SACR_CAVEAT_HEADING = "## SaCR Caveat"
REGRESSION_NOTE_HEADING = "## Block R Regression Note"


def _find_pairwise_mae_row(
    mae_deltas: pl.DataFrame,
    baseline_a: str,
    baseline_b: str,
    split: str,
) -> dict[str, object] | None:
    """Return one pairwise MAE comparison row when present."""
    match = mae_deltas.filter(
        (pl.col("baseline_a") == baseline_a)
        & (pl.col("baseline_b") == baseline_b)
        & (pl.col("split") == split)
    )
    return match.row(0, named=True) if not match.is_empty() else None


def _row_float(row: dict[str, object], key: str) -> float:
    """Return one row value as a float for report rendering."""
    return float(cast(float | int | str, row[key]))


def _row_bool(row: dict[str, object], key: str) -> bool:
    """Return one row value as a bool for report rendering and comparisons."""
    return bool(row[key])


def build_team_report_decision_lines(
    metrics: pl.DataFrame,
    mae_deltas: pl.DataFrame,
    *,
    base_team_stability: dict[str, object] | None,
    t4_team_stability: dict[str, float | int] | None,
    rolling_epa_st_baseline: str,
    play_level_epa_st_baseline: str,
) -> list[str]:
    """Build the archived team decision narrative for the validation report."""
    overall_metrics = {
        str(row["baseline"]): row
        for row in metrics.filter(pl.col("split") == "overall").iter_rows(named=True)
    }
    t2_row = overall_metrics.get(rolling_epa_st_baseline)
    t4_row = overall_metrics.get(play_level_epa_st_baseline)
    srs_row = overall_metrics.get("SRS")
    if t2_row is None or t4_row is None or srs_row is None:
        return []

    candidate = (
        play_level_epa_st_baseline
        if float(t4_row["mae"]) < float(t2_row["mae"])
        else rolling_epa_st_baseline
    )
    candidate_row = t4_row if candidate == play_level_epa_st_baseline else t2_row
    candidate_vs_raw = _find_pairwise_mae_row(mae_deltas, candidate, "RawEPA", "overall")
    candidate_vs_saovr = _find_pairwise_mae_row(mae_deltas, candidate, "SaOvR", "overall")
    candidate_vs_srs = _find_pairwise_mae_row(mae_deltas, candidate, "SRS", "overall")
    t4_vs_t2 = _find_pairwise_mae_row(
        mae_deltas,
        play_level_epa_st_baseline,
        rolling_epa_st_baseline,
        "overall",
    )

    stability_ok = False
    if (
        candidate == play_level_epa_st_baseline
        and base_team_stability is not None
        and t4_team_stability is not None
    ):
        stability_ok = float(t4_team_stability["pearson"]) >= _row_float(
            base_team_stability,
            "pearson",
        ) and float(t4_team_stability["spearman"]) >= _row_float(base_team_stability, "spearman")

    if candidate == rolling_epa_st_baseline:
        stability_ok = True

    promotion_pass = bool(
        candidate_vs_raw is not None
        and candidate_vs_saovr is not None
        and candidate_vs_srs is not None
        and _row_bool(candidate_vs_raw, "distinguishable_from_zero")
        and _row_float(candidate_vs_raw, "mae_delta") < 0.0
        and _row_bool(candidate_vs_saovr, "distinguishable_from_zero")
        and _row_float(candidate_vs_saovr, "mae_delta") < 0.0
        and _row_float(candidate_row, "mae") < _row_float(srs_row, "mae")
        and _row_float(candidate_row, "rmse") < _row_float(srs_row, "rmse")
        and _row_float(candidate_vs_srs, "ci_lower") <= 0.0
        and stability_ok
    )

    t4_displacement_lines = [
        f"- Play-level displacement check: {play_level_epa_st_baseline} overall MAE "
        f"{_row_float(t4_row, 'mae'):.3f} and RMSE {_row_float(t4_row, 'rmse'):.3f}\n  versus "
        f"{rolling_epa_st_baseline} MAE {_row_float(t2_row, 'mae'):.3f} and RMSE "
        f"{_row_float(t2_row, 'rmse'):.3f}."
    ]
    if t4_vs_t2 is not None:
        t4_displacement_lines.append(
            "  Bootstrap delta "
            f"{_row_float(t4_vs_t2, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(t4_vs_t2, 'ci_lower'):.3f}, {_row_float(t4_vs_t2, 'ci_upper'):.3f}] "
            f"and P(A<=B) {_row_float(t4_vs_t2, 'probability_baseline_a_not_worse'):.3f}."
        )

    lines = [
        TEAM_DECISION_RULE_HEADING,
        "",
        "> A candidate team backbone is promoted to the published ratings if, on the full held-out",
        "> walk-forward window: (1) it is significantly better than RawEPA and than the Stage 1",
        "> SaOvR (95% paired-bootstrap CI excluding zero); (2) it is numerically better than SRS",
        "> on both overall MAE and overall RMSE, and not significantly worse than SRS; and (3)",
        "> adopting it does not degrade team year-over-year stability below the Stage 3 recorded",
        "> value. Statistical parity with SRS plus the construct advantages (schedule-adjusted,",
        "> outcome-free components, unit-level decomposition) is sufficient and will be stated",
        "> plainly, as parity, in the methodology documentation — never overclaimed as",
        "> superiority.",
        "",
        '- Rationale: the stricter "beat SRS with CI clearing zero" bar is statistically',
        "  unattainable on this sample, and the current report already shows SRS itself does not",
        "  separate from RawEPA at 95%.",
        "",
        TEAM_OUTCOME_HEADING,
        "",
        f"- Candidate selected for the final Stage 3c gate: {candidate}.",
        *t4_displacement_lines,
    ]

    if candidate_vs_raw is not None:
        lines.append(
            "- Candidate vs RawEPA: MAE delta "
            f"{_row_float(candidate_vs_raw, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_raw, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_raw, 'ci_upper'):.3f}]\n  "
            f"and P(A<=B) {_row_float(candidate_vs_raw, 'probability_baseline_a_not_worse'):.3f}."
        )
    if candidate_vs_saovr is not None:
        lines.append(
            "- Candidate vs Stage 1 SaOvR: MAE delta "
            f"{_row_float(candidate_vs_saovr, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_saovr, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_saovr, 'ci_upper'):.3f}]\n  "
            f"and P(A<=B) {_row_float(candidate_vs_saovr, 'probability_baseline_a_not_worse'):.3f}."
        )
    if candidate_vs_srs is not None:
        lines.append(
            "- Candidate vs SRS: overall MAE/RMSE "
            f"{_row_float(candidate_row, 'mae'):.3f}/"
            f"{_row_float(candidate_row, 'rmse'):.3f} versus "
            f"{_row_float(srs_row, 'mae'):.3f}/{_row_float(srs_row, 'rmse'):.3f}."
        )
        lines.append(
            "  Bootstrap delta "
            f"{_row_float(candidate_vs_srs, 'mae_delta'):.3f} with 95% CI "
            f"[{_row_float(candidate_vs_srs, 'ci_lower'):.3f}, "
            f"{_row_float(candidate_vs_srs, 'ci_upper'):.3f}] and P(A<=B) "
            f"{_row_float(candidate_vs_srs, 'probability_baseline_a_not_worse'):.3f}."
        )

    if (
        candidate == play_level_epa_st_baseline
        and base_team_stability is not None
        and t4_team_stability is not None
    ):
        lines.append(
            f"- Stability guard: {play_level_epa_st_baseline} Pearson/Spearman "
            f"{float(t4_team_stability['pearson']):.3f}/{float(t4_team_stability['spearman']):.3f} "
            f"versus Stage 3 SaOvR {_row_float(base_team_stability, 'pearson'):.3f}/"
            f"{_row_float(base_team_stability, 'spearman'):.3f}."
        )

    promotion_label = "Pass" if promotion_pass else "Fail"
    lines.append(f"- Promotion decision under the fixed Stage 3c rule: {promotion_label}.")
    lines.append("")
    return lines


def build_qb_report_status_lines() -> list[str]:
    """Return the archived quarterback-methodology status note."""
    return [
        QB_OPEN_STATUS_HEADING,
        "",
        "- The published QB composite target and weights remain unchanged, and the split-half",
        "  companion metric is not promoted to a published surface.",
        "- The earlier QB audit continues to stand as a positive linear-adjustment result:",
        "  the additive adjustment operated at full strength in EPA units, the identity checks",
        "  held, and the fixed-defense and lighter-defense-penalty variants were correctly not",
        "  adopted.",
        "- The only remaining QB follow-up is the opponent-context batch below. If those checks",
        "  also come back null, the current published composite stands as the system's answer.",
    ]


def report_history_lines() -> list[str]:
    """Return the archived report-history heading and paragraph."""
    return [
        REPORT_HISTORY_HEADING,
        "",
        (
            "The original Stage 3 headline compared prior-carrying Elo against "
            "within-season-only backbones."
        ),
        "That result is preserved here as history rather than deleted or rewritten.",
        "",
    ]


def report_league_criterion_lines() -> list[str]:
    """Return the archived league-split criterion language."""
    return [
        LEAGUE_CRITERION_HEADING,
        "",
        "Stage 3b re-registers the validation target into information-matched leagues.",
        "",
        (
            "- League 1 is binding: within-season-only team backbones must beat SRS and "
            "RawEPA on held-out\n  MAE, with paired-bootstrap support."
        ),
        (
            "- League 2 is informative: prior-carrying forecast-only variants can be "
            "compared against Elo,\n  but that is not the binding published-rating gate."
        ),
        "",
    ]


def report_league_acceptance_heading() -> str:
    """Return the archived acceptance-check heading."""
    return LEAGUE_ACCEPTANCE_HEADING


def opponent_offense_report_heading_lines() -> list[str]:
    """Return the archived opponent-offense heading."""
    return [OPPONENT_OFFENSE_SECTION_HEADING, ""]


def leverage_report_heading_lines() -> list[str]:
    """Return the archived leverage heading."""
    return [LEVERAGE_SECTION_HEADING, ""]


def split_half_report_heading_lines() -> list[str]:
    """Return the archived split-half heading."""
    return [SPLIT_HALF_DIAGNOSTICS_HEADING, ""]


def playoff_validation_report_intro_lines() -> list[str]:
    """Return the archived playoff-validation heading and interpretation note."""
    return [
        PLAYOFF_VALIDATION_HEADING,
        "",
        "- Interpretation rule: whichever metric best predicts playoff performance is evidence",
        "  about that metric, not about 2025 specifically.",
        "  If QSaCR wins, that is vindicating evidence for the current composite and must be",
        "  recorded as such.",
        "",
    ]


def sacr_report_caveat_lines() -> list[str]:
    """Return the archived caveat about the frozen composite evaluation line."""
    return [
        SACR_CAVEAT_HEADING,
        "",
        "SaCR may be evaluated as a secondary line with a caveat:",
        "its frozen Stage 2 weights were fit on the full 1999-2025 history.",
        "A walk-forward SaCR line over that same window has look-ahead in the weights.",
        (
            "SaOvR is the headline walk-forward metric because it does not depend on "
            "a fitted Stage 2 weight snapshot."
        ),
        "",
    ]


def report_regression_note_lines() -> list[str]:
    """Return the archived regression-note block for the validation report."""
    return [
        REGRESSION_NOTE_HEADING,
        "",
        (
            "- A Stage 3c regression combined pooled offense/defense reference arrays with\n  "
            "current-season-only special-teams reference values, causing the team ratings path\n  "
            "to raise a NumPy broadcast error before `*_combined.parquet` and\n  "
            "`*_ratings.parquet` wrote."
        ),
        (
            "- The fix backfills historical `st_rating` values from\n  "
            "`*_simultaneous_team_adjustments.parquet` when rebuilding pooled team references\n  "
            "and makes the multi-season pipeline exit non-zero with a failure summary if any\n  "
            "season data step fails."
        ),
        "",
    ]
