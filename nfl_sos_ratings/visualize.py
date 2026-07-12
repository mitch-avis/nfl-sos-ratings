"""Generate visualizations from the NFL Strength of Schedule combined.csv output.

Run this script after main.py has produced output/combined.csv:

    python visualize.py

Plots are saved to output/plots/.
"""

import os
import sys
from pathlib import Path

# Allow direct execution via `python nfl_sos_ratings/visualize.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nfl_sos_ratings.config import OUTPUT_DIR, SEASON

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

sns.set_theme(style="whitegrid", font_scale=0.85)

# ---------------------------------------------------------------------------
# Stat specifications: (column_base_name, display_label, higher_diff_is_team_advantage)
# "higher_diff_is_team_advantage" = True  →  diff > 0 means team has an edge
#                                  = False →  diff < 0 means team has an edge
# ---------------------------------------------------------------------------

OFFENSE_SPECS: list[tuple[str, str, bool]] = [
    ("points_for", "Pts Scored/G", True),
    ("total_yards", "Total Yds/G", True),
    ("passing_yards", "Pass Yds/G", True),
    ("rushing_yards", "Rush Yds/G", True),
    ("passing_epa", "Pass EPA/G", True),
    ("rushing_epa", "Rush EPA/G", True),
]

DEFENSE_SPECS: list[tuple[str, str, bool]] = [
    ("points_allowed", "Pts Allowed/G", False),
    ("sacks_suffered", "Sacks Taken/G", False),
    ("passing_interceptions", "INT Thrown/G", False),
    ("def_sacks", "Def Sacks/G", True),
    ("def_interceptions", "Def INT/G", True),
    ("def_pass_defended", "Pass Def/G", True),
]

QB_SPECS: list[tuple[str, str, bool]] = [
    ("qb_passer_rating", "QB Rating", True),
    ("qb_completion_percentage_above_expectation", "CPOE", True),
    ("qb_aggressiveness", "Aggressiveness", True),
    ("qb_avg_intended_air_yards", "Intended Air Yds", True),
    ("qb_avg_time_to_throw", "Time to Throw (s)", False),
    ("qb_avg_air_yards_to_sticks", "Air Yds to Sticks", True),
]

# Overall summary diffs (team minus opponent average)
OVERALL_DIFF_SPECS: list[tuple[str, str, bool]] = [
    ("points_for", "Pts Scored/G", True),
    ("points_allowed", "Pts Allowed/G", False),
    ("total_yards", "Total Yds/G", True),
    ("passing_epa", "Pass EPA/G", True),
    ("rushing_epa", "Rush EPA/G", True),
]

# Raw opp_ columns to show in schedule-strength views
OFFENSE_OPP_SPECS: list[tuple[str, str]] = [
    ("opp_points_for", "Opp Pts Scored/G"),
    ("opp_total_yards", "Opp Total Yds/G"),
    ("opp_passing_yards", "Opp Pass Yds/G"),
    ("opp_rushing_yards", "Opp Rush Yds/G"),
    ("opp_passing_epa", "Opp Pass EPA/G"),
    ("opp_rushing_epa", "Opp Rush EPA/G"),
]

DEFENSE_OPP_SPECS: list[tuple[str, str]] = [
    ("opp_points_allowed", "Opp Pts Allowed/G"),
    ("opp_def_sacks", "Opp Def Sacks/G"),
    ("opp_def_interceptions", "Opp Def INT/G"),
    ("opp_def_pass_defended", "Opp Pass Def/G"),
    ("opp_def_tackles_for_loss", "Opp TFL/G"),
    ("opp_def_qb_hits", "Opp QB Hits/G"),
]

QB_OPP_SPECS: list[tuple[str, str]] = [
    ("opp_qb_passer_rating", "Opp QB Rating"),
    ("opp_qb_completion_percentage_above_expectation", "Opp QB CPOE"),
    ("opp_qb_aggressiveness", "Opp QB Aggressiveness"),
    ("opp_qb_avg_intended_air_yards", "Opp QB Intended Air Yds"),
    ("opp_qb_avg_time_to_throw", "Opp QB Time to Throw"),
]

OVERALL_OPP_SPECS: list[tuple[str, str]] = [
    ("opp_points_for", "Opp Pts Scored/G"),
    ("opp_points_allowed", "Opp Pts Allowed/G"),
    ("opp_total_yards", "Opp Total Yds/G"),
    ("opp_passing_epa", "Opp Pass EPA/G"),
    ("opp_rushing_epa", "Opp Rush EPA/G"),
]

OFFENSE_TEAM_SPECS: list[tuple[str, str]] = [
    ("points_for", "Team Pts Scored/G"),
    ("total_yards", "Team Total Yds/G"),
    ("passing_yards", "Team Pass Yds/G"),
    ("rushing_yards", "Team Rush Yds/G"),
    ("passing_epa", "Team Pass EPA/G"),
    ("rushing_epa", "Team Rush EPA/G"),
]

DEFENSE_TEAM_SPECS: list[tuple[str, str]] = [
    ("points_allowed", "Team Pts Allowed/G"),
    ("def_sacks", "Team Def Sacks/G"),
    ("def_interceptions", "Team Def INT/G"),
    ("def_pass_defended", "Team Pass Def/G"),
    ("def_tackles_for_loss", "Team TFL/G"),
    ("def_qb_hits", "Team QB Hits/G"),
]

QB_TEAM_SPECS: list[tuple[str, str]] = [
    ("qb_passer_rating", "Team QB Rating"),
    ("qb_completion_percentage_above_expectation", "Team QB CPOE"),
    ("qb_aggressiveness", "Team QB Aggressiveness"),
    ("qb_avg_intended_air_yards", "Team QB Intended Air Yds"),
    ("qb_avg_time_to_throw", "Team QB Time to Throw"),
]

OVERALL_TEAM_SPECS: list[tuple[str, str]] = [
    ("points_for", "Team Pts Scored/G"),
    ("points_allowed", "Team Pts Allowed/G"),
    ("total_yards", "Team Total Yds/G"),
    ("passing_epa", "Team Pass EPA/G"),
    ("rushing_epa", "Team Rush EPA/G"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_diff_specs(
    df: pl.DataFrame, specs: list[tuple[str, str, bool]]
) -> list[tuple[str, str, bool]]:
    """Filter specs to those whose diff_ column exists in df."""
    return [s for s in specs if f"diff_{s[0]}" in df.columns]


def _available_opp_specs(df: pl.DataFrame, specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filter opponent stat specs to columns that exist in df."""
    return [s for s in specs if s[0] in df.columns]


def _draw_diff_bars(
    ax: Axes,
    teams: list[str],
    diffs: list[float],
    label: str,
    higher_is_better: bool,
) -> None:
    """Draw a single horizontal bar chart of diff values."""
    # Convert to "advantage score" so sorting always puts best team at top
    advantage = [v if higher_is_better else -v for v in diffs]
    order = np.argsort(advantage)  # ascending → best at the top of barh
    s_teams = [teams[i] for i in order]
    s_diffs = [diffs[i] for i in order]
    s_adv = [advantage[i] for i in order]

    colors = ["#0f9960" if a >= 0 else "#c0392b" for a in s_adv]
    ax.barh(s_teams, s_diffs, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_facecolor("#fffdf7")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_stat_bars(ax: Axes, teams: list[str], values: list[float], label: str) -> None:
    """Draw a ranked horizontal chart for raw opponent stats."""
    order = np.argsort(values)
    s_teams = [teams[i] for i in order]
    s_values = [values[i] for i in order]

    colors = sns.color_palette("crest", n_colors=len(s_values))
    ax.barh(s_teams, s_values, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_facecolor("#fffdf7")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save_fig(fig: Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _sorted_rating_rows(df: pl.DataFrame, rating_col: str) -> tuple[list[str], list[float]]:
    """Return teams and values sorted descending by the selected rating column."""
    ranked = df.sort(rating_col, descending=True)
    teams = ranked.select("team").to_series().cast(pl.String).to_list()
    values = ranked.select(rating_col).to_series().cast(pl.Float64).fill_null(0.0).to_list()
    return teams, values


def _clear_stale_season_plots(files_to_keep: set[str]) -> None:
    """Delete stale season plots so only currently selected outputs remain."""
    season_prefix = f"{SEASON}_"
    for name in os.listdir(PLOTS_DIR):
        if not name.startswith(season_prefix) or not name.endswith(".png"):
            continue
        if name in files_to_keep:
            continue
        os.remove(os.path.join(PLOTS_DIR, name))


def _filter_qb_plot_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return qualified QB rows for player-level QB plots when available."""
    if "qb_is_eligible" in df.columns:
        return df.filter(pl.col("qb_is_eligible"))
    return df


def _qb_display_labels(df: pl.DataFrame) -> list[str]:
    """Return readable QB plot labels without collapsing players to teams."""
    if "qb_name" in df.columns and "team" in df.columns:
        rows = df.select(["qb_name", "team"]).to_dicts()
        return [f"{row['qb_name']} ({row['team']})" for row in rows]
    if "qb_name" in df.columns:
        return df.select("qb_name").to_series().cast(pl.String).to_list()
    return df.select("team").to_series().cast(pl.String).to_list()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_diff_grid(
    df: pl.DataFrame,
    specs: list[tuple[str, str, bool]],
    title: str,
    filename: str,
    ncols: int = 3,
) -> None:
    """Multi-panel horizontal bar chart — one panel per diff stat."""
    available = _available_diff_specs(df, specs)
    if not available:
        print(f"  Skipping {filename}: no matching diff_ columns found.")
        return

    teams = df.select("team").to_series().to_list()
    n = len(available)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6.5), squeeze=False)
    fig.patch.set_facecolor("#f6f3eb")
    fig.suptitle(
        f"{title}  (Team − Opp Avg) — {SEASON} NFL",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    ax_flat: list[Axes] = axes.flatten().tolist()  # type: ignore[attr-defined]

    for i, (col, label, higher_good) in enumerate(available):
        vals = df.select(f"diff_{col}").to_series().cast(pl.Float64).fill_null(0.0).to_list()
        _draw_diff_bars(ax_flat[i], teams, vals, label, higher_good)

    for j in range(len(available), len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_sos_overview(
    df: pl.DataFrame,
    filename: str,
    specs: list[tuple[str, str]] | None = None,
    title: str = "Opponent Strength Profile",
) -> None:
    """Ranked bar charts of raw opponent-strength metrics."""
    chosen_specs = specs if specs is not None else OVERALL_OPP_SPECS
    available = _available_opp_specs(df, chosen_specs)
    if not available:
        print(f"  Skipping {filename}: no opp_ columns found.")
        return

    teams = df.select("team").to_series().to_list()
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6.5), squeeze=False)
    fig.patch.set_facecolor("#f6f3eb")
    fig.suptitle(
        f"{title} — {SEASON} NFL",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    ax_flat: list[Axes] = axes.flatten().tolist()  # type: ignore[attr-defined]

    for i, (col, label) in enumerate(available):
        vals = df.select(col).to_series().cast(pl.Float64).fill_null(0.0).to_list()
        _draw_stat_bars(ax_flat[i], teams, vals, label)

    for j in range(len(available), len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_diff_heatmap(df: pl.DataFrame, filename: str) -> None:
    """Z-score normalized heatmap of all key diff stats.

    Green = team has the statistical edge; red = opponent has the edge.
    Lower-is-better stats (e.g. points_allowed) are sign-flipped so green
    always represents a team advantage.
    """
    all_specs = OFFENSE_SPECS + DEFENSE_SPECS + QB_SPECS + OVERALL_DIFF_SPECS
    available = _available_diff_specs(df, all_specs)
    if not available:
        print(f"  Skipping {filename}: no diff_ columns found.")
        return

    df_sorted = df.sort("team")
    teams = df_sorted.select("team").to_series().to_list()

    matrix_cols: list[np.ndarray] = []
    col_labels: list[str] = []

    for col, label, higher_good in available:
        raw = np.array(
            df_sorted.select(f"diff_{col}").to_series().cast(pl.Float64).fill_null(0.0).to_list()
        )
        std = raw.std()
        z = (raw - raw.mean()) / std if std > 0 else raw - raw.mean()
        # Flip so green always = team advantage
        matrix_cols.append(z if higher_good else -z)
        col_labels.append(label)

    matrix = np.column_stack(matrix_cols)  # shape: (n_teams, n_stats)

    fig_w = max(14, len(col_labels) * 0.85 + 3)
    fig_h = max(10, len(teams) * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=teams,
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Z-score (green = team advantage)", "shrink": 0.6},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(
        f"Differential Stats Heatmap (Z-scored, team − opp avg) — {SEASON} NFL",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(axis="x", labelsize=7.5, rotation=40)
    ax.tick_params(axis="y", labelsize=7.5, rotation=0)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_composite_sos(df: pl.DataFrame, filename: str) -> None:
    """Single bar chart: composite schedule difficulty score per team.

    The composite is the average z-score across key opponent offensive stats
    (higher = faced tougher opponents).
    """
    opp_cols = [
        ("opp_points_for", True),
        ("opp_total_yards", True),
        ("opp_passing_epa", True),
        ("opp_rushing_epa", True),
    ]
    available = [(c, d) for c, d in opp_cols if c in df.columns]
    if not available:
        print(f"  Skipping {filename}: no opp_ columns found.")
        return

    teams = df.select("team").to_series().to_list()
    z_scores: list[np.ndarray] = []

    for col, _ in available:
        raw = np.array(df.select(col).to_series().cast(pl.Float64).fill_null(0.0).to_list())
        std = raw.std()
        z_scores.append((raw - raw.mean()) / std if std > 0 else raw - raw.mean())

    composite = np.mean(np.column_stack(z_scores), axis=1)
    order = np.argsort(composite)
    s_teams = [teams[i] for i in order]
    s_vals = composite[order]
    colors = ["#e74c3c" if v >= 0 else "#95a5a6" for v in s_vals]

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(s_teams, s_vals, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"Composite Schedule Difficulty — {SEASON} NFL\n"
        "(avg z-score of opp pts, yards & EPA; red = harder schedule)",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("Composite Difficulty Score (z-score)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_adjusted_ratings(df: pl.DataFrame, filename: str) -> None:
    """Three-panel horizontal bar chart of SaOR, SaDR, SaCR.

    Bars are sorted by SaCR (overall) across all three panels so teams stay
    in the same vertical position, making side-by-side reading easy.
    Green = above league average (z > 0), red = below average (z < 0).
    """
    rating_cols = [c for c in ("SaCR", "SaOR", "SaDR") if c in df.columns]
    if not rating_cols:
        print(f"  Skipping {filename}: no SaCR/SaOR/SaDR columns found.")
        return

    # Sort order driven by SaCR (or first available rating)
    sort_col = "SaCR" if "SaCR" in df.columns else rating_cols[0]
    df_sorted = df.sort(sort_col)
    teams = df_sorted.select("team").to_series().to_list()

    labels = {"SaCR": "SaCR — Composite", "SaOR": "SaOR — Offense", "SaDR": "SaDR — Defense"}
    n_panels = len(rating_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 6, 11), squeeze=False)
    fig.suptitle(
        f"Schedule-Adjusted Team Ratings — {SEASON} NFL\n"
        "(z-score: 0 = league avg, +1 = 1 SD above avg)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    ax_flat: list[Axes] = axes.flatten().tolist()  # type: ignore[attr-defined]

    for i, col in enumerate(rating_cols):
        vals = df_sorted.select(col).to_series().cast(pl.Float64).fill_null(0.0).to_list()
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
        ax_flat[i].barh(teams, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
        ax_flat[i].axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_flat[i].set_title(labels.get(col, col), fontsize=10, fontweight="bold", pad=4)
        ax_flat[i].set_xlabel("z-score", fontsize=8)
        ax_flat[i].tick_params(axis="y", labelsize=7.5)
        ax_flat[i].tick_params(axis="x", labelsize=7.5)
        for spine in ("top", "right"):
            ax_flat[i].spines[spine].set_visible(False)
        # Hide y-axis labels on non-leftmost panels to reduce clutter
        if i > 0:
            ax_flat[i].set_yticklabels([])

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_adjusted_rating_single(
    df: pl.DataFrame,
    rating_col: str,
    label: str,
    filename: str,
) -> None:
    """Single full-size horizontal ranking chart for one schedule-adjusted team rating."""
    if rating_col not in df.columns:
        print(f"  Skipping {filename}: no {rating_col} column found.")
        return

    teams, values = _sorted_rating_rows(df, rating_col)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(teams, values, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"{label} — {SEASON} NFL\n(z-score: 0 = league avg, +1 = 1 SD above avg)",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("z-score", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_qb_adjusted_ratings(df: pl.DataFrame, filename: str) -> None:
    """Single-panel QB schedule-adjusted composite ranking chart.

    Expects `team` and `QSaCR` columns from `{SEASON}_qb_combined.csv`.
    """
    if "QSaCR" not in df.columns:
        print(f"  Skipping {filename}: no QSaCR column found.")
        return

    ranked = _filter_qb_plot_df(df).filter(pl.col("QSaCR").is_finite()).sort("QSaCR")
    if ranked.is_empty():
        print(f"  Skipping {filename}: no qualified QB rows with finite QSaCR found.")
        return

    labels = _qb_display_labels(ranked)
    values = (
        ranked.select("QSaCR").to_series().cast(pl.Float64).fill_nan(None).fill_null(0.0).to_list()
    )
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(8, len(labels) * 0.34)))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"QB Schedule-Adjusted Composite (QSaCR) — {SEASON} NFL",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("QSaCR z-score", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_qb_raw_vs_adjusted(df: pl.DataFrame, filename: str) -> None:
    """Scatter plot of raw QB composite versus schedule-adjusted offense rating."""
    if "QRaw" not in df.columns or "QSaOR" not in df.columns:
        print(f"  Skipping {filename}: missing QRaw/QSaOR columns.")
        return

    df = _filter_qb_plot_df(df)
    if df.is_empty():
        print(f"  Skipping {filename}: no qualified QB rows found.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    x = df.select("QRaw").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    y = df.select("QSaOR").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    labels = _qb_display_labels(df)

    ax.scatter(x, y, s=60, color="#0f9960", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for label, raw_score, adjusted_score in zip(labels, x, y, strict=True):
        ax.text(raw_score + 0.02, adjusted_score + 0.02, label, fontsize=6.5)
    ax.set_title(
        f"QB Raw vs Adjusted Offense Rating — {SEASON} NFL", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("QRaw: raw QB performance composite", fontsize=9)
    ax.set_ylabel("QSaOR: schedule-adjusted QB offense rating", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, filename)


def plot_qb_schedule_vs_performance(df: pl.DataFrame, filename: str) -> None:
    """Scatter plot of QB schedule difficulty versus adjusted composite performance."""
    if "QSoS" not in df.columns or "QSaCR" not in df.columns:
        print(f"  Skipping {filename}: missing QSoS/QSaCR columns.")
        return

    df = _filter_qb_plot_df(df)
    if df.is_empty():
        print(f"  Skipping {filename}: no qualified QB rows found.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    x = df.select("QSoS").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    y = df.select("QSaCR").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    labels = _qb_display_labels(df)

    ax.scatter(x, y, s=60, color="#1f78b4", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for label, strength, performance in zip(labels, x, y, strict=True):
        ax.text(strength + 0.02, performance + 0.02, label, fontsize=6.5)
    ax.set_title(
        f"QB Schedule Difficulty vs Adjusted Performance — {SEASON} NFL",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("QSoS: opponent QB-defense difficulty", fontsize=9)
    ax.set_ylabel("QSaCR: final QB schedule-adjusted composite", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, filename)


def plot_qb_raw_vs_schedule(df: pl.DataFrame, filename: str) -> None:
    """Scatter plot of raw QB performance versus QB schedule difficulty (QSoS)."""
    if "QRaw" not in df.columns or "QSoS" not in df.columns:
        print(f"  Skipping {filename}: missing QRaw/QSoS columns.")
        return

    df = _filter_qb_plot_df(df)
    if df.is_empty():
        print(f"  Skipping {filename}: no qualified QB rows found.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    x = df.select("QSoS").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    y = df.select("QRaw").to_series().cast(pl.Float64).fill_null(0.0).to_list()
    labels = _qb_display_labels(df)
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    ax.scatter(x, y, s=60, color="#f39c12", alpha=0.85, edgecolor="white", linewidth=0.6)
    x_pad = 0.25
    y_pad = 0.25
    x_min = float(x_arr.min()) - x_pad
    x_max = float(x_arr.max()) + x_pad
    y_min = float(y_arr.min()) - y_pad
    y_max = float(y_arr.max()) + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    tier_score = x_arr + y_arr
    tier_cuts = np.percentile(tier_score, [75, 50, 25]).tolist()
    tier_lines = [
        (tier_cuts[0], "#2166ac", "Upper tier guide"),
        (tier_cuts[1], "#67a9cf", "Mid tier guide"),
        (tier_cuts[2], "#b2182b", "Lower tier guide"),
    ]
    for intercept, color, _ in tier_lines:
        ax.plot(
            [x_min, x_max],
            [-x_min + intercept, -x_max + intercept],
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=1,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for label, sos, raw in zip(labels, x, y, strict=True):
        ax.text(sos + 0.02, raw + 0.02, label, fontsize=6.5)
    ax.set_title(
        f"QB Raw Performance vs Schedule Difficulty — {SEASON} NFL",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("QSoS: opponent QB-defense difficulty", fontsize=9)
    ax.set_ylabel("QRaw: raw QB performance composite", fontsize=9)

    legend_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#f39c12",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            label="Qualified QB",
        )
    ]
    legend_handles.extend(
        Line2D([], [], color=color, linestyle="--", linewidth=1.1, label=line_label)
        for _, color, line_label in tier_lines
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    _save_fig(fig, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(season: int | None = None) -> None:
    """Generate selected team and QB plots from combined outputs.

    Args:
        season: Override the season to visualize. Defaults to the configured SEASON constant.

    """
    global SEASON
    if season is not None:
        SEASON = season

    combined_path = os.path.join(OUTPUT_DIR, f"{SEASON}_combined.csv")
    if not os.path.exists(combined_path):
        print(f"ERROR: {combined_path} not found. Run main.py first.")
        return

    df = pl.read_csv(combined_path)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    selected_files = {
        f"{SEASON}_sos_composite_ranking.png",
        f"{SEASON}_adjusted_ratings_offense.png",
        f"{SEASON}_adjusted_ratings_defense.png",
        f"{SEASON}_adjusted_ratings_overall.png",
        f"{SEASON}_qb_adjusted_ratings.png",
        f"{SEASON}_qb_raw_vs_schedule.png",
    }
    _clear_stale_season_plots(selected_files)

    print(f"Generating {SEASON} NFL SoS visualizations → {PLOTS_DIR}/\n")

    plot_composite_sos(df, f"{SEASON}_sos_composite_ranking.png")
    plot_adjusted_rating_single(
        df,
        rating_col="SaOR",
        label="Schedule-Adjusted Team Offense (SaOR)",
        filename=f"{SEASON}_adjusted_ratings_offense.png",
    )
    plot_adjusted_rating_single(
        df,
        rating_col="SaDR",
        label="Schedule-Adjusted Team Defense (SaDR)",
        filename=f"{SEASON}_adjusted_ratings_defense.png",
    )
    plot_adjusted_rating_single(
        df,
        rating_col="SaCR",
        label="Schedule-Adjusted Team Composite (SaCR)",
        filename=f"{SEASON}_adjusted_ratings_overall.png",
    )

    qb_combined_path = os.path.join(OUTPUT_DIR, f"{SEASON}_qb_combined.csv")
    if os.path.exists(qb_combined_path):
        qb_df = pl.read_csv(qb_combined_path)
        plot_qb_adjusted_ratings(qb_df, f"{SEASON}_qb_adjusted_ratings.png")
        plot_qb_raw_vs_schedule(qb_df, f"{SEASON}_qb_raw_vs_schedule.png")

    print(f"\nDone! {len(os.listdir(PLOTS_DIR))} plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
