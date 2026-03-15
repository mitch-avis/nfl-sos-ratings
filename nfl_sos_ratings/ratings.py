"""Schedule-Adjusted Team Ratings: SaOR, SaDR, SaCR.

All weights are derived objectively from the data rather than chosen by hand:

1. **Stat weights**: Every stat in the offensive and defensive pools is correlated
   against each team's actual win percentage.  A stat's weight equals its
   correlation with win_pct (normalised so all weights sum to 1).  Stats whose
   correlation falls below MIN_CORRELATION are excluded — they don't carry enough
   signal at n=32 to be useful.

2. **Offense / defense blend**: After building the adjusted offensive and defensive
   composites, we measure each composite's correlation with win_pct.  The blend
   is proportional to those correlations, so the data determines whether offense
   or defense "matters more" for the season being analysed.

3. **Schedule adjustment**: A SoS correction (scaled by SOS_WEIGHT) is still applied
   on top of the raw composites, using opponent profile columns available in
   combined.csv as described below.

All ratings are expressed as z-scores: 0 = league average, ±1 = ±1 SD.

Stat pools
----------
``_OFF_STAT_POOL`` and ``_DEF_STAT_POOL`` list every column that *could* contribute
to the respective composite.  Columns absent from the DataFrame are silently
skipped.  Adding or removing entries here is all that is needed to change which
stats are considered.

Tunable constants
-----------------
``MIN_CORRELATION``  — minimum correlation with win_pct for a stat to receive weight
``SOS_WEIGHT``       — fraction of the schedule-difficulty signal added to the raw
                       composite (0 = no SoS adjustment, 1 = full)
"""

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Stat pools: (column_name, True if higher value = better for the team)
#
# Include all plausibly relevant stats; correlation-threshold filtering removes
# anything that doesn't actually predict winning.  Don't pre-select — let the
# data decide.
# ---------------------------------------------------------------------------

_OFF_STAT_POOL: list[tuple[str, bool]] = [
    ("passing_epa", True),
    ("rushing_epa", True),
    ("points_for", True),
    ("total_yards", True),
    ("passing_yards", True),
    ("rushing_yards", True),
    ("passing_tds", True),
    ("rushing_tds", True),
    ("passing_first_downs", True),
    ("rushing_first_downs", True),
    ("passing_cpoe", True),  # completion % over expectation
    ("sacks_suffered", False),  # sacks taken — bad for offense
    ("passing_interceptions", False),  # interceptions thrown — bad for offense
    ("sack_fumbles_lost", False),
    ("rushing_fumbles_lost", False),
]

_DEF_STAT_POOL: list[tuple[str, bool]] = [
    ("points_allowed", False),  # lower = better defense
    ("def_sacks", True),
    ("def_interceptions", True),
    ("def_pass_defended", True),
    ("def_tackles_for_loss", True),
    ("def_qb_hits", True),
    ("def_fumbles_forced", True),
    ("def_safeties", True),
]

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Minimum correlation with win_pct for a stat to receive any weight.
# At n=32 a correlation below ~0.15 is indistinguishable from noise.
MIN_CORRELATION: float = 0.15

# How strongly schedule difficulty shifts the raw composite.
# 0 = ignore schedule; 1 = equal weight to raw performance.
SOS_WEIGHT: float = 0.25


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore(values: list[float]) -> np.ndarray:
    """Z-score using sample standard deviation (ddof=1)."""
    arr = np.array(values, dtype=np.float64)
    std = float(arr.std(ddof=1))
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def _col(df: pl.DataFrame, name: str) -> np.ndarray | None:
    """Return a DataFrame column as float64 ndarray, or None if absent."""
    if name not in df.columns:
        return None
    return np.array(
        df.select(name).to_series().cast(pl.Float64).fill_null(0.0).to_list(),
        dtype=np.float64,
    )


def _derive_weights(
    df: pl.DataFrame,
    stat_pool: list[tuple[str, bool]],
    win_pct: np.ndarray,
    label: str,
) -> list[tuple[str, float, bool]]:
    """Correlate each stat with win_pct; return (stat, weight, higher_is_better).

    Weight = correlation / sum_of_correlations, so weights sum to 1.
    Stats below MIN_CORRELATION are excluded.  If nothing clears the threshold
    (unlikely) the function falls back to equal weighting.
    """
    candidates: list[tuple[str, float, bool]] = []
    for stat, higher_better in stat_pool:
        v = _col(df, stat)
        if v is None:
            continue
        # Orient so that "good" direction always correlates positively with wins
        signed = v if higher_better else -v
        r = float(np.corrcoef(signed, win_pct)[0, 1])
        if r >= MIN_CORRELATION:
            candidates.append((stat, r, higher_better))

    if not candidates:
        # Fallback: equal weight on every column that exists
        present = [(s, h) for s, h in stat_pool if _col(df, s) is not None]
        n = len(present)
        weight = 1.0 / n if n else 0.0
        candidates = [(s, weight, h) for s, h in present]
        print(f"  [{label}] No stats cleared the correlation threshold — using equal weights.")
        return candidates

    total_r = sum(r for _, r, _ in candidates)
    weighted = [(s, r / total_r, h) for s, r, h in candidates]

    print(f"\n  {label} stat weights (by |r| with win_pct):")
    for stat, w, _ in sorted(weighted, key=lambda x: -x[1]):
        orig_r = w * total_r
        print(f"    {stat:<38}  weight={w:.3f}  r={orig_r:+.3f}")

    return weighted


def _build_composite(df: pl.DataFrame, weights: list[tuple[str, float, bool]]) -> np.ndarray:
    """Weighted average of signed z-scores for the given stat weights."""
    composite = np.zeros(df.height)
    for stat, weight, higher_better in weights:
        v = _col(df, stat)
        if v is None:
            continue
        z = _zscore(v.tolist())
        composite += (z if higher_better else -z) * weight
    return composite


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ratings(df: pl.DataFrame) -> pl.DataFrame:
    """Compute SaOR, SaDR, SaCR for every team in *df*.

    The ``win_pct`` column must be present (added by ``main.py`` via
    ``compute_win_totals``).  If it is absent, the function falls back to equal
    weighting and a 50/50 blend with a warning.

    Returns a DataFrame ``[team, SaOR, SaDR, SaCR]`` sorted by team.
    """
    n = df.height
    teams = df.select("team").to_series().to_list()

    win_pct_arr = _col(df, "win_pct")
    if win_pct_arr is None:
        print("  WARNING: 'win_pct' not found in DataFrame — falling back to equal weights.")
        win_pct_arr = np.full(n, 0.5)
        use_wins = False
    else:
        use_wins = True

    # --- Derive stat weights ---
    off_weights = _derive_weights(df, _OFF_STAT_POOL, win_pct_arr, "Offensive")
    def_weights = _derive_weights(df, _DEF_STAT_POOL, win_pct_arr, "Defensive")

    # --- Schedule difficulty signals ---
    # Offensive SoS: proxy for quality of defenses faced.
    # Lower opp_points_allowed → opponents held teams to fewer points → harder defenses.
    sos_off = np.zeros(n)
    opp_pa = _col(df, "opp_points_allowed")
    if opp_pa is not None:
        sos_off = -_zscore(opp_pa.tolist())  # flip: low = harder = positive signal

    # Defensive SoS: proxy for quality of offenses faced.
    # Higher opp_points_for / opp_passing_epa → more potent opponents.
    sos_def_parts: list[np.ndarray] = []
    for col_name in ("opp_points_for", "opp_passing_epa"):
        v = _col(df, col_name)
        if v is not None:
            sos_def_parts.append(_zscore(v.tolist()))
    sos_def = np.mean(sos_def_parts, axis=0) if sos_def_parts else np.zeros(n)

    # --- Raw composites + schedule adjustment ---
    raw_off = _build_composite(df, off_weights)
    raw_def = _build_composite(df, def_weights)
    adj_off = raw_off + SOS_WEIGHT * sos_off
    adj_def = raw_def + SOS_WEIGHT * sos_def

    # --- Offense / defense blend: data-derived ---
    saor = _zscore(adj_off.tolist())
    sadr = _zscore(adj_def.tolist())

    if use_wins:
        r_off = max(0.0, float(np.corrcoef(saor, win_pct_arr)[0, 1]))
        r_def = max(0.0, float(np.corrcoef(sadr, win_pct_arr)[0, 1]))
        total_r = r_off + r_def
        off_blend = r_off / total_r if total_r > 0 else 0.5
    else:
        off_blend = 0.5

    def_blend = 1.0 - off_blend
    print(
        f"\n  Off/def blend: {off_blend:.1%} offense / {def_blend:.1%} defense"
        f"{'  (data-derived from win_pct correlations)' if use_wins else '  (fallback: 50/50)'}"
    )

    sacr = _zscore((off_blend * saor + def_blend * sadr).tolist())

    return pl.DataFrame(
        {
            "team": teams,
            "SaOR": np.round(saor, 3).tolist(),
            "SaDR": np.round(sadr, 3).tolist(),
            "SaCR": np.round(sacr, 3).tolist(),
        }
    ).sort("team")
