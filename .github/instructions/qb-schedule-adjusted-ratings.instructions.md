---
description: "Implementation roadmap for quarterback schedule-adjusted ratings in nfl-sos-ratings"
applyTo: "nfl_sos_ratings/**/*.py,tests/**/*.py,README.md"
---

# QB Schedule-Adjusted Ratings Roadmap

This file defines the project plan for extending team schedule-strength analysis to individual NFL
quarterbacks. Future agents should use this roadmap as the implementation sequence unless the user
asks for a different order.

## Goal

Measure each quarterback's season performance relative to the difficulty of defenses they faced, in
a way that is comparable league-wide and auditable from intermediate outputs.

## Target Outputs

- `output/{SEASON}_qb_per_game_stats.csv`: QB per-game season stats with volume fields.
- `output/{SEASON}_qb_opponent_profiles.csv`: averaged defensive profile of opponents faced by each
  QB.
- `output/{SEASON}_qb_combined.csv`: QB stats + opponent stats + diff columns.
- `output/{SEASON}_qb_ratings.csv`: `QSaCR`, `QSaOR`, and `QSoS` leaderboard.
- `output/plots/{SEASON}_qb_*`: QB-focused charts (leaderboard, raw vs adjusted, difficulty views).

## Implementation Phases

### Phase 1: Canonical QB Season Table

1. Build a canonical QB-game dataset with one row per team-week primary QB.
2. Add a QB season aggregation module for per-game stats and eligibility fields:
   - `qb_games_played`
   - `qb_attempts_total`
   - optional per-game attempt rate fields
3. Add baseline tests for aggregation and eligibility behavior.
4. Emit `qb_per_game_stats` CSV from `main.py` without changing existing team outputs.

### Phase 2: QB Opponent Profile Engine

1. Create QB-specific opponent profile logic keyed by QB season row.
2. Ensure head-to-head self-game exclusion to avoid circularity.
3. Support both unweighted and volume-weighted averaging modes.
4. Emit `qb_opponent_profiles` and diagnostic opponent details.

### Phase 3: QB Differentials and Composite Construction

1. Build `diff_qb_*` columns from paired QB/opponent metrics.
2. Define QB performance pools and directionality (higher-is-better vs lower-is-better).
3. Derive data-driven weights with thresholded correlations and equal-weight fallback.
4. Produce raw QB composite score before schedule adjustment.

### Phase 4: Schedule Adjustment and Final Ratings

1. Build `QSoS` from opponent defensive context columns.
2. Blend raw composite with schedule signal via calibrated constant(s).
3. Standardize outputs as z-scores and include percentile rank columns.
4. Emit `QSaOR`, `QSoS`, `QSaCR` in `qb_ratings.csv`.

### Phase 5: Visualizations and Documentation

1. Add QB-focused ranking and calibration plots.
2. Add README method section, formulas, and interpretation guidance.
3. Document eligibility thresholds and caveats.
4. Keep naming and season-prefix conventions consistent with existing outputs.

## Testing Strategy

- Use synthetic Polars DataFrames in unit tests.
- Cover happy path + sparse-data/guard paths.
- Add integration tests for new CSV creation in `main.py`.
- Ensure all existing tests still pass to preserve team pipeline behavior.

## Guardrails

- Keep regular season filtering unless user requests expansion.
- Do not break existing output schemas unless explicitly approved.
- Prefer additive changes and backwards compatibility.
- Keep robust handling for missing optional columns.

## Recommended Execution Order

1. Phase 1 (foundation + tests)
2. Phase 2 (opponent profiles)
3. Phase 3 (composites)
4. Phase 4 (adjusted ratings)
5. Phase 5 (plots and docs)
