# Planned-Metric ETL Expansion Plan

## Living document rules

Active handoff document for implementing the `status: planned` registry metrics in the ETL and
the category-sectioned UI. Read before substantive changes; update in the same change set when
scope, status, decisions, or next steps change.

## Scope (user request, 2026-07-14)

1. Implement the `planned` catalog metrics from the registry in the ETL, flipping each entry's
   `status` to `implemented` as it lands (TDD; every self-computed metric needs a formula note
   and a correctness test per AGENTS.md).
2. Rework the web UI table views into category-sectioned layouts using the registry's
   category/subcategory data (`/api/metadata` + per-column `column_metadata`).
3. Frontend fixes: duplicate React keys on the detail "All Stats" views; Sched Tier bucket
   thresholds (Tougher ≥ +0.5, Softer ≤ −0.5, Middle otherwise) plus a color gradient
   (Tougher = "better" color); remove the redundant ratings strip under detail-page titles.
4. Align the team/QB stats catalogs, registry payloads, and ETL terminology to the six-view
  model (`Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`,
  `Opponent Per-Game Rates`, `Opponent Per-Play Rates`) without treating `Opponent Context`
  as a standalone category.
5. Prove with focused tests that team and QB opponent outputs still carry genuinely distinct
  opponent-per-game and opponent-per-play series after the terminology refactor.

## Standing decisions

- **Registry stays typed Python** (user asked whether to refactor to YAML + pyyaml now that the
  dependency policy allows it; assessment: YAML would add a dependency, a parse step, and a
  runtime-only validation layer while losing pyright-strict checking of every entry, with no
  consumer that benefits — declined, documented 2026-07-14).
- AGENTS.md dependency policy relaxed: new dependencies allowed with good reason via `.in`
  files + `update_requirements.sh`.
- Season aggregation convention: season values are per-game means of weekly values (existing
  pipeline behavior), except `longest_*`/`fg_long` which aggregate with max.
- Weekly team metrics are added in `team_stats.compute_team_game_stats_from_pbp` (plus new
  helper aggregations joined in); everything downstream (per-game means, opponent profiles,
  `opp_`/`diff_` products, game logs) flows automatically.
- Rating pools are untouched — new metrics are display/context surface only.

## Phases

- **E1 Team Tier 1 (PBP)** — **complete**. 155 registry entries flipped to implemented via the
  new `team_stats_expanded.py` (joined into `compute_team_game_stats_from_pbp`): passing/rushing
  extras, scoring, downs/series, drives & field position, turnovers, penalties, and all
  defensive mirrors (mirror-by-construction: each offense row is renamed onto its opponent).
  Verified on live 2025 data (league norms: 64.6% comp, 38.9% third down, 4.32 YPC, 2.11
  points/drive; league giveaways == takeaways; ratings byte-identical). Combined output grew
  344 -> 809 columns. One rename during rollout: `opp_avg_starting_field_position` ->
  `avg_starting_field_position_allowed` (collided with the `opp_` prefix mechanism — caught by
  write-time validation).
- **E2 Special teams** — **not started** (next ETL phase). Sketch: aggregate kickoff plays by
  kicking team (`defteam` on kickoffs — posteam is the receiving team in nflverse), punts/FG/PAT
  by `posteam`, returns via `return_team`/`touchback`/fair-catch flags; `fg_pct_over_expected`
  deferred (needs a fitted distance model); `avg_opponent_start_after_kickoff` deferred (needs
  next-drive linkage).
- **E3 Receiving display mirrors** — **complete** (14 alias metrics in
  `team_stats._add_receiving_display_mirrors` plus PBP receiving fumbles).
- **E4 QB Tier 1** — **core complete** (11 metrics): official rushing family from weekly
  player stats (`_OFFICIAL_QB_RUSHING_FIELDS` in data_loader), `qb_completion_pct`,
  `qb_yards_per_carry`, `qb_epa_per_carry`, with season totals/per-game/rates in `qb_stats`.
  Remaining QB planned metrics (scramble/designed splits, success rate, clutch extras,
  turnover aggregates, `qb_games_started`/`qb_snap_share`/`qb_plays`) still `planned` —
  they need per-QB PBP rusher attribution in `compute_qb_game_stats_from_pbp`.
- **E5 Tier 2 sources** — **not started**. NGS team passing aggregates
  (attempt-weighted, 2016+), PFR packs (weekly def file aggregates directly to team-week;
  season passing needs `2TM` handling via weekly rows), QBR QB columns (join by `espn_id`
  via `load_players()`; loader `load_espn_qbr()` already exists). All registry entries exist —
  flip `status` per metric as each lands.
- **E6 Category-sectioned views** — **complete**: detail-page "Metric Family" sections now
  come from registry categories/subcategories (`getColumnSection` in metricMetadata.ts,
  hydrated category order from `/api/metadata`; legacy heuristics remain only as a
  pre-hydration fallback), and `ui_data._order_columns_by_category` orders index-table group
  columns by the taxonomy.
- **E7 Frontend fixes** — **complete**: Sched Tier buckets now threshold the raw league
  z-score at +/-0.5 (the old code re-standardized within the faced-opponents sample — the
  reported bug); Sched Tier cells get gradient colors (Tougher = good end, Softer = bad end,
  Middle uncolored); the Unique Opponents table dedupes reserved column ids (fixes the
  duplicate React `games` key from the All Stats group); the redundant five-rating strip
  under the detail-page title was removed.
- **E8 View taxonomy alignment** — **complete** for the backend/docs slice. Delivered in the
  current session: revised both stats catalogs so views are documented separately from
  categories/subcategories; removed standalone `Opponent Context` treatment from the registry
  payload/category lists; mapped opponent columns back onto the team/QB taxonomies for the
  non-ratings views; and added focused regression coverage that proves opponent-per-game vs.
  opponent-per-play outputs are distinct for both teams and QBs.

Deferred with reasons (still `planned` in the registry): `fg_pct_over_expected` (model),
`time_of_possession` family and `timeouts_used` (clock/timeout parsing), `avg_rest_days`,
`one_score_game_record` (non-numeric), `pythagorean_win_pct`/`sos` (season-level post-steps),
`offensive_points` (drive-point attribution nuances), `td_rate_per_drive`,
`first_downs_rush`/`first_downs_pass` (pure duplicates), tackle-accounting PLS extras.

## Handoff notes

- 2026-07-14: plan created; E1 starting.
- 2026-07-14: E1, E3, E4-core, E6, E7 complete in-session; all gates green. Next agent:
  E2 special teams, then E5 Tier 2, then the remaining QB planned metrics (see E4 note).
- 2026-07-14: E8 completed for the six-view taxonomy refactor and opponent-view registry/ETL
  alignment; validation green with `markdownlint`, `ruff format .`, `ruff check .`, `ty check .`,
  `pyright .`, and `pytest`.
