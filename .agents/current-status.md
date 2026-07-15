# Current Project Status

## Purpose

This is the consolidated handoff document for the repo's current state. It replaces completed
one-off plan files as the single place to record:

- what is fully done
- what is still active
- what is deferred
- what the next agent should do first

Update this file in the same change set whenever the repo's active backlog, validation state, or
current-status summary changes.

## Current state

- Status: release-gate green, with the PBP-first pipeline, Parquet contract, metrics registry, and
  analyst UI shell all implemented.
- Active correctness blocker: none currently known in the published team/QB pipeline.
- Primary source of truth for metrics: `nfl_sos_ratings/metrics/`.
- Human-readable metric companions: `docs/stats-catalog.md` and `docs/qb-stats-catalog.md`.
- Active workstreams: remaining metric expansion work and UI/detail-page follow-up.

## Recent work summary

The current uncommitted diff reflects five major work themes.

1. Metrics registry SSOT:
   - added the typed registry package in `nfl_sos_ratings/metrics/`
   - moved rating-pool membership and column metadata into the registry
   - added write-time output-column validation in `main.py`
   - switched published outputs from CSV to Parquet
   - added a direct ESPN QBR Parquet loader
2. PBP/methodology stabilization:
   - canonical QB identity via GSIS/PFR crosswalks
   - official weekly QB/team stat joins from nflverse sources
   - fixed late-game QB outcome attribution
   - clarified explicit QB season-summary naming
   - added season-floor handling for newer upstream sources
3. Metric-surface expansion:
   - added `team_stats_expanded.py`
   - implemented the first large wave of planned team metrics and defensive mirrors
   - added core QB rushing/rate metrics
4. UI/API contract work:
   - added FastAPI metadata and season/game-log endpoints
   - switched the UI data contract to Parquet-backed payloads
   - hydrated frontend metric metadata from `/api/metadata`
   - improved detail pages, grouped-opponent views, compare behavior, and schedule-tier handling
   - replaced the old index/detail metric-family toggles with the six-view control model
     (`Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`,
     `Opponent Per-Game Rates`, `Opponent Per-Play Rates`)
   - moved weekly and unique-opponent detail tables onto the top-of-page control state
   - pinned the detail-page header pane and switched QB detail headers to `QB Detail` plus
     `QB Name - Full Team Name`
5. Docs/tests/plans:
   - added registry, ETL-expansion, and expanded-metric tests
   - refreshed README, UI docs, and plan docs for the Parquet/registry transition
   - built and installed the local wheel successfully with `uv build` and `uv pip install`

## Validation snapshot

Latest recorded green state across the current worktree:

- `ruff format .`
- `ruff check .`
- `ty check .`
- `pyright .`
- `pytest`
- `markdownlint .`
- `uv build && uv pip install dist/nfl_sos_ratings-0.1.0-py3-none-any.whl`
- `cd ui/web && npm exec -- tsc -p tsconfig.detail-tests.json && node --test`
  `src/detailAnalytics.test.mjs src/detailUi.test.mjs`
- `cd ui/web && npm run build`

## Completed workstreams

### PBP overhaul

- PBP-first team and QB loaders are in place.
- Official weekly QB and team stat surfaces are joined from nflverse sources.
- Canonical QB identity duplication and non-QB passer leakage are fixed.
- Team and QB rating paths are release-gate green.
- Simultaneous-adjustment outputs are emitted alongside the diff-based outputs.
- No active correctness blocker is currently known in the live methodology path.

### Metrics registry and Parquet migration

- The typed registry is the SSOT for labels, descriptions, category ordering, polarity,
  implementation status, and rating-pool membership.
- Published outputs, UI loaders, and visualization readers use Parquet.
- API metadata and per-column metadata are served from the registry.
- Historical CSV files may still exist in `data/`, but they are legacy artifacts only.

### UI shell

- Season-aware Teams and QBs index pages exist.
- Team and QB detail pages exist with weekly and grouped-opponent views.
- Compare, filtering, sticky columns, and glossary/tooltip support are shipped.
- Registry-backed category ordering and metric metadata are wired through the API and frontend.

## Active backlog

### A. Metric expansion still open

Tracked in `.agents/metric-expansion-plan.md`.

Still open:

- E2 special teams implementation
- E5 Tier 2 source integration
  - NGS team passing aggregates
  - PFR pack integration
  - ESPN QBR joins onto the published UI surface
- remaining planned QB metrics that need per-QB PBP rusher attribution
- E8 view-taxonomy alignment
  - complete for the catalog/registry/ETL slice and the shipped Teams/QBs page control refactor
  - keep supporting docs aligned with the six-view model as new surfaces land

Still deferred in the registry:

- `fg_pct_over_expected` model-based work
- `avg_opponent_start_after_kickoff` next-drive linkage
- `time_of_possession` family and `timeouts_used` clock parsing
- `avg_rest_days`, `one_score_game_record`, `pythagorean_win_pct`, `sos`
- some tackle-accounting and duplicate display-only metrics

### B. Frontend/UI still open

Tracked in `.agents/frontend-ui-kickoff-plan.md`.

Still open:

- Phase 2/3 control-model refactor
  - shipped for the current index/detail tables and detail-page weekly surfaces
  - next follow-up, if needed, is polish on how the selected slice is summarized in secondary UI
    chrome such as overview cards
- Phase 2 comparison UX follow-up
  - pinned side-by-side compare layout
  - only after detail-page workflows settle
- Phase 3 ratings explanation follow-up
  - clearer inline source/type labeling for metrics
- Phase 3b detail-page follow-up
  - continue strengthening weekly views and grouped-opponent ledgers
  - keep the next slice table-first and dependency-light
- Phase 4 high-value charts
  - not started
- Phase 5 design polish
  - still in progress

### C. Documentation maintenance still open

- Keep README, AGENTS, and the two docs catalogs aligned with the current registry and output
  contract.
- Do not let the catalogs drift into a second spec; if registry and docs diverge, fix the docs.

## Deferred but intentionally retained notes

These are not active workstreams, but they are still useful context.

- `load_officials()` remains a possible future source for referee-crew or penalty-context work.
- `load_teams()` remains a possible future source for team metadata and UI chrome.
- When a future nflverse-backed loader has a season floor later than 1999, guard it in
  `data_loader.py` and return a typed empty frame there instead of widening season-loop exception
  handling.

## What the next agent should do first

1. Read this file and confirm whether the task belongs to the metric-expansion or frontend plan.
2. If the task touches planned metrics, continue from `.agents/metric-expansion-plan.md`.
3. If the task touches the analyst UI, continue from `.agents/frontend-ui-kickoff-plan.md`.
4. Keep `docs/stats-catalog.md` and `docs/qb-stats-catalog.md` synchronized with the registry when
   metric taxonomy or implementation status changes.

## File-keeping rule

Do not create a new one-off `.agents/` plan file for completed work. If a workstream is done, fold
its lasting context into this file or into the still-active plan that owns the remaining backlog.
