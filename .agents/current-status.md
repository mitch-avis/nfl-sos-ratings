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
- Ratings methodology overhaul: Stage 2 principled composite weights are complete in the current
  worktree; Stage 3 validation is the next active slice.
- Primary source of truth for metrics: `nfl_sos_ratings/metrics/`.
- Human-readable metric companions: `docs/stats-catalog.md` and `docs/qb-stats-catalog.md`.
- Active workstreams: ratings-methodology Stage 3, remaining metric expansion work, and
  UI/detail-page follow-up.

## Recent work summary

The current uncommitted diff reflects five major work themes.

The latest methodology work added a sixth theme.

The current methodology work adds a seventh theme.

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
6. Ratings methodology Stage 0:
   - neutralized `SaOvR` as a Stage 0 placeholder so team outcome fields no longer feed published
     team quality ratings
   - made `QOutcome` descriptive-only in the published QB quality path by setting the default
     outcome weight to zero
   - removed redundant `*_allowed` mirror responses from the `team_simultaneous` pool
   - removed dead `min_correlation` parameters from the live QB helper signatures
   - added methodology contract tests asserting published quality ratings are invariant to banned
     outcome-only inputs
   - applied the default turnover treatment decision: turnover margin remains descriptive-only and
     out of published quality ratings until a future split/redefinition is approved
7. Ratings methodology Stage 1:
   - added `is_home` to the team game rows and fit home-field advantage directly in the team ridge
     solve
   - added deterministic ridge-lambda tuning and dropback-weighted QB WLS in
     `simultaneous_adjustment.py`
   - promoted the ridge backbone into the published team ratings (`SaOR`, `SaDR`, `SaOvR`,
     `SaCR`), with SaDR explicitly oriented so higher = better defense
   - promoted the ridge backbone into the published QB ratings (`QSaOR`, `QSaCR`) and redefined
     `QSoS` as the mean faced-defense coefficient from the QB ridge solve
   - removed the live games-played reliability multiplier from the published QB quality path
   - threaded the faced-defense ridge schedule column through `main.py` as
     `adj_def_qb_epa_per_dropback_faced`
   - refreshed registry metadata, methodology docs, and Stage 1 contract tests for the new
     backbone
8. Ratings methodology Stage 2:
   - added `nfl_sos_ratings/composite_weights.py` with season-pair builders, dropback-weighted QB
     WLS fitting, leave-one-season-out diagnostics, frozen-weight snapshots, and a reproducible
     `python -m nfl_sos_ratings.composite_weights` report command
   - replaced the interim `SaCR` equal blend with the frozen Stage 2 team composite over
     ridge-adjusted offensive passing EPA, offensive rushing EPA, defensive passing EPA, and
     defensive rushing EPA using published weights `0.4046 / 0.2016 / 0.2946 / 0.0992`
   - replaced the interim `QSaCR` passthrough with the frozen Stage 2 QB composite over adjusted
     EPA/dropback, adjusted CPOE, adjusted sack rate, and adjusted TD-INT margin rate using
     published weights `0.6688 / 0.2146 / 0.0673 / 0.0493`
   - tested the approved takeaway-creation candidate and excluded it from the frozen team
     composite after a small negative fitted weight (`-0.04081182634425329`)
   - recorded frozen-weight provenance, holdout diagnostics, and refit policy in the registry,
     README, and Stage 2 plan docs

## Validation snapshot

Latest recorded green state across the current worktree:

- `ruff format .`
- `ruff check .`
- `ty check .`
- `pyright .`
- `pytest`
- `markdownlint README.md docs/stats-catalog.md docs/qb-stats-catalog.md .agents/current-status.md .agents/ratings-methodology-overhaul-plan.md`
- `python -m nfl_sos_ratings.composite_weights`

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

### 0. Ratings methodology overhaul

Tracked in `.agents/ratings-methodology-overhaul-plan.md`.

Current state:

- Stage 0 is complete.
- Stage 1 is complete in the current worktree.
- Stage 2 is complete in the current worktree.
- Stage 3 is next: build the walk-forward validation harness and Elo baseline.
- Stage 1b play-level ridge fitting is re-sequenced until after Stage 3, where the validation
  harness will compare game-level and play-level backbones on held-out margin MAE and
  year-over-year stability.
- Turnover margin remains descriptive-only and is intentionally out of published quality ratings
  unless a maintainer approves a later split into more causal sub-signals.
- The approved team takeaway-creation candidate was tested in Stage 2 and excluded after a small
  negative fitted weight, so the frozen SaCR blend stays EPA-only for now.

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

1. Read this file and confirm whether the task belongs to the ratings-methodology, metric-expansion,
  or frontend plan.
2. If the task touches published rating methodology, continue from
  `.agents/ratings-methodology-overhaul-plan.md` and start at Stage 3 unless a maintainer directs
  otherwise.
3. If the task touches planned metrics, continue from `.agents/metric-expansion-plan.md`.
4. If the task touches the analyst UI, continue from `.agents/frontend-ui-kickoff-plan.md`.
5. Keep `docs/stats-catalog.md` and `docs/qb-stats-catalog.md` synchronized with the registry when
   metric taxonomy or implementation status changes.

## File-keeping rule

Do not create a new one-off `.agents/` plan file for completed work. If a workstream is done, fold
its lasting context into this file or into the still-active plan that owns the remaining backlog.
