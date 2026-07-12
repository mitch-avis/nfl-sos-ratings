# Frontend / UI Kickoff Plan

## Purpose

This document is the handoff plan for the first dedicated visualization/UI session after the current
methodology blockers are cleared.

The goal is to turn the generated team and QB outputs into a browsable, polished interface that can
show raw totals, per-game/per-snap/per-dropback rates, opponent context, ratings, and rankings in a
way that is much easier to interpret than the current static plots.

## Current status

- Status: in progress, with the analyst shell shipped, index-page compare/reset behavior now
  stable, and detail pages upgraded from a simple season snapshot into a first real analyst-facing
  profile surface.
- Last updated: 2026-07-12.
- No new methodology blocker was found in `.agents/pbp-overhaul-plan.md`; late-game QB outcome work
  remains green for UI purposes.
- Implemented this session:
  - a tested CSV-backed backend contract in `nfl_sos_ratings.ui_data`
  - a thin FastAPI app in `nfl_sos_ratings.ui_api`
  - a first-pass React + Vite shell in `ui/web/` with season-aware Teams and QBs index routes,
    sortable/filterable TanStack tables, and column-group toggles
  - deep-linkable team and QB detail routes backed by the normalized season payload
  - a first compare workflow driven by URL query state and contract-backed rating columns
  - dedicated frontend documentation in `ui/web/README.md`
  - built-in light/dark theme controls and a Broncos-inspired palette toggle for color-blind-
    friendlier viewing
  - glossary route plus header/detail tooltips for rating and metric explanations
  - default QB filtering that hides unrated rows unless the user explicitly reveals them
  - numeric cell heatmaps and a compact compare strip that follows the currently visible columns
  - sticky identity columns and integer rank columns for both team and QB tables
  - viewport-clamped tooltip portals that now stay anchored near the hovered header or metric label,
    including on sticky tables and detail pages
  - shared metric-direction metadata that now drives both first-click sort direction and numeric
    heatmap polarity
  - restored content-driven table sizing so sticky identity columns keep explicit widths while the
    rest of the table expands to fit real headers and values again
  - compare add/remove/reset for both Teams and QBs now works immediately without a page refresh;
    the `compare` query param is hydrated once into page state and app state remains the live
    source of truth afterward
  - team index defaults now use `SaCR` sorting plus `Ratings` and `Per-Game Rates`; QB defaults now
    use `QSaCR` sorting plus `Ratings` and `Per-Game Rates`
  - Teams `Raw Totals` was renamed to `Per-Game Rates` across the season-index/detail surfaces to
    match what the team payload actually represents
  - metadata-driven compact metric labels across tables, compare panels, detail cards, and glossary
    surfaces, with full metric names and explanations now shown in tooltips
  - tooltip/glossary description cleanup that removes the stale “generated season contract” wording
    and avoids repeating a metric’s full name inside its own tooltip body
  - persistent Teams/QBs page view state in the app layer for group toggles, sorting, compare
    selections, search text, the QB unrated-row toggle, and detail-page active metric-family tabs
  - a page-aware Reset control beside the group toggles that restores the current entity view's
    default groups, sorting, compare selections, and related local view state without touching theme
    or palette preferences
  - additive team and QB game-log CSV exports from `nfl_sos_ratings.main`
  - additive team and QB game-log loaders plus entity-specific FastAPI endpoints in
    `nfl_sos_ratings.ui_data` and `nfl_sos_ratings.ui_api`
  - compact single-section ratings grids on Team and QB detail pages so five rating outputs no
    longer waste vertical space
  - Team detail `Opponent Context` now splits into offense, defense, and opponent-outcome buckets
    instead of one flat wall of metrics
  - a richer weekly log surface on both Team and QB detail pages with category buttons, full
    single-game stat access, opponent season-rating context columns, and grouped opponent
    breakdown tables for repeated opponents
  - an explanatory note for the QB unrated-row toggle that documents the one-snap minimum display
    rule and the 238-attempt rating qualifier
  - the left menu panel is now slimmer and cleaner, with `NFL SOS Ratings` promoted as the primary
    menu title, `Analyst Console` demoted to a smaller single-line subtitle, and the old
    `Contract-backed views` footer link removed
  - the left menu is now fixed in place with four clearly labeled vertical sections (`Season`,
    `Views`, `Theme`, `Palette`), stacked option controls, and a gradient brand card that stays in
    view while the main content scrolls
  - the sidebar brand card now uses the same gradient treatment in both light and dark themes, and
    the `Analyst Console` subtitle line was removed
  - browser page titles now reflect the current route, season, and entity where applicable, such
    as Teams, QBs, Glossary, and individual Team/QB detail pages
  - switching to a season where a requested Team/QB detail entity does not exist now redirects
    back to that entity index for the selected season instead of leaving the user on an error page
  - grouped opponent-breakdown tables on detail pages are now sortable by column so users can rank
    repeated-opponent summaries more easily
  - QB eligibility now uses a season-aware attempt qualifier derived from team games when weekly
    data is present, which fixes pre-2021 16-game seasons such as 2017 where the previous static
    238-attempt threshold incorrectly excluded borderline qualifiers like C.J. Beathard
  - regenerated 2016-2020 output CSVs so the UI now reflects the corrected pre-2021 QB qualifier
- Validation completed this session:
  - focused pytest coverage for the backend loader and API
  - live contract sanity check against real `output/2025_*` files
  - `npm run build` for the frontend shell
  - focused pytest coverage for the new team/QB game-log exports, loaders, and API endpoints
  - `ruff format .`, `ruff check .`, `ty check .`, `pyright .`, and `pytest`
  - live browser verification that Teams/QBs compare add/remove/reset now update immediately
    without refresh on the current frontend shell
  - live browser verification for route-aware page titles and the QB-detail-to-index redirect when
    the selected season no longer contains the requested QB
  - live browser verification that the fixed sidebar remains pinned at the top of the viewport even
    after scrolling to the bottom of a long page
  - live browser verification that the light-theme sidebar brand card now shows the expected
    gradient styling and that the 2017 QB filter note uses the corrected 224-attempt qualifier
  - focused pytest regression coverage for the pre-2021 QB qualifier and regenerated 2016-2020
    output CSVs

Notes from live verification:

- The grouped opponent table briefly surfaced a duplicate React key issue while this work was in
  progress; that was fixed in the current code by excluding duplicate `games`/`weeks` columns from
  the grouped summary schema.

Phase status summary:

- Phase 0. Data contract lock: complete.
- Phase 1. Analyst shell: complete.
- Phase 2. Comparison UX: partial.
- Phase 3. Ratings explanation layer: partial.
- Phase 3b. Detail-page enrichment: in progress.
- Phase 4. High-value charts: not started.
- Phase 5. Design polish: in progress.

Outstanding follow-ups explicitly queued for the next agent session:

1. Revise the left-sidebar description under `NFL SOS Ratings` so it reads as a clearer,
   more useful explanation for any analyst user rather than a narrow project-local blurb.
2. Remove the `Identity` summary box from the Teams and QBs index pages; it is currently low-value
   compared with the stat/rating group boxes.
3. Add an `All Stats` category button to both Teams and QBs index pages, matching the newer
   weekly-log detail-page pattern. Place it between `Opponent Context` and `Reset`.
4. Add fixed bottom-right navigation arrows on all pages as shortcuts for `Home` and `End`:
   only the down arrow at the very top, only the up arrow at the very bottom, and both arrows
   while the user is between those extremes.

## Hard prerequisite

Check `.agents/pbp-overhaul-plan.md` first and confirm there is still no active methodology blocker
on the QB outcome fields before doing frontend work.

The previously open late-game 4QC/GWD bug was fixed in the latest remediation pass, but the UI
session should still treat the overhaul plan as the source of truth in case new findings appear.

## What is already trustworthy enough to design around

These surfaces were rechecked against authoritative nflreadpy sources and are suitable as the core
data model for the first UI pass:

- Team published weekly/season offense stats now match official nflverse team stats.
- Team published split stats now match official nflverse values for:
  - passing yards / rushing yards / total yards
  - passing TDs / rushing TDs
  - passing first downs / rushing first downs
  - passing EPA / rushing EPA
  - passing CPOE
  - sacks suffered / passing interceptions / sack fumbles lost / rushing fumbles lost
- QB attempt-based weekly/season stats now match official nflverse player stats for:
  - attempts / completions / passing yards / passing TDs / interceptions
  - sacks / sack yards lost / passing EPA / passing CPOE
- Canonical QB identity is now GSIS-based, with duplicate season rows removed.
- Non-QB trick passers are excluded from the QB outputs.
- Derived rate formulas such as ANY/A, YPA, EPA/dropback, sack rate, TD-INT margin rate, and passer
  rating were independently rechecked and matched the implemented formulas.

## Product direction

The first UI should not be a marketing page. It should be an analyst-facing exploratory interface.

Primary jobs to support:

1. Compare teams to teams.
2. Compare QBs to QBs.
3. See raw stats, derived rates, opponent context, and ratings side by side.
4. Understand how a final rating was produced.
5. Move between season-wide views and single-entity detail views quickly.

## Recommended implementation approach

Build a lightweight local web app that reads the generated CSV outputs rather than rebuilding the
rating pipeline inside the UI.

Recommended stack for the first pass:

1. Backend/API layer: Python + FastAPI.
2. Frontend: React + Vite.
3. Data transport: preloaded JSON from the existing CSV outputs, served by the API.
4. Tables/charts: TanStack Table plus a charting library with good tooltip/brush support.

Why this direction:

- The repo is already Python-first.
- FastAPI gives a thin layer for file loading, normalization, filtering, and future caching.
- React/Vite is a fast path to a richer UI than matplotlib output without committing to a heavy
  product architecture too early.
- Using the generated outputs as the contract keeps the UI session focused on presentation, not
  methodology.

## Recommended phase breakdown

### Phase 0. Data contract lock

Goal: define exactly which files and columns the UI consumes.

Status: complete.

Delivered:

1. Treat these outputs as the first UI contract:
   - `output/{season}_team_per_game_stats.csv`
   - `output/{season}_qb_per_game_stats.csv`
   - `output/{season}_combined.csv`
   - `output/{season}_qb_combined.csv`
   - `output/{season}_ratings.csv`
   - `output/{season}_qb_ratings.csv`
2. Create a small backend loader that exposes one normalized response per season.
3. Document stable field groups:
   - team raw totals/per-game
   - team opponent context
   - team ratings/simultaneous adjustments
   - QB raw totals/per-game
   - QB attempt-based rates
   - QB dropback-based rates
   - QB opponent context
   - QB ratings/outcome metrics

Notes:

- The current contract is intentionally season-aggregate first.
- Richer detail pages should extend the contract additively with game-log payloads rather than
  overloading the existing combined rows.

### Phase 1. Analyst shell

Goal: ship a usable skeleton before polishing charts.

Status: complete.

Delivered views:

1. Season selector.
2. Teams index page.
3. QBs index page.
4. Team detail page.
5. QB detail page.

Delivered interaction requirements:

1. Sortable/filterable tables.
2. Column toggles by metric family.
3. Search by team or QB.
4. Deep-linkable URLs for season and entity selection.

### Phase 2. Comparison UX

Goal: make the UI materially better than CSV inspection.

Status: partial.

Delivered:

1. Multi-select compare mode for teams.
2. Multi-select compare mode for QBs.
3. A compact compare strip keyed to the currently visible columns.
4. Paired subject columns plus context columns in the main index tables.
5. Stable compare-state persistence and reset behavior for the current compact compare workflow.

Remaining:

1. Metric normalization toggle:
   - raw totals
   - per game
   - per offensive snap
   - per defensive snap
   - per dropback
2. A more deliberate pinned side-by-side compare layout instead of the current compact strip.
3. Derived differential views where they materially improve interpretation.

### Phase 3. Ratings explanation layer

Goal: show how ratings relate to the underlying stat surface.

Status: partial.

Delivered:

1. Surface `SaOR`, `SaDR`, `SaOvR`, `SaCR`, `SRS` clearly for teams.
2. Surface `QRaw`, `QSaOR`, `QSoS`, `QSaCR`, `QOutcome` clearly for QBs.
3. Show supporting metrics beneath each rating family rather than presenting ratings as black-box
   scores.
4. Use compact shared display labels throughout the UI while preserving fuller glossary names and
   tooltip explanations.

Remaining:

1. Add short inline labels explaining whether a metric is:
   - official nflverse
   - PBP-derived
   - opponent-context
   - rating output

### Phase 3b. Detail-page enrichment

Goal: turn the current season-row detail views into analyst-friendly profile pages with weekly and
opponent-by-opponent context.

Status: in progress.

Delivered so far:

1. Export team game logs as `output/{season}_team_game_logs.csv`.
2. Export QB game logs as `output/{season}_qb_game_logs.csv`.
3. Extend `nfl_sos_ratings.ui_data` and `nfl_sos_ratings.ui_api` with additive entity-specific
   game-log payloads/endpoints.
4. Add weekly game-log tables to Team and QB detail pages, keeping the existing season snapshot
   metric grid intact.
5. Add category-filter buttons so weekly logs can pivot between compact views and the full stat
   surface.
6. Join opponent season ratings onto the weekly logs for additional game-level context.
7. Add grouped opponent-breakdown tables so repeated opponents are summarized instead of repeated
   as near-duplicate weekly rows.
8. Compact the detail-page ratings view and split Team opponent context into more readable offense
   and defense sections.

Required backend additions:

1. Keep the additive game-log contract stable as the detail pages expand.
2. Preserve normalized rate columns per snap or per dropback alongside raw game values where both
   are useful.

Recommended frontend additions:

1. Keep the current `EntityDetail` metric grid as the season snapshot tab.
2. Expand the weekly game-log section into a stronger weekly trend surface for teams and QBs.
3. Add more analytical grouped-opponent views on top of the now-sortable grouped tables.
4. Add a rank-or-rating trend view that shows how the subject moved through the season.
5. Add an opponent-strength overlay or per-game rating delta surface so weekly performance can be
   read against defensive or offensive difficulty.

Design direction inspired by the nfelo screenshots, but not copied:

1. Use a strong summary header with a small set of high-signal cards.
2. Keep the detail page table-first and analysis-first rather than prediction- or betting-first.
3. Let charts explain season shape, not decorate the page.
4. Prefer clean weekly trend and opponent-breakdown views over radar charts or novelty widgets.

### Phase 4. High-value charts

Goal: replace the current static plots with interactive views that answer real questions.

Status: not started.

Recommended first chart set:

1. Team offense vs defense quadrant:
   - x-axis `SaOR`
   - y-axis `SaDR`
2. Team overall rating rank chart:
   - sortable bar/dot plot with `SaCR`, `SaOvR`, `SRS`
3. QB adjusted performance vs schedule strength:
   - x-axis `QSoS`
   - y-axis `QSaOR`
4. QB outcome overlay:
   - x-axis `QSaOR`
   - y-axis `QOutcome`
   - now unblocked by the late-game metrics fix, but still worth spot-checking against the current
     overhaul plan before implementation
5. Weekly trend charts for detail pages should come before optional radar or profile charts.

### Phase 5. Design polish

Goal: make the app feel deliberate, not like a generic internal dashboard.

Status: in progress.

Visual direction recommendation:

- Tone: editorial sports almanac meets modern quantitative dashboard.
- Avoid generic SaaS gradients and default component-library aesthetics.
- Use a strong typographic hierarchy and a dense-but-readable table-first layout.
- Prefer a light background with assertive ink colors and restrained accent colors keyed to metric
  families.

Design principles:

1. Tables are first-class, not an afterthought.
2. Charts should explain, not decorate.
3. Density is acceptable if typography and spacing stay disciplined.
4. Mobile support matters, but desktop analyst workflows come first.

Remaining polish tasks already requested by the user:

1. Rework the left-sidebar descriptive copy so it better explains the app to a general analyst.
2. Remove the `Identity` summary card from the Teams and QBs index overviews.
3. Add `All Stats` index-page category buttons for Teams and QBs.
4. Add page navigation arrows in the bottom-right corner with top/middle/bottom visibility rules.

## Suggested repository shape for the first UI session

One reasonable path:

```text
ui/
  api/
    app.py
    loaders.py
    schemas.py
  web/
    src/
      app/
      components/
      routes/
      charts/
      tables/
      styles/
```

Alternative:

- Keep FastAPI under `nfl_sos_ratings/` if the team wants one Python package.
- Put the Vite app in `web/` at repo root.

Chosen structure for the current scaffold:

```text
nfl_sos_ratings/
   ui_api.py
   ui_data.py
ui/
   web/
```

Reasoning:

- Backend logic stays inside the Python package so the repo's existing test, type-check, and
  coverage workflow can validate it directly.
- The frontend remains isolated under `ui/web/`, which keeps Node tooling out of the core Python
  package.

## Risks to watch

1. Do not let the UI session silently redefine metric semantics.
2. Do not build charts around columns that are still under methodology review.
3. Do not couple the frontend directly to ad hoc CSV parsing in the browser.
4. Do not overbuild authentication, persistence, or deployment before the local analyst workflow is
   strong.
5. Do not treat aggregated opponent profiles as if they were game logs; weekly detail views need
   their own contract surface.
6. Do not double-count repeat opponents in detail-page opponent tables.

## What the next agent should do first

1. Read `.agents/pbp-overhaul-plan.md` and confirm no new methodology blocker has appeared.
2. Address the queued shell/index follow-ups first.
   Rewrite the left-sidebar descriptive copy for a broader analyst audience.
   Remove the `Identity` summary box from Teams and QBs.
   Add `All Stats` category buttons to the Teams and QBs index pages.
   Add bottom-right `Home`/`End` navigation arrows with the requested visibility behavior.
3. After that shell/index cleanup, continue building on the shipped game-log contract by turning
   the current weekly log tables into stronger weekly trend surfaces and more analytical grouped-
   opponent views.
4. Add opponent-strength overlays or rank/rating movement summaries only after the grouped opponent
   view is in place.
5. Choose the first charting library only when the next detail-page chart is ready to implement,
   and ask before adding that new dependency.
6. Revisit the compare panel after the richer detail pages settle, with a pinned side-by-side
   workflow as the next comparison upgrade rather than more patching on the compact strip.
