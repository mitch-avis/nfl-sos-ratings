# Frontend / UI Kickoff Plan

## Purpose

This document is the handoff plan for the first dedicated visualization/UI session after the current
methodology blockers are cleared.

The goal is to turn the generated team and QB outputs into a browsable, polished interface that can
show raw totals, per-game/per-snap/per-dropback rates, opponent context, ratings, and rankings in a
way that is much easier to interpret than the current static plots.

## Current status

- Status: in progress.
- Last updated: 2026-07-11.
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
  - viewport-aware tooltips that no longer clip against table columns or browser edges
  - team table now defaults to `SRS` ordering in the UI while keeping `SaCR` and related ratings as
    supporting component views
- Validation completed this session:
  - focused pytest coverage for the backend loader and API
  - live contract sanity check against real `output/2025_*` files
  - `npm run build` for the frontend shell

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

Tasks:

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

### Phase 1. Analyst shell

Goal: ship a usable skeleton before polishing charts.

Views:

1. Season selector.
2. Teams index page.
3. QBs index page.
4. Team detail page.
5. QB detail page.

Interaction requirements:

1. Sortable/filterable tables.
2. Column toggles by metric family.
3. Search by team or QB.
4. Deep-linkable URLs for season and entity selection.

### Phase 2. Comparison UX

Goal: make the UI materially better than CSV inspection.

Features:

1. Multi-select compare mode for teams.
2. Multi-select compare mode for QBs.
3. Metric normalization toggle:
   - raw totals
   - per game
   - per offensive snap
   - per defensive snap
   - per dropback
4. Show paired context columns beside subject columns.
5. Add derived differential views where they aid interpretation.

### Phase 3. Ratings explanation layer

Goal: show how ratings relate to the underlying stat surface.

Features:

1. Surface `SaOR`, `SaDR`, `SaOvR`, `SaCR`, `SRS` clearly for teams.
2. Surface `QRaw`, `QSaOR`, `QSoS`, `QSaCR`, `QOutcome` clearly for QBs.
3. Show supporting metrics beneath each rating family rather than presenting ratings as black-box
   scores.
4. Add short inline labels explaining whether a metric is:
   - official nflverse
   - PBP-derived
   - opponent-context
   - rating output

### Phase 4. High-value charts

Goal: replace the current static plots with interactive views that answer real questions.

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
5. Detail-page metric profile radar is optional and lower priority than clean scatter/table views.

### Phase 5. Design polish

Goal: make the app feel deliberate, not like a generic internal dashboard.

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

## What the next agent should do first

1. Read `.agents/pbp-overhaul-plan.md` and confirm no new methodology blocker has appeared.
2. Add entity detail routes and URL-deep-linked row drilldown for teams and QBs.
3. Expand compare mode from its current visible-column table into a more deliberate pinned side-by-
  side workflow.
4. Add chart layers only after validating that the new glossary, theme controls, and reduced-
  default index surfaces feel stable in use.
