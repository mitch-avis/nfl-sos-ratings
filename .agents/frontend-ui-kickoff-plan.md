# Frontend / UI Kickoff Plan

## Purpose

This document is the handoff plan for the first dedicated visualization/UI session after the current
methodology blockers are cleared.

The goal is to turn the generated team and QB outputs into a browsable, polished interface that can
show raw totals, per-game/per-snap/per-dropback rates, opponent context, ratings, and rankings in a
way that is much easier to interpret than the current static plots.

## Current status

- Status: in progress, with the analyst shell shipped, index-page compare/reset behavior now stable,
  and detail pages upgraded from a simple season snapshot into a first real analyst-facing profile
  surface.
- Last updated: 2026-07-14.
- No new methodology blocker was found in `.agents/current-status.md`; late-game QB outcome work
  remains green for UI purposes.
- Current queued scope for this session:
  - complete in the current session:
    - replaced the old multi-select metric-family controls with a single-select six-view primary row
      (`Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`, `Opponent Per-Game Rates`,
      `Opponent Per-Play Rates`) plus a page-level `Reset`
    - removed the old `Opponent Context` and `All Stats` buttons
    - added persistent Teams category and Teams/QB subcategory rows that hide without collapsing
      layout when `Ratings` is selected
    - drove the detail-page weekly and unique-opponent tables from that top-of-page selection state
      instead of their own local metric-family row
- Implemented in the most recent session:
  - bottom-right page-jump controls now keep fixed vertical slots, with the up arrow always above
    the down arrow when both are visible and each button staying in a stable position when hidden
  - a tested Parquet-backed backend contract in `nfl_sos_ratings.ui_data`
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
  - tooltip overlays now keep a readable minimum width and deliberately shift left when the hover
    point is too close to the right edge of the viewport, which avoids the narrow hard-to-read
    header-tooltip cases on sticky tables and detail pages
  - shared metric-direction metadata that now drives both first-click sort direction and numeric
    heatmap polarity
  - restored content-driven table sizing so sticky identity columns keep explicit widths while the
    rest of the table expands to fit real headers and values again
  - compare add/remove/reset for both Teams and QBs now works immediately without a page refresh;
    the `compare` query param is hydrated once into page state and app state remains the live source
    of truth afterward
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
  - additive team and QB game-log Parquet exports from `nfl_sos_ratings.main`
  - additive team and QB game-log loaders plus entity-specific FastAPI endpoints in
    `nfl_sos_ratings.ui_data` and `nfl_sos_ratings.ui_api`
  - compact single-section ratings grids on Team and QB detail pages so five rating outputs no
    longer waste vertical space
  - Team detail `Opponent Context` now splits into offense, defense, and opponent-outcome buckets
    instead of one flat wall of metrics
  - a richer weekly log surface on both Team and QB detail pages with category buttons, full
    single-game stat access, opponent season-rating context columns, and grouped opponent breakdown
    tables for repeated opponents
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
  - browser page titles now reflect the current route, season, and entity where applicable, such as
    Teams, QBs, Glossary, and individual Team/QB detail pages
  - Teams and QBs index pages now use a single-select six-view row (`Ratings`, `Raw Total Stats`,
    `Per-Game Rates`, `Per-Play Rates`, `Opponent Per-Game Rates`, `Opponent Per-Play Rates`)
    instead of the old multi-select metric-family toggles
  - the old `Opponent Context` and `All Stats` buttons are gone; Teams now show a single-select
    category row plus a multi-select subcategory row for non-`Ratings` views, and QBs show the
    matching single-tier subcategory row
  - those non-`Ratings` rows now hide without collapsing their reserved space, which prevents the
    main table from jumping vertically when the primary view changes
  - Team and QB detail pages now use the same top-of-page control model, with the entire header pane
    pinned at the top of the page and the old descriptive blurb / redundant header links removed
  - QB detail headers now render as `QB Detail` plus `QB Name - Full Team Name`, and Team detail
    headers now show the full team name
  - Game-by-Game Details and Unique Opponents no longer maintain their own local metric-family row;
    both now follow the top-of-page control state, with result-context columns always pinned first
  - those weekly/detail result-context columns now read as `Points For`, `Points Allowed`, `Point
    Margin`, `Outcome`, and `T/O Margin`
  - switching to a season where a requested Team/QB detail entity does not exist now redirects back
    to that entity index for the selected season instead of leaving the user on an error page
  - grouped opponent-breakdown tables on detail pages are now sortable by column so users can rank
    repeated-opponent summaries more easily
  - QB eligibility now uses a season-aware attempt qualifier derived from team games when weekly
    data is present, which fixes pre-2021 16-game seasons such as 2017 where the previous static
    238-attempt threshold incorrectly excluded borderline qualifiers like C.J. Beathard
  - regenerated 2016-2020 data Parquet files so the UI now reflects the corrected pre-2021 QB
    qualifier
  - the sidebar brand-card description under `NFL SOS Ratings` now reads as a broader analyst-facing
    explanation instead of a narrow project-local note
  - the low-value `Identity` summary box was removed from the Teams and QBs index overview grids
  - Teams and QBs index controls now include an `All Stats` button between `Opponent Context` and
    `Reset`, which enables every metric-family toggle in one click
  - fixed bottom-right page navigation arrows now appear across routes as `Home`/`End` shortcuts,
    with only down shown at the top, only up shown at the bottom, and both shown in the middle of
    the page
  - Team and QB detail pages now include a compact weekly summary strip that surfaces peak-week
    performance, opening-versus-closing form on a core metric, and the toughest opponent draw from
    the joined season-rating context
  - Team and QB detail pages now add a recent-three-game form card to that weekly summary strip,
    with the current core metric averaged over the latest window and compared back to the subject's
    full-season baseline on the same metric
  - grouped opponent-breakdown tables now read as compact opponent ledgers keyed to the active
    weekly surface instead of mirroring the current visible weekly columns one-for-one
  - the new grouped opponent ledgers keep identity first, favor one overall difficulty column plus
    one matched opponent-context column where appropriate, add a same-metric season-baseline delta,
    and label opponent difficulty with a simple schedule-tier bucket
  - weekly and grouped-opponent surfaces now explicitly describe repeated `opp_*` ratings as season
    opponent context rather than game-specific grades
  - extracted weekly/detail analytics into a small pure frontend helper surface, with focused
    zero-dependency Node coverage around the recent-form card and curated opponent-ledger logic
  - Team/QB detail weekly sections now read as `Game-by-Game Details` and `Unique Opponents`, with
    more analyst-facing copy, less contract jargon, and a clearer note that unique-opponent rows
    collapse repeat opponents such as division rivals into one averaged line
  - Team/QB detail weekly and unique-opponent tables now render `Win Value` as `W`, `L`, or `T` for
    exact 1.0, 0.0, and 0.5 values instead of raw decimals
  - Team/QB detail weekly game IDs now link out to the matching NFL Savant game overview pages
  - the `Unique Opponents` section now mirrors the same category-toggle row used by the weekly table
    so both surfaces switch together without extra page scanning
  - `Unique Opponents` now keeps the selected surface's full stat set instead of a heavily curated
    mini-ledger, while still pinning the chosen opponent rating plus `Sched Tier` first
  - Team detail `Sched Tier` now switches by active category: `Opp SaDR` for offense, `Opp SaOR` for
    defense, and `Opp SaCR` for results, per-snap, opponent-ratings, and all-stats views
  - `Sched Tier` now uses opponent-rating z-score thresholds instead of rank thirds, with `Tougher`
    at `z >= 0.5`, `Softer` at `z <= -0.5`, and `Middle` in between, plus custom sort ordering so it
    no longer sorts alphabetically
  - numeric heatmaps now extend into the `Unique Opponents` tables so repeated-opponent summaries
    are as scannable as the main index tables
  - team and QB z-score rating outputs now standardize against the current raw season plus any
    available historical `*_combined.parquet` / `*_qb_combined.parquet` reference data files, rather
    than normalizing only within the current season frame
- Validation completed this session:
  - focused pure-frontend helper coverage for the new tooltip, page-jump, game-link, win-value, and
    unique-opponent behavior via `npm exec -- tsc -p tsconfig.detail-tests.json && node --test
src/detailAnalytics.test.mjs src/detailUi.test.mjs`
  - focused pytest coverage for the historical-reference rating path in `ratings`, `qb_ratings`, and
    `main`
  - focused pytest coverage for the backend loader and API
  - live contract sanity check against real `data/2025_*` files
  - `npm run build` for the frontend shell
  - focused pytest coverage for the new team/QB game-log exports, loaders, and API endpoints
  - `ruff format .`, `ruff check .`, `ty check .`, `pyright .`, and `pytest`
  - live browser verification that Teams/QBs compare add/remove/reset now update immediately without
    refresh on the current frontend shell
  - live browser verification for route-aware page titles and the QB-detail-to-index redirect when
    the selected season no longer contains the requested QB
  - live browser verification that the fixed sidebar remains pinned at the top of the viewport even
    after scrolling to the bottom of a long page
  - live browser verification that the light-theme sidebar brand card now shows the expected
    gradient styling and that the 2017 QB filter note uses the corrected 224-attempt qualifier
  - focused pytest regression coverage for the pre-2021 QB qualifier and regenerated 2016-2020 data
    Parquet files
  - `npm run build` after the shell/index cleanup and weekly-summary-strip changes
  - focused zero-dependency frontend helper coverage via `npm exec -- tsc -p
tsconfig.detail-tests.json && node --test src/detailAnalytics.test.mjs`
  - `npm run build` after the recent-form and grouped-opponent-ledger changes
  - focused zero-dependency frontend helper coverage for the new six-view selection model and the
    weekly base-column behavior via `npm exec -- tsc -p tsconfig.detail-tests.json && node --test
src/detailAnalytics.test.mjs src/detailUi.test.mjs`
  - `npm run build` after the six-view control-model refactor and pinned detail-header changes
  - live browser verification that the primary row is single-select, the subcategory row remains
    multi-select, reset returns to `Ratings`, the extra rows hide without collapsing table position,
    and Team/QB detail headers use the new pinned pane and full-name formatting

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

1. Keep strengthening the weekly-log detail pages now that the shell/index cleanup and the new
  control-model refactor are complete, especially if another compact trend primitive still feels
  justified after the new recent-form card, but keep it table-first and dependency-light.
2. Refine the new grouped opponent ledgers after live use, especially if one team or QB weekly
  surface wants a different primary performance metric or a tighter default column mix.
3. Add opponent-strength or rating-delta context to the weekly views carefully, without implying
  that a repeated season-long opponent rating is a true single-game rating.
4. Revisit the compare workflow only after the weekly/detail surfaces settle, with the next step
  being a pinned side-by-side layout rather than more compact-strip patching.

## Hard prerequisite

Check `.agents/current-status.md` first and confirm there is still no active methodology blocker on
the QB outcome fields before doing frontend work.

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

Build a lightweight local web app that reads the generated Parquet outputs rather than rebuilding
the rating pipeline inside the UI.

Recommended stack for the first pass:

1. Backend/API layer: Python + FastAPI.
2. Frontend: React + Vite.
3. Data transport: preloaded JSON from the existing Parquet outputs, served by the API.
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
   - `data/{season}_team_per_game_stats.parquet`
   - `data/{season}_qb_per_game_stats.parquet`
   - `data/{season}_combined.parquet`
   - `data/{season}_qb_combined.parquet`
   - `data/{season}_ratings.parquet`
   - `data/{season}_qb_ratings.parquet`

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

Goal: make the UI materially better than raw data inspection.

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

1. Export team game logs as `data/{season}_team_game_logs.parquet`.
2. Export QB game logs as `data/{season}_qb_game_logs.parquet`.
3. Extend `nfl_sos_ratings.ui_data` and `nfl_sos_ratings.ui_api` with additive entity-specific
   game-log payloads/endpoints.
4. Add weekly game-log tables to Team and QB detail pages, keeping the existing season snapshot
   metric grid intact.
5. Add category-filter buttons so weekly logs can pivot between compact views and the full stat
   surface.
6. Join opponent season ratings onto the weekly logs for additional game-level context.
7. Add grouped opponent-breakdown tables so repeated opponents are summarized instead of repeated as
   near-duplicate weekly rows.
8. Compact the detail-page ratings view and split Team opponent context into more readable offense
   and defense sections.
9. Add a compact weekly summary strip that highlights peak performance, closing-form deltas, and the
   toughest opponent draw from the season-rating context.

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

Execution notes for the next detail-page slice:

1. Keep the next weekly enhancement table-first and dependency-light.
2. A good next compact trend primitive would be one of:
   - a rolling three-game average card for the current core metric
   - a week-over-week change callout for the current core metric
   - a best-stretch / worst-stretch summary over a fixed recent window
3. Keep those summaries grounded in real game-log columns that already exist in the additive
   contract. Do not invent pseudo-expected values just to make the summaries look richer.
4. The grouped opponent-breakdown table should graduate from "sortable summary" into "analytical
   opponent ledger" without becoming another wide uncurated dump.
5. Default grouped-opponent columns should stay compact and high-signal.
6. For teams, the default grouped view should keep identity first (`opponent_team`, `games`,
   `weeks`) and then prefer one overall difficulty column (`opp_SaCR` or `opp_SRS`), one
   opponent-side context column (`opp_SaDR` for offense reading or `opp_SaOR` for defense reading),
   and one or two subject-performance columns such as `point_margin` or a core efficiency rate when
   those columns exist for the active surface.
7. For QBs, the default grouped view should keep identity first and then prefer `opp_SaDR`,
   `opp_SaCR`, `point_margin`, and one or two core QB performance columns such as
   `qb_epa_per_dropback`, `qb_any_a`, or `qb_passer_rating` when available.
8. Derived grouped-opponent columns should be additive summaries that are easy to defend from the
   current trustworthy inputs.
9. Good examples are:
   - subject average performance against that opponent on a core metric
   - subject season-baseline delta on the same metric
   - a simple opponent-difficulty bucket derived from season-long ratings
10. Avoid derived columns that pretend the opponent season rating was a true single-game forecast or
    a directly comparable game-level stat.
11. Weekly opponent-strength overlays should be treated as season-context labels, not as single-game
    ratings.
12. Match the context column to the surface being read: team offense views should emphasize
    `opp_SaDR`; team defense views should emphasize `opp_SaOR`; team overall/result views should
    emphasize `opp_SaCR` or `opp_SRS`; QB passing views should emphasize `opp_SaDR`, with `opp_SaCR`
    only as broader team context.
13. If a weekly delta surface is added, the safest default is subject-week versus subject-season
    baseline on the same metric.
14. Do not subtract unlike units or imply that a season rating is the expected output of one game
    unless that methodology has been explicitly defined and approved elsewhere.
15. Any tooltip or label for these columns should say `season opponent context` or similarly clear
    wording so analysts do not read the value as a game-specific rating.

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

1. Carry the same clarity and utility standard from the cleaned-up shell into the grouped weekly and
   opponent-breakdown surfaces.
2. Only revisit the compare panel once the richer detail-page workflows have settled.

Current recommended polish target inside Phase 5:

1. Make the grouped weekly surfaces feel intentionally edited rather than merely sortable.
2. Prefer a small number of clearly named summary cards and default columns over another broad wall
   of stats.
3. Use copy and tooltips to make the difference between weekly performance, season baseline, and
   season-long opponent context explicit everywhere those surfaces intersect.

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
3. Do not couple the frontend directly to ad hoc backend-specific file parsing in the browser.
4. Do not overbuild authentication, persistence, or deployment before the local analyst workflow is
   strong.
5. Do not treat aggregated opponent profiles as if they were game logs; weekly detail views need
   their own contract surface.
6. Do not double-count repeat opponents in detail-page opponent tables.
7. Do not let a season-long opponent rating read like a game-specific expectation, forecast, or
   single-game grade.
8. Do not add derived delta columns that subtract unlike units just because the result looks
   visually convenient.

## What the next agent should do first

1. Read `.agents/current-status.md` and confirm no new methodology blocker has appeared.
2. Inspect the current detail-page weekly surfaces first and pick one compact trend primitive to
   land cleanly without adding a charting dependency.
3. In the same session, upgrade the grouped opponent-breakdown tables so their default columns are
   more analytical and less dump-like.
4. If there is room after that, add opponent-strength or rating-delta context to the weekly views,
   but only with labels and tooltips that make the season-context nature of those values obvious.
5. Prefer subject-season baseline deltas over any cross-unit subtraction, and prefer
   offense-versus-defense matched context (`opp_SaDR`, `opp_SaOR`) over generic overall context when
   the view is reading a specific side of the ball.
6. Choose the first charting library only when the next detail-page chart is ready to implement, and
   ask before adding that new dependency.
7. Revisit the compare panel after the richer detail pages settle, with a pinned side-by-side
   workflow as the next comparison upgrade rather than more patching on the compact strip.
