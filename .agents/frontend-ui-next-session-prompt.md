# Frontend UI Next-Session Prompt

You are working in `/home/mitch/workspace/nfl-sos-ratings`.

Before editing anything:

1. Read `AGENTS.md`.
2. Read `.agents/pbp-overhaul-plan.md` and confirm there is still no active methodology blocker
   for QB outcome fields.
3. Read `.agents/frontend-ui-kickoff-plan.md` and treat it as the authoritative frontend handoff
   document.
4. Inspect the current frontend ownership surfaces first:
   - `ui/web/src/App.tsx`
   - `ui/web/src/components/DataTable.tsx`
   - `ui/web/src/components/ComparisonPanel.tsx`
   - `ui/web/src/components/EntityDetail.tsx`
   - `ui/web/src/components/GlossaryPage.tsx`
   - `ui/web/src/metricMetadata.ts`
   - `ui/web/src/entityConfig.ts`
   - `ui/web/src/styles.css`
5. Inspect the backend UI contract surfaces if the next step touches them:
   - `nfl_sos_ratings/main.py`
   - `nfl_sos_ratings/ui_data.py`
   - `nfl_sos_ratings/ui_api.py`
   - `tests/test_ui_data.py`
   - `tests/test_ui_api.py`
   - `tests/test_main.py`

Current known state from the last session:

- The compare-state bug was fixed by hydrating the `compare` query param into page state once and
  then treating app state as the source of truth afterward.
- Compare add/remove/reset now works immediately on the current frontend shell for both Teams and
  QBs without needing a refresh.
- Teams now default to `SaCR` sorting and to `Ratings` + `Per-Game Rates` as the enabled groups.
- QBs now default to `QSaCR` sorting and to `Ratings` + `Per-Game Rates` as the enabled groups.
- Teams `Raw Totals` was renamed to `Per-Game Rates` in code; if a live browser still shows the old
  label, restart the running local FastAPI/UI API process before doing more browser QA.
- Tooltip and glossary copy was cleaned up to remove stale “generated season contract” phrasing and
  to avoid repeating a metric name inside its own tooltip body.
- Team and QB detail pages now have compact ratings grids instead of vertically stretched single-
  metric sections.
- Team detail `Opponent Context` is now split into opponent offense, opponent defense, and opponent
  outcomes.
- Weekly log sections now support category-filter buttons, full all-stats tables, joined opponent
  season rating context, and grouped opponent-breakdown tables.
- Grouped opponent-breakdown tables are now sortable by column.
- The QB unrated-row toggle now explains the one-snap display rule and the 238-attempt rating
  qualifier.
- The sidebar was slimmed down, the old `Contract-backed views` footer link was removed, and the
  browser title now reflects the current route/season/entity.
- The sidebar is now fixed in place, with a gradient brand card and four vertical sections for
  Season, Views, Theme, and Palette.
- The sidebar brand card now renders correctly in both light and dark themes, and the old
  `Analyst Console` subtitle line was removed.
- If a Team or QB detail route points at an entity that does not exist for the selected season,
  the app now redirects back to that entity index for the selected season instead of showing an
  error page.
- QB eligibility is now season-aware when weekly team data is available, which fixes the pre-2021
  16-game qualifier bug; 2016-2020 outputs were regenerated so the current UI reflects that fix.

Outstanding follow-ups that should be treated as explicitly requested next-session work:

1. Rewrite the left-sidebar description under `NFL SOS Ratings` so it is more descriptive and
   useful to any analyst user rather than reading like a narrow local note.
2. Remove the `Identity` summary box from the Teams and QBs index pages.
3. Add an `All Stats` category button to the Teams and QBs index pages, placed between
   `Opponent Context` and `Reset`.
4. Add bottom-right page navigation arrows on all pages. They should behave like `Home` and `End`,
   with only the down arrow visible at the top, only the up arrow visible at the bottom, and both
   visible while the user is in the middle of the page.

Validation already passed for the current code before this handoff file was added:

- `npm run build`
- focused `pytest tests/test_ui_data.py tests/test_ui_api.py`
- earlier in-session full repo validation: `ruff format .`, `ruff check .`, `ty check .`,
  `pyright .`, `pytest`

Primary goal for the next session:

Build on the current detail-page enrichment work rather than jumping straight to new top-level
charts or a large compare-panel redesign.

Priorities for the next session:

1. Handle the explicitly queued shell/index follow-ups first.
   Rewrite the sidebar descriptive copy for a broader audience.
   Remove the `Identity` summary box from Teams and QBs.
   Add `All Stats` category buttons to Teams and QBs.
   Add the bottom-right `Home`/`End` navigation arrows with the requested visibility rules.

2. Strengthen the new weekly log surfaces.
   Add clearer weekly trend summaries or compact trend primitives without adding a new charting
   dependency unless you ask first.
   Improve the grouped opponent-breakdown tables so they are more analytical and easier to scan.
   Build on the new sortable grouped-opponent tables with more deliberate default columns and useful
   derived context columns.

3. Add opponent-strength overlays or rating-delta context to the weekly views.
   Use the existing season ratings and opponent ratings carefully.
   Do not imply that a repeated season rating column is a true single-game rating.

4. Only after the detail pages feel stable, revisit the compare workflow.
   The next compare step should be a more deliberate pinned side-by-side layout.
   Do not regress the newly fixed add/remove/reset behavior.

5. If live browser behavior does not match the code, verify whether the local API process is stale
   before assuming the implementation is wrong.

Constraints and reminders:

- This is not a planning-only session.
- Work in small, validated slices.
- Use TDD where there is an existing backend test surface.
- Do not add a charting library or a frontend test framework without asking first.
- Update `.agents/frontend-ui-kickoff-plan.md` in the same change set to reflect real progress,
  decisions, validation, blockers, and next steps.
- If you touch repo-owned Markdown, run `markdownlint` on the touched Markdown files.
- Before finishing, run:
  - `npm run build` in `ui/web`
  - `ruff format .`
  - `ruff check .`
  - `ty check .`
  - `pyright .`
  - `pytest`

If scope gets tight, prefer landing one polished detail-page weekly-log improvement slice and
updating the plan accurately rather than starting several unfinished UI directions.
