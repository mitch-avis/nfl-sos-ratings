# NFL SOS Ratings UI

## Purpose

This frontend is the first analyst-facing interface for `nfl-sos-ratings`. It is not a public
marketing site. It is a local exploration console for the generated season outputs under `data/`.

The UI is meant to answer questions like:

- Which teams rate best once opponent strength is accounted for?
- Which QBs look strongest after separating raw performance from schedule strength?
- How do the rating outputs sit next to the raw totals, rate metrics, and opponent context that
  produced them?
- How can I quickly compare a few teams or QBs without dropping back to raw data inspection?

## What It Includes Today

- Season-aware Teams index route
- Season-aware QBs index route
- Sortable and filterable TanStack tables
- A single-select six-view control row on Teams and QBs pages:
  `Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`,
  `Opponent Per-Game Rates`, `Opponent Per-Play Rates`
- Teams category and team/QB subcategory rows that appear for non-`Ratings` views while keeping
  their vertical space reserved when hidden
- Built-in light/dark theme toggle
- Built-in classic/Broncos palette toggle for cleaner color-blind-friendly viewing
- In-table heatmap styling for numeric metrics
- Sticky identity columns and integer rank columns in the main tables
- Deep-linkable detail routes for individual teams and QBs
- A compare workflow driven by URL query state and current visible columns
- A glossary route for rating meanings and reading guidance
- A thin FastAPI backend that serves normalized JSON from the generated Parquet contract

## Data Contract

The UI does **not** rebuild the methodology from pipeline internals. It reads the generated Parquet
outputs through the backend contract in `nfl_sos_ratings.ui_data`.

Current required files per season:

- `{season}_team_per_game_stats.parquet`
- `{season}_qb_per_game_stats.parquet`
- `{season}_combined.parquet`
- `{season}_qb_combined.parquet`
- `{season}_ratings.parquet`
- `{season}_qb_ratings.parquet`

If one of those files is missing, the backend will not advertise that season as available.

## Project Layout

```text
ui/web/
  index.html
  package.json
  src/
    App.tsx
    api.ts
    entityConfig.ts
    format.ts
    main.tsx
    styles.css
    types.ts
    vite-env.d.ts
    components/
      AppShell.tsx
      ComparisonPanel.tsx
      DataTable.tsx
      EntityDetail.tsx
      OverviewCards.tsx
```

Backend files live in the Python package:

- `nfl_sos_ratings/ui_data.py`
- `nfl_sos_ratings/ui_api.py`

## How To Run

Start from the repository root in the project virtual environment.

### 1. Ensure the generated outputs exist

```bash
python -m nfl_sos_ratings.main
```

### 2. Start the backend

```bash
python -m nfl_sos_ratings.ui_api
```

The API serves on `http://127.0.0.1:8000` by default.

### 3. Start the frontend

```bash
cd ui/web
npm install
npm run dev
```

The Vite dev server serves on `http://127.0.0.1:5173` and proxies `/api` to the backend.

## How To Build

```bash
cd ui/web
npm run build
```

This runs TypeScript project builds and then a Vite production build.

## How To Use

### Index views

- Use the left rail to switch between Teams and QBs.
- Change the season from the selector.
- Use the theme toggle if you want a built-in dark presentation instead of relying on browser-side
  inversion tools.
- Use the palette toggle to switch to the Broncos-inspired orange/navy palette for a more
  color-blind-friendly view.
- Search filters only the currently visible columns.
- Identity columns are always visible and remain pinned while side-scrolling.
- The primary view row is single-select. Exactly one of the six views is active at a time.
- On Teams pages, non-`Ratings` views reveal a single-select category row (`Overall`, `Offense`,
  `Defense`, `Special Teams`) plus a multi-select subcategory row for the selected category.
- On QB pages, non-`Ratings` views reveal a single multi-select subcategory row.
- `Reset` restores the default state: `Ratings`, Teams category `Offense`, all subcategories
  enabled, default sorting, empty compare state, and cleared search.
- Team index defaults are intentionally narrower now so the first view emphasizes rankings before
  you expand additional stat families.
- QB index hides unrated rows by default; use the row-filter toggle to reveal all QB rows.
- Team view currently sorts by `SRS` by default in the UI, while `SaCR`, `SaOR`, `SaDR`, and
  `SaOvR` remain available as supporting schedule-adjusted component views.

### Detail views

- Click a team code or QB name from the table to open that entity's detail route.
- Detail pages use the same top-of-page six-view row and category/subcategory rows as the index.
- The detail header pane is pinned while the page scrolls.
- Weekly and unique-opponent tables now follow the top-of-page selection state instead of
  maintaining their own local metric-family toggle row.

### Compare mode

- Use the compare checkbox in the first table column.
- The selected entities appear in a compact comparison strip above the table.
- Compare state is stored in the query string, so you can bookmark or share a local URL.
- The comparison table follows the currently visible columns in the index view instead of being
  limited to ratings only.

### Glossary

- Use the Glossary route from the left rail when abbreviations are unclear.
- The glossary explains which ratings are the primary overall rankings and which are supporting or
  context-only signals.

## Planned Near-Term Work

- Refine which columns are visible by default for first-pass analysis
- Improve detail-page narrative and metric annotations
- Expand compare mode into a more deliberate multi-entity analysis workflow
- Add chart views only after the table/detail workflows feel strong

## Troubleshooting

### `npm run build` fails on `./styles.css`

The frontend requires the ambient Vite declarations in `src/vite-env.d.ts`. If that file is missing,
newer TypeScript versions can fail on the CSS side-effect import in `src/main.tsx`.

### The season selector is empty

The backend only exposes seasons that have the complete six-file UI contract. Re-run the Python
pipeline and confirm the expected Parquet files exist under `data/`.

### The frontend loads but API calls fail

Check that the backend is running on `127.0.0.1:8000`. The Vite config proxies `/api` to that
address.

### The wrong season stays selected after navigation

Season state is query-string-driven. Make sure the current URL still contains `?season=YYYY`.

### The browser console reports missing TanStack column IDs

The table now sanitizes sort state whenever the visible column set changes.
If you still see old column-ID warnings, hard-refresh the page so any stale client bundle is
discarded.

### Early seasons fail during the full pipeline

The early-season path previously failed inside the simultaneous QB adjustment step when a row had a
missing `opponent_team`, and older schedule aliases such as `SD` and `OAK` could also poison joins.
Those issues are now handled in the current code, so re-running the affected seasons should rebuild
cleanly.

### A team or QB detail page says the entity was not found

The detail route depends on the ID or label in the normalized backend payload. If the season changed
or the compare query references stale IDs, navigate back to the index and reselect the entity.

## Debugging Notes

- Frontend build and type issues are usually reproducible with `npm run build`.
- Backend contract issues are usually reproducible with `pytest tests/test_ui_data.py
  tests/test_ui_api.py`.
- If the UI looks wrong for a specific metric, inspect the backend payload before editing the
  frontend assumptions.
- The UI should never silently redefine a data column's meaning.

## Validation Commands

From the repository root:

```bash
ruff format .
ruff check .
ty check .
pyright .
pytest
```

From `ui/web/`:

```bash
npm run build
```
