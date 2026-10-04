# NFL SOS Ratings web app

A local analyst console for the season outputs that the Python pipeline writes to `data/`. It is
not a public site. It answers questions like:

- Which teams and quarterbacks rate best once opponent strength is accounted for?
- How do the ratings sit next to the raw totals, rates, and opponent context that produced them?
- How did one team or QB trend week to week, and against which opponents?

## Stack

- React 19, TypeScript, and Vite, with React Router for routes.
- Tailwind CSS v4 and shadcn/ui components (Radix primitives, `components.json`) under
  `src/components/ui/`.
- TanStack Query for API data and TanStack Table for the sortable tables.
- Recharts for the weekly trend chart.
- Vitest with Testing Library and jsdom for tests; oxlint for linting.

## Running it

From the repository root, after the pipeline has written at least one season to `data/`:

```bash
cd web && npm ci && npm run build && cd ..
uv run nfl-sos-ratings web   # app and API on http://127.0.0.1:8080
```

`nfl-sos-ratings web` serves the built app from `web/dist` (with a fallback to `index.html` for
client-side routes) and the JSON API under `/api` on one port. Without `web/dist` it still serves
the API and shows a page telling you to build the app. Options: `--host` (default `127.0.0.1`),
`--port` (default `8080`), `--data-dir` (default `data`), `--reload`.

For frontend development with hot reload, keep `nfl-sos-ratings web` running for the API and start
the Vite dev server in a second terminal:

```bash
cd web
npm run dev   # http://127.0.0.1:5280, proxies /api to 127.0.0.1:8080
```

The ports (8080 and 5280) are chosen so this app can run beside nfl-predictor's (8000 and 5173).

## Checks

From `web/`:

```bash
npm run lint        # oxlint
npm run typecheck   # tsc -b
npx vitest run      # unit and component tests
npm run build       # tsc -b, then the Vite production build into dist/
```

`scripts/gate.sh --web` from the repository root runs the Python gate plus these frontend checks.
Run it whenever `web/` or the API payloads change.

## Data contract

The app never recomputes methodology. The backend (`nfl_sos_ratings/ui_data.py` and
`nfl_sos_ratings/ui_api.py`) reads the generated Parquet files and returns JSON:

- `GET /api/health`
- `GET /api/metadata`: the metric registry (labels, descriptions, polarity) for headers and
  tooltips
- `GET /api/seasons`: seasons with a complete contract
- `GET /api/seasons/{season}`: team and QB rows for one season
- `GET /api/seasons/{season}/teams/{team}/game-logs` and
  `GET /api/seasons/{season}/qbs/{qb_id}/game-logs`
- `GET /api/seasons/{season}/teams/{team}/rating-history` and
  `GET /api/seasons/{season}/qbs/{qb_id}/rating-history`: the rating as of each week, each fit on
  the games through that week

A season is listed only when all six contract files exist: `{season}_team_per_game_stats`,
`{season}_qb_per_game_stats`, `{season}_combined`, `{season}_qb_combined`, `{season}_ratings`, and
`{season}_qb_ratings` (all `.parquet`). Game-log views also read `{season}_team_game_logs` and
`{season}_qb_game_logs`.

## Layout

```text
web/
  index.html, vite.config.ts, components.json, package.json
  public/            static assets (favicon)
  src/
    main.tsx, App.tsx, router.tsx, index.css
    api/             fetch client, TanStack Query hooks, payload types
    app/             app shell (sidebar, header, season picker), theme and view-state providers
    pages/           Teams and QBs index, entity detail, glossary, season-data route wrapper
    components/
      ui/            shadcn/ui primitives
      common/        page header, stat tile, loading, empty, and error states, tooltips
      entity/        entity table, view controls, comparison panel, game logs, opponent
                     breakdown, weekly trend chart
    domain/          pure view logic: entity config, view state, formatting, metric metadata,
                     detail analytics, trend points
    hooks/, utils/   small shared helpers
    test/            test setup, fixtures, render helper
```

Keep logic that can be tested without a browser in `src/domain/`, with a `*.test.ts` beside it.

## Using the app

- **Teams** and **Quarterbacks** index pages list one season, chosen in the header. The season is
  in the URL (`?season=YYYY`).
- One view is active at a time: `Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`,
  `Opponent Per-Game Rates`, or `Opponent Per-Play Rates`. Non-`Ratings` views add the team
  category row (`Overall`, `Offense`, `Defense`, `Special Teams`) and subcategory toggles, or the
  QB subcategory toggles.
- Search filters the visible columns; identity columns stay pinned while scrolling sideways.
  `Reset` restores the default view.
- Tick rows to compare them. The selection lives in `?compare=` so a comparison can be shared.
- Click a team or QB to open its detail page: stat tiles, metric sections, the weekly trend chart
  (pick any numeric column of the current view; the season mean is drawn as a reference line), the
  game log, and the unique-opponent breakdown.
- The header toggles light, dark, or system theme, and the classic (green to red) or Broncos
  (orange to navy) heatmap palette.
- **Glossary** explains every rating and which ones are primary rankings versus context.

## Troubleshooting

- **The season picker is empty.** No season in `data/` has all six contract files. Run
  `nfl-sos-ratings season` or `nfl-sos-ratings pipeline` and check `data/`.
- **The dev server loads but API calls fail.** `nfl-sos-ratings web` is not running on port 8080;
  the Vite proxy expects it there.
- **Port 8080 serves an old app.** It serves whatever is in `web/dist`; rerun `npm run build`.
- **A detail link lands on the index page instead.** The ID in the URL is not in the selected
  season, so the app redirects to that season's index. Pick the entity again from there.
- **A metric looks wrong.** Inspect the `/api` payload before changing frontend code; the app must
  never silently redefine a column's meaning. Backend contract tests live in
  `tests/test_ui_data.py` and `tests/test_ui_api.py`.
