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
- `GET /api/seasons/{season}/teams/rating-ranges` and
  `GET /api/seasons/{season}/qbs/rating-ranges`: every team's (or qualifying quarterback's) rating
  and rank percentiles, top-5 and top-10 chances, and the chance of each rank over game-bootstrap
  resamples, ordered by published rank; team payloads add an `offense_range`, `defense_range`, and
  `special_teams_range` column group when the file has them
- `GET /api/seasons/{season}/teams/{team}/rank-history` and
  `GET /api/seasons/{season}/qbs/{qb_id}/rank-history`: the rank percentiles and top-5 and top-10
  chances as of each week, for the season in progress only
- `GET /api/seasons/{season}/teams/{team}/rating-pairs` and
  `GET /api/seasons/{season}/qbs/{qb_id}/rating-pairs`: one team's (or qualifying quarterback's)
  head-to-head chances against every other one: how often it was rated above, the percentiles of
  the rating difference, and (for quarterbacks) the share of resamples with both
- `GET /api/seasons/{season}/teams/wp-ratings?threshold=X` and
  `GET /api/seasons/{season}/qbs/wp-ratings?threshold=X`: the garbage-time filter view, every team's
  (or qualifying quarterback's) ratings refit on the plays whose win probability before the snap
  was between X% and 100% minus X% (X from 0 to 20, default 0; outside that range is a 422), beside
  the published rating and rank, ordered by filtered rank

A season is listed only when all six contract files exist: `{season}_team_per_game_stats`,
`{season}_qb_per_game_stats`, `{season}_combined`, `{season}_qb_combined`, `{season}_ratings`, and
`{season}_qb_ratings` (all `.parquet`). Game-log views also read `{season}_team_game_logs` and
`{season}_qb_game_logs`, and the rating-history chart reads `{season}_ratings_by_week` and
`{season}_qb_ratings_by_week`; the chart is left out when a season has no history file. The
rank-range endpoints read `{season}_rating_ranges` and `{season}_qb_rating_ranges`, and the
head-to-head endpoints `{season}_rating_pairs` and `{season}_qb_rating_pairs`, and the
rank-history endpoints `{season}_rating_ranges_by_week` and `{season}_qb_rating_ranges_by_week`;
each returns 404 when its file is missing. The filter endpoints read the season's game logs, `{season}_team_wp_bins`
or `{season}_qb_wp_bins`, and its ratings file (teams also the previous season's game logs, whose
penalties the team fit reuses), return 404 when one is missing, and cache each season's model and
each threshold's table in process until a file changes.

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
                     breakdown, weekly trend chart, rank-range chart and histogram
    domain/          pure view logic: entity config, view state, formatting, metric metadata,
                     detail analytics, trend points, rank ranges
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
- The `CSV` button above the index table downloads the table as shown: the current view's columns
  in display order and the rows after the search, in the current sort, with raw values (full
  precision, not the rounded display) and column keys as the header.
- Tick rows to compare them. The selection lives in `?compare=` so a comparison can be shared. The
  comparison panel sets the picks side by side: one column each, headed by the name, the published
  rank, the middle 50% of resampled ranks with a mini interval, and a remove button; one row per
  metric of the current view, heat-mapped across the picks, with the metric names pinned while the
  table scrolls sideways.
- Underlined labels, info icons, and chart points explain themselves in a hint card: hover or focus
  with a mouse, tap on a phone or tablet (tap anywhere else to close). On touch screens a tap on a
  column header sorts, and the info button beside it explains the column.
- On phones only the name column stays pinned while the table scrolls sideways, and the view
  toggles scroll on one row.
- When a season has rank-range files, the `Ratings` view gains a `Rank range` column (the middle
  50% of ranks across game-bootstrap resamples, with a mini interval), and a `Rank ranges` chart
  below the table draws every team or qualifying QB: thick bar for the middle 50%, thin bar for the
  middle 95%, a dot for the median, and a diamond for the published rank when it differs. The
  readout above the chart describes the hovered or tapped row (on a phone, its link opens the
  detail page). The detail page adds the rank headline, the top-5 and top-10 chances, the chance
  of each rank, and, for teams, a `Rank range by unit` table: offense, defense, and special teams
  with their rank ranges and mini intervals.
- In the detail page's game-by-game table, each opponent shows its season-long rank range (the
  middle 50% of redrawn ranks, with a mini interval): the team's own on team pages, its defense's
  on quarterback pages.
- For the season in progress, the detail page adds a `Rank by week` chart below `Rating by week`:
  the median rank with bands for the middle 50% and 95% of redraws, rank 1 at the top, a hover or
  tap readout, and a text summary and hidden table for screen readers. The card says that the
  first weeks' bands understate the uncertainty.
- When a season has head-to-head files, the detail page adds a `Head to head` card: a `Compare
  with` picker, starting on the team or QB ranked just above (just below for the leader), and one
  sentence such as "NE rated above BUF in 38% of resampled seasons; difference -1.2 points, 95%:
  -5.0 to +2.8." The comparison panel shows the same sentence when exactly two rows are compared.
- The `Garbage-time filter` card holds a slider from Off to 20%, kept in the address as `?wp=`. A
  threshold of X% asks `/api/seasons/{season}/{teams|qbs}/wp-ratings` for the ratings refit on the
  plays whose win probability before the snap was between X% and 100% minus X%. It then lists
  every team or qualifying QB by filtered rank, beside the change, the published rank and rating,
  and the share of plays kept, under an "Unvalidated exploration view" label. Detail pages show
  the same for one row. The slider waits 250 ms after it stops moving before asking. Rank ranges
  and everything else on the page still count every play.
- Click a team or QB to open its detail page: stat tiles, metric sections, the weekly trend chart
  (pick any numeric column of the current view; the season mean is drawn as a reference line), the
  game log, and the unique-opponent breakdown.
- The header toggles light, dark, or system theme, and its `Palette` menu offers the default palette
  (blue accent, green-to-red heat scale) or any team's, grouped by division; it opens at the chosen
  palette. A team palette recolors the accent, chart colors, heat scale, hover and selected
  backgrounds, tooltips, and app logo in the team's colors and marks the top of the header with its
  two colors; page backgrounds and panels stay the same neutral colors in every palette, changing
  only between light and dark. The chosen palette applies to the Teams, Quarterbacks, and Glossary
  pages; a team or QB page switches to that team's palette, unless the menu's "Use each team's
  colors on its page" switch is off. Every team has its own heat scale: one team color marks the
  better end in both modes; a team whose second color is black, white, or silver, or whose two
  colors tint alike, gets a muted gray at the other end, and the Raiders (black and silver) a scale
  of grays, darker better in light mode and lighter better in dark mode. The palettes live in
  `src/domain/teamPaletteData.json`, generated by `nfl-sos-ratings team-palettes` from nflverse team
  colors with readable contrast checked (`nfl_sos_ratings/team_palettes.py`).
- Whatever the palette, each team abbreviation in the tables, the comparison, and the detail-page
  title carries a small chip in that team's two main colors.
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
- **Charts look different with a dark-mode extension.** `index.html` carries
  `<meta name="darkreader-lock">`, so Dark Reader leaves the app alone; it repainted the rank-range
  marks invisible and overrode the heat-map colors. Use the header's theme toggle for dark mode.
