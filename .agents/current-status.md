# Current Project Status

The handoff document for the repo's current state: what is done, what is open, and what the next
agent should do first. Update it in the same change set whenever any of that changes. Earlier
history (the composite-rating era and its experiments) is in git, before commit `21c5290`.

## Current state (2026-10-08, end of session)

- `main` = `origin/main` = `de3a271` plus this handoff. Pull requests #17-#38 landed on
  2026-10-08, each merged with a merge commit once the gate and CI passed (the maintainer approved
  merging on green; roadmap, "Session plan"):
  - Team colors and palettes: #17 (generated Broncos palette, team heat scales, team chips, logo,
    header stripe, team pages in their team's colors) and #22 (palette menu as a division grid).
  - Speed and tests: #18 (Polars single-threaded by default; `pytest -m published_data` needs no
    `--no-cov`).
  - App fixes and layout: #19 (season-in-progress notice from the API, copy fixes, not-found and
    unavailable-season notices), #21 (straight trend lines, round ticks, rank by week from three
    games), #29 and #30 (index pages: table above the fold, reading-notes popover, full-height
    table, comparison below the table with season-range shading), #31 (detail pages: ranked rating
    summary with the before-adjustment value, stat-view tabs on the stats section, retired jargon
    tiles, QB rating companions in the API), #33 (context columns shaded in one neutral hue), #34
    (registry-built glossary, structured stat hints), #35 (web README), and #37 (the app's
    explanatory copy in plain words).
  - Refresh button: #24 (`nfl-sos-ratings web --allow-refresh`; the app refetches every query
    after a run, so new data shows without a frontend rebuild).
  - Registry: #20 (`team_metrics.py` split into five category modules) and #38 (labels,
    descriptions, formulas, affix notes, and category descriptions in plain words; column keys
    unchanged).
  - Data correctness (2026-10-08 audit): #23 (Rams `LA` codes in every play-by-play column), #25
    (QB sack-yards sign, scrambles and kneels, win %, snaps), #27 (seven team stat fixes), #28 (app
    display: longest plays, vs-season columns, percentages, affix rules), and #36 (season rates
    pooled over the season's plays instead of averaged over games, blanks instead of zeros where
    nflverse has no data, passer-rating ties rounded half up, tests that never download). None
    changes a rating file; all reach the app only after a `data/` rebuild.
  - Preseason prior: #26 (protocol pre-registered before any code) and #32 (the check and its
    results; roadmap P6).
- Polars 2.0 (the maintainer's dependency refresh, `72cc113`) was checked against 1.44.2 on
  2026-10-08 in a worktree of `3b5946f`: no deprecation warnings in the suite (`pytest -W
  default::DeprecationWarning`); the package has no lazy queries (2.0's streaming default reorders
  lazy joins and group-bys); scratch `season` builds of 1999, 2006, 2016, 2025, and 2026 under each
  version gave bit-identical ratings, ranges, pairs, histories, bins, and game logs, with the four
  descriptive opponent-profile files per season differing by float rounding only (`diff-data
  --tolerance 1e-9`: all unchanged); the 1.44.2 builds of the four completed seasons matched
  `data/` exactly; Polars 2.0 is deterministic run to run; `.venv/bin/python
  .agents/findings_2026_10_08/api_parity.py http://127.0.0.1:8090 http://127.0.0.1:8092 1999 2012
  2025 2026` (a server on each version) reported 1,203 payloads compared and 0 differing; and
  `check-additivity` and `check-in-season-penalty` printed identical output.
- `data/` was rebuilt on 2026-10-08 from `485ae56` (maintainer approval) with `nfl-sos-ratings
  pipeline` and `nfl-sos-ratings season --season 2026`, publishing the data fixes (#23, #25, #27,
  #36, #41, #42). `nfl-sos-ratings diff-data --before <copy of data/ before> --after data
  --tolerance 1e-9` reported 280 files unchanged and 226 with changed values: the descriptive stat
  files of every season; no team rating, range, pair, history, or win-probability file, and in the
  QB ratings files only `qb_attempts_total` for one passer in 2001 and one in 2002. The validation
  rerun is in the snapshot below. `.venv/bin/pytest -m published_data` passes on the rebuilt data.
- The preseason prior (#47, maintainer approval) was published by a second 2026-10-08 rebuild from
  `f919beb` (`nfl-sos-ratings pipeline`, then `nfl-sos-ratings season --season 2026`).
  `nfl-sos-ratings diff-data --before <copy of data/ before> --after data --tolerance 1e-9`: 477
  files unchanged, 28 added (`{season}_team_prior`, empty once every team has played 9 games), and
  29 with changed values: `ratings_by_week` for 2003-2025 (weeks 1-9) and the 2026 ratings,
  `combined`, ranges, pairs, and weekly files. No completed season's ratings, ranges, or pairs
  changed, and no QB file did. 2026 DEN is +0.96, 12th. `.venv/bin/pytest -m published_data`
  passes on the rebuilt data, and the validation rerun is in the snapshot below.

## Approved and in progress (maintainer answers of 2026-10-08)

1. Preseason prior at 9 games: done, #47, then the `data/` rebuild and the validation rerun
   above. Left for the maintainer: the nfl-predictor note (roadmap P6, "If adopted").
2. Default heat scale blue to orange, and deeper light-mode shading in every palette (U17): done,
   #45.
3. QB rating from all of a quarterback's plays (scrambles and designed runs, not only dropbacks):
   write the pre-registered protocol first; adoption is the maintainer's. Drafted 2026-10-08 on
   `docs/qb-all-plays-protocol` (roadmap Q1, with `.agents/findings_2026_10_08/qb_play_types.py`
   and `qb_protocol_checks.py`), independently reviewed with every finding resolved; it waits on
   five maintainer questions listed there.
4. The garbage-time filter, kept but more compact: done, #46 (one control above the table).
5. A plan for bringing this app's features into nfl-predictor (heads-up; nothing done there).
6. Scrambles before 2006 in the team ratings (approved 2026-10-08 after a deeper check): nflverse
   codes scrambles as pass plays (`rush` 0) and before 2006 leaves `qb_dropback` at 0 on most of
   them, so the scrimmage-snap filter dropped them from the 1999-2005 team ratings, team
   `dropbacks`, and team passing EPA. Fixed on `fix/pre-2006-scrambles`
   (`pbp_expressions.dropback_expr`); the `data/` rebuild and the `validate` rerun follow (roadmap
   "Data notes", "Team scrimmage plays left out scrambles before 2006").
7. Quarterbacks also listed at another position (approved 2026-10-08): a player any source lists at
   QB in a season counts as a QB that season, in the identity crosswalk and the official weekly
   stats (Taysom Hill, Terrelle Pryor, and a few others had lost every QB row). Fixed on
   `fix/qb-multi-position`; the `data/` rebuild follows (roadmap "Data notes", "QB rows by career
   position").

## Decisions waiting for the maintainer

1. Smaller open items in the roadmap's "Data notes", each a `data/` change: the extra-point drive
   after a return touchdown, the near-duplicate yards-per-snap columns, `opp_longest_*` averaging
   per-game maxima, `fourth_down_aggressiveness` at 2.0 in two 2000 games, play-by-play as a
   2003-2011 source for tackles for loss, and kneel-downs under-recorded in 2000 and 2001.

## State before 2026-10-08

- `main` (pushed, `18c8aaa`) holds the points-based ratings, the follow-up work, previous-season
  ridge penalties for the team fit, the rank ranges, QB data fixes, and UX audit described below
  (pull request #1), the rebuild tooling (pull request #2), and the garbage-time filter's
  win-probability bins (#3), refits per threshold with their API (#4), and slider with its
  exploration view in the web app (#5) on 2026-10-04, and #6 (QB teams in the filtered table, a 20%
  filter maximum) and #7 (the garbage-time filter test) on 2026-10-05.
- Published team ratings: `team_rating` (points per game against an average team) with
  `offense_rating`, `defense_rating`, and `special_teams_rating` adding up to it,
  head-to-head-excluded `sos`, and `SRS` as the score-based reference. The team fit reuses the
  previous season's cross-validated penalties (1999 cross-validates its own). QB ratings:
  `adj_qb_epa_per_dropback` with `qb_epa_per_dropback` and the head-to-head-excluded
  `qb_faced_pass_defense`, penalty cross-validated per season. Definitions:
  `nfl_sos_ratings/team_rating.py`, `nfl_sos_ratings/qb_rating.py`, and `docs/methodology.md`.
- Every season has rating histories (`{season}_ratings_by_week`, `{season}_qb_ratings_by_week`),
  served at `/api/seasons/{season}/{teams|qbs}/{id}/rating-history` and charted on the detail
  page. Week rows reuse the season fit's penalties.
- Decisions, audits, pre-registered rules, and check results (additivity, passer holdout,
  in-season penalty) are in `.agents/ratings-simplification-plan.md`.
- Pull request #1 (`feat/rank-ranges`, merged with a merge commit, branch deleted) added bootstrap
  rank ranges (`{season}_rating_ranges`, `{season}_qb_rating_ranges`, their API and web views),
  the QB data fixes (duplicated QB-game rows, deterministic tie-breaks) and the per-team QB
  qualifier (`qb_attempt_qualifier`), and the UX audit changes (one hint style that also opens on
  tap, phone layout, fixed decimals per column). Details: `.agents/roadmap.md`, "Settled
  background".
- `data/` (1999-2026: range files, fixed row order, win-probability bins; 2026 through week 4) was
  last rebuilt on 2026-10-04 from `c574819` with `nfl-sos-ratings pipeline` and `nfl-sos-ratings
  season --season 2026`; `diff-data` against the previous build showed no value changes
  (`.agents/roadmap.md`, S2).
- `docs/validation-report.md` was regenerated on 2026-10-04 after the QB data fixes; the rule still
  reads adopt.

## Next steps

All open work is in `.agents/roadmap.md`, the single active plan, in the recommended order. The
2026-10-08 session plan is done except the QB all-plays protocol; "Approved and in progress"
above lists what is left.
Earlier work, pull requests #1-#16 (rank ranges, rebuild tooling, the garbage-time filter and its
test, head-to-head chances, unit and weekly rank ranges, the refresh script, the project logger,
opponent context, comparison, CSV export, team palettes), is described in the roadmap. Rebuild 2026
weekly with `scripts/refresh-season.sh` (ask first; it copies `data/`, rebuilds, runs the
`published_data` tests, and prints `diff-data`), or with the app's refresh button when the server
runs with `--allow-refresh`.

## Validation snapshot

Generated by `nfl-sos-ratings validate --data-dir data --start-season 1999 --end-season 2025
--start-week 5 --report-path docs/validation-report.md`, last run on 2026-10-08 after the
preseason prior's `data/` rebuild (before the prior, the same day: `team_rating` 10.601, raw EPA
-0.095, SRS -0.057 and a tie; the QB rows are unchanged):

- Overall walk-forward MAE: `team_rating` 10.567, SRS 10.658, raw EPA 10.695, Elo 10.580.
- `team_rating` versus raw EPA: -0.128 (95% CI -0.190 to -0.069), significantly better.
- `team_rating` versus SRS: -0.091 (95% CI -0.159 to -0.024), significantly better. Decision:
  adopt.
- Prediction weeks 5-7: `team_rating` beats Elo (-0.148, -0.291 to -0.004) and raw EPA (-0.245);
  overall it ties Elo (-0.012, -0.081 to +0.057).
- Year-over-year Pearson: `team_rating` 0.434, SRS 0.437; adjusted EPA per dropback 0.455, passer
  rating 0.460, ANY/A 0.398 (601 QB pairs). Mean QBR correlation 0.892 / 0.874.

Garbage-time filter test (2026-10-05, `nfl-sos-ratings check-wp-filter --data-dir data
--start-season 1999 --end-season 2025 --start-week 5`, the fit before the prior): MAE 10.601 with
every play, 10.623 at 5%, 10.663 at 10%, 10.733 at 20%; 5% and 10% tie, 20% is worse (+0.133, 98.33%
interval +0.046 to +0.222). The published ratings keep every play.

Gate state: `scripts/gate.sh --web` passes on `main` (`f919beb`, 2026-10-08), and
`.venv/bin/pytest -m published_data` passes on `data/` (8 tests, 2026-10-08; with only those tests
selected the coverage floor is lifted, so `--no-cov` is not needed).

Season rollover: after the 2026 season, set `END_YEAR` to 2026 and `SEASON` to 2027 in
`nfl_sos_ratings/config.py`, rebuild (ask first), and regenerate the validation report. The weekly
rank ranges follow `SEASON`, so 2026's stay as last written and 2027 starts its own.

## Open items

Tracked in `.agents/roadmap.md`: F1 and F2 (frontend follow-ups) and M1 (retired stats) wait for
the maintainer's input. `pytest-html` and `pytest-metadata` stay as dev dependencies (decided
2026-10-02).

## What the next agent should do first

1. Read `.agents/roadmap.md` and start at the first unchecked workstream; the decision record
   (rating specifications, pre-registered rules and results) is
   `.agents/ratings-simplification-plan.md`.
2. If the task challenges the rating methodology, start from `docs/methodology.md` and
   `docs/validation-report.md`, and write a falsifiable protocol and decision rule in an `.agents/`
   plan before changing code.
3. After any registry change, run `nfl-sos-ratings catalog`; the catalog drift test fails otherwise.

## File-keeping rule

Do not create one-off `.agents/` plan files for completed work. Fold lasting context into this file
or into the still-active plan that owns the remaining work.
