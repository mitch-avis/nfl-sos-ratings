# Current Project Status

The handoff document for the repo's current state: what is done, what is open, and what the next
agent should do first. Update it in the same change set whenever any of that changes. Earlier
history (the composite-rating era and its experiments) is in git, before commit `21c5290`.

## Current state (2026-10-08)

- `main` = `origin/main` = `72cc113`, the maintainer's dependency refresh (Polars 1.44.2 to 2.0.0,
  a major version, plus fastapi, filelock, ty, and others). Polars 2.0 parity, checked 2026-10-08
  in a worktree of `3b5946f` (Polars 1.44.2) against `main`: no deprecation warnings in the suite
  (`pytest -W default::DeprecationWarning`); the package has no lazy queries (2.0's streaming
  default reorders lazy joins and group-bys); scratch `season` builds of 1999, 2006, 2016, 2025,
  and 2026 under each version gave bit-identical ratings, ranges, pairs, histories, bins, and game
  logs, with the four descriptive opponent-profile files per season differing by float rounding
  only (`diff-data --tolerance 1e-9`: all unchanged); the 1.44.2 builds of the four completed
  seasons matched `data/` exactly; Polars 2.0 is deterministic run to run; 1,203 API payloads were
  identical between servers on each version; and `check-additivity` and
  `check-in-season-penalty` printed identical output.
- Branch `feat/team-color-depth` (2026-10-08, not yet pushed): the generated Broncos palette
  (maintainer approval), tinted surfaces, accent backgrounds, team logo and header stripe, team
  heat scales for all 32 teams, and team color chips (`.agents/roadmap.md`, F7 follow-up).
- New since the last handoff, all in the roadmap: S5 (Polars single-threaded by default for tests
  and builds, plus the `published_data` coverage decision), "Data notes" (three 1999-2000 games
  missing from nflverse play-by-play; the 2026 Broncos rating explained), workstream U (the
  2026-10-08 UX audit), and a parked preseason-prior idea.

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

All open work is in `.agents/roadmap.md`, the single active plan, in the recommended order. Done and
merged: the `feat/rank-ranges` pull request (#1), the nfl-predictor note, the three small fixes in
pull request #2 (single-threaded BLAS by default, one fixed row order per data file, and the
read-only `nfl-sos-ratings diff-data` command), the garbage-time filter's bins (#3), and its refits
per threshold and API (#4, `/api/seasons/{season}/{teams|qbs}/wp-ratings`), and the slider with its
filtered view (#5; phone layout checked by the maintainer). Pull request #6 brought each QB's team
back to the filtered table (the freeze that kept it out did not recur with the maintainer's browser
extensions disabled; roadmap, WP3), lowered the filter's maximum from 30% to 20%, and bumped
`filelock` in `uv.lock`. Pull request #7 (merged 2026-10-05) brought the pre-registered garbage-time
filter test (WP4): no threshold predicts margins better and 20% is significantly worse, so the
published ratings keep every play (maintainer decision; roadmap, WP4, and
`.agents/ratings-simplification-plan.md`). Overnight on 2026-10-05, with the maintainer's
merge-on-green approval, pull requests #8-#16 landed head-to-head chances (R1), unit rank ranges
(R2), weekly rank ranges for the season in progress (R3), the weekly refresh script (A1, no
scheduled task installed), the project logger (S4), opponent rank context (F3), the side-by-side
comparison (F4, F6), CSV export (F5), and team palettes (F7); `data/` was rebuilt after R1+R2 and
2026 again after R3. Decisions waiting for the maintainer are listed in the roadmap: the weekly
chart's early weeks (R3), the Broncos light-mode contrast (F7), installing the scheduled refresh
(A1), and splitting `team_metrics.py`. Still open: F1, F2, and M1 (they need the maintainer's
input). Rebuild 2026 weekly with `scripts/refresh-season.sh` (ask first; it copies `data/`,
rebuilds, runs the `published_data` tests, and prints `diff-data`), or schedule it with the command
in README.

## Validation snapshot

Generated by `nfl-sos-ratings validate --data-dir data --start-season 1999 --end-season 2025
--start-week 5 --report-path docs/validation-report.md`, last run on 2026-10-04 after the QB data
fixes (team numbers unchanged from the run after adopting previous-season penalties):

- Overall walk-forward MAE: `team_rating` 10.601, SRS 10.658, raw EPA 10.695, Elo 10.580.
- `team_rating` versus raw EPA: -0.095 (95% CI -0.154 to -0.038), significantly better.
- `team_rating` versus SRS: -0.057 (95% CI -0.125 to +0.008), a tie. Decision: adopt.
- Year-over-year Pearson: `team_rating` 0.434, SRS 0.437; adjusted EPA per dropback 0.455, passer
  rating 0.464, ANY/A 0.392 (601 QB pairs). Mean QBR correlation 0.892 / 0.874.

Garbage-time filter test (2026-10-05, `nfl-sos-ratings check-wp-filter --data-dir data
--start-season 1999 --end-season 2025 --start-week 5`): MAE 10.601 with every play, 10.623 at 5%,
10.663 at 10%, 10.733 at 20%; 5% and 10% tie, 20% is worse (+0.133, 98.33% interval +0.046 to
+0.222). The published ratings keep every play.

Gate state: `scripts/gate.sh --web` passes, and `.venv/bin/pytest -m published_data --no-cov` passes
on the rebuilt `data/` (6 tests, 2026-10-05). Earlier notes said the command without `--no-cov`
passed; its tests did, but the coverage floor makes that command exit 1.

Season rollover: after the 2026 season, set `END_YEAR` to 2026 and `SEASON` to 2027 in
`nfl_sos_ratings/config.py`, rebuild (ask first), and regenerate the validation report. The weekly
rank ranges follow `SEASON`, so 2026's stay as last written and 2027 starts its own.

## Open items

Tracked in `.agents/roadmap.md` (BLAS threading is S1, the project logger S4, frontend follow-ups
F1-F6). `pytest-html` and `pytest-metadata` stay as dev dependencies (decided 2026-10-02).

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
