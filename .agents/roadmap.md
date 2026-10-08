# Roadmap

The single active plan for `nfl-sos-ratings` (consolidated 2026-10-04 from the rank-range and WP
filter plan and the frontend plan's follow-up queue). It lists every workstream the maintainer
approved on 2026-10-04, in the recommended order, each with its background, the reasoning behind
it, a design, tasks, and how to tell it is done. Work top to bottom unless the maintainer reorders.

Companion files: `.agents/current-status.md` (repo state, validation snapshot, handoff),
`.agents/ratings-simplification-plan.md` (decision record: rating specifications, pre-registered
rules and their results, the retired-metric list), and `.agents/frontend-ui-kickoff-plan.md`
(frontend build history, design direction, and the 2026-10-04 UX audit).

## How to use this plan

- Approval: the maintainer agreed to every workstream below on 2026-10-04. That is approval to
  build them. Steps marked **Ask first** still need a yes at the time, with the usual context
  (AGENTS.md, "Questions and decisions for the user"), because they rebuild `data/`, rerun the
  validation, publish, or change machine or tool configuration.
- Each workstream is sized to land as one or a few gate-green commits. Update its task boxes, and
  `current-status.md` when the repo state changes, in the same change set.
- Numbers in this file come from a repo command or checked-in helper with the command beside them
  (AGENTS.md). Scratch timings are labeled as such and are not citable.
- Plan labels (H1, S2, WP3, ...) stay inside `.agents/`; never use them in code, tests, commits, or
  user-facing docs.

## Where things stand (2026-10-05)

- Pull request #1 (`feat/rank-ranges`, merged as `c396d83`, branch deleted) brought the rank
  ranges (engine, outputs, API, web views), the QB data fixes, the per-team QB qualifier, the UX
  audit changes, the regenerated validation report, and docs.
- S1-S3 merged as pull request #2 (`1d61b2d`), WP1 as #3 (`c574819`), WP2 as #4 (`b70b522`), WP3
  as #5 (`02a2b47`), and its follow-ups (each QB's team in the filtered table, a 20% filter
  maximum, a `filelock` bump) as #6 (`f6da28b`), and WP4 (the protocol, `check-wp-filter`, its
  results, and the decision to keep every play) as #7 (`main` = `18c8aaa`), branches deleted.
- Merged overnight under those approvals: #8 R1, #9 R2, #10 R3, #11 A1, #12 S4, #13 F3, #14 F4
  and F6, #15 F5, #16 F7. Open decisions are marked in their sections (R3, A1, F7, and the
  `team_metrics.py` split proposal under R1).
- Overnight approvals (maintainer, 2026-10-05): the agent may push, merge each pull request to
  `main` with a merge commit once `scripts/gate.sh --web` passes locally and CI is green, delete
  the merged branch, and branch the next task from the updated `main`; rebuild `data/` once after
  R1 and R2 land (copy first, quote `diff-data`, run `pytest -m published_data`), and again if R3
  lands. R3 scope: if weekly ranges add more than about 2 minutes to `pipeline`, build them only
  for the season in progress at the full 1000 resamples. Tonight's scope after R1-R3: A1 (script
  only, no scheduled task), S4, F3-F6, and F7 (team palettes). M1, F1, and F2 wait for the
  maintainer.
- `data/` (1999-2026: range files, fixed row order, and win-probability bins) was rebuilt on
  2026-10-04 from `c574819` with no value changes (S2 records the `diff-data` summary), and
  `.venv/bin/pytest -m published_data` passes on it.
- The maintainer runs `nfl-sos-ratings web --host 0.0.0.0 --port 8081` to view the app on a phone.
  Never stop it; use port 8090 for agent checks.
- 2026-10-08: the maintainer's dependency refresh (`72cc113`, Polars 1.44.2 to 2.0.0) was checked
  for parity (S5 records how); F7's open Broncos decision was settled and team palettes deepened
  (F7, "Palette depth"); a fresh UX audit opened workstream U; S5 (test and pipeline speed) waits
  on one maintainer decision.

## Session plan (from 2026-10-08)

Maintainer decisions of 2026-10-08, in force for this plan: every recommendation in the
2026-10-08 handoff is approved; the agent may merge each pull request itself (merge commit, delete
the branch) once `scripts/gate.sh --web` passes locally and CI is green, then branch the next item
from the updated `main`; the preseason-prior check may be run when built (a `data/` rebuild still
needs a fresh yes); no scheduled refresh task (the maintainer refreshes by hand). Work top to
bottom; update the status boxes in the same change set as the work.

1. [x] P1 Team colors (pull request #17, merged 2026-10-08 as `77d06a0`, F7 follow-up): generated
   Broncos palette, team heat scales, team chips, logo, header stripe. Maintainer changes of
   2026-10-08: page surfaces stay the default neutral in every page and palette (only light or dark
   changes them); accents, charts, heat scale, stripe, logo, and tooltips follow the chosen palette
   on the Teams, Quarterbacks, and Glossary pages; a team or QB page switches to that team's
   palette automatically, with a "Use each team's colors on its page" switch (on by default) in the
   palette menu; the palette menu opens at the chosen team.
2. [x] P2 Test and pipeline speed (S5): `POLARS_MAX_THREADS=1` by default in the front door and the
   test suite; `pytest -m published_data` works without `--no-cov` (conftest hook, approved).
   Done on `perf/single-thread-polars`; timings in S5.
3. [x] P3 Bugs and copy (U1-U4), on `fix/season-notice-and-copy`.
4. [x] P4 Split `nfl_sos_ratings/metrics/team_metrics.py` by category (approved), on
   `refactor/split-team-metrics`: `team_rating_metrics`, `team_overall_metrics`,
   `team_offense_metrics`, `team_passing_metrics`, and `team_defense_metrics` (209-907 lines each);
   `team_metrics.py` assembles `TEAM_METRICS` in the same order. Characterization: the catalog
   drift test (shown to fail when one description changes) and the registry payload, whose JSON was
   identical before and after; `nfl-sos-ratings catalog` left both catalogs unchanged.
5. [x] P5 Tooltip and glossary audit (maintainer request): every registry label, description, and
   formula, the affix rules, the app's own hint text, and a glossary rebuilt from the registry
   (search, categories, a "Start here" section, the methodology linked on GitHub). Tooltips gain a
   generated direction line and a "How it's computed" line. Drafts by six read-only subagents in
   `/tmp/tooltip-audit/` (style brief there), verified and applied centrally. Done in three pull
   requests. `feat/glossary` (#34): the glossary (U18) and the hint format (`MetricHint`: full
   name, sentence, generated direction line, and the registry formula for a base metric), with the
   twelve descriptions that stated a direction trimmed so it reads once. `fix/app-copy` (#37): the
   app's own explanatory text. `fix/registry-text`: the reviewed registry labels, full names,
   descriptions, and formulas (column keys unchanged; season rates described as pooled, after
   #36), the affix rules' wording (garbage-time filter, opponent and faced-defense averages,
   vs-season, per-snap, change, and the rank-range percentiles, whose note takes its count from
   `rating_ranges.BOOTSTRAP_RESAMPLES`), and the category descriptions, with both catalogs
   regenerated. Not done, by choice: composing `filtered_` and percentile columns without the base
   description, a draft idea to shorten those hints that changes how the registry composes text;
   it waits for a request. Waiting on the maintainer: `air_epa_total` polarity (recommend neutral,
   like the other air-yards columns: in a scratch check it tracked throwing depth more than
   quality), removing `player_id` and `player_display_name` from the registry (no file in `data/`
   has them; `ui_data._build_qb_payload` lists them only if present), and accepting nine labels of
   19-20 characters.
6. [ ] P6 Preseason prior for the team fit: the team fit shrinks toward a regressed previous-season
   rating that fades out early in the season (the maintainer expects the prior gone by mid-season
   or earlier; the fade point is for the pre-registered test to settle). Protocol first, then code
   (test-first), an independent review, the check run (approved), the decision, and a `data/`
   rebuild only with a fresh yes. Teams first; QBs as a separate later test. Protocol, reviewed
   and pre-registered: section "P6. Preseason prior for the team fit".
7. [x] P7 Index pages: U5-U8, on `feat/index-layout` (#29) and `feat/index-table` (#30).
8. [x] P8 Detail pages: U9-U12, on `feat/detail-layout` (#31).
9. [x] P9 Charts: U13-U14, with R3's early weeks (chart starts once every team has 3 games;
   approved), on `feat/chart-polish`.
10. [ ] P10 Color semantics: U15-U17. U15 and U16 done on `feat/color-semantics` (#33); U17 waits
    on the maintainer.
11. [x] P11 Palette menu as a division grid (U19), on `feat/palette-grid`; U18 lands with P5.
12. [x] P12 Refresh button (maintainer idea, 2026-10-08), on `feat/refresh-button`: `web
    --allow-refresh` (off by default; only when the server serves the repository's `data/`) runs
    `scripts/refresh-season.sh` in the background, one run at a time (`nfl_sos_ratings/
    refresh_runner.py`), behind `GET` and `POST /api/refresh`; a start needs the app's
    `X-Requested-With` header and, when the browser sends an `Origin`, this server's own. The header
    button opens a panel that says what a refresh does before its "Start refresh" button, then
    shows the run's start time and last output line, the `diff-data` summary and "the data checks
    passed" on success, or the last eight output lines on failure; the app polls every two seconds
    while a run goes and refetches every query when it ends. The maintainer turns it on by
    restarting their server with `--allow-refresh`.

## Recommended order

| Order | Workstream | Why here |
| --- | --- | --- |
| 1 | H1 Push and open a pull request | The branch is easier to review now than after the next features land on top |
| 2 | H2 Note for nfl-predictor | It reads this repo's docs; the qualifier and QB data changed |
| 3 | S1 Single-threaded BLAS by default | Makes every later rebuild faster without anyone remembering a flag |
| 4 | S2 Deterministic row order in outputs | Makes rebuild diffs show only real changes |
| 5 | S3 Data-diff helper | Turns the rebuild checks this session ran by hand into one command |
| 6 | WP Garbage-time WP filter (WP1-WP4) | The largest planned feature; decisions are already made |
| 7 | R1 Head-to-head chances | Answers "is NE better than X?" more directly than two ranges |
| 8 | R2 Rank ranges for the unit ratings | Nearly free: every resample already computes them |
| 9 | R3 Rank ranges by week | Shows how 2026 uncertainty narrows; costs more compute |
| 10 | A1 Weekly 2026 refresh automation | Best after S1-S3 make refreshes cheap and diffable |
| 11 | F1-F7 Frontend follow-ups | Detail pages, opponent context, compare layout, export, team palettes |
| 12 | S4 Project logger | Touches many modules; any quiet stretch works |
| 13 | M1 Retired stats, one at a time | On request; each needs its own verification |
| 14 | S5 Test and pipeline speed | Single-threaded Polars: a season build 32.7 s to 4.8 s (scratch) |
| 15 | U UX audit (2026-10-08) | Bugs first (U1-U4), then layout, detail page, charts, color semantics |
| 16 | P6 Preseason prior for the team fit | Pre-registered test first; adoption needs the maintainer |

## Settled background the workstreams build on

### The weighted bootstrap engine

With a fixed penalty, a game drawn k times in a bootstrap resample enters the least-squares fit
exactly as one copy with k times the weight; game identity mattered only for cross-validation
folds, which a fixed-penalty fit does not use. So the design matrix is built once per season
(`ridge.build_unit_design`) and each resample solves `(X' W X + penalty) b = X' W y`
(`ridge.solve_unit_design`), W being row weights times multiplicities. A zero multiplier removes
games, which is how head-to-head refits and any per-threshold refit (WP2) reuse the same engine.
`team_rating.TeamRatingResampler` and `qb_rating.QbRatingResampler` wrap it; an equivalence test
checks one resample against `fit_team_ratings` on the duplicated-and-relabeled frame.

### Rank ranges (done)

- Method: resample a season's `game_id`s with replacement (1000 resamples, seed 0), refit with the
  season fit's penalties, and rank every resample. Teams rank among all teams; QBs among the
  full-season `qb_is_eligible` passers present in the resample (`qb_rank_missing_share` says how
  often one was absent). Summaries come from `rating_ranges.summarize_rank_ranges`.
- Calibration tolerance, fixed before the first run: an eight-team synthetic league, each pair
  meeting four times, 30 replications of 200 resamples; pooled 95% intervals must cover the truth
  85-100% of the time and 80% intervals 65-95% (`tests/test_rating_ranges.py`). It passes.
- The ranges cover game-to-game sampling noise of the shrunken estimate, not model error
  (`docs/methodology.md`, "Rank Ranges"). Any new probability built from the same resamples (R1,
  R3) carries the same caveat.
- 2025 results, read with this command (and the same on `2025_qb_rating_ranges` for
  `qb_name == 'Drake Maye'`):

  ```bash
  .venv/bin/python -c "import polars as pl; print(pl.read_parquet(
      'data/2025_rating_ranges.parquet').filter(pl.col('team') == 'NE'))"
  ```

  - NE: published 5th; median 6th; middle 50% 3rd-9th; 95% 1st-16th; P(top 5) 0.488, P(top 10)
    0.820; P(rank 10-12) 0.130 and P(rank >= 10) 0.229 (sums of `team_rank_probabilities`).
  - Maye: published 1st of 33 qualifying QBs; median 2nd; middle 50% 1st-4th; 95% 1st-9th;
    P(top 5) 0.882, P(top 10) 0.984; P(rank 8-10) 0.047, P(rank >= 8) 0.063; missing share 0.
  - Re-read after the QB data fixes (same command): unchanged, as 2025 had no duplicated
    QB-games.

### QB data fixes (done 2026-10-04)

- Primary-QB and primary-team ties now break the same way every run (lowest `qb_id` after snaps,
  dropbacks, and attempts; most recent team on a games tie).
- Old play-by-play leaves `posteam` empty (`""`) on non-plays; the late-game flags had paired each
  team with that phantom team, doubling flag rows and QB-game rows (80 duplicated QB-games in
  1999, 87 in 2000). The loader now turns `""` into null.
- A passer tagged two ways in one game (`T.Pike (3rd QB)`) had split into two rows (9 QB-games in
  2004 and 2008-2011); the aggregates group by passer id.
- Qualifier (maintainer decision): 14 pass attempts per game the QB's own team has played,
  published as `qb_attempt_qualifier`. Only 2026 eligibility moved (6 QBs, Drew Lock among them).
- Effects, read against pre-fix copies of `data/`: 1999 lost 3 qualifiers and 2000 lost 2; 1999
  and 2000 `adj_qb_epa_per_dropback` moved by up to 0.041 and 0.064; 2004-2011 by at most 0.0125.
  Team outputs did not change. The rerun of `nfl-sos-ratings validate --data-dir data
  --start-season 1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md`
  left the team results and the adopt decision unchanged and moved the QB year-over-year Pearson to
  0.455 (adjusted EPA per dropback), 0.464 (passer rating), and 0.392 (ANY/A) over 601 pairs.

### Scratch measurements (2026-10-04, not citable)

- Range files for one season: about 20 s wall and 7 CPU minutes with default OpenBLAS threading,
  2.3 s with `OPENBLAS_NUM_THREADS=1`, identical output. NumPy here links `scipy-openblas64`
  (OpenBLAS 0.3.34, from `numpy.show_config()`); the solves are many small ones, where threading
  costs more than it saves.
- `pipeline` with `OPENBLAS_NUM_THREADS=1`: 13 min 43 s to 15 min 26 s across three runs.
- Fixed-penalty `fit_team_ratings` about 78 ms and `fit_qb_ratings` about 37 ms (mostly the Polars
  design build); head-to-head `compute_team_schedule_strength` about 2.7 s for 64 refits.

## H1. Push and open a pull request for `feat/rank-ranges`

Background: everything since `d5929ed` is local. CI (`.github/workflows/validation.yml`) runs
`scripts/gate.sh` and the `web/` checks on every push and pull request.

Approved 2026-10-04: push the branch and open a pull request into `main`. **Ask first** before
the merge itself, and ask which merge method the maintainer wants (recommended: a merge commit or
rebase-merge, which keep the logical commits; a squash would lose them).

Tasks:

- [x] Check each commit on its own (some were staged from partial files and only the final tree
  ran the full gate): in a throwaway worktree, run `scripts/gate.sh --quick` and `npm run
  typecheck` per commit, for example with `git rebase --exec` on a scratch copy of the branch. Fix
  failures with new commits; do not rewrite pushed history. Done 2026-10-04 with more than the
  minimum: a detached worktree checked out each of the 30 commits in turn and ran the full
  `scripts/gate.sh` (pytest included) plus `npm run lint`, `npm run typecheck`, and
  `npx vitest run` in `web/`; all 30 passed, so no fix commits were needed.
- [x] `git push -u origin feat/rank-ranges`, then open the pull request with `gh pr create`. The
  body groups the commits (rank ranges, QB data fixes and the qualifier, UX audit, docs and the
  validation rerun), says `data/` is gitignored and was rebuilt locally, and lists the published
  output changes (1999-2000 and 2004-2011 QB ratings, 2026 eligibility, validation QB rows).
  Opened as <https://github.com/mitch-avis/nfl-sos-ratings/pull/1>.
- [x] Wait for CI; report the result. Merge only after the maintainer says so. CI passed (`gate`
  and `web`, push and pull-request runs); the maintainer chose a merge commit, and the pull request
  merged on 2026-10-04 as `c396d83` with the branch deleted.
- [x] After the merge: `git switch main && git pull`, then `uv sync` (a checkout that rewrites
  `pyproject.toml` needs it, AGENTS.md), and branch the next workstream from `main`.

## H2. Note for nfl-predictor

Background: `../nfl-predictor` ports this repo's head-to-head-excluded opponent profiling and the
simultaneous ridge, and reads `README.md`, `AGENTS.md`, and `docs/` as reference. This repo never
edits it. Hand the maintainer this note to paste into a session there:

> nfl-sos-ratings changed on 2026-10-04 (branch `feat/rank-ranges`): (1) the QB qualifier is now 14
> pass attempts per game the QB's own team has played (`qb_attempt_qualifier`), not 14 times the
> league's most games; (2) the loader turns empty team abbreviations (`""` on old play-by-play
> non-plays) into null, which removed duplicated 1999-2000 QB game rows; (3) passer plays are
> grouped by id, not name; (4) primary-QB ties break deterministically; (5) new bootstrap rank
> ranges (`{season}_rating_ranges`, `docs/methodology.md` "Rank Ranges"). If nfl-predictor copies
> any QB aggregation or the empty-`posteam` handling, check it for the same duplicate-row bug.

Tasks:

- [x] Give the maintainer the note above when H1's pull request is open (it can cite the PR).
  Given on 2026-10-04 with the pull request link.

## S1. Single-threaded BLAS by default

Background and reasoning: see the scratch measurements above. Today the speed-up depends on
someone remembering to set `OPENBLAS_NUM_THREADS=1`.

Design: in `cli.main`, before `importlib.import_module` loads a command (NumPy is imported only
then), call `os.environ.setdefault(...)` for `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and
`MKL_NUM_THREADS` with `"1"`. `setdefault` keeps an explicit setting. The `nfl-sos` and
`nfl-sos-pipeline` shortcuts import `main` / `pipeline` directly, so either route them through the
same helper at the top of those modules or note in README that they keep the default threading.
If import order makes the environment route unreliable, `threadpoolctl.threadpool_limits(1)` is
the fallback (a new dependency; say what it is for).

Tasks:

- [x] Test first: `cli.main` sets the three variables when unset and leaves an existing value.
- [x] Implement; time `nfl-sos-ratings season --season 2025` before and after (wall and CPU
  with `time`) and record both here with the command. `cli.limit_blas_threads` runs first in
  `cli.main`; the shortcuts now point at `cli.season_shortcut` and `cli.pipeline_shortcut`, which
  go through the front door. Timings, 2026-10-04, from a scratch working directory (so `data/` was
  untouched) holding a copy of `data/2024_team_game_logs.parquet`, download cache warm, 24 cores:
  `time .venv/bin/nfl-sos-ratings season --season 2025` took 46.5 s wall, 8 min 3 s user, 1 min
  26 s sys before, and 30.0 s wall, 39 s user, 45 s sys after. All 14 output files matched
  `data/` exactly after sorting by key columns.
- [x] README (pipeline timing note) and AGENTS.md (drop the "run with `OPENBLAS_NUM_THREADS=1`"
  advice once it is the default). README gained the note; AGENTS.md had no such advice left.

Done when: a plain `nfl-sos-ratings season` builds the range files in a few seconds.

Done (2026-10-04): the range step alone took 2.7 s for 2025 under the new default (scratch
timing of `main.build_team_rating_ranges` plus `main.build_qb_rating_ranges` on the files in
`data/`, not citable).

## S2. Deterministic row order in written files

Background: comparing rebuilds showed files whose values were unchanged but whose rows came out in
a different order each run (`ratings_by_week`, `qb_ratings_by_week`, `qb_per_game_stats`,
`qb_combined`, `qb_game_logs`), from `group_by` without `maintain_order`. Harmless for values, but
it buries real changes in diffs, and any later order-dependent step can turn it into a value
change, which is how the comeback drift went unnoticed.

Design: `main._write_data_file` sorts every frame by that file's key columns before writing (team
files by `team` plus `week` / `game_id` where present; QB files by `qb_id` plus `week` /
`game_id`), unless the caller passes an explicit order (the ratings files keep their
rating-descending order for readers). Keep the rule in one place; derive keys from columns
present.

Tasks:

- [x] Test first: a shuffled frame written through `_write_data_file` reads back in key order; the
  ratings file keeps rating order. The rule lives in `main.data_file_row_order`: published order
  for `ratings`, `qb_ratings`, `rating_ranges`, and `qb_rating_ranges` (`PUBLISHED_ROW_ORDER`,
  keyed by file name rather than passed by each caller, so the rule stays in one place), otherwise
  `qb_id` or `team`, then `week` and `game_id` where present; a file with neither identity column
  fails the write. Before relying on it, a scratch script confirmed these keys are unique and
  non-null in all 392 files in `data/`.
- [x] Acceptance: two consecutive `nfl-sos-ratings season --season 1999` runs (**Ask first** with
  the other rebuilds) give frames that are `equals()`-identical for every file, with no sorting in
  the comparison. Run 2026-10-04 in scratch working directories, so `data/` was untouched (no
  rebuild needed asking): before the change, `qb_combined`, `qb_per_game_stats`,
  `qb_ratings_by_week`, and `ratings_by_week` differed in row order between two runs; after it,
  all 14 files were identical, and every file's values matched the pre-change build after sorting
  by key.
- [x] After the next rebuild (**Ask first**): `.venv/bin/pytest -m published_data` includes
  `test_every_published_file_is_stored_in_its_row_order`, which fails on the current `data/` (built
  before this change: 168 files, every season's `qb_combined`, `qb_game_logs`,
  `qb_opponent_profiles`, `qb_per_game_stats`, `qb_ratings_by_week`, and `ratings_by_week`) and must
  pass once `data/` is rebuilt. Rebuilt 2026-10-04 (maintainer approved): `cp -r data
  /tmp/data-before-wp1`, then `nfl-sos-ratings pipeline` (exit 0, 857 s) and `nfl-sos-ratings season
  --season 2026` (exit 0), run detached from `main` at `c574819`. `nfl-sos-ratings diff-data
  --before /tmp/data-before-wp1 --after data` reported "224 unchanged, 168 row order only, 0 values
  changed, 0 schema changed, 56 added, 0 removed": the six reordered kinds above in all 28 seasons,
  plus the two bins files per season. `.venv/bin/pytest -m published_data` then passed (5 tests).

## S3. Data-diff helper

Background: this session compared `data/` before and after four rebuilds with throwaway scripts.
AGENTS.md wants numbers from repo commands or checked-in helpers, so the comparison should be one.

Design: a read-only `nfl-sos-ratings diff-data --before DIR --after DIR [--season N]` command (add
it to `cli.COMMANDS`; the gate runs every command's `--help`). Per file it reports: unchanged, row
order only, values changed (changed columns with the count of changed rows and the largest absolute
difference), schema changed (added or removed columns), added, or removed. Optional `--keys` to
align rows by identity instead of sorted position. Writes to stdout with `sys.stdout.write` (see
`schedules.py`).

Tasks:

- [x] Tests first on two tiny directories covering each outcome (`tests/test_data_diff.py`).
- [x] Implement; document in README (Commands) and AGENTS.md (the commands block).
  `nfl_sos_ratings/data_diff.py`; the identity keys are shared with the season writer through the
  new `nfl_sos_ratings/row_order.py`. Beyond the design: `--tolerance` for float noise, rows
  matched by `qb_id` or `team` plus `week` and `game_id` by default (sorted position only when
  those do not identify every row, or `--keys` to override), and rows added or removed counted
  when rows are matched by identity. AGENTS.md also gained the rule to quote this command's
  summary for any rebuild.
- [ ] Use it for every later rebuild and paste its summary into the relevant plan entry. First
  real runs (2026-10-04, not rebuilds of `data/`): the S2 scratch builds,
  `diff-data --before /tmp/nfl-s2-before1/data --after /tmp/nfl-s2-after1/data`, reported "8
  unchanged, 6 row order only, 0 values changed, 0 schema changed, 0 added, 0 removed", and
  `diff-data --before data --after /tmp/nfl-s1-timing/data --season 2025` (the S1 timing build)
  reported "10 unchanged, 4 row order only" with no value changes. A full `data/` against itself
  (392 files) takes about 6 s (scratch timing).

## WP. Garbage-time win-probability filter

Background: the maintainer asked for an rbsdm-style slider to see how the 2025 results (NE 5th,
Maye 1st) depend on garbage-time plays, shown honestly rather than arguing for a ranking.

What rbsdm.com does (checked 2026-10-04, no public source found): its "Garbage-Time WP Filter"
slider defaults to 0% and runs 0-30%; a setting of X drops plays whose win probability is below X
or above 1 - X (Football Perspective, Adam Steele's QB recaps: 4% drops plays "below 4% or above
96%"). Not confirmed: whether it uses `wp` or `vegas_wp`, and whether the cut is inclusive.

Decisions (maintainer, 2026-10-04):

- WP column: `wp` (score, clock, field position). `vegas_wp` adds the pregame spread, so early
  plays of a mismatch would already look like garbage time.
- Special teams: filtered too. Onside kicks and backup coverage units are concrete garbage-time
  effects, and one play set keeps `team_rating`'s three parts consistent. Special-teams plays get
  the same bins, so the choice stays cheap to revisit.
- Range and step: 0-30% in 1% steps; default 0%. Lowered to 0-20% on 2026-10-05 (maintainer, on
  the agent's recommendation): 20% is the largest WP4 candidate, and past it the filter drops
  more than a third of the plays (2025 mean `wp_kept_play_share` 0.623 at 20%, 0.453 at 30%; WP4
  gives the command). A shared link above 20% now opens with the filter off.
- Rule: keep a play when `min(wp, 1 - wp) >= X`.
- The published default stays 0% unless WP4 says otherwise and the maintainer agrees.

Settled at WP1 (maintainer, 2026-10-04): plays with a null `wp` are kept at every threshold,
because they cannot be judged, so 0% reproduces the published ratings exactly. The "about 0.6% of
rows" figure counted non-play rows; among the plays the ratings use, a scratch count
(`load_pbp_data` with the rating filters, 1999, 2006, 2015, 2025, 2026) found one null-`wp`
scrimmage play (1999, EPA 0.0) and no special-teams ones.

Data availability (scratch check, 2026-10-04): nflverse play-by-play has `wp`, `vegas_wp`,
`def_wp`, `home_wp`, and `vegas_home_wp` in 1999, 2006, and 2025. Verify per season in loader
tests before relying on it, and guard the column (AGENTS.md: do not assume a column exists).

Architecture: per team-game, scrimmage and special-teams plays and EPA summed in 1% bins of
`min(wp, 1 - wp)` (bins 0-50, plus a null-`wp` bin); a threshold X keeps bins >= X, so any
threshold is a cumulative sum and the API refits on demand with the weighted engine (milliseconds,
`sos` included). QB: per passer-game dropbacks and play-level passing EPA in the same bins.

### WP1. Bins in the loader layer

- [x] Find where team-game scrimmage and special-teams plays and EPA are summed today
  (`team_stats_expanded.py`, around the per-team aggregation) and bin with exactly the same play
  filters, so the sum over all bins equals `offensive_snaps`, `offensive_epa`, `st_plays`, and
  `st_epa` per team-game. That equality is the acceptance test. `nfl_sos_ratings/wp_bins.py`
  reuses `scrimmage_snap_expr` and the newly shared `special_teams_play_expr` in two separate
  passes (a fake punt counts in both totals); bin = `floor(round(100 * min(wp, 1 - wp), 9))`, the
  rounding so an exact 0.29 lands in bin 29. Checked three ways: a synthetic test against
  `compute_expanded_team_game_stats`, `compute_team_snap_counts_from_pbp`, and
  `compute_qb_game_volumes_from_pbp`; a scratch run on real 1999, 2012, 2025, and 2026
  play-by-play against the game logs in `data/` (every count exact, EPA within 1e-14, no unmatched
  rows); and `published_data` tests for after the rebuild.
- [x] Outputs: `{season}_team_wp_bins` (`game_id`, `team`, `opponent_team`, unit, bin, plays,
  EPA) and `{season}_qb_wp_bins` (`game_id`, `qb_id`, bin, dropbacks, passing EPA). Registry
  entries first, then `catalog`. Group passers by id (AGENTS.md domain rule). Both files also
  carry `week`; columns `wp_unit`, `wp_bin`, `wp_bin_plays`, `wp_bin_epa`, `qb_wp_bin_dropbacks`,
  `qb_wp_bin_epa`. QB rows without a passer id are left out (no published QB row can match them).
  `row_order` sorts bin rows by identity, week, game, then `wp_unit` and `wp_bin` (null last), and
  `diff-data` now matches null keys. Written by `run_season` through `data_loader.load_wp_bins`
  (one play-by-play load for both); about 143 KB and 84 KB for 2025.
- [x] Loader tests: `wp` present and null counts on scrimmage plays per season; a typed empty frame
  when a source lacks `wp`. The per-season check is a `published_data` test (null-bin share of
  scrimmage plays at most 0.1% in every season, plus a bins file for every season with game logs),
  so it runs after the rebuild; the typed empty frame is a unit test.
- [x] After the rebuild (**Ask first**): `.venv/bin/pytest -m published_data` must pass, including
  the three bins tests. A scratch build of 1999 and 2025 in `/tmp` passed all five
  `published_data` tests on 2026-10-04, and `diff-data --before data --after /tmp/nfl-wp1/data
  --season 2025` reported "8 unchanged, 6 row order only, 0 values changed, 0 schema changed, 2
  added, 0 removed". Done with the full rebuild recorded under S2. Plays without a bin, all seasons,
  from `.venv/bin/python -c "import polars as pl; print(pl.read_parquet('data/*_team_wp_bins.parquet',
  include_file_paths='p').filter(pl.col('wp_bin').is_null()).group_by(pl.col('p').str.extract(r'(\d{4})_'),
  'wp_unit').agg(pl.col('wp_bin_plays').sum()))"`: 17 scrimmage plays (1999: 1, 2000: 9, 2001: 6,
  2007: 1), none on special teams.

### WP2. Refits per threshold and the API

- [x] A resampler-like class built from the bins: for threshold X, cumulative sums give each
  design row's plays (weight) and EPA (response); reuse one `build_unit_design` structure and swap
  weights and responses. Penalties: the season fit's (previous-season penalties for teams, the
  season's own for QBs), as rating histories do. `nfl_sos_ratings/wp_filter.py`
  (`TeamWpFilter`, `QbWpFilter`); `dataclasses.replace` swaps weights and responses into the
  frozen `UnitDesign`, so `ridge.py` is unchanged. Decision made here (agent, reported to the
  maintainer): filtered ratings keep the season fit's per-game scales, as the head-to-head refits
  already do, so a filtered rating reads on the published scale; filtered plays per game would
  shrink every rating as the threshold rises.
- [x] `sos` and `qb_faced_pass_defense` per threshold through the same head-to-head refits.
- [x] QB basis: the published `qb_epa_per_dropback` uses official weekly passing EPA, while the
  bins use play-level EPA. Recommended: the exploration view uses play-level EPA at every
  threshold, shows the published value beside it, and states the measured gap at 0% (compute it
  and record it here with the command). Done as recommended; `_change` columns compare with the
  0% play-level value. 2025 gap, from `curl -s
  'http://127.0.0.1:8090/api/seasons/2025/qbs/wp-ratings?threshold=0'` (server: `nfl-sos-ratings
  web --port 8090`) comparing `adj_qb_epa_per_dropback` with `filtered_adj_qb_epa_per_dropback`
  over the 33 qualifying QBs: mean absolute 0.0005, largest 0.0017 EPA per dropback; 5 QBs rank
  differently at 0%, by at most 2 places.
- [x] API: `GET /api/seasons/{season}/{teams|qbs}/wp-ratings?threshold=X` (0-30), cached per
  season and threshold in process. Performance target: under 200 ms per uncached threshold for
  teams; measure and record. 2025, `curl -w '%{time_total}'` against the 8090 server on
  2026-10-04: teams 0.098 s median and 0.104 s largest over thresholds 1-30 (uncached), QBs 0.104
  s and 0.112 s; a season's first request (model build) 0.47 s for teams and 0.23 s for QBs;
  cached repeats 3-4 ms. The cache key includes each input file's modification time and size, so a
  rebuilt file brings a new model. Payload columns use a new `filtered_` prefix rule and `_change`
  suffix rule plus `wp_kept_play_share` and `wp_kept_dropback_share` in the registry.
- [x] Tests: X = 0 reproduces the published team ratings to float tolerance; a synthetic league
  where dropping a bin changes a rating by a known amount. `tests/test_wp_filter.py` checks every
  threshold result against an independent computation (a fresh fit on the kept plays rescaled to
  the season scale, `compute_team_schedule_strength`, `compute_qb_faced_pass_defense`); the
  shared league is `tests/wp_league.py`. On real 2025 data the 0% team ratings matched the
  published file to 4e-15 and `sos` to 2e-15 (same `curl` on `teams/wp-ratings?threshold=0`).

### WP3. Slider in the web app

- [x] shadcn Slider (`npx shadcn@latest add slider`; the CLI misreads this repo's `utils` alias
  and installs an npm package named `cn`, so `npm uninstall cn` and point the import at
  `@/utils/cn` afterwards). 0-30 in 1% steps, value shown beside it, debounced about 250 ms,
  threshold in the URL (`?wp=`), keyboard and touch friendly. Done as described; the slider uses
  the `radix-ui` package already installed, so `package.json` and the lockfile are unchanged. One
  minimal edit to the generated `slider.tsx`: `aria-label` is forwarded to the thumb (Radix puts
  `role="slider"` there). The root gets `py-3` for a larger touch area.
- [x] Non-zero thresholds show an "Unvalidated exploration view" label and the published values
  next to the filtered ones (with the change), on the team and QB index and detail pages.
  `web/src/components/entity/WpFilterPanel.tsx` (one card per page; a sortable table on index
  pages, stat tiles on detail pages), logic in `web/src/domain/wpFilter.ts` with tests, URL state
  in `web/src/app/useWpThreshold.ts`. Links from the filtered table carry `?wp=`.
- [x] Rank ranges stay at 0% (out of scope at other thresholds; say so in the UI).
- [x] Use the dataviz, frontend-design, and frontend-react skills; phone layout and `Hint` /
  `SortableHeader` patterns as established by the UX audit. The dataviz skill was not used: the
  view has no chart. `react-doctor` (run once with `npx`, not added to the repo) flagged only the
  existing complexity of `EntityDetailPage`.
- [x] Phone layout: the maintainer checked every page on a phone after the merge (pull request
  #5, 2026-10-04) and found it fine.
- [x] QB index freeze, most likely a browser extension: the agent's Chrome froze on
  `/qbs?season=2025&wp=10` whenever the filtered table showed each QB's team, so #5 shipped
  without it. That Chrome profile had the FantasyPros extension enabled on all sites; with it and
  a few others disabled (2026-10-04), the same page with the team shown stayed responsive in the
  agent's Chrome: about 20 s idle, scrolling, and the slider moved to 11% and 20%, every row
  showing its team. One screenshot timed out once, right after scripted key presses on the slider,
  while the page's scripts kept answering at once; it did not recur. The filtered QB table now
  shows each QB's team in its own column, as the main QB table does (pull request after #5).

### WP4. Pre-registered walk-forward test

Pre-registered on 2026-10-05, committed before any run of the check (AGENTS.md). The maintainer
approved the design choices below on 2026-10-04 and the slider change on 2026-10-05.

- **Hypothesis (falsifiable):** rating teams on the plays a garbage-time filter keeps predicts
  game margins out of sample better than rating them on every play. It is refuted for a threshold
  if that threshold's paired interval does not lie entirely below zero.
- **Candidates, fixed in advance:** thresholds of 5%, 10%, and 20%, each against 0%. A threshold
  of X keeps a play when `min(wp, 1 - wp) >= X` (the `team_wp_bins` at or above X), plus the plays
  without a win probability, as the exploration view does.
- **Estimator:** each candidate is the published team fit run unchanged on the kept plays. Kept
  plays and kept EPA from `{season}_team_wp_bins` replace `offensive_snaps`, `offensive_epa`,
  `st_plays`, and `st_epa` in the team game logs, and the rows go through
  `fit_team_ratings_with_previous_penalties` with the previous season's full-season fit on that
  season's kept plays (its penalties cross-validated at the same threshold; 1999 cross-validates
  its own, as the published fit does). Ratings use the kept plays' own per-game scale; each
  candidate gets its own margin model, so a scale difference cannot help or hurt it. Reason for
  re-tuning the penalties: the kept share falls fast (mean `wp_kept_play_share` over the 32 teams
  in 2025: 0.846 at 5%, 0.772 at 10%, 0.623 at 20%, 0.453 at 30%, from `curl -s
  'http://127.0.0.1:8090/api/seasons/2025/teams/wp-ratings?threshold=X'` against `nfl-sos-ratings
  web --port 8090`, measured before the maximum dropped to 20%), so the 0% penalties would shrink
  filtered ratings harder for a reason unrelated to garbage time.
- **Allowed information set:** to predict the games of week w in season s, a candidate sees only
  season s's team-game rows and bins from weeks before w, and season s - 1's full-season rows and
  bins (for the penalties); its margin model sees only its own earlier predictions. This is the
  walk-forward harness in `nfl_sos_ratings/validation/walk_forward.py`
  (`build_snapshot_feature_rows`, `evaluate_feature_rows`). Known leak, judged negligible:
  nflverse's `wp` model was fit on many seasons, later ones included, but it only decides which
  plays enter a rating, never the prediction.
- **Metric and window:** mean absolute error of the predicted home margin, prediction weeks 5 and
  later, seasons 1999-2025, over the games every candidate predicts (every home game in the
  window). Margin model: `home_margin = k * rating_gap + home_edge`, fit by least squares on the
  candidate's earlier predictions, as in `validate`.
- **Integrity check, before reading any result:** at 0% the kept columns must equal the game-log
  columns on every row (to 1e-9), and the 0% candidate's overall MAE must reproduce `validate`'s
  `team_rating` MAE (10.601, `.agents/current-status.md`) to three decimals. If either fails, stop
  and investigate; read nothing else.
- **Inference:** for each candidate X, a paired bootstrap of the per-game difference
  |error at X| - |error at 0%|: 10,000 resamples of the games with replacement, seed 0 (the same
  draws for all three comparisons), percentile intervals at 98.33% (quantiles 1/120 and 119/120),
  which is 95% Bonferroni-adjusted for three comparisons. Games are resampled independently, as
  `validate` does, though games in one week share a rating snapshot.
- **Decision rule:** a threshold qualifies for the default only if its overall interval lies
  entirely below zero; among qualifying thresholds, the one with the lowest overall MAE is the
  recommendation. If none qualifies, the result is a tie and the recommendation is 0% (no filter,
  the simpler option). Every interval that excludes zero is reported, in either direction. The
  decision goes to the maintainer either way. A win applies to the team ratings only: QB ratings
  stay at 0% unless a separate test supports a change, and adopting a threshold for teams is a
  published-rating change (**Ask first**; registry, `README.md`, `docs/methodology.md`, the
  validation report, and the exploration view's penalties change with it).
- **Descriptive extras (never decision inputs):** each candidate's MAE and RMSE overall and in the
  early (weeks 5-7) and late (week 8 on) splits, with 98.33% intervals; mean kept play share; team
  year-over-year Pearson of full-season `team_rating` (consecutive seasons, by team); QB
  year-over-year Pearson of full-season adjusted EPA per dropback over passers qualifying
  (published `qb_is_eligible`) in both seasons; mean per-season Pearson with ESPN QBR (2006-2025,
  joined as `validate` joins it); NE's 2025 `team_rating` and rank and Drake Maye's 2025 adjusted
  EPA per dropback and rank among qualifying passers, whichever way they move. The QB fit is also
  the published one run unchanged on kept dropbacks and their play-level EPA (`qb_wp_bins`), its
  penalty cross-validated per season at each threshold, so its 0% baseline is the play-level
  value, not the published rating (official weekly passing EPA).
- **Command:** a new read-only `nfl-sos-ratings check-wp-filter --data-dir data --start-season
  1999 --end-season 2025 --start-week 5`, modeled on `check-in-season-penalty`; it reads `data/`,
  downloads ESPN QBR for the extra, and writes only to stdout. `validate` and its report stay as
  they are unless the maintainer adopts a change. The slider's and API's 20% maximum does not
  affect the test.

Tasks:

- [x] Finish the protocol above and commit it before running (2026-10-05).
- [x] Build `check-wp-filter` test-first (kept-play game logs and QB rows at a threshold, the
  98.33% bootstrap, the report). Check the plumbing on real data without computing any non-zero
  threshold: the 0% kept columns equal the game logs in every season, and the 0% candidate's
  walk-forward rows equal `validate`'s `team_rating` rows for one season. Done 2026-10-05:
  `wp_filter.team_game_logs_at_threshold` and `qb_games_at_threshold` build the kept rows,
  `walk_forward.compute_pairwise_mae_bootstrap` gained a `confidence` keyword (default 0.95, so
  `validate` is unchanged), and `nfl_sos_ratings/validation/wp_filter_check.py` is the command.
  Plumbing, from `check_zero_kept_columns` and `check_zero_threshold_rows` called on `data/` in a
  scratch script: the 0% kept columns matched the game logs in all 28 seasons (1999-2026, largest
  gap 7.1e-15), and the 2025 0% rows matched the published rows over 272 games (largest gap
  7.1e-15). Scratch timing (not citable): about 1.3 s per season and threshold for the
  walk-forward rows, so the full run should take a few minutes.
- [x] **Ask first**, then run the full check; write the results here, with the command, and in the
  validation report if the maintainer adopts a change. Run on 2026-10-05 (maintainer approved) at
  `3a17c5a`: `nfl-sos-ratings check-wp-filter --data-dir data --start-season 1999 --end-season
  2025 --start-week 5`, 5 min 44 s wall. Every number below is from that run's output.
  - Integrity check passed: at 0% the kept plays equal the game logs in every season, and the 0%
    rows match the published team rating's (largest gap 3.3e-13); the 0% overall MAE is 10.601,
    as in `validate`.
  - MAE over 5,297 games: 0% 10.601, 5% 10.623, 10% 10.663, 20% 10.733 (RMSE 13.593, 13.633,
    13.690, 13.794).
  - Paired MAE difference from 0%, 98.33% intervals: 5% +0.023 (-0.033 to +0.076), 10% +0.062
    (-0.008 to +0.136), 20% +0.133 (+0.046 to +0.222).
  - Decision rule as written: no threshold qualified, so the recommendation is 0% (no filter). The
    hypothesis is refuted at all three thresholds.
  - Intervals excluding zero, either direction: 20% overall (worse), and in the descriptive splits
    20% late, weeks 8 on (+0.134, +0.034 to +0.237, worse). None in the early split (weeks 5-7).
  - Descriptive extras at 0%, 5%, 10%, 20%: kept play share 1.000, 0.836, 0.757, 0.601; team
    year-over-year Pearson 0.434, 0.430, 0.401, 0.361 (829 pairs); QB year-over-year Pearson
    0.455, 0.419, 0.397, 0.330 (601 pairs); mean per-season Pearson with ESPN QBR 0.892, 0.880,
    0.869, 0.815 (20 seasons); NE 2025 `team_rating` 5.92 (5th), 5.31 (3rd), 4.87 (4th), 3.95
    (5th); Drake Maye 2025 adjusted EPA per dropback 0.210 (1st), 0.219 (3rd), 0.222 (1st), 0.204
    (3rd).
- [x] Maintainer decision on the published default (the rule recommends 0%, no filter); then
  record the outcome in `.agents/ratings-simplification-plan.md` with the other test results.
  Decided 2026-10-05: the published ratings keep every play. Recorded in the decision record;
  `docs/methodology.md` ("Garbage-Time Filter") and the app's exploration note state the result.
  The validation report is unchanged, as no change was adopted.

## R. Rank-range extensions

All three reuse the bootstrap engine and carry the sampling-noise caveat.

### R1. Head-to-head chances

Background: two overlapping rank ranges do not answer "is NE better than BUF?" because the two
teams move together in each resample (same games). The share of resamples in which A is rated
above B, and the interval of A's rating minus B's, answer it directly.

Design (recommended): compute pairwise summaries at build time and do not store raw draws.
`{season}_rating_pairs` (`team`, `opponent_team`, P(rated above), rating-difference quantiles at
2.5/50/97.5) and `{season}_qb_rating_pairs` (eligible QBs; pairs counted over resamples where both
appear, with that share). 32 teams give 992 ordered pairs per season, small. Alternative: store raw
draws (about 32,000 rows per season for teams) and compute in the API; more flexible (R2 and R3
could read them) but larger and slower to serve.

Tasks:

- [x] Registry entries; `summarize_rank_pairs` with tests (a synthetic league where one team is
  clearly better; symmetry P(A over B) + P(B over A) = 1 for teams). Done 2026-10-05 as designed
  (build-time summaries, no raw draws). Columns: `team`, `other_team`,
  `team_rated_above_probability` (a tie counts half), `team_rating_gap_q025`/`_q500`/`_q975`, and
  `team_pair_share`, and the QB counterparts (`qb_id`, `other_qb_id`, `qb_...`). "Other" rather
  than "opponent", because the two need not have played. The entries live in the new
  `nfl_sos_ratings/metrics/pair_metrics.py`: `team_metrics.py` is past 2,700 lines, and AGENTS.md
  asks for a split proposal before growing such a module (proposal for the maintainer: split it by
  category, like the sections it already has; approved and done 2026-10-08, session plan P4).
- [x] `run_season` writes the files (**Ask first** for the rebuild; approved for after R2); API
  `GET /api/seasons/{season}/{teams|qbs}/{id}/rating-pairs`. One bootstrap per season now feeds
  both the ranges and the pairs (`main.build_team_rank_summaries`, `build_qb_rank_summaries`);
  `row_order` sorts pair files by unit, then compared unit. The new
  `test_published_pair_files_cover_every_season_and_add_up` (`published_data`) must pass after the
  rebuild.
- [x] Detail page: "Compare with" picker with a plain sentence ("NE rated above BUF in 38% of
  resampled seasons; difference -1.2 points, 95%: -5.0 to +2.8"); the comparison panel shows the
  same when exactly two rows are compared. `HeadToHeadCard` (starts on the neighbor in the
  published ranking) and `HeadToHeadSentence`; logic in `web/src/domain/ratingPairs.ts`. Hidden
  for seasons built without pair files. Live check pending the rebuild.

### R2. Rank ranges for the unit ratings

Background: `TeamRatingResampler.ratings` already returns `offense_rating`, `defense_rating`, and
`special_teams_rating` for every resample; only `team_rating` is summarized.

Tasks:

- [x] Registry: `offense_rank`, `defense_rank`, `special_teams_rank` (the quantile suffix rules
  already exist); summarize all four ratings in `build_team_rating_ranges`. Done 2026-10-05:
  `bootstrap_team_ratings` now keeps the unit ratings (same draws, so the team ranges and pairs are
  unchanged), `rating_ranges.summarize_unit_rank_ranges` adds each unit's published rank and rating
  and rank percentiles to the team rows (rank chances stay with `team_rating`), and the entries
  live in the new `nfl_sos_ratings/metrics/unit_rank_metrics.py` for the same reason as the pair
  entries.
- [x] Note in `docs/methodology.md` that percentiles do not add up (the median of a sum is not the
  sum of the medians), unlike the published ratings.
- [x] Detail page: unit rows with mini intervals; the API payload gains column groups per unit.
  `UnitRankRangeTable` in the detail page's rank-range card; payload groups `offense_range`,
  `defense_range`, `special_teams_range` (only when present). Live check pending the rebuild.

### Rebuild after R1 and R2 (2026-10-05)

Approved for the overnight run. From `main` at `7c62765`: `cp -r data /tmp/data-before-r1r2`, then
`nfl-sos-ratings pipeline` (exit 0, 20 min 17 s wall) and `nfl-sos-ratings season --season 2026`
(exit 0, 22 s). `nfl-sos-ratings diff-data --before /tmp/data-before-r1r2 --after data` reported
"405 unchanged, 0 row order only, 15 values changed, 28 schema changed, 56 added, 0 removed": the
56 added are the two pair files per season; the 28 schema changes are every season's
`rating_ranges` gaining the unit columns, with no existing values changed in 1999-2025; the 15
value changes are all 2026, which gained the week-4 games played since the last build (team game
logs 98 to 126 rows). `.venv/bin/pytest -m published_data` passed (6 tests, including the new pair
check). Live check on the 8090 server: NE's detail page shows offense 2nd, defense 13th, special
teams 17th by unit, and "NE rated above JAX in 40% of resampled seasons; difference -1.0 points,
95%: -8.3 to +5.9."; the comparison panel for NE and BUF gives 72%. The pipeline ran slower than
the last rebuild (857 s); a scratch timing of 2025 put the new steps at about 0.7 s per season
(unit ranges 0.32 s, team pairs 0.18 s, QB pairs 0.16 s), so they do not explain it; download time
or machine load is the likely cause, not measured.

### R3. Rank ranges by week

Background: a season in progress shows very wide ranges that narrow as games accumulate; a weekly
view makes that visible and complements the rating history.

Design: for each week, the games through that week, bootstrapped with the season fit's penalties
(as rating histories do), giving `{season}_rating_ranges_by_week` (week, team, rank percentiles,
P(top 5), P(top 10)) and the QB counterpart. Cost scales with weeks: measure one season first;
options are all seasons with fewer resamples (for example 200) or only seasons in progress.

Tasks:

- [x] Measure the cost for one season and record it; choose the scope with the maintainer if it
  adds more than a couple of minutes to `pipeline`. Scratch timing (2026-10-05, single-threaded
  BLAS, 1000 resamples, week fits with the season fit's penalties) on 2025: 18 weeks took 37.4 s
  for teams and 24.9 s for QBs, 62.3 s per full season, about 28 minutes over 27 seasons. Over
  the limit, so the maintainer's pre-approved scope applies: weekly ranges only for the season in
  progress (`config.SEASON`, 2026) at the full 1000 resamples, at most about a minute per rebuild
  by the season's end.
- [x] Detail page: median rank by week with 50% and 95% bands (rank 1 at the top), beside "Rating
  by week". Done 2026-10-05: `main.build_team_rank_ranges_by_week` and
  `build_qb_rank_ranges_by_week`, written only when the season is `config.SEASON`; API
  `GET /api/seasons/{season}/{teams|qbs}/{id}/rank-history`; `RankHistoryCard` below "Rating by
  week". A scratch build of 2026 (`season --season 2026` in an empty working directory holding
  only `data/2025_team_game_logs.parquet`) took 35 s against 22 s without weekly ranges.
- [x] Rebuild after R3 (approved): from `main` at `a81ef0e`, `cp -r data /tmp/data-before-r3`, then
  `nfl-sos-ratings season --season 2026` (exit 0, 34.5 s). `nfl-sos-ratings diff-data --before
  /tmp/data-before-r3 --after data --season 2026` reported "18 unchanged, 0 row order only, 0
  values changed, 0 schema changed, 2 added, 0 removed" (the two weekly files);
  `.venv/bin/pytest -m published_data --no-cov` passed (6 tests).
- [x] Decision for the maintainer (found 2026-10-05): with one or two games per team, a game
  bootstrap can only repeat or drop a team's games, so the first weeks' bands understate the
  uncertainty; NE 2026's 95% band was 11th-21st after week 1 but 4th-31st after week 3. The card
  and `docs/methodology.md` now say so. Recommended: start the chart at the first week in which
  every team has played three games (a small frontend filter; the files keep every week), since
  a caveat alone still draws a misleadingly tight band. Alternative: keep every week with the
  caveat, as shipped. Approved 2026-10-08 (the recommendation) and built in session plan P9.

## A1. Weekly refresh automation for 2026

Background: during the season the current season is rebuilt with `nfl-sos-ratings season`
(`config.SEASON` is 2026); the season-in-progress notice and the per-team qualifier update from the
data. The app reads Parquet files per request, so a running server shows new data on reload.
nflverse usually publishes a week's play-by-play within a day of the games.

Design: `scripts/refresh-season.sh` runs `nfl-sos-ratings season`, then `.venv/bin/pytest -m
published_data`, logs to `logs/refresh-YYYYMMDD.log` (gitignored), and exits non-zero on failure;
with S3 it also prints the diff against the previous build. Scheduling options (recommended first):

1. Windows Task Scheduler on Tuesday mornings, running the command below: runs whenever Windows is
   on, even if no WSL shell is open.

   ```text
   wsl.exe -d <distro> -- /home/mitch/workspace/nfl-sos-ratings/scripts/refresh-season.sh
   ```

2. A systemd user timer inside WSL: needs systemd enabled in WSL and the WSL VM running at the
   scheduled time.
3. Manual runs (today's practice).

Tasks:

- [x] Script with shellcheck-clean Bash (shell-scripting skill) and a dry-run flag; tests where
  practical (Bats is optional). Done 2026-10-05: `scripts/refresh-season.sh [--season N]
  [--dry-run]` (shellcheck and shfmt clean); `tests/test_refresh_season_script.py` runs a copy of
  it in a temporary tree with stand-in commands (dry run, clean run, failing step, bad arguments).
  The `published_data` step used `--no-cov` then, since the coverage floor failed a run of only
  those tests; S5 (2026-10-08) lifted the floor for that selection and dropped the flag. Not run
  for real tonight (that is a rebuild; `data/` was current).
- [x] **Ask first** before installing any scheduled task (machine configuration outside the
  repo); provide the exact install command or task XML for the maintainer. Decided 2026-10-08: no
  scheduled task; the app's refresh button (session plan, P12) starts the script instead. Not
  installed. The
  command for this machine (WSL distribution `Ubuntu`), from PowerShell or `cmd`: `schtasks /Create
  /TN "nfl-sos-ratings weekly refresh" /SC WEEKLY /D TUE /ST 09:00 /TR "wsl.exe -d Ubuntu --
  /home/mitch/workspace/nfl-sos-ratings/scripts/refresh-season.sh"`; `schtasks /Delete /TN
  "nfl-sos-ratings weekly refresh"` removes it.
- [x] Season rollover note in `current-status.md`: after the 2026 season, set `END_YEAR` to 2026
  and `SEASON` to 2027 in `config.py`, rebuild (ask first), and regenerate the validation report.

## F. Frontend follow-ups

Carried from the frontend plan's queue (F1-F4) plus three ideas (F5-F7). Use the frontend-design,
frontend-react, and dataviz skills; keep the UX audit's patterns: `Hint` for every tooltip
(hover with a mouse, tap on touch screens), `SortableHeader` for sortable columns, only the name
pinned on phones, fixed decimals per column.

- [ ] F1 Weekly-log detail pages: strengthen them now that the weekly trend chart has landed; keep
  them table-first and check the chart against live use before adding chart types.
- [ ] F2 Grouped opponent ledgers: refine after live use, especially if one team or QB surface
  wants a different primary metric or a tighter default column mix.
- [x] F3 Opponent-strength context in the weekly views: show each opponent's season-long rating and
  rank range, labeled as season-long, never as a single-game rating. Done 2026-10-05: the
  game-by-game table already carried each opponent's season-long ratings; each opponent cell now
  adds its middle-50% rank range and a mini interval (`rankRanges.opponentRankRanges`), the team's
  own on team pages and its defense's (R2 unit ranges) on QB pages, and the card's description
  says they are season-long. The unique-opponents table is unchanged.
- [x] F4 Comparison: a pinned side-by-side layout instead of the compact strip, with rank-range
  mini intervals and R1's head-to-head sentence when two rows are compared. Done 2026-10-05:
  `ComparisonPanel` is now one column per pick (name, published rank, middle-50% range and mini
  interval, remove button) and one row per metric, heat-mapped across the picks, with the metric
  column and header pinned in a scrolling box; the head-to-head sentence (R1) sits above it for
  two picks.
- [x] F5 Idea: CSV export of the current table view, built from the loaded payload (current view,
  sort, and filters), so an analyst can take the numbers elsewhere. Done 2026-10-05: a `CSV`
  button in the index table's header (`CsvExportButton`, `domain/csv.ts`) writes the view's
  columns in order and the rows after the search in the current sort, raw values, column keys as
  the header, RFC 4180 quoting; the file is `nfl-sos-ratings-{teams|qbs}-{season}.csv`.
- [x] F6 Idea: rank-range mini intervals in the comparison panel even before F4. Landed with F4.
- [x] F7 Team palettes (maintainer idea, 2026-10-05): every team now has a palette in the
  `Palette` menu (default first, then teams grouped by division, with brand-color swatches),
  remembered as before (a stored `broncos` reads as `DEN`). Built as stated: colors from
  nflverse's teams table (`team_color` through `team_color4`; the Broncos' `#002244` and `#FB4F14`
  match the hand-tuned palette, which confirmed the source); `nfl-sos-ratings team-palettes`
  (`nfl_sos_ratings/team_palettes.py`) writes `web/src/domain/teamPaletteData.json`; the app sets
  the palette's CSS variables on `<html>` per mode, and the heat scale reads from the same file.
  Rules (in the module docstring): the accent is the more vivid main color that needs at most
  0.15 lightness change, moved until text on it and links in it reach 4.5:1 on background, card,
  and muted surfaces, never darker than 0.4 lightness in light mode so links stay apart from the
  body text; chart colors reach 3:1 on the card; heat tints of the accent and second hues, with
  text at 4.5:1 on every step and the two ends at least 8 OKLab units apart in dark mode and 4.9
  in light mode (the hand-tuned Broncos light scale is 4.98). Python tests check every committed
  palette in both modes. Findings for the maintainer:
  - The hand-tuned Broncos light palette, kept exactly, falls below 4.5:1: white text on its orange
    3.28:1 and orange links on the background 3.24:1 (3:1 is the large-text level; dark mode is
    7.2:1). Recommended: darken the light-mode orange to the generator's fit of `#FB4F14`,
    `oklch(0.554 0.188 36.5)` (5.0:1, the hue kept); the test exempts DEN light until decided.
  - Teams whose second color is black, white, silver, or too close in hue (and whose extra colors
    do not help) keep the default green-to-red heat scale: NYJ, CIN, CLE, IND, LV, DAL, PHI, DET,
    ATL, CAR, NO, TB in both modes, and WAS in dark mode only.
  - A team's good end uses its accent hue, so for some teams red means better (KC, NE); the
    hand-tuned Broncos scale already worked that way (orange good, navy bad).
  - Settled 2026-10-08: the maintainer approved replacing the hand-tuned Broncos palette with the
    generated one (nflverse's `#002244` and `#FB4F14`; light accent `oklch(0.554 0.188 36.5)`), so
    the DEN light exemption in the palette test is gone.

### F7 follow-up: palette depth (2026-10-08)

The maintainer asked for more of every team's colors on all pages without overdoing it, clean and
legible for every palette in both modes, and for team heat scales in place of the green-to-red
fallback (suggesting the darkest official color in light mode and the lightest in dark mode).
Built on branch `feat/team-color-depth` (`nfl_sos_ratings/team_palettes.py`, rules in its module
docstring):

- Surfaces: a first version tinted page surfaces with each team's base color (dark mode up to 0.022
  chroma); after seeing it the maintainer chose neutral page surfaces in every palette (2026-10-08),
  so moving between teams changes only accents, never backgrounds or panels. Removed before merge.
- Hover and selected backgrounds (`accent`, `sidebar_accent`) and tooltips (the hint card,
  `hint` and `hint_border`) are tints of the accent hue, the card bordered in the accent; the app
  logo is drawn in the team's colors (`mark` in the palette file); a 3 px two-color stripe marks the
  top of the header under a team palette.
- Team and QB pages show that team's palette (a QB's team that season), the chosen palette applying
  to the Teams, Quarterbacks, and Glossary pages; the palette menu's "Use each team's colors on its
  page" switch (on by default, stored) turns it off (maintainer request, 2026-10-08). The menu opens
  with the chosen palette focused and scrolled into view.
- Heat scales for all 32 teams in both modes. The literal suggestion was tried against every team
  (scratch renders, 2026-10-08): with black as the darkest color, it makes the team hue the bad
  end in light mode for eight teams (NYJ, CIN, ATL, CAR, TB, LV, PHI, DET), inverting the brand.
  Adopted instead: the accent hue always marks better; when the second color is black, white,
  silver, or its tint looks like the accent's, the bad end is a muted gray of it (a muted gray of
  another listed color when that separates the ends better); the Raiders, with no hue at all, get
  the suggested lightness scale (darker better in light mode, lighter in dark mode).
- Team chips (independent of the palette): a two-color dot beside every team abbreviation in the
  index tables, game log, opponent table, comparison, filtered table, and detail-page title.
- Checks: `is_readable` now covers body text on every tinted surface (hover, selected, and hint
  card), secondary text and links on the hint card, the accent surfaces' own text, the hint
  border, and heat-scale ends at least 3 OKLab units from the card; every committed palette passes
  in both modes, and every team has a heat scale. The heat scale's better end uses the same team
  color in both modes (the light-mode accent's), so the Packers' better cells are green by day and
  night (an independent review found 9 teams swapping between modes before this).
  Visual check: all 32 palettes, both modes, index and detail pages, screenshotted with headless
  Chrome against a scratch build (`vite build --outDir /tmp/...`, so `web/dist` was untouched).

## S4. Project logger

Background: package modules print progress, which needs per-file `T201` ignores in
`pyproject.toml`; nfl-predictor has a small colored logger. Approved 2026-10-04, including
removing those ignores once nothing prints (a ruff configuration change the maintainer has
agreed to).

Tasks:

- [x] Use the observability skill; stdlib `logging` with one small formatter (color only on a
  TTY), INFO by default, `--verbose` for DEBUG on the front door. Done 2026-10-05:
  `nfl_sos_ratings/logger.py` (`configure_logging`, one stderr handler on the `nfl_sos_ratings`
  logger, INFO plain, other levels prefixed, ANSI color only on a terminal); `nfl-sos-ratings
  -v/--verbose`; the modules' `__main__` guards configure it too.
- [x] Replace progress `print`s; keep data written to stdout (for example `schedules`) on
  `sys.stdout.write`. All 32 in `main`, `pipeline`, and `walk_forward`; "Saved ... to ..." per file
  is DEBUG; the season's ratings table stays on stdout; a failed pipeline season logs its
  traceback (`logger.exception`).
- [x] Remove the `T201` per-file ignores; the gate must pass without new suppressions. Removed for
  the three modules; the one left is the test fixture builders'. The pipeline's broad `except` no
  longer needs its `BLE001` suppression, since it now logs the exception.

## M1. Retired stats, one at a time

Background: 122 registry entries (57 team, 65 QB: special-teams detail, NGS/PFR/QBR joins, and
QB splits) were catalogued but never computed and were retired on 2026-10-04; the list is in
`.agents/ratings-simplification-plan.md` ("Retired metric backlog"), and commit `c178744` still
has them. The maintainer agreed any of them can come back individually.

Process per stat: the maintainer picks it; registry entry first; compute it in the loader or
aggregation layer with a test against an official value or an independent aggregation (AGENTS.md:
verify every self-computed metric); `nfl-sos-ratings catalog`; the web app picks it up by
category. Ask the maintainer which group to start with (special-teams detail and QBR/NGS joins
are the largest).

## S5. Test and pipeline speed (proposed 2026-10-08)

Background: the maintainer found `pytest` slow (about a minute on their run). Measured 2026-10-08
(scratch timings, not citable): the suite took 133 s wall with 6 min 15 s of system time, the same
under Polars 1.44.2 and 2.0.0, and 17 s with `POLARS_MAX_THREADS=1`. Polars starts a 24-thread
pool for each of thousands of tiny frames, and the threads spend their time waking each other. The
same holds for real builds: `nfl-sos-ratings season --season 2025` took 32.7 s wall (105 s system)
with default threads and 4.8 s with one thread; `diff-data --tolerance 1e-9` between the two
builds reported "19 unchanged" (without a tolerance, four descriptive opponent-profile files differ
by at most 8.5e-14, float summation order).

Proposal (one small pull request after the current one merges):

- [x] `cli.limit_blas_threads` also defaults `POLARS_MAX_THREADS` to 1 (before any command imports
  Polars), with a test like the BLAS ones; `tests/conftest.py` sets the same default before the
  test modules import Polars. Timings (2026-10-08, 24 cores): `time .venv/bin/pytest` 18.7 s wall
  with the new default, against 133 s before (scratch run with all cores, not citable); `time
  .venv/bin/nfl-sos-ratings season --season 2025` 5.3 s with the default and 32.2 s with
  `POLARS_MAX_THREADS=24`, outputs equal (`diff-data --tolerance 1e-9`, README). A full
  `pipeline` was not timed (it rewrites `data/`).
- [x] `pytest -m published_data` (the checks of the generated files in `data/` after a rebuild:
  row order, bins adding up, pair files, eligible passers) failed on the coverage floor because
  `addopts` always measures coverage and those 6 tests touch little code. The maintainer approved
  option (a) on 2026-10-08: a `tests/conftest.py` hook lifts the coverage floor and report only
  when every selected test is a `published_data` one. Checked on the real `data/`: the plain
  command passes (6 tests, no coverage report), and `pytest -m "published_data or not
  published_data"` still enforces the floor (100%). `scripts/refresh-season.sh` dropped
  `--no-cov`.

## Data notes (2026-10-08)

- nflverse play-by-play has no rows for three regular-season games, so the team game logs lack
  them: `1999_01_BAL_STL`, `2000_03_SD_KC`, and `2000_06_BUF_MIA`, from `POLARS_MAX_THREADS=1
  .venv/bin/python .agents/findings_2026_10_08/missing_pbp_games.py 1999 2000 2022` (nflverse's
  schedule against `data/{season}_team_game_logs.parquet`; play-by-play rows for the missing games:
  0). BAL and LAR (1999) and KC, LAC, BUF, and MIA (2000) are rated on 15 games. 2022 BUF and CIN
  have 16 games because their game was cancelled; the same command lists no missing 2022 game.
- Team stat fixes (found by the P5 audit, on `fix/team-stat-bugs`): `turnover_pct_per_drive`
  counts nflverse "Turnover" drives plus "Opp touchdown" drives with an interception or lost
  fumble (it matched "Interception" and "Fumble", which nflverse never uses); `turnover_margin` is
  `takeaways` minus `giveaways` (it added forced fumbles and missed fumbles lost after a catch);
  `targets` counts official attempts with a named receiver (it copied attempts) and `catch_rate`
  is completions per target, both null in 2003-2008, when play-by-play names almost no receiver
  on an incompletion; `air_yards_per_attempt` keeps its per-attempt definition and loses the aDOT
  label; receiving fumbles are the receiver's own; scrambles, designed carries, and rush success
  rate leave out plays a penalty wiped out; `epa_per_carry` sums EPA over the carries it divides
  by; stuff rates leave kneel-downs out; `drive_penalty_yards` (net penalty yards gained) grades
  higher as better; and the penalty columns' text says they include special-teams plays.
  `data/` keeps the old values until a rebuild (ask first): a scratch 2025 build compared with
  `nfl-sos-ratings diff-data --season 2025 --tolerance 1e-9` (smaller differences are rounding in
  opponent averages) changed only those columns and their `opp_` averages in the team game logs,
  per-game stats, combined, and opponent-profile files, plus `turnover_margin` in the QB game
  logs; no rating, range, pair, or win-probability file changed. Still open: the one-play
  extra-point group after a return touchdown counts as a drive in `drives` and the per-drive rates.
- Lost fumbles go to the team that fumbled (branch `fix/fumble-attribution`; `data/` not rebuilt).
  nflverse's `fumble_lost` flags a play on which a fumble was lost, whichever team fumbled, and
  `fumbled_1_team` names the fumbler; on a punt `posteam` is the punting team (Pro Football
  Reference's expected points also take the punting team's side). The stats read `fumble_lost` as
  the possession team's, so a returner's lost muff or fumble was the punting team's giveaway in
  `turnover_epa` (with positive EPA) and the receiving team's takeaway in `takeaway_epa`, and an
  intercepting defender's fumble the offense recovered was an offensive fumble lost (`fumbles_lost`,
  `giveaways`, `turnover_margin`). nflverse's data is right; the reading was wrong (an earlier note
  here blamed nflverse). `POLARS_MAX_THREADS=1 .venv/bin/python
  .agents/findings_2026_10_08/lost_fumbles.py 1999 2025` counts 706 punts whose lost fumble was the
  receiving team's (4 the punting team's), 70 interceptions fumbled back, and 7,564 other plays
  whose fumble was the possession team's. `POLARS_MAX_THREADS=1 .venv/bin/python
  .agents/findings_2026_10_08/espn_fumbles_lost.py 2025,2024,2019,2015,2010,2005,2002,1999 12`
  compares every team's fumbles lost in those punt games with ESPN's box scores: the fumbling team
  matches in all 188 team-games ESPN reports (4 have no count), the possession team in 2.
  Pro Football Reference's team stats agree in the four games checked by hand (1999 DEN at TB, 2015
  DEN at KC, 2021 GB at ARI, 2025 TEN at DEN). `pbp_expressions.lost_fumble_team_expr` now names the
  fumbling team (the usual side when `fumbled_1_team` is missing), and `giveaway_team_expr` the
  first team to give the ball away (the passer's after an interception). `turnover_epa` is the EPA
  of a team's giveaways from its own side, a returner's fumble included, `takeaway_epa` the
  opponent's reversed, and `fumbles_lost`, the sack and rushing fumbles lost, the QB's sack fumbles
  lost, and the drive giveaway flag count the offense's own fumbles. Scratch 2025 builds from `main`
  and the branch, compared with `nfl-sos-ratings diff-data --before <main> --after <branch>
  --season 2025 --tolerance 1e-9`, changed `turnover_epa` and `takeaway_epa` in 28 team-game rows
  and the fumble, giveaway, takeaway, and margin columns in 4, with their season and `opp_`
  averages; no rating, range, pair, history, or win-probability file changed.
- The 2026 Broncos question (maintainer, 2026-10-08): through week 4, DEN's head-to-head-excluded
  `sos` (5.28, `data/2026_ratings.parquet`) is the hardest in 2026 and above every completed
  season's (2009 TB, 2.83, from `nfl-sos-ratings schedules`), yet `team_rating` is -0.37 (15th).
  `POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/den_rating_split.py --season
  2026 --team DEN` refits `data/2026_team_game_logs.parquet` with 2025's cross-validated penalty
  (316): before adjustment DEN was 23rd by point margin (-5.00 per game) and by EPA (-6.55); fully
  adjusted with no penalty it would be +6.19 (8th), a schedule credit of about 12.7 points that
  rests on 4-game estimates of SF, KC, JAX, and LAR; the penalty keeps 43% of DEN's offensive
  evidence and 46% of its defensive (the shrink factors), schedule credit included; the 95% rank
  range is 5th-30th. Nothing is wrong with the fit; the ratings know nothing about 2025 (LAR 1st,
  JAX 4th, DEN 6th in `data/2025_ratings.parquet`), which P6 tests. The same command's prior sketch:
  DEN +1.01 (12th), +1.51 (10th), and +2.43 (9th) with priors at 33%, 45%, and 67% of 2025, and
  +0.32 (13th) at 45% without DEN's own prior.
- Rams team codes (found by the P5 audit, fixed on `fix/team-abbreviations`): play-by-play writes
  the Rams as `LA` in every team column and yard line, and the loader normalized only `posteam`,
  `defteam`, `home_team`, and `away_team`. So `penalty_team`, `td_team`, and the team in
  `drive_start_yard_line` never matched `LAR`: in every season file in `data/`, LAR's
  committed-penalty columns (counts, yards, splits, and rates) and touchdown columns read 0, its
  drive starts in its own half are mirrored (a start at its 25 reads 75), and
  `long_field_score_pct` is null; its opponents' `avg_starting_field_position_allowed`, both
  sides' penalty differentials, and the `opp_` averages that include LAR are off too. The loader
  now normalizes every team column and yard line. `data/` keeps the old values until a rebuild
  (ask first): scratch 2025 builds before and after the fix, compared with `nfl-sos-ratings
  diff-data --season 2025`, changed only those columns, in the team game logs, per-game stats,
  combined, and opponent-profile files; no rating, range, pair, QB, or win-probability file
  changed.
- QB stat fixes from the 2026-10-08 data audit (branch `fix/qb-stat-bugs`; `data/` not rebuilt):
  `qb_sack_yards_lost` was stored negative, as nflverse publishes it, so ANY/A added sack yards;
  scrambles and kneels were 0 because nflverse sets `rush = 0` on them; `qb_win_pct` was .500
  without a primary-passer decision (now null); `qb_offense_snaps` was 0 before 2013 (now null;
  nflverse's 2012 snap-count file has no rows). Scratch `season` builds of 2001, 2012, and 2025
  with and without the fixes, compared with `nfl-sos-ratings diff-data --before <before> --after
  <after> --season <season>`, changed only `qb_combined`, `qb_per_game_stats`, `qb_game_logs`, and
  `qb_opponent_profiles`; every rating, range, and pair file is unchanged. In the fixed build's
  `2025_qb_per_game_stats.parquet`, Drake Maye's ANY/A is 8.26 (9.01 before) and Josh Allen's 6.84
  (8.03). Next, each ask-first: rebuild `data/`, then rerun `validate`, whose QB ANY/A stability
  (0.392, quoted in `docs/methodology.md`) came from the sign-flipped ANY/A.
- Open decision (maintainer): scrambles in QB dropbacks. nflverse leaves the passer empty on
  scrambles, so `qb_dropbacks`, `qb_passing_epa`, and the QB rating count pass attempts and sacks
  only, while team `dropbacks` and the registry text include scrambles. `POLARS_MAX_THREADS=1
  .venv/bin/python .agents/findings_2026_10_08/qb_scramble_dropbacks.py --season 2025` (reads
  `data/`): 18709 QB dropbacks carry 580.2 passing EPA, and the 1087 scrambles on the same QB rows
  another 520.7; refit with scrambles counted, 23 of 33 qualified passers change rank (by at most
  4; Spearman 0.986 between the two ratings), Maye stays 1st (0.210 to 0.254), and Allen moves
  from 11th to 8th (0.102 to 0.149). Counting them changes a published rating, so it needs a
  protocol here before any code. Until then `qb_scramble_rate` divides scrambles by dropbacks that
  leave them out.
- Jacksonville's 2001-2002 home games (branch `fix/jax-team-codes`; `data/` not rebuilt). In these
  16 games nflverse credits every player to the visiting team: the play-by-play player team columns
  (`td_team`, `penalty_team`, `fumbled_1_team`, recoveries, tackles), the weekly player stats, and
  the weekly team stats, which have one row holding both teams' totals (nflverse-pbp issue 92,
  open; `posteam`, `defteam`, and EPA are right). So JAX's touchdowns, penalties, and fumbles went
  to the opponent, the official-stat override put both teams' totals in the visitor's offense and
  JAX's defense allowed, JAX's defenders' sacks and tackles went to the visitor, and JAX's QBs found
  no official row. `POLARS_MAX_THREADS=1 .venv/bin/python
  .agents/findings_2026_10_08/one_team_games.py 1999 2025` lists the same 16 games in both
  sources and no others. The loaders now find such games (`player_team_repair.one_team_games`),
  rebuild each player's team from the season roster restricted to the game's two teams (2002's
  roster writes league codes such as HST and CLV, mapped in `data_loader`), take a penalty with no
  player from the play text, blank `return_team` and `timeout_team` there, and leave those games
  out of the official team stats so the play-by-play values stand. Scratch 2001 and 2002 builds
  from `main` and the branch, compared with `nfl-sos-ratings diff-data --before <main> --after
  <branch> --season <season> --tolerance 1e-9`, changed the team game logs, per-game stats,
  combined, opponent-profile, and QB stat files; no team rating, range, pair, history, or
  win-probability file changed, and in the QB ratings file only `qb_attempts_total`, for one
  passer a season (sacks no longer counted as attempts). JAX's passing yards allowed per game fell
  from 347.0 to 234.8 in 2001 and from 303.6 to 218.0 in 2002 (the `combined` files).
- App and registry fixes from the P5 audit (branch `fix/app-stat-display`; no `data/` change):
  Raw Total Stats multiplied `longest_pass` and `longest_rush` by games played (they now have the
  registry shape `max`); the detail page's "vs Season" column compared per-game matchup averages
  with the Raw Total Stats row's season totals (the page's game-by-game analytics now use the
  API's per-game row); `filtered_` dropped a context metric's `contextual` flag; `_change`
  inherited the base polarity (now neutral); the `_per_dropback`, `_per_attempt`, `_per_carry`,
  and `_per_drive` suffix rules matched no column (removed), and the `season_delta_` rule never
  reached the app (`/api/metadata` now serves the prefix rules, and the app applies it); the seven
  player-stat `def_` columns named play-by-play as their source (now `PLS`); proportions showed as
  fractions (the registry's new `percent` flag; the app shows 65.3%, CSV keeps 0.653).
- Not done, from `POLARS_MAX_THREADS=1 .venv/bin/python
  .agents/findings_2026_10_08/app_stat_display.py --team NE --season 2025` (reads `data/`):
  `yards_per_defensive_snap_allowed` is not a duplicate of the derived
  `total_yards_allowed_per_defensive_snap` (play-by-play yards over scrimmage snaps against total
  yards allowed, from official team stats where available), so it has no `duplicate_of`: they
  differ in 23 of 28 seasons (12 games in 2001, largest gap 6.739 yards per snap), as do
  `yards_per_offensive_snap` and `total_yards_per_offensive_snap`. Whether the Per-Play view keeps
  both near-identical columns is the maintainer's call. `opp_longest_pass` and `opp_longest_rush`
  average each opponent's per-game longest play, because the opponent profile averages them per
  game (it pools rates since `fix/season-rates-and-nulls`): NE 2025 shows 34.17 and 22.45 against
  63.57 and 54.64 for its opponents' season maxima. Keeping the maximum there, as the team's own
  season row does, changes `data/` (ask first). `fourth_down_aggressiveness` is 2.0 in two games
  (`2000_04_CIN_BAL` CIN, `2000_10_SF_NO` SF), the only values outside 0-1 among the 136
  percentage columns in `data/`.
- Season rates pooled, missing data blank, tests offline (branch `fix/season-rates-and-nulls`;
  `data/` not rebuilt). Rates: season rows averaged each rate's game values; they now divide the
  summed numerator by the summed denominator (`nfl_sos_ratings/pooled_rates.py`: every game rate
  carries hidden `_num_` and `_den_` parts, which the writers drop; passer ratings and margins
  are rebuilt from pooled inputs). Opponent profiles pool each opponent over its games without the
  head-to-head ones, then average once per opponent; `opp_longest_*` still average per-game
  maxima. `POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/season_rate_pooling.py
  --season 2025` lists each changed column's largest change: 123 team season columns change, led
  (in league standard deviations) by `fourth_down_pct` (TB 0.611 to 0.448),
  `red_zone_td_pct_allowed` (JAX 0.486 to 0.596), `points_per_red_zone_trip` (LV 4.96 to 4.50),
  `red_zone_td_pct` (LV 0.594 to 0.500), and `fourth_down_aggressiveness` (ATL 0.410 to 0.545);
  QB season rows change only in passer rating and CPOE (among qualified passers at most 4.52
  rating points, Spencer Rattler 81.98 to 86.50, and 5.24 CPOE points, Jacoby Brissett -3.60 to
  1.64).
- Blanks: `POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/source_coverage.py
  1999 2025` counts what nflverse fills: play-by-play has no air yards, air or YAC EPA, or
  expected YAC before 2006, yards after catch on 337 of 9,522 completions and pass depth on 344 of
  16,651 passes in 1999 and on none in 2000-2005, a QB-hit flag on no play in 2003-2005 (969 to
  1,048 a season in 1999-2002, all but 7 of them sacks), no drive penalty yards in 1999-2000, and
  a no-huddle flag on 1, 0, 0, and 22 dropbacks and runs in 1999-2002; weekly player stats credit
  0 tackles for loss in 2003-2011 (1 in 2006) and 0 QB hits in 2003-2005. The loaders blank those
  fields for those seasons (`_PBP_FIELD_GAPS` and `_PLAYER_STAT_GAPS` in `data_loader.py`), the
  stats built on them are null for every team, and the registry's `since` moves to 2006 (YAC, air
  and YAC EPA, deep attempts), 2003 (no-huddle rate), and 2001 (drive penalty yards), with gap
  notes on tackles for loss, QB hits, the pressure-events rate, and the no-huddle rate.
- No-huddle from 2003 on stays as recorded: the same command counts 275 to 506 flagged dropbacks
  and runs a season in 2003-2005 and 1,139 to 4,318 from 2006, but also the flagged share of a
  trailing offense's snaps in the last two minutes of a half, where a hurry-up offense almost
  never huddles: 0.0% in 1999-2001, 0.2% in 2002, 3.9-5.1% in 2003-2005, 7.7-13.7% in 2006-2012,
  and 15.5-21.0% since 2013. The flag follows the play text, which marks no-huddle snaps less
  often the older the season, so 2003-2005 cannot be told apart from low usage; the registry note
  says to compare teams within a season. Not fixed, from the same command: nflverse flags 26
  kneel-downs in 2000 and in 2001 against 232 to 434 in every other season, so most kneel-downs of
  those two seasons count as designed carries and stuffs.
- Tests: seven loader tests read real nflverse files, alone by download and in the full suite
  from the on-disk nflreadpy cache the front-door test switches to (the two weekly team stats
  tests through `load_team_stats`, four QB tests through `load_player_stats`, a playoff QB test
  through `load_rosters_weekly`). They are stubbed, and `tests/conftest.py` makes nflreadpy's
  downloader and `urllib.request.urlopen` (the ESPN QBR release assets) raise in every test.
  Guards: a test fails for a team rate without hidden parts or a formula, another for a QB rate
  outside `QB_RATES` without parts, and two `published_data` tests check that their fixtures build
  every rate column the team and QB game logs in `data/` carry.
- Passer rating ties round half up (78.75 shows as 78.8): the rating rounded half-to-even after
  floating-point noise, so a tie could land either way, and the QB opponent-profile rating changed
  with how many pairs were computed together.
- Checks of that branch: scratch `season` builds of 1999, 2004, 2010, 2025, and 2026 from
  `origin/main` and from the branch, compared with `nfl-sos-ratings diff-data --before <before>
  --after <after> --season <season>`, changed no rating, rating history, rank range, weekly rank
  range, pair, or win-probability bin file; the QB game logs changed only in `qb_passer_rating`,
  by 0.1 on 4 to 14 tie games a season; the team game logs changed only in the blanked columns of
  1999, 2004, and 2010; the season-stat, combined, and opponent-profile files changed in rate
  columns and the blanked ones. Next, each ask-first: rebuild `data/`, then rerun
  `validate`, whose QB passer-rating stability (0.464 in `docs/methodology.md`) came from passer
  ratings averaged over games. Left as they are: the walk-forward `RawEPA` baseline averages game
  EPA margins per play (`_raw_epa_snapshot` in `validation/walk_forward.py`; changing it moves the
  validation report); the app's opponent ledger averages a division rival's two game rates and its
  recent-form highlights average game values (`web/src/domain/detailAnalytics.ts`); QB hits in
  1999-2002 stay as recorded, on sacks only (the registry note says so); play-by-play's
  `tackled_for_loss` covers 2003-2011, so counting it could fill the player-stat gap (a source
  change for the maintainer). The P5 tooltip drafts that end formulas with "Season: average of the
  game values" no longer apply.

## U. UX audit (2026-10-08)

A fresh audit of the whole app, as a newcomer and as a frontend developer: every page in both
themes at 1440 px and 390 px, all 32 palettes, edge states (a QB below the qualifier, a QB without
dropbacks, an unknown team or season, 1999). Each item names where it lives; the groups are sized
as pull requests, in the recommended order.

Bugs and copy (first):

- [x] U1 The "Season in progress" notice shows on three completed seasons, 1999, 2000, and 2022,
  because `seasonRules.getInProgressGames` treats any team below 16 or 17 games as unfinished (see
  "Data notes"); the QB qualifier text then says "so far". Let the API say which season is in
  progress (`config.SEASON` while its games are still being played) instead of counting games.
  Done: the season payload's `in_progress` (`ui_data.season_in_progress`) drives the notice.
- [x] U2 Plurals: "1 games" and "1 opponents" badges on detail pages; "How often this qb was rated"
  (`HeadToHeadCard`); QB titles use a hyphen ("Brock Purdy - San Francisco 49ers"); the glossary
  says "used throughout the current shell" (`metricMetadata.ts`) and lists QB EPA Per Dropback twice.
  Done except the glossary items, which move to P5 with the glossary rebuild.
- [x] U3 An unknown team (`/teams/XYZ`) or season (`?season=1990`) silently lands on the current
  index; say what was not found. Done: a notice names the missing team or QB, and an unavailable
  season says which season shows instead (carried through the index page's URL rewrite).
- [x] U4 "no games in 1%" / "no dropbacks in 2%" in rank-range summaries is opaque; say "missing
  from 1% of resampled seasons".

Index pages (layout):

- [x] U5 The table starts far below the fold: the season notice, the "Use first" card, and the
  garbage-time filter put it at about 700 px on desktop and 1,170 px on a 390 px phone, so a phone's
  first screen has no data. Fold the notice and the primary-rank note into one line under the title,
  move "Reading notes" into a hint or the glossary, and make the garbage-time filter a toolbar
  control beside the views. Done on `feat/index-layout`: the ranking sentence is the page
  description, the reading notes open from "How to read this page", a season in progress adds one
  line under it, the QB qualifier switch joined the table toolbar (its explanation in a hint), and
  the table starts at about 340 px (completed season) or 415 px (in progress) on desktop, with
  data rows on a phone's first screen. Changed from the wording above: the garbage-time filter
  moved below the rank ranges as a folded section (open when the address has `?wp=`), not into the
  toolbar, because it shows its own table of filtered ratings rather than filtering the main one.
- [x] U6 Nested scrolling: the table scrolls inside the scrolling page (`max-h-[75vh]`), so the
  wheel gets captured, and at 1440 px the SRS column hides behind a horizontal scroll while the
  Compare column (108 px for a checkbox) and Rank range (196 px) take room. Let the page scroll
  with a sticky table header and tighten those widths. Done on `feat/index-table`: the table box
  fills the screen below the app header at every width (as it already did on phones), so once
  the page reaches it, it reads as one sheet with a sticky header row; Compare is a 44 px icon
  column and Rank range 168 px, so SRS fits at 1440 px. Not done literally: a page-level sticky
  header cannot coexist with the table's sideways scroll in one box, so the box keeps its own
  scroll at full height.
- [x] U7 Noise: the "32 rows / 7 columns / 0 compared" and "6 columns / 110 columns" pills, the
  "USE FIRST" eyebrow, the card title repeating the page title, and the rank-range readout box
  ("Tap a row for its numbers.") that looks like an empty input. The floating scroll buttons cover
  the table's last column at the bottom right. Done on `feat/index-layout`: the index pills, the
  eyebrow, the repeated title, and the readout box (plain text until a row is picked). Left for
  the U6 change: the floating scroll buttons; the detail page's "6 columns / 110 columns" pills
  go with P8. Floating buttons, on `feat/index-table`: smaller (32 px) and translucent until
  hovered or focused, so the corner they cover stays readable; they still sit over it.
- [x] U8 Comparison: the panel appears above the table, so ticking a box pushes the row under the
  cursor down; with two picks every heat cell is fully green or red (min-max over two values). Show
  a "N selected, compare" bar and the panel below or in a drawer, and shade cells against the
  season's range, not the picks'. Done on `feat/index-table`: the panel sits below the table, the
  toolbar shows "N selected" with "View comparison" (scrolls to it) and "Clear selection", and
  compared cells take the season's heat (the same color as the row's cell in the table).

Detail pages:

- [x] U9 The view tabs sit above the whole page but only change the Ratings tiles and the game-log
  columns; put them on the sections they drive. Done on `feat/detail-layout`: the page leads with
  the rating summary, rank range, head-to-head, and weekly charts; the tabs head a stats section
  (sticky only while it scrolls by) holding the view's stats, the game log, and the unique
  opponents. They offer the five stat views; a Ratings choice carried over from the index reads
  as Per-Game Rates there, as the game log already did.
- [x] U10 The stat tiles show bare values: add each one's rank ("-0.37, 15th of 32") and the
  unadjusted counterpart (team EPA margin, QB raw EPA per dropback) so the schedule adjustment is
  visible, the question the 2026 Broncos raised. The QB Ratings view (index and detail) leaves out
  raw EPA per dropback and dropbacks. Done on `feat/detail-layout`: `domain/ratingSummary` ranks
  each rating (QBs among the qualifiers; context columns such as SoS unranked) and the headline
  tile adds "Before the schedule adjustment" (2026 DEN: -0.37, 15th; EPA margin -0.108 per play,
  26th). The API's new `rating_companions` group puts raw EPA per dropback and total dropbacks
  beside the QB ratings in the Ratings view, while each column stays in its own view too.
- [x] U11 The garbage-time filter card sits mid-page (and shows for a QB without dropbacks); move
  it to the end as an exploration section, collapsed. Done on `feat/detail-layout`: the folded
  section from the index pages closes the page, and a team or QB without a rating gets none.
- [x] U12 "Game by game" tiles (Peak week, Recent 3-game, Closing form, Schedule edge) use jargon
  and a secondary stat (Points/Off Snap, EPA/DB); retire them or base them on the rating's own
  stat with plain labels. Done on `feat/detail-layout`: retired, with their domain code; the chart
  above the log already opens on the rating's per-game stat. The detail page's count pills (games,
  columns, opponents) went with them, the last of U7's noise items.

Charts:

- [x] U13 The weekly chart draws smoothed lines (`type="monotone"`) that overshoot between games,
  and defaults to the first numeric column (Pass Yds on team pages, 4QC on QB pages) instead of the
  rating's per-play stat. Straight segments, and a rating-first default.
- [x] U14 Axis ticks: Rating by week uses uneven ticks (-3.60, -2.70, ...); Rank by week crowds
  25th and 32nd. Round ticks from a nice-number step. Settle R3's early-week decision with it.
  Done: straight segments, `trend.niceTicks` (1, 2, 2.5, or 5 times a power of ten, covering the
  reference line), rank ticks every eighth rank on the short weekly axis, the game-by-game chart
  opening on `epa_margin_per_play` (teams) or `qb_epa_per_dropback` (QBs), and the weekly rank API
  starting at `ui_data.first_rank_history_week` (every team at 3 games; R3 settled).

Color semantics:

- [x] U15 Schedule strength (SoS, Faced Pass D) is heat-mapped as good or bad, so a hard schedule
  is green; it is context, not quality. Give context columns a single-hue or no heat scale, from
  registry metadata (the backend defines meaning). Done on `feat/color-semantics`: every column
  the registry marks `contextual` (SoS, Faced Pass D, the `opp_` columns) and the unique-opponent
  schedule tier shade in one neutral slate hue, deeper toward the tougher end, in every palette;
  the reading notes say so.
- [x] U16 The unique-opponent table heat-maps raw counts against one opponent (completions,
  attempts); shade rates only. Done on `feat/color-semantics` (`tableState.shadedColumns`).
- [ ] U17 The default palette's green-to-red heat scale is hard to read with red-green color
  blindness (about 1 in 12 men); consider blue to orange for the default. Waiting on the
  maintainer (asked 2026-10-08): a visible change to the default look.

Glossary and navigation:

- [x] U18 The glossary covers about a dozen metrics, points at a repository path instead of linking
  the methodology, and explains neither rank ranges, head-to-head chances, the garbage-time filter,
  nor what positive SoS means. Build it from the registry with search and categories. Done on
  `feat/glossary` (`domain/glossary`): "Start here" with plain-language entries for EPA, the
  schedule adjustment, schedule strength, rank ranges, head-to-head chances, the garbage-time
  filter, and the shading, then the headline ratings; every other registry metric by entity and
  category in the registry's order, with its table label, direction, formula, first season, and
  any duplicate it repeats; a search over all of it; the methodology linked on GitHub. The old
  hand-picked sections (with the "current shell" text and the duplicate entry) are gone.
- [x] U19 The palette menu is a 33-item scrolling list; an eight-division grid of team chips would
  be faster, especially on a phone.

## P6. Preseason prior for the team fit

Background: the 2026 Broncos question (see "Data notes"). After four games the team fit pulls every
estimate about halfway toward average (zero), and it knows nothing about the previous season.
A scratch refit of 2026 with a prior inside the solve (2026-10-08, `/tmp` script, not citable)
moved DEN from 15th to 10th with 45% of 2025 carried over, and a prior of zero reproduced the
published ratings exactly. `../nfl-predictor` blends a regressed prior after its solve
(`strength_snapshot._blend_prior`: two-thirds of last season, weight `games / (games + 4)`); its
own 2026-09-25 review found that this shrinks the in-season solve twice and lets last season
dominate at week 3 (correlation 0.96 with the prior against 0.45 with the in-season solve), and
its constants were not tuned. This design puts the prior inside the solve instead.

Maintainer decisions (2026-10-08): combine both kinds of regression to the mean; fade the prior out
early (gone by mid-season or sooner, the exact point for the test to settle), so completed seasons
keep their published ratings; teams first, QBs as a separate later test; the check may be run when
built; a `data/` rebuild needs a fresh yes.

Protocol history: drafted 2026-10-08; an independent review the same day (two blockers, eight
majors, six minors, all resolved below) saw only descriptive inputs: the 2026 DEN sketch (outside
the window), carryover slopes, shrink factors, and game counts. No candidate's walk-forward error
has been computed by anyone before this protocol was committed. Two scope choices the review left
to the maintainer were made conservatively, so the maintainer can revisit them before any
adoption. The prior covers scrimmage only; the alternative is a special-teams prior built from a
fit at a fixed, season-independent penalty (the published special-teams penalty sometimes reaches
the top of its grid, which flattens that unit's effects for a season, as in 2020 and 2022). An
adopted prior keeps the head-to-head exclusion in `sos` exactly; the alternative is no prior in
the `sos` refit, at the cost of rating opponents with a different estimator from `team_rating`
early in a season.

### Estimator

The team fit has two ridge units, scrimmage and special teams. In the scrimmage unit, the penalty
pulls each offense and defense effect toward a prior mean instead of toward zero; the
special-teams unit is unchanged (prior means zero):

```text
minimize  sum_rows w (y - a - o[team] + d[opp] - h s)^2
        + lambda * sum_teams ((o[t] - m_o[t])^2 + (d[t] - m_d[t])^2)
raw_o[t] = c(g_t) * rho_o * o_prev[t]        (raw_d[t] likewise with rho_d and d_prev)
m_o[t]   = raw_o[t] - mean over the fit's offense units of raw_o   (m_d likewise)
c(g)     = max(0, 1 - g / G)
```

- Units: `y` is EPA per play and every effect is per play, as in `ridge`; a rating is an effect
  times the season's plays per team-game, as published.
- `lambda`: unchanged, the scrimmage penalty cross-validated on season `s-1`, as published today.
  It is not re-tuned for the prior; around an informative mean the matching penalty would be
  larger, so keeping it under-weights the prior. That is conservative: a tie does not show that
  no prior can help. Cross-validation never uses a prior, so every penalty is today's.
- `o_prev`, `d_prev`: the scrimmage offense and defense effects of `fit_unit_ridge` on season
  `s-1`'s rows with the scrimmage penalty cross-validated on season `s-2` (1999's own for 2000):
  the fit behind `{s-1}_ratings.parquet`. Teams are linked across seasons by the normalized team
  code (franchise-normalized for 1999-2025; HOU first appears in 2002). A team in season `s`
  without a season `s-1` effect raises an error instead of defaulting to zero, except 2002 HOU,
  which gets a raw prior of zero (outside the window either way).
- `rho_o`, `rho_d`: the least-squares slope, through the origin and unweighted over team pairs, of
  season `t`'s near-unpenalized scrimmage effects (`fit_unit_ridge` with `ridge_lambda = 1e-6`, an
  unbiased estimate of that season's effects) on season `t-1`'s `o_prev` (likewise `d_prev`),
  pooled over every pair `(t-1, t)` with `t < s` (expanding window, no look-ahead). Reason: the
  penalty treats `m` as the expected true effect, which this slope estimates without bias;
  shrinkage in `o_prev` itself is absorbed by the slope. Three pairs are needed, so 2003 is the
  first season with a prior.
- `c(g)`: the fade. `g_t` is the number of games team `t` has played in season `s` at the
  snapshot (all of its games before the prediction week). Every refit of a snapshot reuses the
  snapshot's `g`; no refit recounts games. Bootstrap resamples also reuse the snapshot's prior
  means; head-to-head-excluded refits take theirs as described under "If adopted". With every
  candidate `G` at most 9, below the shortest completed season in the
  window (16 games), every completed-season fit has a zero prior, and so does every snapshot in
  which every team has played at least `G` games.
- Centering: the fitted effects average to the average prior mean (the intercept is unpenalized),
  so the prior means are centered within each snapshot and side, which keeps the effects averaging
  zero when byes give teams different `c(g)`.
- A team with no games at a snapshot after week 1 (2017 MIA and TB before week 2) is rated at its
  prior means on that snapshot's per-game scale: offense `m_o[t]`, defense
  `m_d[t]` (its raw means shifted by the same centering as the fitted teams'), special teams zero.
  Week-1 snapshots have no games at all; their feature rows are the harness's zeros (below), and
  prior-only ratings appear only in the descriptive week-1 extra.
- Solved directly: `(X'WX + lambda P) b = X'Wy + lambda P m`, with `P` the penalty mask and `m`
  zero on the intercept and home field (`ridge.UnitPrior`, `fit_unit_ridge(..., prior=...)`). A
  prior needs a fixed penalty, so cross-validation never sees one. The equivalent residual form,
  the ordinary ridge on `y - (m_o[team] - m_d[opp])` with the means added back, is the independent
  side of check 5. The ridge code reads a missing prior key as zero, so the prior-construction
  module raises on a missing previous-season effect before calling it.

### Pre-registered test

- **Hypothesis (falsifiable):** shrinking the scrimmage effects toward the faded prior predicts
  game margins out of sample better than shrinking toward zero. Not supported for a horizon `G`
  unless its paired interval lies entirely below zero.
- **Candidates, fixed in advance:** `G` = 3, 6, and 9 games, each against today's fit (no prior).
  With `G = g`, a team's prior is gone once it has played `g` games; the DEN question is at
  `g = 4`, where the prior weights are 0, 1/3, and 5/9.
- **Allowed information set:** to predict week `w` of season `s`, a candidate sees season `s`'s
  games before `w`; seasons before `s` (for `o_prev`, the penalties, and the slopes); and the
  scrimmage penalty cross-validated on `s-1`, as published. The prior construction for season `s`
  is computed by a function that receives only the game logs of seasons before `s` (checked
  below). Its margin model sees only its own earlier predictions. Shared caveat, not a leak
  between candidates: nflverse's expected-points model is trained on many seasons, including ones
  after the season being predicted.
- **Harness:** `nfl_sos_ratings/validation/walk_forward.py`, as `check-wp-filter` uses it. Feature
  rows are built for every season from 1999 for every candidate (prior candidates equal today's
  fit before 2003, where they have no prior), so every margin model has the same warm-up. Week-1
  feature rows are the harness's zeros for every candidate. The margin model is `validate`'s:
  `home_margin = k * rating_gap + home_edge`, least squares on all of the candidate's earlier
  feature rows.
- **Metric and window:** mean absolute error of the predicted home margin over prediction weeks 2
  and later, seasons 2003-2025, on the games every candidate predicts.
- **Inference:** a paired bootstrap of the per-game difference |error at G| - |error today| that
  resamples whole seasons (23 clusters, 2003-2025) with replacement, keeping every game of a drawn
  season together; 10,000 resamples, seed 0, the same season draws for every candidate and every
  week band; 98.33% percentile intervals. "Qualifies" is a one-sided test at 0.83% per horizon, so
  the chance of adopting any horizon when none helps is at most 2.5% (Bonferroni, conservative
  here because the horizons are nested and their differences correlated).
- **Decision rule:** a horizon qualifies only if its overall interval lies entirely below zero and
  none of its week-band intervals (prediction weeks 2-4, 5-8, and 9 on; same draws and level)
  lies entirely above zero. The recommendation is the qualifying horizon with the lowest overall
  MAE. None qualifying is a tie, and the recommendation is no prior (the simpler option). Every
  interval that excludes zero is reported, in either direction. The decision goes to the
  maintainer either way; adopting changes the published early-week ratings, weekly histories, and
  the season in progress (ask before the rebuild).
- **Integrity checks, run before reading any result; if one fails, stop, investigate, and read
  nothing else:**
  1. Reproduction: for every `s` in the window, the per-game ratings rebuilt from `o_prev`,
     `d_prev`, and season `s-1`'s special-teams effects equal `{s-1}_ratings.parquet`'s four
     rating columns to 1e-9.
  2. Today's fit: its rating gaps and predictions for weeks 5 on equal the `TeamRating` rows of
     `walk_forward.run_walk_forward_backtest` over 1999-2025 with start week 5, computed in
     memory, to 1e-9 (the command never runs `validate`, which rewrites the report).
  3. Zero prior: with every prior mean zero, a candidate's ratings equal today's to 1e-9 on every
     snapshot.
  4. Fade: on every snapshot where every team has played at least `G` games, a candidate equals
     today's fit to 1e-9; on every snapshot, the prior means used equal `c(g_t) * rho * o_prev[t]`,
     centered, recomputed from the game counts, with `g_t` counting games before the prediction
     week only.
  5. Residual form: on every snapshot of 2003, 2014, and 2025, each candidate's scrimmage effects
     equal an independent residual-form fit (an ordinary `fit_unit_ridge` with no prior on the
     response `y - (m_o[team] - m_d[opp])`, the means added back) to 1e-9.
  6. Limit: with `lambda = 1e12` and `c = 1`, every scrimmage effect equals its prior mean to 1e-6.
  7. Coverage: in every season of the window, every team in the fit has a previous-season effect
     (no default fills a missing key).
  8. Information set: the prior construction for each `s`, run on a data directory holding only
     the seasons before `s`, gives the same `rho` and `o_prev` as the full run.
  9. Penalties: every penalty a candidate uses equals today's exactly.
  10. Inputs: the check prints a SHA-256 fingerprint of exactly the files its decision reads, in
      name order: `{season}_team_game_logs.parquet` for 1999-2025 and `{season}_ratings.parquet`
      for 2002-2024 (each file's `sha256sum` line, hashed again), recorded with the results. The
      2026 game logs, read only by the DEN extra, get a fingerprint of their own.
- **Descriptive extras (never decision inputs):** MAE and paired intervals for weeks 2-4, 5-8, and
  9 on; the single-game bootstrap intervals; each candidate's fitted margin slope `k` by band, so
  a gain from rescaling rather than information shows; Elo's MAE by band as the existing carryover
  reference; the same results for `G = 17` and a no-fade prior (both change completed seasons, so
  neither is adoptable); `rho` by season; DEN's 2026 rating and rank after week 4 for each
  candidate; the week-1 MAE of the prior alone against the home-edge-only prediction.
- **Command:** a new read-only `nfl-sos-ratings check-team-prior --data-dir data --start-season
  2003 --end-season 2025`, modeled on `check-wp-filter`; it reads `data/` and writes only to
  stdout.

### If adopted (after the maintainer's decision; not needed for the test)

- Every refit takes the snapshot's prior: weekly rating histories, rank ranges, and head-to-head
  chances reuse the snapshot's `g` and prior means. Test: a bootstrap refit in which a team is
  drawn fewer than `G` times keeps the snapshot's prior. Integrity check before the rebuild: for
  every completed season, `sos`, rank ranges, and head-to-head chances (same seeds and draws)
  equal the published files.
- Head-to-head-excluded `sos`: the opponents' prior means come from a refit of season `s-1`
  without the evaluated team's games, at the same penalty as `o_prev` (the scrimmage penalty
  cross-validated on `s-2`) and with the same `rho` and the snapshot's `g`, centered over that
  refit's own units (every team but the evaluated one). So the evaluated team's results never
  move its opponents' ratings. Test: changing the evaluated team's previous-season games against
  an opponent leaves that team's `sos` unchanged.
- `docs/methodology.md` states that early-week rank ranges hold the prior fixed and so understate
  uncertainty until week `G`.

### Tests (test-first) for the prior construction and the check

- A synthetic league whose true effects carry over with a known slope: the near-unpenalized slope
  recovers it within sampling error while the published-on-published slope does not, and the
  prior beats the zero prior out of sample.
- Sign: one strong offense and one strong defense, each prior pulling its effect the right way.
- Units: at `lambda -> infinity` a rating equals `m * plays_per_game`.
- Fade: `c(g)` at `g = 0, G/2, G, G + 3`.
- Centering: on a snapshot with byes, the effects average zero to 1e-12.
- A prior without a fixed penalty raises: the cross-validation path takes no prior.
- The season bootstrap: when the paired differences are constant within each season, its interval
  equals the interval from resampling the season means with the same draws.
- The bootstrap pivot has the same game rows for every candidate (no row dropped).
- A team without games at a snapshot gets its prior rating; a missing previous-season key raises.

Tasks:

- [x] Commit this protocol before any run (pre-registration): its own pull request, merged
  before any of the check's code.
- [x] Independent review of the protocol (a fresh subagent, 2026-10-08), findings resolved here.
- [x] The prior construction (`o_prev`, slopes, fade, centering) in a new module, on the `ridge`
  prior already built (`UnitPrior`); test-first as listed. Done on `feat/team-prior`:
  `nfl_sos_ratings/team_prior.py` (`PriorHistory`, `snapshot_fit`), `fit_team_ratings(...,
  scrimmage_prior=...)`, and a prior that refuses to run without a fixed penalty.
- [x] `check-team-prior` test-first; integrity checks run on real data before reading any result.
  Done: `nfl_sos_ratings/validation/team_prior_check.py`.
- [x] Independent code review (2026-10-08): 0 blockers, 1 major, 7 minors, all fixed before any run.
  The major: no check read the prior the candidates use (swapping its offense and defense passed
  every check); check 1 now rebuilds all four published columns from that prior, and a new check
  recomputes the carryover slopes by exact least squares from the published files (tolerance
  1e-5). The minors: the snapshot audit now checks the means each fit used (`snapshot_fit`),
  compares teams without games with their prior, and runs the residual and limit checks on every
  window snapshot rather than three seasons (a superset of checks 5 and 6); candidates must equal
  today's fit in the seasons before a prior; duplicate rows are rejected before pairing; NaN
  counts as a mismatch; the single-game bootstrap draws in chunks; and failure tests break the
  production code (wrong means, a solver that drops the prior, a drifting warm-up).
- [x] Run the check (approved), 2026-10-08, from commit `f274469` on `feat/team-prior`:
  `POLARS_MAX_THREADS=1 .venv/bin/nfl-sos-ratings check-team-prior --data-dir data
  --start-season 2003 --end-season 2025` (input fingerprint `1fe53e2b...16fa`, 53 files).
  Integrity: every check passed (previous seasons' ratings rebuilt from each prior with gap 0;
  slopes against the exact recompute 9.2e-10; fully faded rows equal today's on 10,941 rows;
  1,119 audited snapshots, means 1.4e-17, residual form 2.1e-14; teams without games on 3
  snapshots, 2017 week 2; limit 2.4e-10 on 373 snapshots; 23 information-set seasons; 27 seasons'
  penalties). Results, MAE of predicted home margins, prediction weeks 2+ of 2003-2025 (5,600
  games): today 10.750, 3 games 10.714, 6 games 10.685, 9 games 10.668. Paired differences,
  98.33% season-bootstrap intervals: 3 games -0.036 (-0.053 to -0.017), 6 games -0.064 (-0.090 to
  -0.038), 9 games -0.082 (-0.114 to -0.050). Every interval excluding zero favors the prior:
  each horizon overall and in weeks 2-4 (3: -0.189; 6: -0.253; 9: -0.264), and 6 and 9 in weeks
  5-8 (-0.069, -0.136); no band interval lies above zero, so the guard removes none. Decision
  rule: all three qualify; the recommendation is the 9-game horizon (lowest MAE). Descriptive:
  weeks 9+ are unchanged for every horizon; 17 games and no fade gain little more overall
  (-0.095, -0.093) and are not adoptable; the fitted margin slope is about 0.89 for every
  candidate (no rescaling); Elo by band 10.996 / 10.730 / 10.605; carryover slopes 0.68-0.82
  (offense) and 0.38-0.47 (defense); 2026 DEN after week 4 -0.37 (15th) today, +0.96 (12th) at 9
  games; week 1 by the prior alone MAE 10.341 against 10.689 for the home edge alone.
- [ ] Maintainer decision on the 9-game horizon (asked 2026-10-08).
- [ ] If adopted: the refits above, then ask before the `data/` rebuild, then update the registry,
  `README.md`, `docs/methodology.md`, the validation report, and the nfl-predictor note.

## Ideas parking lot (not approved yet)

- Rank ranges at a WP threshold, once WP4 has run.
- A league heatmap of head-to-head chances (row team rated above column team), after R1.
- Season-over-season rank change on the detail page.
- A CI job that runs `scripts/gate.sh --quick` on every commit of a pull request (needs approval:
  CI changes are ask-first).
