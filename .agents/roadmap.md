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

## Where things stand (2026-10-04)

- Pull request #1 (`feat/rank-ranges`, merged as `c396d83`, branch deleted) brought the rank
  ranges (engine, outputs, API, web views), the QB data fixes, the per-team QB qualifier, the UX
  audit changes, the regenerated validation report, and docs.
- S1-S3 merged as pull request #2 (`1d61b2d`), WP1 as #3 (`c574819`), WP2 as #4 (`b70b522`), and
  WP3 as #5 (`main` = `02a2b47`), branches deleted. Branch `fix/wp-qb-team` (from `02a2b47`)
  restores each QB's team in the filtered table, lowers the filter's maximum to 20%, and carries
  a `filelock` lockfile bump.
- `data/` (1999-2026: range files, fixed row order, and win-probability bins) was rebuilt on
  2026-10-04 from `c574819` with no value changes (S2 records the `diff-data` summary), and
  `.venv/bin/pytest -m published_data` passes on it.
- The maintainer runs `nfl-sos-ratings web --host 0.0.0.0 --port 8081` to view the app on a phone.
  Never stop it; use port 8090 for agent checks.

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
| 11 | F1-F6 Frontend follow-ups | Detail pages, opponent context, compare layout, export |
| 12 | S4 Project logger | Touches many modules; any quiet stretch works |
| 13 | M1 Retired stats, one at a time | On request; each needs its own verification |

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

Write this section's protocol here, in full, before any run (AGENTS.md):

- Hypothesis: dropping garbage-time plays from the rating inputs improves out-of-sample prediction
  of game margins.
- Candidates fixed in advance: 5%, 10%, and 20%, each against 0%.
- Metric and window: walk-forward MAE of predicted margins, weeks >= 5, 1999-2025, as the current
  rule (`nfl_sos_ratings/validation/walk_forward.py`).
- Inference: paired bootstrap of per-game error differences against 0%, Bonferroni-adjusted for
  three comparisons (98.33% intervals).
- Decision rule: a threshold is a candidate for the default only if its interval excludes zero in
  its favor; among such thresholds, the lowest MAE. Inconclusive means a tie, and the simpler
  option (0%, no filter) is the recommendation; the decision goes to the maintainer either way.
  Report every interval that excludes zero, in either direction.
- Descriptive extras (not decision inputs): QB year-over-year stability and QBR correlation at
  each candidate; NE 2025 and Maye across thresholds, whichever way they move.

Decided with the maintainer (2026-10-04), to be written into the protocol above:

- Each candidate is the published team fit run unchanged on the kept plays: kept plays and EPA
  replace the game-log columns `fit_team_ratings` reads, and each threshold's penalties are
  cross-validated on the previous season's kept plays (1999 cross-validates its own), as the
  published fit does with every play. Reason: the kept share falls fast, so 0% penalties would
  shrink filtered ratings harder for a reason unrelated to garbage time. Mean
  `wp_kept_play_share` over the 32 teams in 2025: 0.846 at 5%, 0.772 at 10%, 0.623 at 20%, 0.453
  at 30%, from `curl -s 'http://127.0.0.1:8090/api/seasons/2025/teams/wp-ratings?threshold=X'`
  (server: `nfl-sos-ratings web --port 8090`). If a threshold is adopted, the exploration view
  switches to the same penalties.
- A separate read-only command, like `check-in-season-penalty`; `validate` and its report stay as
  they are unless the maintainer adopts a change.
- 10,000 paired-bootstrap resamples rather than the validation's 2,000: at 2,000, each tail of a
  98.33% interval rests on about 17 draws.
- The QB extras compare each threshold with the 0% play-level value, not the published rating,
  which uses official weekly passing EPA.
- The slider's and API's maximum is now 20% (see the decisions above). It does not affect the
  test.

Tasks:

- [ ] Finish the protocol above (anything left open) and commit it before running.
- [ ] **Ask first**, then run; write the results here and in the validation report if the
  maintainer adopts a change.

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

- [ ] Registry entries; `summarize_rank_pairs` with tests (a synthetic league where one team is
  clearly better; symmetry P(A over B) + P(B over A) = 1 for teams).
- [ ] `run_season` writes the files (**Ask first** for the rebuild); API
  `GET /api/seasons/{season}/{teams|qbs}/{id}/rating-pairs`.
- [ ] Detail page: "Compare with" picker with a plain sentence ("NE rated above BUF in 38% of
  resampled seasons; difference -1.2 points, 95%: -5.0 to +2.8"); the comparison panel shows the
  same when exactly two rows are compared.

### R2. Rank ranges for the unit ratings

Background: `TeamRatingResampler.ratings` already returns `offense_rating`, `defense_rating`, and
`special_teams_rating` for every resample; only `team_rating` is summarized.

Tasks:

- [ ] Registry: `offense_rank`, `defense_rank`, `special_teams_rank` (the quantile suffix rules
  already exist); summarize all four ratings in `build_team_rating_ranges`.
- [ ] Note in `docs/methodology.md` that percentiles do not add up (the median of a sum is not the
  sum of the medians), unlike the published ratings.
- [ ] Detail page: unit rows with mini intervals; the API payload gains column groups per unit.

### R3. Rank ranges by week

Background: a season in progress shows very wide ranges that narrow as games accumulate; a weekly
view makes that visible and complements the rating history.

Design: for each week, the games through that week, bootstrapped with the season fit's penalties
(as rating histories do), giving `{season}_rating_ranges_by_week` (week, team, rank percentiles,
P(top 5), P(top 10)) and the QB counterpart. Cost scales with weeks: measure one season first;
options are all seasons with fewer resamples (for example 200) or only seasons in progress.

Tasks:

- [ ] Measure the cost for one season and record it; choose the scope with the maintainer if it
  adds more than a couple of minutes to `pipeline`.
- [ ] Detail page: median rank by week with 50% and 95% bands (rank 1 at the top), beside "Rating
  by week".

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

- [ ] Script with shellcheck-clean Bash (shell-scripting skill) and a dry-run flag; tests where
  practical (Bats is optional).
- [ ] **Ask first** before installing any scheduled task (machine configuration outside the
  repo); provide the exact install command or task XML for the maintainer.
- [ ] Season rollover note in `current-status.md`: after the 2026 season, set `END_YEAR` to 2026
  and `SEASON` to 2027 in `config.py`, rebuild (ask first), and regenerate the validation report.

## F. Frontend follow-ups

Carried from the frontend plan's queue (F1-F4) plus two ideas (F5-F6). Use the frontend-design,
frontend-react, and dataviz skills; keep the UX audit's patterns: `Hint` for every tooltip
(hover with a mouse, tap on touch screens), `SortableHeader` for sortable columns, only the name
pinned on phones, fixed decimals per column.

- [ ] F1 Weekly-log detail pages: strengthen them now that the weekly trend chart has landed; keep
  them table-first and check the chart against live use before adding chart types.
- [ ] F2 Grouped opponent ledgers: refine after live use, especially if one team or QB surface
  wants a different primary metric or a tighter default column mix.
- [ ] F3 Opponent-strength context in the weekly views: show each opponent's season-long rating and
  rank range, labeled as season-long, never as a single-game rating.
- [ ] F4 Comparison: a pinned side-by-side layout instead of the compact strip, with rank-range
  mini intervals and R1's head-to-head sentence when two rows are compared.
- [ ] F5 Idea: CSV export of the current table view, built from the loaded payload (current view,
  sort, and filters), so an analyst can take the numbers elsewhere.
- [ ] F6 Idea: rank-range mini intervals in the comparison panel even before F4.

## S4. Project logger

Background: package modules print progress, which needs per-file `T201` ignores in
`pyproject.toml`; nfl-predictor has a small colored logger. Approved 2026-10-04, including
removing those ignores once nothing prints (a ruff configuration change the maintainer has
agreed to).

Tasks:

- [ ] Use the observability skill; stdlib `logging` with one small formatter (color only on a
  TTY), INFO by default, `--verbose` for DEBUG on the front door.
- [ ] Replace progress `print`s; keep data written to stdout (for example `schedules`) on
  `sys.stdout.write`.
- [ ] Remove the `T201` per-file ignores; the gate must pass without new suppressions.

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

## Ideas parking lot (not approved yet)

- Rank ranges at a WP threshold, once WP4 has run.
- A league heatmap of head-to-head chances (row team rated above column team), after R1.
- Season-over-season rank change on the detail page.
- A CI job that runs `scripts/gate.sh --quick` on every commit of a pull request (needs approval:
  CI changes are ask-first).
