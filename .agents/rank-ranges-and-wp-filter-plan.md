# Rank Ranges and Garbage-Time WP Filter Plan

Active plan (opened 2026-10-04) for two maintainer-requested features: bootstrap rank ranges for
teams and quarterbacks, and a garbage-time win-probability (WP) filter with a slider. Both exist so
the maintainer can see the uncertainty in the published 2025 results (NE 5th, Drake Maye 1st)
for themselves; they must show it honestly, not argue for a ranking. Background on the dispute and
the earlier checks: `.agents/ratings-simplification-plan.md` (checks A and B, in-season penalty
test). The brief came from a handoff prompt at `/tmp/ne/handoff-prompt.md`; its numbers came from
throwaway scripts (`/tmp/ne/analysis.py`, `/tmp/ne/qb.py`) and are not citable until a repo
command reproduces them.

## Order

1. Rank ranges (adds outputs, changes no published rating).
2. WP filter data layer and slider as an unvalidated exploration view.
3. Pre-registered walk-forward test of WP thresholds (only that could change the published 0%).

## Measurements (2026-10-04, single-threaded BLAS, 2025 data)

Scratch timings, not citable numbers; a repo command will report its own timings.

- Fixed-penalty `fit_team_ratings` about 78 ms, `fit_qb_ratings` about 37 ms; most of it is the
  Polars design build, not the solve. Head-to-head `compute_team_schedule_strength` about 2.7 s
  (64 refits).
- A naive 1000-resample bootstrap would cost about 2 minutes per season (about an hour for
  `pipeline`). See the weighted-solve design below, which avoids that.
- nflverse play-by-play has `wp`, `vegas_wp`, `def_wp`, `home_wp`, and `vegas_home_wp` in 1999,
  2006, and 2025; about 0.6% of rows have null `wp`, none of them 2025 scrimmage plays. Check null
  counts on scrimmage plays for every season in the loader tests before relying on it.

## Shared engine: a weighted ridge on a fixed design

With a fixed penalty, a game drawn k times in a bootstrap resample enters the least-squares fit
exactly as one copy with k times the weight; game identity mattered only for cross-validation
folds, which a fixed-penalty fit does not use. So build the design matrix once per season and
solve `(X' W X + penalty) b = X' W y` per resample with W = row weights times multiplicities.

- Same engine serves the head-to-head refits (weight 0 for the excluded games) and WP thresholds
  (per-row plays and EPA sums change with the threshold), so `sos` per threshold becomes cheap.
- Equivalence test (required): one resample through the engine equals `fit_team_ratings` on the
  explicitly duplicated-and-relabeled frame with the same fixed penalties.
- Points scale per resample: league-average plays per team-game recomputed from the resample, as
  `fit_team_ratings` would on the duplicated frame.

## Feature: rank ranges

- Method: resample a season's `game_id`s with replacement (default 1000 resamples, seed 0),
  refit with the season fit's penalties, rank every resample. Teams rank among all teams; QBs among
  published-eligible QBs (`qb_is_eligible` from the full season) present in the resample. A QB with
  no dropbacks in a resample has no rank there; report how often (`missing` share) and compute his
  quantiles over the resamples where he appears.
- Outputs per season: `{season}_rating_ranges` (teams) and `{season}_qb_rating_ranges`, one row per
  entity: rating quantiles (2.5, 10, 25, 50, 75, 90, 97.5), rank quantiles at the same levels, the
  published rank, P(top 5), P(top 10), the QB missing share, and the full P(rank = k) as one list
  column. Registry first: quantile suffix rules (`_q025` ... `_q975`) on `team_rating`,
  `adj_qb_epa_per_dropback`, and new rank metrics, so names resolve without dozens of entries;
  then `catalog`.
- Tests: the engine equivalence test above; a synthetic league with known effects and noise whose
  80% and 95% rating intervals cover the true effects near the nominal rate. Shrinkage biases
  extreme units toward average, so the tolerance is fixed before the first run (2026-10-04): an
  eight-team league, each pair meeting four times, 30 replications of 200 resamples; pooled over
  teams and replications, 95% intervals cover the truth 85-100% of the time and 80% intervals
  65-95%. Report the observed coverage either way. Optional: analytic ridge-posterior draws as a
  cross-check.
- Docs: `docs/methodology.md` states that the ranges cover game-to-game sampling noise of the
  shrunken estimate, not model error, and that in-progress seasons show very wide ranges.
- API: `GET /api/seasons/{season}/{teams|qbs}/rating-ranges` (one payload per kind; the detail page
  filters it). Missing file is a 404 and the UI leaves the views out, as for rating histories.
- UI (use the dataviz, frontend-design, frontend-react skills; mobile-first, dark mode, color never
  the only channel, text alternatives):
  - League view: one row per team or QB sorted by median rank, rank 1 left; thick bar for the
    middle 50%, thin bar for 95%, dot for the median, published rank marked when different. An
    interval chart from the box-plot family, not a Tukey box plot (outlier dots mean nothing here).
    Check Recharts 3 range bars (`Bar` with `[low, high]` data) versus custom shapes.
  - Detail page: P(rank = k) histogram with a plain headline ("6th; middle 50%: 4th-8th; 95%:
    1st-16th").
  - Table: a "Rank range" column with an inline mini interval.
- Producing the files needs `season` / `pipeline` runs: ask first.

## Feature: garbage-time WP filter

What rbsdm.com does (checked 2026-10-04, no public source found): its "Garbage-Time WP Filter"
slider defaults to 0% and runs 0-30%; a setting of X drops plays whose win probability is below X
or above 1 - X (Football Perspective, Adam Steele's QB recaps: 4% drops plays "below 4% or above
96%"). Not confirmed: whether it uses `wp` or `vegas_wp`, and whether the cut is inclusive.

Decisions (answered by the maintainer 2026-10-04):

- WP column: `wp` (score, clock, field position). `vegas_wp` adds the pregame spread, so early
  plays of a mismatch would already look like garbage time.
- Special teams: filtered too. The maintainer leaned that way; onside kicks and backup coverage
  units are concrete garbage-time effects on special teams, and one play set keeps
  `team_rating`'s three parts consistent. Special-teams plays get the same WP bins, so the choice
  stays cheap to revisit.
- Range and step: 0-30% in 1% steps, as rbsdm; default 0%.
- Rule: keep a play when `min(wp, 1 - wp) >= X`; X = 0 keeps every play with a `wp`.

Architecture: per team-game, store scrimmage and special-teams plays and EPA summed in 1% bins of
`min(wp, 1 - wp)` (bins 0-50); a threshold X keeps bins >= X, so any threshold is a cumulative sum
and the API refits on demand with the weighted engine (milliseconds, `sos` included). QB: per
passer-game dropbacks and play-level passing EPA in the same bins; at 0% compare with the official
`qb_epa_per_dropback` and state the tolerance (official passing EPA comes from weekly player stats,
not plays).

Methodology: the published default stays 0%. Before any run, write the walk-forward test here:
hypothesis, a small fixed candidate set (for example 5%, 10%, 20%), multiple-comparison handling
(for example Bonferroni-adjusted intervals), window (weeks >= 5, 1999-2025, as the current rule),
paired bootstrap, and decision rule. The UI labels non-zero thresholds as an unvalidated
exploration view; the threshold lives in the URL; refits are debounced. Report NE 2025 and Maye
across thresholds whichever way they move.

## Tasks

- [x] Weighted ridge engine with the equivalence test (scratch timing: 1000 resamples of 2025 take
  about 1 s each for teams and QBs).
- [x] Rank-range computation: `team_rating.bootstrap_team_ratings`,
  `qb_rating.bootstrap_qb_ratings`, and `rating_ranges.summarize_rank_ranges` (`RangeColumns`,
  `TEAM_RANGE_COLUMNS`,
  `QB_RANGE_COLUMNS`); the calibration test passes the pre-set tolerance. Column names it emits:
  `team_rank` / `qb_rank` (published rank), `{rating}_q025` ... `_q975` and `{rank}_q025` ...
  `_q975`, `{rank}_missing_share`, `{rank}_top5_probability`, `{rank}_top10_probability`,
  `{rank}_probabilities` (list).
- [x] Registry: base metrics `team_rank`, `qb_rank`, and their `_missing_share`,
  `_top5_probability`, `_top10_probability`, `_probabilities` columns (shape `rate`, denominator
  bootstrap resamples; ranks are `score` with polarity `lower`); quantile suffix rules `_q025` ...
  `_q975` built from `rating_ranges.RANGE_QUANTILES` in `DEFAULT_SUFFIX_RULES`; catalogs
  regenerated. A test resolves every column `summarize_rank_ranges` emits.
- [x] `main.run_season` writes `{season}_rating_ranges` (`build_team_rating_ranges`: the season
  team fit, every team ranked) and `{season}_qb_rating_ranges` (`build_qb_rating_ranges`: ranked
  among `qb_is_eligible`, with `qb_name` and primary `team` joined in), 1000 resamples, seed 0.
  The season-pipeline tests patch the count to 50 for speed. README data-files list and the
  methodology caveats (`docs/methodology.md`, "Rank Ranges") landed with it.
- [ ] Rank-range API and web views.
- [ ] Ask to run `season` / `pipeline` for the range files.
- [x] WP decisions answered by the maintainer.
- [ ] WP bins in the loader layer (guarded columns), engine refits per threshold, API parameter.
- [ ] Slider (shadcn) with URL state, debounce, exploration label.
- [ ] Pre-register and (after asking) run the WP walk-forward test.
