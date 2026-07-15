# Ratings Methodology Overhaul Plan

Master plan for replacing the diff-based, equal-weight rating path with a ridge-backed,
predictive-validity-weighted, era-honest rating system for teams and quarterbacks. This document
is the single source of truth for the overhaul. Agents working on any stage must read it fully,
follow it, and update `.agents/current-status.md` in the same change set as their work.

## Background and motivation

This project exists because win-loss record is a poor indicator of team or QB quality. The
motivating example: the 2025 Patriots went 14-3 against a historically easy schedule (two
opponents with winning records all season), and Drake Maye was an MVP finalist largely on that
record. The project's founding premise, restated in `AGENTS.md`: wins and losses are noisy
outcomes the ratings are meant to see past, not ground truth.

Two external analyses inform this plan and are preserved for reference:

1. An initial methodology report (research-based, written **without** repo access — its
   codebase-mapping claims are superseded, but its methodology survey, public-systems comparison,
   and validation argument remain valid).
2. A repo-grounded revision (written **with** full repo access) containing the authoritative gap
   analysis and staged plan that this document operationalizes.

Both reports converge on the same methodology, which is the target state:

- **Opponent adjustment**: simultaneous ridge regression (offense, defense, and home-field solved
  jointly over the full opponent graph), replacing the diff-based opponent-profile path as the
  backbone of published ratings.
- **Composite weighting**: predictive-validity weights (coefficients that best predict
  _next-season opponent-adjusted performance_), replacing equal weights. This is the construction
  behind nflfastR's DAKOTA metric. Never derive weights from correlation with win rate.
- **Standardization**: within-season z-scores as the published scale, enabling era-honest
  cross-season comparison ("+2.0 = two standard deviations above contemporaries").
- **Validation**: out-of-sample, walk-forward prediction of future game point margins (MAE/RMSE)
  as the primary criterion — the least circular objective standard — plus year-over-year rating
  stability and external references (ESPN QBR, Elo) as secondary checks.

## Non-negotiable principles

1. Win rate, wins, comeback counts, and game-winning drives must never be inputs to any published
   quality rating (SaOR, SaDR, SaOvR, SaCR, QRaw, QSaOR, QSaCR). They may exist only as
   clearly-labeled descriptive/outcome metrics with `ratings_eligible=False` semantics.
2. Elo ratings (team or QB, imported or ported from `nfl-predictor`) are outcome-derived and
   recency-weighted. They are welcome as **descriptive context metrics** and as **validation
   baselines**, never as composite inputs. See "Elo integration" below.
3. Composite weights must be derived from a documented, reproducible procedure whose target is
   never win rate. The fitted weights, the target definition, and the fit window get recorded in
   the registry and surfaced on the methodology page.
4. All standardization for published ratings is within-season. Pooled multi-season reference
   distributions may only appear as separately-named, explicitly-labeled variants.
5. The metrics registry (`nfl_sos_ratings/metrics/`) remains the single source of truth. Any
   rating-definition change lands in the registry and its docs in the same change set.
6. House engineering rules from `AGENTS.md` apply to every stage: TDD (failing test first),
   Polars-only dataframes, docstrings everywhere, `ruff format`, `ruff check`, `ty check`,
   `pyright`, `pytest` with >90% branch coverage on logic-bearing code, and `markdownlint` on
   touched repo-owned Markdown. Never weaken lint/type/coverage settings.

## Cross-repo context

Both projects live side by side on the developer's machine:

- `~/workspace/nfl-sos-ratings` — this repo.
- `~/workspace/nfl-predictor` — sibling repo. Agents may read it for reference. Relevant assets:
  - `scripts/walk_forward_backtest.py` — the conceptual template for Stage 3's validation
    harness (walk-forward evaluation, leakage discipline).
  - `scripts/leakage_audit.py`, `scripts/validate_offline.py` — leakage-prevention patterns.
  - Team Elo and QB Elo implementations and their feature pipeline — source material for the
    Elo descriptive metrics and validation baseline (Stage 3 / Elo integration).

Do not import code across repos at runtime. Port what is needed, with tests, honoring this
repo's stack (Polars, no pandas).

## Findings inventory (from the repo-grounded gap analysis)

These are the confirmed issues the stages below resolve. File/line references are as of the
plan's writing; re-verify before editing.

- **F1 — Ridge outputs are display-only.** `simultaneous_adjustment.py` implements
  `solve_srs`, `solve_team_stat_ridge`, `solve_qb_stat_ridge`, and multi-stat wrappers. `main.py`
  computes and writes `simultaneous_team_adjustments` / `simultaneous_qb_adjustments` and joins
  `adj_*` columns into `combined`, but `compute_ratings()` (`ratings.py`) and
  `compute_qb_ratings()` (`qb_ratings.py`) ignore them entirely; published ratings still come
  from the legacy diff-based equal-weight path.
- **F2 — Premise violations.** `ratings.py::_build_overall_raw` builds SaOvR from
  z(`win_pct`) + z(turnover margin); SaOvR feeds SaCR at `OVERALL_COMPOSITE_WEIGHT = 0.25`.
  `qb_ratings.py` blends a wins/4QC/GWD outcome signal into QSaCR at `_OUTCOME_WEIGHT = 0.75`,
  making team outcomes the heaviest single component of the flagship QB rating.
- **F3 — No home-field advantage term.** `team_stats.py` concatenates home/away perspectives
  without an `is_home` column, so the ridge design matrix cannot include HFA (~1.5–2.5 points in
  the NFL). Schedule data already contains home/away.
- **F4 — Untuned shrinkage; unweighted QB rows.** `ridge_lambda` is hard-coded to `1.0`
  everywhere. QB ridge rows are unweighted (a 4-dropback relief appearance counts the same as a
  45-dropback start). `qb_ratings.py::_reliability_weights` applies a crude linear games-played
  multiplier that would double-shrink once tuned ridge shrinkage is the backbone.
- **F5 — Redundant mirror solves with inverted labels.** `weekly_df` contains both perspectives
  of each game, so solving an offensive response already yields defense ratings for all teams.
  The `team_simultaneous` pool also solves `*_allowed` mirrors, duplicating work — and for
  "allowed" responses the `adj_off_*` / `adj_def_*` output labels invert meaning.
- **F6 — Hidden unequal weighting via pool overlap.** Equal weights over overlapping members
  (total + passing + rushing yards per snap; QB EPA + ANY/A + passer rating + sack rate + sacks)
  silently multi-count stat families. Registry docstrings still claim correlation filtering
  downweights redundancy, but that filtering is disabled (vestigial `min_correlation` params).
- **F7 — Era leakage in cross-season standardization.** `main.py::_build_historical_reference_frame`
  pools all seasons (1999–2025) into one reference distribution; z-scoring against it rewards
  playing in the pass-inflated modern era. Also, CPOE-bearing inputs are unavailable before 2006
  (nflfastR) / 2016 (NGS), so equal-weight composites silently change meaning across eras as
  pools thin out.
- **F8 — QSoS is decorative.** `_SOS_WEIGHT = 0.0`, so QSoS contributes nothing to QSaOR/QSaCR.

## Elo integration (decision)

Team Elo and QB Elo from `nfl-predictor` **should** be added to the stats catalog and metrics
registry, with these constraints:

- New category placement under team/QB context (e.g., "External & Reference Ratings"), each
  metric documented with source, formula/provenance, and polarity.
- `ratings_eligible=False` — the registry's pool validation must reject any attempt to add Elo
  (or ESPN QBR, which has a loader already) to a rating pool. Add a test asserting this.
- Primary purpose: (a) analyst-facing context columns in the UI; (b) the Stage 3 validation
  baseline — the headline acceptance claim is "ridge-backed SaOvR beats a simple Elo baseline on
  held-out margin MAE," which is stronger than beating SRS alone.
- Optional later use: preseason priors for early-week in-season stabilization. Out of scope for
  this overhaul; requires its own plan entry if pursued.

## Staged plan

Stages are ordered by dependency and leverage. Each stage lists scope, files, tests, and
acceptance criteria. A stage is done only when all its criteria pass and
`.agents/current-status.md` is updated.

### Stage 0 — Surgical premise fixes (no new methodology)

Scope:

- Remove `win_pct` (and `win_value`) from `_build_overall_raw` in `ratings.py`. Interim SaOvR
  may be turnover-margin-only or zeroed pending Stage 1's redefinition — prefer redefining
  SaOvR in Stage 1 and, in Stage 0, simply stop feeding win rate into it.
- Set `_OUTCOME_WEIGHT = 0.0` in `qb_ratings.py`; keep computing and publishing `QOutcome` /
  `QOutcome_pct` as descriptive columns. Update registry descriptions to state QOutcome is
  outcome context, not a quality input.
- Delete vestigial `min_correlation` parameters and stale "correlation filtering" docstrings in
  `ratings.py`, `qb_ratings.py`, and `metrics/catalog.py` pool descriptions.
- Remove `*_allowed` response columns from the `team_simultaneous` pool (F5). If any are kept
  deliberately, rename their outputs so labels match polarity, and document why.
- Decide and document turnover-margin treatment: keep as descriptive only, or split takeaway
  creation rate from recovery luck. Default: descriptive only, out of quality ratings.

Files: `nfl_sos_ratings/ratings.py`, `nfl_sos_ratings/qb_ratings.py`,
`nfl_sos_ratings/metrics/catalog.py`, `tests/test_ratings.py`,
`tests/test_qb_opponent_and_ratings.py`, `tests/test_metrics_registry.py`,
`tests/test_simultaneous_adjustment.py`, `README.md` (Derived Formulas + pipeline sections).

Acceptance criteria:

- Grep-level guarantee: no published quality rating consumes `win_pct`, `win_value`, `qb_wins`,
  `qb_win_pct`, `qb_fourth_quarter_comebacks`, or `qb_game_winning_drives`. Enforced by a test.
- Full gate suite green; coverage floor maintained; README and registry docs updated.

### Stage 1 — Promote the ridge to backbone

Status in the current worktree:

- Implemented at the game-level v1 scope.
- The published team ratings now use the ridge-adjusted EPA backbone.
- The published QB ratings now use the ridge-adjusted `adj_qb_epa_per_dropback` backbone, with
  `QSoS` coming from the mean faced-defense ridge coefficient.
- The Stage 1 synthetic-fixture solver tests and the full repo gate suite are green.
- Stage 1b play-level fitting is re-sequenced, not abandoned. After Stage 3 lands, the
  walk-forward harness will compare the game-level and play-level backbones on held-out margin
  MAE and year-over-year stability before any play-level promotion decision.

Scope:

- Add `is_home` to `weekly_df` construction in `team_stats.py` (home half `True`, away half
  `False`), threaded through game logs and the UI contract as appropriate.
- Extend `simultaneous_adjustment.py`:
  - HFA column in the team design matrix (+1 offense-at-home, −1 offense-away, 0 neutral).
  - `tune_ridge_lambda()` — k-fold CV (or closed-form GCV) over a log-spaced grid; pure
    function; per-response tuning permitted. Replace all hard-coded `ridge_lambda=1.0` call
    sites with tuned values (allow explicit override for tests).
  - Dropback-weighted WLS in `solve_qb_stat_ridge` (scale design rows and response by
    √dropbacks).
- Redefine published team ratings in `ratings.py` on the ridge outputs:
  - `SaOR` := ridge offense composite over a small EPA-centric response set
    (`passing_epa_per_offensive_snap`, `rushing_epa_per_offensive_snap`; optionally a success
    rate if added to the pool).
  - `SaDR` := ridge defense composite over the same responses, oriented so higher = better
    defense. Verify and test sign conventions explicitly.
  - `SaOvR` := `SaOR − SaDR` — wait: with SaDR oriented higher-is-better, `SaOvR := SaOR + SaDR`
    on the standardized scale. Whichever orientation is chosen, encode it in one place, document
    it in the registry, and cover it with an orientation test (a team that dominates on both
    sides must rank top-3 in SaOvR on a synthetic fixture).
  - `SaCR` := interim simple average of standardized SaOR/SaDR (equal two-way blend) until
    Stage 2 replaces the weighting. SRS remains published as reference.
- Redefine `QSaOR` := tuned, dropback-weighted ridge-adjusted `qb_epa_per_dropback`. Delete
  `_reliability_weights` (F4). Retire the paired-diff adjustment path from rating computation;
  diff columns may remain as UI-only descriptive surfaces.
- `QSoS` := mean faced-defense coefficient from the QB ridge solve, published as descriptive
  context (the adjustment already lives inside the QB coefficient). Remove `_SOS_WEIGHT`.
- Head-to-head exclusion machinery (`opponent_stats.py`, `qb_opponent_stats.py`) becomes
  legacy for rating purposes: the simultaneous solve conditions on all games at once. Keep the
  opponent-profile outputs for the UI's opponent views; stop feeding them into ratings.
- Optional (may split into Stage 1b): fit ridge on play-level EPA rows instead of game-level
  per-snap aggregates. `data_loader.load_pbp_data` already pulls what is needed; preserve a
  pre-aggregation play frame (`posteam`, `defteam`, `epa`, `qb_epa`, home flag, WP for a
  garbage-time config toggle). Game-level is an acceptable v1; play-level adds sample size,
  situational filtering, and pass/rush splits.

Files: `nfl_sos_ratings/team_stats.py`, `nfl_sos_ratings/simultaneous_adjustment.py`,
`nfl_sos_ratings/ratings.py`, `nfl_sos_ratings/qb_ratings.py`, `nfl_sos_ratings/main.py`,
`nfl_sos_ratings/metrics/catalog.py`, matching tests, `README.md`.

Acceptance criteria:

- Synthetic-fixture tests: ridge with HFA recovers known planted offense/defense/HFA parameters
  within tolerance; dropback weighting shrinks a low-volume QB toward the mean more than a
  high-volume QB with identical per-dropback stats; lambda tuner selects a larger lambda for a
  noisier synthetic season.
- Published ratings Parquet contract updated and validated in `main.py` write-time checks.
- Full gate suite green.

### Stage 2 — Principled composite weights (SaCR, QSaCR)

Status in the current worktree:

- Implemented at the frozen-weight publish scope.
- Added `nfl_sos_ratings/composite_weights.py` with season-pair builders, team OLS and
  dropback-weighted QB WLS fitting, leave-one-season-out diagnostics, frozen snapshots, and a
  reproducible `python -m nfl_sos_ratings.composite_weights` report command.
- Frozen `SaCR` to the normalized Stage 2 weights
  `0.4046255151410425 / 0.20159591248913308 / 0.29457048409623865 / 0.0992080882735857` over
  `adj_off_passing_epa_per_offensive_snap`, `adj_off_rushing_epa_per_offensive_snap`,
  `adj_def_passing_epa_per_offensive_snap`, and `adj_def_rushing_epa_per_offensive_snap`.
- The approved team takeaway-creation candidate
  (`adj_def_takeaway_creation_rate_per_defensive_snap`) produced a small negative fitted weight
  (`-0.04081182634425329`) and was excluded from the frozen composite.
- Frozen `QSaCR` to the normalized Stage 2 weights
  `0.6687790473858877 / 0.21464381898367774 / 0.06725872314827445 / 0.04931841048216012` over
  `adj_qb_epa_per_dropback`, `adj_qb_completion_percentage_above_expectation`,
  `adj_qb_sack_rate`, and `adj_qb_td_int_margin_rate`.
- Held-out leave-one-season-out diagnostics: team weighted RMSE `0.745182` vs equal-weight RMSE
  `0.752449`, team weighted MAE `0.603618` vs equal-weight MAE `0.611575`; QB weighted RMSE
  `0.093348` vs equal-weight RMSE `0.093446`, QB weighted MAE `0.073908` vs equal-weight MAE
  `0.074351`.

Scope:

- New module `nfl_sos_ratings/composite_weights.py`: loads the multi-season ratings back-catalog
  (Parquet outputs, 1999–2025), builds season-pair training rows, and fits weights by regressing
  season _t+1_ targets on season _t_ standardized sub-ratings.
  - Team target: next-season ridge SaOvR. QB target: next-season adjusted `qb_epa_per_dropback`
    (dropback-weighted), matched on canonical QB identity.
  - QB sub-ratings: adjusted EPA/dropback, CPOE, sack rate, TD-INT margin rate; optional rushing
    value term if added. Expect ANY/A and passer rating to receive near-zero weight — that is
    redundancy handling working, not a bug.
- Freeze fitted coefficients as published weights recorded in the registry (values, target
  definition, fit window, refit policy). Weight refits are a deliberate, documented act — never
  a silent side effect of a pipeline run.
- `SaCR` and `QSaCR` become the frozen-weight blends of standardized sub-ratings.

Files: new `nfl_sos_ratings/composite_weights.py` + `tests/test_composite_weights.py`,
`nfl_sos_ratings/ratings.py`, `nfl_sos_ratings/qb_ratings.py`,
`nfl_sos_ratings/metrics/catalog.py`, `README.md`.

Acceptance criteria:

- Weight-fitting is reproducible from committed code + published Parquet history (documented
  command), with a fixed random seed where applicable.
- On held-out season pairs, the weighted composite predicts the target at least as well as the
  equal-weight blend of the same standardized inputs.
- Registry records weights + provenance; methodology docs updated.

### Stage 3 — Validation harness and Elo baseline

Scope:

- New package `nfl_sos_ratings/validation/` with `walk_forward.py`: compute ratings through week
  _n_, predict week _n+1_ margin as `k · (SaOvR_home − SaOvR_away) + HFA` with `k` fit on
  training weeks only; score MAE/RMSE across seasons. Port leakage discipline (not code) from
  `~/workspace/nfl-predictor` (`walk_forward_backtest.py`, `leakage_audit.py`).
- Baselines evaluated in the same harness: SRS, raw (unadjusted) EPA/snap differential, and a
  simple team Elo ported from `nfl-predictor` (Polars-native, tested).
- Secondary checks: year-over-year stability of QSaCR vs. passer rating and ANY/A on the same QB
  population (literature reference points: ~0.48 passer rating, ~0.30 ANY/A; nflWAR-class
  shrinkage metrics reach ~0.60); correlation of QSaCR against the already-loadable ESPN QBR as
  an external sanity reference (divergences are content to inspect, not automatic failures).
- Registry/catalog additions: team Elo, QB Elo, ESPN QBR as descriptive metrics,
  `ratings_eligible=False`, with a registry test asserting pool ineligibility.

Files: new `nfl_sos_ratings/validation/` + tests, `nfl_sos_ratings/metrics/` additions,
`nfl_sos_ratings/data_loader.py` (if Elo requires new inputs), docs.

Acceptance criteria (headline claims of the project):

- Ridge-backed SaOvR beats SRS, raw EPA differential, **and** the Elo baseline on held-out
  walk-forward margin MAE across the multi-season window.
- QSaCR year-over-year stability exceeds passer rating and ANY/A on the same population.
- A committed validation report (generated artifact or doc) records the numbers.

### Stage 4 — Era-honest standardization and methodology transparency

Scope:

- Switch final published standardization to within-season z-scores in both rating modules.
  Retire `_build_historical_reference_frame` from the published path; if all-time-distribution
  variants are kept, publish them under distinct `*_alltime` names with explicit registry
  descriptions.
- Handle the CPOE era boundary explicitly: either exclude CPOE-bearing components before 2006
  with a registry-documented note, or define the comparable-era window as 2006+ for CPOE-bearing
  composites. Decide, document, and test.
- Write the methodology page content (served via the UI or docs): the ratings' definitions, the
  weight provenance, the validation numbers, the era-comparability assumption ("+2.0 = two SDs
  above contemporaries"), and the remaining documented subjective choices (component selection,
  predictive target, garbage-time toggle).

Acceptance criteria:

- Cross-season outputs regenerate cleanly for 1999–2025 under the new standardization.
- No published rating draws on a pooled multi-season reference without an `*_alltime` label.
- Methodology documentation complete; full gate suite green.

## Recorded decisions and remaining open decisions

Record decisions in `.agents/current-status.md` when made.

1. Resolved: game-level ridge stays the published v1 backbone. Stage 1b play-level fitting is
  re-sequenced until after Stage 3, where the walk-forward harness will compare both backbones
  on held-out margin MAE and year-over-year stability.
2. Resolved: turnover margin stays descriptive-only. The approved Stage 2 takeaway-creation
  candidate was tested and excluded after a small negative fitted weight, so the frozen team
  composite stays EPA-only for now.
3. Resolved: the frozen team menu is `adj_off_passing_epa_per_offensive_snap`,
  `adj_off_rushing_epa_per_offensive_snap`, `adj_def_passing_epa_per_offensive_snap`, and
  `adj_def_rushing_epa_per_offensive_snap`; the frozen QB menu is `adj_qb_epa_per_dropback`,
  `adj_qb_completion_percentage_above_expectation`, `adj_qb_sack_rate`, and
  `adj_qb_td_int_margin_rate`.
4. Open: Stage 4 CPOE era-boundary handling (exclude pre-2006 components vs. 2006+ window).
5. Open: garbage-time filter default (config flag exists either way; default off vs. moderate WP
  band).

## Working agreements for agents on this plan

- One stage (or clearly-scoped sub-stage) per session. Do not start a later stage while an
  earlier stage's acceptance criteria are unmet.
- TDD strictly: every behavior change lands as a failing test first. Synthetic fixtures with
  known planted parameters are the required pattern for solver tests.
- Registry-first: rating-definition changes are registry changes; pool membership edits need the
  explicit sign-off noted in `metrics/catalog.py`'s header.
- Update `README.md` (Current Methodology, Derived Formulas) and `.agents/current-status.md` in
  the same change set as behavior changes; run `markdownlint` on touched Markdown.
- Never reintroduce outcome stats into quality ratings — if a change makes a rating correlate
  better with wins, that is not evidence it is better.
