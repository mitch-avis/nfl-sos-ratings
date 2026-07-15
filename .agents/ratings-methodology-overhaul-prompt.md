# Next Prompt: Ratings Methodology Overhaul — Stage 2 Kickoff

Copy everything below the divider into a fresh agent session started in
`~/workspace/nfl-sos-ratings`.

---

You are working in the `nfl-sos-ratings` repository at `~/workspace/nfl-sos-ratings`. The sibling
repository `~/workspace/nfl-predictor` is available read-only for reference; do not import from
it at runtime.

## Where the previous sessions left off

Stages 0 and 1 of the ratings methodology overhaul are complete and green (229 tests passing,
full gate suite run). The ridge solver is now the published backbone:

- Team game rows carry `is_home`; team solves estimate home-field advantage; ridge lambda is
  tuned via `tune_ridge_lambda()` instead of fixed; QB solves are dropback-weighted WLS.
- `SaOR`/`SaDR` come from ridge-adjusted passing/rushing EPA components, with `SaDR` oriented
  higher = better defense; `SaOvR` combines them; `SaCR` is an **interim equal two-way blend**
  awaiting this stage's principled weights.
- `QSaOR`/`QSaCR` use ridge-adjusted `adj_qb_epa_per_dropback` (QSaCR is an **interim
  passthrough** awaiting this stage); `QSoS` is the mean faced-defense ridge coefficient
  (descriptive); the paired-diff path and `_reliability_weights` are retired.
- The methodology contract test (`tests/test_rating_methodology_contract.py`) encodes the
  no-outcome-stats invariant and must stay green through all of your work.
- Stage 1b (play-level fitting) is deferred by maintainer decision — see "Recorded decisions"
  below.

## Required reading, in order, before writing any code

1. `AGENTS.md` — house rules: stack (Python 3.14+, Polars only, NumPy for math), TDD, docstrings,
   and the full quality-gate suite (`ruff format`, `ruff check`, `ty check`, `pyright`, `pytest`
   with >90% branch coverage, `markdownlint` on touched repo-owned Markdown).
2. `.agents/ratings-methodology-overhaul-plan.md` — the master plan and single source of truth.
   Stage 2 is your assignment; the Non-negotiable Principles (especially 1 and 3) and the Stage 1
   status notes are your context.
3. `.agents/current-status.md` — handoff notes and recorded maintainer decisions.
4. `tests/test_rating_methodology_contract.py` — the standing invariant.
5. The surfaces you will build on: `nfl_sos_ratings/simultaneous_adjustment.py` (the `adj_*`
   outputs your training rows consume), `nfl_sos_ratings/ratings.py`, `nfl_sos_ratings/qb_ratings.py`,
   `nfl_sos_ratings/pipeline.py` and `nfl_sos_ratings/main.py` (per-season Parquet outputs),
   `nfl_sos_ratings/metrics/catalog.py`, `nfl_sos_ratings/data_loader.py` (QB identity
   crosswalk), and their tests.

## Recorded decisions that bind this session

Record each of these in `.agents/current-status.md` and update the master plan's relevant
sections as part of this session's change set:

1. **Stage 1b is re-sequenced, not abandoned.** Play-level ridge fitting is deferred until after
   Stage 3, where it becomes a measured experiment: fit both backbones and let the walk-forward
   harness decide (held-out margin MAE and year-over-year stability). Update the plan's Stage 1
   status notes and open-decision #1 to say exactly this.
2. **Turnover margin stays descriptive-only** (Stage 0 default confirmed by the maintainer).
   Rationale to record: EPA-based responses already price turnovers, so re-adding turnover margin
   would double-count skill and import fumble-recovery luck. **However**, takeaway-creation rate
   (interceptions + forced fumbles — creation, not recoveries) is approved as a *candidate*
   sub-rating in this stage's team menu; the predictive-validity fit decides whether it earns
   weight. If its fitted weight is negligible, exclude it from the frozen composite and record
   that outcome. Resolve open-decision #2 accordingly.
3. No Elo or ESPN QBR registry work (Stage 3 scope). No standardization-scheme changes beyond
   what weight fitting requires (Stage 4 scope). No garbage-time filtering work.

## Prerequisite: the multi-season back-catalog

Stage 2 fits weights on season-pair training rows, which requires per-season Parquet outputs for
1999–2025 **generated under the Stage 1 ridge backbone**. Any back-catalog generated before the
Stage 1 cutover is stale for this purpose.

- First, check `data/` and `.agents/current-status.md` to determine whether the maintainer has
  already regenerated the back-catalog post-Stage-1.
- If not regenerated: run `python -m nfl_sos_ratings.pipeline` (this is download- and
  compute-heavy; it is acceptable and expected). If the environment cannot complete it, stop,
  leave the repo green, and record precisely what is missing in `.agents/current-status.md`
  rather than fitting weights on stale artifacts.
- Never mix pre- and post-Stage-1 season outputs in one training set. Add a provenance guard if
  feasible (e.g., assert expected Stage 1 columns such as `adj_*` fields exist in every season
  file consumed).

## Your assignment for this session: Stage 2 — Principled composite weights

Execute Stage 2 of the master plan. The plan is authoritative where this summary is terse. Work
in the ordered blocks below; each block lands tests-first.

### Block A — `composite_weights.py` (fitting machinery)

New module `nfl_sos_ratings/composite_weights.py` with pure, testable functions that:

1. Load the per-season ratings/adjusted-stats Parquet back-catalog and build season-pair
   training rows (season *t* predictors → season *t+1* target).
2. Standardize predictors within season *t* before fitting (z-scores over that season's
   qualifying population).
3. Fit weights by linear regression (OLS is acceptable given the small predictor count; ridge
   permitted — if used, tune and record lambda). Fix all random seeds; make the fit reproducible
   from committed code + the back-catalog via one documented command (add it to README's
   development commands or a small CLI entry point).

Targets and matching:

- **Team target:** season *t+1* `SaOvR` (ridge-backed). Match on canonical franchise; apply the
  existing `TEAM_ABBR_ALIASES` handling so relocations/renames (OAK→LV, SD→LAC, STL→LAR) pair
  correctly across seasons.
- **QB target:** season *t+1* `adj_qb_epa_per_dropback`, dropback-weighted in the fit. Match on
  the canonical GSIS `qb_id` via the existing identity crosswalk.
- Drop pairs where the subject is absent in season *t+1*. Document the resulting survivorship
  bias for QBs (fitting on consecutive-season QBs skews toward established starters) as a known,
  accepted limitation in the module docstring and methodology docs.
- Apply the existing QB eligibility gate (`qb_is_eligible`) or a documented dropback floor for
  inclusion in the fit; document whichever is used.

### Block B — Sub-rating menus and the fit

Default menus (deviations require flagging to the maintainer, not improvising):

- **Team (SaCR candidates):** standardized ridge-adjusted offensive passing EPA, offensive
  rushing EPA, defensive passing EPA, defensive rushing EPA — plus the approved
  takeaway-creation-rate candidate (which may need a small, registry-documented metric addition
  if the exact creation-rate column does not yet exist; interceptions + forced fumbles per
  defensive snap from existing fields is the expected construction).
- **QB (QSaCR candidates):** standardized `adj_qb_epa_per_dropback`,
  `adj_qb_completion_percentage_above_expectation`, `adj_qb_sack_rate`,
  `adj_qb_td_int_margin_rate` (all already produced by the QB simultaneous solve). Do not
  include ANY/A or passer rating as candidates — they are redundant compositions of the above;
  note this exclusion in the docs. Handle the pre-2006 CPOE gap by fitting on season pairs where
  CPOE exists and recording the fit window; do not impute CPOE.

Fit diagnostics to compute and record: per-candidate fitted weights, the held-out (leave-one-
season-pair-out or k-fold over seasons) predictive performance of the weighted composite versus
the equal-weight blend of the same standardized inputs, and the fit window.

### Block C — Freeze and publish

- Record the frozen weights in the registry with full provenance: values, target definition, fit
  window, fitting command, and refit policy (refits are deliberate, documented acts — never a
  side effect of a pipeline run). Follow `metrics/catalog.py` header rules for any pool/metric
  changes.
- Add a snapshot test asserting the published weights match the committed frozen values, so any
  silent refit fails CI.
- Update `SaCR` in `ratings.py` and `QSaCR` in `qb_ratings.py` to be the frozen-weight blends of
  the standardized sub-ratings, replacing the interim equal blend / passthrough.
- Extend the methodology contract test if the new definitions create new invariants worth
  locking (e.g., SaCR consumes only registry-listed, ratings-eligible components).

### Block D — Pipeline, docs, and plan upkeep

- Ensure `main.py` writes any new columns through the write-time validation contract.
- Update `README.md` (Current Methodology, Derived Formulas), the registry descriptions, the
  master plan's Stage 2 status notes and open-decisions #1–#3 resolutions, and
  `.agents/current-status.md` — all in the same change set. Run `markdownlint` on touched
  Markdown.

## Acceptance criteria (from the master plan, made concrete)

- Weight fitting is reproducible from committed code + published Parquet history via one
  documented command, with fixed seeds.
- On held-out season pairs, the weighted composite predicts its target at least as well as the
  equal-weight blend of the same standardized inputs — for both teams and QBs. Record the
  numbers in the plan's Stage 2 status notes.
- The takeaway-creation candidate's outcome (weight kept or excluded) is recorded with its
  fitted value.
- Frozen weights + provenance live in the registry; the snapshot test guards them.
- The methodology contract test remains green; full gate suite green; coverage floor maintained.

## If the session runs long

Finish at a clean block boundary (never mid-block), leave the repo fully green, and write
precise resume notes in `.agents/current-status.md`: blocks done, blocks remaining, whether the
back-catalog regeneration completed, and any plan-vs-code discrepancies found. A green repo with
Blocks A–B done and honest notes beats a red repo with all four attempted.

## Definition of done for this session

- Blocks A–D implemented tests-first (or a clean partial per the section above).
- All acceptance criteria above pass.
- Recorded decisions #1–#3 written into the plan and `.agents/current-status.md`.
- A short handoff summary: what changed, the fitted weights and held-out numbers, what Stage 3
  should read first, and any open questions for the maintainer.
