# Next Prompt: Ratings Methodology Overhaul — Stage 3 Kickoff

Copy everything below the divider into a fresh agent session started in
`~/workspace/nfl-sos-ratings`.

---

You are working in the `nfl-sos-ratings` repository at `~/workspace/nfl-sos-ratings`. The sibling
repository `~/workspace/nfl-predictor` is available read-only for reference; do not import from
it at runtime.

## Where the previous sessions left off

Stages 0–2 of the ratings methodology overhaul are complete and green. The ridge solver is the
published backbone (HFA term, tuned lambda, dropback-weighted QB solves). SaCR and QSaCR are
frozen-weight blends fit by season-pair predictive validity (team weights 0.4046/0.2016/0.2946/
0.0992 over adjusted off-pass/off-rush/def-pass/def-rush EPA; QB weights 0.6688/0.2146/0.0673/
0.0493 over adjusted EPA-per-dropback/CPOE/sack-rate/TD-INT-margin), with provenance recorded in
the registry, a snapshot test guarding against silent refits, and a reproducible report command
(`python -m nfl_sos_ratings.composite_weights`). The takeaway-creation candidate was fitted,
came out negligible, and was excluded. The methodology contract test
(`tests/test_rating_methodology_contract.py`) remains the standing invariant.

**A structural fact that shapes this stage:** the pipeline is entirely season-level. `main.py`
and `pipeline.py` compute full-season ratings only; no "ratings through week *n*" capability
exists anywhere. Building it is Block A below, and everything else depends on it.

## Required reading, in order, before writing any code

1. `AGENTS.md` — house rules: stack (Python 3.14+, Polars only, NumPy for math), TDD, docstrings,
   and the full quality-gate suite (`ruff format`, `ruff check`, `ty check`, `pyright`, `pytest`
   with >90% branch coverage, `markdownlint` on touched repo-owned Markdown).
2. `.agents/ratings-methodology-overhaul-plan.md` — the master plan. Stage 3 is your assignment;
   the Non-negotiable Principles, the Elo integration decision, and Stage 1/2 status notes are
   your context.
3. `.agents/current-status.md` — handoff notes and recorded maintainer decisions.
4. `tests/test_rating_methodology_contract.py` — the standing invariant; keep it green.
5. Surfaces you will build on: `nfl_sos_ratings/simultaneous_adjustment.py` (the solvers your
   snapshots re-run), `nfl_sos_ratings/team_stats.py` (weekly rows with `is_home`),
   `nfl_sos_ratings/data_loader.py` (`load_schedule`, ESPN QBR loader),
   `nfl_sos_ratings/composite_weights.py` (fit-window provenance), `nfl_sos_ratings/metrics/`
   (registry), and their tests.
6. Reference-only, in `~/workspace/nfl-predictor`: `scripts/walk_forward_backtest.py` and
   `scripts/leakage_audit.py` for walk-forward and leakage discipline patterns, and the team/QB
   Elo implementations as source material for the Elo baseline. Port ideas and constants, not
   code; everything landed here is Polars-native and tested.

## Your assignment for this session: Stage 3 — Validation harness and Elo baseline

Execute Stage 3 of the master plan. The plan is authoritative where this summary is terse. Work
in the ordered blocks below; each block lands tests-first.

### Block A — Partial-season rating snapshots

New package `nfl_sos_ratings/validation/` with a `snapshots` module providing a pure function
that, given a season's weekly team rows and a cutoff week *n*, returns ridge-backed ratings
computed **only from games in weeks < n** (filter rows, re-run `solve` with lambda tuned on
those rows only, derive SaOR/SaDR/SaOvR exactly as the published path does). Requirements:

- Zero leakage by construction: no input row from week ≥ *n* may influence the snapshot. Add a
  test that perturbs week-*n* data and asserts the snapshot through week *n* is unchanged.
- Reuse the published rating-derivation code rather than duplicating it — refactor shared logic
  out of `ratings.py` into importable pure functions if needed (behavior-preserving; existing
  tests must stay green).
- Handle degenerate early cutoffs gracefully (teams with 0–1 games; the tuned lambda will shrink
  hard toward zero — that is correct behavior, not an error).

### Block B — Walk-forward harness

`nfl_sos_ratings/validation/walk_forward.py`:

- For each season and each prediction week *n* (parameterized `start_week`, default 5, with the
  default recorded and justified in docs — early-week ridge snapshots are near-uniform by
  design), predict each week-*n* game's home margin as
  `k · (SaOvR_home − SaOvR_away) + HFA_points`, where `k` and `HFA_points` are fit on
  **training data only** (prior weeks of the current season and/or prior seasons — document the
  choice; never the week being predicted or later).
- Score MAE and RMSE against actual margins from `load_schedule`, aggregated overall, per
  season, and split by early (weeks < 8) vs. late weeks.
- **Headline metric uses SaOvR, not SaCR.** The frozen SaCR/QSaCR weights were fit on the full
  1999–2025 window, so walk-forward evaluation of SaCR over those seasons has look-ahead in the
  weights. SaOvR has no fitted weights and is clean. SaCR may be evaluated as a secondary line
  with this caveat disclosed verbatim in the report; do not present it as the headline.
- Expose the harness as a reproducible CLI command
  (`python -m nfl_sos_ratings.validation.walk_forward`), mirroring the `composite_weights`
  pattern, with fixed seeds where randomness exists.

### Block C — Baselines in the same harness

All baselines predicted and scored identically to Block B (same weeks, same games, same
train-only fitting of any scale/HFA parameters):

1. **SRS** — computed through week *n* via the existing `solve_srs` on the same filtered rows.
2. **Raw EPA differential** — unadjusted per-snap EPA margin through week *n* (no opponent
   adjustment); this isolates the value of the ridge adjustment itself.
3. **Team Elo** — a simple, standard Elo ported conceptually from `nfl-predictor`: fixed K,
   Elo-points HFA, optional margin-of-victory multiplier, season-boundary mean reversion toward
   1500. Constants are fixed a priori from the reference implementation and documented — do not
   tune Elo on the evaluation window (tuning the baseline on the data it is judged on would be
   unfair in Elo's favor; fixed published constants are the honest comparison). Elo carries
   across seasons by design, which advantages it in early weeks — this is expected; the
   early/late split in Block B's reporting exists to make that visible.

QB Elo is **stretch scope only**: attempt it only if Blocks A–E are complete and validated, and
land it as descriptive-only if you do.

### Block D — Secondary checks and the validation report

- **Year-over-year stability:** Pearson and Spearman correlations of season-*t* vs. season-*t+1*
  values for QSaCR, passer rating, and ANY/A, computed on the identical eligible-QB population
  (existing eligibility gate; consecutive-season QBs matched on canonical `qb_id`). Reference
  points from the literature: passer rating ≈ 0.48, ANY/A ≈ 0.30, shrinkage-based metrics
  ≈ 0.60. Also report SaOvR team stability for context.
- **External reference:** per-season correlation of QSaCR against ESPN QBR (2006+ only; use the
  existing loader). Divergences are content to inspect and note, not automatic failures.
- **Committed validation report:** the CLI writes a Markdown report (e.g.,
  `docs/validation-report.md`, markdownlint-clean) containing the walk-forward table (SaOvR vs.
  SRS vs. raw EPA vs. Elo; overall, per-season, early/late), the stability table, the QBR
  correlations, the SaCR caveat text, and the exact command + parameters that produced it.
  Commit the generated report.

### Block E — Registry additions and docs/plan upkeep

- Register team Elo and ESPN QBR (and QB Elo only if the stretch landed) as descriptive metrics
  under an external/reference category, `ratings_eligible=False`, with source and provenance in
  their descriptions. Add a registry test asserting these metrics are rejected from every rating
  pool. Follow `metrics/catalog.py` header rules; flag-and-stop on anything beyond this scope.
- Publishing Elo as output columns is optional this session; the required deliverable is the
  registry entries + baseline. If published, season-end values only, through the write-time
  column contract in `main.py`.
- Update `README.md` (add a Validation section summarizing the criterion and pointing at the
  report and CLI), the master plan's Stage 3 status notes, and `.agents/current-status.md` — all
  in the same change set. If acceptance results bear on the deferred Stage 1b experiment, note
  that the harness is now the instrument for it.

## Acceptance criteria (headline claims of the project)

- Ridge-backed SaOvR beats SRS, raw EPA differential, **and** the fixed-constant Elo baseline on
  held-out walk-forward margin MAE aggregated across the multi-season window (report the
  early/late split even if Elo wins early weeks — full-window MAE is the criterion).
- QSaCR year-over-year stability exceeds passer rating and ANY/A on the identical population.
- The leakage test (week-*n* perturbation) passes; `k`/HFA fitting provably uses training data
  only (covered by test or audit function).
- The committed validation report exists, is reproducible from the documented command, and
  records all numbers.
- Methodology contract test green; full gate suite green; coverage floor maintained.

**If an acceptance criterion fails, that is a finding, not a formatting problem.** Do not tweak
the harness until the number crosses the line. Record the failure honestly in the report and
`.agents/current-status.md`, investigate obvious implementation bugs (sign conventions, leakage,
join errors) — and if the implementation is sound, stop and surface the result to the maintainer
with your analysis. A true negative here is exactly what the validation framework exists to
catch.

## If the session runs long

Finish at a clean block boundary (never mid-block), leave the repo fully green, and write
precise resume notes in `.agents/current-status.md`: blocks done, blocks remaining, any
plan-vs-code discrepancies, and partial numbers if the harness ran but the report is unwritten.
A green repo with Blocks A–C done and honest notes beats a red repo with all five attempted.

## Definition of done for this session

- Blocks A–E implemented tests-first (or a clean partial per the section above); QB Elo only as
  completed stretch.
- All acceptance criteria pass, or failures are documented per the finding protocol above.
- `docs/validation-report.md` committed and reproducible.
- Registry entries + pool-ineligibility test landed.
- `README.md`, the master plan, and `.agents/current-status.md` updated in the same change set,
  with a short handoff summary: the headline numbers, what Stage 4 should read first, and any
  open questions for the maintainer.
