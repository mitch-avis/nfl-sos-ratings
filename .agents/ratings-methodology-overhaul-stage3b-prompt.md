# Next Prompt: Ratings Methodology Overhaul — Stage 3b Kickoff (Diagnosis and Backbone Revision)

You are working in the `nfl-sos-ratings` repository at `~/workspace/nfl-sos-ratings`. The sibling
repository `~/workspace/nfl-predictor` is available read-only for reference; do not import from
it at runtime.

## Where things stand

Stages 0–3 are implemented and green as engineering. Stage 3's walk-forward harness recorded a
**negative headline finding**: overall held-out margin MAE is Elo 10.580, SRS 10.658, RawEPA
10.695, SaOvR 10.701 (`docs/validation-report.md`). Secondary checks passed: QSaCR stability
beats passer rating and ANY/A on the matched population, and QSaCR–QBR correlation averages
0.893 across 2006–2025.

The maintainer has additionally identified a **face-validity failure in the QB adjustment**: in
2025, Drake Maye (QSoS −1.54, a historically soft slate) retains the #1 QSaOR/QSaCR over Matthew
Stafford (QSoS +0.376) despite near-equal raw production — a ~1.9σ faced-defense gap moved their
relative adjusted standing by only ~0.13σ. The working hypothesis is over-shrunk defense
coefficients: prediction-tuned ridge lambda shrinks the opponent-adjustment effects themselves.

This stage is **diagnosis first, revision second, all of it measured**. It is a research stage:
negative experiment results are recorded, not retried until they pass.

## Required reading, in order, before writing any code

1. `AGENTS.md` — house rules and the full quality-gate suite.
2. `.agents/ratings-methodology-overhaul-plan.md` — the master plan; Non-negotiable Principles
   still bind everything here.
3. `.agents/current-status.md` — the recorded Stage 3 negative finding.
4. `docs/validation-report.md` — the numbers you are diagnosing.
5. `tests/test_rating_methodology_contract.py` — the standing invariant; keep it green.
6. The surfaces you will work in: `nfl_sos_ratings/validation/` (snapshots, walk_forward),
   `nfl_sos_ratings/simultaneous_adjustment.py`, `nfl_sos_ratings/ratings.py`,
   `nfl_sos_ratings/qb_ratings.py`, `nfl_sos_ratings/composite_weights.py`, and their tests.

## Pre-registered criterion revision (record before running anything)

The Stage 3 acceptance criterion compared systems with unequal information sets: Elo carries
prior-season knowledge across season boundaries; SaOvR/SRS/RawEPA are within-season only. The
project's stated purpose is *retrospective within-season evaluation*, so the criterion is revised
into two information-matched leagues. Record this revision — including the fact that it follows
a failed test and the rationale — in the master plan and `.agents/current-status.md` **before**
running experiments, so it is a documented re-registration, not a post-hoc rationalization:

- **League 1 (within-season information; the binding headline):** the team backbone (SaOvR or
  its revised successor) must beat SRS and RawEPA on full-window held-out MAE, with a
  statistically meaningful margin per the bootstrap procedure in Block A.
- **League 2 (prior-carrying information; informative, not binding):** the backbone equipped
  with a simple previous-season prior (forecast-only device; published season ratings remain
  full-season, within-season) is compared against fixed-constant Elo. Beating Elo here is the
  stretch goal; losing narrowly is a recordable, acceptable outcome.
- The original criterion and its failure remain in the validation report's history section —
  do not delete or reword the recorded negative finding.

## Block A — Diagnostics (no methodology changes yet)

1. **Significance on the existing table.** Add paired-bootstrap confidence intervals (resample
   games; fixed seed) for pairwise MAE deltas between all four systems, overall and by split.
   Report which gaps are distinguishable from zero. Add this to the harness output permanently.
2. **Where does SaOvR lose to SRS?** Per-week-number MAE curves for SaOvR vs. SRS vs. Elo;
   per-season deltas; and a decomposition check on whether games involving strong
   special-teams/kicking teams drive the gap (SaOvR sees only pass+rush EPA; SRS sees full
   margins).
3. **QB adjustment-magnitude audit.** For each season: (a) verify the identity
   `adj_qb_epa_per_dropback − raw qb_epa_per_dropback ≈ −(dropback-weighted mean faced-defense
   coefficient)` in EPA units — a large violation is a bug, stop and fix; (b) plot/tabulate the
   slope of (adjusted − raw) against mean faced-defense quality — the slope should be ≈ −1 in
   common units if adjustments act at full strength, and its actual value quantifies the
   under-adjustment; (c) include the 2025 Maye/Stafford pair as a named case study with all
   intermediate quantities.
4. **Defense-coefficient shrinkage audit.** Compare the spread (SD) of defense coefficients from
   the QB solve against the spread of defense-vs-pass effects from the team solve on the same
   seasons. A much smaller QB-solve spread confirms the over-shrinkage hypothesis.

Commit the diagnostics (code + a short findings section appended to the validation report)
before starting Block B.

## Block B — Pre-registered team-backbone experiments

Run each through the unchanged harness; adopt only what improves League 1 held-out MAE without
degrading team stability; record every result, adopted or not, in the validation report:

- **T1 — Nested, leakage-free component weighting.** Inside the walk-forward, fit the
  pass/rush × off/def component weights on *prior seasons only* (rolling), so the headline team
  metric gets principled weights with zero look-ahead. This supersedes the equal blend that
  handicapped SaOvR in Stage 3.
- **T2 — Special-teams component.** Add a ridge-adjusted special-teams EPA-per-snap response
  (kicking/punting/returns from existing PBP fields) as a third unit alongside offense/defense,
  entering the composite via T1's nested weighting.
- **T3 — Early-week prior (League 2 only).** Previous-season backbone rating times a reversion
  factor fit on prior seasons, blended with the in-season snapshot by games played. Forecast-only
  device; must not alter published season ratings.
- **T4 — Stage 1b play-level backbone.** Fit the team ridge on play-level EPA rows (the deferred
  experiment; the harness is now the instrument). Compare game-level vs. play-level on League 1
  MAE and stability; adopt the winner per the recorded Stage 1b decision protocol.

Order: T1 → T2 → T4, with T3 independent (League 2). If context runs short, T1 and T2 are the
priority.

## Block C — QB backbone revision (gated on Block A findings)

If A3/A4 confirm under-adjustment (expected):

- **Q1 — Two-stage adjustment (primary candidate).** Estimate defense-vs-pass effects from the
  *team* ridge (large samples, tuned lambda), then hold them **fixed as offsets** when solving
  QB effects, penalizing only QB coefficients (and HFA). The opponent adjustment is no longer
  shrunk by the QB model's lambda.
- **Q2 — Separate penalties (fallback).** If Q1 is impractical, keep the joint solve but give
  defense coefficients a separate, smaller (or CV-tuned separately) penalty than QB
  coefficients.
- Acceptance for adoption: the A3 adjustment-slope moves materially toward −1; QSaCR
  year-over-year stability does not degrade below the Stage 3 recorded values (0.486/0.480);
  QBR correlation stays in a comparable range; and the Maye/Stafford case study shows the
  faced-defense gap actually expressed in the adjusted ratings. If adopted, refreeze the Stage 2
  QSaCR weights via the documented refit procedure (this is a deliberate, recorded refit — the
  snapshot test must be updated in the same change set, never silently).
- If A3/A4 do **not** confirm under-adjustment, stop, record what the audit shows instead, and
  surface it to the maintainer before changing the QB backbone.

## Block D — Report, plan, and docs

Regenerate `docs/validation-report.md` with: the original Stage 3 finding preserved as history,
the criterion revision and its rationale, all Block A diagnostics, every Block B/C experiment
result (adopted and rejected), the new League 1/League 2 tables with bootstrap CIs, and the
Maye/Stafford case study. Update `README.md`, the master plan (Stage 3b status), and
`.agents/current-status.md` in the same change set. Full gate suite green throughout; the
methodology contract test never breaks.

## Rules of engagement (this stage's version of the finding protocol)

- **No metric-hacking.** Experiments are defined above before results are seen. Adoption
  decisions use held-out numbers plus the stated stability guards — never "run variants until
  one wins." If none of T1–T4 lifts the backbone past SRS with a meaningful margin, that is a
  major recorded finding to surface to the maintainer, not a defeat to be massaged.
- Published-rating semantics are unchanged by forecast-only devices (T3, nested weighting):
  end-of-season published ratings remain full-season, within-season quantities.
- Commit at block boundaries; leave the repo green at every commit; precise resume notes in
  `.agents/current-status.md` if the session ends mid-stage.

## Definition of done for this session

- Block A diagnostics committed with findings (including the adjustment-slope number and the
  Maye/Stafford quantities).
- T1 and T2 run and recorded (T4/T3 if context allows); Block C run if gated open, with the
  adoption decision and any refreeze recorded.
- Regenerated validation report with history preserved; plan and status updated.
- A short handoff summary: the League 1 result vs. SRS with CI, the QB adjustment slope before
  and after any revision, what remains for Stage 4, and open questions for the maintainer.
