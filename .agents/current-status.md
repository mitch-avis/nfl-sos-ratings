# Current Project Status

## Purpose

This is the consolidated handoff document for the repo's current state. It replaces completed
one-off plan files as the single place to record:

- what is fully done
- what is still active
- what is deferred
- what the next agent should do first

Update this file in the same change set whenever the repo's active backlog, validation state, or
current-status summary changes.

## Current state

- Status: code-health green, with the PBP-first pipeline, Parquet contract, metrics registry, and
  analyst UI shell implemented. The ratings-methodology workstream is closed: published team and QB
  ratings now use within-season standardization, the pooled reference path is removed from the live
  pipeline, the CPOE-bearing QB headline composites (`QRaw`, `QSaCR`) are intentionally null for
  1999-2005, the public write-up lives in `docs/methodology.md`, and archived stage/block wording is
  isolated to `nfl_sos_ratings/validation/history_strings.py` for validation-report continuity.
- Active correctness blocker: none currently known in the published team/QB pipeline.
- Active methodology blocker: none currently known. The recorded QB null results close that question
  with evidence unless a maintainer explicitly opens a new workstream.
- Latest schedule-strength audit result: no evidence that QB faced-defense coefficients are
  compressed relative to the team pass-defense coefficients. The 2025 Drake Maye versus Tyler
  Shough discrepancy is now explained by construct plus weighting, with one smaller published-path
  bug fixed for multi-team QBs.
- Ratings methodology record: the archived validation history remains in
  `docs/validation-report.md`, while the reader-facing explanation is in `docs/methodology.md`.
- Primary source of truth for metrics: `nfl_sos_ratings/metrics/`.
- Human-readable metric companions: `docs/stats-catalog.md` and `docs/qb-stats-catalog.md`.
- Active workstreams: remaining metric expansion work and UI/detail-page follow-up.

## Recent work summary

The current uncommitted diff reflects five major work themes.

The latest methodology work added a sixth theme.

The current methodology work adds a seventh theme.

The current terminology/provenance cleanup adds an eighth theme.

1. Metrics registry SSOT:
   - added the typed registry package in `nfl_sos_ratings/metrics/`
   - moved rating-pool membership and column metadata into the registry
   - added write-time output-column validation in `main.py`
   - switched published outputs from CSV to Parquet
   - added a direct ESPN QBR Parquet loader
2. PBP/methodology stabilization:
   - canonical QB identity via GSIS/PFR crosswalks
   - official weekly QB/team stat joins from nflverse sources
   - fixed late-game QB outcome attribution
   - clarified explicit QB season-summary naming
   - added season-floor handling for newer upstream sources
3. Metric-surface expansion:
   - added `team_stats_expanded.py`
   - implemented the first large wave of planned team metrics and defensive mirrors
   - added core QB rushing/rate metrics
4. UI/API contract work:
   - added FastAPI metadata and season/game-log endpoints
   - switched the UI data contract to Parquet-backed payloads
   - hydrated frontend metric metadata from `/api/metadata`
   - improved detail pages, grouped-opponent views, compare behavior, and schedule-tier handling
   - replaced the old index/detail metric-family toggles with the six-view control model (`Ratings`,
     `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`, `Opponent Per-Game Rates`, `Opponent
     Per-Play Rates`)
   - moved weekly and unique-opponent detail tables onto the top-of-page control state
   - pinned the detail-page header pane and switched QB detail headers to `QB Detail` plus `QB Name
     - Full Team Name`
5. Docs/tests/plans:
   - added registry, ETL-expansion, and expanded-metric tests
   - refreshed README, UI docs, and plan docs for the Parquet/registry transition
   - built and installed the local wheel successfully with `uv build` and `uv pip install`
6. Ratings methodology Stage 0:
   - neutralized `SaOvR` as a Stage 0 placeholder so team outcome fields no longer feed published
     team quality ratings
   - made `QOutcome` descriptive-only in the published QB quality path by setting the default
     outcome weight to zero
   - removed redundant `*_allowed` mirror responses from the `team_simultaneous` pool
   - removed dead `min_correlation` parameters from the live QB helper signatures
   - added methodology contract tests asserting published quality ratings are invariant to banned
     outcome-only inputs
   - applied the default turnover treatment decision: turnover margin remains descriptive-only and
     out of published quality ratings until a future split/redefinition is approved
7. Ratings methodology Stage 1:
   - added `is_home` to the team game rows and fit home-field advantage directly in the team ridge
     solve
   - added deterministic ridge-lambda tuning and dropback-weighted QB WLS in
     `simultaneous_adjustment.py`
   - promoted the ridge backbone into the published team ratings (`SaOR`, `SaDR`, `SaOvR`, `SaCR`),
     with SaDR explicitly oriented so higher = better defense
   - promoted the ridge backbone into the published QB ratings (`QSaOR`, `QSaCR`) and redefined
     `QSoS` as the mean faced-defense coefficient from the QB ridge solve
   - removed the live games-played reliability multiplier from the published QB quality path
   - threaded the faced-defense ridge schedule column through `main.py` as
     `adj_def_qb_epa_per_dropback_faced`
   - refreshed registry metadata, methodology docs, and Stage 1 contract tests for the new backbone
8. Ratings methodology Stage 2:
   - added `nfl_sos_ratings/composite_weights.py` with season-pair builders, dropback-weighted QB
     WLS fitting, leave-one-season-out diagnostics, frozen-weight snapshots, and a reproducible
     `python -m nfl_sos_ratings.composite_weights` report command
   - replaced the interim `SaCR` equal blend with the frozen Stage 2 team composite over
     ridge-adjusted offensive passing EPA, offensive rushing EPA, defensive passing EPA, and
     defensive rushing EPA using published weights `0.4046 / 0.2016 / 0.2946 / 0.0992`
   - replaced the interim `QSaCR` passthrough with the frozen Stage 2 QB composite over adjusted
     EPA/dropback, adjusted CPOE, adjusted sack rate, and adjusted TD-INT margin rate using
     published weights `0.6688 / 0.2146 / 0.0673 / 0.0493`
   - tested the approved takeaway-creation candidate and excluded it from the frozen team composite
     after a small negative fitted weight (`-0.04081182634425329`)
   - recorded frozen-weight provenance, holdout diagnostics, and refit policy in the registry,
     README, and Stage 2 plan docs
9. Ratings methodology Stage 3:
   - added `nfl_sos_ratings/validation/` with pre-cutoff team snapshot builders, a walk-forward
     backtest runner, fixed-constant Elo / SRS / raw-EPA baselines, stability metrics, QBR
     correlations, and a `python -m nfl_sos_ratings.validation.walk_forward` report command
   - added leakage-guard tests proving future-week perturbations do not affect earlier snapshots and
     that week-*n* projections only fit on prior weeks
   - added the generated `docs/validation-report.md` and documented the current negative finding:
     Elo wins overall MAE (`10.580`) ahead of SRS (`10.658`), RawEPA (`10.695`), and SaOvR
     (`10.701`)
   - recorded the positive secondary checks: QSaCR year-over-year stability clears passer rating and
     ANY/A on the matched QB population (`0.486 / 0.480` vs `0.473 / 0.475` and `0.403 / 0.388`),
     and mean QSaCR-to-QBR correlation is `0.893 / 0.873` across 2006-2025
   - added external/reference metric taxonomy entries and pool-ineligibility tests for team Elo and
     ESPN QBR surfaces

10. Ratings methodology Stage 3b:

    - added paired-bootstrap MAE deltas, weekly MAE curves, season-delta diagnostics, and QB audit
      helpers under `nfl_sos_ratings/validation/`
    - fixed a correctness bug in `main._build_qb_faced_defense_adjustments()` by switching the
      faced-defense mean from an unweighted average to a dropback-weighted average
    - implemented the `T1Weighted` League 1 team experiment: rolling prior-season component weights
      over the published four EPA backbone pieces; overall walk-forward MAE improved to `10.648`
    - implemented the `T2Weighted` League 1 team experiment: `T1Weighted` plus a special-teams
      component built from raw special-play PBP; overall walk-forward MAE improved further to
      `10.630`
    - recorded the key League 1 finding: `T2Weighted` beats RawEPA with a nonzero paired- bootstrap
      edge (`-0.066`, CI `[-0.113, -0.016]`) and beats SaOvR clearly, but its edge over SRS remains
      statistically ambiguous (`-0.028`, CI `[-0.089, 0.036]`)
    - ran the QB revision sweep and did not adopt a backbone change: Q1 fixed team-defense offsets
      reduced the eligible-QB schedule slope, while the best Q2 separate-penalty variant improved it
      only modestly (`0.505 -> 0.557`) and did not justify a published-path refreeze
    - added the Maye/Stafford case study and the season-level QB slope/spread diagnostics to the
      validation report
11. Ratings methodology Stage 3c:
    - pre-registered the final team promotion rule before the deferred T4 run and added
      probability-of-superiority to the paired-bootstrap output tables
    - implemented the required T4 play-level team backbone experiment using offensive-snap EPA rows
      plus the existing special-teams backbone
    - recorded the final team result: `T4Weighted` improved overall MAE to `10.600` and RMSE to
      `13.626`, beating `T2Weighted` (`10.630` / `13.680`), `RawEPA` (`10.695` / `13.771`), and
      Stage 1 `SaOvR` (`10.701` / `13.744`)
    - recorded the SRS parity result that satisfied the fixed rule: `T4Weighted` beat SRS on overall
      MAE/RMSE (`10.600` / `13.626` vs `10.658` / `13.746`) and was not significantly worse than SRS
      overall (`95% CI [-0.118, 0.001]`, `P(T4 <= SRS)=0.9695`)
    - confirmed the stability guard: team year-over-year stability improved from `0.417 / 0.414` for
      Stage 1 `SaOvR` to `0.445 / 0.434` for the promoted T4 history
    - promoted the winning team backbone into the published team ratings path by adding `SaSTR`,
      redefining published `SaOvR` to include offense, defense, and special teams, and refreezing
      `SaCR` to the five-component Stage 3c weights
12. Ratings methodology Stage 3d and Block R:
    - fixed the Stage 3c pooled-reference regression in the team ratings path: historical
      `st_rating` values are now backfilled from `*_simultaneous_team_adjustments.parquet` when
      rebuilding pooled team references, so the published team ratings path no longer mixes
      current-season-only special teams with pooled offense/defense references
    - hardened the multi-season pipeline to fail loudly: season data-step failures now produce an
      end-of-run summary, skip visualization for failed seasons, and exit non-zero instead of
      silently continuing on stale artifacts
    - regenerated the full 1999-2025 back-catalog and verified fresh `*_combined.parquet` and
      `*_ratings.parquet` outputs for every season; published team outputs now carry populated
      `SaSTR` in both the 31-team and 32-team eras
    - added a validation-only `load_playoff_pbp_data()` path plus contract coverage proving
      postseason data stays out of published regular-season rating inputs
    - ran Stage 3d D1: pooled top-half residual slope `0.488` with 95% CI `[0.154, 0.818]` and `19 /
      27` positive seasons (`p = 0.026`), but the bottom-half placebo was similarly positive
      (`0.473`, CI `[0.118, 0.836]`), so the decision-gate reading is `not_supported`
    - skipped D2 by rule because D1 did not read `supported`
    - ran Stage 3d D3: pooled playoff ranking is `QRaw` (`Spearman 0.343`) ahead of `QSaCR`
      (`0.326`), `qb_any_a` (`0.325`), `QSaOR` (`0.323`), `qb_passer_rating` (`0.296`), and the
      Stage 3d split-half metric (`0.257`) across `313` playoff QB-seasons and `21,245` playoff
      dropbacks
13. Terminology and provenance cleanup (current session):
    - added a durable-language policy to `AGENTS.md`, limiting campaign/process vocabulary to
      `.agents/` files and the historical sections of `docs/validation-report.md`
    - added a methodology-language guard test covering registry/API text, published rating column
      names, durable docs, validation CLI labels, and composite-weight CLI output
    - moved composite-fit provenance for `SaCR` and `QSaCR` into structured registry metadata fields
      instead of user-facing prose notes
    - renamed validation baseline labels from experiment codes to descriptive names and scrubbed
      stage/experiment wording from durable registry/docs/CLI/test surfaces
14. Ratings methodology Stage 3e:
    - recorded the QB composite-target closure: published QB composite target and weights remain
      unchanged, and the split-half companion metric is not promoted to a published surface
    - added QB-season bootstrap confidence intervals to the playoff-prediction table in
      `docs/validation-report.md`
    - ran D5 opponent-offense diagnostics: pooled weighted slope `-0.0414` with 95% CI `[-0.1342,
      0.0515]`, `10 / 27` positive seasons, and sign-consistency `p = 0.1239`, so the gate reading
      is `not_supported`
    - ran D6 leverage diagnostics with the pre-registered `0.05-0.95` moderate-WP band: pooled slope
      `-0.0257` with 95% CI `[-0.2016, 0.1527]`, `14 / 27` positive seasons, and `p = 0.500`, so the
      primary leverage-share signal is `not_supported`
    - recorded the 2025 named rows: Drake Maye faced softer opponent offenses (`-0.0265`) and a
      softer schedule context (`0.0295`) with a higher low-leverage share (`0.1206`) than Matthew
      Stafford (`0.0110`, `-0.0066`, `0.0908`), but neither pooled signal cleared its support gate
    - rejected the leverage-filtered companion: year-over-year stability `0.4386 / 0.4222` versus
      published `QSaCR` `0.4978 / 0.4842`, and pooled playoff Spearman `0.3149` versus `QSaOR`
      `0.3226`
15. Ratings methodology Stage 4 and closure:
    - switched the published team and QB scales to within-season standardization and removed the
      pooled historical-reference path from the live pipeline and validation snapshot path
    - documented the explicit 2006+ publication boundary for the CPOE-bearing QB headline
      composites; `QRaw` and `QSaCR` now stay null for 1999-2005 instead of silently renormalizing a
      reduced-input formula
    - added `docs/methodology.md` as the durable public methodology page and linked it from the
      README and the existing glossary UI route
    - moved archived validation-report stage/block wording into
      `nfl_sos_ratings/validation/history_strings.py`, leaving the rest of the source tree under the
      durable-language policy
    - folded the methodology workstream handoff into this file and removed the completed one-off
      methodology plan/prompt files per the repo file-keeping rule
16. Schedule-strength audit and product follow-up:
    - reproduced the 2025 four-QB Maye/Shough/Flacco/McCarthy opponent-quality anchor directly from
      the published Parquet outputs and pinned it in `tests/test_qsos_audit.py`
    - diagnosed one prompt-side data issue before code changes: the hand-entered `SaCR` / `SaDR`
      values were transposed for Tyler Shough, Joe Flacco, and J.J. McCarthy relative to the
      published Parquet files
    - fixed a correctness bug in `main._build_qb_faced_defense_adjustments()`: multi-team QBs were
      being grouped by QB-plus-team even though the published season surface is one row per QB,
      which had understated Joe Flacco's 2025 schedule softness (`QSoS -2.026 -> -2.150` after the
      fix and output refresh)
    - hardened the same schedule helper to normalize opponent aliases before the join and fail
      loudly on any unmatched opponent-defense row instead of silently diluting toward league
      average
    - aligned `validation.diagnostics.build_qb_adjustment_audit_frame()` with the same one-row-per-
      QB grouping rule so the audit math matches the published surface
    - recorded the calibration result: the QB defense coefficients are not compressed versus the
      team pass-defense coefficients (`2025` raw SD ratio `1.633`, pooled `1999-2025` raw SD ratio
      `1.626`), so the under-docking-by-compression hypothesis is not supported
    - recorded the ranking explanation: `QSoS` tracks opponent `SaDR` / pass-defense much more
      closely than opponent `SaCR` / `SRS`, and dropback weighting flips the Maye-vs-Shough
      pass-defense ordering even before the overall-team-quality lens is considered
    - published two new descriptive schedule-context surfaces: team `sos` (played-game mean
      opponent `SaCR`) and QB `faced_opp_SaCR` (equal-game mean opponent `SaCR` over the games the
      QB played)
    - made the QSoS registry/UI description explicit that it is the dropback-weighted pass-defense
      lens from the QB solve, not a general overall-opponent-strength measure

## QB composite-target closure

Recorded decision:

1. The published QB composite target and weights remain unchanged.
2. The split-half companion metric is not promoted to a published surface.
3. `QRaw` remains contextual despite leading the pooled playoff ranking.
4. The only remaining QB methodology follow-up is the pre-registered Stage 3e opponent-context
  batch. That batch is now complete and returned a double-null, so the current composite stands.

## Validation snapshot

Latest recorded green state across the current worktree:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run ty check .`
- `uv run pyright .`
- `uv run pytest`
- `uv run python -m nfl_sos_ratings.main`
- `cd ui/web && npm run build`
- `markdownlint README.md docs/methodology.md docs/stats-catalog.md docs/qb-stats-catalog.md`
  `docs/validation-report.md .agents/current-status.md`

## Completed workstreams

### PBP overhaul

- PBP-first team and QB loaders are in place.
- Official weekly QB and team stat surfaces are joined from nflverse sources.
- Canonical QB identity duplication and non-QB passer leakage are fixed.
- Team and QB rating paths are release-gate green.
- Simultaneous-adjustment outputs are emitted alongside the diff-based outputs.
- No active correctness blocker is currently known in the live methodology path.

### Metrics registry and Parquet migration

- The typed registry is the SSOT for labels, descriptions, category ordering, polarity,
  implementation status, and rating-pool membership.
- Published outputs, UI loaders, and visualization readers use Parquet.
- API metadata and per-column metadata are served from the registry.
- Historical CSV files may still exist in `data/`, but they are legacy artifacts only.

### UI shell

- Season-aware Teams and QBs index pages exist.
- Team and QB detail pages exist with weekly and grouped-opponent views.
- Compare, filtering, sticky columns, and glossary/tooltip support are shipped.
- Registry-backed category ordering and metric metadata are wired through the API and frontend.

### Ratings methodology

- Published team and QB ratings are standardized within their own season.
- The live pipeline no longer uses a pooled all-time reference frame to scale published ratings.
- The CPOE-bearing QB headline composites (`QRaw`, `QSaCR`) publish for 2006+ only; `QSaOR`, `QSoS`,
  and `QOutcome` still publish across the full 1999-2025 span.
- The durable reader-facing explanation lives in `docs/methodology.md`.
- The archived experimental record lives in `docs/validation-report.md`.
- New methodology challenges should follow the same pattern: define a falsifiable hypothesis,
  pre-register the information set and gate, then compare against the published path and the
  relevant baselines.

## Active backlog

### A. Metric expansion still open

Tracked in `.agents/metric-expansion-plan.md`.

Still open:

- E2 special teams implementation
- E5 Tier 2 source integration
  - NGS team passing aggregates
  - PFR pack integration
  - ESPN QBR joins onto the published UI surface
- remaining planned QB metrics that need per-QB PBP rusher attribution
- E8 view-taxonomy alignment
  - complete for the catalog/registry/ETL slice and the shipped Teams/QBs page control refactor
  - keep supporting docs aligned with the six-view model as new surfaces land

Still deferred in the registry:

- `fg_pct_over_expected` model-based work
- `avg_opponent_start_after_kickoff` next-drive linkage
- `time_of_possession` family and `timeouts_used` clock parsing
- `avg_rest_days`, `one_score_game_record`, `pythagorean_win_pct`
- some tackle-accounting and duplicate display-only metrics

### B. Frontend/UI still open

Tracked in `.agents/frontend-ui-kickoff-plan.md`.

Still open:

- Phase 2/3 control-model refactor
  - shipped for the current index/detail tables and detail-page weekly surfaces
  - next follow-up, if needed, is polish on how the selected slice is summarized in secondary UI
    chrome such as overview cards
- Phase 2 comparison UX follow-up
  - pinned side-by-side compare layout
  - only after detail-page workflows settle
- Phase 3 ratings explanation follow-up
  - clearer inline source/type labeling for metrics
- Phase 3b detail-page follow-up
  - continue strengthening weekly views and grouped-opponent ledgers
  - keep the next slice table-first and dependency-light
- Phase 4 high-value charts
  - not started
- Phase 5 design polish
  - still in progress

### C. Documentation maintenance still open

- Keep README, AGENTS, and the two docs catalogs aligned with the current registry and output
  contract.
- Do not let the catalogs drift into a second spec; if registry and docs diverge, fix the docs.

## Deferred but intentionally retained notes

These are not active workstreams, but they are still useful context.

- `load_officials()` remains a possible future source for referee-crew or penalty-context work.
- `load_teams()` remains a possible future source for team metadata and UI chrome.
- When a future nflverse-backed loader has a season floor later than 1999, guard it in
  `data_loader.py` and return a typed empty frame there instead of widening season-loop exception
  handling.

## What the next agent should do first

1. Read this file and confirm whether the task belongs to metric expansion, the analyst UI, or a new
  methodology challenge.
2. If the task challenges published rating methodology, start from `docs/methodology.md` and
  `docs/validation-report.md`, then define a falsifiable protocol and decision gate before making
  code changes.
3. If the task touches planned metrics, continue from `.agents/metric-expansion-plan.md`.
4. If the task touches the analyst UI, continue from `.agents/frontend-ui-kickoff-plan.md`.
5. Keep `docs/stats-catalog.md` and `docs/qb-stats-catalog.md` synchronized with the registry when
   metric taxonomy or implementation status changes.

## File-keeping rule

Do not create a new one-off `.agents/` plan file for completed work. If a workstream is done, fold
its lasting context into this file or into the still-active plan that owns the remaining backlog.
