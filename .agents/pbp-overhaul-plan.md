# PBP Overhaul Implementation Plan

## Living document rules

This file is the active handoff document for the approved PBP-based overhaul. Agents working on this
effort must read it before making substantive changes. Agents must update it in the same change set
whenever scope, status, decisions, blockers, validation results, or next steps change. Keep it
accurate enough for a new agent to resume work without rereading the full chat history. Stale
status, stale decisions, or missing handoff notes are bugs.

## Current status

- Status: release-gate green, with the late-game QB outcome bug fixed and the QB season summary
  schema made explicit enough for downstream interpretation/UI work.
- Last updated: 2026-07-10.
- Code and validation state: `ruff format . --check`, `ruff check .`, `ty check .`, `pyright .`,
  `pytest`, and `python -m nfl_sos_ratings.main` all pass right now.
- Completed so far: the repo already has PBP-first team and QB loaders, equal-weight team and QB
  rating models, simultaneous-adjustment outputs, derived 4QC/GWD features, canonical GSIS-based QB
  identity stitching, authoritative weekly QB/team stat joins, and the QB outcome weighting fix.
- Automated validation now confirms for 2025 that published team columns match
  `load_team_stats(summary_level="week")`, QB attempt-based columns match
  `load_player_stats(summary_level="week")`, there are no duplicate canonical QB season rows, and no
  non-QB passers remain in QB outputs.
- Remaining blocker: no active correctness blocker is known in the remediated QB path, but
  documentation refresh and any remaining interpretation review still need to be completed.
- Next focus: continue the planned visualization/UI work on top of the now-tested CSV contract and
  circle back for any remaining documentation refresh that should accompany the UI rollout.

## Review findings to fix before trusting outputs

1. QB identity stitching is incorrect. PBP passer rows and snap-count rows are being merged by
   `qb_name`, which creates duplicate season rows for the same real player when abbreviated PBP
   names and full snap-count names do not match.
2. `qb_attempts` is not an official pass-attempt field. It is currently behaving like a dropback
   count, which corrupts pass-attempt rates, passer rating inputs, eligibility, and ANY/A.
3. The QB pipeline currently admits non-QB trick passers. That is outside project scope and
   contaminates QB outputs.
4. Team `passing_epa`, `rushing_epa`, and first-down splits do not match nflverse's published team
   stats. Those columns must either match official nflverse values exactly or be removed/renamed;
   the approved direction is to make them match.
5. The QB outcome signal overcounts winning. It currently mixes multiple win-based fields in a way
   that gives team result signal more weight than intended.
6. Automated external 4QC/GWD validation is limited by public-source access. The values can still be
   verified manually in a browser or against another accessible source.

## Current code state

This section is intentionally about what already exists, so the next agent does not waste time
redoing completed scaffolding.

- Team and QB public loaders are already PBP-first.
- Team ratings already emit `SaOR`, `SaDR`, `SaOvR`, `SaCR`, and `SRS`.
- QB ratings already use fixed constants; the old historical calibration loop is gone.
- Simultaneous-adjustment helpers already exist in `simultaneous_adjustment.py`, and `main.py`
  already writes simultaneous team and QB CSV outputs.
- 4QC and GWD derivation already exists in the QB path, but those values still need authoritative
  sanity review after the identity and official-stat fixes.
- The next agent should fix correctness on top of this structure, not rebuild the whole architecture
  from scratch.

## Remediation plan

Phase status summary as of 2026-07-10:

- Phase 1. Canonical identity layer: complete.
- Phase 2. Official attempt-based QB stats: complete.
- Phase 3. Dropback-based QB stats: complete on corrected official attempt totals.
- Phase 4. Exclude non-QB trick passers: complete.
- Phase 5. Official team stat alignment: complete.
- Phase 6. Rebuild team and QB rate surfaces from authoritative totals: complete.
- Phase 7. Fix QB outcome weighting: complete.
- Phase 8. Revalidate simultaneous adjustment on corrected data: complete for current pipeline run.
- Phase 9. Documentation and interpretation refresh: not started.
- Phase 10. Release gate: code/test gates complete for the current methodology; remaining work is
  documentation refresh and follow-on presentation/UI work.

### Phase 1. Canonical identity layer

Goal: eliminate duplicate QB season rows and establish one authoritative player identity across PBP,
snap counts, player stats, and outputs.

Approach:

1. Use GSIS `player_id`/`gsis_id` as the canonical QB identity.
2. Build a crosswalk from `load_players()`, `load_rosters()`, or `load_rosters_weekly()` using
   available `gsis_id` and `pfr_id` columns.
3. Stop grouping QB game rows by `qb_name`; group by canonical player ID plus team/week.
4. Merge snap-count rows onto canonical QB rows through the crosswalk instead of using `qb_name`
   heuristics.
5. Remove season-row duplication caused by fallback IDs such as `snap_player_id` or full names.
6. Add tests proving that the same real player is represented by one QB season row only.

### Phase 2. Official attempt-based QB stats

Goal: distinguish official pass attempts from dropbacks and use the correct denominator for each
stat.

Approach:

1. Treat weekly `load_player_stats(summary_level="week")` as the authoritative source for official
   QB passing totals when those fields exist.
2. Join weekly player stats to QB game rows by canonical player ID, team, and week.
3. Replace PBP-derived placeholders for the following attempt-based fields with official values:
   - `qb_attempts`
   - `qb_completions`
   - `qb_pass_yards`
   - `qb_pass_touchdowns`
   - `qb_interceptions`
   - `qb_sacks`
   - `qb_sack_yards_lost`
   - `qb_passing_epa`
   - `qb_completion_percentage_above_expectation`
4. Keep PBP as the source of record for:
   - `qb_dropbacks`
   - `qb_offense_snaps`
   - 4QC and GWD
   - opponent-allowed and defensive-mirror constructions
5. Rename or retire misleading fields as needed so attempt-based columns are not carrying dropback
   semantics.

Expected result:

- `qb_attempts_total` matches official nflverse attempt totals.
- `qb_yards_per_attempt`, `qb_touchdown_rate`, `qb_interception_rate`, and `qb_passer_rating` use
  true attempts.
- `qb_any_a` uses official attempts plus sacks, not dropbacks plus sacks.

### Phase 3. Dropback-based QB stats

Goal: keep dropback-based metrics separate, correctly named, and internally consistent.

Approach:

1. Keep `qb_dropbacks` and `qb_dropbacks_total` as explicit dropback fields.
2. Compute only the following from dropbacks:
   - `qb_epa_per_dropback`
   - `qb_pass_yards_per_dropback`
   - `qb_td_int_margin_rate`
   - `qb_sack_rate`
3. Recompute ANY/A from official attempt-based totals: `(pass_yards + 20 * pass_tds - 45 *
interceptions - sack_yards_lost) / (attempts + sacks)`.
4. Add source-of-truth tests comparing attempt-based season totals to
   `load_player_stats(summary_level="reg")` and rate formulas to independent calculations.

### Phase 4. Exclude non-QB trick passers

Goal: keep the QB system limited to actual quarterbacks.

Approach:

1. Use the authoritative roster/player position from the identity crosswalk.
2. Filter the QB pipeline to canonical players whose position is QB.
3. Leave trick-pass attempts inside team offense, not QB outputs.
4. Add tests covering known trick-pass cases.

### Phase 5. Official team stat alignment

Goal: ensure published team columns match nflverse's official weekly team data.

Approach:

1. Treat `load_team_stats(summary_level="week")` as authoritative for published team stat columns.
2. Replace custom PBP-derived values for the following official columns with the published
   weekly-team values:
   - `passing_yards`, `rushing_yards`, `total_yards`
   - `passing_tds`, `rushing_tds`
   - `passing_first_downs`, `rushing_first_downs`
   - `passing_epa`, `rushing_epa`
   - `passing_cpoe`
   - `sacks_suffered`, `passing_interceptions`
   - `sack_fumbles_lost`, `rushing_fumbles_lost`
   - defense-only published fields when available
3. Keep PBP as the source for snap counts, per-snap rates, point margin, turnover margin, late-game
   state, and defensive mirrors not available in the official weekly team table.
4. Add tests and live sampled comparisons requiring exact or near-exact match to nflverse official
   team outputs.

### Phase 6. Rebuild team and QB rate surfaces from authoritative totals

Goal: recompute derived rates after the official totals are corrected.

Approach:

1. Recalculate team per-snap rates from authoritative totals and PBP snap counts.
2. Recalculate QB attempt-based and dropback-based rates from the corrected game and season totals.
3. Re-run opponent-profile builders on those corrected fields.

### Phase 7. Fix QB outcome weighting

Goal: keep wins and late-game stats secondary without counting wins multiple times.

Approach:

1. Choose one win-based signal as primary outcome input: `qb_win_pct` or `qb_wins`, but not both.
2. Remove team `win_pct` from the QB outcome blend once QB wins are assigned to the primary QB.
3. Blend 4QC and GWD in as secondary fixed-weight or equal-z-score additions.
4. Document the exact outcome formula and add tests proving that 4QC/GWD can move `QOutcome` without
   re-anchoring the model to team wins multiple times.

### Phase 8. Revalidate simultaneous adjustment on corrected data

Goal: ensure the new simultaneous-adjustment outputs are fed by corrected, canonical inputs.

Approach:

1. Recompute SRS from corrected team-game point margins.
2. Recompute team ridge outputs from corrected rate columns only.
3. Recompute QB ridge outputs from canonical QB rows only.
4. Keep the simultaneous outputs side-by-side with diff-based outputs; do not promote them to
   primary automatically.

### Phase 9. Documentation and interpretation refresh

Goal: align human-facing docs to the corrected methodology.

Approach:

1. Update README field descriptions once the official-vs-derived split is fixed.
2. Document which columns are official nflverse stats and which are PBP-derived.
3. Add a short note about the manual/public-source validation path for 4QC/GWD.

### Phase 10. Release gate

Goal: do not ship the corrections until both code and data sanity checks pass.

Required checks:

1. `ruff format . --check`
2. `ruff check .`
3. `ty check .`
4. `pyright .`
5. `pytest`
6. `python -m nfl_sos_ratings.main`
7. Live comparisons against nflverse source tables:
   - team official columns match `load_team_stats(summary_level="week")`
   - QB official columns match `load_player_stats(summary_level="reg")`
   - no duplicate canonical QB season rows
   - no non-QB passers in QB outputs

## Approved clarifications

- Use PBP as the source of record for both team and QB metrics.
- Use QB dropbacks as the primary QB denominator and the practical qualifier for QB-specific rate
  stats.
- Also assign QB snap counts for every QB game and use snap-count data as an auxiliary signal and
  tiebreaker when determining who played the majority of snaps.
- Replace TD/INT ratio with TD-INT differential.
- Wins remain a secondary QB comparison stat, but wins must not drive any weighting, calibration, or
  model tuning.
- Fourth-quarter comebacks and game-winning drives are lower-priority QB stats, but they should
  still be included.

## Validation requirements

- Follow TDD for each implementation slice.
- For each self-computed metric, document the public formula used and add a correctness test against
  a known or independently aggregated result.
- Before closing the overhaul, run `ruff format .`, `ruff check .`, `ty check .`, `pyright .`, and
  `pytest`.

## Open execution notes

- `player_stats` appears to provide useful weekly aggregates such as `passing_epa`, `passing_cpoe`,
  `passing_interceptions`, `sacks_suffered`, and `passing_first_downs`, but it does not expose
  dropbacks, so PBP remains the source of record for QB denominators.
- `load_player_stats(summary_level="week")` does expose `team`, `opponent_team`, and the official QB
  attempt-based stat surface needed for the remediation plan.
- `load_team_stats(summary_level="week")` does expose the official team EPA and first-down split
  fields that the current published team columns should match exactly.
- `load_players()`, `load_rosters()`, and `load_rosters_weekly()` expose GSIS and PFR identity
  fields that can support the canonical QB crosswalk.
- Snap-count data should be treated as supporting QB participation context, not as a replacement for
  PBP-derived dropbacks.
- nflreadpy does not appear to expose ready-made 4QC or GWD columns in the currently used tables, so
  those are now derived from late-game PBP score state instead.
- The simultaneous-adjustment outputs are now emitted alongside the existing diff-based outputs.
  They are not yet promoted to the primary published ratings, which is intentional and matches the
  approved plan.

## Progress log

- 2026-07-01: plan approved with clarifications on QB dropbacks, snap-count support, TD-INT
  differential, QB wins as a secondary stat, and inclusion of fourth-quarter comebacks and
  game-winning drives.
- 2026-07-01: added `load_pbp_data()`, `load_weekly_player_stats()`, and `load_snap_counts_data()`
  in `data_loader.py` with focused tests in `tests/test_data_loader.py`.
- 2026-07-01: added `compute_team_snap_counts_from_pbp()` in `team_stats.py` with a focused test in
  `tests/test_team_and_opponent_stats.py`.
- 2026-07-01: added `compute_qb_game_volumes_from_pbp()` and `compute_qb_game_stats_from_pbp()` in
  `qb_stats.py` with focused tests in `tests/test_qb_stats.py`.
- 2026-07-01: switched `load_weekly_team_stats()` to a PBP-backed team-game aggregation and
  `load_qb_stats()` to a PBP-plus-snap-count QB-game aggregation.
- 2026-07-01: added team-game `win_value` and `turnover_margin` source fields for the planned
  overall team rating.
- 2026-07-01: added QB season totals and rate metrics for dropbacks, EPA per dropback, ANY/A, sack
  rate, pass yards per dropback, and TD-INT differential style fields.
- 2026-07-01: added matching opponent-allowed QB rate metrics including EPA per dropback, ANY/A,
  sack rate, pass yards per dropback, and TD-INT margin rate.
- 2026-07-01: rewrote `ratings.py` to use equal stat weights and emit `SaOvR`.
- 2026-07-01: rewrote `qb_opponent_stats.py` to key details by QB identity, use primary-QB games,
  deduplicate opponents, and skip unreconstructable QBs.
- 2026-07-01: removed QB win-based calibration search and correlation weighting from
  `qb_ratings.py`; calibration now returns fixed documented constants.
- 2026-07-01: switched the team rating pools to per-snap and rate-like fields and prevented
  raw-total-only frames from driving team ratings.
- 2026-07-01: added `simultaneous_adjustment.py` with SRS, team ridge, QB ridge, and wrapper
  helpers; main now writes simultaneous team and QB adjustment CSVs and joins those columns into the
  combined outputs.
- 2026-07-01: added QB game-level and season-level late-game stats for fourth-quarter comebacks and
  game-winning drives, assigning them only to the primary QB for a team-week.
- 2026-07-01: expanded the QB outcome signal to include wins, 4QC, and GWD as secondary inputs
  alongside `qb_win_pct`.
- 2026-07-01: updated README, AGENTS, and source-module docstrings to reflect the PBP-first
  pipeline, equal-weight models, simultaneous-adjustment outputs, and the formulas used for derived
  metrics.
- 2026-07-01: release-gate checks passed: `ruff format . --check`, `ruff check .`, `ty check .`,
  `pyright .`, and `pytest` all green; overall test coverage is above 90%; `python -m
nfl_sos_ratings.main` completes.
- 2026-07-10: comprehensive methodology review against live outputs and authoritative nflreadpy
  sources found that most published team counting stats already match official sources, but team EPA

- 2026-07-11: UI kickoff started after re-reading this plan and confirming no new methodology
  blocker. Added a tested CSV-backed `nfl_sos_ratings.ui_data` contract layer, a thin FastAPI app in
  `nfl_sos_ratings.ui_api`, and a first-pass React/Vite analyst shell in `ui/web/` for season-aware
  Teams and QBs index views. Focused pytest coverage, a live 2025 contract sanity check, and a
  frontend production build all passed.
  and first-down splits do not.
- 2026-07-10: the review also found that QB season outputs currently have duplicated real players,
  include non-QB trick passers, and use a misnamed `qb_attempts` field that behaves like dropbacks.
- 2026-07-10: the remediation plan was rewritten around canonical GSIS identity, authoritative
  weekly `player_stats` joins for official QB totals, authoritative weekly `team_stats` joins for
  published team fields, and a cleanup of the QB outcome signal.
- 2026-07-10: all code checks still pass, but the current generated outputs should not be treated as
  validated until the remediation phases are completed and the official-source comparisons are
  rerun.
- 2026-07-10: added a canonical GSIS/PFR QB identity crosswalk from `load_players()` and
  `load_rosters_weekly()`, rewired QB game-row merging to use canonical identity instead of
  `qb_name`, restored same-name fallback behavior for direct `qb_stats` helpers without a crosswalk,
  and added focused tests covering name-mismatch merges and non-QB trick passer exclusion.
- 2026-07-10: `load_qb_stats()` now joins authoritative weekly `player_stats` attempt-based QB
  fields onto canonical QB game rows, recomputes dropback-based/attempt-based derived rates from the
  corrected totals, and is covered by focused tests that prove official values override the old PBP
  placeholders.
- 2026-07-10: `load_weekly_team_stats()` now joins authoritative weekly `team_stats` onto the
  PBP-derived team-game frame, replaces the published team offense surface and mirrored allowed
  columns, and recomputes the affected per-snap rates from those official totals.
- 2026-07-10: `_build_outcome_signal()` in `qb_ratings.py` now uses one primary win-based QB input
  (`qb_win_pct`, with `qb_wins` only as fallback) plus 4QC/GWD, and no longer double-counts team
  `win_pct`.
- 2026-07-10: live 2025 validation confirms exact matches for the corrected official team/QB
  columns, zero duplicate canonical QB season rows, zero non-QB passers in QB outputs, green repo
  gates (`ruff format . --check`, `ruff check .`, `ty check .`, `pyright .`, `pytest`), and a
  successful `python -m nfl_sos_ratings.main` pipeline run.
- 2026-07-10: follow-up methodology/output review found one new correctness bug in the derived
  late-game QB metrics. In 2025, four primary-QB game rows carry 4QC/GWD flags despite the team not
  winning, which violates the documented definition. Example: Tyrod Taylor (NYJ) week 3 is credited
  with both a 4QC and GWD in a 27-29 loss. The controlling code path is in `qb_stats.py` around
  `_compute_team_late_game_flags_from_pbp()` and the season aggregation that sums
  `qb_fourth_quarter_comeback` / `qb_game_winning_drive`.
- 2026-07-10: fixed the late-game QB metric bug by deriving team wins from final game scores rather
  than a team's last offensive score-differential snapshot, added a regression test proving a QB
  cannot retain 4QC/GWD flags in a loss, and revalidated that 2025 now has zero primary-QB rows
  with late-game flags on non-wins.
- 2026-07-10: cleaned up the QB season summary schema by replacing ambiguous raw season columns like
  `qb_attempts`, `qb_completions`, and `qb_pass_yards` with explicit per-game columns such as
  `qb_attempts_per_game`, `qb_completions_per_game`, and `qb_pass_yards_per_game`, while adding the
  missing `qb_completions_total` season total.

## Handoff checklist

Before ending a work session on this effort, update this file with:

1. What changed.
2. What remains blocked or unresolved.
3. Which numbered items are complete, in progress, or not started.
4. What the next agent should do first.

Current handoff note:

- What changed: Phases 1-8 were implemented in code for the loader/rating path, with live 2025
  source checks confirming the official team/QB surfaces now match nflreadpy and that QB identity
  duplication/non-QB leakage are gone.
- What remains blocked or unresolved: no active correctness blocker is known in the remediated QB
  loader/rating path; remaining work is docs/interpretation refresh and UI/visualization follow-up.
- Which numbered items are complete, in progress, or not started: 1-8 complete; 9 not started; 10
  code/test gates complete for the current methodology.
- What the next agent should do first: update README/field documentation for the explicit QB season
  schema and the final official-vs-derived split, then start the UI scaffold in
  `.agents/frontend-ui-kickoff-plan.md`.
