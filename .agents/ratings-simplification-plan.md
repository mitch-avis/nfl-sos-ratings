# Ratings Simplification Plan

Active plan for replacing the composite team and QB ratings with one points-based team rating and
one adjusted-EPA QB rating, pruning the surfaces that grew around them, and then adding in-season
2026 support. Branch: `refactor/points-based-ratings`.

## Decisions (maintainer sign-off, 2026-10-03)

The maintainer chose these after the 2026-10-03 audit (findings below):

1. Team headline: one points-based net rating built from opponent-adjusted offense, defense, and
   special-teams EPA, summed in points with no fitted weights. It replaces `SaCR`, `SaOvR`, `SaOR`,
   `SaDR`, and `SaSTR`. `SRS` stays as the score-based reference.
2. QB headline: adjusted EPA per dropback from the simultaneous QB solve, published in EPA units
   with the raw value and the faced-defense value beside it. It replaces `QSaCR`, `QRaw`,
   `QOutcome`, and the `_pct` columns; the stats behind them stay as context columns.
3. Cleanup depth: aggressive. Remove the fitted composites and their weight-fitting command, the
   `_alltime` companions, concluded validation experiments and their named-QB hard-codes, the
   `planned` registry stubs, and dead pools; trim the opponent-profile and `diff_*` surfaces to a
   curated stat set while keeping the head-to-head-excluded opponent profile.
4. 2026: add in-season support after the rework, with a small-sample flag.

## Audit findings that motivated the change

Recorded qualitatively; each item names the code that shows it.

- The registry's headline team rating `SaCR` was never evaluated by the walk-forward harness
  (`validation/walk_forward.py` has no `SaCR` baseline).
- The walk-forward `SaOvR` row was a game-level ridge without special teams
  (`validation/snapshots.build_team_rating_snapshot`), while the published `SaOvR` used the
  play-level ridge plus special teams (`main._build_team_adjustments`). The play-level backbone
  was promoted in `deb4785`, and its re-run under within-season scaling failed the promotion rule
  in `docs/validation-report.md` without the promotion being revisited.
- `SaOvR` summed within-season z-scores of offense, defense, and special teams, which gives
  special teams far more weight than its spread in points.
- `SaCR` weights were fit to next-season `SaOvR` over the full window and applied in-sample.
- `_alltime` companions standardized inputs that were already standardized within season, so each
  season's companion mean is zero and they carry no era information.
- `team_stats.compute_qb_stats_excluding_opponent` averages per QB appearance, not per team game.
- `QRaw` blends seven stats (README said five) and silently skips `qb_sacks`.
- The ridge solves have no intercept, the QB penalty search tops out at its grid edge, and
  cross-validation folds split plays from the same game.

## Team rating specification

- Rows: one row per team-game offense, from regular-season scrimmage plays
  (`pbp_expressions.scrimmage_snap_expr`): offensive plays and offensive EPA sum.
- Model: `mean EPA per play = intercept + offense[team] - defense[opponent] + home_field`, weighted
  by plays (equivalent to the play-level least-squares fit), with the ridge penalty on the offense
  and defense effects only. The penalty is chosen by deterministic k-fold cross-validation with
  folds grouped by game over a grid wide enough that the choice is interior.
- Points: `offense_rating` and `defense_rating` are the effects times the season's league-average
  offensive plays per team-game, so both read as points per game versus an average unit.
- Special teams: one row per team-game with net special-teams EPA (own special-play EPA minus the
  opponent's), solved with the same ridge as `margin = intercept + st[team] - st[opponent] +
  home_field`; `special_teams_rating` is in points per game.
- `team_rating = offense_rating + defense_rating + special_teams_rating`.
- `sos`: for each team, refit all three solves without any game involving that team, then average
  its opponents' `team_rating` over the games it played (played-game weighting, so a twice-played
  division rival counts twice, matching how SRS and the old `sos` weight games). This is the
  head-to-head exclusion rule applied to the simultaneous solve.
- One function computes all of this from frames, and both the published pipeline and every
  walk-forward snapshot call it, so the validated estimator is the published estimator.

## QB rating specification

- Same ridge machinery with an unpenalized intercept, dropback weights, and an interior penalty
  choice: `EPA per dropback = intercept + qb[passer] - defense[opponent]`.
- Published per QB: raw EPA per dropback, the dropback-weighted faced pass-defense effect (EPA per
  dropback, positive means tougher), and the adjusted EPA per dropback (intercept plus the QB
  effect). The adjusted value also includes small-sample shrinkage, which the docs explain.

## Pre-registered team validation

Written before any run of the new estimator.

- Hypothesis: the points-based `team_rating`, built only from games before each cutoff week,
  predicts held-out home margins at least as well as raw EPA margin and SRS built from the same
  games.
- Information set: within-season, pre-cutoff regular-season games only, identical for the
  candidate, `RawEPA`, and `SRS`. `Elo` carries prior seasons, so it is reported as a reference,
  never as a gate.
- Window and metric: seasons 1999-2025, prediction weeks 5 and later, the existing prior-only
  margin projection in `walk_forward.evaluate_feature_rows`, overall MAE as the binding metric,
  paired bootstrap with 2000 resamples and seed 0. Early and late splits are informative.
- Decision rule: publish `team_rating` as the headline if its overall MAE delta against `RawEPA`
  and against `SRS` is not significantly positive (the 95% interval does not lie entirely above
  zero). Parity is enough, because the construct was chosen on principle and the check guards
  against a broken implementation. If it is significantly worse than either, stop, do not tune,
  and bring the result to the maintainer. Every interval that excludes zero, in either direction,
  is reported.
- Informative only: year-over-year Pearson correlation of full-season `team_rating` beside `SRS`.
- Generating command (after the rewrite): `nfl-sos-ratings validate --data-dir data
  --start-season 1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md`.

## Retired metric backlog

The registry's `planned` entries (stats catalogued but never computed) were removed on 2026-10-04
under the aggressive-cleanup decision, which also retired `.agents/metric-expansion-plan.md`
(special-teams detail stats, NGS/PFR/QBR joins, and the remaining QB splits). The last commit
that still had them is `c178744`; this lists the 57 team and 65 QB entries:

```bash
for f in team_metrics qb_metrics; do
  git show c178744:nfl_sos_ratings/metrics/$f.py | python3 -c 'import re, sys; print("\n".join(
    re.findall(r"name=\"(\w+)\",(?:(?!\n    \),).)*?\n        status=\"planned\"",
               sys.stdin.read(), re.S)))'
done
```

## Tasks

- [x] Shared ridge helper with an unpenalized intercept, grouped cross-validation, and a wide grid.
- [x] `team_rating` module (rows, solves, points scaling, leave-one-team-out `sos`) with tests.
- [x] Walk-forward rewrite: candidate, `RawEPA`, `SRS`, `Elo` only; concluded experiments removed.
- [x] Wire the team rating into `main.py`; drop the composite, companion, and old rating columns.
- [x] QB rating rewrite on the shared helper; drop `QSaCR`, `QRaw`, `QOutcome`, percentiles.
- [x] Drop the never-displayed `diff_*`, team-level `opp_qb_*`, and team-level `qb_*` columns
  (which also removes the opponent QB profile's per-appearance averaging bug).
- [x] Prune the registry: old ratings, rating pools, fit provenance, `planned` stubs, and the
  categories they left empty. The docs catalogs still need the matching trim.
- [ ] Update `web/` for the new columns and views.
- [ ] Load PBP once per season, write each output once, enable the nflreadpy filesystem cache.
- [ ] Regenerate `data/` and the validation report (ask first), apply the decision rule.
- [ ] Rewrite `README.md`, `docs/methodology.md`, `AGENTS.md`, and `current-status.md`.
- [ ] In-season 2026 support with a small-sample flag.
