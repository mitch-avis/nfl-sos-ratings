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

## Pre-registered checks (2026-10-04)

The maintainer questioned whether the additive model ("strength minus opponent strength") credits
teams and passers who feasted on a soft schedule, citing 2025 New England and Drake Maye's last
seven games (four 2025 playoff games and 2026 weeks 1-3). Written before either check runs.

### Check A: additivity of the scrimmage model

- Hypothesis (maintainer's): strong offenses outperform the additive prediction against weak
  defenses and fall short of it against strong defenses, so ratings built on soft schedules run
  high.
- Information set: regular-season team-game rows 1999-2025 (`*_team_game_logs.parquet`). For each
  pair of teams that met, refit the scrimmage ridge (that season's full-fit penalty) without their
  games, predict each of their games from the refit, and take the residual (actual minus predicted
  EPA per play). Offense and defense terciles use the refit's effects against cut points from the
  season's full fit.
- Primary statistic: the play-weighted mean residual for top-tercile offenses against
  bottom-tercile defenses minus the same for top-tercile offenses against top-tercile defenses.
  Additivity predicts zero; the hypothesis predicts a positive value.
- Decision rule: 95% season-block bootstrap interval (2000 resamples, seed 0). Entirely above
  zero: the additive model misses the claimed effect, and a non-additive team model becomes a new
  pre-registered candidate for the walk-forward rule. Including zero: no evidence against
  additivity. Entirely below zero: the opposite of the claim. The same contrast for passers
  against pass defenses (QB-game rows, dropback weights) is reported alongside, read the same way.

### Check B: a passer's later games against his season model

- Question: were a passer's games after the season he was rated on below what that season's
  model predicted for those opponents?
- Information set: the 2025 regular-season QB fit (intercept, passer effect, pass-defense
  effects, home field). Games: the passer's 2025 postseason games (neutral-site Super Bowl) and his
  2026 games to date, each predicted from the opponent's 2025 pass-defense effect. 2026 rosters
  differ from 2025, so the 2026 part is the weaker evidence.
- Statistic: dropback-weighted mean residual (actual minus predicted EPA per dropback) and
  z = residual / (sigma / sqrt(dropbacks)), with sigma estimated from the 2025 regular-season
  residuals of qualifying passers.
- Reading: z at or below -2 says the 2025 rating overstated the passer against these opponents;
  otherwise the games are within the noise the model expects. Any window picked after seeing the
  results overstates the evidence, so the report also gives each part (postseason, 2026) on its
  own.

### Implementation choices for checks A and B (written 2026-10-04, before either ran)

The checks above leave some details open; these are fixed before any result is seen.

- Check A, teams: the penalty is the scrimmage penalty `fit_team_ratings` chooses for the full
  season, and the full fit and every refit use `fit_unit_ridge` on the same scrimmage rows. Cut
  points are the 1/3 and 2/3 quantiles (NumPy linear interpolation) of the full fit's 32 offense
  effects and, separately, its 32 defense effects. A unit is top-tercile when its refit effect is
  above the upper cut and bottom-tercile when below the lower cut; a positive defense effect is a
  better defense, so a top-tercile defense is a strong one. Weights are scrimmage plays.
- Check A, passers: rows are `qb_game_logs` rows with a dropback, the penalty is the one
  `fit_qb_ratings` chooses, and for each pair of teams that met the refit drops every passer row of
  their games. Rows whose passer has no other rows cannot be predicted; they are dropped and
  counted. Passer cut points are the 1/3 and 2/3 quantiles of the full-fit effects of the season's
  qualifying passers (`qb_is_eligible`), defense cut points those of the 32 pass-defense effects.
  Weights are dropbacks.
- Check A, both: the statistic pools 1999-2025; each bootstrap resample draws 27 seasons with
  replacement, and the interval is the 2.5th to 97.5th percentile.
- Check B: postseason rows are built like regular-season rows (play-by-play dropbacks, official
  weekly passing EPA) from postseason play-by-play, snap counts, and weekly player stats. A
  postseason host is home (+1, the visitor -1) and the Super Bowl is neutral (0); 2026 rows use
  `is_home` as the published fit does. sigma squared is the mean of dropbacks times squared
  residual over the 2025 regular-season rows of qualifying passers, residuals from the full 2025
  fit with no degrees-of-freedom correction. z is the dropback-weighted mean residual divided by
  sigma over the square root of total dropbacks, reported for the postseason, 2026, and both.
- Commands: `nfl-sos-ratings check-additivity --data-dir data --start-season 1999 --end-season
  2025` and `nfl-sos-ratings check-passer --data-dir data --model-season 2025 --name "Drake
  Maye"`. Both only read; the passer check downloads the postseason data.

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
  categories they left empty.
- [x] Update `web/` for the new columns and views (rank-based schedule tiers).
- [x] Enable the nflreadpy filesystem cache and write each output once. Loading PBP once per
  season was dropped: nflreadpy's in-process cache already avoids the repeat download.
- [x] Regenerate `data/` and the validation report (approved 2026-10-04). Decision rule: adopt
  (`team_rating` tied with SRS, significantly better than raw EPA; numbers in
  `docs/validation-report.md` and `docs/methodology.md`).
- [x] Rewrite `README.md`, `docs/methodology.md`, `AGENTS.md`, and `current-status.md`; generate
  the stats catalogs from the registry.
- [x] In-season 2026 support: played games only, `SEASON` defaults to 2026, and the web app flags
  a season in progress. Building 2026 into `data/` is still open (needs the go-ahead).
- [x] All-time schedule leaderboard (`nfl-sos-ratings schedules`).
