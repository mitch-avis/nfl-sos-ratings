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

### Results of checks A and B (2026-10-04)

Check A, from `nfl-sos-ratings check-additivity --data-dir data --start-season 1999 --end-season
2025` (2000 season-block resamples, seed 0):

- Teams: top offenses' mean residual -0.0156 EPA per play against bottom-tercile defenses (101,601
  plays) and +0.0123 against top-tercile defenses (105,725 plays). Contrast -0.0278, interval
  -0.0403 to -0.0161: entirely below zero, the opposite of the claim.
- Passers: -0.0201 against bottom-tercile defenses (53,575 dropbacks), +0.0362 against top-tercile
  defenses (55,551 dropbacks), 305 rows without a prediction. Contrast -0.0563, interval -0.0738 to
  -0.0405: entirely below zero, the opposite of the claim.
- Strong units fall short of the additive prediction against weak opponents and beat it against
  strong ones, so a soft schedule does not inflate a rating through this mechanism; if anything it
  deflates it. The check does not say why (game script is one untested possibility).

Check B, from `nfl-sos-ratings check-passer --data-dir data --model-season 2025 --name "Drake
Maye"` (2025 fit: adjusted EPA per dropback +0.210, penalty 177.828, sigma 1.535 per dropback):

- 2025 postseason (LAC, HOU, DEN, SEA; 4 games, 142 dropbacks): actual -0.285, predicted +0.112,
  residual -0.398, z -3.09.
- 2026 weeks 1-3 (SEA, PIT, JAX; 3 games, 89 dropbacks): actual -0.225, predicted +0.141, residual
  -0.366, z -2.25.
- Both: 7 games, 231 dropbacks, residual -0.386, z -3.82.
- Every part reads "the 2025 rating overstated the passer against these opponents". The window was
  named after the games were seen, which overstates the evidence, and the 2026 part faces changed
  rosters. Check A says the overstatement is not the additive model's systematic bias.

## Pre-registered in-season penalty test (written 2026-10-04, before any run)

The 2026 build (`nfl-sos-ratings season`, games through week 4) rates every team within 0.04
points of zero (`data/2026_ratings.parquet`): cross-validation picks the grid's largest scrimmage
penalty (100,000). Over 2000-2025 that happens through week 2 in 7 of 26 seasons, week 3 in 3,
week 4 in 1 (2003), and week 5 in none, and full-season penalties are always 316, 562, or 1000
(the penalty table printed by the command below). The maintainer chose to keep the fixed season
penalty for rating histories and to test a better in-season penalty.

- Hypothesis: a team rating fit with the previous season's full-season penalties (scrimmage and
  special teams, each chosen by that season's cross-validation) predicts held-out home margins in
  the first weeks at least as well as one that cross-validates each in-season fit.
- Candidate `TeamRatingPriorPenalty`: `fit_team_ratings` on the pre-week games with the previous
  season's full-season penalties. Incumbent `TeamRating`: `fit_team_ratings` cross-validating
  every snapshot, as published. Both use only games before the predicted week; the candidate also
  uses the previous completed season's games, through its two penalties only.
- Window: seasons 2000-2025 (1999 has no previous season in `data/`), the walk-forward harness's
  prior-only margin projection, prediction weeks 2 and later. Primary metric: MAE over prediction
  weeks 2-5 (snapshots from weeks 1-4, where cross-validation is unstable). Guard: MAE over weeks 6
  and later.
- Statistics: candidate-minus-incumbent MAE with the harness's paired game bootstrap (2000
  resamples, seed 0, 95% percentile interval), primary and guard separately.
- Decision rule: primary interval entirely below zero and guard interval not entirely above zero:
  recommend the candidate. Primary interval including zero and guard not entirely above zero: a
  tie, and the recommendation is the candidate as the simpler option (no in-season tuning, and no
  all-zero ratings). Primary or guard interval entirely above zero: keep cross-validation. Every
  interval that excludes zero is reported, and the decision goes to the maintainer either way.
- Scope if adopted: the published team fit uses the previous season's penalties from 2000 on
  (1999 keeps cross-validation), so the validated estimator stays the published one. That changes
  every published team rating slightly and needs `data/` and the validation report regenerated,
  each asked for separately. The QB fit keeps cross-validation; 2026 QB penalties are interior.
- Command: `nfl-sos-ratings check-in-season-penalty --data-dir data --start-season 2000
  --end-season 2025` (read-only). It also prints each season's cross-validated scrimmage penalty
  through weeks 2-5 and the full season.

### Results of the in-season penalty test (2026-10-04)

From `nfl-sos-ratings check-in-season-penalty --data-dir data --start-season 2000 --end-season
2025` (paired game bootstrap, 2000 resamples, seed 0):

- Primary, prediction weeks 2-5 (1,567 games): `TeamRating` MAE 11.158, `TeamRatingPriorPenalty`
  10.967; candidate minus incumbent -0.191, interval -0.335 to -0.053, entirely below zero.
- Guard, weeks 6 and later (4,737 games): 10.650 against 10.644; -0.005, interval -0.048 to
  +0.037, a tie.
- Reading as written: recommend the candidate. The only interval that excludes zero is the
  primary one, in the candidate's favor. Adoption waits on the maintainer.

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
