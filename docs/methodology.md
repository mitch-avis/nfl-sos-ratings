# Ratings Methodology

This page explains what the team and quarterback ratings measure, how they are computed, what
they leave out, and how they are checked. The pipeline overview and commands are in [README.md];
the generated validation output is [validation-report.md].

## The Question

How good was a team or quarterback, relative to the opponents it actually faced, and to who those
opponents faced? Records answer a different question. A 14-3 team that played a soft schedule and
a 10-7 team that played a brutal one can be equally good, and the ratings are built to show that.

## Building Blocks

**Expected points added (EPA).** nflverse's play-by-play data assigns every play an EPA value: how
much it changed the offense's expected points, given down, distance, field position, and time. A
3-yard run on 3rd and 2 is worth more than a 3-yard run on 3rd and 8. EPA is already measured in
points, so offense, defense, and special teams can be added together without weights.

**Rates, not totals.** Teams run different numbers of plays, so the fit works on EPA per play and
converts back to points per game only at the end, using the league's average plays per game.

**Regular season only.** Playoff games never feed a rating.

## Team Ratings

Each team-game contributes one row: the offense's EPA per scrimmage play (dropbacks, rushes,
kneels, and spikes). One weighted least-squares fit explains every row at once:

```text
EPA per play = league average + offense strength - opposing defense strength + home field
```

Each row is weighted by its play count, which makes the fit equivalent to fitting every play.
Because all teams are solved together, an offense's strength accounts for the defenses it faced,
those defenses' strengths account for every offense they faced, and so on through the whole
schedule. This is the original idea behind the project, rating a team by the opponents it played
and by who those opponents played, carried through to every level at once.

The fit includes a ridge penalty on the team strengths. The penalty pulls estimates toward average
in proportion to how little evidence there is. Over a full season every team has a similar number
of plays, so the pull is similar for all of them; after three or four games it is much stronger,
which is the point.

The penalty's strength is the one five-fold cross-validation (whole games held out) chose for the
whole previous season. Cross-validating on a season's own games works over a full season but not
over a few weeks, where it sometimes picks the largest penalty on offer and rates every team as
average. In the walk-forward check the previous season's choice had a mean absolute error of 10.967
points against 11.158 for prediction weeks 2-5 (difference -0.191, 95% interval -0.335 to -0.053)
and tied from week 6 on (`nfl-sos-ratings check-in-season-penalty --data-dir data --start-season
2000 --end-season 2025`). 1999, the first season of play-by-play, cross-validates its own.

Special teams get the same fit over kicks, punts, returns, field goals, and extra points, with each
team's possession units and coverage units estimated separately and then added together.

The published team columns, all in points per game against an average team on a neutral field:

- `offense_rating`: offensive strength times league-average scrimmage plays per game.
- `defense_rating`: defensive strength on the same scale; positive means the defense prevented
  points.
- `special_teams_rating`: special-teams strength times league-average special-teams plays per
  game.
- `team_rating`: the sum of the three.

## Strength of Schedule

`sos` is the average `team_rating` of the opponents a team played, one entry per game, so a
division rival met twice counts twice. Each opponent is rated from a refit that leaves out every
game involving the team being evaluated. That keeps a team's own results out of its schedule: a
team that beats an opponent badly cannot make that opponent look weaker in its own `sos`. Early in
a season an opponent may not have played anyone else yet; it is left out until it has, and `sos`
stays empty until at least one opponent can be rated.

`nfl-sos-ratings schedules` ranks every completed team-season on `sos`, so one schedule can be
placed in the full history the data covers.

## Quarterback Ratings

Each quarterback-game contributes one row: EPA per dropback, weighted by dropbacks. The fit has
the same shape, with passers in place of offenses:

```text
EPA per dropback = league average + passer strength - opposing defense strength + home field
```

- `adj_qb_epa_per_dropback` is league average plus passer strength. It reads on the same scale as
  raw `qb_epa_per_dropback`, so the two can be compared directly.
- `qb_faced_pass_defense` is the dropback-weighted average strength of the defenses faced, each
  rated from a refit without that quarterback's dropbacks. Positive means tougher defenses.

The adjusted value differs from the raw one for two reasons: the defenses faced, and the ridge pull
toward average. The pull is strongest for backups with few dropbacks, but a full-season starter's
adjusted value also sits noticeably closer to average than his raw value, even after an average
schedule. Read `qb_faced_pass_defense` to see the schedule part on its own.

A quarterback's EPA also reflects his line, receivers, and play-calling. Nothing in public
play-by-play separates those from the passer, so the rating describes the passing offense the
quarterback led, adjusted for the defenses it faced.

## Ratings Through the Season

Each season also gets a rating history: `team_rating` with its three parts, and
`adj_qb_epa_per_dropback`, refit on the games through each week. Every week reuses the season fit's
ridge penalty (the previous season's for teams, the season's own cross-validated one for
quarterbacks), because one or two weeks of games are too few to choose one. With the penalty fixed,
the pull toward average depends only on how much evidence there is: early-week ratings sit close to
average and spread out as games accumulate, and the last week's ratings are the season's. `sos` and
`qb_faced_pass_defense` are not refit week by week. The histories are the `ratings_by_week` and
`qb_ratings_by_week` files.

## Rank Ranges

A rank depends on which games happened to be played. To show how much, each season's games are
redrawn at random with repeats (a game bootstrap: 1000 resamples, each as many games as the season
has) and the ratings are refit on every resample with the season fit's ridge penalties. Each
resample is ranked: teams among all teams, quarterbacks among those who qualify for the full
season (14 pass attempts per game their team has played). The `rating_ranges` and
`qb_rating_ranges` files give, per team or quarterback, the rating and rank at the 2.5th, 10th,
25th, 50th, 75th, 90th, and 97.5th percentiles of the resamples, the chance of a top-5 and a top-10
rank, and the chance of each rank. Team rows also give each unit's published rank (`offense_rank`,
`defense_rank`, `special_teams_rank`) with its rating and rank percentiles from the same resamples.
Unlike the published ratings, the percentiles do not add up: the median of a sum is not the sum of
the medians.

- **Head-to-head chances.** Two overlapping rank ranges cannot say whether one team is better
  than another, because both move together in each resample. The `rating_pairs` and
  `qb_rating_pairs` files compare every ordered pair (qualifying quarterbacks only) across the same
  resamples: how often the first was rated above the second (a tie counts half, so a pair's two
  chances add up to one), and the 2.5th, 50th, and 97.5th percentiles of the first's rating minus
  the second's. Quarterback pairs count only the resamples with both, and `qb_pair_share` says how
  many those were. The chances carry the same caveat as the ranges.
- **What the ranges cover.** Game-to-game sampling noise in the shrunken estimate, nothing more.
  They say nothing about whether the model is right: a bias every resample shares (for example,
  EPA crediting the passer for his receivers) moves the whole range, not its width.
- **Shrinkage.** The ridge pulls every rating toward average, and each resample is pulled the same
  way, so the extreme teams' ranges sit a little toward the middle of the league.
- **Quarterbacks who miss games.** A quarterback with no dropbacks in a resample has no rank in
  it. The `qb_rank_missing_share` column says how often that happened, and the percentiles come
  from the resamples he appears in.
- **Seasons in progress.** With only a few games played, a resample can leave a team out entirely
  (`team_rank_missing_share`), and the ranges are very wide. They narrow as the season fills in.

## Garbage-Time Filter (Exploration View)

Lopsided game states change how teams play: a big lead brings prevent defenses and run-out-the-
clock offense, a big deficit brings desperation passing against them. The analyst app can refit the
ratings without those plays to show how much a ranking depends on them. It is an exploration view:
a walk-forward test (below) found no threshold that predicts better, so the published ratings
always use every play.

- **The rule.** A threshold of X% (0 to 20, in whole percentages) keeps a play when the offense's
  win probability before the snap (nflverse `wp`, from score, clock, and field position, without
  the pregame spread) was at least X% and at most 100% minus X%. Scrimmage and special-teams plays
  are filtered alike. Plays without a win probability (17 rated plays in all, in 1999-2001 and
  2007) are kept at every threshold, so 0% is the published rating.
- **How it is computed.** Every season writes each team-game's plays and EPA, and each
  passer-game's dropbacks and passing EPA, in 1% bins of `min(wp, 1 - wp)` (`team_wp_bins`,
  `qb_wp_bins`). A threshold keeps the bins at or above it, and the ratings are refit on the kept
  plays with the season fit's ridge penalties and per-game scales, as the rating history is. A
  filtered rating is therefore the per-play estimate on the kept plays over a full game's worth of
  plays, on the published scale. `sos` and faced pass defense repeat their head-to-head exclusion
  at every threshold.
- **Quarterbacks.** The filter needs play-level EPA, so filtered quarterback ratings use the
  play-by-play EPA credited to the passer at every threshold, including 0%. The published rating
  uses official weekly passing EPA; in 2025 the two differed by at most 0.002 EPA per dropback
  among qualifying quarterbacks. Filtered changes are measured against the 0% play-level value.
- **Tested: filtering does not improve predictions.** A walk-forward test, its decision rule
  written before it ran, rated teams with the published fit on the plays kept at 5%, 10%, and
  20%, each threshold with its own cross-validated penalties, and predicted the margin of every
  game from week 5 on in 1999-2025 (5,297 games) from the games before it. Mean absolute error was
  10.601 points with every play, 10.623 at 5%, 10.663 at 10%, and 10.733 at 20%. In a paired game
  bootstrap with 98.33% intervals (95% after a Bonferroni adjustment for three comparisons), 5%
  and 10% tied with every play and 20% was worse (+0.133 points, +0.046 to +0.222). Year-over-year
  stability and the quarterback correlation with ESPN QBR also fell as the threshold rose. The
  command is `nfl-sos-ratings check-wp-filter --data-dir data --start-season 1999 --end-season
  2025 --start-week 5`. The exploration view keeps the season fit's penalties instead, so its
  filtered values differ slightly from the tested fits, and rank ranges are not recomputed at
  other thresholds.

## What the Ratings Leave Out

- **Outcomes.** Wins, comebacks, game-winning drives, and turnover margin are published as context
  and never feed a rating. A team's quality is in how it played, which outcomes only partly reflect.
- **Score-based margin.** `SRS` (the classic point-margin rating) is published beside
  `team_rating` as a reference built from final scores rather than plays.
- **Playoffs.** Postseason games are excluded.
- **Other seasons.** Each season is rated on its own games; only the team fit's two ridge penalties
  carry over from the year before.

## Judgment Calls

Every rating rests on choices. These are the ones that matter most here:

- EPA comes from nflverse's expected-points model; the ratings inherit its strengths and blind
  spots.
- Every scrimmage play counts equally, including plays in lopsided games.
- The model is additive: a strong offense is assumed to gain the same amount against every defense.
- Strengths become points per game through the league's average plays per game, so two teams with
  the same per-play strength get the same rating whatever their pace.
- Ridge penalties are chosen by cross-validation rather than by hand: on the previous season's
  games for teams, on the season's own games for quarterbacks.

## How the Ratings Are Checked

The walk-forward check rebuilds `team_rating` each week from that season's earlier games only (with
the previous season's penalties, as published), fits a margin model on earlier predictions only, and
predicts the coming week's home margins. `SRS` and raw EPA margin built from the same games are the
comparisons, and Elo, which carries ratings across seasons and so sees more information, is shown as
a reference. The decision rule was written before the first run: `team_rating` stays the headline
unless its mean absolute error is significantly worse than raw EPA's or SRS's in a paired bootstrap.

The quarterback checks are year-over-year stability beside passer rating and ANY/A, and the
per-season correlation with ESPN QBR, which is a reference, not a target.

### Results

From [validation-report.md], generated by `nfl-sos-ratings validate --data-dir data --start-season
1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md` over 5,297 games
from week 5 on:

- Overall mean absolute error of the predicted home margin: `team_rating` 10.601 points, SRS
  10.658, raw EPA 10.695, and Elo 10.580.
- `team_rating` beats raw EPA: difference -0.095 (95% interval -0.154 to -0.038).
- `team_rating` against SRS is a statistical tie: difference -0.057 (95% interval -0.125 to
  +0.008). The rule's outcome is adopt.
- Elo is not distinguishable from `team_rating` (difference +0.021, interval -0.047 to +0.088),
  even though Elo carries ratings across seasons and the others rate each season from its own games.
- Year-over-year stability: `team_rating` 0.434 and SRS 0.437 (Pearson). For quarterbacks,
  adjusted EPA per dropback is 0.455, a little below passer rating's 0.464 and above ANY/A's
  0.392. Stability is reported, not optimized; the rating measures the season that was played.
- Adjusted EPA per dropback correlates with ESPN QBR at 0.892 (Pearson) and 0.874 (Spearman) on
  average across 2006-2025.

## Worked Example: 2025 New England

These values come from the 2025 files written by `nfl-sos-ratings pipeline`
(`data/2025_ratings.parquet` and `data/2025_qb_ratings.parquet`) and from `nfl-sos-ratings
schedules --team NE --season 2025`.

- Schedule: `sos` of -3.55 points per game, the softest of the 861 team-seasons from 1999 to
  2025, just ahead of the 1999 Rams (-3.49).
- Team: `team_rating` 5.92 points per game, fifth in 2025, with the second-best `offense_rating`
  (5.20). `SRS`, built from final scores, was 5.72.
- Quarterback: Drake Maye's raw EPA per dropback was 0.308. His `qb_faced_pass_defense` was
  -0.025, the fifth-softest of 33 qualifying passers, and his `adj_qb_epa_per_dropback` was 0.210,
  first in 2025, ahead of Matthew Stafford's 0.193.

The ratings separate the two questions the season raised: the schedule was historically soft, and
the passing offense was still the most efficient in the league after accounting for it.

[README.md]: ../README.md
[validation-report.md]: validation-report.md
