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
  adjusted EPA per dropback is 0.461, a little below passer rating's 0.473 and above ANY/A's
  0.403. Stability is reported, not optimized; the rating measures the season that was played.
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
