# NFL Team Stats Catalog

A human-readable companion to the typed metrics registry under `nfl_sos_ratings/metrics/`.

The registry is the authoritative source of truth for published metric definitions, categories,
polarity, and implementation status.

This document exists to explain the team-side metric surface, sourcing rules, and display taxonomy
in narrative form for maintainers and analysts.

Scope: regular season, team grain. The QB-grain companion is [qb-stats-catalog.md].

## Table of Contents

- [How to Read This Document](#how-to-read-this-document)
- [Sourcing Policy: PBP-First](#sourcing-policy-pbp-first)
- [Data Source Inventory](#data-source-inventory)
- [Duplication Policy and Ratings Safeguards](#duplication-policy-and-ratings-safeguards)
- [Category Taxonomy](#category-taxonomy)
- [Catalog: External & Reference Ratings](#catalog-external--reference-ratings)
- [Catalog: Overall](#catalog-overall)
- [Catalog: Offense](#catalog-offense)
- [Catalog: Defense](#catalog-defense)
- [Catalog: Special Teams](#catalog-special-teams)
- [Situational Splits (Future Display Axis)](#situational-splits-future-display-axis)
- [Secondary-Source Metric Packs (Tier 2)](#secondary-source-metric-packs-tier-2)
- [Appendix A: PBP Field Map](#appendix-a-pbp-field-map)

## How to Read This Document

### Metric shapes

The UI has six primary views:

- `Ratings`
- `Raw Total Stats`
- `Per-Game Rates`
- `Per-Play Rates`
- `Opponent Per-Game Rates`
- `Opponent Per-Play Rates`

`Ratings` is the schedule-adjusted view and stands outside the stat taxonomy below. The other five
are the stat views.

Every stat metric has one of three shapes, and that shape determines which of the five stat views it
appears in and how:

- **count** — a summable total (yards, touchdowns, sacks). Appears in all five views: raw total,
  divided by games, divided by the relevant play denominator, and both opponent-averaged views.
- **rate** — an intrinsic ratio with its own built-in denominator (completion %, yards per carry,
  third-down %, EPA per play). Identical in the Totals / Per-Game / Per-Play views; only the
  team-vs-opponent axis varies. Never divide a rate by games or plays.
- **avg** — a per-event mean (CPOE, aDOT, average drive start). Behaves like **rate** in views;
  aggregate across games by re-averaging over events, not by averaging weekly values.

### Natural denominators and naming

"Per-Play" means "per natural unit for that subcategory," and the unit must be encoded in the column
name and stated in the metric description. Approved denominator suffixes:

| Suffix | Denominator | Applies to |
| --- | --- | --- |
| `_per_game` | games played | any count |
| `_per_offensive_snap` / `_per_defensive_snap` | scrimmage snaps | Total offense / defense |
| `_per_dropback` | dropbacks (attempts + sacks + scrambles) | Passing |
| `_per_attempt` | official pass attempts | conventional passing rates (YPA, TD%, INT%) |
| `_per_carry` | official carries | Rushing |
| `_per_drive` | offensive possessions (`fixed_drive`) | Drives, Scoring, Turnovers |
| `_per_series` | first-down series | Downs & Conversions |
| `_per_target` / `_per_reception` | targets / receptions | Receiving |
| `_per_return`, `_per_punt`, `_per_kickoff`, `_per_fg_att` | the ST event | Special Teams |

Rules:

1. A rate column must carry its denominator suffix (or be a `%`-style named rate like
   `third_down_pct` whose denominator is unambiguous and documented).
2. The UI "Per-Play" view renders each subcategory with its natural denominator; the table header
   and tooltip must state which denominator is in use (via `metricMetadata.ts`).
3. Every metric registry entry records `denominator` explicitly so the ETL and frontend cannot
   disagree.
4. `_per_drive` and `_per_series` are not separate views or special cases. They are the
  Drives/Scoring/Turnovers and Downs & Conversions variants of the same `Per-Play Rates` view.

## Views and Denominators

| View | Denominator | Additional Notes |
| --- | --- | --- |
| Ratings | none | Schedule-adjusted outputs (`SaCR`, `SaOR`, `SaDR`, `SaSTR`, `SaOvR`, `SRS`). `SaCR` is the published weighted team composite over ridge-adjusted passing EPA, rushing EPA, defense, and special teams. This is a view, not a category. |
| Raw Stat Totals | none | For **count** metrics only. **rate** and **avg** metrics keep the same value they show elsewhere. |
| Per-Game Rates | games played | For **count** metrics this is the per-game form. **rate** and **avg** metrics are unchanged. |
| Per-Play Rates | play-specific denominators for the subcategory (dropback, attempt, carry, drive, series, etc.) | For **count** metrics this uses the stat's natural denominator suffix; **rate** and **avg** metrics are unchanged. |
| Opponent Per-Game Rates | games played, measured from the opponent profile | Opponent-profile columns are `opp_`-prefixed. Count metrics keep their per-game opponent form here; intrinsic **rate** and **avg** metrics are unchanged opponent-context values. |
| Opponent Per-Play Rates | play-specific denominators from the opponent profile | Opponent-profile columns are `opp_`-prefixed and keep the same natural-denominator suffixes as the matching subject-side per-play stats (for example `opp_points_per_offensive_snap`, `opp_yards_per_drive`). |

### Source codes and tiers

| Code | Loader | Grain | Coverage | Tier |
| --- | --- | --- | --- | --- |
| PBP | `load_pbp()` | play | 1999+ | 1 |
| TS | `load_team_stats(summary_level="week")` | team-week | 1999+ | 1 (validation) |
| PLS | `load_player_stats(summary_level="week")` | player-week | 1999+ | 1 |
| SCH | `load_schedules()` | game | 1999+ | 1 |
| SNP | `load_snap_counts()` | player-game | 2012+ | 1 (narrow role) |
| NGS | `load_nextgen_stats()` | player-week/season | 2016+ | 2 |
| PFR | `load_pfr_advstats()` | player-week/season | 2018+ (sub-eras vary) | 2 |
| QBR | ESPN QBR nflverse release assets (no `nflreadpy` loader yet; `data_loader.load_espn_qbr()` downloads the Parquet directly) | QB-week/season | 2006+ | 2 |
| D | Derived / computed by this project | — | limited by inputs | — |

Source notation in the tables: `PBP` means computed from PBP; `PBP +TS` means computed from PBP
**and** also published in `team_stats`, so the TS value is available as an automated
cross-validation check. `SCH` scores remain authoritative for game outcomes.

Removed from the catalog per project decision (unreliable, discontinued, or out of scope): FTN
charting, participation, injuries, depth charts, draft picks, combine, contracts, trades.

Starters/primary QBs are identified from PBP usage instead of depth charts.

Coverage caveats:

- PBP EP/EPA and WP models cover 1999+, but `cp`/`cpoe`, xYAC, `xpass`/`pass_oe` start **2006**.
- The 2024 dynamic-kickoff rule change breaks kickoff/return comparability across eras.
- `order_sequence`, `special_teams_play`, `st_play_type`, `time_of_day` exist 2011+ only.
- PFR sub-era coverage varies by column — see the [Tier 2
  packs](#secondary-source-metric-packs-tier-2).

## Sourcing Policy: PBP-First

**Decision: PBP is the primary source for every stat it can produce — raw and derived, totals and
rates, offense and defense. `team_stats` is kept only as an automated validation surface.**

Empirical basis (2025 season, measured in this repo's environment):

- `load_pbp([2025])` → 48,771 plays × 372 cols in ~1s (cached; cold download is a one-time,
  per-season cost). `load_team_stats()` → ~0.3s. The full team-passing aggregation over a season of
  PBP takes **~45ms** in Polars. Pipeline cost is dominated by download/IO, not aggregation — the
  performance concern about PBP-first is **not** a real constraint.
- Exactness was verified team-by-team for 2025 REG: PBP-derived completions, gross passing yards,
  sacks, sack yards, passing TDs, and INTs match `team_stats` **exactly** (32/32 teams, zero diffs).
  Official **attempts** require one nuance: exclude two-point conversion tries (`pass_attempt −
  sack`, where `two_point_attempt == 0`) — with that filter, attempts also match 32/32 exactly.
- Practical consequence: per-game and per-play variants never need TS at all; they are the same PBP
  aggregation grouped by game and divided by PBP-derived snaps.

Validation harness: keep a small pipeline check that recomputes the official surface from PBP and
asserts equality against TS per team-week, so any upstream PBP encoding change is caught
immediately. PLS remains in use where player attribution is genuinely easier from official playstats
(defense-only add-ons: tackles, QB hits, passes defended, safeties) — these are also derivable from
PBP attribution columns if PLS is ever dropped.

### Snap counts: narrowed role

Verified for 2025 (544 team-weeks):

- **Team/unit snap counts need no external source** — they are already derived from PBP
  (`compute_team_snap_counts_from_pbp`). Nothing changes.
- **Primary-QB identification from PBP dropbacks agrees with snap counts in 541/544 team-weeks
  (99.4%)**. The 3 disagreements are games where the starter left early or was pulled (e.g., a
  backup out-threw an injured starter). These are policy questions, not data questions: decide
  whether "primary" means *most snaps* or *most dropbacks* and encode the tie-break.
- What PBP **cannot** provide: true per-player snap participation (PBP only names players involved
  in the action). If QB snap totals / snap-share are wanted as displayed stats, SNP (2012+) is the
  only source.

Recommendation: keep `load_snap_counts()` solely for (a) QB snap/snap-share display fields and (b)
the split-game tie-break, with PBP dropbacks as the 1999–2011 fallback. Do not use SNP for anything
team-level.

### Passing yards: gross vs. net (verified)

nflverse `passing_yards` is **gross** (sack yardage is *not* deducted), and team `receiving_yards`
**equals** team `passing_yards` exactly (verified 32/32 teams, 2025). The NFL.com/PFR team-stat
convention of "net passing yards" deducts sack yardage. Both belong in the catalog with unambiguous
names:

- `passing_yards` — gross (nflverse convention). Identity: team receiving yards ≡ gross passing
  yards (modulo rare laterals).
- `net_passing_yards` — `passing_yards − sack_yards_lost` (NFL.com team convention).
- **Sign gotcha (verified):** `team_stats.sack_yards_lost` is stored as a **negative** number (e.g.,
  CLE 2025 = −345). When deriving from PBP, `yards_gained` on sack plays is also negative. Formulas
  in this catalog treat `sack_yards_lost` as a *positive magnitude*; the ETL must normalize the sign
  at ingest and the validation harness must assert the convention.

## Data Source Inventory

| Loader | Role in this project | Key contents |
| --- | --- | --- |
| `load_pbp()` | **Primary source for all derivable stats** | 372 fields: play typing, yardage, downs/series, drives, EP/EPA, WP/WPA, CP/CPOE, xYAC, xpass, kicking/returns, penalties, fumbles, player attribution, environment |
| `load_team_stats()` | Validation surface for the official stat families | Official passing/rushing/receiving splits, defense box stats, returns, full FG/PAT suite |
| `load_player_stats()` | Defense-only team add-ons; QB official truth; player pages | Same stat families at player grain plus `pacr`, `racr`, `target_share`, `wopr` |
| `load_schedules()` | Official scores, results, game context | Scores, rest days, Vegas lines, weather, coaches, refs, starting QBs, stadium |
| `load_snap_counts()` | QB snap-share display; split-game primary-QB tie-break (2012+) | Offense/defense/ST snaps and percentages per player-game |
| `load_players()` / `load_rosters()` / `load_rosters_weekly()` | Identity crosswalk — `gsis_id` canonical; `espn_id`, `pfr_id` verified present | Names, IDs, positions, status |
| `load_nextgen_stats()` | Tracking-derived QB/rusher/receiver metrics (tested — see Tier 2) | Time to throw, aggressiveness, xCOMP%, CPOE, RYOE, xYAC, separation |
| `load_pfr_advstats()` | Charting-adjacent advanced splits (tested — see Tier 2) | Pressures, blitzes, drops, bad throws, YBC/YAC, broken/missed tackles, coverage |
| ESPN QBR | QB quality input + display (tested — see Tier 2) | Adjusted/raw QBR, points added, EPA splits per QB. No `nflreadpy` loader exists; download the Parquet assets directly: [season level](https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_season_level.parquet), [week level](https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.parquet) (wrapped by `data_loader.load_espn_qbr()`) |

## Duplication Policy and Ratings Safeguards

Display duplication is deliberate; rating-input duplication is forbidden. The enforcement design:

1. Every metric registry entry carries two fields:
   - `ratings_eligible: bool` — whether the metric may ever enter a rating pool.
   - `duplicate_of: str | None` — the canonical metric this one restates (same numbers under a
     different name or trivial transform).
2. **Rating pools are built from explicit allowlists only** (already the practice in `ratings.py`),
   and a pipeline assertion rejects any pool that contains a metric whose `duplicate_of` canonical
   form — or any other pool member's canonical form — collides. Concretely: `passing_yards` and
   `receiving_yards` must never co-exist in a pool; `point_differential` must not join a pool
   already containing both `points_for` and `points_allowed`; a rate must not join alongside the
   exact ratio of two pool members.
3. Display surfaces (Teams page, Team Details, glossary) are free to show duplicates in as many
   subcategories as is useful; the `duplicate_of` field also powers a UI hint ("same as X").

Known display-only duplicate families: the entire Offense–Receiving and Defense–Receiving
subcategories (see below); `point_differential`, `turnover_margin`, and other Overall differentials;
every `*_pct`/`*_per_*` whose numerator and denominator are both catalog members.

## Category Taxonomy

Within the five stat views above, both team pages — the league-wide Teams table and the individual
Team Detail page — use this category and subcategory structure.

`Ratings` stays separate as its own primary view and is not part of this taxonomy:

```text
Teams
├── Overall                    (no subcategories)
├── Offense
    ├── Total
    ├── Passing
    ├── Rushing
    ├── Receiving              (display-only mirror of Passing — see subcategory note)
    ├── Scoring
    ├── Downs & Conversions
    ├── Drives & Field Position
    ├── Turnovers (Giveaways)
    └── Penalties
├── Defense
    ├── Total
    ├── Passing
    ├── Rushing
    ├── Receiving              (display-only mirror of Defense–Passing)
    ├── Scoring
    ├── Downs & Conversions
    ├── Drives & Field Position
    ├── Turnovers (Takeaways)
    ├── Pressure & Playmaking
    └── Penalties
├── Special Teams
    ├── Kicking (FG & XP)
    ├── Kickoffs & Coverage
    ├── Kick Returns
    ├── Punting & Coverage
    ├── Punt Returns
    └── ST Scoring & Blocks
```

Rationale for departures from a flat ESPN-style layout: drive efficiency gets its own subcategory
(highly display-worthy, doesn't fit Downs); defense gets Pressure & Playmaking (sacks/TFL/PD have no
offensive mirror); ST kick/punt units split coverage from returns; Receiving is retained on both
sides **for display purposes only** with every row marked `duplicate_of` its passing counterpart
(see safeguards above).

The `Ratings` view now has two descriptive blocks:

- the project's own schedule-adjusted ratings (`SaCR`, `SaOR`, `SaDR`, `SaSTR`, `SaOvR`, `SRS`)
- external/reference ratings such as the fixed-constant team Elo validation baseline

Those external/reference ratings are `ratings_eligible=False` in the registry and never feed the
published project ratings.

## Catalog: External & Reference Ratings

These metrics live in the `Ratings` view for analyst context and validation baselines. They are not
part of the five stat-view taxonomies below and are never allowed into any rating pool.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `team_elo` | Fixed-constant team Elo benchmark with preseason regression toward 1500, used as a walk-forward validation baseline and as descriptive context | Elo | score | 1999 |

## Catalog: Overall

Season identity and outcome summary. No subcategories.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `games_played` | Regular-season games played | SCH | count | 1999 |
| `wins`, `losses`, `ties` | Official results | SCH | count | 1999 |
| `win_pct` | `(wins + 0.5·ties) / games` | D(SCH) | rate | 1999 |
| `win_value` | 1 / 0.5 / 0 per game (existing pipeline field) | D(SCH) | avg | 1999 |
| `points_for` | Points scored | SCH | count | 1999 |
| `points_allowed` | Points allowed | SCH | count | 1999 |
| `point_differential` | `points_for − points_allowed` | D(SCH) | count | 1999 |
| `pythagorean_win_pct` | `PF^2.37 / (PF^2.37 + PA^2.37)` | D(SCH) | rate | 1999 |
| `total_yards_differential` | Yards gained − yards allowed | PBP | count | 1999 |
| `turnover_margin` | Takeaways − giveaways | PBP | count | 1999 |
| `penalty_differential` | Opponent penalties − own penalties | PBP | count | 1999 |
| `penalty_yards_differential` | Opponent penalty yards − own | PBP | count | 1999 |
| `epa_margin_per_play` | Off EPA/play − def EPA/play allowed | PBP | rate | 1999 |
| `success_rate_margin` | Off success rate − def success rate allowed | PBP | rate | 1999 |
| `time_of_possession_pct` | Share of game clock with possession | PBP | rate | 1999 |
| `avg_scoring_margin` | Point differential per game | D(SCH) | rate | 1999 |
| `one_score_game_record` | W-L-T in games decided by ≤ 8 points | D(SCH) | count | 1999 |
| `avg_rest_days` | Mean of schedule rest fields | SCH | avg | 1999 |
| `srs`, `sos` | Simple Rating System and its schedule component | D(SCH) | rate | 1999 |

All differentials are `duplicate_of` their components for ratings purposes. The app's own ratings
(`SaCR`, `SaOR`, `SaDR`, `SaSTR`, `SaOvR`, `SRS`) belong to the separate `Ratings` view rather than
to the `Overall` / `Offense` / `Defense` / `Special Teams` stat taxonomy. External/reference ratings
such as `team_elo` live in that same top-level view as descriptive-only benchmarks. Within that
view, `SaCR` is the published weighted composite, `SaSTR` is the published special-teams backbone
surface, and `SaOR` / `SaDR` / `SaOvR` remain the ridge-backbone component views.

## Catalog: Offense

### Offense — Total

Per-play denominator: offensive scrimmage snaps (`qb_dropback + rush + qb_kneel + qb_spike`, the
existing pipeline definition).

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `offensive_snaps` | Scrimmage plays run | PBP | count | 1999 |
| `total_yards` | Passing + rushing yards | PBP +TS | count | 1999 |
| `yards_per_offensive_snap` | `total_yards / offensive_snaps` | PBP | rate | 1999 |
| `first_downs` | Rush + pass + penalty first downs | PBP | count | 1999 |
| `scrimmage_tds` | Passing + rushing TDs | PBP +TS | count | 1999 |
| `offensive_epa` | Sum of EPA on scrimmage plays | PBP | count | 1999 |
| `epa_per_offensive_snap` | `offensive_epa / offensive_snaps` | PBP | rate | 1999 |
| `success_rate` | Share of plays with EPA > 0 (`success`) | PBP | rate | 1999 |
| `explosive_play_rate` | Plays of 20+ pass yds or 10+ rush yds / snaps | PBP | rate | 1999 |
| `time_of_possession` | Sum of drive possession time | PBP | count | 1999 |
| `seconds_per_offensive_snap` | Pace: possession seconds / snaps | PBP | rate | 1999 |
| `no_huddle_rate` | `no_huddle` plays / snaps | PBP | rate | 1999 |
| `shotgun_rate` | `shotgun` plays / snaps | PBP | rate | 1999 |
| `pass_rate` | Dropbacks / scrimmage snaps | PBP | rate | 1999 |
| `early_down_pass_rate` | Dropbacks on 1st/2nd down / early-down snaps | PBP | rate | 1999 |
| `pass_rate_over_expected` | Mean `pass_oe` (PROE) | PBP | avg | 2006 |
| `offensive_wpa` | Sum of WPA on scrimmage plays | PBP | count | 1999 |
| `timeouts_used` | Timeouts charged | PBP +TS | count | 1999 |

### Offense — Passing

Per-play denominator: dropbacks. Conventional per-attempt rates flagged by suffix.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `attempts` | Official attempts: `pass_attempt − sack`, excl. 2-pt tries (verified exact vs TS) | PBP +TS | count | 1999 |
| `completions` | `complete_pass` (verified exact vs TS) | PBP +TS | count | 1999 |
| `completion_pct` | `completions / attempts` | PBP | rate | 1999 |
| `passing_yards` | Gross passing yards (≡ receiving yards; verified) | PBP +TS | count | 1999 |
| `net_passing_yards` | `passing_yards − sack_yards_lost` (magnitude) | PBP | count | 1999 |
| `passing_tds` | Passing touchdowns | PBP +TS | count | 1999 |
| `passing_interceptions` | INTs thrown | PBP +TS | count | 1999 |
| `dropbacks` | `qb_dropback` plays | PBP | count | 1999 |
| `sacks_suffered` | Sacks taken | PBP +TS | count | 1999 |
| `sack_yards_lost` | Yards lost to sacks (positive magnitude; TS stores negative) | PBP +TS | count | 1999 |
| `sack_rate_per_dropback` | `sacks_suffered / dropbacks` | PBP | rate | 1999 |
| `scrambles` | `qb_scramble` plays | PBP | count | 1999 |
| `scramble_yards` | Yards on scrambles | PBP | count | 1999 |
| `passing_first_downs` | First downs via pass | PBP +TS | count | 1999 |
| `passing_air_yards` | Air yards, incl. incompletions | PBP +TS | count | 2006 |
| `passing_yards_after_catch` | YAC on completions | PBP +TS | count | 1999 |
| `air_yards_per_attempt` | aDOT: `passing_air_yards / attempts` | PBP | rate | 2006 |
| `yac_per_completion` | `passing_yards_after_catch / completions` | PBP | rate | 1999 |
| `yards_per_attempt` | `passing_yards / attempts` | PBP | rate | 1999 |
| `net_yards_per_attempt` | `(pass_yds − sack_yds) / (att + sacks)` | PBP | rate | 1999 |
| `adjusted_net_yards_per_attempt` | ANY/A: `(yds + 20·TD − 45·INT − sack_yds) / (att + sacks)` | PBP | rate | 1999 |
| `yards_per_dropback` | `passing_yards / dropbacks` | PBP | rate | 1999 |
| `passing_epa` | EPA on dropbacks (`qb_epa` basis, per TS convention) | PBP +TS | count | 1999 |
| `epa_per_dropback` | `passing_epa / dropbacks` | PBP | rate | 1999 |
| `pass_success_rate` | Success rate on dropbacks | PBP | rate | 1999 |
| `passing_cpoe` | Mean CPOE on attempts | PBP +TS | avg | 2006 |
| `team_passer_rating` | NFL passer-rating formula on team totals | PBP | rate | 1999 |
| `passing_td_rate_per_attempt` | `passing_tds / attempts` | PBP | rate | 1999 |
| `int_rate_per_attempt` | `passing_interceptions / attempts` | PBP | rate | 1999 |
| `explosive_pass_rate` | 20+ yard completions / dropbacks | PBP | rate | 1999 |
| `deep_attempt_rate` | `pass_length == "deep"` attempts / attempts | PBP | rate | 1999 |
| `longest_pass` | Max completed pass yards | PBP | count | 1999 |
| `sack_fumbles` / `sack_fumbles_lost` | Strip-sack fumbles / lost | PBP +TS | count | 1999 |
| `passing_2pt_conversions` | Successful 2-pt passes | PBP +TS | count | 1999 |
| `air_epa_total` / `yac_epa_total` | EPA split: through the air vs. after catch | PBP | count | 1999 |
| `xyac_per_completion` | Expected YAC per completion (model) | PBP | avg | 2006 |
| `yac_over_expected_per_completion` | Actual − expected YAC per completion | PBP | avg | 2006 |

### Offense — Rushing

Per-play denominator: carries.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `carries` | Official rush attempts (incl. scrambles, kneels) | PBP +TS | count | 1999 |
| `designed_carries` | Carries excluding scrambles and kneels | PBP | count | 1999 |
| `rushing_yards` | Rushing yards | PBP +TS | count | 1999 |
| `yards_per_carry` | `rushing_yards / carries` | PBP | rate | 1999 |
| `rushing_tds` | Rushing touchdowns | PBP +TS | count | 1999 |
| `rushing_first_downs` | First downs via rush | PBP +TS | count | 1999 |
| `rushing_epa` | EPA on rush plays | PBP +TS | count | 1999 |
| `epa_per_carry` | `rushing_epa / carries` | PBP | rate | 1999 |
| `rush_success_rate` | Success rate on designed rushes | PBP | rate | 1999 |
| `explosive_rush_rate` | 10+ yard rushes / carries | PBP | rate | 1999 |
| `stuffed_run_rate` | Rushes for ≤ 0 yards / carries | PBP | rate | 1999 |
| `rushing_fumbles` / `rushing_fumbles_lost` | Fumbles on rushes / lost | PBP +TS | count | 1999 |
| `longest_rush` | Max rush yards | PBP | count | 1999 |
| `rushing_2pt_conversions` | Successful 2-pt rushes | PBP +TS | count | 1999 |
| `run_location_splits` | Carries/yards by left–middle–right, gap | PBP | count | 1999 |

Tier 2 add-ons (see packs): yards before/after contact per carry, broken tackles (PFR 2018+); RYOE,
8-in-box rate — team aggregation caveats apply (NGS 2016+).

### Offense — Receiving (display-only mirror)

Retained for display purposes on both Teams and Team Details pages. **Every row here is
`duplicate_of` a Passing metric and is `ratings_eligible: false`.** Verified identity: team
`receiving_yards` ≡ gross `passing_yards`; receptions ≡ completions; targets ≡ attempts. The
genuinely receiving-flavored fields (drops, separation) also appear here with their real sources.
Full per-player receiving belongs on the Team Details player breakdown (`receptions`, `targets`,
`receiving_yards`, `receiving_tds`, air-yards family, `racr`, `target_share`, `air_yards_share`,
`wopr` from PLS).

| Column | Definition / formula | Source | Shape | Since | Duplicate of |
| --- | --- | --- | --- | --- | --- |
| `targets` | Team pass attempts | PBP +TS | count | 1999 | `attempts` |
| `receptions` | Team completions | PBP +TS | count | 1999 | `completions` |
| `receiving_yards` | Gross passing yards | PBP +TS | count | 1999 | `passing_yards` |
| `receiving_tds` | Passing TDs | PBP +TS | count | 1999 | `passing_tds` |
| `receiving_air_yards` | Air yards | PBP +TS | count | 2006 | `passing_air_yards` |
| `receiving_yards_after_catch` | YAC | PBP +TS | count | 1999 | `passing_yards_after_catch` |
| `receiving_first_downs` | First downs via reception | PBP +TS | count | 1999 | `passing_first_downs` |
| `receiving_fumbles` / `receiving_fumbles_lost` | Fumbles after catch / lost | PBP +TS | count | 1999 | — (unique) |
| `catch_rate` | Receptions / targets | PBP | rate | 1999 | `completion_pct` |
| `drops` / `drop_rate` | Dropped passes / per target | PFR | count/rate | 2018 | — (unique) |
| `avg_separation`, `avg_cushion` | Receiver openness (tracking) | NGS | avg | 2016 | — (unique; coverage caveat) |

### Offense — Scoring

Per-play denominator: drives.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `points_for` | Total points (duplicated from Overall) | SCH | count | 1999 |
| `total_tds` | All TDs scored (off + def + ST) | PBP | count | 1999 |
| `offensive_tds` | `passing_tds + rushing_tds` | PBP | count | 1999 |
| `offensive_points` | `6·off TDs + 2·(2-pt made)` + kicking points | PBP | count | 1999 |
| `points_per_drive` | Offensive points / offensive drives | PBP | rate | 1999 |
| `td_rate_per_drive` | TD drives / drives | PBP | rate | 1999 |
| `red_zone_trips` | Drives reaching opponent 20 (`drive_inside20`) | PBP | count | 1999 |
| `red_zone_td_pct` | TD trips / red-zone trips | PBP | rate | 1999 |
| `points_per_red_zone_trip` | Points on RZ drives / trips | PBP | rate | 1999 |
| `goal_to_go_td_pct` | TDs / goal-to-go series | PBP | rate | 1999 |
| `two_pt_attempts` / `two_pt_conversions` | 2-pt tries / makes | PBP +TS | count | 1999 |
| `two_pt_conversion_rate` | Makes / tries | PBP | rate | 1999 |

### Offense — Downs & Conversions

Denominators: series for series conversion; third/fourth-down attempts for down conversions.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `first_downs` | Total first downs gained | PBP | count | 1999 |
| `first_downs_rush` / `first_downs_pass` / `first_downs_penalty` | First downs by type | PBP | count | 1999 |
| `third_down_attempts` / `third_down_conversions` | 3rd-down tries / conversions | PBP | count | 1999 |
| `third_down_pct` | Conversions / attempts | PBP | rate | 1999 |
| `third_down_avg_distance` | Mean `ydstogo` on 3rd down | PBP | avg | 1999 |
| `fourth_down_attempts` / `fourth_down_conversions` | 4th-down tries / conversions | PBP | count | 1999 |
| `fourth_down_pct` | Conversions / attempts | PBP | rate | 1999 |
| `fourth_down_go_rate` | Go-for-it plays / 4th downs faced | PBP | rate | 1999 |
| `fourth_down_aggressiveness` | Go rate on 4th-and-short (≤ 2), or go rate vs. league expectation given down-distance-field position | PBP | rate | 1999 |
| `series` | Offensive series run | PBP | count | 1999 |
| `series_conversion_rate` | `series_success` / series (first down or TD) | PBP | rate | 1999 |
| `three_and_out_rate` | Three-and-out drives / drives | PBP | rate | 1999 |
| `turnovers_on_downs` | Failed 4th downs surrendering possession | PBP | count | 1999 |

### Offense — Drives & Field Position

Built from PBP `fixed_drive` / `fixed_drive_result` and the `drive_*` fields.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `drives` | Offensive possessions (`fixed_drive`) | PBP | count | 1999 |
| `yards_per_drive` | Net drive yards / drives | PBP | rate | 1999 |
| `plays_per_drive` | `drive_play_count` mean | PBP | avg | 1999 |
| `time_per_drive` | `drive_time_of_possession` mean | PBP | avg | 1999 |
| `first_downs_per_drive` | `drive_first_downs` mean | PBP | avg | 1999 |
| `score_pct_per_drive` | Drives ending in any score / drives | PBP | rate | 1999 |
| `punt_pct_per_drive` | Drives ending in punt / drives | PBP | rate | 1999 |
| `turnover_pct_per_drive` | Drives ending in giveaway / drives | PBP | rate | 1999 |
| `avg_starting_field_position` | Mean drive-start yardline (own X) | PBP | avg | 1999 |
| `long_field_score_pct` | Score % on drives starting inside own 25 | PBP | rate | 1999 |
| `drive_penalty_yards` | `drive_yards_penalized` sum | PBP | count | 1999 |

### Offense — Turnovers (Giveaways)

Denominators: offensive snaps or drives.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `giveaways` | INTs thrown + fumbles lost | PBP | count | 1999 |
| `passing_interceptions` | INTs thrown (duplicated from Passing) | PBP +TS | count | 1999 |
| `fumbles` | All offensive fumbles (kept + lost) | PBP +TS | count | 1999 |
| `fumbles_lost` | Sack + rushing + receiving fumbles lost | PBP +TS | count | 1999 |
| `giveaway_rate_per_offensive_snap` | Giveaways / offensive snaps | PBP | rate | 1999 |
| `giveaways_per_drive` | Giveaways / drives | PBP | rate | 1999 |
| `turnover_epa` | EPA on giveaway plays (cost of turnovers) | PBP | count | 1999 |

### Offense — Penalties

PBP `penalty_team`, `penalty_type`, `penalty_yards` allow offense/defense/ST attribution and
pre-snap vs. post-snap classification; TS `penalties` / `penalty_yards` are all-units totals.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `penalties` | All penalties committed (all units) | PBP +TS | count | 1999 |
| `penalty_yards` | Yards assessed against team | PBP +TS | count | 1999 |
| `offensive_penalties` / `offensive_penalty_yards` | Penalties while on offense | PBP | count | 1999 |
| `presnap_penalty_rate` | False starts, delay, etc. / snaps | PBP | rate | 1999 |
| `penalty_rate_per_offensive_snap` | Offensive penalties / snaps | PBP | rate | 1999 |
| `penalty_type_breakdown` | Counts by `penalty_type` | PBP | count | 1999 |

## Catalog: Defense

Every Offense subcategory has a defensive mirror built by grouping PBP on `defteam` instead of
`posteam` (the existing `_allowed` convention), with identical formulas, shapes, coverage, and the
same natural denominators (opponent dropbacks, opponent carries, opponent drives):

- **Total (allowed)** — `total_yards_allowed`, `yards_per_defensive_snap_allowed`,
  `epa_per_defensive_snap_allowed`, `success_rate_allowed`, `explosive_play_rate_allowed`,
  `first_downs_allowed`, snaps faced, pace faced, opponent PROE faced.
- **Passing (allowed)** — full mirror: attempts/completions faced, `completion_pct_allowed`,
  `passing_yards_allowed` (gross) and `net_passing_yards_allowed`, `passing_tds_allowed`,
  `epa_per_dropback_allowed`, `passing_cpoe_allowed`, `any_a_allowed`,
  `explosive_pass_rate_allowed`, `air_yards_allowed`, `yac_allowed`, `team_passer_rating_allowed`,
  deep-attempt rate faced.
- **Rushing (allowed)** — `carries_faced`, `rushing_yards_allowed`, `yards_per_carry_allowed`,
  `rushing_tds_allowed`, `rush_success_rate_allowed`, `explosive_rush_rate_allowed`, and the
  positive framing `stuff_rate` (duplicated in Pressure & Playmaking).
- **Receiving (allowed)** — display-only mirror of Defense–Passing, same `duplicate_of` safeguards
  as the offensive Receiving block (`targets_faced`, `receptions_allowed`,
  `receiving_yards_allowed`, `catch_rate_allowed`, YAC allowed), plus genuinely unique coverage
  detail from PFR def (completion % / passer rating / aDOT allowed as nearest defender — see Tier 2
  packs).
- **Scoring (allowed)** — `points_allowed`, `points_per_drive_allowed`, `red_zone_td_pct_allowed`,
  `goal_to_go_td_pct_allowed`, `two_pt_conversion_rate_allowed`.
- **Downs & Conversions (allowed)** — `third_down_pct_allowed`, `fourth_down_pct_allowed`,
  `series_conversion_rate_allowed`, `three_and_outs_forced_rate`, first downs allowed by type.
- **Drives & Field Position (allowed)** — `score_pct_per_drive_allowed`, `punts_forced_pct`,
  opponent average starting field position.
- **Penalties** — defensive penalties committed, yards, defensive PI counts/yards (`penalty_type`),
  penalty first downs gifted.

The two defense-specific subcategories:

### Defense — Turnovers (Takeaways)

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `takeaways` | `def_interceptions + fumble_recovery_opp` | PBP | count | 1999 |
| `def_interceptions` | INTs made | PBP +TS | count | 1999 |
| `def_interception_yards` | INT return yards | PBP +TS | count | 1999 |
| `fumble_recovery_opp` | Opponent fumbles recovered | PBP +TS | count | 1999 |
| `def_fumbles_forced` | Fumbles forced | PBP +TS / PLS | count | 1999 |
| `takeaway_rate_per_defensive_snap` | Takeaways / defensive snaps | PBP | rate | 1999 |
| `takeaways_per_drive` | Takeaways / opponent drives | PBP | rate | 1999 |
| `takeaway_epa` | Opponent EPA lost on takeaway plays | PBP | count | 1999 |
| `def_tds` | Defensive touchdowns | PBP +TS | count | 1999 |
| `fumble_recovery_tds` | Fumble returns for TD | PBP +TS | count | 1999 |

### Defense — Pressure & Playmaking

Denominator: defensive snaps, or opponent dropbacks for pass-rush rates.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `def_sacks` | Sacks recorded | PBP +TS / PLS | count | 1999 |
| `def_sack_yards` | Sack yardage forced | PBP +TS | count | 1999 |
| `def_sack_rate_per_dropback` | Sacks / opponent dropbacks | PBP | rate | 1999 |
| `def_qb_hits` | QB hits (non-sack) | PBP / PLS | count | 1999 |
| `qb_pressure_events_rate` | (Hits + sacks) / opponent dropbacks | PBP | rate | 1999 |
| `def_tackles_for_loss` | TFLs | PBP / PLS | count | 1999 |
| `def_tackles_for_loss_yards` | Yards lost on TFLs | TS / PLS | count | 1999 |
| `stuff_rate` | Opponent rushes stopped ≤ 0 yds / carries faced | PBP | rate | 1999 |
| `def_pass_defended` | Passes broken up | PBP / PLS | count | 1999 |
| `def_safeties` | Safeties forced | PBP / PLS | count | 1999 |
| `def_tackles_solo` / `def_tackle_assists` / `def_tackles_with_assist` | Tackle accounting | PLS +TS | count | 1999 |
| `havoc_rate` | (TFL + FF + INT + PD) / defensive snaps | PBP | rate | 1999 |
| `pressure_rate_allowed_to_qbs` | Pressures / opponent dropbacks | PFR | rate | 2018 |
| `blitz_rate` | Blitzes / opponent dropbacks | PFR | rate | 2018 |
| `missed_tackle_rate` | Missed tackles / tackle attempts | PFR | rate | 2018 |
| `defensive_2pt_conversions` | Defensive 2-pt returns | PBP | count | 1999 |

## Catalog: Special Teams

All PBP-derivable; TS validates the FG/PAT/return counting surface. Remember the 2024 kickoff-rule
discontinuity.

### Special Teams — Kicking (FG & XP)

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `fg_att` / `fg_made` / `fg_missed` / `fg_blocked` | Field-goal outcomes | PBP +TS | count | 1999 |
| `fg_pct` | Makes / attempts | PBP +TS | rate | 1999 |
| `fg_long` | Longest make | PBP +TS | count | 1999 |
| `fg_made_0_19` … `fg_made_60_` | Makes by distance bucket (`kick_distance`) | PBP +TS | count | 1999 |
| `fg_missed_0_19` … `fg_missed_60_` | Misses by distance bucket | PBP +TS | count | 1999 |
| `pat_att` / `pat_made` / `pat_missed` / `pat_blocked` | Extra-point outcomes | PBP +TS | count | 1999 |
| `pat_pct` | PAT make rate | PBP +TS | rate | 1999 |
| `gwfg_att` / `gwfg_made` | Game-winning FG outcomes | TS | count | 1999 |
| `kicking_points` | `3·fg_made + pat_made` | PBP | count | 1999 |
| `fg_epa` | Kicker EPA summed on FG plays | PBP | count | 1999 |
| `fg_pct_over_expected` | Make rate vs. distance-based expectation (simple logistic on `kick_distance`) | PBP | rate | 1999 |

### Special Teams — Kickoffs & Coverage (kicking team)

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `kickoffs` | `kickoff_attempt` plays | PBP | count | 1999 |
| `kickoff_touchbacks` / `kickoff_touchback_pct` | Touchbacks and rate | PBP | count/rate | 1999 |
| `kickoff_return_yards_allowed` | Return yards conceded | PBP | count | 1999 |
| `avg_opponent_start_after_kickoff` | Mean opponent drive start | PBP | avg | 1999 |
| `kickoffs_out_of_bounds` | OOB kicks | PBP | count | 1999 |
| `onside_attempts` / `onside_recoveries` | `own_kickoff_recovery` family | PBP | count | 1999 |
| `kickoff_epa` | EPA on kickoff plays (coverage view) | PBP | count | 1999 |

### Special Teams — Kick Returns

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `kickoff_returns` | Returns | PBP +TS | count | 1999 |
| `kickoff_return_yards` | Return yards | PBP +TS | count | 1999 |
| `yards_per_kickoff_return` | Yards / return | PBP | rate | 1999 |
| `kickoff_return_tds` | Return TDs | PBP | count | 1999 |
| `longest_kickoff_return` | Max return | PBP | count | 1999 |
| `avg_start_after_kickoff_return` | Own drive start after KO | PBP | avg | 1999 |

### Special Teams — Punting & Coverage

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `punts` | `punt_attempt` plays | PBP | count | 1999 |
| `punt_gross_avg` | `kick_distance` mean on punts | PBP | avg | 1999 |
| `punt_net_avg` | (Distance − return yards − 20·touchbacks) / punts | PBP | rate | 1999 |
| `punts_inside_20` / `punt_inside_20_pct` | `punt_inside_twenty` | PBP | count/rate | 1999 |
| `punt_touchbacks` | Punt touchbacks | PBP | count | 1999 |
| `punts_blocked` | `punt_blocked` against | PBP | count | 1999 |
| `punt_return_yards_allowed` | Return yards conceded | PBP | count | 1999 |
| `punt_fair_catches_forced` | Fair catches by opponent | PBP | count | 1999 |
| `punt_epa` | EPA on punt plays (net field-position value) | PBP | count | 1999 |

### Special Teams — Punt Returns

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `punt_returns` | Returns | PBP +TS | count | 1999 |
| `punt_return_yards` | Return yards | PBP +TS | count | 1999 |
| `yards_per_punt_return` | Yards / return | PBP | rate | 1999 |
| `punt_return_tds` | Return TDs | PBP | count | 1999 |
| `punt_return_fair_catches` | Fair catches | PBP | count | 1999 |
| `muffed_punts` | Muffs (fumbles on punt receipt) | PBP | count | 1999 |

### Special Teams — ST Scoring & Blocks

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `special_teams_tds` | Kick + punt return TDs | PBP +TS | count | 1999 |
| `kicks_blocked_forced` | Punts + FGs + PATs blocked by this team | PBP | count | 1999 |
| `kicks_blocked_suffered` | Own kicks blocked | PBP | count | 1999 |
| `st_epa` | Total EPA on `special` plays, team perspective | PBP | count | 1999 |
| `st_penalties` / `st_penalty_yards` | Penalties on ST plays | PBP | count | 1999 |

## Situational Splits (Future Display Axis)

Not new metrics — PBP filters applied to the same definitions above, best implemented as a filter
dimension rather than new subcategories. Highest-value splits:

| Split | PBP filter |
| --- | --- |
| Red zone | `yardline_100 <= 20` |
| Third/fourth down | `down in (3, 4)` |
| Early downs (1st/2nd) | `down in (1, 2)` — best signal for EPA/success splits |
| Two-minute drill | `half_seconds_remaining <= 120` |
| By half / quarter | `game_half`, `qtr` |
| Score state | `score_differential` buckets: leading / trailing / one-score |
| Garbage-time exclusion | `wp` between 0.05 and 0.95 (or `vegas_wp`) |
| Home / road | `posteam_type` |
| Division games | `div_game` |
| Vs. winning teams | join opponent final record |

Note for the schedule-adjustment system: garbage-time-filtered and early-down EPA/success rates are
commonly better rating inputs than full-sample versions.

## Secondary-Source Metric Packs (Tier 2)

All three sources were tested against the 2025 pulls in `data/` (plus full history in the
season-level files). Summary of findings, then per-source packs. Keep these visually separated in
the UI ("Advanced / Tracking") with a source badge so Tier 1 tables stay uniform to 1999.

**Test findings (2025 unless noted):**

- **Join keys are solid — but only by ID, never by name.** NGS carries `player_gsis_id` (0 nulls
  across all files); PFR carries `pfr_id`/`pfr_player_id` (0 nulls except 12 legacy rows in def
  season). ESPN QBR uses ESPN `player_id`. `load_players()` was verified to carry `gsis_id`,
  `espn_id`, and `pfr_id` for the full crosswalk. Names are inconsistent even within one source (NGS
  lists "Cam Ward" weekly but "Cameron Ward" in the season row).
- **Team abbreviations need normalization**: NGS and QBR use `LAR` (nflverse `LA`) and QBR uses
  `WSH` (nflverse `WAS`); QBR history also has `OAK`/`SD`/`STL` era codes. PFR weekly files are
  already nflverse-normalized; PFR season files have `2TM`/`3TM` multi-team rows that must be
  handled (prefer weekly files for team attribution).
- **Weekly files include playoffs** (weeks 1–22): always filter `game_type == "REG"` / `season_type
  == "REG"` / `"Regular"`.
- **Row-level completeness is excellent** — essentially zero nulls in key columns in-era for all
  three sources; PFR def season `rat` is null for never-targeted players (expected).
- **NGS is qualification-filtered**, so team aggregation is *incomplete by design*. Measured 2025
  REG coverage vs. official league totals: passing rows cover **97% of attempts** (only qualified
  QBs; 11 team-games have no row at all), rushing **60% of carries**, receiving **55% of targets**.
  Verdict: NGS passing aggregates acceptably to team level with a small bias; NGS rushing/receiving
  must be shown as *player* stats or clearly-labeled "qualified-players-only" team aggregates, never
  as true team totals.
- **NGS weekly attempts exactly match PFR attempts per QB** (sampled top-8 2025 passers), and QBR
  `qb_plays` is consistent with dropbacks (attempts + sacks + scrambles). Cross-source agreement is
  good.
- **PFR sub-era coverage varies by column** (verified null-by-season): pressures, pocket time,
  drops, bad throws — 2018+; on-target %, batted balls, RPO family — 2019+; **play-action passing
  exists only 2019–2023** (dropped 2024+); intended/completed air yards and scrambles — 2024+ only.
  The weekly PFR files carry a much smaller column set than season files (pass weekly: drops, bad
  throws, sack/pressure counts only — no pocket time, on-target, RPO/PA). Def weekly is rich (full
  coverage/pressure/tackle surface) and is the best PFR table for team aggregation.
- **QBR covers 2006+ with zero nulls** in key fields, 27–32 QBs per week (starters only —
  backups/mop-up QBs absent), and a `qualified` flag at season level (58 rows, 28 qualified in
  2025).

### NGS passing → team passing add-ons (2016+, ~97% attempt coverage)

Aggregate weighted by attempts: `avg_time_to_throw`, `avg_completed_air_yards`,
`avg_intended_air_yards`, `avg_air_yards_differential`, `aggressiveness` (tight-window %),
`avg_air_yards_to_sticks`, `expected_completion_percentage`,
`completion_percentage_above_expectation` (NGS xCOMP-based CPOE — keep distinct from PBP `cpoe`),
`max_completed_air_distance`.

### NGS rushing / receiving → player display only (2016+)

Team aggregation not recommended (60% / 55% coverage). Show on Team Details player breakdowns:
`efficiency`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los`, `expected_rush_yards`,
`rush_yards_over_expected(_per_att)`, `rush_pct_over_expected`; `avg_cushion`, `avg_separation`,
`avg_intended_air_yards`, `percent_share_of_intended_air_yards`, `catch_percentage`, `avg_yac`,
`avg_expected_yac`, `avg_yac_above_expectation`.

### PFR passing → team pocket/accuracy add-ons

From season files (aggregate over team QBs; handle `2TM` via weekly attribution):

- 2018+: `pocket_time`, `times_blitzed`, `times_hurried`, `times_hit`, `times_pressured`,
  `pressure_pct`, `drops`, `drop_pct`, `bad_throws`, `bad_throw_pct`, `throwaways`, `spikes`.
- 2019+: `on_tgt_throws`, `on_tgt_pct`, `batted_balls`, RPO family (`rpo_plays`, `rpo_yards`,
  `rpo_pass_att`, `rpo_rush_att`).
- 2019–2023 only: play-action family (`pa_pass_att`, `pa_pass_yards`) — discontinued upstream.
- 2024+ only: `intended_air_yards`, `completed_air_yards` (+ per-attempt variants),
  `pass_yards_after_catch`, `scrambles`, `scramble_yards_per_attempt`.

### PFR defense → team coverage/pressure add-ons (2018+)

The weekly def file has `team` and full columns — aggregate directly to team-week: `def_pressures` →
`pressure_rate` (per opponent dropback), `def_times_blitzed` → `blitz_rate`, `def_times_hurried`,
`def_times_hitqb`, `def_completion_pct` (coverage completion % allowed),
`def_passer_rating_allowed`, `def_adot` (depth of target faced), `def_air_yards_completed`,
`def_yards_after_catch`, `def_missed_tackles` / `def_missed_tackle_pct`, `def_tackles_combined`.

### PFR rushing / receiving → team contact add-ons (2018+)

Weekly files: `rushing_yards_before_contact(_avg)`, `rushing_yards_after_contact(_avg)`,
`rushing_broken_tackles`; `receiving_drop(_pct)`, `receiving_int`, `receiving_rat`,
`receiving_broken_tackles`. Season files add `adot`, `x1d`, YBC/YAC per reception.

### ESPN QBR → QB-context columns on the team surface (2006+)

`qbr_total` (opponent-adjusted 0–100), `qbr_raw` (unadjusted), `pts_added`, `epa_total` and splits
(`pass`, `run`, `sack`, `penalty`), `qb_plays`, `qualified`. Project stance: ESPN's opponent
adjustment is considered weak — this project will still use historical QBR as an input to its own
schedule-adjusted ratings. Preference: **feed `qbr_raw` into the adjustment system** (avoids
stacking ESPN's opponent correction under ours) and display `qbr_total` alongside for reference.
Weekly grain supports faced-defense profiling back to 2006.

## Appendix A: PBP Field Map

All 372 `load_pbp()` fields, grouped. Fields already used by the live pipeline are the backbone of
Tier 1; the rest are available for derivation.

**Identifiers & game context (18)** — `play_id`, `game_id`, `old_game_id`, `nfl_api_id`,
`home_team`, `away_team`, `season`, `season_type`, `week`, `game_date`, `start_time`, `stadium`,
`stadium_id`, `game_stadium`, `location`, `div_game`, `home_opening_kickoff`, `order_sequence`.

**Possession & field position (8)** — `posteam`, `posteam_type`, `defteam`, `side_of_field`,
`yardline_100`, `yrdln`, `end_yard_line`, `goal_to_go`.

**Clock & situation (13)** — `qtr`, `game_half`, `quarter_end`, `time`, `quarter_seconds_remaining`,
`half_seconds_remaining`, `game_seconds_remaining`, `play_clock`, `time_of_day`, `end_clock_time`,
`timeout`, `timeout_team`, `home_timeouts_remaining` / `away_timeouts_remaining` /
`posteam_timeouts_remaining` / `defteam_timeouts_remaining`.

**Play typing (20)** — `play_type`, `play_type_nfl`, `pass`, `rush`, `pass_attempt`, `rush_attempt`,
`qb_dropback`, `qb_scramble`, `qb_kneel`, `qb_spike`, `shotgun`, `no_huddle`, `play`, `special`,
`special_teams_play`, `st_play_type`, `aborted_play`, `play_deleted`, `desc`, `out_of_bounds`.

**Yardage & pass detail (10)** — `yards_gained`, `ydsnet`, `air_yards`, `yards_after_catch`,
`pass_length`, `pass_location`, `run_location`, `run_gap`, `passing_yards` / `rushing_yards` /
`receiving_yards` (player-credited).

**Downs, series, conversions (13)** — `down`, `ydstogo`, `first_down`, `first_down_rush`,
`first_down_pass`, `first_down_penalty`, `third_down_converted`, `third_down_failed`,
`fourth_down_converted`, `fourth_down_failed`, `series`, `series_success`, `series_result`.

**Drives (20)** — `drive`, `fixed_drive`, `fixed_drive_result`, `drive_play_count`,
`drive_time_of_possession`, `drive_first_downs`, `drive_inside20`, `drive_ended_with_score`,
`drive_quarter_start` / `drive_quarter_end`, `drive_yards_penalized`, `drive_start_transition` /
`drive_end_transition`, `drive_game_clock_start` / `drive_game_clock_end`, `drive_start_yard_line` /
`drive_end_yard_line`, `drive_play_id_started` / `drive_play_id_ended`, `drive_real_start_time`.

**Scoring & score state (17)** — `sp`, `touchdown`, `pass_touchdown`, `rush_touchdown`,
`return_touchdown`, `td_team`, `td_player_name` / `td_player_id`, `safety`, `total_home_score` /
`total_away_score`, `posteam_score` / `defteam_score`, `score_differential`, `posteam_score_post` /
`defteam_score_post`, `score_differential_post`.

**Expected points family (29)** — `ep`, `epa`, `qb_epa`, `success`, `air_epa`, `yac_epa`,
`comp_air_epa`, `comp_yac_epa`, scoring-probability inputs (`no_score_prob`, `fg_prob`,
`safety_prob`, `td_prob`, `opp_fg_prob`, `opp_safety_prob`, `opp_td_prob`, `extra_point_prob`,
`two_point_conversion_prob`), and the cumulative `total_home_epa` / `total_away_epa` +
rush/pass/air/YAC cumulative variants (12 columns).

**Win probability family (27)** — `wp`, `def_wp`, `home_wp`, `away_wp`, `wpa`, `vegas_wp`,
`vegas_home_wp`, `vegas_wpa`, `vegas_home_wpa`, `home_wp_post`, `away_wp_post`, `air_wpa`,
`yac_wpa`, `comp_air_wpa`, `comp_yac_wpa`, plus cumulative home/away rush/pass/air/YAC WPA variants
(12 columns).

**Completion probability & pass expectation (9)** — `cp`, `cpoe`, `xpass`, `pass_oe`, `xyac_epa`,
`xyac_mean_yardage`, `xyac_median_yardage`, `xyac_success`, `xyac_fd`.

**Kicking, punting, returns (24)** — `field_goal_result`, `field_goal_attempt`, `kick_distance`,
`extra_point_result`, `extra_point_attempt`, `two_point_conv_result`, `two_point_attempt`,
`kickoff_attempt`, `punt_attempt`, `punt_blocked`, `touchback`, `punt_inside_twenty`,
`punt_in_endzone`, `punt_out_of_bounds`, `punt_downed`, `punt_fair_catch`, `kickoff_inside_twenty`,
`kickoff_in_endzone`, `kickoff_out_of_bounds`, `kickoff_downed`, `kickoff_fair_catch`,
`own_kickoff_recovery`, `own_kickoff_recovery_td`, `return_team` / `return_yards`.

**Defense & contact events (12)** — `sack`, `qb_hit`, `interception`, `tackled_for_loss`,
`solo_tackle`, `assist_tackle`, `tackle_with_assist`, `incomplete_pass`, `complete_pass`,
`defensive_two_point_attempt` / `defensive_two_point_conv`, `defensive_extra_point_attempt` /
`defensive_extra_point_conv`.

**Fumbles (7 event flags)** — `fumble`, `fumble_forced`, `fumble_not_forced`,
`fumble_out_of_bounds`, `fumble_lost`, plus recovery yardage in the attribution block.

**Penalties & replay (7)** — `penalty`, `penalty_team`, `penalty_type`, `penalty_yards`,
`penalty_player_id` / `penalty_player_name`, `replay_or_challenge`, `replay_or_challenge_result`.

**Player attribution (~110)** — passer/rusher/receiver names + IDs + jersey numbers (both official
`*_player_*` and convenience `passer`/`rusher`/`receiver`/`name`/`id`/`fantasy*` columns), lateral
chains, kicker/punter/returner IDs, `td_player`, `safety_player`, `interception_player`,
`sack_player` + `half_sack_1/2`, `tackle_for_loss_1/2`, `qb_hit_1/2`, `solo_tackle_1/2`,
`assist_tackle_1–4`, `tackle_with_assist_1/2`, `pass_defense_1/2`, `forced_fumble_player_1/2`,
`fumbled_1/2`, `fumble_recovery_1/2` (+ teams and yards), `own_kickoff_recovery_player`,
`blocked_player`. Used for player pages, not team aggregates.

**Environment & betting (10)** — `weather`, `roof`, `surface`, `temp`, `wind`, `spread_line`,
`total_line`, `home_coach`, `away_coach`, plus final `home_score` / `away_score` / `result` /
`total`.

Removed from consideration (project decision): FTN charting and participation (broken / discontinued
/ insufficient history), injuries (unreliable), depth charts (starters are identified from PBP
usage), draft picks, combine, contracts, trades (out of scope).

[qb-stats-catalog.md]: (qb-stats-catalog.md)
