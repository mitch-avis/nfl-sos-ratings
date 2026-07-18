# NFL Quarterback Stats Catalog

A human-readable companion to the typed metrics registry under `nfl_sos_ratings/metrics/`.

The registry is the authoritative source of truth for published metric definitions, categories,
polarity, and implementation status.

This document explains the quarterback-side metric surface, sourcing rules, and display taxonomy for
maintainers and analysts.

ESPN QBR has no `nflreadpy` load function yet; download it directly from the nflverse release assets
(Parquet, smallest format):

- [qbr_season_level.parquet]
- [qbr_week_level.parquet]

The pipeline's `data_loader.load_espn_qbr()` wraps these URLs.

Scope: regular season. Included: stats directly describing individual quarterback play, plus the
opposing-defense stats that are directly related to QB play (the `qopp_` context surface used by the
schedule adjustment). Excluded: general team stats, non-QB player stats.

The sourcing rules, metric shapes (count / rate / avg), tier definitions, and test findings from the
team catalog all apply here; this document only restates what differs for individual QBs.

## Table of Contents

- [NFL Quarterback Stats Catalog](#nfl-quarterback-stats-catalog)
  - [Table of Contents](#table-of-contents)
  - [QB-Level Ground Rules](#qb-level-ground-rules)
  - [Views and Denominators](#views-and-denominators)
  - [Category Taxonomy](#category-taxonomy)
  - [Catalog: External \& Reference Ratings](#catalog-external--reference-ratings)
    - [ESPN QBR (2006+, starters only — 27–32 QBs/week)](#espn-qbr-2006-starters-only--2732-qbsweek)
  - [Catalog: Identity \& Availability](#catalog-identity--availability)
  - [Catalog: Passing Volume](#catalog-passing-volume)
  - [Catalog: Passing Efficiency](#catalog-passing-efficiency)
  - [Catalog: Advanced \& Expected (Tier 2)](#catalog-advanced--expected-tier-2)
    - [NGS passing (2016+, qualified QBs — 97% of league attempts)](#ngs-passing-2016-qualified-qbs--97-of-league-attempts)
    - [PFR advanced passing (2018+ base; sub-eras per column)](#pfr-advanced-passing-2018-base-sub-eras-per-column)
  - [Catalog: Pressure, Sacks \& Pocket](#catalog-pressure-sacks--pocket)
  - [Catalog: Rushing](#catalog-rushing)
  - [Catalog: Scoring, Clutch \& Outcomes](#catalog-scoring-clutch--outcomes)
  - [Catalog: Turnovers \& Ball Security](#catalog-turnovers--ball-security)
  - [Opponent Views](#opponent-views)
  - [Ratings Safeguards for QB Metrics](#ratings-safeguards-for-qb-metrics)
  - [Page Layout Recommendation](#page-layout-recommendation)

## QB-Level Ground Rules

1. **Identity is GSIS-based, joins are ID-only.** The pipeline already canonicalizes QBs to
   `gsis_id`. Verified crosswalk: `load_players()` carries `gsis_id`, `espn_id` (→ QBR `player_id`),
   and `pfr_id` (→ PFR `pfr_id`/`pfr_player_id`); NGS carries `player_gsis_id` directly (0 nulls).
   Never join on names — NGS itself lists "Cam Ward" in weekly rows and "Cameron Ward" in its season
   row.
2. **Official passing truth comes from `player_stats` (PLS)**, per the existing pipeline: attempts,
   completions, yards, TDs, INTs, sacks, sack yards, passing EPA, CPOE are replaced with PLS weekly
   values. PBP remains the source for everything situational and derived (dropbacks, scrambles,
   kneels, red zone, late-game state, success rate, air/YAC EPA splits). The same exactness verified
   at team grain applies: PBP-derived official stats reconcile with PLS, and the validation harness
   should assert it per QB-week.
3. **Sign convention:** PLS `sack_yards_lost` is stored negative upstream; the ETL normalizes to
   positive magnitude (`qb_sack_yards_lost`), and every formula here assumes the magnitude.
4. **Primary-QB policy.** Wins, 4QC, and GWD are assigned only to the team-week's primary QB.
   Verified: PBP dropbacks alone reproduce the snap-count choice in 541/544 team-weeks (2025); the 3
   disagreements are injury/pull games. Policy: primary = most offensive snaps (SNP, 2012+) with
   dropbacks → attempts as tie-breaks; pre-2012 fallback = most dropbacks.
5. **Season attribution for multi-team QBs:** keep QB-team season rows (the current shape) and let
   the UI aggregate across teams for a player view; PFR season files' `2TM` rows are never used
   (weekly rows carry true teams).
6. **Qualification:** league-wide QBs page needs a minimum-dropback threshold (existing
   `qb_is_eligible`). QBR provides its own `qualified` flag (28 of 58 QBs in 2025) — do not mix the
   two; apply this project's threshold consistently and show ESPN's flag only as context.

## Views and Denominators

| View | Denominator | Additional Notes |
| --- | --- | --- |
| Ratings | none | Schedule-adjusted outputs (`QSaCR`, `QSaOR`, `QRaw`, `QSoS`, `QOutcome`). Every published project rating is standardized within its own season, so `0` means that season's average and `+1` means one standard deviation above that season's average. `QSaCR` is the published weighted QB composite over adjusted EPA/dropback, CPOE, sack rate, and TD-INT margin rate. `QRaw` and `QSaCR` are published for `2006+` only because CPOE is part of their formula; the `1999-2005` rows are intentionally null. This is a view, not a category. |
| Raw Stat Totals | none | For **count** metrics only. **rate** and **avg** metrics keep the same value they show elsewhere. |
| Per-Game Rates | games played (or games as primary QB — must be labeled) | For **count** metrics this is the per-game form. **rate** and **avg** metrics are unchanged. |
| Per-Play Rates | play-specific denominators for the subcategory: dropback, attempt, carry, drive, series, etc. | For **count** metrics this uses each subcategory's natural denominator; **rate** and **avg** metrics are unchanged. |
| - | Per-Dropback | dropbacks (attempts + sacks + scrambles) for all passing counts |
| - | Per-Attempt | official attempts — only for the conventional rates flagged `_per_attempt` |
| - | Per-Carry | QB carries — rushing subcategory only |
| Opponent Per-Game Rates | games played, measured from the opponent profile | Opponent-profile columns are `qopp_`-prefixed. Count metrics keep their opponent per-game form here; intrinsic **rate** and **avg** metrics are unchanged opponent-context values. |
| Opponent Per-Play Rates | play-specific denominators from the opponent profile | Opponent-profile columns are `qopp_`-prefixed and keep the same natural-denominator suffixes as the matching subject-side per-play stats (for example `qopp_qb_epa_per_dropback`, `qopp_qb_yards_per_carry`). |

The naming suffix rules from the team catalog apply (`qb_` prefix preserved; every rate carries its
denominator in name or documented convention).

`_per_drive` and `_per_series` are still part of the same `Per-Play Rates` view when those
denominators appear on supporting context columns; they do not create extra view types.

## Category Taxonomy

Modeled on the team taxonomy, PFR passing/advanced-passing pages, and ESPN/NFL.com passing tables:

Both QB pages — the league-wide QBs table and the individual QB Details page — use this
eight-subcategory taxonomy inside the five stat views.

`Ratings` stays separate as its own primary view and is not part of this taxonomy:

```text
QB
├── Identity & Availability     (games, starts, snaps, snap share, dropbacks)
├── Passing Volume              (attempts, completions, yards, air yards, TDs, INTs)
├── Passing Efficiency          (comp %, YPA, ANY/A, EPA/dropback, CPOE, passer rating)
├── Advanced & Expected         (NGS tracking and PFR accuracy — Tier 2)
├── Pressure, Sacks & Pocket    (sacks, pressure, pocket time, scramble escape)
├── Rushing                     (carries, yards, TDs, designed vs. scramble)
├── Scoring, Clutch & Outcomes  (total TDs, red zone, wins, 4QC, GWD)
└── Turnovers & Ball Security   (INTs, fumbles, giveaway rates)
```

Notes: for an individual QB a "Receiving" block is meaningless and omitted. Rushing is a first-class
subcategory (a named gap in the current UI), covering both designed runs and scrambles. Opponent
context is expressed through the two opponent views applied to these same eight subcategories, not
as a ninth tab or category.

The `Ratings` view now has two descriptive blocks:

- the project's own schedule-adjusted ratings (`QSaCR`, `QSaOR`, `QRaw`, `QSoS`, `QOutcome`)
- external/reference ratings such as ESPN QBR and related ESPN reference columns

For the project's own ratings, the displayed scale is always within-season. If a QB is `+2.0`, that
means two standard deviations better than that season's qualifying QBs, not two standard deviations
better than a pooled all-time reference.

Those external/reference ratings are `ratings_eligible=False` in the registry and never feed the
published project ratings.

## Catalog: External & Reference Ratings

These metrics live in the `Ratings` view for analyst context and external reference checks. They are
not part of the five stat-view taxonomies below and are never allowed into any rating pool.

### ESPN QBR (2006+, starters only — 27–32 QBs/week)

| Column | Definition | Shape |
| --- | --- | --- |
| `qb_qbr_total` | Adjusted Total QBR (0–100, opponent-adjusted by ESPN) | rate |
| `qb_qbr_raw` | Raw QBR, unadjusted — **preferred input to this project's own schedule adjustment** (avoids stacking ESPN's opponent correction under ours); `qbr_total` displayed alongside | rate |
| `qb_pts_added` | Points contributed above average QB | count |
| `qb_qbr_plays` | ESPN dropback count (cross-checks PBP dropbacks) | count |

## Catalog: Identity & Availability

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_games_played` | Games with ≥ 1 dropback or carry | PBP | count | 1999 |
| `qb_games_started` | Games as primary QB (policy above) | PBP/SNP | count | 1999 |
| `qb_offense_snaps` | Offensive snaps played | SNP | count | 2012 |
| `qb_snap_share` | Snaps / team offensive snaps | SNP+PBP | rate | 2012 |
| `qb_dropbacks` | Attempts + sacks + scrambles | PBP | count | 1999 |
| `qb_plays` | Dropbacks + designed carries (total usage) | PBP | count | 1999 |
| `qb_is_eligible` | Meets project dropback threshold | D | flag | — |

## Catalog: Passing Volume

All counts appear in Totals / Per-Game / Per-Dropback views.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_attempts` | Official pass attempts | PLS | count | 1999 |
| `qb_completions` | Completions | PLS | count | 1999 |
| `qb_pass_yards` | Gross passing yards | PLS | count | 1999 |
| `qb_net_pass_yards` | `pass_yards − sack_yards_lost` | D(PLS) | count | 1999 |
| `qb_pass_touchdowns` | Passing TDs | PLS | count | 1999 |
| `qb_interceptions` | INTs thrown | PLS | count | 1999 |
| `qb_passing_first_downs` | First downs via pass | PLS | count | 1999 |
| `qb_passing_air_yards` | Air yards incl. incompletions | PLS | count | 2006 |
| `qb_passing_yards_after_catch` | YAC on his completions | PLS | count | 1999 |
| `qb_passing_2pt_conversions` | 2-pt conversion passes | PLS | count | 1999 |
| `qb_explosive_completions` | 20+ yard completions | PBP | count | 1999 |
| `qb_longest_completion` | Max completed pass | PBP | count | 1999 |
| `qb_spikes` / `qb_throwaways` | Clock kills / intentional incompletions | PBP / PFR | count | 1999 / 2018 |

## Catalog: Passing Efficiency

The heart of the QBs page. Current rating-pool members marked ★ (see safeguards).

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_completion_pct` | `completions / attempts` — named UI gap, add | D(PLS) | rate | 1999 |
| `qb_yards_per_attempt` | `pass_yards / attempts` | D | rate | 1999 |
| `qb_net_yards_per_attempt` | `(pass_yds − sack_yds) / (att + sacks)` | D | rate | 1999 |
| `qb_adjusted_yards_per_attempt` | AY/A: `(yds + 20·TD − 45·INT) / att` | D | rate | 1999 |
| `qb_any_a` ★ | ANY/A: `(yds + 20·TD − 45·INT − sack_yds) / (att + sacks)` | D | rate | 1999 |
| `qb_pass_yards_per_dropback` | `pass_yards / dropbacks` | D | rate | 1999 |
| `qb_passing_epa` | EPA on dropbacks (`qb_epa` basis) | PLS | count | 1999 |
| `qb_epa_per_dropback` ★ | `passing_epa / dropbacks` | D | rate | 1999 |
| `qb_success_rate` | Dropbacks with EPA > 0 / dropbacks | PBP | rate | 1999 |
| `qb_completion_percentage_above_expectation` ★ | PBP-model CPOE | PLS | avg | 2006 |
| `qb_passer_rating` | NFL formula | D | rate | 1999 |
| `qb_td_rate_per_attempt` | `pass_TDs / attempts` | D | rate | 1999 |
| `qb_int_rate_per_attempt` | `INTs / attempts` | D | rate | 1999 |
| `qb_td_int_margin_rate` ★ | `(pass_TDs − INTs) / dropbacks` | D | rate | 1999 |
| `qb_first_down_rate_per_dropback` | Passing first downs / dropbacks | D | rate | 1999 |
| `qb_adot` | `air_yards / attempts` (aggressive depth) | D(PLS) | rate | 2006 |
| `qb_pacr` | Passing Air Conversion Ratio: `pass_yards / air_yards` | PLS | rate | 2006 |
| `qb_explosive_pass_rate` | 20+ yard completions / dropbacks | PBP | rate | 1999 |
| `qb_deep_attempt_rate` | Deep attempts / attempts | PBP | rate | 1999 |
| `qb_air_epa_per_dropback` / `qb_yac_epa_per_dropback` | EPA split: air vs. after catch | PBP | rate | 1999 |
| `qb_xyac_per_completion` / `qb_yac_over_expected` | Expected YAC and delta (supporting-cast lens) | PBP | avg | 2006 |
| `qb_wpa_total` | Win probability added on dropbacks | PBP | count | 1999 |

## Catalog: Advanced & Expected (Tier 2)

All verified reliable in-era (see team catalog test findings). Join by `pfr_id` / `player_gsis_id`,
with the ESPN/QBR identity crosswalk handled in the separate external/reference ratings block above.
Filter to regular season; normalize `LAR`→`LA`, `WSH`→`WAS`.

### NGS passing (2016+, qualified QBs — 97% of league attempts)

`qb_avg_time_to_throw`, `qb_avg_completed_air_yards`, `qb_avg_intended_air_yards` (NGS aDOT),
`qb_avg_air_yards_differential`, `qb_aggressiveness` (tight-window attempt %),
`qb_avg_air_yards_to_sticks`, `qb_max_completed_air_distance`, `qb_expected_completion_percentage`
(xCOMP%), `qb_ngs_cpoe` (`completion_percentage_above_expectation` — keep distinct from PBP CPOE;
the two models disagree by design). All **avg** shape.

### PFR advanced passing (2018+ base; sub-eras per column)

- 2018+: `qb_pocket_time`, `qb_times_blitzed`, `qb_times_hurried`, `qb_times_hit`,
  `qb_times_pressured`, `qb_pressure_pct`, `qb_drops_suffered`, `qb_drop_pct_suffered`,
  `qb_bad_throws`, `qb_bad_throw_pct`, `qb_throwaways`, `qb_spikes_pfr`.
- 2019+: `qb_on_tgt_throws`, `qb_on_tgt_pct`, `qb_batted_balls`, RPO usage (`qb_rpo_plays`,
  `qb_rpo_pass_att`, `qb_rpo_rush_att`, yards).
- 2019–2023 only: play-action volume/yards (`qb_pa_pass_att`, `qb_pa_pass_yards`) — discontinued
  upstream, display with era note.
- 2024+ only: PFR air-yards family and `qb_scrambles_pfr` / `qb_scramble_yards_per_attempt` (prefer
  PBP scrambles, 1999+).
- Derived: `qb_drop_adjusted_comp_pct` = `(completions + drops) / (attempts − throwaways − spikes −
  batted_balls)` — accuracy isolated from receiver/system noise.

## Catalog: Pressure, Sacks & Pocket

Sacks are Tier 1; pressure context is PFR Tier 2.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_sacks` | Sacks taken | PLS | count | 1999 |
| `qb_sack_yards_lost` | Yards lost (positive magnitude) | PLS | count | 1999 |
| `qb_sack_rate` ★ | `sacks / dropbacks` | D | rate | 1999 |
| `qb_sack_fumbles` / `qb_sack_fumbles_lost` | Strip-sacks / lost | PLS | count | 1999 |
| `qb_qb_hits_taken` | Hits absorbed (non-sack) | PBP | count | 1999 |
| `qb_pressure_rate_faced` | `times_pressured / dropbacks` | PFR | rate | 2018 |
| `qb_blitz_rate_faced` | `times_blitzed / dropbacks` | PFR | rate | 2018 |
| `qb_pocket_time` | Avg seconds in pocket | PFR | avg | 2018 |
| `qb_scramble_rate` | Scrambles / dropbacks (pressure escape lens) | PBP | rate | 1999 |
| `qb_sack_rate_vs_pressure` | Sacks / times pressured (pressure-to-sack conversion) | D(PFR) | rate | 2018 |

## Catalog: Rushing

A named gap in the current UI. Per-carry denominators. Official carries include scrambles and
kneels; the designed/scramble split is the analytical view.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_carries` | Official rush attempts (incl. scrambles, kneels) | PLS | count | 1999 |
| `qb_rushing_yards` | Rushing yards (incl. scramble yards) | PLS | count | 1999 |
| `qb_yards_per_carry` | `rushing_yards / carries` | D | rate | 1999 |
| `qb_rushing_tds` | Rushing TDs | PLS | count | 1999 |
| `qb_rushing_first_downs` | First downs on rushes | PLS | count | 1999 |
| `qb_rushing_epa` | EPA on rush plays | PLS | count | 1999 |
| `qb_epa_per_carry` | `rushing_epa / carries` | D | rate | 1999 |
| `qb_designed_carries` / `qb_designed_rush_yards` | Excluding scrambles and kneels | PBP | count | 1999 |
| `qb_scrambles` / `qb_scramble_yards` | Scramble volume (duplicated in Pocket) | PBP | count | 1999 |
| `qb_yards_per_scramble` | `scramble_yards / scrambles` | PBP | rate | 1999 |
| `qb_kneels` | Kneel-downs (excluded from analytic rates) | PBP | count | 1999 |
| `qb_rush_success_rate` | Success rate on designed QB runs | PBP | rate | 1999 |
| `qb_explosive_rush_rate` | 10+ yard runs / carries | PBP | rate | 1999 |
| `qb_rushing_fumbles` / `qb_rushing_fumbles_lost` | Fumbles rushing / lost | PLS | count | 1999 |
| `qb_rushing_2pt_conversions` | 2-pt runs | PLS | count | 1999 |

`qb_total_epa_per_play` = (passing EPA + rushing EPA) / (dropbacks + designed carries) is the
dual-threat headline rate combining both subcategories.

## Catalog: Scoring, Clutch & Outcomes

Outcome stats follow the primary-QB assignment policy.

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_total_tds` | Pass TDs + rush TDs (TDs accounted for) | D(PLS) | count | 1999 |
| `qb_total_td_rate` | Total TDs / (dropbacks + designed carries) | D | rate | 1999 |
| `qb_red_zone_td_pass_pct` | RZ TD passes / RZ attempts | PBP | rate | 1999 |
| `qb_red_zone_epa_per_dropback` | EPA/dropback inside the 20 | PBP | rate | 1999 |
| `qb_2pt_conversions` | Pass + rush 2-pt conversions | PLS | count | 1999 |
| `qb_wins` / `qb_losses` / `qb_ties` | Record as primary QB | D(SCH+PBP) | count | 1999 |
| `qb_win_pct` | `(wins + 0.5·ties) / starts` | D | rate | 1999 |
| `qb_fourth_quarter_comebacks` | Existing pipeline definition | D(PBP) | count | 1999 |
| `qb_game_winning_drives` | Existing pipeline definition | D(PBP) | count | 1999 |
| `qb_late_close_epa_per_dropback` | EPA/dropback in Q4/OT, one-score game | PBP | rate | 1999 |
| `qb_third_down_conversion_rate` | 3rd-down dropbacks converted / attempts | PBP | rate | 1999 |

## Catalog: Turnovers & Ball Security

| Column | Definition / formula | Source | Shape | Since |
| --- | --- | --- | --- | --- |
| `qb_interceptions` | INTs (duplicated from Volume) | PLS | count | 1999 |
| `qb_fumbles` | Sack + rush fumbles (kept + lost) | PLS | count | 1999 |
| `qb_fumbles_lost` | Sack + rush fumbles lost | PLS | count | 1999 |
| `qb_giveaways` | INTs + fumbles lost | D | count | 1999 |
| `qb_giveaway_rate` | Giveaways / (dropbacks + designed carries) | D | rate | 1999 |
| `qb_td_int_differential` | `pass_TDs − INTs` | D | count | 1999 |
| `qb_turnover_epa` | EPA on his giveaway plays | PBP | count | 1999 |
| `qb_int_worthy_context` | Bad-throw % as the era-stable proxy (no FTN) | PFR | rate | 2018 |

## Opponent Views

The defenses-faced profile: for each defense the QB actually played (primary-QB games only,
head-to-head games excluded, deduplicated — existing pipeline rules), average what those defenses
allowed to *all other* QBs, then average across the faced list. Every column below is a season-long
context descriptor (UI `contextual: true`), not a QB grade.

These values appear through `Opponent Per-Game Rates` and `Opponent Per-Play Rates`, not through a
standalone `Opponent Context` category.

Tier 1 mirror (PBP, 1999+) — what faced defenses allowed per opposing QB/dropback:

| Column | Mirrors |
| --- | --- |
| `qopp_epa_per_dropback_allowed` | `qb_epa_per_dropback` |
| `qopp_any_a_allowed` | `qb_any_a` |
| `qopp_cpoe_allowed` | `qb_completion_percentage_above_expectation` (2006+) |
| `qopp_completion_pct_allowed` | `qb_completion_pct` |
| `qopp_passer_rating_allowed` | `qb_passer_rating` |
| `qopp_sack_rate` | `qb_sack_rate` (their sacks / opponent dropbacks) |
| `qopp_td_int_margin_rate_allowed` | `qb_td_int_margin_rate` |
| `qopp_int_rate` | `qb_int_rate_per_attempt` (takeaway lens) |
| `qopp_pass_success_rate_allowed` | `qb_success_rate` |
| `qopp_explosive_pass_rate_allowed` | `qb_explosive_pass_rate` |
| `qopp_yards_per_dropback_allowed` | `qb_pass_yards_per_dropback` |
| `qopp_qb_rush_epa_per_carry_allowed` | `qb_epa_per_carry` (QB-run defense) |

Tier 2 context (PFR def weekly aggregated to team, 2018+): `qopp_pressure_rate` (`def_pressures` /
dropbacks faced), `qopp_blitz_rate` (`def_times_blitzed`), `qopp_passer_rating_allowed_pfr`
(coverage-grain cross-check). QBR weekly (2006+) supports a `qopp_qbr_raw_allowed` cross-check: mean
raw QBR that faced defenses allowed to other starters.

`diff_qb_*` columns (QB minus `qopp_` context on the same metric) follow the existing prefix
convention and are derived for every mirrored pair.

## Ratings Safeguards for QB Metrics

Same registry design as the team catalog (`ratings_eligible`, `duplicate_of`, explicit allowlisted
pools). QB-specific collision notes:

- The current primary pool ★ (`qb_epa_per_dropback`, `qb_any_a`, `qb_cpoe`, `qb_td_int_margin_rate`,
  `qb_sack_rate`) has known internal overlap: ANY/A already encodes TDs, INTs, and sack yardage, and
  sack rate overlaps ANY/A's sack term. Documented as an accepted, deliberate weighting — but the
  registry must forbid *adding* further overlapping members (e.g., `qb_passer_rating`, which
  restates comp% / YPA / TD% / INT%; or `qb_int_rate_per_attempt` next to `qb_td_int_margin_rate`).
- `qb_passer_rating`, AY/A, NY/A, `qb_td_int_differential`, and all `_per_game` variants of pool
  members are `duplicate_of` pool inputs → display-only.
- `qb_qbr_total` is ESPN-opponent-adjusted; if QBR enters the adjustment system, use `qb_qbr_raw`
  (project decision) and never both.
- NGS CPOE and PBP CPOE are different models — display both, never pool both.
- Outcome-layer stats (wins, 4QC, GWD) stay quarantined in `QOutcome` and never join the performance
  pools.

## Page Layout Recommendation

**QBs page (league-wide table)**: the top row is a single-select six-view control in this exact
order: `Ratings`, `Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`, `Opponent Per-Game Rates`,
`Opponent Per-Play Rates`, plus `Reset`.

When any non-`Ratings` view is active, the secondary row is the eight-subcategory multi-select in
the taxonomy order above.

Default sort `QSaCR`; qualification filter on `qb_is_eligible` with a "show all" toggle. The
`Ratings` view keeps `QSaOR` as the ridge backbone, `QSaCR` as the published weighted composite, and
`QOutcome` as descriptive-only context.

**QB Details page**: identical section order for cohesion, each section adding: percentile bars vs.
qualified QBs (the existing `_pct` convention), the `qopp_` context and `diff_` comparison inline
per metric, a weekly game log (PLS weekly + QBR weekly), and a faced-defenses table.

Tier 2 sections carry source badges and era notes (PFR sub-eras, NGS 2016+, QBR 2006+) so missing
values in older seasons read as "not tracked yet," not as gaps.

[qbr_season_level.parquet]:
    https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_season_level.parquet
[qbr_week_level.parquet]:
    https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.parquet
