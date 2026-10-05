"""Quarterback metric definitions: every QB column the pipeline publishes.

Human-readable companion: [docs/qb-stats-catalog.md](../../docs/qb-stats-catalog.md).
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_ratings = section("qb", "Schedule-Adjusted Ratings")
_identity = section("qb", "Identity & Availability")
_volume = section("qb", "Passing Volume")
_efficiency = section("qb", "Passing Efficiency")
_pressure = section("qb", "Pressure, Sacks & Pocket")
_rushing = section("qb", "Rushing")
_clutch = section("qb", "Scoring, Clutch & Outcomes")
_turnovers = section("qb", "Turnovers & Ball Security")

QB_RATING_METRICS: tuple[MetricDef, ...] = (
    _ratings(
        name="adj_qb_epa_per_dropback",
        label="Adj EPA/DB",
        full_name="Adjusted EPA Per Dropback",
        description=(
            "The quarterback's expected points added per dropback after adjusting for the pass "
            "defenses he faced. It reads on the same scale as raw EPA per dropback, and small "
            "samples are pulled toward the league average. Higher is better."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
    ),
    _ratings(
        name="qb_faced_pass_defense",
        label="Faced Pass D",
        full_name="Faced Pass Defense",
        description=(
            "The average quality of the pass defenses this quarterback faced, weighted by his "
            "dropbacks, in EPA per dropback prevented. Each defense is rated without its games "
            "against this quarterback. Positive means tougher defenses. Early in a season, "
            "defenses that have faced no other passer yet are left out. Context, not a QB grade."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        contextual=True,
    ),
    _ratings(
        name="qb_rank",
        label="Rank",
        full_name="QB Rating Rank",
        description=(
            "The quarterback's place among eligible quarterbacks by Adjusted EPA Per Dropback, "
            "1 for the best. Quarterbacks with equal ratings share the better rank."
        ),
        shape="score",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _ratings(
        name="qb_rank_missing_share",
        label="No-Dropback Share",
        full_name="Share of Resamples Without the Quarterback",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the quarterback had no dropbacks and so no rank. A "
            "quarterback who played only part of the season is missing more often, and the "
            "rank quantiles come only from the resamples that include him."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
    ),
    _ratings(
        name="qb_rank_top5_probability",
        label="Top-5 Chance",
        full_name="Chance of a Top-5 Rank",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the quarterback ranked in the top five eligible "
            "quarterbacks by Adjusted EPA Per Dropback. It shows how much the ranking depends "
            "on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
    ),
    _ratings(
        name="qb_rank_top10_probability",
        label="Top-10 Chance",
        full_name="Chance of a Top-10 Rank",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the quarterback ranked in the top ten eligible "
            "quarterbacks by Adjusted EPA Per Dropback. It shows how much the ranking depends "
            "on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
    ),
    _ratings(
        name="qb_rank_probabilities",
        label="Rank Chances",
        full_name="Chance of Each Rank",
        description=(
            "A list giving, for each rank from 1 down, the share of game-bootstrap resamples of "
            "the season in which the quarterback finished at exactly that rank among eligible "
            "quarterbacks."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
    ),
)

QB_IDENTITY_METRICS: tuple[MetricDef, ...] = (
    _identity(
        name="qb_id",
        label="QB ID",
        full_name="Quarterback ID",
        description="The canonical GSIS player identifier for this quarterback.",
        shape="id",
        polarity="neutral",
        source="PBP",
    ),
    _identity(
        name="qb_name",
        label="QB",
        full_name="Quarterback",
        description="The quarterback's display name.",
        shape="id",
        polarity="neutral",
        source="PBP",
    ),
    _identity(
        name="player_id",
        label="Player ID",
        full_name="Player ID",
        description="The GSIS player identifier used to join across data sources.",
        shape="id",
        polarity="neutral",
        source="PLS",
    ),
    _identity(
        name="player_display_name",
        label="QB",
        full_name="Quarterback Display Name",
        description="The quarterback's display name from the official player feed.",
        shape="id",
        polarity="neutral",
        source="PLS",
    ),
    _identity(
        name="qb_games_played",
        label="QB Games",
        full_name="QB Games Played",
        description="Games in which this quarterback recorded a dropback.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _identity(
        name="qb_offense_snaps",
        label="QB Snaps",
        full_name="QB Offensive Snaps",
        description="Offensive snaps the quarterback played, from snap-count data.",
        shape="count",
        polarity="neutral",
        source="SNP",
        since=2012,
    ),
    _identity(
        name="qb_dropbacks",
        label="Dropbacks",
        full_name="QB Dropbacks",
        description=(
            "Pass attempts plus sacks plus scrambles — every play that began as a pass. "
            "The natural denominator for QB efficiency stats."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _identity(
        name="qb_is_eligible",
        label="Eligible",
        full_name="QB Eligibility Flag",
        description=(
            "Whether the quarterback has the qualifying number of pass attempts (14 for every "
            "game his team has played) to be ranked on the league-wide QBs page."
        ),
        shape="flag",
        polarity="neutral",
        source="D",
    ),
    _identity(
        name="qb_attempt_qualifier",
        label="Qualifier Att",
        full_name="Qualifying Pass Attempts",
        description=(
            "The pass attempts this quarterback needs to be ranked: 14 for every game his team "
            "has played so far (his main team, for a quarterback who changed teams)."
        ),
        shape="count",
        polarity="neutral",
        source="D",
    ),
)

QB_VOLUME_METRICS: tuple[MetricDef, ...] = (
    _volume(
        name="qb_attempts",
        label="Att",
        full_name="QB Pass Attempts",
        description="Official pass attempts (sacks and two-point tries excluded).",
        shape="count",
        polarity="neutral",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_completions",
        label="Comp",
        full_name="QB Completions",
        description="Passes completed to a teammate.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_pass_yards",
        label="Pass Yds",
        full_name="QB Passing Yards",
        description="Gross passing yards on completions (sack yardage not subtracted).",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_pass_touchdowns",
        label="Pass TDs",
        full_name="QB Passing Touchdowns",
        description="Touchdown passes thrown.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_interceptions",
        label="INTs",
        full_name="QB Interceptions",
        description="Passes intercepted by the defense. Fewer is better.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_passing_epa",
        label="Pass EPA",
        full_name="QB Passing EPA",
        description=(
            "Total expected points added on this quarterback's dropbacks. EPA credits "
            "down, distance, and field position — not just raw yards."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="wp_kept_dropback_share",
        label="Kept Dropbacks",
        full_name="Share of Dropbacks the Filter Keeps",
        description=(
            "The share of this quarterback's dropbacks that the chosen garbage-time filter "
            "keeps. 1.00 means no dropback was left out."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="dropbacks",
        since=1999,
    ),
    _volume(
        name="qb_wp_bin_dropbacks",
        label="Dropbacks",
        full_name="Dropbacks in Win-Probability Bin",
        description=(
            "This quarterback's dropbacks in one game and win-probability bin. Summed over every "
            "bin they equal his dropbacks in that game."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _volume(
        name="qb_wp_bin_epa",
        label="Pass EPA",
        full_name="Passing EPA in Win-Probability Bin",
        description=(
            "Expected points added credited to this quarterback on his dropbacks in one game "
            "and win-probability bin, summed from play-by-play."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

QB_EFFICIENCY_METRICS: tuple[MetricDef, ...] = (
    _efficiency(
        name="qb_epa_per_dropback",
        label="EPA/DB",
        full_name="QB EPA Per Dropback",
        description=(
            "Expected points added per dropback — the single best play-level measure of "
            "quarterback efficiency. League average sits near zero."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
    ),
    _efficiency(
        name="qb_pass_yards_per_dropback",
        label="Pass Yds/DB",
        full_name="QB Passing Yards Per Dropback",
        description="Passing yards divided by dropbacks, so sacks and scrambles count.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
    ),
    _efficiency(
        name="qb_td_int_margin_rate",
        label="TD-INT Margin/DB",
        full_name="QB TD-INT Margin Rate",
        description="Touchdown passes minus interceptions, divided by dropbacks.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        formula="(pass_touchdowns - interceptions) / dropbacks",
    ),
    _efficiency(
        name="qb_any_a",
        label="ANY/A",
        full_name="QB Adjusted Net Yards Per Attempt",
        description=(
            "The best single conventional passing stat: yards per attempt with a +20-yard "
            "bonus per touchdown, a -45-yard penalty per interception, and sacks counted "
            "against."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts + sacks",
        since=1999,
        formula="(yards + 20*TD - 45*INT - sack_yards) / (attempts + sacks)",
        note="Overlaps the TD-INT and sack pool members; accepted, frozen overlap.",
    ),
    _efficiency(
        name="qb_completion_percentage_above_expectation",
        label="CPOE",
        full_name="QB Completion Percentage Above Expectation",
        description=(
            "How much higher the quarterback's completion rate was than the difficulty of "
            "the throws would predict, in percentage points. Positive means more accurate "
            "than expected."
        ),
        shape="avg",
        polarity="higher",
        source="PLS",
        denominator="pass attempts (model-expected completions)",
        since=2006,
    ),
    _efficiency(
        name="qb_passer_rating",
        label="Passer Rating",
        full_name="QB Passer Rating",
        description=(
            "The classic NFL passer-rating formula (0 to 158.3), built from completion "
            "rate, yards, touchdowns, and interceptions per attempt."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="official NFL formula over attempts",
        since=1999,
        note="Restates comp%, Y/A, TD%, and INT%; kept in the pool as a frozen exception.",
    ),
    _efficiency(
        name="qb_yards_per_attempt",
        label="Y/A",
        full_name="QB Yards Per Attempt",
        description="Passing yards divided by official pass attempts.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
    ),
    _efficiency(
        name="qb_touchdown_rate",
        label="TD %",
        full_name="QB Touchdown Rate",
        description="The share of pass attempts that scored touchdowns.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
    ),
    _efficiency(
        name="qb_interception_rate",
        label="INT %",
        full_name="QB Interception Rate",
        description="The share of pass attempts that were intercepted. Lower is better.",
        shape="rate",
        polarity="lower",
        source="D",
        denominator="pass attempts",
        since=1999,
    ),
    _efficiency(
        name="qb_completion_pct",
        label="Comp %",
        full_name="QB Completion Percentage",
        description="Completions divided by official pass attempts.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
    ),
)

QB_PRESSURE_METRICS: tuple[MetricDef, ...] = (
    _pressure(
        name="qb_sacks",
        label="Sacks",
        full_name="QB Sacks Taken",
        description="Times the quarterback was sacked. Avoiding sacks is a QB skill.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _pressure(
        name="qb_sack_yards_lost",
        label="Sack Yds Lost",
        full_name="QB Sack Yards Lost",
        description="Yards lost on sacks, shown as a positive number.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
        note="Stored negative upstream; the ETL normalizes the sign.",
    ),
    _pressure(
        name="qb_sack_rate",
        label="Sack Rate",
        full_name="QB Sack Rate",
        description=(
            "The share of dropbacks that ended in a sack. Lower is better — sack "
            "avoidance tracks quarterbacks more than offensive lines."
        ),
        shape="rate",
        polarity="lower",
        source="D",
        denominator="dropbacks",
        since=1999,
    ),
    _pressure(
        name="qb_sack_fumbles_lost",
        label="Sack Fum Lost",
        full_name="QB Sack Fumbles Lost",
        description="Strip-sack fumbles the defense recovered.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _pressure(
        name="qb_scramble_rate",
        label="Scramble %",
        full_name="QB Scramble Rate",
        description="Scrambles divided by dropbacks — the escape-and-run tendency.",
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
)

QB_RUSHING_METRICS: tuple[MetricDef, ...] = (
    _rushing(
        name="qb_carries",
        label="Carries",
        full_name="QB Carries",
        description="Official rushing attempts, including scrambles and kneel-downs.",
        shape="count",
        polarity="neutral",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_yards",
        label="Rush Yds",
        full_name="QB Rushing Yards",
        description="Rushing yards gained, including scramble yardage.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_yards_per_carry",
        label="Y/C",
        full_name="QB Yards Per Carry",
        description="Rushing yards divided by carries.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="carries",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_tds",
        label="Rush TDs",
        full_name="QB Rushing Touchdowns",
        description="Touchdowns scored on the ground.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_first_downs",
        label="Rush 1Ds",
        full_name="QB Rushing First Downs",
        description="First downs gained on quarterback runs.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_epa",
        label="Rush EPA",
        full_name="QB Rushing EPA",
        description="Expected points added on this quarterback's runs.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_epa_per_carry",
        label="EPA/Carry",
        full_name="QB EPA Per Carry",
        description="Rushing expected points added per carry.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="carries",
        since=1999,
    ),
    _rushing(
        name="qb_designed_carries",
        label="Designed Carries",
        full_name="QB Designed Carries",
        description="Called quarterback runs, excluding scrambles and kneel-downs.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_designed_rush_yards",
        label="Designed Rush Yds",
        full_name="QB Designed Rush Yards",
        description="Yards gained on called quarterback runs.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_designed_rush_epa",
        label="Designed Rush EPA",
        full_name="QB Designed Rush EPA",
        description=(
            "Expected points added on called quarterback runs, excluding scrambles and kneels."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        formula=(
            "sum(epa on rush plays where rusher_player_id == qb_id and qb_scramble == 0 and "
            "qb_kneel == 0)"
        ),
    ),
    _rushing(
        name="qb_designed_yards_per_carry",
        label="Designed Yds/Carry",
        full_name="QB Designed-Rush Yards Per Carry",
        description="Designed-run rushing yards divided by designed carries.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="designed carries",
        since=1999,
    ),
    _rushing(
        name="qb_designed_epa_per_carry",
        label="Designed EPA/Carry",
        full_name="QB Designed-Rush EPA Per Carry",
        description="Designed-run expected points added divided by designed carries.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="designed carries",
        since=1999,
    ),
    _rushing(
        name="qb_scrambles",
        label="Scrambles",
        full_name="QB Scrambles",
        description="Dropbacks on which the quarterback took off and ran.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_scramble_yards",
        label="Scramble Yds",
        full_name="QB Scramble Yards",
        description="Yards gained on scrambles.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_yards_per_scramble",
        label="Yds/Scramble",
        full_name="QB Yards Per Scramble",
        description="Average yards gained per scramble.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrambles",
        since=1999,
    ),
    _rushing(
        name="qb_kneels",
        label="Kneels",
        full_name="QB Kneel-Downs",
        description="Kneel-downs to run out the clock (excluded from efficiency rates).",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_2pt_conversions",
        label="2-Pt Rushes",
        full_name="QB Two-Point Conversion Rushes",
        description="Successful two-point conversions run in.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
)

QB_CLUTCH_METRICS: tuple[MetricDef, ...] = (
    _clutch(
        name="qb_wins",
        label="QB Wins",
        full_name="QB Wins",
        description=(
            "Wins in games where this quarterback was the primary passer. A team outcome, "
            "shown for context — never a rating input."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_losses",
        label="QB Losses",
        full_name="QB Losses",
        description="Losses in games where this quarterback was the primary passer.",
        shape="count",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_ties",
        label="QB Ties",
        full_name="QB Ties",
        description="Ties in games where this quarterback was the primary passer.",
        shape="count",
        polarity="neutral",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_win_pct",
        label="QB Win %",
        full_name="QB Win Percentage",
        description=(
            "Share of primary-QB games won, counting a tie as half a win. Feeds only the "
            "separate outcome layer, never the performance ratings."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="primary-QB games",
        since=1999,
    ),
    _clutch(
        name="qb_fourth_quarter_comeback",
        label="4QC",
        full_name="QB Fourth-Quarter Comeback",
        description=(
            "Credit for a game in which the quarterback's team trailed in the fourth "
            "quarter and he led it to a win."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_fourth_quarter_comebacks",
        label="4QC",
        full_name="QB Fourth-Quarter Comebacks",
        description=(
            "Games in which the quarterback's team trailed in the fourth quarter and he "
            "led it to a win."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_game_winning_drive",
        label="GWD",
        full_name="QB Game-Winning Drive",
        description=(
            "Credit for leading a drive that put the team ahead for good in the fourth "
            "quarter or overtime of a win."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_game_winning_drives",
        label="GWD",
        full_name="QB Game-Winning Drives",
        description=(
            "Drives led that put the team ahead for good in the fourth quarter or "
            "overtime of games the team won."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
)

QB_TURNOVER_METRICS: tuple[MetricDef, ...] = (
    _turnovers(
        name="qb_td_int_differential",
        label="TD-INT Diff",
        full_name="QB Touchdown-Interception Differential",
        description="Touchdown passes minus interceptions across the season.",
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _turnovers(
        name="qb_rushing_fumbles",
        label="Rush Fumbles",
        full_name="QB Rushing Fumbles",
        description="Fumbles on quarterback runs, whether or not lost.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _turnovers(
        name="qb_rushing_fumbles_lost",
        label="Rush Fum Lost",
        full_name="QB Rushing Fumbles Lost",
        description="Fumbles lost on quarterback runs.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
)

QB_METRICS: tuple[MetricDef, ...] = (
    QB_RATING_METRICS
    + QB_IDENTITY_METRICS
    + QB_VOLUME_METRICS
    + QB_EFFICIENCY_METRICS
    + QB_PRESSURE_METRICS
    + QB_RUSHING_METRICS
    + QB_CLUTCH_METRICS
    + QB_TURNOVER_METRICS
)
