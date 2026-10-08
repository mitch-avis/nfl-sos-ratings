"""Team offense passing and receiving definitions.

Passing volume, efficiency, depth, and the receiving side of the same plays.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_off_pass = section("team", "Offense", "Passing")
_off_recv = section("team", "Offense", "Receiving")

_TARGETS_GAP_NOTE = (
    "Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no "
    "incomplete passes."
)

OFFENSE_PASSING_METRICS: tuple[MetricDef, ...] = (
    _off_pass(
        name="passing_yards",
        label="Pass Yds",
        full_name="Passing Yards",
        description=(
            "Gross passing yards on completions. Sack yardage is not subtracted here — see "
            "net passing yards for the NFL.com team convention."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_epa",
        label="Pass EPA",
        full_name="Passing EPA",
        description=(
            "Expected points added on dropbacks. Positive means the passing game moved the "
            "team toward scoring more than an average offense would have."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_tds",
        label="Pass TDs",
        full_name="Passing Touchdowns",
        description="Touchdown passes thrown by the team.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_first_downs",
        label="Pass 1Ds",
        full_name="Passing First Downs",
        description="First downs gained through the air.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_cpoe",
        label="Pass CPOE",
        full_name="Completion Percentage Above Expectation",
        description=(
            "How much higher the team's completion rate was than the difficulty of its "
            "throws would predict, in percentage points. Positive means more accurate than "
            "expected."
        ),
        shape="avg",
        polarity="higher",
        source="PBP +TS",
        denominator="pass attempts (model-expected completions)",
        since=2006,
    ),
    _off_pass(
        name="sacks_suffered",
        label="Sacks Taken",
        full_name="Sacks Suffered",
        description="Times the team's quarterbacks were sacked. Fewer is better.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_interceptions",
        label="INTs Thrown",
        full_name="Interceptions Thrown",
        description="Passes intercepted by the defense. Fewer is better.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="sack_fumbles_lost",
        label="Sack Fum Lost",
        full_name="Sack Fumbles Lost",
        description="Fumbles lost to the defense on sack plays (strip-sacks).",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="attempts",
        label="Att",
        full_name="Pass Attempts",
        description=(
            "Official pass attempts. Excludes sacks and two-point conversion tries, "
            "matching the league's official counting."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
        formula="pass_attempt - sack, excluding two_point_attempt",
    ),
    _off_pass(
        name="completions",
        label="Comp",
        full_name="Completions",
        description="Completed passes by the team.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="completion_pct",
        label="Comp %",
        full_name="Completion Percentage",
        description="The share of official pass attempts that were completed.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="net_passing_yards",
        label="Net Pass Yds",
        full_name="Net Passing Yards",
        description=(
            "Passing yards minus yards lost to sacks — the NFL.com team convention for "
            "passing offense."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        formula="passing_yards - sack_yards_lost (positive magnitude)",
    ),
    _off_pass(
        name="dropbacks",
        label="Dropbacks",
        full_name="Dropbacks",
        description=(
            "Pass attempts plus sacks plus scrambles — every play that started as a pass. "
            "The natural denominator for passing efficiency."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="sack_yards_lost",
        label="Sack Yds Lost",
        full_name="Sack Yards Lost",
        description="Yards lost on sacks, shown as a positive number. Fewer is better.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        note="team_stats stores this negative upstream; the ETL normalizes the sign.",
    ),
    _off_pass(
        name="sack_rate_per_dropback",
        label="Sack Rate",
        full_name="Sack Rate",
        description="The share of dropbacks that ended in a sack. Lower is better.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="scrambles",
        label="Scrambles",
        full_name="QB Scrambles",
        description="Dropbacks on which the quarterback took off and ran.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="scramble_yards",
        label="Scramble Yds",
        full_name="Scramble Yards",
        description="Rushing yards gained on quarterback scrambles.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="passing_air_yards",
        label="Air Yds",
        full_name="Passing Air Yards",
        description=(
            "Total distance the ball traveled past the line of scrimmage on all throws, "
            "including incompletions — a measure of how far downfield the team attacks."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=2006,
    ),
    _off_pass(
        name="passing_yards_after_catch",
        label="YAC",
        full_name="Passing Yards After Catch",
        description="Yards receivers gained after catching the ball.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="air_yards_per_attempt",
        label="aDOT",
        full_name="Air Yards Per Attempt (aDOT)",
        description=(
            "Average depth of target: how far downfield the average throw traveled. A "
            "style stat, not a quality grade."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="pass attempts",
        since=2006,
    ),
    _off_pass(
        name="yac_per_completion",
        label="YAC/Comp",
        full_name="Yards After Catch Per Completion",
        description="Average yards gained after the catch on completed passes.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="completions",
        since=1999,
    ),
    _off_pass(
        name="yards_per_attempt",
        label="Y/A",
        full_name="Yards Per Attempt",
        description="Passing yards divided by official pass attempts.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="net_yards_per_attempt",
        label="NY/A",
        full_name="Net Yards Per Attempt",
        description=(
            "Passing yards minus sack yards, divided by attempts plus sacks — yards per "
            "dropback-style efficiency that charges the offense for sacks."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts + sacks",
        since=1999,
        formula="(passing_yards - sack_yards_lost) / (attempts + sacks)",
    ),
    _off_pass(
        name="adjusted_net_yards_per_attempt",
        label="ANY/A",
        full_name="Adjusted Net Yards Per Attempt",
        description=(
            "The best single conventional passing stat: yards per attempt with a +20-yard "
            "bonus per touchdown, a -45-yard penalty per interception, and sacks counted "
            "against."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts + sacks",
        since=1999,
        formula="(yards + 20*TD - 45*INT - sack_yards) / (attempts + sacks)",
    ),
    _off_pass(
        name="yards_per_dropback",
        label="Yds/DB",
        full_name="Yards Per Dropback",
        description="Passing yards divided by dropbacks, so sacks and scrambles count.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="epa_per_dropback",
        label="EPA/DB",
        full_name="EPA Per Dropback",
        description="Passing expected points added per dropback — passing efficiency.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="pass_success_rate",
        label="Pass Success %",
        full_name="Passing Success Rate",
        description="The share of dropbacks that improved the team's expected points.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="team_passer_rating",
        label="Passer Rating",
        full_name="Team Passer Rating",
        description=(
            "The classic NFL passer-rating formula (0 to 158.3) computed on the team's "
            "combined passing totals."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="official NFL formula over attempts",
        since=1999,
    ),
    _off_pass(
        name="passing_td_rate_per_attempt",
        label="TD %",
        full_name="Passing Touchdown Rate",
        description="The share of pass attempts that scored touchdowns.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="int_rate_per_attempt",
        label="INT %",
        full_name="Interception Rate",
        description="The share of pass attempts that were intercepted. Lower is better.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="explosive_pass_rate",
        label="Explosive Pass %",
        full_name="Explosive Pass Rate",
        description="Completions of 20+ yards divided by dropbacks.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="deep_attempt_rate",
        label="Deep Att %",
        full_name="Deep Attempt Rate",
        description="The share of attempts thrown deep (16+ air yards). A style stat.",
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="longest_pass",
        label="Long Pass",
        full_name="Longest Completed Pass",
        description="The team's longest completed pass of the season, in yards.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="sack_fumbles",
        label="Sack Fumbles",
        full_name="Sack Fumbles",
        description="Fumbles on sack plays, whether or not the team lost the ball.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_2pt_conversions",
        label="2-Pt Passes",
        full_name="Two-Point Conversion Passes",
        description="Successful two-point conversions thrown.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="air_epa_total",
        label="Air EPA",
        full_name="Air EPA",
        description=(
            "The share of passing EPA created by the throw itself (distance and placement) "
            "rather than the run after the catch."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="yac_epa_total",
        label="YAC EPA",
        full_name="Yards-After-Catch EPA",
        description="The share of passing EPA created after the catch by the receivers.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="xyac_per_completion",
        label="xYAC/Comp",
        full_name="Expected Yards After Catch Per Completion",
        description=(
            "How many yards after the catch an average receiver would have gained on the "
            "same catches, per the nflverse model."
        ),
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="completions",
        since=2006,
    ),
    _off_pass(
        name="yac_over_expected_per_completion",
        label="YAC +/-",
        full_name="Yards After Catch Over Expected",
        description="Actual minus expected yards after catch per completion.",
        shape="avg",
        polarity="higher",
        source="PBP",
        denominator="completions",
        since=2006,
    ),
)

OFFENSE_RECEIVING_METRICS: tuple[MetricDef, ...] = (
    _off_recv(
        name="targets",
        label="Targets",
        full_name="Targets",
        description=(
            "Pass attempts thrown to an intended receiver. Throwaways and spikes are attempts "
            "but not targets."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
        note=_TARGETS_GAP_NOTE,
    ),
    _off_recv(
        name="receptions",
        label="Receptions",
        full_name="Receptions",
        description="Team catches — the same number as completions, receiving-side view.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="completions",
    ),
    _off_recv(
        name="receiving_yards",
        label="Rec Yds",
        full_name="Receiving Yards",
        description=(
            "Team receiving yards. At team level this equals gross passing yards exactly "
            "(verified against nflverse data)."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_yards",
    ),
    _off_recv(
        name="receiving_tds",
        label="Rec TDs",
        full_name="Receiving Touchdowns",
        description="Touchdown catches — the same number as passing touchdowns.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_tds",
    ),
    _off_recv(
        name="receiving_air_yards",
        label="Rec Air Yds",
        full_name="Receiving Air Yards",
        description="Air yards on targets — the same number as passing air yards.",
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=2006,
        duplicate_of="passing_air_yards",
    ),
    _off_recv(
        name="receiving_yards_after_catch",
        label="Rec YAC",
        full_name="Receiving Yards After Catch",
        description="Yards gained after the catch — receiving-side view of team YAC.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_yards_after_catch",
    ),
    _off_recv(
        name="receiving_first_downs",
        label="Rec 1Ds",
        full_name="Receiving First Downs",
        description="First downs on catches — the same number as passing first downs.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_first_downs",
    ),
    _off_recv(
        name="receiving_fumbles",
        label="Rec Fumbles",
        full_name="Receiving Fumbles",
        description="Fumbles by receivers after the catch, whether or not lost.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_recv(
        name="receiving_fumbles_lost",
        label="Rec Fum Lost",
        full_name="Receiving Fumbles Lost",
        description="Fumbles lost to the defense after a catch.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_recv(
        name="catch_rate",
        label="Catch %",
        full_name="Catch Rate",
        description="Receptions divided by targets — the receiving view of completion rate.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="targets",
        since=1999,
        note=_TARGETS_GAP_NOTE,
    ),
)

__all__ = [
    "OFFENSE_PASSING_METRICS",
    "OFFENSE_RECEIVING_METRICS",
]
