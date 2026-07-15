"""Quarterback metric definitions — every QB stat published or planned.

Entries with ``status="implemented"`` cover every QB column the pipeline
writes today. Entries with ``status="planned"`` encode the full QB catalog in
[docs/qb-stats-catalog.md](../../docs/qb-stats-catalog.md).
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_ratings = section("qb", "Schedule-Adjusted Ratings")
_identity = section("qb", "Identity & Availability")
_volume = section("qb", "Passing Volume")
_efficiency = section("qb", "Passing Efficiency")
_advanced = section("qb", "Advanced & Expected")
_pressure = section("qb", "Pressure, Sacks & Pocket")
_rushing = section("qb", "Rushing")
_clutch = section("qb", "Scoring, Clutch & Outcomes")
_turnovers = section("qb", "Turnovers & Ball Security")

QB_RATING_METRICS: tuple[MetricDef, ...] = (
    _ratings(
        name="QSaCR",
        label="QSaCR",
        full_name="QB Schedule-Adjusted Composite Rating",
        description=(
            "The site's headline quarterback rating. In Stage 2 of the methodology overhaul it "
            "is a frozen-weight blend of standardized adjusted EPA/dropback, CPOE, sack rate, "
            "and TD-INT margin rate, while wins and late-game results stay in QOutcome only. "
            "0 is league average; positive is better."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        note=(
            "Frozen Stage 2 weights: adj_qb_epa_per_dropback=0.6687790473858877, "
            "adj_qb_completion_percentage_above_expectation=0.21464381898367774, "
            "adj_qb_sack_rate=0.06725872314827445, adj_qb_td_int_margin_rate=0.04931841048216012. "
            "Target: next-season adj_qb_epa_per_dropback with dropback-weighted WLS. Fit window: "
            "2006-2025 season pairs because adjusted CPOE is unavailable before 2006. Held-out "
            "leave-one-season-out metrics: weighted RMSE 0.093348 vs equal-weight RMSE 0.093446; "
            "weighted MAE 0.073908 vs equal-weight MAE 0.074351. Fitting command: uv run python "
            "-m nfl_sos_ratings.composite_weights. Refit only with an explicit maintainer-"
            "approved Stage 2 methodology update."
        ),
    ),
    _ratings(
        name="QSaOR",
        label="QSaOR",
        full_name="QB Schedule-Adjusted Offense Rating",
        description=(
            "Passing performance after adjusting for the defenses actually faced, using the "
            "simultaneous ridge estimate of QB EPA per dropback. This is the published "
            "opponent-adjusted QB quality signal in Stage 1."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="QRaw",
        label="QRaw",
        full_name="QB Raw Performance Composite",
        description=(
            "The unadjusted composite of the core passing stat pool, before any schedule "
            "context is applied."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="QSoS",
        label="QSoS",
        full_name="QB Strength of Schedule",
        description=(
            "How tough the defenses this quarterback faced were, measured as the mean faced-"
            "defense coefficient from the simultaneous ridge QB solve. Higher means a harder "
            "slate — it describes the schedule, not the quarterback's play."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        contextual=True,
    ),
    _ratings(
        name="adj_def_qb_epa_per_dropback_faced",
        label="Faced Adj Def EPA/DB",
        full_name="Faced Defense Rating on QB EPA Per Dropback",
        description=(
            "The mean defense-side ridge coefficient, on QB EPA per dropback, for the "
            "defenses this quarterback actually faced. This is the raw schedule-context "
            "input behind QSoS."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        contextual=True,
    ),
    _ratings(
        name="QOutcome",
        label="QOutcome",
        full_name="QB Outcome Layer",
        description=(
            "A secondary signal built from wins and late-game results such as "
            "fourth-quarter comebacks and game-winning drives. Descriptive-only: kept "
            "separate so results never contaminate the performance ratings."
        ),
        shape="score",
        polarity="higher",
        source="D",
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
            "Whether the quarterback meets the project's minimum-dropback threshold to be "
            "ranked on the league-wide QBs page."
        ),
        shape="flag",
        polarity="neutral",
        source="D",
    ),
    _identity(
        name="qb_games_started",
        label="QB Starts",
        full_name="QB Games Started",
        description="Games in which this quarterback was the team's primary passer.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
        status="planned",
    ),
    _identity(
        name="qb_snap_share",
        label="Snap Share",
        full_name="QB Snap Share",
        description="The quarterback's snaps divided by the team's offensive snaps.",
        shape="rate",
        polarity="neutral",
        source="SNP",
        denominator="team offensive snaps",
        since=2012,
        status="planned",
    ),
    _identity(
        name="qb_plays",
        label="QB Plays",
        full_name="QB Plays",
        description="Dropbacks plus designed carries — the quarterback's total usage.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
        status="planned",
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
        name="qb_net_pass_yards",
        label="Net Pass Yds",
        full_name="QB Net Passing Yards",
        description="Passing yards minus yards lost to sacks.",
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_passing_first_downs",
        label="Pass 1Ds",
        full_name="QB Passing First Downs",
        description="First downs gained on this quarterback's passes.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_passing_air_yards",
        label="Air Yds",
        full_name="QB Passing Air Yards",
        description=(
            "Total distance the ball traveled past the line of scrimmage on all throws, "
            "including incompletions."
        ),
        shape="count",
        polarity="neutral",
        source="PLS",
        since=2006,
        status="planned",
    ),
    _volume(
        name="qb_passing_yards_after_catch",
        label="YAC",
        full_name="QB Passing Yards After Catch",
        description="Yards receivers gained after catching this quarterback's passes.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_passing_2pt_conversions",
        label="2-Pt Passes",
        full_name="QB Two-Point Conversion Passes",
        description="Successful two-point conversions thrown.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_explosive_completions",
        label="20+ Yd Comp",
        full_name="QB Explosive Completions",
        description="Completions that gained 20 or more yards.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_longest_completion",
        label="Long",
        full_name="QB Longest Completion",
        description="The quarterback's longest completed pass, in yards.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_spikes",
        label="Spikes",
        full_name="QB Spikes",
        description="Clock-stopping spikes (excluded from accuracy rates).",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
        status="planned",
    ),
    _volume(
        name="qb_throwaways",
        label="Throwaways",
        full_name="QB Throwaways",
        description="Intentional incompletions thrown away under pressure, per PFR charting.",
        shape="count",
        polarity="neutral",
        source="PFR",
        since=2018,
        status="planned",
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
    _efficiency(
        name="qb_net_yards_per_attempt",
        label="NY/A",
        full_name="QB Net Yards Per Attempt",
        description="Passing yards minus sack yards, divided by attempts plus sacks.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts + sacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_adjusted_yards_per_attempt",
        label="AY/A",
        full_name="QB Adjusted Yards Per Attempt",
        description=(
            "Yards per attempt with a +20-yard bonus per touchdown and a -45-yard penalty "
            "per interception (ANY/A without the sack terms)."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
        duplicate_of="qb_any_a",
        status="planned",
    ),
    _efficiency(
        name="qb_success_rate",
        label="Success %",
        full_name="QB Success Rate",
        description=(
            "The share of dropbacks that improved the team's expected points — a "
            "consistency measure that ignores how big each play was."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_first_down_rate_per_dropback",
        label="1D %/DB",
        full_name="QB First Down Rate Per Dropback",
        description="Passing first downs divided by dropbacks.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_adot",
        label="aDOT",
        full_name="QB Average Depth of Target",
        description=(
            "How far downfield the average throw traveled. A style measure of "
            "aggressiveness, not a quality grade."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="pass attempts",
        since=2006,
        status="planned",
    ),
    _efficiency(
        name="qb_pacr",
        label="PACR",
        full_name="QB Passing Air Conversion Ratio",
        description=(
            "Passing yards divided by air yards — how efficiently intended depth turned "
            "into actual yards."
        ),
        shape="rate",
        polarity="higher",
        source="PLS",
        denominator="air yards",
        since=2006,
        status="planned",
    ),
    _efficiency(
        name="qb_explosive_pass_rate",
        label="Explosive %",
        full_name="QB Explosive Pass Rate",
        description="Completions of 20+ yards divided by dropbacks.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_deep_attempt_rate",
        label="Deep Att %",
        full_name="QB Deep Attempt Rate",
        description="The share of attempts thrown deep. A style stat.",
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="pass attempts",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_air_epa_per_dropback",
        label="Air EPA/DB",
        full_name="QB Air EPA Per Dropback",
        description="The share of EPA created by the throw itself, per dropback.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_yac_epa_per_dropback",
        label="YAC EPA/DB",
        full_name="QB YAC EPA Per Dropback",
        description=(
            "The share of EPA created after the catch, per dropback — largely a "
            "supporting-cast measure."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        status="planned",
    ),
    _efficiency(
        name="qb_xyac_per_completion",
        label="xYAC/Comp",
        full_name="QB Expected YAC Per Completion",
        description=(
            "Yards after catch an average receiver would have gained on the same catches "
            "— context for how much help the quarterback received."
        ),
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="completions",
        since=2006,
        status="planned",
    ),
    _efficiency(
        name="qb_yac_over_expected",
        label="YAC +/-",
        full_name="QB YAC Over Expected",
        description="Actual minus expected yards after catch on this QB's completions.",
        shape="avg",
        polarity="higher",
        source="PBP",
        denominator="completions",
        since=2006,
        status="planned",
    ),
    _efficiency(
        name="qb_wpa_total",
        label="WPA",
        full_name="QB Win Probability Added",
        description=(
            "How much this quarterback's dropbacks moved the team's chance of winning, "
            "summed over the season."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        status="planned",
    ),
)

QB_ADVANCED_METRICS: tuple[MetricDef, ...] = (
    _advanced(
        name="qb_qbr_total",
        label="QBR",
        full_name="ESPN Total QBR",
        description=(
            "ESPN's 0-100 quarterback rating, which weights plays by importance and "
            "includes ESPN's own opponent adjustment. Shown for reference alongside this "
            "site's schedule-adjusted ratings."
        ),
        shape="score",
        polarity="higher",
        source="QBR",
        since=2006,
        status="planned",
    ),
    _advanced(
        name="qb_qbr_raw",
        label="Raw QBR",
        full_name="ESPN Raw QBR",
        description=(
            "ESPN's QBR before its opponent adjustment — the preferred version to feed "
            "this project's own schedule adjustment, so opponent strength is not counted "
            "twice."
        ),
        shape="score",
        polarity="higher",
        source="QBR",
        since=2006,
        status="planned",
    ),
    _advanced(
        name="qb_pts_added",
        label="Pts Added",
        full_name="ESPN Points Added",
        description="ESPN's estimate of points contributed above an average quarterback.",
        shape="count",
        polarity="higher",
        source="QBR",
        since=2006,
        status="planned",
    ),
    _advanced(
        name="qb_qbr_plays",
        label="QBR Plays",
        full_name="ESPN QBR Plays",
        description="ESPN's action-play count, a useful cross-check on dropbacks.",
        shape="count",
        polarity="neutral",
        source="QBR",
        since=2006,
        status="planned",
    ),
    _advanced(
        name="qb_avg_time_to_throw",
        label="Time to Throw",
        full_name="NGS Average Time to Throw",
        description=(
            "Seconds from snap to release, from player tracking. Quick releases mitigate "
            "pressure; a style measure."
        ),
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_avg_completed_air_yards",
        label="Comp Air Yds",
        full_name="NGS Average Completed Air Yards",
        description="Average downfield distance of completed passes, from tracking.",
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="completions (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_avg_intended_air_yards",
        label="Intended Air Yds",
        full_name="NGS Average Intended Air Yards",
        description="Average downfield distance of all throws, from tracking (NGS aDOT).",
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_avg_air_yards_differential",
        label="Air Yds Diff",
        full_name="NGS Air Yards Differential",
        description="Completed minus intended air yards — how much depth is completed.",
        shape="avg",
        polarity="higher",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_aggressiveness",
        label="AGG %",
        full_name="NGS Aggressiveness",
        description=(
            "The share of throws into tight coverage (a defender within a yard), from "
            "tracking. A risk-style measure."
        ),
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_avg_air_yards_to_sticks",
        label="AYTS",
        full_name="NGS Air Yards to the Sticks",
        description=(
            "How far beyond (or short of) the first-down marker the average throw "
            "traveled. Positive means attacking past the sticks."
        ),
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_max_completed_air_distance",
        label="Max Air Dist",
        full_name="NGS Max Completed Air Distance",
        description="The longest true air distance on a completion, from tracking.",
        shape="count",
        polarity="higher",
        source="NGS",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_expected_completion_percentage",
        label="xCOMP %",
        full_name="NGS Expected Completion Percentage",
        description=(
            "The completion rate an average quarterback would post on the same throws, "
            "from tracking — context for throw difficulty."
        ),
        shape="avg",
        polarity="neutral",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_ngs_cpoe",
        label="NGS CPOE",
        full_name="NGS Completion Percentage Above Expectation",
        description=(
            "Tracking-based completion rate above expectation. A different model from the "
            "play-by-play CPOE shown elsewhere — the two are never mixed in ratings."
        ),
        shape="avg",
        polarity="higher",
        source="NGS",
        denominator="pass attempts (tracking)",
        since=2016,
        status="planned",
    ),
    _advanced(
        name="qb_on_tgt_pct",
        label="On-Target %",
        full_name="PFR On-Target Percentage",
        description="The share of throws charted as accurate to the receiver, per PFR.",
        shape="rate",
        polarity="higher",
        source="PFR",
        denominator="charted attempts",
        since=2019,
        status="planned",
    ),
    _advanced(
        name="qb_bad_throw_pct",
        label="Bad Throw %",
        full_name="PFR Bad Throw Percentage",
        description="The share of throws charted as poor, per PFR. Lower is better.",
        shape="rate",
        polarity="lower",
        source="PFR",
        denominator="charted attempts",
        since=2018,
        status="planned",
    ),
    _advanced(
        name="qb_batted_balls",
        label="Batted",
        full_name="PFR Batted Balls",
        description="Passes knocked down at the line of scrimmage, per PFR charting.",
        shape="count",
        polarity="lower",
        source="PFR",
        since=2019,
        status="planned",
    ),
    _advanced(
        name="qb_drop_adjusted_comp_pct",
        label="Drop-Adj Comp %",
        full_name="Drop-Adjusted Completion Percentage",
        description=(
            "Completion rate after crediting receiver drops back to the quarterback and "
            "removing throwaways, spikes, and batted balls — accuracy isolated from "
            "supporting-cast noise."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="attempts - throwaways - spikes - batted balls",
        since=2019,
        formula="(completions + drops) / (attempts - throwaways - spikes - batted_balls)",
        status="planned",
    ),
    _advanced(
        name="qb_rpo_plays",
        label="RPO Plays",
        full_name="PFR Run-Pass Option Plays",
        description="Run-pass option plays run with this quarterback, per PFR charting.",
        shape="count",
        polarity="neutral",
        source="PFR",
        since=2019,
        status="planned",
    ),
    _advanced(
        name="qb_pa_pass_att",
        label="PA Att",
        full_name="PFR Play-Action Attempts",
        description=(
            "Pass attempts off play-action fakes, per PFR charting. Only tracked from "
            "2019 through 2023."
        ),
        shape="count",
        polarity="neutral",
        source="PFR",
        since=2019,
        note="Discontinued upstream after 2023.",
        status="planned",
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
        ratings_eligible=True,
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
        ratings_eligible=True,
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
        name="qb_sack_fumbles",
        label="Sack Fumbles",
        full_name="QB Sack Fumbles",
        description="Fumbles on sacks, whether or not the ball was lost.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _pressure(
        name="qb_qb_hits_taken",
        label="Hits Taken",
        full_name="QB Hits Taken",
        description="Hits absorbed beyond sacks.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
        status="planned",
    ),
    _pressure(
        name="qb_pressure_rate_faced",
        label="Pressure % Faced",
        full_name="QB Pressure Rate Faced",
        description=(
            "The share of dropbacks under pressure, per PFR charting. Context for the "
            "efficiency stats: some quarterbacks live under siege."
        ),
        shape="rate",
        polarity="lower",
        source="PFR",
        denominator="dropbacks",
        since=2018,
        status="planned",
    ),
    _pressure(
        name="qb_blitz_rate_faced",
        label="Blitz % Faced",
        full_name="QB Blitz Rate Faced",
        description="The share of dropbacks against a blitz, per PFR charting.",
        shape="rate",
        polarity="neutral",
        source="PFR",
        denominator="dropbacks",
        since=2018,
        status="planned",
    ),
    _pressure(
        name="qb_pocket_time",
        label="Pocket Time",
        full_name="QB Average Pocket Time",
        description="Average seconds the pocket held before pressure or throw, per PFR.",
        shape="avg",
        polarity="neutral",
        source="PFR",
        denominator="dropbacks (charted)",
        since=2018,
        status="planned",
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
        status="planned",
    ),
    _pressure(
        name="qb_sack_rate_vs_pressure",
        label="Sack %/Pressure",
        full_name="QB Sacks Per Pressure",
        description=(
            "Sacks divided by pressures faced — how often pressure turned into a sack. "
            "Lower means better escape ability."
        ),
        shape="rate",
        polarity="lower",
        source="D",
        denominator="pressures faced",
        since=2018,
        status="planned",
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
        status="planned",
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
        status="planned",
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
        status="planned",
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
        status="planned",
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
        status="planned",
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
        status="planned",
    ),
    _rushing(
        name="qb_rush_success_rate",
        label="Rush Success %",
        full_name="QB Rushing Success Rate",
        description="The share of designed QB runs that improved expected points.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="designed carries",
        since=1999,
        status="planned",
    ),
    _rushing(
        name="qb_explosive_rush_rate",
        label="Explosive Rush %",
        full_name="QB Explosive Rush Rate",
        description="Runs of 10+ yards divided by carries.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="carries",
        since=1999,
        status="planned",
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
    _rushing(
        name="qb_total_epa_per_play",
        label="Total EPA/Play",
        full_name="QB Total EPA Per Play",
        description=(
            "Passing plus rushing expected points added, divided by dropbacks plus "
            "designed carries — the headline dual-threat efficiency stat."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks + designed carries",
        since=1999,
        formula="(passing_epa + rushing_epa) / (dropbacks + designed_carries)",
        status="planned",
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
    _clutch(
        name="qb_total_tds",
        label="Total TDs",
        full_name="QB Total Touchdowns",
        description="Passing plus rushing touchdowns — all scores the QB accounted for.",
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_total_td_rate",
        label="Total TD %",
        full_name="QB Total Touchdown Rate",
        description="Total touchdowns divided by dropbacks plus designed carries.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks + designed carries",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_red_zone_td_pass_pct",
        label="RZ TD %",
        full_name="QB Red Zone Touchdown Pass Percentage",
        description="Touchdown passes divided by attempts inside the opponent's 20.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="red-zone pass attempts",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_red_zone_epa_per_dropback",
        label="RZ EPA/DB",
        full_name="QB Red Zone EPA Per Dropback",
        description="Expected points added per dropback inside the opponent's 20.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="red-zone dropbacks",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_2pt_conversions",
        label="2-Pt Conv",
        full_name="QB Two-Point Conversions",
        description="Two-point conversions passed or run in.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_late_close_epa_per_dropback",
        label="Clutch EPA/DB",
        full_name="QB Late & Close EPA Per Dropback",
        description=(
            "EPA per dropback in the fourth quarter or overtime of one-score games — "
            "performance when it mattered most."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="late-and-close dropbacks",
        since=1999,
        status="planned",
    ),
    _clutch(
        name="qb_third_down_conversion_rate",
        label="3rd Down Conv %",
        full_name="QB Third Down Conversion Rate",
        description="Third-down dropbacks converted into first downs or touchdowns.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="third-down dropbacks",
        since=1999,
        status="planned",
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
        name="qb_fumbles",
        label="Fumbles",
        full_name="QB Fumbles",
        description="Sack and rushing fumbles combined, whether or not lost.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
        status="planned",
    ),
    _turnovers(
        name="qb_fumbles_lost",
        label="Fumbles Lost",
        full_name="QB Fumbles Lost",
        description="Sack and rushing fumbles the defense recovered.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
        status="planned",
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
    _turnovers(
        name="qb_giveaways",
        label="Giveaways",
        full_name="QB Giveaways",
        description="Interceptions plus fumbles lost.",
        shape="count",
        polarity="lower",
        source="D",
        since=1999,
        status="planned",
    ),
    _turnovers(
        name="qb_giveaway_rate",
        label="Giveaway %",
        full_name="QB Giveaway Rate",
        description="Giveaways divided by dropbacks plus designed carries.",
        shape="rate",
        polarity="lower",
        source="D",
        denominator="dropbacks + designed carries",
        since=1999,
        status="planned",
    ),
    _turnovers(
        name="qb_turnover_epa",
        label="TO EPA",
        full_name="QB Turnover EPA",
        description=(
            "Expected points lost on this quarterback's giveaway plays — how costly the "
            "turnovers were, not just how many."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        note="Values are negative; closer to zero means cheaper turnovers.",
        status="planned",
    ),
)

QB_METRICS: tuple[MetricDef, ...] = (
    QB_RATING_METRICS
    + QB_IDENTITY_METRICS
    + QB_VOLUME_METRICS
    + QB_EFFICIENCY_METRICS
    + QB_ADVANCED_METRICS
    + QB_PRESSURE_METRICS
    + QB_RUSHING_METRICS
    + QB_CLUTCH_METRICS
    + QB_TURNOVER_METRICS
)
