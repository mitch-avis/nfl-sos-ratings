"""Team overall definitions.

Results, margins, and the stats that span offense and defense.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_overall = section("team", "Overall")

OVERALL_METRICS: tuple[MetricDef, ...] = (
    _overall(
        name="team",
        label="Team",
        full_name="Team",
        description="The team's standard NFL abbreviation, such as KC or PHI.",
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="game_id",
        label="Game ID",
        full_name="Game ID",
        description="The unique nflverse identifier for one game, useful for deep links.",
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="week",
        label="Week",
        full_name="Week",
        description=(
            "The regular-season week: on a game row, the week the game was played; on a "
            "rating-history row, the last week of games the rating includes."
        ),
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="opponent_team",
        label="Opponent",
        full_name="Opponent Team",
        description="The opposing team in this game or in the summarized opponent row.",
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="is_home",
        label="Home",
        full_name="Home Team Flag",
        description="Whether this team was the home side in the game represented by the row.",
        shape="flag",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="games_played",
        label="Games",
        full_name="Games Played",
        description="Regular-season games played. Rate stats divide by this number.",
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="games",
        label="Games",
        full_name="Games",
        description="The number of games represented by this row of the table.",
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="wins",
        label="Wins",
        full_name="Wins",
        description="Regular-season games won. Treated as an outcome, not a rating input.",
        shape="count",
        polarity="higher",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="losses",
        label="Losses",
        full_name="Losses",
        description="Regular-season games lost. Treated as an outcome, not a rating input.",
        shape="count",
        polarity="lower",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="ties",
        label="Ties",
        full_name="Ties",
        description="Regular-season games that ended tied after overtime.",
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="win_pct",
        label="Win %",
        full_name="Win Percentage",
        description=(
            "Share of games won, counting a tie as half a win: (wins + 0.5 x ties) divided "
            "by games played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="games played",
        since=1999,
        formula="(wins + 0.5 * ties) / games_played",
        percent=True,
    ),
    _overall(
        name="win_value",
        label="Win Value",
        full_name="Win Value",
        description=(
            "The game result as a number: 1 for a win, 0.5 for a tie, 0 for a loss. In "
            "summary rows it is the average across the games included."
        ),
        shape="avg",
        polarity="higher",
        source="D",
        denominator="games included",
        since=1999,
    ),
    _overall(
        name="points_for",
        label="Points For",
        full_name="Points Scored",
        description="Total points the team scored, including all offense, defense, and kicks.",
        shape="count",
        polarity="higher",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="points_allowed",
        label="Points Allowed",
        full_name="Points Allowed",
        description="Total points the team gave up. Fewer points allowed is better.",
        shape="count",
        polarity="lower",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="point_margin",
        label="Point Margin",
        full_name="Point Margin",
        description=(
            "Points scored minus points allowed. A restatement of the two point totals, "
            "kept for display because it is the most intuitive whole-team summary."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
        formula="points_for - points_allowed",
        note="Restates points_for and points_allowed.",
    ),
    _overall(
        name="turnover_margin",
        label="TO Margin",
        full_name="Turnover Margin",
        description=(
            "Takeaways minus giveaways. Positive means the team won the turnover battle "
            "across the season."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="points_per_offensive_snap",
        label="Points/Off Snap",
        full_name="Points Per Offensive Snap",
        description=(
            "Points scored divided by offensive snaps — scoring efficiency that does not "
            "reward teams simply for running more plays."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="offensive snaps",
        since=1999,
    ),
    _overall(
        name="points_allowed_per_defensive_snap",
        label="Points Allowed/Def Snap",
        full_name="Points Allowed Per Defensive Snap",
        description=(
            "Points given up divided by defensive snaps — defensive scoring efficiency that "
            "does not punish defenses simply for facing more plays."
        ),
        shape="rate",
        polarity="lower",
        source="D",
        denominator="defensive snaps",
        since=1999,
    ),
    _overall(
        name="total_yards_differential",
        label="Total Yds Diff",
        full_name="Total Yards Differential",
        description="Yards gained minus yards allowed across the season.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="penalty_differential",
        label="Penalty Diff",
        full_name="Penalty Differential",
        description="Opponent penalties minus the team's own penalties; positive is good.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="penalty_yards_differential",
        label="Pen Yds Diff",
        full_name="Penalty Yards Differential",
        description="Opponent penalty yards minus the team's own penalty yards.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="epa_margin_per_play",
        label="EPA Margin/Play",
        full_name="EPA Margin Per Play",
        description=(
            "Offensive expected points added per play minus defensive EPA allowed per play "
            "— the single best play-level summary of team strength."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrimmage snaps",
        since=1999,
    ),
    _overall(
        name="success_rate_margin",
        label="Success Margin",
        full_name="Success Rate Margin",
        description=(
            "Offensive success rate minus defensive success rate allowed. Success means a "
            "play that improved the team's expected points."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrimmage snaps",
        since=1999,
        percent=True,
    ),
    _overall(
        name="wp_unit",
        label="Unit",
        full_name="Play Unit",
        description=(
            "Which plays a win-probability bin row counts: scrimmage plays (the team's offense "
            "against the opponent's defense) or special-teams plays where the team had "
            "possession."
        ),
        shape="id",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="wp_bin",
        label="WP Bin",
        full_name="Win-Probability Bin",
        description=(
            "How far from decided the game was before the snap, in whole percentage points: "
            "the smaller of the offense's win probability and its chance of losing, rounded "
            "down. 0 means one side was already more than 99% to win; 50 means a toss-up. Plays "
            "without a win probability have no bin and are kept by every garbage-time filter."
        ),
        shape="id",
        polarity="neutral",
        source="PBP",
        since=1999,
        formula="floor(round(100 * min(wp, 1 - wp), 9))",
    ),
    _overall(
        name="wp_kept_play_share",
        label="Kept Plays",
        full_name="Share of Plays the Filter Keeps",
        description=(
            "The share of this team's scrimmage and special-teams plays (with the ball) that the "
            "chosen garbage-time filter keeps. 1.00 means no play was left out."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="scrimmage and special-teams plays",
        since=1999,
        percent=True,
    ),
    _overall(
        name="wp_bin_plays",
        label="Plays",
        full_name="Plays in Win-Probability Bin",
        description=(
            "Plays this team ran in one game, unit, and win-probability bin. Summed over every "
            "bin they equal the game's scrimmage plays or special-teams plays."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="wp_bin_epa",
        label="EPA",
        full_name="EPA in Win-Probability Bin",
        description=(
            "Expected points added on this team's plays in one game, unit, and win-probability "
            "bin. Summed over every bin it equals the game's scrimmage EPA or special-teams EPA."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

__all__ = [
    "OVERALL_METRICS",
]
