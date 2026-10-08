"""Team rating and special-teams definitions.

The published ratings, schedule strength, and SRS, plus the special-teams play counts the
ratings use.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_ratings = section("team", "Schedule-Adjusted Ratings")
_special_teams = section("team", "Special Teams")

RATING_METRICS: tuple[MetricDef, ...] = (
    _ratings(
        name="team_rating",
        label="Team Rating",
        full_name="Team Rating",
        description=(
            "How many points per game better than an average team this team was on a neutral "
            "field, after adjusting for every opponent it faced. It is the sum of the offense, "
            "defense, and special-teams ratings, all built from expected points added (EPA). "
            "0 is an average team."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="offense_rating",
        label="Off Rating",
        full_name="Offense Rating",
        description=(
            "Points per game the offense produced above an average offense, measured by "
            "scrimmage EPA per play and adjusted for the defenses it faced."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="defense_rating",
        label="Def Rating",
        full_name="Defense Rating",
        description=(
            "Points per game the defense prevented compared with an average defense, measured "
            "by scrimmage EPA per play allowed and adjusted for the offenses it faced. Higher "
            "is better."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="special_teams_rating",
        label="ST Rating",
        full_name="Special Teams Rating",
        description=(
            "Points per game gained on special-teams plays (kicks, punts, returns, field goals, "
            "and extra points) compared with an average team, adjusted for the opponents faced."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="SRS",
        label="SRS",
        full_name="Simple Rating System",
        description=(
            "A classic point-margin rating solved across the whole league at once. Positive "
            "means the team outscored opponents by more than an average team would have "
            "against the same schedule, measured in points per game."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _ratings(
        name="sos",
        label="SoS",
        full_name="Strength of Schedule",
        description=(
            "The average Team Rating of the opponents this team played, one entry per game, in "
            "points per game. Each opponent is rated without its games against this team, so "
            "beating an opponent badly cannot make that opponent look weaker here. Positive "
            "means a harder-than-average schedule. Early in a season, opponents that have played "
            "no one else yet are left out. Context, not a team grade."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        contextual=True,
    ),
    _ratings(
        name="team_rank",
        label="Rank",
        full_name="Team Rating Rank",
        description=(
            "The team's place in the league by Team Rating, 1 for the best. Teams with equal "
            "ratings share the better rank."
        ),
        shape="score",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _ratings(
        name="team_rank_missing_share",
        label="No-Game Share",
        full_name="Share of Resamples Without the Team",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the team had no games and so no rank. It is essentially "
            "zero for a finished season and grows early in a season, when each team has played "
            "only a few games."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_top5_probability",
        label="Top-5 Chance",
        full_name="Chance of a Top-5 Rank",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the team ranked in the top five by Team Rating. It shows how "
            "much the ranking depends on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_top10_probability",
        label="Top-10 Chance",
        full_name="Chance of a Top-10 Rank",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which the team ranked in the top ten by Team Rating. It shows how "
            "much the ranking depends on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_probabilities",
        label="Rank Chances",
        full_name="Chance of Each Rank",
        description=(
            "A list giving, for each rank from 1 down, the share of game-bootstrap resamples of "
            "the season in which the team finished at exactly that rank by Team Rating."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

SPECIAL_TEAMS_METRICS: tuple[MetricDef, ...] = (
    _special_teams(
        name="st_plays",
        label="ST Plays",
        full_name="Special Teams Plays",
        description=(
            "Special-teams plays where this team had possession: its punts, field goals, and "
            "extra points, plus kickoffs it received."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _special_teams(
        name="st_epa",
        label="ST EPA",
        full_name="Special Teams EPA",
        description=(
            "Expected points added on the special-teams plays where this team had possession "
            "(its punts, field goals, and extra points, plus kickoffs it received)."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

__all__ = [
    "RATING_METRICS",
    "SPECIAL_TEAMS_METRICS",
]
