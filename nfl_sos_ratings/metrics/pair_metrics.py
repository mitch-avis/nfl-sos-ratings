"""Head-to-head pair metrics: how often one team or quarterback is rated above another.

Each row of the ``rating_pairs`` and ``qb_rating_pairs`` files compares two teams or two
quarterbacks across the same game-bootstrap resamples that give the rank ranges. Kept apart from
the team and QB definition modules, which are already long.
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.schema import MetricDef, section

_team = section("team", "Schedule-Adjusted Ratings")
_qb = section("qb", "Schedule-Adjusted Ratings")

TEAM_PAIR_METRICS: tuple[MetricDef, ...] = (
    _team(
        name="other_team",
        label="Compared Team",
        full_name="Compared Team",
        description=(
            "The other team in a head-to-head comparison, by its standard NFL abbreviation. The "
            "two teams need not have played each other."
        ),
        shape="id",
        polarity="neutral",
        source="D",
    ),
    _team(
        name="team_rated_above_probability",
        label="Rated-Above Chance",
        full_name="Chance of Being Rated Above the Compared Team",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats) in which this team's Team Rating came out above the compared team's. "
            "Both teams move together in each resample, so this answers 'is A better than B?' "
            "more directly than two overlapping rank ranges. A tie counts as half."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples with both teams",
        since=1999,
        percent=True,
    ),
    _team(
        name="team_rating_gap",
        label="Rating Gap",
        full_name="Team Rating Minus the Compared Team's",
        description=(
            "How many points per game this team's Team Rating exceeds the compared team's in a "
            "game-bootstrap resample of the season; published as percentiles across resamples."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _team(
        name="team_pair_share",
        label="Both-Teams Share",
        full_name="Share of Resamples With Both Teams",
        description=(
            "The share of game-bootstrap resamples of the season in which both teams had games, "
            "so the comparison could be made. It is essentially one for a finished season."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

QB_PAIR_METRICS: tuple[MetricDef, ...] = (
    _qb(
        name="other_qb_id",
        label="Compared QB ID",
        full_name="Compared Quarterback ID",
        description=(
            "The canonical GSIS player identifier of the other quarterback in a head-to-head "
            "comparison. The two need not have faced each other."
        ),
        shape="id",
        polarity="neutral",
        source="D",
    ),
    _qb(
        name="qb_rated_above_probability",
        label="Rated-Above Chance",
        full_name="Chance of Being Rated Above the Compared Quarterback",
        description=(
            "The share of game-bootstrap resamples of the season (its games redrawn at random, "
            "with repeats), among those including both quarterbacks, in which this quarterback's "
            "Adjusted EPA Per Dropback came out above the compared quarterback's. A tie counts "
            "as half."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples with both quarterbacks",
        since=1999,
        percent=True,
    ),
    _qb(
        name="qb_rating_gap",
        label="Rating Gap",
        full_name="Adjusted EPA Per Dropback Minus the Compared Quarterback's",
        description=(
            "How much this quarterback's Adjusted EPA Per Dropback exceeds the compared "
            "quarterback's in a game-bootstrap resample of the season; published as percentiles "
            "across the resamples including both."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _qb(
        name="qb_pair_share",
        label="Both-QBs Share",
        full_name="Share of Resamples With Both Quarterbacks",
        description=(
            "The share of game-bootstrap resamples of the season that include games by both "
            "quarterbacks, so the comparison could be made. A backup who played little is "
            "missing from many resamples."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

PAIR_METRICS: tuple[MetricDef, ...] = TEAM_PAIR_METRICS + QB_PAIR_METRICS

__all__ = ["PAIR_METRICS", "QB_PAIR_METRICS", "TEAM_PAIR_METRICS"]
