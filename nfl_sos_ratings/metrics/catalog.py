"""Assembly of the project registry: categories and the cached builder.

Category order here is the display order for both the index pages and the
detail pages — the project's own schedule-adjusted ratings always lead.
"""

from __future__ import annotations

from functools import lru_cache

from nfl_sos_ratings.metrics.pair_metrics import PAIR_METRICS
from nfl_sos_ratings.metrics.qb_metrics import QB_METRICS
from nfl_sos_ratings.metrics.registry import MetricRegistry
from nfl_sos_ratings.metrics.schema import CategoryDef
from nfl_sos_ratings.metrics.team_metrics import TEAM_METRICS
from nfl_sos_ratings.metrics.unit_rank_metrics import UNIT_RANK_METRICS

TEAM_CATEGORIES: tuple[CategoryDef, ...] = (
    CategoryDef(
        name="Schedule-Adjusted Ratings",
        entity="team",
        description=(
            "The project's own ratings: how good each team was after accounting for the "
            "opponents it actually played. Start here when ranking teams."
        ),
    ),
    CategoryDef(
        name="Overall",
        entity="team",
        description="Season identity and whole-game outcomes: record, points, and margins.",
    ),
    CategoryDef(
        name="Offense",
        entity="team",
        description="Everything the team did with the ball.",
        subcategories=(
            "Total",
            "Passing",
            "Rushing",
            "Receiving",
            "Scoring",
            "Downs & Conversions",
            "Drives & Field Position",
            "Turnovers",
            "Penalties",
        ),
    ),
    CategoryDef(
        name="Defense",
        entity="team",
        description="Everything the team allowed, plus the plays its defense made.",
        subcategories=(
            "Total",
            "Passing",
            "Rushing",
            "Receiving",
            "Scoring",
            "Downs & Conversions",
            "Drives & Field Position",
            "Turnovers",
            "Pressure & Playmaking",
            "Penalties",
        ),
    ),
    CategoryDef(
        name="Special Teams",
        entity="team",
        description="Kicks, punts, returns, field goals, and extra points.",
    ),
)

QB_CATEGORIES: tuple[CategoryDef, ...] = (
    CategoryDef(
        name="Schedule-Adjusted Ratings",
        entity="qb",
        description=(
            "The project's own quarterback ratings, adjusted for the defenses each "
            "quarterback actually faced. Start here when ranking QBs."
        ),
    ),
    CategoryDef(
        name="Identity & Availability",
        entity="qb",
        description="Who the quarterback is and how much he played.",
    ),
    CategoryDef(
        name="Passing Volume",
        entity="qb",
        description="Raw passing production: attempts, completions, yards, and scores.",
    ),
    CategoryDef(
        name="Passing Efficiency",
        entity="qb",
        description="Quality per play: the rates that separate good QBs from busy ones.",
    ),
    CategoryDef(
        name="Pressure, Sacks & Pocket",
        entity="qb",
        description="Sacks taken, pressure faced, and how the quarterback handled it.",
    ),
    CategoryDef(
        name="Rushing",
        entity="qb",
        description="Quarterback runs: designed carries, scrambles, and their value.",
    ),
    CategoryDef(
        name="Scoring, Clutch & Outcomes",
        entity="qb",
        description=(
            "Results and late-game moments: wins, comebacks, and game-winning drives. "
            "Context stats — they never feed the performance ratings."
        ),
    ),
    CategoryDef(
        name="Turnovers & Ball Security",
        entity="qb",
        description="Interceptions, fumbles, and how costly the giveaways were.",
    ),
)


def build_registry() -> MetricRegistry:
    """Build and validate the full project registry."""
    return MetricRegistry(
        metrics=TEAM_METRICS + UNIT_RANK_METRICS + QB_METRICS + PAIR_METRICS,
        categories=TEAM_CATEGORIES + QB_CATEGORIES,
    )


@lru_cache(maxsize=1)
def get_registry() -> MetricRegistry:
    """Build, validate, and cache the project registry."""
    return build_registry()
