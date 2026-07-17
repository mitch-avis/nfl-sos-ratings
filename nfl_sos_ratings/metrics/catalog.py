"""Assembly of the project registry: categories, rating pools, and builder.

Category order here is the display order for both the index pages and the
detail pages — the project's own schedule-adjusted ratings always lead.
Rating pools are copied exactly from the live rating code; changing pool
membership changes published ratings and needs explicit sign-off.
"""

from __future__ import annotations

from nfl_sos_ratings.metrics.qb_metrics import QB_METRICS
from nfl_sos_ratings.metrics.registry import MetricRegistry
from nfl_sos_ratings.metrics.schema import CategoryDef, RatingPool
from nfl_sos_ratings.metrics.team_metrics import TEAM_METRICS

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
        name="External & Reference Ratings",
        entity="team",
        description=(
            "Third-party or benchmark rating systems used for analyst context and validation "
            "baselines only. They never feed the project's published ratings."
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
        description="Kicking, punting, returns, and coverage units.",
        subcategories=(
            "Kicking",
            "Kickoffs & Coverage",
            "Kick Returns",
            "Punting & Coverage",
            "Punt Returns",
            "ST Scoring & Blocks",
        ),
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
        name="External & Reference Ratings",
        entity="qb",
        description=(
            "Third-party quarterback ratings used for analyst context and external "
            "sanity checks only. They never feed the published project ratings."
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
        name="Advanced & Expected",
        entity="qb",
        description=(
            "Tracking- and charting-based metrics (ESPN QBR, Next Gen Stats, PFR). "
            "Coverage varies by era — missing values mean not tracked yet, not zero."
        ),
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

# Pool membership mirrors ratings.py, qb_ratings.py, and main.py exactly.
RATING_POOLS: tuple[RatingPool, ...] = (
    RatingPool(
        name="team_offense",
        entity="team",
        description=(
            "Legacy raw offensive rate pool kept for descriptive surfaces and raw helper "
            "tests. Published SaOR now comes from the ridge-adjusted EPA backbone, not "
            "directly from this pool."
        ),
        members=(
            "points_per_offensive_snap",
            "total_yards_per_offensive_snap",
            "passing_yards_per_offensive_snap",
            "rushing_yards_per_offensive_snap",
            "passing_epa_per_offensive_snap",
            "rushing_epa_per_offensive_snap",
            "passing_tds_per_offensive_snap",
            "rushing_tds_per_offensive_snap",
            "passing_first_downs_per_offensive_snap",
            "rushing_first_downs_per_offensive_snap",
            "passing_cpoe",
            "sacks_suffered_per_offensive_snap",
            "passing_interceptions_per_offensive_snap",
            "sack_fumbles_lost_per_offensive_snap",
            "rushing_fumbles_lost_per_offensive_snap",
        ),
    ),
    RatingPool(
        name="team_defense",
        entity="team",
        description=(
            "Legacy raw defensive rate pool kept for descriptive surfaces and raw helper "
            "tests. Published SaDR now comes from the defense side of the ridge EPA backbone, "
            "not directly from this pool."
        ),
        members=(
            "points_allowed_per_defensive_snap",
            "total_yards_allowed_per_defensive_snap",
            "passing_yards_allowed_per_defensive_snap",
            "rushing_yards_allowed_per_defensive_snap",
            "passing_epa_allowed_per_defensive_snap",
            "rushing_epa_allowed_per_defensive_snap",
            "passing_tds_allowed_per_defensive_snap",
            "rushing_tds_allowed_per_defensive_snap",
            "passing_first_downs_allowed_per_defensive_snap",
            "rushing_first_downs_allowed_per_defensive_snap",
            "passing_cpoe_allowed",
            "def_sacks_per_defensive_snap",
            "def_interceptions_per_defensive_snap",
            "def_pass_defended_per_defensive_snap",
            "def_tackles_for_loss_per_defensive_snap",
            "def_qb_hits_per_defensive_snap",
            "def_fumbles_forced_per_defensive_snap",
            "def_safeties_per_defensive_snap",
        ),
    ),
    RatingPool(
        name="qb_primary",
        entity="qb",
        description=(
            "QB performance inputs to QRaw. Published QSaOR and QSaCR now come from the "
            "ridge-adjusted QB EPA per dropback backbone. ANY/A, sack rate, sacks, and passer "
            "rating overlap by construction — an accepted overlap; adding further overlapping "
            "members is forbidden."
        ),
        members=(
            "qb_epa_per_dropback",
            "qb_any_a",
            "qb_completion_percentage_above_expectation",
            "qb_td_int_margin_rate",
            "qb_sack_rate",
            "qb_pass_yards_per_dropback",
            "qb_sacks",
            "qb_passer_rating",
        ),
    ),
    RatingPool(
        name="qb_paired",
        entity="qb",
        description=(
            "The qb_primary members that have qopp_ opponent mirrors. Retained for legacy "
            "context and descriptive surfaces; published QB ratings no longer use the paired "
            "diff path."
        ),
        members=(
            "qb_epa_per_dropback",
            "qb_any_a",
            "qb_completion_percentage_above_expectation",
            "qb_td_int_margin_rate",
            "qb_sack_rate",
            "qb_pass_yards_per_dropback",
            "qb_passer_rating",
        ),
    ),
    RatingPool(
        name="team_simultaneous",
        entity="team",
        description=(
            "Response columns for the simultaneous team ridge adjustment: offensive rate "
            "inputs plus direct defensive playmaking rates, excluding redundant *_allowed "
            "mirrors."
        ),
        members=(
            "points_per_offensive_snap",
            "total_yards_per_offensive_snap",
            "passing_yards_per_offensive_snap",
            "rushing_yards_per_offensive_snap",
            "passing_epa_per_offensive_snap",
            "rushing_epa_per_offensive_snap",
            "passing_tds_per_offensive_snap",
            "rushing_tds_per_offensive_snap",
            "passing_first_downs_per_offensive_snap",
            "rushing_first_downs_per_offensive_snap",
            "passing_cpoe",
            "sacks_suffered_per_offensive_snap",
            "passing_interceptions_per_offensive_snap",
            "sack_fumbles_lost_per_offensive_snap",
            "rushing_fumbles_lost_per_offensive_snap",
            "def_sacks_per_defensive_snap",
            "def_interceptions_per_defensive_snap",
            "def_pass_defended_per_defensive_snap",
            "def_tackles_for_loss_per_defensive_snap",
            "def_qb_hits_per_defensive_snap",
            "def_fumbles_forced_per_defensive_snap",
            "def_safeties_per_defensive_snap",
        ),
    ),
    RatingPool(
        name="qb_simultaneous",
        entity="qb",
        description="Response columns for the simultaneous QB ridge adjustment.",
        members=(
            "qb_epa_per_dropback",
            "qb_any_a",
            "qb_completion_percentage_above_expectation",
            "qb_td_int_margin_rate",
            "qb_sack_rate",
            "qb_pass_yards_per_dropback",
            "qb_passer_rating",
        ),
    ),
)


def build_registry() -> MetricRegistry:
    """Build and validate the full project registry."""
    return MetricRegistry(
        metrics=TEAM_METRICS + QB_METRICS,
        categories=TEAM_CATEGORIES + QB_CATEGORIES,
        pools=RATING_POOLS,
    )
