"""Contract tests for the metric registry, the single source of truth for published columns."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.metrics import CategoryDef, MetricDef, RegistryValidationError, get_registry
from nfl_sos_ratings.metrics.registry import MetricRegistry

if TYPE_CHECKING:
    from nfl_sos_ratings.metrics.schema import Entity

# A representative sample of every column shape the pipeline writes. The full guarantee is
# enforced at write time by main._write_data_file and by the published-data test below.
_OUTPUT_COLUMN_SAMPLES = (
    "team",
    "game_id",
    "is_home",
    "games_played",
    "win_pct",
    "point_margin",
    "passing_yards",
    "offensive_snaps",
    "offensive_epa",
    "st_plays",
    "st_epa",
    "points_per_offensive_snap",
    "passing_yards_per_offensive_snap",
    "def_qb_hits_per_defensive_snap",
    "qb_dropbacks",
    "qb_epa_per_dropback",
    "qb_designed_epa_per_carry",
    "qb_attempts_total",
    "qb_completions_per_game",
    "qb_is_eligible",
    "opp_passing_yards",
    "opp_qb_epa_per_dropback",
    "qopp_qb_sack_rate",
    "team_rating",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "sos",
    "SRS",
    "adj_qb_epa_per_dropback",
    "qb_faced_pass_defense",
)


@pytest.fixture(scope="module")
def registry() -> MetricRegistry:
    """Return the validated project registry once per module."""
    return get_registry()


def _metric(name: str, category: str = "Test Category") -> MetricDef:
    """Return a minimal valid metric for synthetic-registry tests."""
    return MetricDef(
        name=name,
        label=name.title(),
        full_name=name.replace("_", " ").title(),
        description="A synthetic metric used only for validation tests.",
        entity="team",
        category=category,
        shape="count",
        polarity="higher",
        source="D",
    )


def _category() -> CategoryDef:
    """Return a minimal category for synthetic-registry tests."""
    return CategoryDef(name="Test Category", entity="team", description="Synthetic category.")


def test_every_rate_and_average_declares_a_denominator(registry: MetricRegistry) -> None:
    # Act
    missing = [
        metric.name
        for metric in registry.metrics.values()
        if metric.shape in ("rate", "avg") and not metric.denominator
    ]

    # Assert
    assert missing == []


def test_every_description_is_a_full_sentence(registry: MetricRegistry) -> None:
    # Act
    short = [
        metric.name
        for metric in registry.metrics.values()
        if len(metric.description) < 20 or not metric.description.endswith(".")
    ]

    # Assert
    assert short == []


def test_every_duplicate_of_target_exists(registry: MetricRegistry) -> None:
    # Act
    dangling = [
        metric.name
        for metric in registry.metrics.values()
        if metric.duplicate_of is not None and metric.duplicate_of not in registry.metrics
    ]

    # Assert
    assert dangling == []


@pytest.mark.parametrize("entity", ["team", "qb"])
def test_ratings_category_leads_each_entity(registry: MetricRegistry, entity: Entity) -> None:
    # Act
    first = registry.categories(entity)[0].name

    # Assert
    assert first == "Schedule-Adjusted Ratings"


@pytest.mark.parametrize("column", _OUTPUT_COLUMN_SAMPLES)
def test_output_column_resolves(registry: MetricRegistry, column: str) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None


def test_unknown_column_resolves_to_none(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("definitely_not_a_metric")

    # Assert
    assert resolved is None


def test_exact_match_wins_over_affix_decomposition(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("adj_qb_epa_per_dropback")

    # Assert
    assert resolved is not None
    assert resolved.base.name == "adj_qb_epa_per_dropback"


def test_per_game_suffix_keeps_the_base_metric(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("qb_attempts_per_game")

    # Assert
    assert resolved is not None
    assert resolved.base.name == "qb_attempts"
    assert resolved.polarity == resolved.base.polarity


@pytest.mark.parametrize(
    ("column", "category"),
    [
        ("opp_passing_yards", "Offense"),
        ("qopp_points_allowed", "Scoring, Clutch & Outcomes"),
        ("qopp_qb_sack_rate", "Pressure, Sacks & Pocket"),
    ],
)
def test_opponent_columns_stay_contextual_inside_the_taxonomy(
    registry: MetricRegistry, column: str, category: str
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert (resolved.contextual, resolved.category) == (True, category)


def test_qopp_prefix_inverts_qb_polarity(registry: MetricRegistry) -> None:
    # Act
    mirrored = registry.resolve_column("qopp_qb_epa_per_dropback")

    # Assert
    assert mirrored is not None
    assert mirrored.polarity == "lower"


def test_validate_columns_returns_exactly_the_unknown_names(registry: MetricRegistry) -> None:
    # Act
    unknown = registry.validate_columns(["team", "team_rating", "not_a_real_column"])

    # Assert
    assert unknown == ["not_a_real_column"]


def test_payload_survives_a_json_round_trip(registry: MetricRegistry) -> None:
    # Act
    payload = json.loads(json.dumps(registry.payload()))

    # Assert
    assert payload["metrics"]["team_rating"]["polarity"] == "higher"
    assert "Opponent Context" not in [
        category["name"] for category in payload["entities"]["team"]["categories"]
    ]


def test_column_metadata_carries_label_and_description(registry: MetricRegistry) -> None:
    # Act
    metadata = registry.column_metadata(["team_rating", "opp_passing_yards"])

    # Assert
    assert metadata["team_rating"]["label"] == "Team Rating"
    assert metadata["opp_passing_yards"]["contextual"] is True


def test_registry_rejects_a_metric_defined_twice() -> None:
    # Arrange
    metrics = [_metric("alpha"), _metric("alpha")]

    # Act & Assert
    with pytest.raises(RegistryValidationError, match="defined twice"):
        MetricRegistry(metrics, [_category()])


def test_registry_rejects_an_unknown_category() -> None:
    # Arrange
    metrics = [_metric("alpha", category="Nowhere")]

    # Act & Assert
    with pytest.raises(RegistryValidationError, match="unknown category"):
        MetricRegistry(metrics, [_category()])


@pytest.mark.published_data
def test_every_published_column_resolves(registry: MetricRegistry) -> None:
    # Arrange
    files = sorted(Path("data").glob("*.parquet"))

    # Act
    failures = {
        path.name: unknown
        for path in files
        if (unknown := registry.validate_columns(pl.read_parquet_schema(path).keys()))
    }

    # Assert
    assert files
    assert failures == {}


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("opp_qb_offense_snaps", ("Offense", "Total")),
        ("opp_qb_rushing_yards", ("Offense", "Rushing")),
        ("opp_qb_fourth_quarter_comeback", ("Offense", "Scoring")),
        ("opp_qb_td_int_differential", ("Offense", "Turnovers")),
        ("opp_qb_passer_rating", ("Offense", "Passing")),
        ("qopp_points_allowed", ("Scoring, Clutch & Outcomes", None)),
        ("qopp_total_tds", ("Scoring, Clutch & Outcomes", None)),
        ("qopp_def_interceptions", ("Turnovers & Ball Security", None)),
        ("qopp_def_sacks", ("Pressure, Sacks & Pocket", None)),
        ("qopp_def_fumbles_forced", ("Pressure, Sacks & Pocket", None)),
        ("qopp_rushing_yards_allowed", ("Rushing", None)),
        ("qopp_passing_yards_allowed", ("Passing Efficiency", None)),
    ],
)
def test_cross_entity_context_columns_map_onto_the_viewing_taxonomy(
    registry: MetricRegistry, column: str, expected: tuple[str, str | None]
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert (resolved.category, resolved.subcategory) == expected


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                subcategory="Nowhere",
                shape="count",
                polarity="higher",
                source="D",
            ),
            "unknown subcategory",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                shape="count",
                polarity="higher",
                source="D",
                duplicate_of="missing",
            ),
            "duplicates unknown metric",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                shape="rate",
                polarity="higher",
                source="D",
            ),
            "declares no denominator",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="Too short",
                entity="team",
                category="Test Category",
                shape="count",
                polarity="higher",
                source="D",
            ),
            "full-sentence",
        ),
    ],
)
def test_registry_rejects_an_invalid_metric(metric: MetricDef, message: str) -> None:
    # Act & Assert
    with pytest.raises(RegistryValidationError, match=message):
        MetricRegistry([metric], [_category()])


def test_column_metadata_skips_unknown_columns(registry: MetricRegistry) -> None:
    # Act
    metadata = registry.column_metadata(["team_rating", "not_a_real_column"])

    # Assert
    assert list(metadata) == ["team_rating"]


def test_suffix_without_a_known_base_resolves_to_none(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("opp_not_a_metric_per_game")

    # Assert
    assert resolved is None


def test_qopp_prefix_keeps_neutral_polarity_neutral(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("qopp_qb_offense_snaps")

    # Assert
    assert resolved is not None
    assert resolved.polarity == "neutral"
