"""Contract tests for the metric registry single source of truth."""

from pathlib import Path
from typing import Any, cast

import pytest

from nfl_sos_ratings.metrics import (
    CategoryDef,
    MetricDef,
    RatingPool,
    RegistryValidationError,
    get_registry,
)
from nfl_sos_ratings.metrics.registry import MetricRegistry

# Representative sample of every affix pattern the pipeline emits today. The
# full-output guarantee is enforced at write time by validate_frame_columns()
# in main.py and by the on-disk integration test below.
CURRENT_OUTPUT_COLUMN_SAMPLES = [
    # identity and results
    "team",
    "game_id",
    "week",
    "opponent_team",
    "is_home",
    "games_played",
    "games",
    "wins",
    "win_pct",
    "points_for",
    "points_allowed",
    "point_margin",
    "win_value",
    "turnover_margin",
    # team offense / defense bases
    "passing_yards",
    "passing_cpoe",
    "sacks_suffered",
    "rushing_fumbles_lost",
    "total_yards_allowed",
    "passing_epa_allowed",
    "def_tackles_for_loss",
    "def_safeties",
    "offensive_snaps",
    "defensive_snaps",
    # irregular per-snap rate (no *_for_ infix)
    "points_per_offensive_snap",
    # suffix products
    "passing_yards_per_offensive_snap",
    "rushing_first_downs_allowed_per_defensive_snap",
    "def_qb_hits_per_defensive_snap",
    # team-surface QB aggregates
    "qb_dropbacks",
    "qb_attempts",
    "qb_passing_epa",
    "qb_epa_per_dropback",
    "qb_any_a",
    "qb_sack_rate",
    "qb_designed_carries",
    "qb_designed_rush_epa",
    "qb_designed_epa_per_carry",
    "qb_scrambles",
    "qb_kneels",
    "qb_fourth_quarter_comeback",
    "qb_passer_rating",
    # QB season shapes
    "qb_id",
    "qb_name",
    "qb_games_played",
    "qb_attempts_total",
    "qb_completions_per_game",
    "qb_fourth_quarter_comebacks",
    "qb_game_winning_drives_per_game",
    "qb_yards_per_attempt",
    "qb_touchdown_rate",
    "qb_interception_rate",
    "qb_td_int_differential",
    "qb_is_eligible",
    # prefix products
    "opp_passing_yards",
    "opp_points_per_offensive_snap",
    "opp_qb_epa_per_dropback",
    "qopp_points_allowed",
    "qopp_def_sacks",
    "qopp_qb_sack_rate",
    "qopp_qb_passer_rating",
    "diff_passing_epa_per_offensive_snap",
    "diff_qb_any_a",
    "adj_qb_epa_per_dropback",
    "adj_qb_designed_rush_epa_per_carry",
    "adj_def_qb_epa_per_dropback_faced",
    "adj_def_rushing_epa_per_offensive_snap_faced",
    "adj_off_points_per_offensive_snap",
    "adj_def_passing_interceptions_per_offensive_snap",
    "adj_off_def_sacks_per_defensive_snap",
    # ratings and percentiles
    "SaCR",
    "SaCR_alltime",
    "SaOR",
    "SaDR",
    "SaSTR",
    "SaOvR",
    "SaOvR_alltime",
    "SRS",
    "sos",
    "QSaCR_alltime",
    "QSaOR",
    "QSaOR_alltime",
    "QRaw",
    "QSoS",
    "faced_opp_SaCR",
    "adj_qb_designed_rush_epa_per_carry",
    "adj_def_rushing_epa_per_offensive_snap_faced",
    "QSaCR",
    "QOutcome",
    "QRaw_pct",
    "QSoS_pct",
]


@pytest.fixture(scope="module")
def registry() -> MetricRegistry:
    """Return the validated project registry once per module."""
    return get_registry()


class TestRegistryIntegrity:
    """The shipped registry must satisfy every structural invariant."""

    def test_registry_builds_and_validates(self, registry: MetricRegistry) -> None:
        """Building the registry runs full validation without raising."""
        assert registry.metrics

    def test_every_duplicate_of_target_exists(self, registry: MetricRegistry) -> None:
        """duplicate_of links must point at real canonical metrics."""
        for metric in registry.metrics.values():
            if metric.duplicate_of is not None:
                assert metric.duplicate_of in registry.metrics, metric.name

    def test_every_metric_category_exists(self, registry: MetricRegistry) -> None:
        """Each metric's category and subcategory must exist for its entity."""
        for metric in registry.metrics.values():
            categories = {
                category.name: category for category in registry.categories(metric.entity)
            }
            assert metric.category in categories, metric.name
            if metric.subcategory is not None:
                assert metric.subcategory in categories[metric.category].subcategories, metric.name

    def test_rates_and_averages_declare_denominators(self, registry: MetricRegistry) -> None:
        """Every rate/avg metric names the denominator that defines it."""
        for metric in registry.metrics.values():
            if metric.shape in ("rate", "avg"):
                assert metric.denominator, metric.name

    def test_descriptions_are_full_sentences(self, registry: MetricRegistry) -> None:
        """Layman-facing descriptions must be non-trivial sentences."""
        for metric in registry.metrics.values():
            assert len(metric.description) >= 20, metric.name
            assert metric.description.endswith("."), metric.name

    def test_schedule_adjusted_ratings_lead_both_entities(self, registry: MetricRegistry) -> None:
        """The project's own ratings category is first for teams and QBs."""
        for entity in ("team", "qb"):
            assert registry.categories(entity)[0].name == "Schedule-Adjusted Ratings"


class TestColumnResolution:
    """Concrete data columns resolve to base metrics plus affix context."""

    @pytest.mark.parametrize("column", CURRENT_OUTPUT_COLUMN_SAMPLES)
    def test_current_output_columns_resolve(self, registry: MetricRegistry, column: str) -> None:
        """Every column shape the pipeline currently writes must resolve."""
        assert registry.resolve_column(column) is not None, column

    def test_unknown_column_returns_none(self, registry: MetricRegistry) -> None:
        """Nonsense columns resolve to None instead of guessing."""
        assert registry.resolve_column("definitely_not_a_metric") is None

    def test_exact_match_wins_over_affix_decomposition(self, registry: MetricRegistry) -> None:
        """win_pct is an intrinsic rate, not win + percentile suffix."""
        resolved = registry.resolve_column("win_pct")
        assert resolved is not None
        assert resolved.base.name == "win_pct"

    def test_suffix_product_inherits_base_and_annotates(self, registry: MetricRegistry) -> None:
        """Per-game products keep the base identity with per-game labeling."""
        resolved = registry.resolve_column("qb_attempts_per_game")
        assert resolved is not None
        assert resolved.base.name == "qb_attempts"
        assert "per game" in resolved.label.lower() or "/g" in resolved.label.lower()
        assert resolved.polarity == resolved.base.polarity

    def test_opponent_context_stays_contextual_inside_real_taxonomy(
        self, registry: MetricRegistry
    ) -> None:
        """Context columns stay contextual but inherit the team/QB taxonomy."""
        expectations = {
            "opp_passing_yards": ("Offense", "Passing"),
            "opp_qb_passer_rating": ("Offense", "Passing"),
            "qopp_points_allowed": ("Scoring, Clutch & Outcomes", None),
            "qopp_def_sacks": ("Pressure, Sacks & Pocket", None),
            "qopp_qb_sack_rate": ("Pressure, Sacks & Pocket", None),
        }

        for column, (category, subcategory) in expectations.items():
            resolved = registry.resolve_column(column)
            assert resolved is not None, column
            assert resolved.contextual is True, column
            assert resolved.category == category, column
            assert resolved.subcategory == subcategory, column

    def test_qopp_inverts_polarity_for_qb_metrics(self, registry: MetricRegistry) -> None:
        """What faced defenses allowed flips good/bad relative to the QB stat."""
        base = registry.resolve_column("qb_epa_per_dropback")
        mirrored = registry.resolve_column("qopp_qb_epa_per_dropback")
        assert base is not None and mirrored is not None
        assert base.polarity == "higher"
        assert mirrored.polarity == "lower"

    def test_adjusted_columns_keep_base_polarity(self, registry: MetricRegistry) -> None:
        """Ridge adjustment sides keep the base stat's good direction."""
        for column in (
            "adj_off_points_per_offensive_snap",
            "adj_def_points_per_offensive_snap",
            "adj_qb_sack_rate",
        ):
            resolved = registry.resolve_column(column)
            assert resolved is not None, column
            assert resolved.polarity == resolved.base.polarity

    def test_percentile_suffix_resolves_ratings(self, registry: MetricRegistry) -> None:
        """QSaCR_pct resolves as the percentile view of QSaCR."""
        resolved = registry.resolve_column("QSaCR_pct")
        assert resolved is not None
        assert resolved.base.name == "QSaCR"
        assert "percentile" in resolved.full_name.lower()

    def test_validate_columns_reports_unknowns(self, registry: MetricRegistry) -> None:
        """validate_columns returns exactly the unresolvable names."""
        unknown = registry.validate_columns(["team", "SaCR", "not_a_real_column"])
        assert unknown == ["not_a_real_column"]


class TestRatingPools:
    """Rating pools are the registry's enforcement point for double counting."""

    def test_team_pools_match_ratings_module(self, registry: MetricRegistry) -> None:
        """The registry pools are the source for ratings.py stat pools."""
        from nfl_sos_ratings.ratings import _DEF_STAT_POOL, _OFF_STAT_POOL

        assert registry.pool_stats("team_offense") == _OFF_STAT_POOL
        assert registry.pool_stats("team_defense") == _DEF_STAT_POOL

    def test_qb_pool_matches_qb_ratings_module(self, registry: MetricRegistry) -> None:
        """The registry pool is the source for qb_ratings.py stat pools."""
        from nfl_sos_ratings.qb_ratings import (
            _QB_DIFF_STAT_POOL,
            _QB_PAIRED_STAT_POOL,
            _QB_STAT_POOL,
        )

        assert registry.pool_stats("qb_primary") == _QB_STAT_POOL
        assert registry.pool_stats("qb_paired") == _QB_PAIRED_STAT_POOL
        expected_diff_pool = [(f"diff_{name}", higher) for name, higher in _QB_STAT_POOL]
        assert expected_diff_pool == _QB_DIFF_STAT_POOL

    def test_simultaneous_pools_match_main_module(self, registry: MetricRegistry) -> None:
        """The registry pools are the source for main.py response columns."""
        from nfl_sos_ratings.main import _QB_SIMULTANEOUS_COLS, _TEAM_SIMULTANEOUS_COLS

        assert registry.pool_columns("team_simultaneous") == _TEAM_SIMULTANEOUS_COLS
        assert registry.pool_columns("qb_simultaneous") == _QB_SIMULTANEOUS_COLS

    def test_pool_membership_is_frozen(self, registry: MetricRegistry) -> None:
        """Pool membership is a rating contract; changes need sign-off."""
        assert registry.pool_stats("qb_primary") == [
            ("qb_epa_per_dropback", True),
            ("qb_any_a", True),
            ("qb_completion_percentage_above_expectation", True),
            ("qb_td_int_margin_rate", True),
            ("qb_sack_rate", False),
            ("qb_pass_yards_per_dropback", True),
            ("qb_sacks", False),
            ("qb_passer_rating", True),
        ]
        team_offense = registry.pool_columns("team_offense")
        team_defense = registry.pool_columns("team_defense")
        team_simultaneous = registry.pool_columns("team_simultaneous")
        team_defense_playmaking = [member for member in team_defense if "allowed" not in member]
        assert len(team_offense) == 15
        assert len(team_defense) == 18
        assert team_offense[0] == "points_per_offensive_snap"
        assert team_defense[0] == "points_allowed_per_defensive_snap"
        assert team_simultaneous == team_offense + team_defense_playmaking
        assert all("allowed" not in member for member in team_simultaneous[len(team_offense) :])
        assert registry.pool_columns("qb_simultaneous") == registry.pool_columns("qb_paired")

    def test_rating_provenance_is_structured(self, registry: MetricRegistry) -> None:
        """SaCR and QSaCR should expose structured provenance for the published weights."""
        sacr = registry.resolve_column("SaCR")
        qsacr = registry.resolve_column("QSaCR")

        assert sacr is not None and sacr.base.provenance is not None
        assert sacr.base.provenance.fit_window == (1999, 2025)
        assert sacr.base.provenance.fitting_command == (
            "uv run python -m nfl_sos_ratings.composite_weights"
        )
        assert sacr.base.provenance.target == "next-season SaOvR"
        assert sacr.base.provenance.weight_snapshot[-1] == ("st_rating", 0.0575025275920063)
        assert sacr.base.provenance.excluded_weight_candidates == (
            ("adj_def_takeaway_creation_rate_per_defensive_snap", -0.04081182634425329),
        )

        assert qsacr is not None and qsacr.base.provenance is not None
        assert qsacr.base.provenance.fit_window == (2006, 2025)
        assert qsacr.base.provenance.sample_weighting == "dropback-weighted weighted least squares"
        assert qsacr.base.provenance.fitting_command == (
            "uv run python -m nfl_sos_ratings.composite_weights"
        )
        assert qsacr.base.provenance.weight_snapshot[0] == (
            "adj_qb_epa_per_dropback",
            0.6687790473858877,
        )

    def test_published_rating_descriptions_define_the_within_season_scale(
        self, registry: MetricRegistry
    ) -> None:
        """Published rating tooltips should explain the within-season z-score scale."""
        for column in (
            "SaCR",
            "SaOR",
            "SaDR",
            "SaSTR",
            "SaOvR",
            "QRaw",
            "QSaOR",
            "QSoS",
            "QSaCR",
            "QOutcome",
        ):
            resolved = registry.resolve_column(column)
            assert resolved is not None
            assert "that season" in resolved.description.lower(), column

    def test_alltime_companion_descriptions_carry_era_and_moving_baseline_caveats(
        self, registry: MetricRegistry
    ) -> None:
        """All-time companion descriptions should warn about era context and moving baselines."""
        for column in ("SaCR_alltime", "SaOvR_alltime", "QSaCR_alltime", "QSaOR_alltime"):
            resolved = registry.resolve_column(column)
            assert resolved is not None
            text = f"{resolved.description} {resolved.base.note or ''}".lower()
            assert "era context" in text
            assert "future season" in text or "future seasons" in text

    def test_schedule_strength_entries_are_precise_and_visible(
        self, registry: MetricRegistry
    ) -> None:
        """Schedule-context surfaces should resolve with precise metric descriptions."""
        qsos = registry.resolve_column("QSoS")
        raw_qsos = registry.resolve_column("adj_def_qb_epa_per_dropback_faced")
        rush_qsos = registry.resolve_column("adj_def_rushing_epa_per_offensive_snap_faced")
        team_sos = registry.resolve_column("sos")
        qb_overall = registry.resolve_column("faced_opp_SaCR")
        adjusted_rush = registry.resolve_column("adj_qb_designed_rush_epa_per_carry")

        assert qsos is not None
        assert "dropback-weighted" in qsos.description.lower()
        assert "pass-defense" in qsos.description.lower()

        assert raw_qsos is not None
        assert "dropback-weighted" in raw_qsos.description.lower()

        assert rush_qsos is not None
        assert "carry-weighted" in rush_qsos.description.lower()
        assert "rush-defense" in rush_qsos.description.lower()

        assert team_sos is not None
        assert team_sos.category == "Schedule-Adjusted Ratings"
        assert team_sos.contextual is True

        assert qb_overall is not None
        assert qb_overall.category == "Schedule-Adjusted Ratings"
        assert qb_overall.contextual is True

        assert adjusted_rush is not None
        assert adjusted_rush.category == "Schedule-Adjusted Ratings"
        assert adjusted_rush.base.ratings_eligible is True

    def test_qb_cpoe_era_boundary_is_documented_on_published_composites(
        self, registry: MetricRegistry
    ) -> None:
        """QRaw and QSaCR should explain the 2006 CPOE-era publication boundary."""
        for column in ("QRaw", "QSaCR"):
            resolved = registry.resolve_column(column)
            assert resolved is not None
            assert resolved.base.note is not None
            assert "1999-2005" in resolved.base.note
            assert "2006" in resolved.base.note

    def test_pool_members_are_ratings_eligible(self, registry: MetricRegistry) -> None:
        """Only ratings_eligible metrics may enter any pool."""
        for pool in registry.pools.values():
            for member in pool.members:
                resolved = registry.resolve_column(member)
                assert resolved is not None, (pool.name, member)
                assert resolved.base.ratings_eligible, (pool.name, member)

    def test_pool_rejects_metric_with_its_duplicate(self) -> None:
        """A pool containing a metric and its duplicate must fail validation."""
        metrics = (
            _metric("canonical_stat"),
            _metric("restated_stat", duplicate_of="canonical_stat"),
        )
        pool = RatingPool(
            name="bad_pool",
            entity="team",
            description="Pool that double counts.",
            members=("canonical_stat", "restated_stat"),
        )
        with pytest.raises(RegistryValidationError, match="duplicate"):
            MetricRegistry(
                metrics=metrics,
                categories=(_category(),),
                pools=(pool,),
            )

    def test_pool_rejects_unknown_member(self) -> None:
        """A pool naming a metric that does not exist must fail validation."""
        pool = RatingPool(
            name="bad_pool",
            entity="team",
            description="Pool with a ghost member.",
            members=("missing_stat",),
        )
        with pytest.raises(RegistryValidationError, match="missing_stat"):
            MetricRegistry(
                metrics=(_metric("canonical_stat"),), categories=(_category(),), pools=(pool,)
            )

    @pytest.mark.parametrize(
        ("entity", "member"),
        (("team", "team_elo"), ("qb", "qb_qbr_total")),
    )
    def test_pool_rejects_external_reference_metrics(
        self,
        entity: str,
        member: str,
    ) -> None:
        """External reference metrics must never be allowed into rating pools."""
        from nfl_sos_ratings.metrics.catalog import QB_CATEGORIES, RATING_POOLS, TEAM_CATEGORIES
        from nfl_sos_ratings.metrics.qb_metrics import QB_METRICS
        from nfl_sos_ratings.metrics.team_metrics import TEAM_METRICS

        pool = RatingPool(
            name="bad_pool",
            entity=cast(Any, entity),
            description="Pool with a descriptive-only external reference metric.",
            members=(member,),
        )
        with pytest.raises(RegistryValidationError, match="not ratings_eligible"):
            MetricRegistry(
                metrics=TEAM_METRICS + QB_METRICS,
                categories=TEAM_CATEGORIES + QB_CATEGORIES,
                pools=RATING_POOLS + (pool,),
            )


class TestPayload:
    """The API payload must be JSON-serializable and complete."""

    def test_payload_serializes_to_json(self, registry: MetricRegistry) -> None:
        """The payload survives a JSON round trip with categories in order."""
        import json

        payload = json.loads(json.dumps(registry.payload()))
        assert [category["name"] for category in payload["entities"]["team"]["categories"]][
            0
        ] == "Schedule-Adjusted Ratings"
        assert "passing_yards" in payload["metrics"]

    def test_payload_omits_standalone_opponent_context_categories(
        self, registry: MetricRegistry
    ) -> None:
        """Opponent context is expressed through views, not entity category lists."""
        payload = cast(dict[str, Any], registry.payload())
        entities = cast(dict[str, Any], payload["entities"])

        for entity in ("team", "qb"):
            entity_payload = cast(dict[str, Any], entities[entity])
            categories = cast(list[dict[str, Any]], entity_payload["categories"])
            category_names = [str(category["name"]) for category in categories]
            assert "Opponent Context" not in category_names

    def test_resolved_metadata_payload_for_columns(self, registry: MetricRegistry) -> None:
        """Column metadata payloads carry label, tooltip text, and polarity."""
        metadata = registry.column_metadata(["SaCR", "opp_passing_yards"])
        assert metadata["SaCR"]["polarity"] == "higher"
        assert metadata["SaCR"]["label"]
        description = metadata["SaCR"]["description"]
        assert isinstance(description, str)
        assert description.endswith(".")
        assert metadata["opp_passing_yards"]["contextual"] is True


@pytest.mark.skipif(
    not Path("data/2025_combined.csv").exists() and not Path("data/2025_combined.parquet").exists(),
    reason="generated outputs not present",
)
class TestLiveOutputCoverage:
    """When generated outputs exist on disk, every column must resolve."""

    def test_all_output_columns_resolve(self, registry: MetricRegistry) -> None:
        """No data file may contain a column the registry cannot explain."""
        import polars as pl

        data_dir = Path("data")
        failures: dict[str, list[str]] = {}
        for pattern in ("2025_*.parquet", "2025_*.csv"):
            for file_path in sorted(data_dir.glob(pattern)):
                if file_path.suffix == ".parquet":
                    columns = pl.read_parquet_schema(file_path).keys()
                else:
                    columns = pl.read_csv(file_path, n_rows=0).columns
                unknown = registry.validate_columns(columns)
                if unknown:
                    failures[file_path.name] = unknown
        assert not failures, failures


def _metric(name: str, duplicate_of: str | None = None) -> MetricDef:
    """Return a minimal valid metric for synthetic-registry tests."""
    return MetricDef(
        name=name,
        label=name.title(),
        full_name=name.replace("_", " ").title(),
        description="A synthetic metric used only for validation tests.",
        entity="team",
        category="Test Category",
        shape="count",
        polarity="higher",
        source="D",
        ratings_eligible=True,
        duplicate_of=duplicate_of,
    )


def _category() -> CategoryDef:
    """Return a minimal category for synthetic-registry tests."""
    return CategoryDef(
        name="Test Category",
        entity="team",
        description="Synthetic category for tests.",
    )
