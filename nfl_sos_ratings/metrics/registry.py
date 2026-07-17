"""Registry engine: validation, column resolution, pools, and API payloads.

The registry is the single source of truth for every published metric. The
ETL validates its data columns against it, the API serves it, and the
frontend derives labels, tooltips, sort defaults, and color-gradient
direction from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from nfl_sos_ratings.metrics.schema import (
    CategoryDef,
    Entity,
    MetricDef,
    MetricProvenance,
    Polarity,
    PrefixRule,
    RatingPool,
    ResolvedColumn,
    SuffixRule,
)


class RegistryValidationError(ValueError):
    """Raised when the registry data violates a structural invariant."""


# Longest prefixes first so adj_off_/adj_def_ win over the bare adj_ rule.
DEFAULT_PREFIX_RULES: tuple[PrefixRule, ...] = (
    PrefixRule(
        prefix="adj_off_",
        label_template="Adj Off {label}",
        full_name_template="{full_name} (Offense Side, Simultaneously Adjusted)",
        description_note=(
            "This is the offense-side share of the stat from the simultaneous ridge model, "
            "which solves every team's offense and defense at once so each value is already "
            "adjusted for the schedule played."
        ),
    ),
    PrefixRule(
        prefix="adj_def_",
        label_template="Adj Def {label}",
        full_name_template="{full_name} (Defense Side, Simultaneously Adjusted)",
        description_note=(
            "This is the defense-side share of the stat from the simultaneous ridge model: "
            "how much of it this defense takes away from (or gives to) opponents compared "
            "with an average defense, already adjusted for the schedule played."
        ),
    ),
    PrefixRule(
        prefix="adj_",
        label_template="Adj {label}",
        full_name_template="Simultaneously Adjusted {full_name}",
        description_note=(
            "This is a schedule-adjusted estimate from the simultaneous ridge model, which "
            "solves every quarterback and defense at the same time instead of averaging "
            "opponents one hop out."
        ),
    ),
    PrefixRule(
        prefix="qopp_",
        label_template="Opp {label}",
        full_name_template="Faced Defenses: {full_name}",
        description_note=(
            "This is season-long context about the defenses this quarterback actually "
            "faced — what those defenses allowed to all other passers — not a grade of the "
            "quarterback."
        ),
        contextual=True,
        invert_polarity_for_qb=True,
    ),
    PrefixRule(
        prefix="opp_",
        label_template="Opp {label}",
        full_name_template="Opponents Faced: {full_name}",
        description_note=(
            "This is season-long context about the opponents actually faced (averaged with "
            "head-to-head games excluded), not a grade of the selected team."
        ),
        contextual=True,
    ),
    PrefixRule(
        prefix="diff_",
        label_template="{label} Diff",
        full_name_template="{full_name} vs. Opponents Faced",
        description_note=(
            "This subtracts the faced-opponents context from the subject's own value on the "
            "same stat, so positive means better than the schedule would suggest."
        ),
    ),
    PrefixRule(
        prefix="season_delta_",
        label_template="{label} vs Season",
        full_name_template="{full_name} vs. Season Baseline",
        description_note=(
            "This compares the average in these matchups with the subject's full-season "
            "average on the same stat."
        ),
    ),
)

DEFAULT_SUFFIX_RULES: tuple[SuffixRule, ...] = (
    SuffixRule(
        suffix="_per_game",
        label_template="{label}/G",
        full_name_template="{full_name} Per Game",
        description_note="Shown per game played.",
    ),
    SuffixRule(
        suffix="_per_offensive_snap",
        label_template="{label}/Off Snap",
        full_name_template="{full_name} Per Offensive Snap",
        description_note=(
            "Shown per offensive snap, so teams with different play volumes compare fairly."
        ),
    ),
    SuffixRule(
        suffix="_per_defensive_snap",
        label_template="{label}/Def Snap",
        full_name_template="{full_name} Per Defensive Snap",
        description_note=(
            "Shown per defensive snap, so teams with different play volumes compare fairly."
        ),
    ),
    SuffixRule(
        suffix="_per_dropback",
        label_template="{label}/DB",
        full_name_template="{full_name} Per Dropback",
        description_note="Shown per dropback (pass attempts plus sacks plus scrambles).",
    ),
    SuffixRule(
        suffix="_per_attempt",
        label_template="{label}/Att",
        full_name_template="{full_name} Per Attempt",
        description_note="Shown per official pass attempt.",
    ),
    SuffixRule(
        suffix="_per_carry",
        label_template="{label}/Carry",
        full_name_template="{full_name} Per Carry",
        description_note="Shown per rushing attempt.",
    ),
    SuffixRule(
        suffix="_per_drive",
        label_template="{label}/Drive",
        full_name_template="{full_name} Per Drive",
        description_note="Shown per offensive possession.",
    ),
    SuffixRule(
        suffix="_total",
        label_template="{label}",
        full_name_template="{full_name} (Season Total)",
        description_note="This is the full season total.",
    ),
    SuffixRule(
        suffix="_pct",
        label_template="{label} Pct",
        full_name_template="{full_name} Percentile",
        description_note=(
            "This is the percentile rank within the current season data "
            "(100 means best of the season, 0 means worst)."
        ),
    ),
)


class MetricRegistry:
    """Validated, queryable collection of metrics, categories, and pools."""

    def __init__(
        self,
        metrics: Sequence[MetricDef],
        categories: Sequence[CategoryDef],
        pools: Sequence[RatingPool] = (),
        prefix_rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
        suffix_rules: Sequence[SuffixRule] = DEFAULT_SUFFIX_RULES,
    ) -> None:
        """Index the definitions and run full structural validation."""
        self.metrics: dict[str, MetricDef] = {}
        for metric in metrics:
            if metric.name in self.metrics:
                raise RegistryValidationError(f"Metric defined twice: {metric.name}")
            self.metrics[metric.name] = metric
        self._categories: tuple[CategoryDef, ...] = tuple(categories)
        self.pools: dict[str, RatingPool] = {pool.name: pool for pool in pools}
        self._prefix_rules = tuple(prefix_rules)
        self._suffix_rules = tuple(suffix_rules)
        self._validate()

    def categories(self, entity: Entity) -> tuple[CategoryDef, ...]:
        """Return the display-ordered categories for one entity."""
        return tuple(category for category in self._categories if category.entity == entity)

    def resolve_column(self, column: str) -> ResolvedColumn | None:
        """Resolve a concrete data column to its base metric plus affixes."""
        exact = self.metrics.get(column)
        if exact is not None:
            return self._finalize(column, exact, prefix=None, suffix=None)

        for prefix_rule in self._prefix_rules:
            if not column.startswith(prefix_rule.prefix):
                continue
            core = column[len(prefix_rule.prefix) :]
            resolved_core = self._resolve_core(core)
            if resolved_core is not None:
                base, suffix_rule = resolved_core
                return self._finalize(column, base, prefix=prefix_rule, suffix=suffix_rule)

        resolved_core = self._resolve_core(column)
        if resolved_core is not None:
            base, suffix_rule = resolved_core
            return self._finalize(column, base, prefix=None, suffix=suffix_rule)
        return None

    def validate_columns(self, columns: Iterable[str]) -> list[str]:
        """Return the columns that do not resolve against the registry."""
        return [column for column in columns if self.resolve_column(column) is None]

    def pool_columns(self, pool_name: str) -> list[str]:
        """Return a pool's member columns in their defined order."""
        return list(self.pools[pool_name].members)

    def pool_stats(self, pool_name: str) -> list[tuple[str, bool]]:
        """Return a pool as (column, higher_is_better) rating-input tuples."""
        stats: list[tuple[str, bool]] = []
        for member in self.pools[pool_name].members:
            resolved = self.resolve_column(member)
            if resolved is None:  # pragma: no cover - guarded by _validate
                raise RegistryValidationError(f"Pool {pool_name} member unknown: {member}")
            stats.append((member, resolved.polarity == "higher"))
        return stats

    def column_metadata(self, columns: Iterable[str]) -> dict[str, dict[str, object]]:
        """Return JSON-safe presentation metadata for the resolvable columns."""
        metadata: dict[str, dict[str, object]] = {}
        for column in columns:
            resolved = self.resolve_column(column)
            if resolved is None:
                continue
            metadata[column] = {
                "label": resolved.label,
                "full_name": resolved.full_name,
                "description": resolved.description,
                "polarity": resolved.polarity,
                "contextual": resolved.contextual,
                "category": resolved.category,
                "subcategory": resolved.subcategory,
                "shape": resolved.base.shape,
                "denominator": resolved.base.denominator,
                "source": resolved.base.source,
                "base_name": resolved.base.name,
            }
        return metadata

    def payload(self) -> dict[str, object]:
        """Return the full registry as a JSON-safe API payload."""
        return {
            "entities": {
                entity: {
                    "categories": [
                        {
                            "name": category.name,
                            "description": category.description,
                            "subcategories": list(category.subcategories),
                        }
                        for category in self.categories(entity)
                    ]
                }
                for entity in ("team", "qb")
            },
            "metrics": {
                metric.name: {
                    "label": metric.label,
                    "full_name": metric.full_name,
                    "description": metric.description,
                    "entity": metric.entity,
                    "category": metric.category,
                    "subcategory": metric.subcategory,
                    "shape": metric.shape,
                    "polarity": metric.polarity,
                    "source": metric.source,
                    "denominator": metric.denominator,
                    "since": metric.since,
                    "ratings_eligible": metric.ratings_eligible,
                    "duplicate_of": metric.duplicate_of,
                    "status": metric.status,
                    "contextual": metric.contextual,
                    "formula": metric.formula,
                    "note": metric.note,
                    "provenance": self._provenance_payload(metric.provenance),
                }
                for metric in self.metrics.values()
            },
            "pools": {
                pool.name: {
                    "entity": pool.entity,
                    "description": pool.description,
                    "members": list(pool.members),
                }
                for pool in self.pools.values()
            },
        }

    def _resolve_core(self, core: str) -> tuple[MetricDef, SuffixRule | None] | None:
        """Resolve a prefix-stripped column body to a base metric."""
        exact = self.metrics.get(core)
        if exact is not None:
            return exact, None
        for suffix_rule in self._suffix_rules:
            if core.endswith(suffix_rule.suffix):
                base = self.metrics.get(core[: -len(suffix_rule.suffix)])
                if base is not None:
                    return base, suffix_rule
        return None

    @staticmethod
    def _provenance_payload(provenance: MetricProvenance | None) -> dict[str, object] | None:
        """Return a JSON-safe provenance payload when one exists."""
        if provenance is None:
            return None
        return {
            "target": provenance.target,
            "fit_window": list(provenance.fit_window) if provenance.fit_window else None,
            "fitting_command": provenance.fitting_command,
            "refit_policy": provenance.refit_policy,
            "sample_weighting": provenance.sample_weighting,
            "weight_snapshot": [
                {"name": name, "weight": weight} for name, weight in provenance.weight_snapshot
            ],
            "holdout_metrics": dict(provenance.holdout_metrics),
            "excluded_weight_candidates": [
                {"name": name, "weight": weight}
                for name, weight in provenance.excluded_weight_candidates
            ],
        }

    def _finalize(
        self,
        column: str,
        base: MetricDef,
        prefix: PrefixRule | None,
        suffix: SuffixRule | None,
    ) -> ResolvedColumn:
        """Compose the presentation of a resolved column from its parts."""
        label = base.label
        full_name = base.full_name
        description_parts = [base.description]

        if suffix is not None:
            label = suffix.label_template.format(label=label)
            full_name = suffix.full_name_template.format(full_name=full_name)
            description_parts.append(suffix.description_note)
        if prefix is not None:
            label = prefix.label_template.format(label=label)
            full_name = prefix.full_name_template.format(full_name=full_name)
            description_parts.append(prefix.description_note)

        polarity = base.polarity
        if prefix is not None and prefix.invert_polarity_for_qb and base.name.startswith("qb_"):
            polarity = _invert(polarity)

        contextual = prefix.contextual if prefix is not None else base.contextual
        category, subcategory = _resolved_taxonomy(base, prefix)

        return ResolvedColumn(
            column=column,
            base=base,
            label=label,
            full_name=full_name,
            description=" ".join(description_parts),
            polarity=polarity,
            contextual=contextual,
            category=category,
            subcategory=subcategory,
        )

    def _validate(self) -> None:
        """Enforce every structural invariant; raise on the first violation."""
        category_index: dict[tuple[Entity, str], CategoryDef] = {
            (category.entity, category.name): category for category in self._categories
        }

        for metric in self.metrics.values():
            self._validate_metric(metric, category_index)

        for pool in self.pools.values():
            self._validate_pool(pool)

    def _validate_metric(
        self,
        metric: MetricDef,
        category_index: dict[tuple[Entity, str], CategoryDef],
    ) -> None:
        """Check one metric's links, denominator rule, and description."""
        category = category_index.get((metric.entity, metric.category))
        if category is None:
            raise RegistryValidationError(
                f"Metric {metric.name} references unknown category {metric.category!r}"
            )
        if metric.subcategory is not None and metric.subcategory not in category.subcategories:
            raise RegistryValidationError(
                f"Metric {metric.name} references unknown subcategory {metric.subcategory!r}"
            )
        if metric.duplicate_of is not None and metric.duplicate_of not in self.metrics:
            raise RegistryValidationError(
                f"Metric {metric.name} duplicates unknown metric {metric.duplicate_of!r}"
            )
        if metric.shape in ("rate", "avg") and not metric.denominator:
            raise RegistryValidationError(
                f"Metric {metric.name} is a {metric.shape} but declares no denominator"
            )
        if not metric.description.endswith(".") or len(metric.description) < 20:
            raise RegistryValidationError(
                f"Metric {metric.name} needs a full-sentence layman description"
            )

    def _validate_pool(self, pool: RatingPool) -> None:
        """Check pool members exist, are eligible, and never double count."""
        canonical_bases: dict[str, str] = {}
        for member in pool.members:
            resolved = self.resolve_column(member)
            if resolved is None:
                raise RegistryValidationError(
                    f"Rating pool {pool.name} references unknown column {member!r}"
                )
            base = resolved.base
            if not base.ratings_eligible:
                raise RegistryValidationError(
                    f"Rating pool {pool.name} member {member} is not ratings_eligible"
                )
            canonical = base.duplicate_of or base.name
            if canonical in canonical_bases:
                raise RegistryValidationError(
                    f"Rating pool {pool.name} double counts {canonical!r} via "
                    f"{canonical_bases[canonical]!r} and {member!r} (duplicate)"
                )
            canonical_bases[canonical] = member


def _invert(polarity: Polarity) -> Polarity:
    """Flip higher/lower polarity; neutral stays neutral."""
    if polarity == "higher":
        return "lower"
    if polarity == "lower":
        return "higher"
    return "neutral"


def _resolved_taxonomy(base: MetricDef, prefix: PrefixRule | None) -> tuple[str, str | None]:
    """Return the display taxonomy for one resolved column."""
    if prefix is None or prefix.prefix not in {"opp_", "qopp_"}:
        return base.category, base.subcategory

    if prefix.prefix == "opp_":
        if base.entity == "team":
            return base.category, base.subcategory
        return _map_qb_metric_to_team_taxonomy(base)

    if base.entity == "qb":
        return base.category, base.subcategory
    return _map_team_metric_to_qb_taxonomy(base)


def _map_qb_metric_to_team_taxonomy(base: MetricDef) -> tuple[str, str | None]:
    """Project QB metrics onto the team taxonomy for opp_qb_* columns."""
    if base.name == "qb_offense_snaps":
        return "Offense", "Total"
    if base.category == "Rushing":
        return "Offense", "Rushing"
    if base.category == "Scoring, Clutch & Outcomes":
        return "Offense", "Scoring"
    if base.category == "Turnovers & Ball Security":
        return "Offense", "Turnovers"
    return "Offense", "Passing"


def _map_team_metric_to_qb_taxonomy(base: MetricDef) -> tuple[str, str | None]:
    """Project team-defense context metrics onto the QB taxonomy for qopp_* columns."""
    if base.name == "points_allowed" or base.subcategory == "Scoring":
        return "Scoring, Clutch & Outcomes", None
    if base.name == "def_interceptions" or base.subcategory == "Turnovers":
        return "Turnovers & Ball Security", None
    if (
        base.name
        in {
            "def_sacks",
            "def_qb_hits",
            "def_pass_defended",
            "def_tackles_for_loss",
        }
        or base.subcategory == "Pressure & Playmaking"
    ):
        return "Pressure, Sacks & Pocket", None
    if base.subcategory == "Rushing":
        return "Rushing", None
    return "Passing Efficiency", None


@lru_cache(maxsize=1)
def get_registry() -> MetricRegistry:
    """Build, validate, and cache the project registry."""
    from nfl_sos_ratings.metrics.catalog import build_registry

    return build_registry()
