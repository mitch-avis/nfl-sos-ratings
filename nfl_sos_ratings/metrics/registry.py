"""Registry engine: validation, column resolution, and API payloads.

The registry is the single source of truth for every published metric. The
ETL validates its data columns against it, the API serves it, and the
frontend derives labels, tooltips, sort defaults, and color-gradient
direction from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfl_sos_ratings.metrics.schema import (
    CategoryDef,
    Entity,
    MetricDef,
    Polarity,
    PrefixRule,
    ResolvedColumn,
    SuffixRule,
)
from nfl_sos_ratings.rating_ranges import RANGE_QUANTILES, quantile_suffix

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class RegistryValidationError(ValueError):
    """Raised when the registry data violates a structural invariant."""


# A layman description shorter than this is a label, not a sentence.
_MIN_DESCRIPTION_LENGTH = 20


def _quantile_suffix_rule(level: float) -> SuffixRule:
    """Return the suffix rule for one rank-range quantile column, such as ``_q025``."""
    percent = f"{round(level * 100, 1):g}"
    return SuffixRule(
        suffix=quantile_suffix(level),
        label_template=f"{{label}} ({percent}th pct)",
        full_name_template=f"{{full_name}}, {percent}th percentile",
        description_note=(
            f"Shown as the {percent}th percentile across game-bootstrap resamples of the season: "
            f"{percent}% of resampled seasons came out at or below it. The spread reflects "
            "which games happened to be played, not whether the model is right."
        ),
    )


# Longest prefixes first so qopp_ wins over opp_.
DEFAULT_PREFIX_RULES: tuple[PrefixRule, ...] = (
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
    *(_quantile_suffix_rule(level) for level in RANGE_QUANTILES),
)


class MetricRegistry:
    """Validated, queryable collection of metrics and categories."""

    def __init__(
        self,
        metrics: Sequence[MetricDef],
        categories: Sequence[CategoryDef],
        prefix_rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
        suffix_rules: Sequence[SuffixRule] = DEFAULT_SUFFIX_RULES,
    ) -> None:
        """Index the definitions and run full structural validation."""
        self.metrics: dict[str, MetricDef] = {}
        for metric in metrics:
            if metric.name in self.metrics:
                msg = f"Metric defined twice: {metric.name}"
                raise RegistryValidationError(msg)
            self.metrics[metric.name] = metric
        self._categories: tuple[CategoryDef, ...] = tuple(categories)
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
                    "duplicate_of": metric.duplicate_of,
                    "contextual": metric.contextual,
                    "formula": metric.formula,
                    "note": metric.note,
                }
                for metric in self.metrics.values()
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

    def _validate_metric(
        self,
        metric: MetricDef,
        category_index: dict[tuple[Entity, str], CategoryDef],
    ) -> None:
        """Check one metric's links, denominator rule, and description."""
        category = category_index.get((metric.entity, metric.category))
        if category is None:
            msg = f"Metric {metric.name} references unknown category {metric.category!r}"
            raise RegistryValidationError(msg)
        if metric.subcategory is not None and metric.subcategory not in category.subcategories:
            msg = f"Metric {metric.name} references unknown subcategory {metric.subcategory!r}"
            raise RegistryValidationError(msg)
        if metric.duplicate_of is not None and metric.duplicate_of not in self.metrics:
            msg = f"Metric {metric.name} duplicates unknown metric {metric.duplicate_of!r}"
            raise RegistryValidationError(msg)
        if metric.shape in ("rate", "avg") and not metric.denominator:
            msg = f"Metric {metric.name} is a {metric.shape} but declares no denominator"
            raise RegistryValidationError(msg)
        if (
            not metric.description.endswith(".")
            or len(metric.description) < _MIN_DESCRIPTION_LENGTH
        ):
            msg = f"Metric {metric.name} needs a full-sentence layman description"
            raise RegistryValidationError(msg)


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
