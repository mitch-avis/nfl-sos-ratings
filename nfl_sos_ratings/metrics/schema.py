"""Typed building blocks for the metric registry single source of truth.

Every stat, rating, and metric the project publishes is defined once as a
:class:`MetricDef`. Concrete data columns are either exact metric names or
affix products (a prefix such as ``opp_`` and/or a suffix such as
``_per_game`` around a base metric), resolved by the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict, Unpack

Entity = Literal["team", "qb"]
"""Which page family a metric belongs to: the Teams pages or the QBs pages."""

Shape = Literal["count", "rate", "avg", "flag", "id", "score"]
"""How a metric behaves across views.

- ``count``: a summable total (yards, touchdowns). Valid in every view.
- ``rate``: an intrinsic ratio with its own denominator. Never divided again.
- ``avg``: a per-event mean re-averaged over events, not over weeks.
- ``flag``: a boolean marker (eligibility, comeback credit).
- ``id``: identity text (team codes, player names, game ids).
- ``score``: a model output on its own scale (z-scored ratings, SRS points).
"""

Polarity = Literal["higher", "lower", "neutral"]
"""Which end of the scale is good for the subject of the row."""

Status = Literal["implemented", "planned"]
"""Whether the pipeline currently produces the metric or it is catalogued
for the play-by-play metric expansion."""


@dataclass(frozen=True, slots=True)
class MetricProvenance:
    """Structured provenance for fitted or externally maintained metrics."""

    target: str | None = None
    fit_window: tuple[int, int] | None = None
    fitting_command: str | None = None
    refit_policy: str | None = None
    sample_weighting: str | None = None
    weight_snapshot: tuple[tuple[str, float], ...] = ()
    holdout_metrics: tuple[tuple[str, float], ...] = ()
    excluded_weight_candidates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class MetricDef:
    """One stat/rating/metric definition — the single source of truth entry."""

    name: str
    label: str
    full_name: str
    description: str
    entity: Entity
    category: str
    shape: Shape
    polarity: Polarity
    source: str
    subcategory: str | None = None
    denominator: str | None = None
    since: int | None = None
    ratings_eligible: bool = False
    duplicate_of: str | None = None
    status: Status = "implemented"
    contextual: bool = False
    formula: str | None = None
    note: str | None = None
    provenance: MetricProvenance | None = None


@dataclass(frozen=True, slots=True)
class CategoryDef:
    """A display category for one entity, with ordered subcategories."""

    name: str
    entity: Entity
    description: str
    subcategories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RatingPool:
    """An explicit allowlist of columns that feed one rating computation."""

    name: str
    entity: Entity
    description: str
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrefixRule:
    """How a column prefix transforms the base metric's presentation."""

    prefix: str
    label_template: str
    full_name_template: str
    description_note: str
    contextual: bool = False
    invert_polarity_for_qb: bool = False
    category_override: str | None = None


@dataclass(frozen=True, slots=True)
class SuffixRule:
    """How a column suffix transforms the base metric's presentation."""

    suffix: str
    label_template: str
    full_name_template: str
    description_note: str


@dataclass(frozen=True, slots=True)
class ResolvedColumn:
    """A concrete data column resolved to its base metric plus affix context."""

    column: str
    base: MetricDef
    label: str
    full_name: str
    description: str
    polarity: Polarity
    contextual: bool
    category: str
    subcategory: str | None


class MetricFields(TypedDict):
    """Keyword fields accepted by section builders (entity/category preset)."""

    name: str
    label: str
    full_name: str
    description: str
    shape: Shape
    polarity: Polarity
    source: str
    denominator: NotRequired[str | None]
    since: NotRequired[int | None]
    ratings_eligible: NotRequired[bool]
    duplicate_of: NotRequired[str | None]
    status: NotRequired[Status]
    contextual: NotRequired[bool]
    formula: NotRequired[str | None]
    note: NotRequired[str | None]
    provenance: NotRequired[MetricProvenance | None]


class MetricBuilder(Protocol):
    """A callable that builds metrics with entity/category/subcategory preset."""

    def __call__(self, **fields: Unpack[MetricFields]) -> MetricDef:
        """Build one metric definition from the remaining fields."""
        ...


def section(entity: Entity, category: str, subcategory: str | None = None) -> MetricBuilder:
    """Return a builder that stamps entity, category, and subcategory."""

    def build(**fields: Unpack[MetricFields]) -> MetricDef:
        """Build one metric definition inside the preset section."""
        return MetricDef(entity=entity, category=category, subcategory=subcategory, **fields)

    return build
