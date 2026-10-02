"""Metric registry package — the single source of truth for published stats."""

from nfl_sos_ratings.metrics.catalog import get_registry
from nfl_sos_ratings.metrics.registry import MetricRegistry, RegistryValidationError
from nfl_sos_ratings.metrics.schema import (
    CategoryDef,
    MetricDef,
    MetricProvenance,
    RatingPool,
    ResolvedColumn,
)

__all__ = [
    "CategoryDef",
    "MetricDef",
    "MetricProvenance",
    "MetricRegistry",
    "RatingPool",
    "RegistryValidationError",
    "ResolvedColumn",
    "get_registry",
]
