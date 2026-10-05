"""The fixed row order of every Parquet data file the pipeline writes.

Every data file is written in one fixed row order, so two builds from the same inputs give
identical files and a rebuild's diff shows only real changes. Files read top-down keep their
published order (best first, ties broken by id); every other file is in key order: its first
identity column present, then week and game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

PUBLISHED_ROW_ORDER: dict[str, tuple[tuple[str, bool], ...]] = {
    "ratings": (("team_rating", True), ("team", False)),
    "qb_ratings": (("adj_qb_epa_per_dropback", True), ("qb_id", False)),
    "rating_ranges": (("team_rank", False), ("team", False)),
    "qb_rating_ranges": (("qb_rank", False), ("qb_id", False)),
}
ROW_IDENTITY_KEYS = ("qb_id", "team")
ROW_EVENT_KEYS = ("week", "game_id")


def data_file_row_order(suffix: str, columns: Collection[str]) -> tuple[tuple[str, bool], ...]:
    """Return the ``(column, descending)`` sort that fixes a data file's row order.

    Args:
        suffix: The file name after the season, for example ``ratings_by_week``.
        columns: The file's columns.

    Raises:
        ValueError: The file has no published order and no identity column to sort by.

    """
    if suffix in PUBLISHED_ROW_ORDER:
        return PUBLISHED_ROW_ORDER[suffix]
    identity = next((key for key in ROW_IDENTITY_KEYS if key in columns), None)
    if identity is None:
        msg = (
            f"Output {suffix} has no row order: it needs one of {', '.join(ROW_IDENTITY_KEYS)} "
            "or an entry in PUBLISHED_ROW_ORDER"
        )
        raise ValueError(msg)
    keys = (identity, *(key for key in ROW_EVENT_KEYS if key in columns))
    return tuple((key, False) for key in keys)


__all__ = ["PUBLISHED_ROW_ORDER", "ROW_EVENT_KEYS", "ROW_IDENTITY_KEYS", "data_file_row_order"]
