"""Compare two directories of Parquet outputs file by file, without changing either.

A rebuild check: point ``--before`` at a copy of ``data/`` taken before a rebuild and ``--after``
at the rebuilt directory. Each file is reported as unchanged, row order only, values changed,
schema changed, added, or removed; changed columns list how many rows changed and the largest
absolute difference. Rows are matched by the identity columns the season writer sorts by
(``row_order.row_identity_keys``) when those identify every row on both sides, by ``--keys`` when
given, and otherwise by sorted position.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from nfl_sos_ratings.row_order import row_identity_keys

if TYPE_CHECKING:
    from collections.abc import Sequence

_PARQUET_GLOB = "*.parquet"
_SEASON_PREFIX = re.compile(r"^(?P<season>\d{4})_")
# Suffix for the after-side copy of each column while two aligned frames sit side by side.
_AFTER_SUFFIX = "__after"


class FileStatus(StrEnum):
    """How one data file differs between the two directories, in report order."""

    UNCHANGED = "unchanged"
    ROW_ORDER = "row order only"
    VALUES = "values changed"
    SCHEMA = "schema changed"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class ColumnChange:
    """One column's count of changed rows and, for numbers, the largest absolute difference."""

    column: str
    changed_rows: int
    max_abs_difference: float | None


@dataclass(frozen=True, slots=True)
class FileDiff:
    """The comparison of one data file.

    ``aligned_by`` names the columns rows were matched on; it is empty when rows were matched by
    sorted position, or not matched at all because the row counts differ. ``rows_added`` and
    ``rows_removed`` count rows whose identity appears on only one side.
    """

    name: str
    status: FileStatus
    rows_before: int | None = None
    rows_after: int | None = None
    aligned_by: tuple[str, ...] = ()
    rows_added: int = 0
    rows_removed: int = 0
    added_columns: tuple[str, ...] = ()
    removed_columns: tuple[str, ...] = ()
    retyped_columns: tuple[str, ...] = ()
    column_changes: tuple[ColumnChange, ...] = ()


def _identifies_rows(frame: pl.DataFrame, keys: Sequence[str]) -> bool:
    """Return whether ``keys`` are columns of ``frame`` with no duplicated or null combination."""
    if not keys or not set(keys) <= set(frame.columns):
        return False
    key_frame = frame.select(keys)
    return (
        not key_frame.is_duplicated().any() and key_frame.null_count().sum_horizontal().item() == 0
    )


def _alignment_keys(
    before: pl.DataFrame, after: pl.DataFrame, keys: Sequence[str] | None
) -> tuple[str, ...]:
    """Return the columns to match rows on, or ``()`` to match by sorted position."""
    candidates = tuple(keys) if keys is not None else row_identity_keys(before.columns)
    if _identifies_rows(before, candidates) and _identifies_rows(after, candidates):
        return candidates
    return ()


def _changed(column: str, *, numeric: bool, tolerance: float) -> pl.Expr:
    """Return whether ``column`` differs from its after-side copy, row by row.

    Matching nulls and matching NaNs count as equal; numbers within ``tolerance`` do too.
    """
    before, after = pl.col(column), pl.col(f"{column}{_AFTER_SUFFIX}")
    same = before.eq_missing(after)
    if numeric and tolerance > 0:
        within = (before.cast(pl.Float64) - after.cast(pl.Float64)).abs() <= tolerance
        same |= within.fill_null(value=False)
    return ~same


def _column_changes(
    before: pl.DataFrame, after: pl.DataFrame, tolerance: float
) -> tuple[ColumnChange, ...]:
    """Compare two frames with the same columns and height row by row; return changed columns."""
    if before.height == 0 or before.width == 0:
        return ()
    pairs = pl.concat(
        [before, after.rename({column: f"{column}{_AFTER_SUFFIX}" for column in after.columns})],
        how="horizontal",
    )
    numeric = {column: dtype.is_numeric() for column, dtype in before.schema.items()}
    expressions: list[pl.Expr] = []
    for index, column in enumerate(before.columns):
        changed = _changed(column, numeric=numeric[column], tolerance=tolerance)
        expressions.append(changed.sum().alias(f"changed_{index}"))
        if numeric[column]:
            difference = pl.col(column).cast(pl.Float64) - pl.col(f"{column}{_AFTER_SUFFIX}").cast(
                pl.Float64
            )
            expressions.append(difference.abs().filter(changed).max().alias(f"largest_{index}"))
    stats = pairs.select(expressions).row(0, named=True)
    changes: list[ColumnChange] = []
    for index, column in enumerate(before.columns):
        changed_rows = int(cast("int", stats[f"changed_{index}"]))
        if changed_rows == 0:
            continue
        largest = cast("float | None", stats.get(f"largest_{index}"))
        changes.append(
            ColumnChange(column, changed_rows, None if largest is None else float(largest))
        )
    return tuple(changes)


def diff_frames(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    name: str = "",
    keys: Sequence[str] | None = None,
    tolerance: float = 0.0,
) -> FileDiff:
    """Compare one data file's two versions.

    Args:
        before: The file as it was.
        after: The file as it is now.
        name: The file name to report.
        keys: Columns that identify a row; by default the data files' identity columns. Rows are
            matched by sorted position when the keys do not identify every row on both sides.
        tolerance: Numeric differences up to this absolute size count as unchanged.

    Returns:
        The file's status, row counts and alignment, schema changes, and changed columns
        (compared over the columns both versions share with the same type).

    """
    before_types, after_types = before.schema, after.schema
    added = tuple(column for column in after.columns if column not in before_types)
    removed = tuple(column for column in before.columns if column not in after_types)
    retyped = tuple(
        column
        for column in before.columns
        if column in after_types and before_types[column] != after_types[column]
    )
    shared = [
        column for column in before.columns if column in after_types and column not in retyped
    ]
    left, right = before.select(shared), after.select(shared)
    aligned_by = _alignment_keys(left, right, keys)
    if not (added or removed or retyped) and before.equals(after):
        # Exact equality (matching nulls and NaNs included) settles most files of a rebuild
        # without the per-column comparison.
        return FileDiff(
            name=name,
            status=FileStatus.UNCHANGED,
            rows_before=before.height,
            rows_after=after.height,
            aligned_by=aligned_by,
        )

    rows_added = rows_removed = 0
    changes: tuple[ColumnChange, ...] = ()
    if aligned_by:
        on = list(aligned_by)
        rows_removed = left.join(right.select(on), on=on, how="anti").height
        rows_added = right.join(left.select(on), on=on, how="anti").height
        values = [column for column in shared if column not in aligned_by]
        changes = _column_changes(
            left.join(right.select(on), on=on, how="semi").sort(on).select(values),
            right.join(left.select(on), on=on, how="semi").sort(on).select(values),
            tolerance,
        )
    elif left.height == right.height:
        changes = _column_changes(left.sort(shared), right.sort(shared), tolerance)

    if added or removed or retyped:
        status = FileStatus.SCHEMA
    elif changes or rows_added or rows_removed or left.height != right.height:
        status = FileStatus.VALUES
    elif _column_changes(left, right, tolerance):
        status = FileStatus.ROW_ORDER
    else:
        status = FileStatus.UNCHANGED
    return FileDiff(
        name=name,
        status=status,
        rows_before=before.height,
        rows_after=after.height,
        aligned_by=aligned_by,
        rows_added=rows_added,
        rows_removed=rows_removed,
        added_columns=added,
        removed_columns=removed,
        retyped_columns=retyped,
        column_changes=changes,
    )


def _file_names(directory: Path, season: int | None) -> set[str]:
    """Return the Parquet file names in ``directory``, limited to one season when given."""
    names = {path.name for path in directory.glob(_PARQUET_GLOB)}
    if season is None:
        return names
    return {
        name
        for name in names
        if (match := _SEASON_PREFIX.match(name)) is not None and int(match["season"]) == season
    }


def diff_data_dirs(
    before_dir: Path,
    after_dir: Path,
    *,
    season: int | None = None,
    keys: Sequence[str] | None = None,
    tolerance: float = 0.0,
) -> list[FileDiff]:
    """Compare every Parquet file in two directories, in file-name order.

    Args:
        before_dir: The directory as it was, for example a copy of ``data/`` before a rebuild.
        after_dir: The directory as it is now.
        season: Only compare files whose names start with this season.
        keys: Row identity columns passed to ``diff_frames`` for every file.
        tolerance: Numeric tolerance passed to ``diff_frames``.

    Returns:
        One ``FileDiff`` per file name found in either directory.

    """
    before_names, after_names = _file_names(before_dir, season), _file_names(after_dir, season)
    diffs: list[FileDiff] = []
    for name in sorted(before_names | after_names):
        if name not in before_names:
            diffs.append(FileDiff(name=name, status=FileStatus.ADDED))
        elif name not in after_names:
            diffs.append(FileDiff(name=name, status=FileStatus.REMOVED))
        else:
            diffs.append(
                diff_frames(
                    pl.read_parquet(before_dir / name),
                    pl.read_parquet(after_dir / name),
                    name=name,
                    keys=keys,
                    tolerance=tolerance,
                )
            )
    return diffs


def _detail(diff: FileDiff) -> str:
    """Return the parenthetical row summary for a file whose values or schema changed."""
    if diff.aligned_by:
        alignment = f"aligned by {', '.join(diff.aligned_by)}"
    elif diff.rows_before == diff.rows_after:
        alignment = "aligned by sorted position"
    else:
        alignment = "rows not aligned; pass --keys to compare values"
    moved = (
        f"; {diff.rows_added} added, {diff.rows_removed} removed"
        if diff.rows_added or diff.rows_removed
        else ""
    )
    return f" ({diff.rows_before} -> {diff.rows_after} rows, {alignment}{moved})"


def _file_lines(diff: FileDiff) -> list[str]:
    """Return the report lines for one file."""
    if diff.status not in {FileStatus.VALUES, FileStatus.SCHEMA}:
        return [f"{diff.name}: {diff.status}"]
    lines = [f"{diff.name}: {diff.status}{_detail(diff)}"]
    for label, columns in (
        ("added columns", diff.added_columns),
        ("removed columns", diff.removed_columns),
        ("retyped columns", diff.retyped_columns),
    ):
        if columns:
            lines.append(f"  {label}: {', '.join(columns)}")
    for change in diff.column_changes:
        rows = "row" if change.changed_rows == 1 else "rows"
        largest = (
            ""
            if change.max_abs_difference is None
            else f", largest absolute difference {change.max_abs_difference:.6g}"
        )
        lines.append(f"  {change.column}: {change.changed_rows} {rows} changed{largest}")
    return lines


def format_diffs(diffs: Sequence[FileDiff], before_dir: Path, after_dir: Path) -> str:
    """Return the plain-text report: a header, each file's lines, and a summary of statuses."""
    lines = [f"Comparing {before_dir} (before) with {after_dir} (after): {len(diffs)} files"]
    for diff in diffs:
        lines.extend(_file_lines(diff))
    counts = ", ".join(
        f"{sum(diff.status is status for diff in diffs)} {status}" for status in FileStatus
    )
    lines.append(f"Summary: {counts}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``diff-data`` command's options and check both directories exist."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings diff-data",
        description=(
            "Compare two directories of Parquet outputs file by file (read-only), for example a "
            "copy of data/ taken before a rebuild against the rebuilt data/."
        ),
    )
    parser.add_argument("--before", type=Path, required=True, help="Directory as it was.")
    parser.add_argument("--after", type=Path, required=True, help="Directory as it is now.")
    parser.add_argument("--season", type=int, help="Only compare this season's files.")
    parser.add_argument(
        "--keys",
        help=(
            "Comma-separated columns that identify a row in every file (default: qb_id or team, "
            "then week and game_id where present)."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Treat numeric differences up to this absolute size as unchanged (default: 0).",
    )
    args = parser.parse_args(argv)
    for directory in (args.before, args.after):
        if not directory.is_dir():
            parser.error(f"not a directory: {directory}")
    return args


def main(argv: list[str] | None = None) -> None:
    """Print the comparison of two data directories."""
    args = _parse_args(argv)
    keys = None if args.keys is None else [key.strip() for key in args.keys.split(",")]
    diffs = diff_data_dirs(
        args.before, args.after, season=args.season, keys=keys, tolerance=args.tolerance
    )
    sys.stdout.write(format_diffs(diffs, args.before, args.after))


__all__ = [
    "ColumnChange",
    "FileDiff",
    "FileStatus",
    "diff_data_dirs",
    "diff_frames",
    "format_diffs",
    "main",
]
