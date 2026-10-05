"""Tests for the read-only comparison of two directories of Parquet outputs."""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.data_diff import (
    ColumnChange,
    FileStatus,
    diff_data_dirs,
    diff_frames,
    format_diffs,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(directory: Path, name: str, frame: pl.DataFrame) -> None:
    """Write ``frame`` as ``name`` under ``directory``, creating the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / name)


def _ratings(values: list[float]) -> pl.DataFrame:
    """Return a three-team ratings frame with the given ratings."""
    return pl.DataFrame({"team": ["BUF", "MIA", "NE"], "team_rating": values})


def test_diff_frames_reports_identical_frames_unchanged() -> None:
    # Arrange
    frame = _ratings([1.0, 2.0, 3.0])

    # Act
    diff = diff_frames(frame, frame.clone())

    # Assert
    assert diff.status is FileStatus.UNCHANGED
    assert diff.column_changes == ()


def test_diff_frames_reports_reordered_rows_as_row_order_only() -> None:
    # Arrange
    frame = _ratings([1.0, 2.0, 3.0])

    # Act
    diff = diff_frames(frame, frame.reverse())

    # Assert
    assert diff.status is FileStatus.ROW_ORDER


def test_diff_frames_counts_changed_rows_and_the_largest_difference() -> None:
    # Arrange
    before = _ratings([1.0, 2.0, 3.0])
    after = _ratings([1.0, 2.5, 2.0])

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.status is FileStatus.VALUES
    assert diff.column_changes == (ColumnChange("team_rating", 2, 1.0),)


def test_diff_frames_aligns_rows_by_identity_before_comparing() -> None:
    # Arrange
    before = pl.DataFrame({"team_rating": [1.0, 2.0], "team": ["BUF", "NE"]})
    after = pl.DataFrame({"team_rating": [2.0, 3.0], "team": ["NE", "BUF"]})

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.aligned_by == ("team",)
    assert diff.column_changes == (ColumnChange("team_rating", 1, 2.0),)


def test_diff_frames_aligns_qb_rows_by_passer_and_week() -> None:
    # Arrange
    before = pl.DataFrame({"qb_id": ["00-1", "00-1"], "team": ["NE", "NE"], "week": [1, 2]})
    after = before.reverse()

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert (diff.status, diff.aligned_by) == (FileStatus.ROW_ORDER, ("qb_id", "week"))


def test_diff_frames_counts_rows_added_and_removed_by_identity() -> None:
    # Arrange
    before = pl.DataFrame({"team": ["BUF", "MIA"], "team_rating": [1.0, 2.0]})
    after = pl.DataFrame({"team": ["BUF", "NE", "NYJ"], "team_rating": [1.0, 3.0, 4.0]})

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.status is FileStatus.VALUES
    assert (diff.rows_added, diff.rows_removed) == (2, 1)
    assert diff.column_changes == ()


def test_diff_frames_aligns_by_explicit_keys() -> None:
    # Arrange
    before = pl.DataFrame({"value": [1.0, 2.0], "game_id": ["g1", "g2"]})
    after = pl.DataFrame({"value": [5.0, 2.0], "game_id": ["g1", "g2"]})

    # Act
    diff = diff_frames(before, after, keys=["game_id"])

    # Assert
    assert diff.aligned_by == ("game_id",)
    assert diff.column_changes == (ColumnChange("value", 1, 4.0),)


def test_diff_frames_matches_rows_whose_key_is_null() -> None:
    # Arrange
    before = pl.DataFrame({"team": ["NE", "NE"], "wp_bin": [None, 3], "value": [1.0, 2.0]})
    after = pl.DataFrame({"team": ["NE", "NE"], "wp_bin": [3, None], "value": [2.0, 5.0]})

    # Act
    diff = diff_frames(before, after, keys=["team", "wp_bin"])

    # Assert
    assert diff.aligned_by == ("team", "wp_bin")
    assert diff.column_changes == (ColumnChange("value", 1, 4.0),)


def test_diff_frames_compares_sorted_rows_when_identity_keys_repeat() -> None:
    # Arrange
    before = pl.DataFrame({"team": ["NE", "NE"], "value": [1.0, 2.0]})
    after = before.reverse()

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert (diff.status, diff.aligned_by) == (FileStatus.ROW_ORDER, ())


def test_diff_frames_reports_column_changes_as_a_schema_change() -> None:
    # Arrange
    before = pl.DataFrame({"team": ["NE"], "old": [1.0], "games": [17]})
    after = pl.DataFrame({"team": ["NE"], "new": [1.0], "games": [17.0]})

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.status is FileStatus.SCHEMA
    assert (diff.added_columns, diff.removed_columns, diff.retyped_columns) == (
        ("new",),
        ("old",),
        ("games",),
    )


def test_diff_frames_ignores_differences_within_the_tolerance() -> None:
    # Arrange
    before = _ratings([1.0, 2.0, 3.0])
    after = _ratings([1.0, 2.0 + 1e-12, 3.0])

    # Act
    diff = diff_frames(before, after, tolerance=1e-9)

    # Assert
    assert diff.status is FileStatus.UNCHANGED


def test_diff_frames_treats_matching_nan_and_null_as_unchanged() -> None:
    # Arrange
    frame = pl.DataFrame({"team": ["BUF", "NE"], "value": [float("nan"), None]})

    # Act
    diff = diff_frames(frame, frame.clone())

    # Assert
    assert diff.status is FileStatus.UNCHANGED


def test_diff_frames_reports_text_changes_without_a_numeric_difference() -> None:
    # Arrange
    before = pl.DataFrame({"team": ["NE"], "qb_name": ["T.Brady"]})
    after = pl.DataFrame({"team": ["NE"], "qb_name": ["D.Maye"]})

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.column_changes == (ColumnChange("qb_name", 1, None),)


def test_diff_frames_does_not_compare_values_of_rows_it_cannot_align() -> None:
    # Arrange
    before = pl.DataFrame({"week": [1, 2], "value": [1.0, 2.0]})
    after = pl.DataFrame({"week": [1, 2, 3], "value": [1.0, 2.0, 3.0]})

    # Act
    diff = diff_frames(before, after)

    # Assert
    assert diff.status is FileStatus.VALUES
    assert (diff.rows_before, diff.rows_after, diff.aligned_by) == (2, 3, ())
    assert diff.column_changes == ()


def test_diff_data_dirs_reports_added_and_removed_files(tmp_path: Path) -> None:
    # Arrange
    _write(tmp_path / "before", "2025_ratings.parquet", _ratings([1.0, 2.0, 3.0]))
    _write(tmp_path / "before", "2025_old.parquet", _ratings([1.0, 2.0, 3.0]))
    _write(tmp_path / "after", "2025_ratings.parquet", _ratings([1.0, 2.0, 3.0]))
    _write(tmp_path / "after", "2025_new.parquet", _ratings([1.0, 2.0, 3.0]))

    # Act
    diffs = diff_data_dirs(tmp_path / "before", tmp_path / "after")

    # Assert
    assert [(diff.name, diff.status) for diff in diffs] == [
        ("2025_new.parquet", FileStatus.ADDED),
        ("2025_old.parquet", FileStatus.REMOVED),
        ("2025_ratings.parquet", FileStatus.UNCHANGED),
    ]


def test_diff_data_dirs_limits_the_comparison_to_one_season(tmp_path: Path) -> None:
    # Arrange
    for season in (2024, 2025):
        _write(tmp_path / "before", f"{season}_ratings.parquet", _ratings([1.0, 2.0, 3.0]))
        _write(tmp_path / "after", f"{season}_ratings.parquet", _ratings([1.0, 2.0, 3.0]))

    # Act
    diffs = diff_data_dirs(tmp_path / "before", tmp_path / "after", season=2024)

    # Assert
    assert [diff.name for diff in diffs] == ["2024_ratings.parquet"]


def test_format_diffs_lists_changed_columns_and_a_summary(tmp_path: Path) -> None:
    # Arrange
    diffs = [
        diff_frames(_ratings([1.0, 2.0, 3.0]), _ratings([1.0, 2.5, 3.0]), name="2025_ratings"),
        diff_frames(_ratings([1.0, 2.0, 3.0]), _ratings([1.0, 2.0, 3.0]), name="2025_combined"),
    ]

    # Act
    report = format_diffs(diffs, tmp_path / "before", tmp_path / "after")

    # Assert
    lines = report.splitlines()
    assert "2025_ratings: values changed (3 -> 3 rows, aligned by team)" in lines
    assert "  team_rating: 1 row changed, largest absolute difference 0.5" in lines
    assert "2025_combined: unchanged" in lines
    assert lines[-1] == (
        "Summary: 1 unchanged, 0 row order only, 1 values changed, 0 schema changed, 0 added, "
        "0 removed"
    )


def test_format_diffs_says_how_rows_were_matched_and_what_moved(tmp_path: Path) -> None:
    # Arrange
    weeks = pl.DataFrame({"week": [1, 2], "value": [1.0, 2.0]})
    diffs = [
        diff_frames(weeks, weeks.with_columns(pl.col("value") * 2), name="sorted"),
        diff_frames(weeks, weeks.head(1), name="unaligned"),
        diff_frames(_ratings([1.0, 2.0, 3.0]), _ratings([1.0, 2.0, 3.0]).head(2), name="moved"),
        diff_frames(weeks, weeks.rename({"value": "renamed"}), name="schema"),
    ]

    # Act
    report = format_diffs(diffs, tmp_path / "before", tmp_path / "after")

    # Assert
    lines = report.splitlines()
    assert "sorted: values changed (2 -> 2 rows, aligned by sorted position)" in lines
    assert (
        "unaligned: values changed (2 -> 1 rows, rows not aligned; pass --keys to compare values)"
        in lines
    )
    assert "moved: values changed (3 -> 2 rows, aligned by team; 0 added, 1 removed)" in lines
    assert "  added columns: renamed" in lines
    assert "  removed columns: value" in lines


def test_main_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    _write(tmp_path / "before", "2025_ratings.parquet", _ratings([1.0, 2.0, 3.0]))
    _write(tmp_path / "after", "2025_ratings.parquet", _ratings([3.0, 2.0, 1.0]))

    # Act
    main(["--before", str(tmp_path / "before"), "--after", str(tmp_path / "after")])

    # Assert
    output = capsys.readouterr().out
    assert "2025_ratings.parquet: values changed (3 -> 3 rows, aligned by team)" in output
    assert "  team_rating: 2 rows changed, largest absolute difference 2" in output


def test_main_rejects_a_missing_directory(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "before").mkdir()

    # Act & Assert
    with pytest.raises(SystemExit) as exit_info:
        main(["--before", str(tmp_path / "before"), "--after", str(tmp_path / "missing")])

    # Assert
    assert exit_info.value.code == 2


def test_main_matches_rows_by_the_keys_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    before = pl.DataFrame({"value": [1.0, 2.0], "game_id": ["g1", "g2"]})
    _write(tmp_path / "before", "2025_games.parquet", before)
    _write(tmp_path / "after", "2025_games.parquet", before.reverse())
    _write(tmp_path / "before", "2024_games.parquet", before)

    # Act
    main(
        [
            "--before",
            str(tmp_path / "before"),
            "--after",
            str(tmp_path / "after"),
            "--season",
            "2025",
            "--keys",
            "game_id",
        ]
    )

    # Assert
    output = capsys.readouterr().out
    assert "2025_games.parquet: row order only" in output
    assert "2024_games.parquet" not in output
