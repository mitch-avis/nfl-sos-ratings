"""Tests for scripts/refresh-season.sh, the weekly rebuild of the season in progress.

Each test copies the script into a temporary tree whose ``.venv/bin`` holds stand-ins for the
project's commands, so the script runs end to end without touching the real ``data/``.
"""

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh-season.sh"


def _tree(tmp_path: Path, *, failing: str | None = None) -> Path:
    """Return a repo-shaped tree with the script, a data file, and recording stand-in commands.

    Each stand-in appends its arguments to ``calls.txt``; the one named ``failing`` exits 1.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy2(_SCRIPT, tmp_path / "scripts" / _SCRIPT.name)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "2026_ratings.parquet").write_text("before", encoding="utf-8")
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("nfl-sos-ratings", "pytest"):
        status = 1 if name == failing else 0
        stand_in = bin_dir / name
        stand_in.write_text(
            f'#!/usr/bin/env bash\necho "{name} $*" >> "{tmp_path}/calls.txt"\nexit {status}\n',
            encoding="utf-8",
        )
        stand_in.chmod(0o755)
    return tmp_path


def _run(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copied script with ``args`` and capture its output."""
    return subprocess.run(  # noqa: S603  # fixed argv: the copied repo script and test flags
        [str(tree / "scripts" / _SCRIPT.name), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _calls(tree: Path) -> list[str]:
    """Return the stand-in commands the script ran, in order."""
    path = tree / "calls.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_help_prints_the_usage(tmp_path: Path) -> None:
    # Arrange
    tree = _tree(tmp_path)

    # Act
    result = _run(tree, "--help")

    # Assert
    assert result.returncode == 0
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("args", [["--bogus"], ["--season", "26"], ["--season"]])
def test_bad_arguments_exit_with_a_usage_error(tmp_path: Path, args: list[str]) -> None:
    # Arrange
    tree = _tree(tmp_path)

    # Act
    result = _run(tree, *args)

    # Assert
    assert result.returncode == 2
    assert "refresh-season" in result.stderr


def test_dry_run_lists_every_step_and_runs_none(tmp_path: Path) -> None:
    # Arrange
    tree = _tree(tmp_path)

    # Act
    result = _run(tree, "--dry-run", "--season", "2026")

    # Assert
    assert result.returncode == 0
    steps = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert [step.split()[1] for step in steps] == [
        "cp",
        ".venv/bin/nfl-sos-ratings",
        ".venv/bin/pytest",
        ".venv/bin/nfl-sos-ratings",
    ]
    assert "season --season 2026" in steps[1]
    assert "diff-data" in steps[3]
    assert steps[3].endswith("--season 2026")
    assert _calls(tree) == []
    assert not (tree / "logs").exists()


def test_a_refresh_runs_the_steps_logs_them_and_removes_the_copy(tmp_path: Path) -> None:
    # Arrange
    tree = _tree(tmp_path)

    # Act
    result = _run(tree, "--season", "2026")

    # Assert
    assert result.returncode == 0, result.stderr
    calls = _calls(tree)
    assert calls[0] == "nfl-sos-ratings season --season 2026"
    assert calls[1].startswith("pytest -m published_data")
    assert calls[2].startswith("nfl-sos-ratings diff-data --before ")
    backup = Path(calls[2].split()[3])
    assert not backup.exists()
    log = tree / "logs" / f"refresh-{datetime.now(UTC).astimezone():%Y%m%d}.log"
    assert "diff-data" in log.read_text(encoding="utf-8")


def test_a_failing_step_exits_nonzero_and_keeps_the_copy(tmp_path: Path) -> None:
    # Arrange
    tree = _tree(tmp_path, failing="pytest")

    # Act
    result = _run(tree, "--season", "2026")

    # Assert
    assert result.returncode != 0
    kept = next(
        line.split("kept at ", 1)[1].strip()
        for line in result.stderr.splitlines()
        if "kept at " in line
    )
    assert (Path(kept) / "2026_ratings.parquet").read_text(encoding="utf-8") == "before"
    shutil.rmtree(kept)
