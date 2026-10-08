"""Tests for the background runner behind the app's refresh button."""

import time
from typing import TYPE_CHECKING

import pytest

from nfl_sos_ratings.refresh_runner import RefreshRunner

if TYPE_CHECKING:
    from pathlib import Path


def _script(tmp_path: Path, body: str) -> list[str]:
    """Return a command that runs a small Bash script with ``body``."""
    path = tmp_path / "refresh.sh"
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return [str(path)]


def _wait(runner: RefreshRunner) -> None:
    """Wait up to 10 seconds for the runner to finish."""
    deadline = time.monotonic() + 10
    while runner.status().state == "running" and time.monotonic() < deadline:
        time.sleep(0.02)


def test_status_is_idle_before_any_run(tmp_path: Path) -> None:
    # Arrange
    runner = RefreshRunner(_script(tmp_path, "exit 0"), cwd=tmp_path)

    # Act
    status = runner.status()

    # Assert
    assert (status.state, status.exit_code, status.log_tail) == ("idle", None, [])


def test_a_clean_run_succeeds_and_keeps_the_diff_summary(tmp_path: Path) -> None:
    # Arrange
    body = 'echo "rebuilding"\necho "Summary: 18 unchanged, 2 values changed"\nexit 0'
    runner = RefreshRunner(_script(tmp_path, body), cwd=tmp_path)

    # Act
    started = runner.start()
    _wait(runner)

    # Assert
    status = runner.status()
    assert started is True
    assert (status.state, status.exit_code) == ("succeeded", 0)
    assert status.summary == "Summary: 18 unchanged, 2 values changed"
    assert status.log_tail == ["rebuilding", "Summary: 18 unchanged, 2 values changed"]
    assert status.started_at is not None
    assert status.finished_at is not None


def test_a_failing_run_reports_failure_and_its_last_lines(tmp_path: Path) -> None:
    # Arrange
    runner = RefreshRunner(_script(tmp_path, 'echo "season failed" >&2\nexit 3'), cwd=tmp_path)

    # Act
    runner.start()
    _wait(runner)

    # Assert
    status = runner.status()
    assert (status.state, status.exit_code, status.summary) == ("failed", 3, None)
    assert status.log_tail == ["season failed"]


def test_start_refuses_a_second_run_while_one_is_running(tmp_path: Path) -> None:
    # Arrange
    runner = RefreshRunner(_script(tmp_path, "sleep 1"), cwd=tmp_path)
    runner.start()

    # Act
    second = runner.start()

    # Assert
    assert second is False
    _wait(runner)


def test_the_log_tail_keeps_only_the_last_lines(tmp_path: Path) -> None:
    # Arrange
    runner = RefreshRunner(_script(tmp_path, "seq 1 10"), cwd=tmp_path, max_lines=3)

    # Act
    runner.start()
    _wait(runner)

    # Assert
    assert runner.status().log_tail == ["8", "9", "10"]


def test_a_command_that_cannot_start_fails_without_running(tmp_path: Path) -> None:
    # Arrange
    runner = RefreshRunner([str(tmp_path / "missing.sh")], cwd=tmp_path)

    # Act
    started = runner.start()

    # Assert
    status = runner.status()
    assert started is True
    assert status.state == "failed"
    assert status.log_tail[0].startswith("Could not start")


@pytest.mark.parametrize("state", ["idle", "succeeded"])
def test_a_finished_or_idle_runner_can_start_again(tmp_path: Path, state: str) -> None:
    # Arrange
    runner = RefreshRunner(_script(tmp_path, "exit 0"), cwd=tmp_path)
    if state == "succeeded":
        runner.start()
        _wait(runner)

    # Act
    started = runner.start()
    _wait(runner)

    # Assert
    assert started is True
    assert runner.status().state == "succeeded"
