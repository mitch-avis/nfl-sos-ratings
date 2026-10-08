"""Run the weekly data refresh in the background for the app's refresh button.

The analyst server (``nfl-sos-ratings web --allow-refresh``) owns one ``RefreshRunner``. A refresh
runs ``scripts/refresh-season.sh`` (rebuild the season in progress, run the published-data checks,
and print ``diff-data`` against a copy of ``data/``) as a child process, one at a time, while the
server keeps answering requests; the app polls ``status`` until the run finishes, then refetches
its data. The server reads ``data/`` per request, so the new numbers show without a restart.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

type RefreshState = Literal["idle", "running", "succeeded", "failed"]

# How many of the run's last output lines the status keeps for the app.
DEFAULT_MAX_LINES = 200
# The line ``diff-data`` ends with, which the app shows as the run's result.
SUMMARY_PREFIX = "Summary:"


@dataclass(frozen=True, slots=True)
class RefreshStatus:
    """A snapshot of the runner: its state, timing, exit code, and the last lines of output."""

    state: RefreshState
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    summary: str | None
    log_tail: list[str]


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class RefreshRunner:
    """Run one refresh command at a time in the background and report its progress."""

    def __init__(
        self, command: Sequence[str], *, cwd: Path, max_lines: int = DEFAULT_MAX_LINES
    ) -> None:
        """Remember the command to run (argv form) and the directory to run it in."""
        self._command = list(command)
        self._cwd = cwd
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._state: RefreshState = "idle"
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._exit_code: int | None = None
        self._summary: str | None = None

    def status(self) -> RefreshStatus:
        """Return the current state and the last lines of output."""
        with self._lock:
            return RefreshStatus(
                state=self._state,
                started_at=self._started_at,
                finished_at=self._finished_at,
                exit_code=self._exit_code,
                summary=self._summary,
                log_tail=list(self._lines),
            )

    def start(self) -> bool:
        """Start a refresh unless one is running; return whether this call started one."""
        with self._lock:
            if self._state == "running":
                return False
            self._lines.clear()
            self._state = "running"
            self._started_at = _now()
            self._finished_at = None
            self._exit_code = None
            self._summary = None
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is the fixed refresh script
                self._command,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            self._finish(-1, [f"Could not start the refresh: {error}"])
            return True
        threading.Thread(target=self._follow, args=(process,), daemon=True).start()
        return True

    def _follow(self, process: subprocess.Popen[str]) -> None:
        """Collect the child's output line by line, then record how it ended."""
        for line in process.stdout or ():
            text = line.rstrip("\n")
            with self._lock:
                self._lines.append(text)
                if text.startswith(SUMMARY_PREFIX):
                    self._summary = text
        self._finish(process.wait(), [])

    def _finish(self, exit_code: int, lines: list[str]) -> None:
        """Record the exit code, any final lines, and the finished state."""
        with self._lock:
            self._lines.extend(lines)
            self._exit_code = exit_code
            self._state = "succeeded" if exit_code == 0 else "failed"
            self._finished_at = _now()


__all__ = ["DEFAULT_MAX_LINES", "RefreshRunner", "RefreshState", "RefreshStatus"]
