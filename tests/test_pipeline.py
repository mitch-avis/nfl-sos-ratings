"""Tests for nfl_sos_ratings.pipeline."""

import io
import logging
from types import SimpleNamespace

import pytest
from tests.stubs import stub

from nfl_sos_ratings import pipeline


def test_pipeline_raises_on_failures_and_exits_nonzero_for_failed_seasons(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify failed data seasons are summarized and exit non-zero."""
    # Arrange
    caplog.set_level(logging.INFO, logger="nfl_sos_ratings")
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(pipeline, "START_YEAR", 2024)
    monkeypatch.setattr(pipeline, "END_YEAR", 2025)

    def fake_run_season(season: int) -> None:
        calls.append(("data", season))
        if season == 2024:
            msg = "boom"
            raise RuntimeError(msg)

    monkeypatch.setattr(pipeline, "run_season", fake_run_season)

    # Act & Assert
    with pytest.raises(SystemExit) as excinfo:
        pipeline.main([])

    # Assert
    assert calls == [
        ("data", 2024),
        ("data", 2025),
    ]
    assert excinfo.value.code == 1
    messages = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert ("INFO", "Phase 1 of 1: Data gathering") in messages
    assert ("ERROR", "season 2024 data step failed") in messages
    errors: list[BaseException] = []
    for record in caplog.records:
        exc_info = record.exc_info
        if not isinstance(exc_info, tuple):
            continue
        error = exc_info[1]
        if error is not None:
            errors.append(error)
    assert [str(error) for error in errors] == ["boom"]
    assert ("ERROR", "Data step failures: 2024") in messages
    assert ("ERROR", "Pipeline finished with failures.") in messages


def test_pipeline_main_handles_windows_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the pipeline executes the Windows UTF-8 stdout branch."""
    # Arrange
    monkeypatch.setattr(pipeline, "START_YEAR", 2025)
    monkeypatch.setattr(pipeline, "END_YEAR", 2025)
    monkeypatch.setattr(pipeline, "run_season", stub(lambda: None))
    monkeypatch.setattr(pipeline.sys, "platform", "win32")
    monkeypatch.setattr(pipeline.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(pipeline.io, "TextIOWrapper", stub(io.StringIO))

    # Act
    pipeline.main([])

    # Assert
    assert isinstance(pipeline.sys.stdout, io.StringIO)


def test_pipeline_runs_every_configured_season_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The multi-season pipeline runs each configured season once, in order."""
    # Arrange
    calls: list[int] = []
    monkeypatch.setattr(pipeline, "START_YEAR", 2024)
    monkeypatch.setattr(pipeline, "END_YEAR", 2025)
    monkeypatch.setattr(pipeline, "run_season", calls.append)

    # Act
    pipeline.main([])

    # Assert
    assert calls == [2024, 2025]


class Record:
    """A small generic data container with useful serialization helpers."""

    def __init__(self, **fields: object) -> None:
        self._values = dict(fields)

    @property
    def key(self) -> str:
        if "id" in self._values:
            return str(self._values["id"])
        if "key" in self._values:
            return str(self._values["key"])
        return ""

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Record:
        return cls(**values)

    def as_mapping(self) -> dict[str, object]:
        return dict(self._values)

    def __iter__(self):
        return iter(self.as_mapping().items())

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, item: str) -> object:
        return self._values[item]

    def __setitem__(self, item: str, value: object) -> None:
        self._values[item] = value

    def __contains__(self, item: object) -> bool:
        return item in self._values

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.as_mapping()!r})"


def test_record_round_trip_mapping() -> None:
    """Generic record data can be serialized and reconstructed from a mapping."""
    # Arrange
    item = Record(id=7, team="NE")

    # Act
    rebuilt_item = Record.from_mapping({"id": 7, "team": "NE"})

    # Assert
    assert item.key == "7"
    assert item.as_mapping() == {"id": 7, "team": "NE"}
    assert rebuilt_item.as_mapping() == item.as_mapping()
