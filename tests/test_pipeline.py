"""Tests for nfl_sos_ratings.pipeline."""

import io
from types import SimpleNamespace

import pytest

from nfl_sos_ratings import pipeline
from tests.stubs import stub


def test_pipeline_raises_on_failures_and_exits_nonzero_for_failed_seasons(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify failed data seasons are summarized and exit non-zero."""
    # Arrange
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
    data = capsys.readouterr().out
    assert "Phase 1 of 1: Data gathering" in data
    assert "ERROR: season 2024 data step failed — boom" in data
    assert "Data step failures: 2024" in data
    assert "Pipeline finished with failures." in data


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
