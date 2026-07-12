"""Tests for nfl_sos_ratings.pipeline."""

import io
from types import SimpleNamespace

import pytest

from nfl_sos_ratings import pipeline


def test_pipeline_runs_data_then_visualizations_and_continues_on_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the multi-season pipeline preserves phase order and logs failures."""
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(pipeline, "START_YEAR", 2024)
    monkeypatch.setattr(pipeline, "END_YEAR", 2025)

    def fake_run_season(season: int) -> None:
        calls.append(("data", season))
        if season == 2024:
            raise RuntimeError("boom")

    def fake_visualize(season: int) -> None:
        calls.append(("viz", season))
        if season == 2025:
            raise RuntimeError("plot boom")

    monkeypatch.setattr(pipeline, "run_season", fake_run_season)
    monkeypatch.setattr(pipeline.visualize, "main", fake_visualize)

    pipeline.main()

    assert calls == [
        ("data", 2024),
        ("data", 2025),
        ("viz", 2024),
        ("viz", 2025),
    ]
    output = capsys.readouterr().out
    assert "Phase 1 of 2: Data gathering" in output
    assert "ERROR: season 2024 data step failed — boom" in output
    assert "Phase 2 of 2: Visualizations" in output
    assert "ERROR: season 2025 visualization failed — plot boom" in output
    assert "Pipeline complete — 2 seasons processed." in output


def test_pipeline_main_handles_windows_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the pipeline executes the Windows UTF-8 stdout branch."""
    monkeypatch.setattr(pipeline, "START_YEAR", 2025)
    monkeypatch.setattr(pipeline, "END_YEAR", 2025)
    monkeypatch.setattr(pipeline, "run_season", lambda season: None)
    monkeypatch.setattr(pipeline.visualize, "main", lambda season: None)
    monkeypatch.setattr(pipeline.sys, "platform", "win32")
    monkeypatch.setattr(pipeline.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(pipeline.io, "TextIOWrapper", lambda buffer, encoding: io.StringIO())

    pipeline.main()
