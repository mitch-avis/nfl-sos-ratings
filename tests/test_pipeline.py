"""Tests for nfl_sos_ratings.pipeline."""

import io
from types import SimpleNamespace

import pytest

from nfl_sos_ratings import pipeline


def test_pipeline_raises_on_failures_and_skips_visualization_for_failed_seasons(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify failed data seasons are summarized, skipped in Phase 2, and exit non-zero."""
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

    with pytest.raises(SystemExit) as excinfo:
        pipeline.main()

    assert calls == [
        ("data", 2024),
        ("data", 2025),
        ("viz", 2025),
    ]
    assert excinfo.value.code == 1
    data = capsys.readouterr().out
    assert "Phase 1 of 2: Data gathering" in data
    assert "ERROR: season 2024 data step failed — boom" in data
    assert "Phase 2 of 2: Visualizations" in data
    assert "Skipping visualization for season 2024 due to failed data step." in data
    assert "ERROR: season 2025 visualization failed — plot boom" in data
    assert "Data step failures: 2024" in data
    assert "Visualization failures: 2025" in data
    assert "Pipeline finished with failures." in data


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
