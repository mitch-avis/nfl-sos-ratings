"""Tests for the nfl-sos-ratings front-door command."""

from typing import TYPE_CHECKING

import pytest

from nfl_sos_ratings import cli, composite_weights, main, pipeline, ui_api
from nfl_sos_ratings.config import SEASON
from nfl_sos_ratings.validation import qsos_audit, walk_forward

if TYPE_CHECKING:
    from pathlib import Path


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command.name in output
        assert command.summary in output


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["not-a-command"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize("command", [command.name for command in cli.COMMANDS])
def test_every_command_prints_its_own_help_without_running(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])

    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_season_runs_the_configured_season_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    cli.main(["season"])

    assert seasons == [SEASON]


def test_season_accepts_an_explicit_season(monkeypatch: pytest.MonkeyPatch) -> None:
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    cli.main(["season", "--season", "2019"])

    assert seasons == [2019]


def test_pipeline_runs_the_multi_season_build(monkeypatch: pytest.MonkeyPatch) -> None:
    seasons: list[int] = []
    companions: list[list[int]] = []
    monkeypatch.setattr(pipeline, "START_YEAR", 2023)
    monkeypatch.setattr(pipeline, "END_YEAR", 2024)
    monkeypatch.setattr(pipeline, "run_season", seasons.append)

    def fake_companions(data_dir: Path, all_seasons: list[int]) -> None:
        companions.append(all_seasons)

    monkeypatch.setattr(pipeline, "apply_alltime_rating_companions", fake_companions)

    cli.main(["pipeline"])

    assert seasons == [2023, 2024]
    assert companions == [[2023, 2024]]


@pytest.mark.parametrize(
    ("command", "module"),
    [("validate", walk_forward), ("qsos-audit", qsos_audit), ("web", ui_api)],
)
def test_options_pass_through_to_the_command(
    command: str, module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> None:
        received.append(argv)

    monkeypatch.setattr(module, "main", fake_main)

    cli.main([command, "--data-dir", "elsewhere"])

    assert received == [["--data-dir", "elsewhere"]]


def test_weights_prints_the_composite_report(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(composite_weights, "print_weight_report", lambda: calls.append("report"))

    cli.main(["weights"])

    assert calls == ["report"]
