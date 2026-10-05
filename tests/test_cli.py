"""Tests for the nfl-sos-ratings front-door command."""

import os
import sys

import pytest

from nfl_sos_ratings import cli, main, pipeline, ui_api
from nfl_sos_ratings.config import SEASON
from nfl_sos_ratings.validation import walk_forward


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    # Act & Assert
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    # Assert
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command.name in output
        assert command.summary in output


def test_unknown_command_exits_with_usage_error() -> None:
    # Act & Assert
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["not-a-command"])

    # Assert
    assert exit_info.value.code == 2


@pytest.mark.parametrize("command", [command.name for command in cli.COMMANDS])
def test_every_command_prints_its_own_help_without_running(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act & Assert
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])

    # Assert
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(("argv", "expected"), [(["season"], [False]), (["-v", "season"], [True])])
def test_front_door_configures_logging_before_the_command(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: list[bool]
) -> None:
    # Arrange
    configured: list[bool] = []
    seasons: list[int] = []

    def record(*, verbose: bool = False, stream: object = None) -> None:
        """Remember the verbosity the front door asked for."""
        del stream
        configured.append(verbose)

    monkeypatch.setattr(cli, "configure_logging", record)
    monkeypatch.setattr(main, "run_season", seasons.append)

    # Act
    cli.main(argv)

    # Assert
    assert configured == expected


def test_season_runs_the_configured_season_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    # Act
    cli.main(["season"])

    # Assert
    assert seasons == [SEASON]


def test_season_accepts_an_explicit_season(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    # Act
    cli.main(["season", "--season", "2019"])

    # Assert
    assert seasons == [2019]


def test_pipeline_runs_the_multi_season_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seasons: list[int] = []
    monkeypatch.setattr(pipeline, "START_YEAR", 2023)
    monkeypatch.setattr(pipeline, "END_YEAR", 2024)
    monkeypatch.setattr(pipeline, "run_season", seasons.append)

    # Act
    cli.main(["pipeline"])

    # Assert
    assert seasons == [2023, 2024]


@pytest.mark.parametrize(
    ("command", "module"),
    [("validate", walk_forward), ("web", ui_api)],
)
def test_options_pass_through_to_the_command(
    command: str, module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    received: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> None:
        received.append(argv)

    monkeypatch.setattr(module, "main", fake_main)

    # Act
    cli.main([command, "--data-dir", "elsewhere"])

    # Assert
    assert received == [["--data-dir", "elsewhere"]]


def _unset_blas_thread_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the BLAS thread variables for one test, restoring them afterwards.

    ``setenv`` first records each variable's original state, so the undo also removes values the
    code under test sets on ``os.environ`` directly.
    """
    for name in cli.BLAS_THREAD_VARIABLES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_main_limits_blas_to_one_thread_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _unset_blas_thread_variables(monkeypatch)
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    # Act
    cli.main(["season"])

    # Assert
    assert {name: os.environ.get(name) for name in cli.BLAS_THREAD_VARIABLES} == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_main_keeps_an_explicit_blas_thread_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _unset_blas_thread_variables(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)

    # Act
    cli.main(["season"])

    # Assert
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_season_shortcut_runs_season_with_one_blas_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _unset_blas_thread_variables(monkeypatch)
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)
    monkeypatch.setattr(sys, "argv", ["nfl-sos", "--season", "2019"])

    # Act
    cli.season_shortcut()

    # Assert
    assert seasons == [2019]
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"


def test_pipeline_shortcut_runs_pipeline_with_one_blas_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _unset_blas_thread_variables(monkeypatch)
    seasons: list[int] = []
    monkeypatch.setattr(pipeline, "START_YEAR", 2023)
    monkeypatch.setattr(pipeline, "END_YEAR", 2023)
    monkeypatch.setattr(pipeline, "run_season", seasons.append)
    monkeypatch.setattr(sys, "argv", ["nfl-sos-pipeline"])

    # Act
    cli.pipeline_shortcut()

    # Assert
    assert seasons == [2023]
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
