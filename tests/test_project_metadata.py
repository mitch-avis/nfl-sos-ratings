"""Tests for project metadata invariants."""

import tomllib
from pathlib import Path
from typing import Any


def _load_pyproject() -> dict[str, Any]:
    """Return the parsed root pyproject file."""
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_pyproject_matches_current_cli_and_runtime_surface() -> None:
    """The package metadata should not reference the removed visualization surface."""
    # Act
    pyproject = _load_pyproject()

    # Assert
    project = pyproject["project"]
    scripts = project["scripts"]
    dependencies = project["dependencies"]

    assert scripts["nfl-sos-ratings"] == "nfl_sos_ratings.cli:main"
    assert scripts["nfl-sos"] == "nfl_sos_ratings.cli:season_shortcut"
    assert scripts["nfl-sos-pipeline"] == "nfl_sos_ratings.cli:pipeline_shortcut"
    assert "nfl-sos-ui-api" not in scripts
    assert "nfl-sos-viz" not in scripts
    assert any(dependency.startswith("numpy") for dependency in dependencies)
    assert all(not dependency.startswith("matplotlib") for dependency in dependencies)


def test_pyright_execution_environments_share_the_missing_stub_policy() -> None:
    """Pyright should treat known untyped third-party packages the same in tests and runtime."""
    # Act
    pyproject = _load_pyproject()

    # Assert
    execution_environments = {
        environment["root"]: environment
        for environment in pyproject["tool"]["pyright"]["executionEnvironments"]
    }

    assert execution_environments["."]["reportMissingTypeStubs"] is False
    assert execution_environments["tests"]["reportMissingTypeStubs"] is False


def test_pytest_default_addopts_keep_published_data_opt_in() -> None:
    """The default pytest run should keep generated-data checks opt-in for local and CI runs."""
    # Act
    pyproject = _load_pyproject()

    # Assert
    addopts = pyproject["tool"]["pytest"]["addopts"]

    marker_option_index = addopts.index("-m")
    assert addopts[marker_option_index + 1] == "not published_data"
