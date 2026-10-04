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
    assert "nfl-sos" in scripts
    assert "nfl-sos-pipeline" in scripts
    assert "nfl-sos-ui-api" not in scripts
    assert "nfl-sos-viz" not in scripts
    assert any(dependency.startswith("numpy") for dependency in dependencies)
    assert all(not dependency.startswith("matplotlib") for dependency in dependencies)
