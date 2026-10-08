"""Tests for the shared pytest setup: Polars threads and the published-data coverage floor."""

from argparse import Namespace
from types import SimpleNamespace
from typing import cast

import pytest

from tests.conftest import limit_polars_threads, only_published_data, pytest_collection_modifyitems


class _FakeItem:
    """A collected test that carries a ``published_data`` marker or not."""

    def __init__(self, *, published_data: bool) -> None:
        """Remember whether the test reads the generated data files."""
        self._published_data = published_data

    def get_closest_marker(self, name: str) -> object | None:
        """Return a marker for ``published_data`` when the test carries it."""
        return object() if name == "published_data" and self._published_data else None


def _config(plugin: object | None) -> pytest.Config:
    """Return a stand-in pytest config whose plugin manager knows ``plugin`` as the coverage one."""

    def getplugin(name: str) -> object | None:
        return plugin if name == "_cov" else None

    return cast(
        "pytest.Config", SimpleNamespace(pluginmanager=SimpleNamespace(getplugin=getplugin))
    )


def _coverage_plugin() -> SimpleNamespace:
    """Return a stand-in pytest-cov plugin with the floor and report the repo configures."""
    return SimpleNamespace(
        options=Namespace(cov_fail_under=90.0, cov_report={"term-missing": None})
    )


def test_limit_polars_threads_defaults_to_one_thread() -> None:
    # Arrange
    environ: dict[str, str] = {}

    # Act
    limit_polars_threads(environ)

    # Assert
    assert environ == {"POLARS_MAX_THREADS": "1"}


def test_limit_polars_threads_keeps_an_explicit_setting() -> None:
    # Arrange
    environ = {"POLARS_MAX_THREADS": "8"}

    # Act
    limit_polars_threads(environ)

    # Assert
    assert environ == {"POLARS_MAX_THREADS": "8"}


@pytest.mark.parametrize(
    ("published", "expected"),
    [((True, True), True), ((True, False), False), ((), False)],
)
def test_only_published_data_needs_every_selected_test_to_read_data(
    published: tuple[bool, ...], *, expected: bool
) -> None:
    # Arrange
    items = [_FakeItem(published_data=flag) for flag in published]

    # Act
    result = only_published_data(items)

    # Assert
    assert result is expected


def test_collection_hook_lifts_the_coverage_floor_for_published_data_only() -> None:
    # Arrange
    plugin = _coverage_plugin()
    items = cast("list[pytest.Item]", [_FakeItem(published_data=True)])

    # Act
    pytest_collection_modifyitems(_config(plugin), items)

    # Assert
    assert plugin.options.cov_fail_under == 0
    assert plugin.options.cov_report == {}


def test_collection_hook_keeps_the_floor_when_code_tests_run() -> None:
    # Arrange
    plugin = _coverage_plugin()
    items = cast(
        "list[pytest.Item]", [_FakeItem(published_data=True), _FakeItem(published_data=False)]
    )

    # Act
    pytest_collection_modifyitems(_config(plugin), items)

    # Assert
    assert plugin.options.cov_fail_under == 90.0


def test_collection_hook_does_nothing_without_coverage() -> None:
    # Arrange
    items = cast("list[pytest.Item]", [_FakeItem(published_data=True)])

    # Act
    result = pytest_collection_modifyitems(_config(None), items)

    # Assert
    assert result is None
