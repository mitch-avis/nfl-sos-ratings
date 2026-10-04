"""Tests for the nflverse download-cache default."""

from typing import TYPE_CHECKING

import pytest
from nflreadpy.config import CacheMode, get_config, update_config

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_cache_mode() -> Iterator[None]:
    """Put nflreadpy's process-wide cache mode back after each test."""
    original = get_config().cache_mode
    update_config(cache_mode=CacheMode.MEMORY)
    yield
    update_config(cache_mode=original)


def test_use_disk_cache_unless_configured_defaults_to_the_filesystem_cache() -> None:
    # Act
    use_disk_cache_unless_configured(environ={})

    # Assert
    assert get_config().cache_mode == CacheMode.FILESYSTEM


def test_use_disk_cache_unless_configured_respects_an_explicit_choice() -> None:
    # Act
    use_disk_cache_unless_configured(environ={"NFLREADPY_CACHE": "memory"})

    # Assert
    assert get_config().cache_mode == CacheMode.MEMORY
