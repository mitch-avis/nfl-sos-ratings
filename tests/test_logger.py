"""Tests for the project logger: one stderr handler, plain progress lines, color on terminals."""

import io
import logging
from typing import TYPE_CHECKING

import pytest

from nfl_sos_ratings.logger import LOGGER_NAME, configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Terminal(io.StringIO):
    """A text stream that reports itself as a terminal."""

    def isatty(self) -> bool:
        """Claim to be a terminal, as an interactive stderr would."""
        return True


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    """Put the package logger's handlers and level back after each test."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = logger.handlers[:], logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


def _module_logger() -> logging.Logger:
    """Return a logger named like one of the package's modules."""
    return logging.getLogger(f"{LOGGER_NAME}.main")


def test_info_lines_read_like_plain_progress() -> None:
    # Arrange
    stream = io.StringIO()
    configure_logging(stream=stream)

    # Act
    _module_logger().info("Fitting %s ratings...", "team")

    # Assert
    assert stream.getvalue() == "Fitting team ratings...\n"


def test_warnings_and_errors_name_their_level() -> None:
    # Arrange
    stream = io.StringIO()
    configure_logging(stream=stream)

    # Act
    _module_logger().error("season %s data step failed: %s", 2024, "boom")

    # Assert
    assert stream.getvalue() == "ERROR: season 2024 data step failed: boom\n"


def test_debug_lines_are_hidden_by_default() -> None:
    # Arrange
    stream = io.StringIO()
    configure_logging(stream=stream)

    # Act
    _module_logger().debug("Saved %s", "ratings")

    # Assert
    assert stream.getvalue() == ""


def test_debug_lines_appear_when_verbose() -> None:
    # Arrange
    stream = io.StringIO()
    configure_logging(verbose=True, stream=stream)

    # Act
    _module_logger().debug("Saved %s", "ratings")

    # Assert
    assert stream.getvalue() == "DEBUG: Saved ratings\n"


def test_configuring_twice_keeps_one_handler() -> None:
    # Arrange
    first, second = io.StringIO(), io.StringIO()
    configure_logging(stream=first)
    configure_logging(stream=second)

    # Act
    _module_logger().info("once")

    # Assert
    assert (first.getvalue(), second.getvalue()) == ("", "once\n")


def test_levels_are_colored_on_a_terminal() -> None:
    # Arrange
    terminal = _Terminal()
    configure_logging(stream=terminal)

    # Act
    _module_logger().warning("careful")

    # Assert
    assert terminal.getvalue().startswith("\x1b[")
    assert "WARNING: careful" in terminal.getvalue()


def test_plain_streams_get_no_color() -> None:
    # Arrange
    stream = io.StringIO()
    configure_logging(stream=stream)

    # Act
    _module_logger().warning("careful")

    # Assert
    assert "\x1b[" not in stream.getvalue()
