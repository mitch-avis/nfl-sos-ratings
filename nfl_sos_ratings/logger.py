"""The project logger: progress and problems on stderr through one handler, colored on terminals.

Every module logs through ``logging.getLogger(__name__)``, a child of the ``nfl_sos_ratings``
logger. :func:`configure_logging`, which the ``nfl-sos-ratings`` front door calls once, gives that
logger one stderr handler: INFO lines read as plain progress, DEBUG (with ``--verbose``), WARNING,
and ERROR lines name their level, and a terminal shows them in color. Output meant as data, such
as the season's ratings table, goes to stdout instead.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOGGER_NAME = "nfl_sos_ratings"
_HANDLER_NAME = "nfl_sos_ratings_stderr"
_RESET = "\x1b[0m"
# ANSI styles by level; INFO stays in the terminal's own color.
_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: "\x1b[2m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;31m",
}


class _ProgressFormatter(logging.Formatter):
    """Format INFO as the bare message and every other level with its name first."""

    def __init__(self, *, color: bool) -> None:
        """Remember whether to wrap non-INFO lines in their level's color."""
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        """Return the message (and any traceback), prefixed and colored by level."""
        message = super().format(record)
        if record.levelno == logging.INFO:
            return message
        message = f"{record.levelname}: {message}"
        style = _LEVEL_STYLES.get(record.levelno) if self._color else None
        return f"{style}{message}{_RESET}" if style else message


def configure_logging(*, verbose: bool = False, stream: TextIO | None = None) -> None:
    """Send the package's log records to ``stream`` (stderr by default) through one handler.

    Args:
        verbose: Show DEBUG records (each file written, for example) as well as INFO and above.
        stream: Where records go; color is used only when it is a terminal.

    Calling it again replaces the handler it added rather than adding a second one.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [handler for handler in logger.handlers if handler.get_name() == _HANDLER_NAME]:
        logger.removeHandler(handler)
    target = sys.stderr if stream is None else stream
    handler = logging.StreamHandler(target)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_ProgressFormatter(color=target.isatty()))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


__all__ = ["LOGGER_NAME", "configure_logging"]
