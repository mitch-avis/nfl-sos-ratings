"""The ``nfl-sos-ratings`` front door: one command name, then that command's own options."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

PROG = "nfl-sos-ratings"

# The rating fits are many small linear solves, where BLAS worker threads cost far more CPU than
# they save in wall time, so every command runs single-threaded BLAS unless the caller set these.
# BLAS reads them when NumPy first loads, which happens only once a command module is imported.
BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class Command:
    """One front-door command and the module whose ``main(argv)`` runs it."""

    name: str
    group: str
    summary: str
    module: str


COMMANDS: tuple[Command, ...] = (
    Command(
        "season",
        "data",
        "Build one season's Parquet outputs in data/.",
        "nfl_sos_ratings.main",
    ),
    Command(
        "pipeline",
        "data",
        "Build every configured season into data/.",
        "nfl_sos_ratings.pipeline",
    ),
    Command(
        "schedules",
        "data",
        "Rank every team-season in data/ by strength of schedule.",
        "nfl_sos_ratings.schedules",
    ),
    Command(
        "validate",
        "validation",
        "Run the walk-forward validation and rewrite docs/validation-report.md.",
        "nfl_sos_ratings.validation.walk_forward",
    ),
    Command(
        "check-additivity",
        "validation",
        "Test whether strong units beat the additive prediction against weak ones (read-only).",
        "nfl_sos_ratings.validation.additivity",
    ),
    Command(
        "check-passer",
        "validation",
        "Score a passer's later games against his rating season's QB model (read-only).",
        "nfl_sos_ratings.validation.passer_holdout",
    ),
    Command(
        "check-in-season-penalty",
        "validation",
        "Compare previous-season penalties with per-fit cross-validation (read-only).",
        "nfl_sos_ratings.validation.in_season_penalty",
    ),
    Command(
        "catalog",
        "docs",
        "Regenerate the stats catalogs in docs/ from the metric registry.",
        "nfl_sos_ratings.metrics.catalog_docs",
    ),
    Command(
        "web",
        "web",
        "Serve the analyst web app (built into web/dist) and its API.",
        "nfl_sos_ratings.ui_api",
    ),
)
COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
GROUP_TITLES = {"data": "data", "validation": "validation", "docs": "docs", "web": "web"}


def _command_list() -> str:
    """Return the grouped command table shown by ``nfl-sos-ratings --help``."""
    width = max(len(command.name) for command in COMMANDS)
    blocks: list[str] = []
    for group, title in GROUP_TITLES.items():
        rows = [
            f"  {command.name:<{width}}  {command.summary}"
            for command in COMMANDS
            if command.group == group
        ]
        blocks.append(f"{title}:\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _build_parser() -> argparse.ArgumentParser:
    """Build the front-door parser: a command name, then that command's own options."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Schedule-strength-adjusted NFL team and quarterback ratings.",
        epilog=_command_list() + f"\n\nRun '{PROG} <command> --help' for a command's options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=sorted(COMMANDS_BY_NAME), metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def limit_blas_threads() -> None:
    """Default every BLAS thread variable to one thread, keeping any value already set."""
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the named command's ``main`` with the remaining arguments."""
    limit_blas_threads()
    args = _build_parser().parse_args(argv)
    module = importlib.import_module(COMMANDS_BY_NAME[args.command].module)
    command_main = cast("Callable[[list[str]], None]", module.main)
    command_main(list(args.args))


def season_shortcut() -> None:
    """Run the ``nfl-sos`` shortcut as ``nfl-sos-ratings season`` with its arguments."""
    main(["season", *sys.argv[1:]])


def pipeline_shortcut() -> None:
    """Run the ``nfl-sos-pipeline`` shortcut as ``nfl-sos-ratings pipeline`` with its arguments."""
    main(["pipeline", *sys.argv[1:]])


if __name__ == "__main__":
    main()
