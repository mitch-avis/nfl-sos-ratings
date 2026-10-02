"""The ``nfl-sos-ratings`` front door: one command name, then that command's own options."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

PROG = "nfl-sos-ratings"


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
        "Build every configured season, then the all-time companion columns.",
        "nfl_sos_ratings.pipeline",
    ),
    Command(
        "validate",
        "validation",
        "Run the walk-forward validation and rewrite docs/validation-report.md.",
        "nfl_sos_ratings.validation.walk_forward",
    ),
    Command(
        "weights",
        "validation",
        "Print the composite-weight fit and held-out diagnostics.",
        "nfl_sos_ratings.composite_weights",
    ),
    Command(
        "qsos-audit",
        "validation",
        "Print (or write) the QB schedule-strength audit.",
        "nfl_sos_ratings.validation.qsos_audit",
    ),
    Command(
        "web",
        "web",
        "Serve the analyst web app (built into web/dist) and its API.",
        "nfl_sos_ratings.ui_api",
    ),
)
COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
GROUP_TITLES = {"data": "data", "validation": "validation", "web": "web"}


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


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the named command's ``main`` with the remaining arguments."""
    args = _build_parser().parse_args(argv)
    module = importlib.import_module(COMMANDS_BY_NAME[args.command].module)
    command_main = cast("Callable[[list[str]], None]", module.main)
    command_main(list(args.args))


if __name__ == "__main__":
    main()
