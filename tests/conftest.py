"""Shared pytest fixtures and import aliases for this test suite."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _alias_module(alias: str, target: str) -> None:
    module = importlib.import_module(target)
    sys.modules.setdefault(alias, module)


_alias_module("config", "nfl_sos_ratings.config")
_alias_module("team_stats", "nfl_sos_ratings.team_stats")
_alias_module("data_loader", "nfl_sos_ratings.data_loader")
_alias_module("opponent_stats", "nfl_sos_ratings.opponent_stats")
_alias_module("ratings", "nfl_sos_ratings.ratings")
_alias_module("qb_stats", "nfl_sos_ratings.qb_stats")
_alias_module("main", "nfl_sos_ratings.main")
