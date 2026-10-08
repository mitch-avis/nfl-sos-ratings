"""List the games nflverse credits wholly to one team (roadmap, "Data notes").

For each season, prints the games whose raw play-by-play player team columns name one team, and
the games whose weekly team stats have a row for one team only (both teams' totals in it). Reads
nflverse through nflreadpy (cached), before the loaders repair anything.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/one_team_games.py 1999 2025
"""

from __future__ import annotations

import sys

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured
from nfl_sos_ratings.player_team_repair import PBP_PLAYER_TEAM_IDS, one_team_games


def main() -> None:
    """Print the one-team games for the seasons from the first argument to the second."""
    use_disk_cache_unless_configured()
    first, last = (int(argument) for argument in sys.argv[1:3])
    for season in range(first, last + 1):
        pbp = nfl.load_pbp([season]).filter(pl.col("season_type") == "REG")
        team_stats = nfl.load_team_stats([season], summary_level="week").filter(
            pl.col("season_type") == "REG"
        )
        play_games = one_team_games(pbp, list(PBP_PLAYER_TEAM_IDS))
        stat_games = (
            team_stats.group_by("game_id")
            .agg(pl.col("team").n_unique().alias("teams"))
            .filter(pl.col("teams") == 1)
            .get_column("game_id")
            .sort()
            .to_list()
        )
        if play_games or stat_games:
            sys.stdout.write(
                f"{season}: play-by-play {len(play_games)} {play_games}; "
                f"team stats {len(stat_games)} {stat_games}\n"
            )


if __name__ == "__main__":
    main()
