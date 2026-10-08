"""Cross-check each team's fumbles lost in punt-muff games against ESPN box scores (read-only).

For the first games per season (in game-id order) with a punt on which the receiving team lost a
fumble, prints each team's fumbles lost as ESPN's box score reports it, as nflverse's fumbling
team (``fumbled_1_team``) gives it, and as the team listed with the ball (``posteam``) would, then
how many team-games each count matches. Downloads nflverse play-by-play and schedules through
nflreadpy (cached) and one ESPN game summary per game (the schedule's ``espn`` id).

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/espn_fumbles_lost.py \
        2025,2024,2019,2015,2010,2005,2002,1999 12
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured

ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
# ESPN codes that differ from nflverse's normalized ones.
ESPN_ALIASES = {"WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
PAUSE_SECONDS = 0.6


def espn_fumbles_lost(espn_id: str) -> dict[str, int | None]:
    """Return each team's fumbles lost from ESPN's box score, keyed by nflverse team code."""
    with urllib.request.urlopen(ESPN_SUMMARY.format(espn_id), timeout=30) as response:  # noqa: S310 - fixed https URL
        summary = json.load(response)
    counts: dict[str, int | None] = {}
    for team in summary["boxscore"]["teams"]:
        code = team["team"]["abbreviation"]
        stats = {stat["name"]: stat.get("displayValue") for stat in team["statistics"]}
        value = stats.get("fumblesLost")
        counts[ESPN_ALIASES.get(code, code)] = int(value) if value not in {None, ""} else None
    return counts


def season_rows(season: int, games_per_season: int) -> list[dict[str, object]]:
    """Return the per-team comparison rows for one season's sampled games."""
    pbp = nfl.load_pbp([season]).filter(pl.col("season_type") == "REG")
    espn_ids = dict(nfl.load_schedules([season]).select("game_id", "espn").iter_rows())
    games = (
        pbp.filter(
            (pl.col("punt_attempt") == 1)
            & (pl.col("fumble_lost") == 1)
            & (pl.col("fumbled_1_team") == pl.col("defteam"))
        )
        .get_column("game_id")
        .unique()
        .sort()
        .head(games_per_season)
        .to_list()
    )
    rows: list[dict[str, object]] = []
    for game in games:
        plays = pbp.filter(pl.col("game_id") == game)
        lost = plays.filter(pl.col("fumble_lost") == 1)
        espn = espn_fumbles_lost(str(espn_ids[game]))
        rows.extend(
            {
                "game_id": game,
                "team": team,
                "espn": espn.get(team),
                "fumbling_team": lost.filter(pl.col("fumbled_1_team") == team).height,
                "posteam": lost.filter(pl.col("posteam") == team).height,
            }
            for team in (plays.get_column("away_team")[0], plays.get_column("home_team")[0])
        )
        time.sleep(PAUSE_SECONDS)
    return rows


def main() -> None:
    """Print the comparison: seasons in the first argument, games per season in the second."""
    use_disk_cache_unless_configured()
    seasons = [int(season) for season in sys.argv[1].split(",")]
    games_per_season = int(sys.argv[2])
    table = pl.DataFrame(
        [row for season in seasons for row in season_rows(season, games_per_season)],
        schema_overrides={"espn": pl.Int64},
    )
    reported = table.filter(pl.col("espn").is_not_null())
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f"{table}\n")
    sys.stdout.write(
        f"{reported.height} team-games with an ESPN count ({table.height - reported.height} "
        "without); matching the fumbling team: "
        f"{(reported['espn'] == reported['fumbling_team']).sum()}; matching posteam: "
        f"{(reported['espn'] == reported['posteam']).sum()}\n"
    )


if __name__ == "__main__":
    main()
