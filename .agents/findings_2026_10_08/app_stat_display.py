"""Evidence for the app display fixes' open questions (roadmap, "Data notes"). Reads ``data/`` only.

Prints four checks to stdout:

- per season, the team-games where ``yards_per_defensive_snap_allowed`` (play-by-play yards over
  scrimmage snaps, mirrored from the opponent's offense) differs from the derived
  ``total_yards_allowed_per_defensive_snap`` (total yards allowed, from official team stats where
  available, over defensive snaps), and the same for the offense pair, with the largest gap;
- one team's longest pass and run (season maxima) beside its ``opp_`` columns, which average each
  opponent's per-game longest play, and beside the opponents' season maxima with head-to-head
  games left out;
- every team-game whose ``fourth_down_aggressiveness`` exceeds 1;
- how many columns in ``data/`` the registry marks as percentages, and their values outside 0 to 1
  (-1 to 1 for a margin, the difference of two shares).

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/app_stat_display.py \
        --team NE --season 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_sos_ratings.metrics import get_registry

DATA = Path("data")
YARDS_PER_SNAP_PAIRS = (
    ("yards_per_defensive_snap_allowed", "total_yards_allowed_per_defensive_snap"),
    ("yards_per_offensive_snap", "total_yards_per_offensive_snap"),
)
# Gaps below this are float noise, not a different value.
TOLERANCE = 1e-9


def _seasons() -> list[int]:
    """Return the seasons with team game logs in ``data/``."""
    return sorted(int(path.name[:4]) for path in DATA.glob("*_team_game_logs.parquet"))


def _game_logs(season: int) -> pl.DataFrame:
    """Return one season's team game logs."""
    return pl.read_parquet(DATA / f"{season}_team_game_logs.parquet")


def yards_per_snap_pairs() -> None:
    """Print, per season, how many team-games each yards-per-snap pair disagrees on."""
    seasons_with_gaps = 0
    for season in _seasons():
        logs = _game_logs(season)
        parts: list[str] = []
        differing = 0
        for left, right in YARDS_PER_SNAP_PAIRS:
            gap = (logs.get_column(left) - logs.get_column(right)).abs()
            count = int((gap > TOLERANCE).sum())
            differing = max(differing, count)
            parts.append(f"{left} vs {right}: {count} games, largest gap {gap.max():.3f}")
        seasons_with_gaps += differing > 0
        sys.stdout.write(f"{season}  {'; '.join(parts)}\n")
    sys.stdout.write(f"seasons with a gap: {seasons_with_gaps} of {len(_seasons())}\n")


def opponent_longest_plays(team: str, season: int) -> None:
    """Print a team's longest plays, its opp_ averages, and its opponents' season maxima."""
    combined = pl.read_parquet(DATA / f"{season}_combined.parquet").filter(pl.col("team") == team)
    logs = _game_logs(season)
    opponents = logs.filter(pl.col("team") == team).get_column("opponent_team").unique().to_list()
    other_games = logs.filter(pl.col("team").is_in(opponents) & (pl.col("opponent_team") != team))
    for column in ("longest_pass", "longest_rush"):
        season_maxima = other_games.group_by("team").agg(pl.col(column).max()).get_column(column)
        sys.stdout.write(
            f"{team} {season} {column}: {combined.get_column(column).item():.0f}; "
            f"opp_{column}: {combined.get_column(f'opp_{column}').item():.2f}; "
            f"mean of the opponents' season maxima: {season_maxima.mean():.2f}\n"
        )


def aggressiveness_above_one() -> None:
    """Print every team-game whose fourth-and-short go rate exceeds 1."""
    for season in _seasons():
        rows = _game_logs(season).filter(pl.col("fourth_down_aggressiveness") > 1)
        for game_id, team, value in rows.select(
            "game_id", "team", "fourth_down_aggressiveness"
        ).iter_rows():
            sys.stdout.write(f"fourth_down_aggressiveness above 1: {game_id} {team} {value}\n")


def percentage_ranges() -> None:
    """Print how many percentage columns ``data/`` holds and any value outside its range."""
    registry = get_registry()
    columns: set[str] = set()
    outside: dict[str, int] = {}
    for path in sorted(DATA.glob("*.parquet")):
        frame = pl.read_parquet(path)
        for column, dtype in frame.schema.items():
            resolved = registry.resolve_column(column)
            if resolved is None or not resolved.base.percent:
                continue
            columns.add(column)
            values = frame.get_column(column)
            if isinstance(dtype, pl.List):
                values = values.explode()
            values = values.drop_nulls().drop_nans()
            low = -1.0 if resolved.base.name.endswith("_margin") else 0.0
            count = int(((values < low) | (values > 1.0)).sum())
            if count:
                outside[column] = outside.get(column, 0) + count
    sys.stdout.write(
        f"percentage columns in data/: {len(columns)}; outside their range: {outside}\n"
    )


def main() -> None:
    """Run every check for the requested team-season."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="NE")
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    yards_per_snap_pairs()
    opponent_longest_plays(args.team, args.season)
    aggressiveness_above_one()
    percentage_ranges()


if __name__ == "__main__":
    main()
