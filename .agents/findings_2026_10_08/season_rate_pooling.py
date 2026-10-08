"""How much pooling season rates moves them, for one season (roadmap, "Data notes").

Season rows used to average each rate's game values; they now divide the season's summed
numerator by its summed denominator (``nfl_sos_ratings.pooled_rates``). This loads one season
through the package loaders (nflverse through nflreadpy, cached), builds the rows both ways from
the same game rows, and prints, for every column that changed, its largest absolute change, the
team or passer it occurs for, and the value before and after. Sections: team season rows, the
team opponent profiles (each opponent's rate without its games against the team, averaged over
opponents), and QB season rows (all passers, then the qualified ones). Columns are ordered by
their largest change in league standard deviations (the spread of the column across the season's
rows after pooling), so rates on different scales compare. Writes nothing.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/season_rate_pooling.py \
        --season 2025
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from nfl_sos_ratings.data_loader import (
    load_qb_stats,
    load_schedule,
    load_weekly_team_stats,
    use_disk_cache_unless_configured,
)
from nfl_sos_ratings.main import played_schedule
from nfl_sos_ratings.opponent_stats import compute_all_opponent_profiles, get_opponents
from nfl_sos_ratings.pooled_rates import drop_rate_parts
from nfl_sos_ratings.qb_stats import compute_qb_season_stats
from nfl_sos_ratings.team_stats import compute_all_teams_per_game

# Changes below this are float rounding, not a different value.
TOLERANCE = 1e-9
IDENTIFIERS = {"season", "week", "season_type", "games"}
# The QB season rates the old season rows averaged over games.
AVERAGED_QB_RATES = ["qb_passer_rating", "qb_completion_percentage_above_expectation"]


def _mean_of_games(
    games: pl.DataFrame, by: list[str], *, longest_as_max: bool = True
) -> pl.DataFrame:
    """Return the old aggregation: every stat's mean over the games.

    A season row kept the longest plays' maximum; an opponent row averaged them too.
    """
    stats = [
        column
        for column, dtype in games.schema.items()
        if dtype.is_numeric() and column not in IDENTIFIERS
    ]
    return games.group_by(by).agg(
        pl.col(column).max()
        if longest_as_max and column.startswith("longest_")
        else pl.col(column).mean()
        for column in stats
    )


def _old_opponent_profiles(games: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Return the old opponent profiles: each opponent's per-game means without the team."""
    published = drop_rate_parts(games)
    rows: list[pl.DataFrame] = []
    for team in sorted(published.get_column("team").unique().to_list()):
        opponents = get_opponents(schedule, team)
        others = published.filter(
            pl.col("team").is_in(opponents) & (pl.col("opponent_team") != team)
        )
        per_opponent = _mean_of_games(others, ["team"], longest_as_max=False).drop("team")
        rows.append(per_opponent.select(pl.lit(team).alias("team"), pl.all().mean()))
    return pl.concat(rows)


def _old_qb_season_rows(qb_games: pl.DataFrame, qb_season: pl.DataFrame) -> pl.DataFrame:
    """Return the old QB season rows: passer rating and CPOE as means of the game values.

    Every other QB season rate was already rebuilt from season totals.
    """
    averaged = (
        drop_rate_parts(qb_games)
        .group_by("qb_id")
        .agg(pl.col(column).mean() for column in AVERAGED_QB_RATES)
    )
    return qb_season.drop(AVERAGED_QB_RATES).join(averaged, on="qb_id", how="left")


def _profiles(profiles: pl.DataFrame | None) -> pl.DataFrame:
    """Return the opponent profiles, or an empty frame keyed by team when there are none."""
    return profiles if profiles is not None else pl.DataFrame(schema={"team": pl.String})


def changes(
    before: pl.DataFrame, after: pl.DataFrame, key: str, label: str | None = None
) -> pl.DataFrame:
    """Return each changed column's largest change, where it occurs, and its size in SDs.

    ``label`` names a column of ``after`` to print beside ``key`` (a passer's name).
    """
    joined = before.join(after, on=key, suffix="__after")
    rows: list[dict[str, object]] = []
    for column in after.columns:
        if column in {key, label} or column not in before.columns:
            continue
        if not after.schema[column].is_numeric():
            continue
        delta = (pl.col(f"{column}__after") - pl.col(column)).abs()
        changed = joined.with_columns(delta.alias("delta")).filter(pl.col("delta") > TOLERANCE)
        if changed.is_empty():
            continue
        worst = changed.sort("delta", descending=True).row(0, named=True)
        spread = after.get_column(column).std()
        rows.append(
            {
                "column": column,
                "rows_changed": changed.height,
                key: worst[key],
                **({label: worst[f"{label}__after"]} if label else {}),
                "before": worst[column],
                "after": worst[f"{column}__after"],
                "largest_change": worst["delta"],
                "change_in_sd": worst["delta"] / spread if spread else None,
            }
        )
    return pl.DataFrame(rows).sort("change_in_sd", descending=True, nulls_last=True)


def main() -> None:
    """Print the three sections for the season given by ``--season``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--season", type=int, required=True)
    season = parser.parse_args().season
    use_disk_cache_unless_configured()
    games = load_weekly_team_stats(season)
    schedule = played_schedule(load_schedule(season))
    qb_games = load_qb_stats(season)
    qb_season = compute_qb_season_stats(qb_games, weekly_df=games)

    sections = {
        "Team season rows": changes(
            _mean_of_games(drop_rate_parts(games), ["team"]),
            compute_all_teams_per_game(games),
            "team",
        ),
        "Team opponent profiles": changes(
            _old_opponent_profiles(games, schedule),
            _profiles(compute_all_opponent_profiles(games, schedule)[0]),
            "team",
        ),
        "QB season rows": changes(
            _old_qb_season_rows(qb_games, qb_season), qb_season, "qb_id", "qb_name"
        ),
        "QB season rows, qualified passers": changes(
            _old_qb_season_rows(qb_games, qb_season).filter(pl.col("qb_is_eligible")),
            qb_season.filter(pl.col("qb_is_eligible")),
            "qb_id",
            "qb_name",
        ),
    }
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        for title, table in sections.items():
            sys.stdout.write(f"\n{season} {title}: {table.height} columns changed\n{table}\n")


if __name__ == "__main__":
    main()
