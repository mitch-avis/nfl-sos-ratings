"""Points-based team ratings and head-to-head-excluded schedule strength.

A team's rating answers one question: how many points per game better than an average team was it,
after adjusting for every opponent it actually faced? It is built from expected points added
(EPA), which is already measured in points, so the pieces add up without any weighting:

- ``offense_rating``: scrimmage EPA per play the offense produced above an average offense,
  adjusted for the defenses it faced, times the league-average scrimmage plays per team-game.
- ``defense_rating``: scrimmage EPA per play the defense prevented relative to an average defense,
  adjusted for the offenses it faced, on the same per-game scale.
- ``special_teams_rating``: the same two-sided adjustment over special-teams plays (kicks, punts,
  returns, field goals, extra points), counting both the team's possession units and its coverage
  units, times the league-average special-teams plays per team-game.
- ``team_rating``: the sum of the three, in points per game on a neutral field.

Each piece comes from one simultaneous ridge fit (``nfl_sos_ratings.ridge``), so opponents are
judged against their own opponents, and so on through the whole schedule.

``sos`` is the average ``team_rating`` of the opponents a team played, one entry per game. Each
opponent is rated from a refit that leaves out every game involving the team being evaluated, so a
team beating up on an opponent cannot make that opponent look weaker in its own schedule strength.

The rating history (``fit_team_ratings_by_week``) refits the ratings on the games through each
week with the season's penalties, so each week's row shows the rating as the evidence then
supported it.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_sos_ratings.ridge import UnitColumns, UnitFit, fit_unit_ridge

SCRIMMAGE_PLAYS_COLUMN = "offensive_snaps"
SCRIMMAGE_EPA_COLUMN = "offensive_epa"
SPECIAL_TEAMS_PLAYS_COLUMN = "st_plays"
SPECIAL_TEAMS_EPA_COLUMN = "st_epa"
TEAM_RATING_COLUMNS = ("offense_rating", "defense_rating", "special_teams_rating", "team_rating")

_KEY_COLUMNS = ("game_id", "team", "opponent_team", "is_home")
_RESPONSE = "epa_per_play"
_WEIGHT = "plays"
# The ridge columns for the rows `scrimmage_rows` returns (special-teams rows use the same names).
TEAM_UNIT_COLUMNS = UnitColumns(response=_RESPONSE, weight=_WEIGHT)


@dataclass(frozen=True, slots=True)
class TeamRatingFit:
    """Team ratings plus the fitted constants a head-to-head-excluded refit must reuse."""

    ratings: pl.DataFrame
    scrimmage_lambda: float
    special_teams_lambda: float
    scrimmage_plays_per_game: float
    special_teams_plays_per_game: float


def _require_columns(game_logs: pl.DataFrame) -> None:
    """Raise ``ValueError`` naming every input column the team rating needs but lacks."""
    required = {
        *_KEY_COLUMNS,
        SCRIMMAGE_PLAYS_COLUMN,
        SCRIMMAGE_EPA_COLUMN,
        SPECIAL_TEAMS_PLAYS_COLUMN,
        SPECIAL_TEAMS_EPA_COLUMN,
    }
    missing = sorted(required - set(game_logs.columns))
    if missing:
        msg = f"team game logs are missing team-rating columns: {', '.join(missing)}"
        raise ValueError(msg)


def _unit_rows(game_logs: pl.DataFrame, plays_column: str, epa_column: str) -> pl.DataFrame:
    """Return one offense-versus-defense row per team-game with EPA per play and its play count."""
    return game_logs.filter(pl.col(plays_column) > 0).select(
        *_KEY_COLUMNS,
        pl.col(plays_column).cast(pl.Float64).alias(_WEIGHT),
        (pl.col(epa_column).cast(pl.Float64) / pl.col(plays_column)).alias(_RESPONSE),
    )


def scrimmage_rows(game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return the rows the scrimmage fit uses, for analyses that refit or predict them.

    One row per team-game with plays: ``game_id``, ``team`` (the offense), ``opponent_team`` (the
    defense), ``is_home``, ``plays``, and ``epa_per_play``; fit them with ``TEAM_UNIT_COLUMNS``.

    Raises:
        ValueError: If a team-rating column is missing.

    """
    _require_columns(game_logs)
    return _unit_rows(game_logs, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN)


def _plays_per_game(rows: pl.DataFrame) -> float:
    """Return the league-average play count per team-game row."""
    return float(rows.get_column(_WEIGHT).sum()) / rows.height


def _ratings_frame(
    scrimmage: UnitFit,
    special_teams: UnitFit,
    scrimmage_plays_per_game: float,
    special_teams_plays_per_game: float,
) -> pl.DataFrame:
    """Convert per-play effects into the four per-game rating columns, one row per team."""
    teams = sorted(
        set(scrimmage.offense)
        | set(scrimmage.defense)
        | set(special_teams.offense)
        | set(special_teams.defense)
    )
    offense = [scrimmage.offense.get(team, 0.0) * scrimmage_plays_per_game for team in teams]
    defense = [scrimmage.defense.get(team, 0.0) * scrimmage_plays_per_game for team in teams]
    special = [
        (special_teams.offense.get(team, 0.0) + special_teams.defense.get(team, 0.0))
        * special_teams_plays_per_game
        for team in teams
    ]
    return pl.DataFrame(
        {
            "team": teams,
            "offense_rating": offense,
            "defense_rating": defense,
            "special_teams_rating": special,
            "team_rating": [o + d + s for o, d, s in zip(offense, defense, special, strict=True)],
        }
    )


def fit_team_ratings(
    game_logs: pl.DataFrame,
    *,
    scrimmage_lambda: float | None = None,
    special_teams_lambda: float | None = None,
) -> TeamRatingFit:
    """Fit points-based offense, defense, special-teams, and overall team ratings.

    Args:
        game_logs: One row per team-game (the team on offense), with ``game_id``, ``team``,
            ``opponent_team``, ``is_home``, scrimmage plays and EPA, and special-teams plays and EPA
            from the possession team's side.
        scrimmage_lambda: Fixed scrimmage ridge penalty; cross-validated when ``None``.
        special_teams_lambda: Fixed special-teams ridge penalty; cross-validated when ``None``.

    Returns:
        The ratings frame plus the penalties and per-game scales used to build it.

    Raises:
        ValueError: If a required column is missing or no rows have plays.

    """
    _require_columns(game_logs)
    scrimmage_rows = _unit_rows(game_logs, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN)
    special_rows = _unit_rows(game_logs, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN)
    scrimmage = fit_unit_ridge(scrimmage_rows, TEAM_UNIT_COLUMNS, ridge_lambda=scrimmage_lambda)
    special_teams = fit_unit_ridge(
        special_rows, TEAM_UNIT_COLUMNS, ridge_lambda=special_teams_lambda
    )
    scrimmage_plays_per_game = _plays_per_game(scrimmage_rows)
    special_teams_plays_per_game = _plays_per_game(special_rows)
    return TeamRatingFit(
        ratings=_ratings_frame(
            scrimmage, special_teams, scrimmage_plays_per_game, special_teams_plays_per_game
        ),
        scrimmage_lambda=scrimmage.ridge_lambda,
        special_teams_lambda=special_teams.ridge_lambda,
        scrimmage_plays_per_game=scrimmage_plays_per_game,
        special_teams_plays_per_game=special_teams_plays_per_game,
    )


def fit_team_ratings_by_week(game_logs: pl.DataFrame, fit: TeamRatingFit) -> pl.DataFrame:
    """Return each team's ratings as of every week, each fit on the games through that week.

    Every week reuses the season fit's penalties rather than cross-validating its own: after one
    or two weeks there are too few games for cross-validation to choose a penalty reliably. With
    the penalty held fixed, the pull toward average depends only on how many plays a team has, so
    early weeks sit close to average and spread out as games accumulate. The per-game scales are
    each week's own league averages, as a fit of those games alone would use. Head-to-head-excluded
    schedule strength is not refit week by week.

    Args:
        game_logs: The team-game rows passed to :func:`fit_team_ratings`, plus ``week``.
        fit: The season fit of ``game_logs`` whose penalties every week reuses.

    Returns:
        One row per week and team that has played by then, with ``week``, ``team``,
        ``games_played`` through that week, and the four rating columns. The last week's ratings
        are the season fit's.

    Raises:
        ValueError: If ``week`` or a team-rating column is missing.

    """
    _require_columns(game_logs)
    if "week" not in game_logs.columns:
        msg = "team game logs need a week column for the weekly rating history"
        raise ValueError(msg)
    weeks: list[int] = sorted(game_logs.get_column("week").unique().to_list())
    frames: list[pl.DataFrame] = []
    for week in weeks:
        through_week = game_logs.filter(pl.col("week") <= week)
        games_played = through_week.group_by("team").len("games_played")
        ratings = fit_team_ratings(
            through_week,
            scrimmage_lambda=fit.scrimmage_lambda,
            special_teams_lambda=fit.special_teams_lambda,
        ).ratings
        frames.append(
            ratings.join(games_played, on="team").select(
                pl.lit(week, dtype=pl.Int64).alias("week"),
                "team",
                pl.col("games_played").cast(pl.Int64),
                *TEAM_RATING_COLUMNS,
            )
        )
    return pl.concat(frames)


def _ratings_without(game_logs: pl.DataFrame, team: str, fit: TeamRatingFit) -> pl.DataFrame:
    """Rate every other team from a refit that drops all games involving ``team``.

    Returns an empty frame when no games remain, as can happen in the first weeks of a season.
    """
    others = game_logs.filter((pl.col("team") != team) & (pl.col("opponent_team") != team))
    if others.is_empty():
        return pl.DataFrame(schema={"team": pl.String, "team_rating": pl.Float64})
    scrimmage = fit_unit_ridge(
        _unit_rows(others, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN),
        TEAM_UNIT_COLUMNS,
        ridge_lambda=fit.scrimmage_lambda,
    )
    special_teams = fit_unit_ridge(
        _unit_rows(others, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN),
        TEAM_UNIT_COLUMNS,
        ridge_lambda=fit.special_teams_lambda,
    )
    return _ratings_frame(
        scrimmage,
        special_teams,
        fit.scrimmage_plays_per_game,
        fit.special_teams_plays_per_game,
    )


def compute_team_schedule_strength(game_logs: pl.DataFrame, fit: TeamRatingFit) -> pl.DataFrame:
    """Return each team's played-game mean opponent ``team_rating``, head-to-head excluded.

    Args:
        game_logs: The same team-game rows passed to :func:`fit_team_ratings`.
        fit: The full-season fit whose penalties and per-game scales the refits reuse.

    Returns:
        One row per team with ``team`` and ``sos`` in points per game. Early in a season an
        opponent may have played no one else yet; such games are skipped, and ``sos`` is null
        when no opponent can be rated.

    """
    _require_columns(game_logs)
    teams: list[str] = fit.ratings.get_column("team").to_list()
    values: list[float | None] = []
    for team in teams:
        opponent_ratings = _ratings_without(game_logs, team, fit).select(
            pl.col("team").alias("opponent_team"), pl.col("team_rating").alias("sos")
        )
        rated = (
            game_logs.filter(pl.col("team") == team)
            .select("opponent_team")
            .join(opponent_ratings, on="opponent_team", how="inner")
        )
        values.append(float(np.mean(rated.get_column("sos").to_numpy())) if rated.height else None)
    return pl.DataFrame({"team": teams, "sos": values})


__all__ = [
    "SCRIMMAGE_EPA_COLUMN",
    "SCRIMMAGE_PLAYS_COLUMN",
    "SPECIAL_TEAMS_EPA_COLUMN",
    "SPECIAL_TEAMS_PLAYS_COLUMN",
    "TEAM_RATING_COLUMNS",
    "TEAM_UNIT_COLUMNS",
    "TeamRatingFit",
    "compute_team_schedule_strength",
    "fit_team_ratings",
    "fit_team_ratings_by_week",
    "scrimmage_rows",
]
