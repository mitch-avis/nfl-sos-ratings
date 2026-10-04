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
_RESPONSE = "_epa_per_play"
_WEIGHT = "_plays"
_UNIT_COLUMNS = UnitColumns(response=_RESPONSE, weight=_WEIGHT)


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
    scrimmage = fit_unit_ridge(scrimmage_rows, _UNIT_COLUMNS, ridge_lambda=scrimmage_lambda)
    special_teams = fit_unit_ridge(special_rows, _UNIT_COLUMNS, ridge_lambda=special_teams_lambda)
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


def _ratings_without(game_logs: pl.DataFrame, team: str, fit: TeamRatingFit) -> pl.DataFrame:
    """Rate every other team from a refit that drops all games involving ``team``."""
    others = game_logs.filter((pl.col("team") != team) & (pl.col("opponent_team") != team))
    scrimmage = fit_unit_ridge(
        _unit_rows(others, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN),
        _UNIT_COLUMNS,
        ridge_lambda=fit.scrimmage_lambda,
    )
    special_teams = fit_unit_ridge(
        _unit_rows(others, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN),
        _UNIT_COLUMNS,
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
        One row per team with ``team`` and ``sos`` in points per game.

    Raises:
        ValueError: If an opponent has no games left once the team's own games are removed.

    """
    _require_columns(game_logs)
    teams: list[str] = fit.ratings.get_column("team").to_list()
    values: list[float] = []
    for team in teams:
        opponent_ratings = _ratings_without(game_logs, team, fit).select(
            pl.col("team").alias("opponent_team"), pl.col("team_rating").alias("sos")
        )
        faced = (
            game_logs.filter(pl.col("team") == team)
            .select("opponent_team")
            .join(opponent_ratings, on="opponent_team", how="left")
        )
        unmatched = faced.filter(pl.col("sos").is_null()).get_column("opponent_team").to_list()
        if unmatched:
            msg = f"no head-to-head-excluded rating for {team} opponents: {', '.join(unmatched)}"
            raise ValueError(msg)
        values.append(float(np.mean(faced.get_column("sos").to_numpy())))
    return pl.DataFrame({"team": teams, "sos": values})


__all__ = [
    "SCRIMMAGE_EPA_COLUMN",
    "SCRIMMAGE_PLAYS_COLUMN",
    "SPECIAL_TEAMS_EPA_COLUMN",
    "SPECIAL_TEAMS_PLAYS_COLUMN",
    "TEAM_RATING_COLUMNS",
    "TeamRatingFit",
    "compute_team_schedule_strength",
    "fit_team_ratings",
]
