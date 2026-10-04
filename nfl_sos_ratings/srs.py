"""Simple rating system (SRS): point-margin ratings solved across the whole league at once.

SRS is the score-based reference beside the EPA-based team rating: each team's rating is its
average margin adjusted for the average rating of the opponents it played, solved for every
team simultaneously by least squares and centered so the league average is zero.
"""

import numpy as np
import polars as pl


def _sorted_entities(df: pl.DataFrame, *columns: str) -> list[str]:
    """Return sorted unique entity labels from one or more string columns."""
    values: set[str] = set()
    for column in columns:
        values.update(df.get_column(column).drop_nulls().cast(pl.String).to_list())
    return sorted(values)


def solve_srs(
    team_games: pl.DataFrame,
    response_col: str,
    team_col: str = "team",
    opponent_col: str = "opponent_team",
) -> pl.DataFrame:
    """Solve a centered simple rating system from team-game responses."""
    if team_games.is_empty():
        return pl.DataFrame(schema={team_col: pl.String, "srs_rating": pl.Float64})

    teams = _sorted_entities(team_games, team_col, opponent_col)
    team_index = {team: index for index, team in enumerate(teams)}
    design = np.zeros((team_games.height, len(teams)), dtype=np.float64)
    response = np.array(
        team_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )

    for row_index, row in enumerate(team_games.select([team_col, opponent_col]).iter_rows()):
        team, opponent = row
        design[row_index, team_index[str(team)]] = 1.0
        design[row_index, team_index[str(opponent)]] = -1.0

    ratings, *_ = np.linalg.lstsq(design, response, rcond=None)
    ratings = ratings - ratings.mean()
    return pl.DataFrame({team_col: teams, "srs_rating": np.round(ratings, 6).tolist()}).sort(
        team_col
    )


__all__ = ["solve_srs"]
