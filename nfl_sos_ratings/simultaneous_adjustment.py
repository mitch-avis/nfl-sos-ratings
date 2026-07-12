"""Simultaneous opponent-adjustment helpers for teams and quarterbacks."""

import numpy as np
import polars as pl


def _sorted_entities(df: pl.DataFrame, *columns: str) -> list[str]:
    """Return sorted unique entity labels from one or more string columns."""
    values: set[str] = set()
    for column in columns:
        if column in df.columns:
            values.update(df.select(column).drop_nulls().to_series().cast(pl.String).to_list())
    return sorted(values)


def _solve_linear_system(
    design: np.ndarray, response: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    """Solve an ordinary or ridge least-squares system."""
    if ridge_lambda <= 0.0:
        solution, *_ = np.linalg.lstsq(design, response, rcond=None)
        return solution

    xtx = design.T @ design
    ridge = ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    return np.linalg.solve(xtx + ridge, design.T @ response)


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


def solve_team_stat_ridge(
    team_games: pl.DataFrame,
    response_col: str,
    team_col: str = "team",
    opponent_col: str = "opponent_team",
    ridge_lambda: float = 1.0,
) -> pl.DataFrame:
    """Jointly estimate team offense and defense ratings for one response column."""
    if team_games.is_empty():
        return pl.DataFrame(
            schema={team_col: pl.String, "offense_rating": pl.Float64, "defense_rating": pl.Float64}
        )

    teams = _sorted_entities(team_games, team_col, opponent_col)
    team_index = {team: index for index, team in enumerate(teams)}
    team_count = len(teams)
    design = np.zeros((team_games.height, team_count * 2), dtype=np.float64)
    response = np.array(
        team_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )

    for row_index, row in enumerate(team_games.select([team_col, opponent_col]).iter_rows()):
        team, opponent = row
        design[row_index, team_index[str(team)]] = 1.0
        design[row_index, team_count + team_index[str(opponent)]] = -1.0

    coefficients = _solve_linear_system(design, response, ridge_lambda)
    offense = coefficients[:team_count] - coefficients[:team_count].mean()
    defense = coefficients[team_count:] - coefficients[team_count:].mean()
    return pl.DataFrame(
        {
            team_col: teams,
            "offense_rating": np.round(offense, 6).tolist(),
            "defense_rating": np.round(defense, 6).tolist(),
        }
    ).sort(team_col)


def solve_qb_stat_ridge(
    qb_games: pl.DataFrame,
    response_col: str,
    qb_col: str = "qb_id",
    defense_col: str = "opponent_team",
    ridge_lambda: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Jointly estimate quarterback offense and defense-allowed ratings."""
    qb_games = qb_games.drop_nulls([qb_col, defense_col, response_col])
    if qb_games.is_empty():
        return (
            pl.DataFrame(schema={qb_col: pl.String, "offense_rating": pl.Float64}),
            pl.DataFrame(schema={"team": pl.String, "defense_rating": pl.Float64}),
        )

    quarterbacks = _sorted_entities(qb_games, qb_col)
    defenses = _sorted_entities(qb_games, defense_col)
    qb_index = {qb: index for index, qb in enumerate(quarterbacks)}
    defense_index = {team: index for index, team in enumerate(defenses)}
    qb_count = len(quarterbacks)
    defense_count = len(defenses)
    design = np.zeros((qb_games.height, qb_count + defense_count), dtype=np.float64)
    response = np.array(
        qb_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )

    for row_index, row in enumerate(qb_games.select([qb_col, defense_col]).iter_rows()):
        quarterback, defense = row
        design[row_index, qb_index[str(quarterback)]] = 1.0
        design[row_index, qb_count + defense_index[str(defense)]] = -1.0

    coefficients = _solve_linear_system(design, response, ridge_lambda)
    offense = coefficients[:qb_count] - coefficients[:qb_count].mean()
    defense = coefficients[qb_count:] - coefficients[qb_count:].mean()
    return (
        pl.DataFrame({qb_col: quarterbacks, "offense_rating": np.round(offense, 6).tolist()}).sort(
            qb_col
        ),
        pl.DataFrame({"team": defenses, "defense_rating": np.round(defense, 6).tolist()}).sort(
            "team"
        ),
    )


def compute_team_adjusted_stats(
    team_games: pl.DataFrame,
    response_cols: list[str],
    team_col: str = "team",
    opponent_col: str = "opponent_team",
    ridge_lambda: float = 1.0,
) -> pl.DataFrame:
    """Compute prefixed offense and defense adjustments for multiple team stats."""
    teams = _sorted_entities(team_games, team_col, opponent_col)
    result = pl.DataFrame({team_col: teams})

    for response_col in response_cols:
        if response_col not in team_games.columns:
            continue
        solved = solve_team_stat_ridge(
            team_games,
            response_col=response_col,
            team_col=team_col,
            opponent_col=opponent_col,
            ridge_lambda=ridge_lambda,
        ).rename(
            {
                "offense_rating": f"adj_off_{response_col}",
                "defense_rating": f"adj_def_{response_col}",
            }
        )
        result = result.join(solved, on=team_col, how="left")

    return result.sort(team_col)


def compute_qb_adjusted_stats(
    qb_games: pl.DataFrame,
    response_cols: list[str],
    qb_col: str = "qb_id",
    defense_col: str = "opponent_team",
    ridge_lambda: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute prefixed QB and defense adjustments for multiple QB stats."""
    qb_games = qb_games.drop_nulls([qb_col, defense_col])
    quarterbacks = _sorted_entities(qb_games, qb_col)
    defenses = _sorted_entities(qb_games, defense_col)
    qb_result = pl.DataFrame({qb_col: quarterbacks})
    defense_result = pl.DataFrame({"team": defenses})

    for response_col in response_cols:
        if response_col not in qb_games.columns:
            continue
        qb_ratings, defense_ratings = solve_qb_stat_ridge(
            qb_games,
            response_col=response_col,
            qb_col=qb_col,
            defense_col=defense_col,
            ridge_lambda=ridge_lambda,
        )
        qb_result = qb_result.join(
            qb_ratings.rename({"offense_rating": f"adj_{response_col}"}),
            on=qb_col,
            how="left",
        )
        defense_result = defense_result.join(
            defense_ratings.rename({"defense_rating": f"adj_def_{response_col}"}),
            on="team",
            how="left",
        )

    return qb_result.sort(qb_col), defense_result.sort("team")
