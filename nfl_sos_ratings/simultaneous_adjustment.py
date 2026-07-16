"""Simultaneous opponent-adjustment helpers for teams and quarterbacks."""

import numpy as np
import polars as pl

_DEFAULT_RIDGE_LAMBDAS = np.logspace(-6, 2, 17, dtype=np.float64)


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


def _solve_linear_system_with_penalties(
    design: np.ndarray,
    response: np.ndarray,
    penalties: np.ndarray,
) -> np.ndarray:
    """Solve a ridge system with a per-coefficient non-negative penalty vector."""
    if penalties.ndim != 1 or penalties.shape[0] != design.shape[1]:
        raise ValueError("penalties must be a 1D vector matching the design column count")
    if np.allclose(penalties, 0.0):
        solution, *_ = np.linalg.lstsq(design, response, rcond=None)
        return solution

    xtx = design.T @ design
    ridge = np.diag(np.clip(penalties.astype(np.float64), 0.0, None))
    return np.linalg.solve(xtx + ridge, design.T @ response)


def _apply_sample_weights(
    design: np.ndarray,
    response: np.ndarray,
    sample_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a linear system for weighted least squares when weights are provided."""
    if sample_weights is None:
        return design, response

    clipped_weights = np.clip(sample_weights.astype(np.float64), 0.0, None)
    sqrt_weights = np.sqrt(clipped_weights)
    return design * sqrt_weights[:, np.newaxis], response * sqrt_weights


def tune_ridge_lambda(
    design: np.ndarray,
    response: np.ndarray,
    candidate_lambdas: np.ndarray | None = None,
    folds: int = 5,
    sample_weights: np.ndarray | None = None,
) -> float:
    """Choose a ridge penalty by deterministic k-fold cross-validation."""
    if design.ndim != 2:
        raise ValueError("design must be a 2D matrix")
    if response.ndim != 1:
        raise ValueError("response must be a 1D vector")
    if design.shape[0] != response.shape[0]:
        raise ValueError("design and response must have the same number of rows")

    lambdas = (
        np.array(candidate_lambdas, dtype=np.float64)
        if candidate_lambdas is not None
        else _DEFAULT_RIDGE_LAMBDAS
    )
    if lambdas.size == 0:
        raise ValueError("candidate_lambdas must contain at least one value")

    row_count = design.shape[0]
    effective_folds = min(max(folds, 2), row_count)
    if row_count < 2:
        return float(lambdas[0])

    fold_ids = np.arange(row_count, dtype=np.int64) % effective_folds
    best_lambda = float(lambdas[0])
    best_error = float("inf")

    for ridge_lambda in lambdas:
        fold_errors: list[float] = []
        for fold_id in range(effective_folds):
            validation_mask = fold_ids == fold_id
            training_mask = ~validation_mask
            if not validation_mask.any() or not training_mask.any():
                continue

            train_design, train_response = _apply_sample_weights(
                design[training_mask],
                response[training_mask],
                sample_weights[training_mask] if sample_weights is not None else None,
            )
            coefficients = _solve_linear_system(train_design, train_response, float(ridge_lambda))
            residuals = response[validation_mask] - (design[validation_mask] @ coefficients)

            if sample_weights is None:
                fold_errors.append(float(np.mean(residuals**2)))
                continue

            validation_weights = np.clip(
                sample_weights[validation_mask].astype(np.float64), 0.0, None
            )
            positive_mask = validation_weights > 0.0
            if positive_mask.any():
                fold_errors.append(
                    float(
                        np.average(
                            residuals[positive_mask] ** 2,
                            weights=validation_weights[positive_mask],
                        )
                    )
                )

        if not fold_errors:
            continue

        mean_error = float(np.mean(fold_errors))
        if mean_error < best_error or (
            np.isclose(mean_error, best_error) and float(ridge_lambda) < best_lambda
        ):
            best_error = mean_error
            best_lambda = float(ridge_lambda)

    return best_lambda


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
    home_field_col: str = "is_home",
    ridge_lambda: float | None = None,
) -> tuple[pl.DataFrame, float]:
    """Jointly estimate team offense and defense ratings for one response column."""
    if team_games.is_empty():
        return (
            pl.DataFrame(
                schema={
                    team_col: pl.String,
                    "offense_rating": pl.Float64,
                    "defense_rating": pl.Float64,
                }
            ),
            0.0,
        )

    teams = _sorted_entities(team_games, team_col, opponent_col)
    team_index = {team: index for index, team in enumerate(teams)}
    team_count = len(teams)
    design = np.zeros((team_games.height, team_count * 2 + 1), dtype=np.float64)
    response = np.array(
        team_games.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )
    selected_columns = [team_col, opponent_col]
    has_home_field = home_field_col in team_games.columns
    if has_home_field:
        selected_columns.append(home_field_col)

    for row_index, row in enumerate(team_games.select(selected_columns).iter_rows()):
        team, opponent, *home_field_value = row
        design[row_index, team_index[str(team)]] = 1.0
        design[row_index, team_count + team_index[str(opponent)]] = -1.0
        if has_home_field:
            is_home = home_field_value[0]
            if is_home is not None:
                design[row_index, -1] = 1.0 if bool(is_home) else -1.0

    solved_lambda = (
        ridge_lambda if ridge_lambda is not None else tune_ridge_lambda(design, response)
    )
    coefficients = _solve_linear_system(design, response, solved_lambda)
    offense = coefficients[:team_count] - coefficients[:team_count].mean()
    defense_coefficients = coefficients[team_count : team_count * 2]
    defense = defense_coefficients - defense_coefficients.mean()
    home_field_advantage = float(coefficients[-1])
    return (
        pl.DataFrame(
            {
                team_col: teams,
                "offense_rating": np.round(offense, 6).tolist(),
                "defense_rating": np.round(defense, 6).tolist(),
            }
        ).sort(team_col),
        round(home_field_advantage, 6),
    )


def solve_qb_stat_ridge(
    qb_games: pl.DataFrame,
    response_col: str,
    qb_col: str = "qb_id",
    defense_col: str = "opponent_team",
    dropback_col: str = "qb_dropbacks",
    ridge_lambda: float | None = None,
    defense_ridge_lambda: float | None = None,
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

    sample_weights = None
    if dropback_col in qb_games.columns:
        sample_weights = np.array(
            qb_games.select(pl.col(dropback_col).cast(pl.Float64).fill_null(0.0))
            .to_series()
            .to_list(),
            dtype=np.float64,
        )

    solved_lambda = (
        ridge_lambda
        if ridge_lambda is not None
        else tune_ridge_lambda(design, response, sample_weights=sample_weights)
    )
    weighted_design, weighted_response = _apply_sample_weights(design, response, sample_weights)
    if defense_ridge_lambda is None:
        coefficients = _solve_linear_system(weighted_design, weighted_response, solved_lambda)
    else:
        penalties = np.concatenate(
            [
                np.full(qb_count, solved_lambda, dtype=np.float64),
                np.full(defense_count, defense_ridge_lambda, dtype=np.float64),
            ]
        )
        coefficients = _solve_linear_system_with_penalties(
            weighted_design,
            weighted_response,
            penalties,
        )
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


def solve_qb_stat_with_fixed_defense_offsets(
    qb_games: pl.DataFrame,
    response_col: str,
    fixed_defense_ratings: pl.DataFrame,
    qb_col: str = "qb_id",
    defense_col: str = "opponent_team",
    dropback_col: str = "qb_dropbacks",
    ridge_lambda: float | None = None,
) -> pl.DataFrame:
    """Estimate QB offense ratings with opponent defense effects held fixed.

    The supplied defense ratings are treated as known offsets in the same response units as
    ``response_col``. Only QB offense coefficients are penalized and fit.
    """
    qb_games = qb_games.drop_nulls([qb_col, defense_col, response_col])
    if qb_games.is_empty() or fixed_defense_ratings.is_empty():
        return pl.DataFrame(schema={qb_col: pl.String, "offense_rating": pl.Float64})

    joined = qb_games.join(
        fixed_defense_ratings.rename({"team": defense_col}),
        on=defense_col,
        how="inner",
    ).drop_nulls(["defense_rating"])
    if joined.is_empty():
        return pl.DataFrame(schema={qb_col: pl.String, "offense_rating": pl.Float64})

    quarterbacks = _sorted_entities(joined, qb_col)
    qb_index = {qb: index for index, qb in enumerate(quarterbacks)}
    design = np.zeros((joined.height, len(quarterbacks)), dtype=np.float64)

    for row_index, (quarterback,) in enumerate(joined.select([qb_col]).iter_rows()):
        design[row_index, qb_index[str(quarterback)]] = 1.0

    response = np.array(
        joined.select(pl.col(response_col).cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )
    defense_offsets = np.array(
        joined.select(pl.col("defense_rating").cast(pl.Float64)).to_series().to_list(),
        dtype=np.float64,
    )
    target = response + defense_offsets

    sample_weights = None
    if dropback_col in joined.columns:
        sample_weights = np.array(
            joined.select(pl.col(dropback_col).cast(pl.Float64).fill_null(0.0))
            .to_series()
            .to_list(),
            dtype=np.float64,
        )

    solved_lambda = (
        ridge_lambda
        if ridge_lambda is not None
        else tune_ridge_lambda(design, target, sample_weights=sample_weights)
    )
    weighted_design, weighted_target = _apply_sample_weights(design, target, sample_weights)
    coefficients = _solve_linear_system(weighted_design, weighted_target, solved_lambda)
    offense = coefficients - coefficients.mean()
    return pl.DataFrame(
        {qb_col: quarterbacks, "offense_rating": np.round(offense, 6).tolist()}
    ).sort(qb_col)


def compute_team_adjusted_stats(
    team_games: pl.DataFrame,
    response_cols: list[str],
    team_col: str = "team",
    opponent_col: str = "opponent_team",
    ridge_lambda: float | None = None,
) -> pl.DataFrame:
    """Compute prefixed offense and defense adjustments for multiple team stats."""
    teams = _sorted_entities(team_games, team_col, opponent_col)
    result = pl.DataFrame({team_col: teams})

    for response_col in response_cols:
        if response_col not in team_games.columns:
            continue
        solved, _ = solve_team_stat_ridge(
            team_games,
            response_col=response_col,
            team_col=team_col,
            opponent_col=opponent_col,
            ridge_lambda=ridge_lambda,
        )
        solved = solved.rename(
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
    ridge_lambda: float | None = None,
    defense_ridge_lambda: float | None = None,
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
            defense_ridge_lambda=defense_ridge_lambda,
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
