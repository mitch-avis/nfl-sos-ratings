"""Tests for the in-season penalty test: previous-season penalties against per-fit tuning."""

import itertools
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.team_rating import fit_team_ratings
from nfl_sos_ratings.validation import in_season_penalty
from nfl_sos_ratings.validation.in_season_penalty import (
    CANDIDATE_BASELINE,
    build_prior_penalty_feature_rows,
    compare_windows,
    reading,
    scrimmage_penalty_table,
)
from nfl_sos_ratings.validation.walk_forward import TEAM_RATING_BASELINE

if TYPE_CHECKING:
    from pathlib import Path

_STRENGTH = {"AAA": 0.08, "BBB": 0.03, "CCC": -0.02, "DDD": -0.09}


def _game_logs(season: int, *, rounds: int = 2) -> pl.DataFrame:
    """Return team-game rows for ``rounds`` double round-robins, one game per week."""
    rows: list[dict[str, object]] = []
    matchups = itertools.product(range(rounds), itertools.permutations(_STRENGTH, 2))
    for week, (game_round, (home, away)) in enumerate(matchups, start=1):
        margin = round((_STRENGTH[home] - _STRENGTH[away]) * 100) + 2 + game_round
        for team, opponent, is_home, team_margin in (
            (home, away, True, margin),
            (away, home, False, -margin),
        ):
            epa = 0.01 + _STRENGTH[team] - _STRENGTH[opponent] / 2 + (week % 3) * 0.01
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "point_margin": team_margin,
                    "offensive_snaps": 62,
                    "offensive_epa": epa * 62,
                    "st_plays": 13,
                    "st_epa": 0.05 if is_home else -0.05,
                }
            )
    return pl.DataFrame(rows)


def _predictions(errors: dict[tuple[str, int], float]) -> pl.DataFrame:
    """Return one prediction per (baseline, week) with the given error, one game per week."""
    return pl.DataFrame(
        [
            {
                "season": 2020,
                "week": week,
                "game_id": f"g{week:02d}",
                "home_team": "AAA",
                "away_team": "BBB",
                "baseline": baseline,
                "error": error,
            }
            for (baseline, week), error in errors.items()
        ]
    )


def test_build_prior_penalty_feature_rows_rate_teams_with_the_previous_seasons_penalties() -> None:
    # Arrange
    game_logs = _game_logs(2021)
    prior_fit = fit_team_ratings(_game_logs(2020), scrimmage_lambda=40.0, special_teams_lambda=80.0)
    snapshot = fit_team_ratings(
        game_logs.filter(pl.col("week") < 7), scrimmage_lambda=40.0, special_teams_lambda=80.0
    ).ratings
    week_seven = game_logs.filter((pl.col("week") == 7) & pl.col("is_home")).row(0, named=True)
    rating = dict(snapshot.select("team", "team_rating").iter_rows())

    # Act
    rows = build_prior_penalty_feature_rows(game_logs, 2021, prior_fit)

    # Assert
    row = rows.filter(pl.col("week") == 7).row(0, named=True)
    assert row["baseline"] == CANDIDATE_BASELINE
    assert row["rating_diff"] == pytest.approx(
        rating[week_seven["team"]] - rating[week_seven["opponent_team"]]
    )


def test_compare_windows_scores_weeks_two_to_five_apart_from_later_weeks() -> None:
    # Arrange
    errors = {(CANDIDATE_BASELINE, week): 1.0 for week in range(2, 9)}
    errors |= {(TEAM_RATING_BASELINE, week): 2.0 for week in range(2, 6)}
    errors |= {(TEAM_RATING_BASELINE, week): 1.0 for week in range(6, 9)}

    # Act
    windows = compare_windows(_predictions(errors))

    # Assert
    deltas = dict(windows.select("window", "mae_delta").iter_rows())
    assert deltas == pytest.approx({"primary": -1.0, "guard": 0.0})


@pytest.mark.parametrize(
    ("primary", "guard", "expected"),
    [
        ((-0.2, -0.1), (-0.1, 0.1), "recommend the candidate"),
        ((-0.2, 0.1), (-0.1, 0.1), "a tie"),
        ((0.1, 0.2), (-0.1, 0.1), "keep cross-validation"),
        ((-0.2, -0.1), (0.1, 0.2), "keep cross-validation"),
    ],
)
def test_reading_applies_the_decision_rule(
    primary: tuple[float, float], guard: tuple[float, float], expected: str
) -> None:
    # Act
    text = reading(primary, guard)

    # Assert
    assert text.startswith(expected)


def test_scrimmage_penalty_table_lists_the_week_by_week_choices(tmp_path: Path) -> None:
    # Arrange
    game_logs = _game_logs(2020)
    game_logs.write_parquet(tmp_path / "2020_team_game_logs.parquet")

    # Act
    table = scrimmage_penalty_table(tmp_path, [2020])

    # Assert
    assert table.columns == ["season", "week_2", "week_3", "week_4", "week_5", "full"]
    assert table.get_column("full").item() == fit_team_ratings(game_logs).scrimmage_lambda


def test_main_reports_both_windows_and_the_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    for season in (2019, 2020, 2021):
        _game_logs(season).write_parquet(tmp_path / f"{season}_team_game_logs.parquet")

    # Act
    in_season_penalty.main(
        ["--data-dir", str(tmp_path), "--start-season", "2020", "--end-season", "2021"]
    )

    # Assert
    output = capsys.readouterr().out
    assert "Primary (prediction weeks 2-5)" in output
    assert "Guard (prediction weeks 6 and later)" in output
    assert "Seasons at the grid's largest penalty" in output
    assert output.count("Reading: ") == 1
