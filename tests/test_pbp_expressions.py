"""Tests for the shared play-by-play expressions: who lost a fumble and who gave the ball away."""

import polars as pl
import pytest

from nfl_sos_ratings.pbp_expressions import giveaway_team_expr, lost_fumble_team_expr


def _play(**overrides: object) -> dict[str, object]:
    """Return one DEN-possession play with no turnover."""
    play: dict[str, object] = {
        "posteam": "DEN",
        "defteam": "KC",
        "punt_attempt": 0,
        "interception": 0,
        "fumble_lost": 0,
        "fumbled_1_team": None,
    }
    play.update(overrides)
    return play


@pytest.mark.parametrize(
    ("play", "expected"),
    [
        (_play(), None),
        (_play(fumble_lost=1, fumbled_1_team="DEN"), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(interception=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(fumble_lost=1), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1), "KC"),
        (_play(interception=1, fumble_lost=1), "KC"),
    ],
    ids=[
        "no lost fumble",
        "offense fumble",
        "returner muff",
        "interception fumbled back",
        "fumbler unknown on a scrimmage play",
        "fumbler unknown on a punt",
        "fumbler unknown after an interception",
    ],
)
def test_lost_fumble_team_names_the_fumbling_team(
    play: dict[str, object], expected: str | None
) -> None:
    # Arrange
    plays = pl.DataFrame([play], schema_overrides={"fumbled_1_team": pl.String})

    # Act
    team = plays.select(lost_fumble_team_expr(plays.columns)).item()

    # Assert
    assert team == expected


def test_lost_fumble_team_falls_back_without_a_fumbler_column() -> None:
    # Arrange
    plays = pl.DataFrame([_play(fumble_lost=1)]).drop("fumbled_1_team")

    # Act
    team = plays.select(lost_fumble_team_expr(plays.columns)).item()

    # Assert
    assert team == "DEN"


@pytest.mark.parametrize(
    ("play", "expected"),
    [
        (_play(), None),
        (_play(interception=1), "DEN"),
        (_play(interception=1, fumble_lost=1, fumbled_1_team="KC"), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(fumble_lost=1, fumbled_1_team="DEN"), "DEN"),
    ],
    ids=[
        "no giveaway",
        "interception",
        "interception fumbled back",
        "returner muff",
        "offense fumble",
    ],
)
def test_giveaway_team_names_the_first_team_to_give_the_ball_away(
    play: dict[str, object], expected: str | None
) -> None:
    # Arrange
    plays = pl.DataFrame([play], schema_overrides={"fumbled_1_team": pl.String})

    # Act
    team = plays.select(giveaway_team_expr(plays.columns)).item()

    # Assert
    assert team == expected
