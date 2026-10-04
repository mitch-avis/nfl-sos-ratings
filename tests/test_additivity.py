"""Tests for the additivity check: pair-held-out residuals by offense and defense tercile."""

import itertools
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.ridge import UnitColumns
from nfl_sos_ratings.validation import additivity
from nfl_sos_ratings.validation.additivity import (
    TercileCuts,
    contrast,
    pair_holdout_residuals,
    passer_residuals,
    reading,
    season_block_interval,
    team_residuals,
)

if TYPE_CHECKING:
    from pathlib import Path

_OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.0, "DDD": -0.02, "EEE": -0.05, "FFF": -0.08}
_DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.02, "DDD": -0.06, "EEE": 0.05, "FFF": -0.05}
_COLUMNS = UnitColumns(response="epa_per_play", weight="plays")
_TINY_LAMBDA = 1e-6


def _rows(shock: float = 0.0) -> pl.DataFrame:
    """Return noiseless offense rows for a six-team double round-robin.

    ``shock`` is added to AAA's offense in its first game against BBB.
    """
    rows: list[dict[str, object]] = []
    for game_number, (home, away) in enumerate(itertools.permutations(_OFFENSE, 2)):
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            value = 0.02 + _OFFENSE[team] - _DEFENSE[opponent] + (0.01 if is_home else -0.01)
            if (game_number, team) == (0, "AAA"):
                value += shock
            rows.append(
                {
                    "game_id": f"g{game_number:02d}",
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "plays": 60,
                    "epa_per_play": value,
                }
            )
    return pl.DataFrame(rows)


def _cuts() -> tuple[TercileCuts, TercileCuts]:
    """Return tercile cut points of the known offense and defense effects."""
    return TercileCuts.from_effects(_OFFENSE.values()), TercileCuts.from_effects(_DEFENSE.values())


def _team_game_logs(season: int) -> pl.DataFrame:
    """Return team-game logs with the columns the team rating needs, built from ``_rows``."""
    return (
        _rows()
        .with_columns(
            pl.lit(season).alias("season"),
            pl.col("plays").alias("offensive_snaps"),
            (pl.col("epa_per_play") * pl.col("plays")).alias("offensive_epa"),
            pl.lit(12).alias("st_plays"),
            pl.lit(0.0).alias("st_epa"),
        )
        .drop("plays", "epa_per_play")
    )


def _qb_game_logs() -> pl.DataFrame:
    """Return one passer per team-game; AAA uses two passers, one only against BBB."""
    return _rows().select(
        "game_id",
        "team",
        "opponent_team",
        "is_home",
        pl.when((pl.col("team") == "AAA") & (pl.col("opponent_team") == "BBB"))
        .then(pl.lit("qb_AAA_backup"))
        .otherwise(pl.format("qb_{}", pl.col("team")))
        .alias("qb_id"),
        pl.lit(35).alias("qb_dropbacks"),
        pl.col("epa_per_play").alias("qb_epa_per_dropback"),
    )


def _tiered(
    residuals: list[float], offense: list[str], defense: list[str], season: int = 2020
) -> pl.DataFrame:
    """Return residual rows with fixed tiers and unit weights."""
    return pl.DataFrame(
        {
            "season": [season] * len(residuals),
            "weight": [1.0] * len(residuals),
            "residual": residuals,
            "offense_tier": offense,
            "defense_tier": defense,
        }
    )


def test_tercile_cuts_split_effects_at_the_one_third_and_two_third_quantiles() -> None:
    # Act
    cuts = TercileCuts.from_effects([0.0, 3.0, 6.0, 9.0])

    # Assert
    assert (cuts.lower, cuts.upper) == pytest.approx((3.0, 6.0))


@pytest.mark.parametrize(("effect", "tier"), [(2.9, "bottom"), (4.5, "middle"), (6.1, "top")])
def test_tercile_cuts_place_an_effect_in_its_tier(effect: float, tier: str) -> None:
    # Arrange
    cuts = TercileCuts(lower=3.0, upper=6.0)

    # Act
    result = cuts.tier(effect)

    # Assert
    assert result == tier


def test_pair_holdout_residuals_are_zero_for_an_additive_league() -> None:
    # Arrange
    offense_cuts, defense_cuts = _cuts()

    # Act
    residuals = pair_holdout_residuals(
        _rows(),
        _COLUMNS,
        ridge_lambda=_TINY_LAMBDA,
        offense_cuts=offense_cuts,
        defense_cuts=defense_cuts,
    )

    # Assert
    assert residuals.get_column("residual").abs().max() == pytest.approx(0.0, abs=1e-6)


def test_pair_holdout_residuals_predict_a_pair_without_its_own_games() -> None:
    # Arrange
    offense_cuts, defense_cuts = _cuts()

    # Act
    residuals = pair_holdout_residuals(
        _rows(shock=0.3),
        _COLUMNS,
        ridge_lambda=_TINY_LAMBDA,
        offense_cuts=offense_cuts,
        defense_cuts=defense_cuts,
    )

    # Assert
    shocked = residuals.filter((pl.col("game_id") == "g00") & (pl.col("team") == "AAA"))
    assert shocked.get_column("residual").item() == pytest.approx(0.3, abs=1e-6)


def test_pair_holdout_residuals_tier_units_by_their_held_out_effects() -> None:
    # Arrange
    offense_cuts, defense_cuts = _cuts()

    # Act
    residuals = pair_holdout_residuals(
        _rows(),
        _COLUMNS,
        ridge_lambda=_TINY_LAMBDA,
        offense_cuts=offense_cuts,
        defense_cuts=defense_cuts,
    )

    # Assert
    row = residuals.filter((pl.col("team") == "AAA") & (pl.col("opponent_team") == "FFF")).row(
        0, named=True
    )
    assert (row["offense_tier"], row["defense_tier"]) == ("top", "bottom")


def test_contrast_is_top_offense_residual_against_weak_minus_strong_defenses() -> None:
    # Arrange
    rows = _tiered(
        [0.3, 0.1, -0.2, 5.0],
        ["top", "top", "top", "bottom"],
        ["bottom", "bottom", "top", "bottom"],
    )

    # Act
    value = contrast(rows)

    # Assert
    assert value == pytest.approx(0.2 - (-0.2))


def test_season_block_interval_collapses_when_every_season_agrees() -> None:
    # Arrange
    rows = pl.concat(
        [
            _tiered([0.3, -0.1], ["top", "top"], ["bottom", "top"], season=season)
            for season in (2019, 2020, 2021)
        ]
    )

    # Act
    lower, upper = season_block_interval(rows)

    # Assert
    assert (lower, upper) == pytest.approx((0.4, 0.4))


def test_season_block_interval_resamples_whole_seasons() -> None:
    # Arrange
    rows = pl.concat(
        [
            _tiered([0.5, 0.0], ["top", "top"], ["bottom", "top"], season=2019),
            _tiered([-0.5, 0.0], ["top", "top"], ["bottom", "top"], season=2020),
        ]
    )

    # Act
    lower, upper = season_block_interval(rows, resamples=200)

    # Assert
    assert (lower, upper) == pytest.approx((-0.5, 0.5))


def test_team_residuals_score_every_team_game() -> None:
    # Arrange
    game_logs = _team_game_logs(2020)

    # Act
    residuals = team_residuals(game_logs)

    # Assert
    assert residuals.height == game_logs.height
    assert residuals.get_column("residual").null_count() == 0


def test_passer_residuals_leave_out_a_passer_seen_only_in_the_held_out_games() -> None:
    # Arrange
    qb_game_logs = _qb_game_logs()

    # Act
    residuals = passer_residuals(qb_game_logs, {"qb_AAA", "qb_BBB", "qb_CCC", "qb_DDD"})

    # Assert
    backup = residuals.filter(pl.col("qb_id") == "qb_AAA_backup")
    assert backup.height == 2
    assert backup.get_column("residual").null_count() == 2


def test_main_reports_the_contrast_interval_and_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    for season in (2019, 2020):
        _team_game_logs(season).write_parquet(tmp_path / f"{season}_team_game_logs.parquet")
        _qb_game_logs().write_parquet(tmp_path / f"{season}_qb_game_logs.parquet")
        pl.DataFrame({"qb_id": ["qb_AAA", "qb_BBB", "qb_CCC", "qb_DDD"]}).write_parquet(
            tmp_path / f"{season}_qb_ratings.parquet"
        )

    # Act
    additivity.main(["--data-dir", str(tmp_path), "--start-season", "2019", "--end-season", "2020"])

    # Assert
    output = capsys.readouterr().out
    assert "Teams (scrimmage EPA per play)" in output
    assert "Passers (EPA per dropback)" in output
    assert "95% season-block interval" in output
    assert output.count("Reading: ") == 2


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.01, 0.05, "the additive model misses the claimed effect"),
        (-0.01, 0.05, "no evidence against additivity"),
        (-0.05, -0.01, "the opposite of the claim"),
    ],
)
def test_reading_applies_the_decision_rule(lower: float, upper: float, expected: str) -> None:
    # Act
    text = reading(lower, upper)

    # Assert
    assert text.startswith(expected)


def test_contrast_accepts_one_seasons_rows_without_a_season_column() -> None:
    # Arrange
    rows = _tiered([0.3, -0.1], ["top", "top"], ["bottom", "top"]).drop("season")

    # Act
    value = contrast(rows)

    # Assert
    assert value == pytest.approx(0.4)
