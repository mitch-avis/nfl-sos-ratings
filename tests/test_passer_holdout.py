"""Tests for the passer holdout: a passer's later games against his rating season's model."""

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.ridge import UnitFit
from nfl_sos_ratings.validation import passer_holdout
from nfl_sos_ratings.validation.passer_holdout import (
    postseason_rows,
    reading,
    residual_sigma,
    score_part,
)
from tests.stubs import stub

if TYPE_CHECKING:
    from pathlib import Path

_FIT = UnitFit(
    intercept=0.05,
    home_field=0.02,
    ridge_lambda=100.0,
    offense={"qb-1": 0.15, "qb-2": -0.05},
    defense={"BUF": 0.04, "MIA": -0.03},
)


def _rows(
    qb_ids: list[str], opponents: list[str], dropbacks: list[int], epa: list[float]
) -> pl.DataFrame:
    """Return neutral-site passer-game rows."""
    return pl.DataFrame(
        {
            "qb_id": qb_ids,
            "opponent_team": opponents,
            "is_home": [None] * len(qb_ids),
            "qb_dropbacks": dropbacks,
            "qb_epa_per_dropback": epa,
        },
        schema_overrides={"is_home": pl.Boolean},
    )


def test_residual_sigma_uses_qualifying_passers_dropback_scaled_residuals() -> None:
    # Arrange
    # Predictions: qb-1 vs BUF 0.16, qb-1 vs MIA 0.23, qb-2 vs BUF -0.04.
    rows = _rows(["qb-1", "qb-1", "qb-2"], ["BUF", "MIA", "BUF"], [25, 36, 49], [0.26, 0.13, 0.96])

    # Act
    sigma = residual_sigma(rows, _FIT, {"qb-1"})

    # Assert
    assert sigma == pytest.approx(math.sqrt((25 * 0.1**2 + 36 * 0.1**2) / 2))


def test_score_part_reports_the_weighted_residual_and_its_z() -> None:
    # Arrange
    rows = _rows(["qb-1", "qb-1"], ["BUF", "MIA"], [30, 10], [0.06, 0.23])

    # Act
    part = score_part("postseason", rows, _FIT, sigma=0.5)

    # Assert
    # Residuals: 0.06 - 0.16 = -0.10 over 30 dropbacks, 0.23 - 0.23 = 0 over 10.
    assert (part.games, part.dropbacks) == (2, 40)
    assert part.residual == pytest.approx(-0.075)
    assert part.z == pytest.approx(-0.075 / (0.5 / math.sqrt(40)))


def test_score_part_adds_home_field_for_a_home_game() -> None:
    # Arrange
    rows = _rows(["qb-1"], ["BUF"], [40], [0.18]).with_columns(pl.lit(True).alias("is_home"))

    # Act
    part = score_part("later season", rows, _FIT, sigma=0.5)

    # Assert
    assert part.predicted == pytest.approx(0.05 + 0.15 - 0.04 + 0.02)


def test_score_part_with_an_unknown_opponent_raises_value_error() -> None:
    # Arrange
    rows = _rows(["qb-1"], ["XXX"], [40], [0.1])

    # Act & Assert
    with pytest.raises(ValueError, match="XXX"):
        score_part("later season", rows, _FIT, sigma=0.5)


def test_postseason_rows_name_the_opponent_and_treat_the_super_bowl_as_neutral() -> None:
    # Arrange
    playoff_qb = pl.DataFrame(
        {
            "game_id": ["g-wc", "g-div", "g-sb"],
            "week": [19, 20, 22],
            "team_abbr": ["NE", "NE", "NE"],
            "qb_id": ["qb-1", "qb-1", "qb-1"],
            "qb_dropbacks": [30, 35, 40],
            "qb_epa_per_dropback": [0.1, 0.2, 0.3],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": ["g-wc", "g-div", "g-sb"],
            "game_type": ["WC", "DIV", "SB"],
            "home_team": ["NE", "HOU", "SEA"],
            "away_team": ["LAC", "NE", "NE"],
        }
    )

    # Act
    rows = postseason_rows(playoff_qb, schedule, "qb-1")

    # Assert
    assert rows.select("opponent_team", "is_home").rows() == [
        ("LAC", True),
        ("HOU", False),
        ("SEA", None),
    ]


@pytest.mark.parametrize(
    ("z", "expected"),
    [(-2.0, "the model season's rating overstated"), (-1.9, "within the noise")],
)
def test_reading_applies_the_z_threshold(z: float, expected: str) -> None:
    # Act
    text = reading(z)

    # Assert
    assert expected in text


def _write_model_season(data_dir: Path) -> None:
    """Write a 2025 double round-robin of three passers plus a 2026 game for qb-1."""
    teams = {"qb-1": "NE", "qb-2": "BUF", "qb-3": "MIA"}
    opponents = {"NE": ("BUF", "MIA"), "BUF": ("NE", "MIA"), "MIA": ("NE", "BUF")}
    strength = {"qb-1": 0.2, "qb-2": 0.0, "qb-3": -0.1}
    rows: list[dict[str, object]] = []
    for game_round in range(2):
        for qb_id, team in teams.items():
            for opponent in opponents[team]:
                pair = "_".join(sorted((team, opponent)))
                rows.append(
                    {
                        "game_id": f"2025_{game_round}_{pair}",
                        "week": game_round + 1,
                        "team": team,
                        "opponent_team": opponent,
                        "qb_id": qb_id,
                        "qb_name": f"Passer {qb_id}",
                        "is_home": team < opponent,
                        "qb_dropbacks": 35,
                        "qb_epa_per_dropback": 0.05 + strength[qb_id] + 0.01 * game_round,
                    }
                )
    pl.DataFrame(rows).write_parquet(data_dir / "2025_qb_game_logs.parquet")
    pl.DataFrame({"qb_id": ["qb-1", "qb-2", "qb-3"]}).write_parquet(
        data_dir / "2025_qb_ratings.parquet"
    )
    pl.DataFrame(
        {
            "game_id": ["2026_01_BUF_NE"],
            "week": [1],
            "team": ["NE"],
            "opponent_team": ["BUF"],
            "qb_id": ["qb-1"],
            "qb_name": ["Passer qb-1"],
            "is_home": [True],
            "qb_dropbacks": [40],
            "qb_epa_per_dropback": [0.1],
        }
    ).write_parquet(data_dir / "2026_qb_game_logs.parquet")


def test_main_reports_each_part_with_its_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_model_season(tmp_path)
    playoff_qb = pl.DataFrame(
        {
            "game_id": ["g-sb"],
            "week": [22],
            "team_abbr": ["NE"],
            "qb_id": ["qb-1"],
            "qb_dropbacks": [40],
            "qb_epa_per_dropback": [0.3],
        }
    )
    schedule = pl.DataFrame(
        {"game_id": ["g-sb"], "game_type": ["SB"], "home_team": ["MIA"], "away_team": ["NE"]}
    )
    monkeypatch.setattr(passer_holdout, "load_playoff_qb_stats", stub(lambda: playoff_qb))
    monkeypatch.setattr(passer_holdout, "load_playoff_schedule", stub(lambda: schedule))
    monkeypatch.setattr(passer_holdout, "use_disk_cache_unless_configured", stub(lambda: None))

    # Act
    passer_holdout.main(["--data-dir", str(tmp_path), "--name", "Passer qb-1"])

    # Assert
    output = capsys.readouterr().out
    assert "2025 postseason: 1 games, 40 dropbacks" in output
    assert "2026 regular season: 1 games, 40 dropbacks" in output
    assert "both: 2 games, 80 dropbacks" in output
    assert output.count("Reading: ") == 3


def test_main_rejects_a_name_with_no_model_season_games(tmp_path: Path) -> None:
    # Arrange
    _write_model_season(tmp_path)

    # Act & Assert
    with pytest.raises(SystemExit, match="No 2025 passer named Nobody"):
        passer_holdout.main(["--data-dir", str(tmp_path), "--name", "Nobody"])


def test_main_reports_no_games_for_a_later_season_not_yet_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_model_season(tmp_path)
    (tmp_path / "2026_qb_game_logs.parquet").unlink()
    empty_playoffs = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team_abbr": pl.String,
            "qb_id": pl.String,
            "qb_dropbacks": pl.Int64,
            "qb_epa_per_dropback": pl.Float64,
        }
    )
    schedule = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "game_type": pl.String,
            "home_team": pl.String,
            "away_team": pl.String,
        }
    )
    monkeypatch.setattr(passer_holdout, "load_playoff_qb_stats", stub(lambda: empty_playoffs))
    monkeypatch.setattr(passer_holdout, "load_playoff_schedule", stub(lambda: schedule))
    monkeypatch.setattr(passer_holdout, "use_disk_cache_unless_configured", stub(lambda: None))

    # Act
    passer_holdout.main(["--data-dir", str(tmp_path), "--name", "Passer qb-1"])

    # Assert
    output = capsys.readouterr().out
    assert "2026 regular season: no games" in output
    assert "both: no games" in output
