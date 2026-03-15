import polars as pl
import pytest

from nfl_sos_ratings import data_loader


def test_extract_points_per_team_week() -> None:
    schedule = pl.DataFrame(
        {
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "week": [1],
            "home_score": [24],
            "away_score": [17],
        }
    )

    result = data_loader._extract_points_per_team_week(schedule).sort(["team"])

    assert result.to_dicts() == [
        {"team": "DEN", "week": 1, "points_for": 24, "points_allowed": 17},
        {"team": "KC", "week": 1, "points_for": 17, "points_allowed": 24},
    ]


def test_load_weekly_team_stats_enriches_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    weekly = pl.DataFrame(
        {
            "team": ["DEN", "DEN"],
            "week": [1, 2],
            "season_type": ["REG", "POST"],
            "passing_yards": [200, 250],
            "rushing_yards": [100, 90],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_type": ["REG", "POST"],
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
            "week": [1, 2],
            "home_score": [24, 17],
            "away_score": [17, 21],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_team_stats", lambda seasons, summary_level: weekly)
    monkeypatch.setattr(data_loader.nfl, "load_schedules", lambda seasons: schedule)

    result = data_loader.load_weekly_team_stats(2025)

    assert result.height == 1
    assert result.select("total_yards").item() == 300
    assert result.select("points_for").item() == 24
    assert result.select("points_allowed").item() == 17


def test_load_schedule_filters_regular_season(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = pl.DataFrame(
        {
            "game_type": ["REG", "POST"],
            "home_team": ["DEN", "DEN"],
            "away_team": ["KC", "BUF"],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_schedules", lambda seasons: schedule)

    result = data_loader.load_schedule(2025)

    assert result.height == 1
    assert result.select("away_team").item() == "KC"


def test_load_qb_stats_selects_primary_qb_and_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    qb = pl.DataFrame(
        {
            "season_type": ["REG", "REG", "REG", "POST"],
            "week": [1, 1, 2, 3],
            "team_abbr": ["DEN", "DEN", "KC", "DEN"],
            "attempts": [20, 30, 25, 40],
            "passer_rating": [90.0, 110.0, 95.0, 120.0],
            "avg_time_to_throw": [2.8, 2.5, 2.7, 2.9],
        }
    )

    monkeypatch.setattr(data_loader.nfl, "load_nextgen_stats", lambda seasons, stat_type: qb)

    result = data_loader.load_qb_stats(2025).sort(["team_abbr", "week"])

    assert result.columns == [
        "team_abbr",
        "week",
        "qb_avg_time_to_throw",
        "qb_attempts",
        "qb_passer_rating",
    ]
    assert result.to_dicts() == [
        {
            "team_abbr": "DEN",
            "week": 1,
            "qb_avg_time_to_throw": 2.5,
            "qb_attempts": 30,
            "qb_passer_rating": 110.0,
        },
        {
            "team_abbr": "KC",
            "week": 2,
            "qb_avg_time_to_throw": 2.7,
            "qb_attempts": 25,
            "qb_passer_rating": 95.0,
        },
    ]
