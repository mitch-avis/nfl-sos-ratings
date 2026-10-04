"""End-to-end tests of the single-season pipeline on a small synthetic league."""

import io
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings import main
from nfl_sos_ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    TeamRatingFit,
    fit_team_ratings,
    fit_team_ratings_with_previous_penalties,
)
from tests.stubs import stub

if TYPE_CHECKING:
    from pathlib import Path

_TEAMS = ("BUF", "MIA", "NE", "NYJ")
_STRENGTH = {"BUF": 0.08, "MIA": -0.02, "NE": 0.04, "NYJ": -0.10}


def _games() -> list[tuple[int, str, str, str]]:
    """Return (week, game_id, home, away) for a double round-robin."""
    return [
        (week, f"2025_{week:02d}_{away}_{home}", home, away)
        for week, (home, away) in enumerate(itertools.permutations(_TEAMS, 2), start=1)
    ]


def _weekly_df() -> pl.DataFrame:
    """Return team-game rows with the columns the loaders publish for the rating."""
    rows: list[dict[str, object]] = []
    for week, game_id, home, away in _games():
        margin = round((_STRENGTH[home] - _STRENGTH[away]) * 100) + 2
        for team, opponent, is_home, team_margin in (
            (home, away, True, margin),
            (away, home, False, -margin),
        ):
            epa_per_play = 0.01 + _STRENGTH[team] - _STRENGTH[opponent] / 2 + (week % 3) * 0.01
            points_for = 20 + max(team_margin, 0)
            rows.append(
                {
                    "game_id": game_id,
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "points_for": points_for,
                    "points_allowed": points_for - team_margin,
                    "point_margin": team_margin,
                    "offensive_snaps": 62,
                    "defensive_snaps": 61,
                    "offensive_epa": epa_per_play * 62,
                    "st_plays": 13,
                    "st_epa": 0.05 if is_home else -0.05,
                }
            )
    return pl.DataFrame(rows)


def _qb_df() -> pl.DataFrame:
    """Return one starting passer per team-game."""
    rows: list[dict[str, object]] = []
    for row in _weekly_df().iter_rows(named=True):
        epa = float(row["offensive_epa"]) / 62 + 0.02
        rows.append(
            {
                "game_id": row["game_id"],
                "week": row["week"],
                "team_abbr": row["team"],
                "qb_id": f"qb-{row['team']}",
                "qb_name": f"{row['team']} Passer",
                "qb_dropbacks": 38,
                "qb_attempts": 35,
                "qb_passing_epa": epa * 38,
                "qb_epa_per_dropback": epa,
            }
        )
    return pl.DataFrame(rows)


def _schedule_df() -> pl.DataFrame:
    """Return the schedule matching the synthetic games."""
    return pl.DataFrame(
        {
            "game_id": [game_id for _, game_id, _, _ in _games()],
            "week": [week for week, _, _, _ in _games()],
            "home_team": [home for _, _, home, _ in _games()],
            "away_team": [away for _, _, _, away in _games()],
        }
    )


def _no_profiles() -> tuple[pl.DataFrame | None, dict[str, list[dict[str, str | bool | int]]]]:
    """Stand in for a profile builder that finds no opponent games."""
    return None, {}


@pytest.fixture
def season_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run the real pipeline on the synthetic league with only the loaders patched."""
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_weekly_team_stats", stub(_weekly_df))
    monkeypatch.setattr(main, "load_schedule", stub(_schedule_df))
    monkeypatch.setattr(main, "load_qb_stats", stub(_qb_df))
    main.run_season(2025)
    return tmp_path


def test_build_team_ratings_publishes_columns_in_order() -> None:
    # Arrange
    weekly_df = _weekly_df()

    # Act
    ratings = main.build_team_ratings(weekly_df)

    # Assert
    assert ratings.columns == ["team", "games_played", *TEAM_RATING_COLUMNS, "sos", "SRS"]


def test_build_team_ratings_ranks_the_strongest_team_first() -> None:
    # Arrange
    weekly_df = _weekly_df()

    # Act
    ratings = main.build_team_ratings(weekly_df)

    # Assert
    assert ratings.get_column("team").to_list()[0] == "BUF"


def test_build_qb_game_logs_carries_opponent_and_venue() -> None:
    # Arrange
    weekly_df = _weekly_df()

    # Act
    logs = main._build_qb_game_logs(_qb_df(), weekly_df)

    # Assert
    first = logs.filter((pl.col("team") == "BUF") & (pl.col("week") == 1)).row(0, named=True)
    assert (first["opponent_team"], first["is_home"]) == ("MIA", True)


def test_run_season_writes_the_team_ratings_file(season_outputs: Path) -> None:
    # Act
    ratings = pl.read_parquet(season_outputs / "2025_ratings.parquet")

    # Assert
    assert ratings.height == len(_TEAMS)
    assert set(TEAM_RATING_COLUMNS) <= set(ratings.columns)


def test_run_season_writes_qb_ratings_for_qualified_passers(season_outputs: Path) -> None:
    # Act
    qb_ratings = pl.read_parquet(season_outputs / "2025_qb_ratings.parquet")

    # Assert
    assert qb_ratings.columns == list(main.QB_RATINGS_ORDER)
    assert qb_ratings.height == len(_TEAMS)


def test_run_season_combined_file_carries_the_team_rating(season_outputs: Path) -> None:
    # Act
    combined = pl.read_parquet(season_outputs / "2025_combined.parquet")

    # Assert
    assert combined.get_column("team_rating").null_count() == 0


def test_run_season_writes_the_weekly_team_rating_history(season_outputs: Path) -> None:
    # Act
    history = pl.read_parquet(season_outputs / "2025_ratings_by_week.parquet")

    # Assert
    last_week = history.filter(pl.col("week") == len(_games()))
    assert last_week.height == len(_TEAMS)


def test_run_season_weekly_team_history_ends_at_the_season_rating(season_outputs: Path) -> None:
    # Arrange
    ratings = pl.read_parquet(season_outputs / "2025_ratings.parquet")

    # Act
    history = pl.read_parquet(season_outputs / "2025_ratings_by_week.parquet")

    # Assert
    last_week = history.filter(pl.col("week") == len(_games())).sort("team")
    assert last_week.get_column("team_rating").to_list() == pytest.approx(
        ratings.sort("team").get_column("team_rating").to_list()
    )


def test_run_season_writes_the_weekly_qb_rating_history(season_outputs: Path) -> None:
    # Act
    history = pl.read_parquet(season_outputs / "2025_qb_ratings_by_week.parquet")

    # Assert
    last_week = history.filter(pl.col("week") == len(_games()))
    assert sorted(last_week.get_column("qb_id").to_list()) == [f"qb-{team}" for team in _TEAMS]


def _recording_previous_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> list[TeamRatingFit | None]:
    """Record the previous-season fit ``run_season`` passes to the team fit; behavior is real."""
    seen: list[TeamRatingFit | None] = []

    def recording(game_logs: pl.DataFrame, previous: TeamRatingFit | None) -> TeamRatingFit:
        seen.append(previous)
        return fit_team_ratings_with_previous_penalties(game_logs, previous)

    monkeypatch.setattr(main, "fit_team_ratings_with_previous_penalties", recording)
    return seen


def _patch_loaders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[int]:
    """Point the pipeline at ``tmp_path`` and the synthetic league; return the loaded seasons."""
    loaded: list[int] = []

    def weekly(season: int) -> pl.DataFrame:
        loaded.append(season)
        return _weekly_df()

    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_weekly_team_stats", weekly)
    monkeypatch.setattr(main, "load_schedule", stub(_schedule_df))
    monkeypatch.setattr(main, "load_qb_stats", stub(_qb_df))
    return loaded


def test_run_season_fits_teams_with_the_previous_seasons_penalties(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    _patch_loaders(monkeypatch, tmp_path)
    previous_logs = main._build_team_game_logs(_weekly_df()).with_columns(
        pl.col("offensive_epa") * 1.5
    )
    previous_logs.write_parquet(tmp_path / "2024_team_game_logs.parquet")
    seen = _recording_previous_fits(monkeypatch)

    # Act
    main.run_season(2025)

    # Assert
    previous = seen[0]
    assert previous is not None
    assert previous.ratings.equals(fit_team_ratings(previous_logs).ratings)


def test_run_season_loads_a_previous_season_missing_from_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    loaded = _patch_loaders(monkeypatch, tmp_path)

    # Act
    main.run_season(2025)

    # Assert
    assert sorted(loaded) == [2024, 2025]


def test_run_season_first_play_by_play_season_cross_validates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    loaded = _patch_loaders(monkeypatch, tmp_path)
    seen = _recording_previous_fits(monkeypatch)

    # Act
    main.run_season(1999)

    # Assert
    assert (loaded, seen) == ([1999], [None])


def test_played_schedule_drops_games_without_final_scores() -> None:
    # Arrange
    schedule = pl.DataFrame(
        {
            "home_team": ["BUF", "NE"],
            "away_team": ["MIA", "NYJ"],
            "home_score": [24, None],
            "away_score": [17, None],
        }
    )

    # Act
    played = main.played_schedule(schedule)

    # Assert
    assert played.select("home_team", "away_team").rows() == [("BUF", "MIA")]


def test_played_schedule_keeps_a_schedule_without_score_columns() -> None:
    # Arrange
    schedule = _schedule_df()

    # Act
    played = main.played_schedule(schedule)

    # Assert
    assert played.equals(schedule)


def test_run_season_profiles_opponents_from_played_games_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    unplayed = pl.DataFrame(
        {"game_id": ["2025_18_BUF_PIT"], "week": [18], "home_team": ["PIT"], "away_team": ["BUF"]}
    )
    schedule = pl.concat([_schedule_df(), unplayed]).with_columns(
        pl.when(pl.col("home_team") == "PIT").then(None).otherwise(20).alias("home_score"),
        pl.when(pl.col("home_team") == "PIT").then(None).otherwise(17).alias("away_score"),
    )
    seen: list[pl.DataFrame] = []
    real_profiles = main.compute_all_opponent_profiles

    def recording_profiles(
        weekly_df: pl.DataFrame, schedule_df: pl.DataFrame
    ) -> tuple[pl.DataFrame | None, dict[str, list[dict[str, str | bool | int]]]]:
        seen.append(schedule_df)
        return real_profiles(weekly_df, schedule_df)

    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_weekly_team_stats", stub(_weekly_df))
    monkeypatch.setattr(main, "load_schedule", stub(lambda: schedule))
    monkeypatch.setattr(main, "load_qb_stats", stub(_qb_df))
    monkeypatch.setattr(main, "compute_all_opponent_profiles", recording_profiles)

    # Act
    main.run_season(2025)

    # Assert
    assert "PIT" not in seen[0].get_column("home_team").to_list()


def test_write_data_file_rejects_columns_missing_from_the_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    frame = pl.DataFrame({"team": ["NE"], "not_a_registered_column": [1]})

    # Act & Assert
    with pytest.raises(ValueError, match="not_a_registered_column"):
        main._write_data_file(frame, 2025, "ratings")


def test_run_season_without_opponent_profiles_still_writes_the_ratings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_weekly_team_stats", stub(_weekly_df))
    monkeypatch.setattr(main, "load_schedule", stub(_schedule_df))
    monkeypatch.setattr(main, "load_qb_stats", stub(_qb_df))
    monkeypatch.setattr(main, "compute_qb_opponent_profiles", stub(_no_profiles))
    monkeypatch.setattr(main, "compute_all_opponent_profiles", stub(_no_profiles))

    # Act
    main.run_season(2025)

    # Assert
    assert (tmp_path / "2025_ratings.parquet").exists()
    assert not (tmp_path / "2025_opponent_profiles.parquet").exists()


def test_main_wraps_stdout_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seasons: list[int] = []
    monkeypatch.setattr(main, "run_season", seasons.append)
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(main.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(main.io, "TextIOWrapper", stub(io.StringIO))

    # Act
    main.main(["--season", "2024"])

    # Assert
    assert seasons == [2024]
    assert isinstance(main.sys.stdout, io.StringIO)
