"""Tests for nfl_sos_ratings.main pipeline behavior."""

import io
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from nfl_sos_ratings import main


def _weekly_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "week": [1, 1],
            "points_for": [24, 17],
            "points_allowed": [17, 24],
        }
    )


def _schedule_df() -> pl.DataFrame:
    return pl.DataFrame({"home_team": ["DEN"], "away_team": ["KC"]})


def _qb_df() -> pl.DataFrame:
    return pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_passer_rating": [100.0]})


def _team_per_game() -> pl.DataFrame:
    return pl.DataFrame({"team": ["DEN"], "points_for": [24.0]})


def _qb_per_game() -> pl.DataFrame:
    return pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [100.0]})


def _empty_qb_per_game() -> pl.DataFrame:
    return pl.DataFrame({"team": ["DEN"]})


def _win_totals() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN"],
            "games_played": [1],
            "wins": [1],
            "losses": [0],
            "ties": [0],
            "win_pct": [1.0],
        }
    )


def _ratings_df() -> pl.DataFrame:
    return pl.DataFrame({"team": ["DEN"], "SaCR": [1.0], "SaOR": [0.8], "SaDR": [0.6]})


def _qb_opp_profiles() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN"],
            "qopp_points_allowed": [19.0],
            "qopp_def_sacks": [2.4],
            "qopp_def_interceptions": [0.9],
            "qopp_qb_passer_rating": [91.0],
            "qopp_qb_completion_percentage_above_expectation": [1.2],
        }
    )


def _qb_ratings_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN"],
            "QRaw": [0.5],
            "QSaOR": [0.7],
            "QSoS": [0.4],
            "QSaCR": [0.7],
            "QRaw_pct": [66.7],
            "QSaOR_pct": [75.0],
            "QSoS_pct": [62.5],
            "QSaCR_pct": [75.0],
        }
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "SEASON", 2025)
    monkeypatch.setattr(main, "load_weekly_team_stats", lambda season: _weekly_df())
    monkeypatch.setattr(main, "load_schedule", lambda season: _schedule_df())
    monkeypatch.setattr(main, "load_qb_stats", lambda season: _qb_df())
    monkeypatch.setattr(main, "compute_all_teams_per_game", lambda weekly_df: _team_per_game())
    monkeypatch.setattr(main, "compute_win_totals", lambda weekly_df: _win_totals())
    monkeypatch.setattr(main, "compute_ratings", lambda combined: _ratings_df())
    monkeypatch.setattr(
        main,
        "compute_qb_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df, qb_season_df: (_qb_opp_profiles(), {"DEN": []}),
    )
    monkeypatch.setattr(main, "compute_qb_ratings", lambda qb_combined: _qb_ratings_df())


def test_main_returns_when_no_opponent_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main exits early with warning when opponent profiles are unavailable."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (None, None, {}),
    )

    main.main()

    assert (tmp_path / f"{main.SEASON}_team_per_game_stats.csv").exists()
    assert (tmp_path / f"{main.SEASON}_qb_per_game_stats.csv").exists()
    assert (tmp_path / f"{main.SEASON}_qb_opponent_profiles.csv").exists()
    assert (tmp_path / f"{main.SEASON}_qb_combined.csv").exists()
    assert (tmp_path / f"{main.SEASON}_qb_ratings.csv").exists()
    assert not (tmp_path / f"{main.SEASON}_combined.csv").exists()
    assert "No opponent profile data was computed" in capsys.readouterr().out


def test_main_handles_both_team_and_qb_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main writes combined outputs when both team and QB profiles are present."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame({"team": ["DEN"], "points_for": [20.0]}),
            pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [90.0]}),
            {"DEN": [{"opponent": "KC", "division": True, "games_included": 1}]},
        ),
    )

    main.main()

    combined = pl.read_csv(tmp_path / f"{main.SEASON}_combined.csv")
    qb_combined = pl.read_csv(tmp_path / f"{main.SEASON}_qb_combined.csv")
    assert combined.select("diff_points_for").item() == 4.0
    assert combined.select("diff_qb_passer_rating").item() == 10.0
    assert qb_combined.select("qopp_points_allowed").item() == 19.0
    assert qb_combined.select("diff_qb_passer_rating").item() == 9.0
    assert qb_combined.select("QSaCR_pct").item() == 75.0
    assert (tmp_path / f"{main.SEASON}_ratings.csv").exists()
    assert (tmp_path / f"{main.SEASON}_qb_ratings.csv").exists()
    assert "KC (DIV): 1 games" in capsys.readouterr().out


def test_main_calibrates_qb_model_with_opponent_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify QB calibration receives opponent context and differential columns."""
    _patch_common(monkeypatch, tmp_path)
    captured: dict[str, list[str]] = {}

    def capture_calibration(historical_df: pl.DataFrame) -> tuple[float, float, float]:
        captured["columns"] = historical_df.columns
        return 0.1, 0.25, 0.5

    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(main, "calibrate_qb_model", capture_calibration)
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame({"team": ["DEN"], "points_for": [20.0]}),
            pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [90.0]}),
            {},
        ),
    )

    main.main()

    assert "qopp_qb_passer_rating" in captured["columns"]
    assert "diff_qb_passer_rating" in captured["columns"]


def test_main_handles_team_only_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify main still writes opponent profiles when only team-level profiles are present."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame({"team": ["DEN"], "points_for": [19.0]}),
            None,
            {},
        ),
    )

    main.main()

    opponents = pl.read_csv(tmp_path / f"{main.SEASON}_opponent_profiles.csv")
    assert opponents.columns == ["team", "points_for"]


def test_main_handles_qb_only_profiles_and_windows_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify main handles QB-only profiles and executes Windows UTF-8 stdout path."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _empty_qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            None,
            pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [90.0]}),
            {},
        ),
    )
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(main.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(main.io, "TextIOWrapper", lambda buffer, encoding: io.StringIO())

    main.main()

    combined = pl.read_csv(tmp_path / f"{main.SEASON}_combined.csv")
    assert "diff_qb_passer_rating" not in combined.columns
