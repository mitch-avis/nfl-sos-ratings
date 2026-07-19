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
            "point_margin": [7, -7],
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
    return pl.DataFrame(
        {
            "team": ["DEN"],
            "SaCR": [1.0],
            "SaOR": [0.8],
            "SaDR": [0.6],
            "SaSTR": [0.2],
            "SaOvR": [0.7],
        }
    )


def _srs_df() -> pl.DataFrame:
    return pl.DataFrame({"team": ["DEN"], "srs_rating": [1.2]})


def _team_adjustments_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["DEN", "KC"],
            "adj_off_points_per_offensive_snap": [0.15, 0.10],
            "adj_def_points_allowed_per_defensive_snap": [0.20, 0.12],
            "adj_def_rushing_epa_per_offensive_snap": [0.05, 0.30],
            "st_rating": [0.25, 0.10],
        }
    )


def _qb_adjustments_df() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame({"qb_id": ["DEN"], "adj_qb_epa_per_dropback": [0.12]}),
        pl.DataFrame({"team": ["KC"], "adj_def_qb_epa_per_dropback": [-0.05]}),
    )


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
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "SEASON", 2025)
    monkeypatch.setattr(main, "load_weekly_team_stats", lambda season: _weekly_df())
    monkeypatch.setattr(main, "load_schedule", lambda season: _schedule_df())
    monkeypatch.setattr(main, "load_qb_stats", lambda season: _qb_df())
    monkeypatch.setattr(main, "compute_all_teams_per_game", lambda weekly_df: _team_per_game())
    monkeypatch.setattr(main, "compute_win_totals", lambda weekly_df: _win_totals())
    monkeypatch.setattr(main, "compute_ratings", lambda combined, **kwargs: _ratings_df())
    monkeypatch.setattr(main, "solve_srs", lambda weekly_df, response_col: _srs_df())
    monkeypatch.setattr(main, "load_pbp_data", lambda season: pl.DataFrame({"week": [1]}))
    monkeypatch.setattr(
        main,
        "compute_team_adjusted_stats",
        lambda weekly_df, response_cols, ridge_lambda=1.0: _team_adjustments_df(),
    )
    monkeypatch.setattr(
        main,
        "build_play_level_team_frame_from_pbp",
        lambda pbp_df: pl.DataFrame({"team": ["DEN"], "week": [1]}),
    )
    monkeypatch.setattr(
        main,
        "build_play_level_team_adjusted_snapshot",
        lambda play_rows, cutoff_week: _team_adjustments_df(),
    )
    monkeypatch.setattr(
        main,
        "build_special_teams_game_frame_from_pbp",
        lambda pbp_df: pl.DataFrame({"team": ["DEN"], "week": [1]}),
    )
    monkeypatch.setattr(
        main,
        "build_special_teams_rating_snapshot",
        lambda st_game_rows, cutoff_week: pl.DataFrame({"team": ["DEN"], "st_rating": [0.25]}),
    )
    monkeypatch.setattr(
        main,
        "compute_qb_adjusted_stats",
        lambda qb_games, response_cols, ridge_lambda=1.0: _qb_adjustments_df(),
    )
    monkeypatch.setattr(
        main,
        "compute_qb_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df, qb_season_df: (_qb_opp_profiles(), {"DEN": []}),
    )
    monkeypatch.setattr(main, "compute_qb_ratings", lambda qb_combined, **kwargs: _qb_ratings_df())
    monkeypatch.setattr(
        main,
        "_build_team_schedule_strength",
        lambda team_games, team_ratings: pl.DataFrame({"team": ["DEN"], "sos": [1.0]}),
    )
    monkeypatch.setattr(
        main,
        "_build_qb_faced_overall_quality",
        lambda qb_games, team_ratings: pl.DataFrame({"team": ["DEN"], "faced_opp_SaCR": [1.0]}),
    )


def test_main_uses_current_season_scaling_for_team_and_qb_ratings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Published ratings should be scaled from the current season only."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame({"team": ["DEN"], "points_for": [20.0]}),
            pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [90.0]}),
            {},
        ),
    )

    pl.read_csv(
        io.BytesIO(b"team,points_for,points_per_offensive_snap\nKC,30,0.41\n")
    ).write_parquet(tmp_path / "2024_combined.parquet")
    pl.read_csv(
        io.BytesIO(
            b"team,qb_id,qb_name,qb_epa_per_dropback,qb_any_a,"
            b"qb_completion_percentage_above_expectation,"
            b"qb_td_int_margin_rate,qb_sack_rate,qb_pass_yards_per_dropback,qb_passer_rating\n"
            b"KC,qb-kc,Patrick Mahomes,0.22,7.6,3.1,0.06,0.04,7.1,103.0\n"
        )
    ).write_parquet(tmp_path / "2024_qb_combined.parquet")

    captured: dict[str, object] = {}

    def capture_team_reference(combined: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
        captured["team_reference_df"] = kwargs.get("reference_df")
        return _ratings_df()

    def capture_qb_reference(qb_combined: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
        captured["qb_reference_df"] = kwargs.get("reference_df")
        return _qb_ratings_df()

    monkeypatch.setattr(main, "compute_ratings", capture_team_reference)
    monkeypatch.setattr(main, "compute_qb_ratings", capture_qb_reference)

    main.main()

    assert captured["team_reference_df"] is None
    assert captured["qb_reference_df"] is None


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

    assert (tmp_path / f"{main.SEASON}_team_per_game_stats.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_per_game_stats.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_team_game_logs.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_game_logs.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_opponent_profiles.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_combined.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_ratings.parquet").exists()
    assert not (tmp_path / f"{main.SEASON}_combined.parquet").exists()
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

    combined = pl.read_parquet(tmp_path / f"{main.SEASON}_combined.parquet")
    qb_combined = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_combined.parquet")
    assert combined.select("diff_points_for").item() == 4.0
    assert combined.select("diff_qb_passer_rating").item() == 10.0
    assert combined.select("SaSTR").item() == 0.2
    assert combined.select("SaOvR").item() == 0.7
    assert combined.select("SRS").item() == 1.2
    assert combined.select("adj_off_points_per_offensive_snap").item() == 0.15
    assert combined.select("adj_def_points_allowed_per_defensive_snap").item() == 0.2
    assert qb_combined.select("qopp_points_allowed").item() == 19.0
    assert qb_combined.select("diff_qb_passer_rating").item() == 9.0
    assert qb_combined.select("QSaCR_pct").item() == 75.0
    assert qb_combined.select("adj_qb_epa_per_dropback").item() == 0.12
    assert qb_combined.select("adj_def_qb_epa_per_dropback_faced").item() == -0.05
    team_game_logs = pl.read_parquet(tmp_path / f"{main.SEASON}_team_game_logs.parquet")
    qb_game_logs = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_game_logs.parquet")
    assert team_game_logs.filter(pl.col("team") == "DEN").select("opponent_team").item() == "KC"
    assert qb_game_logs.filter(pl.col("team") == "DEN").select("opponent_team").item() == "KC"
    assert (tmp_path / f"{main.SEASON}_ratings.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_qb_ratings.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_simultaneous_team_adjustments.parquet").exists()
    assert (tmp_path / f"{main.SEASON}_simultaneous_qb_adjustments.parquet").exists()

    ratings_summary = pl.read_parquet(tmp_path / f"{main.SEASON}_ratings.parquet")
    assert ratings_summary.select("SaSTR").item() == 0.2
    assert ratings_summary.select("SRS").item() == 1.2
    assert "KC (DIV): 1 games" in capsys.readouterr().out


def test_main_preserves_distinct_opponent_per_game_and_per_play_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Team and QB opponent outputs keep per-game-like and per-play-like series distinct."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame(
                {
                    "team": ["DEN"],
                    "points_for": [20.0],
                    "points_per_offensive_snap": [0.33],
                }
            ),
            None,
            {},
        ),
    )
    monkeypatch.setattr(
        main,
        "compute_qb_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df, qb_season_df: (
            pl.DataFrame(
                {
                    "team": ["DEN"],
                    "qopp_points_allowed": [19.0],
                    "qopp_qb_pass_yards": [240.0],
                    "qopp_qb_pass_yards_per_dropback": [5.9],
                }
            ),
            {"DEN": []},
        ),
    )

    main.main()

    team_combined = pl.read_parquet(tmp_path / f"{main.SEASON}_combined.parquet")
    qb_combined = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_combined.parquet")

    assert team_combined.select("opp_points_for").item() == 20.0
    assert team_combined.select("opp_points_per_offensive_snap").item() == 0.33
    assert (
        team_combined.select("opp_points_for").item()
        != team_combined.select("opp_points_per_offensive_snap").item()
    )

    assert qb_combined.select("qopp_qb_pass_yards").item() == 240.0
    assert qb_combined.select("qopp_qb_pass_yards_per_dropback").item() == 5.9
    assert (
        qb_combined.select("qopp_qb_pass_yards").item()
        != qb_combined.select("qopp_qb_pass_yards_per_dropback").item()
    )


def test_main_writes_team_and_qb_schedule_context_companions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Main should publish team and QB overall-opponent schedule context columns."""
    original_team_sos = main._build_team_schedule_strength
    original_qb_overall = main._build_qb_faced_overall_quality
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "load_qb_stats",
        lambda season: pl.DataFrame(
            {
                "qb_id": ["qb-1"],
                "qb_name": ["QB One"],
                "team_abbr": ["DEN"],
                "week": [1],
                "qb_passer_rating": [100.0],
            }
        ),
    )
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_qb_season_stats",
        lambda qb_df, weekly_df=None: pl.DataFrame(
            {
                "qb_id": ["qb-1"],
                "qb_name": ["QB One"],
                "team": ["DEN"],
                "qb_is_eligible": [True],
                "qb_attempts_total": [10],
                "qb_win_pct": [1.0],
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "compute_all_opponent_profiles",
        lambda weekly_df, qb_df, schedule_df: (
            pl.DataFrame({"team": ["DEN"], "points_for": [20.0]}),
            pl.DataFrame({"team": ["DEN"], "qb_passer_rating": [90.0]}),
            {},
        ),
    )
    monkeypatch.setattr(
        main,
        "compute_ratings",
        lambda combined, **kwargs: pl.DataFrame(
            {
                "team": ["DEN", "KC"],
                "SaCR": [0.5, 1.0],
                "SaOR": [0.8, 0.2],
                "SaDR": [0.6, 0.1],
                "SaSTR": [0.2, 0.0],
                "SaOvR": [0.7, 0.3],
            }
        ),
    )
    monkeypatch.setattr(main, "_build_team_schedule_strength", original_team_sos)
    monkeypatch.setattr(main, "_build_qb_faced_overall_quality", original_qb_overall)

    main.main()

    combined = pl.read_parquet(tmp_path / f"{main.SEASON}_combined.parquet")
    ratings = pl.read_parquet(tmp_path / f"{main.SEASON}_ratings.parquet")
    qb_combined = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_combined.parquet")
    qb_ratings = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_ratings.parquet")

    assert combined.filter(pl.col("team") == "DEN").select("sos").item() == pytest.approx(1.0)
    assert ratings.filter(pl.col("team") == "DEN").select("sos").item() == pytest.approx(1.0)
    assert qb_combined.select("faced_opp_SaCR").item() == pytest.approx(1.0)
    assert qb_ratings.select("faced_opp_SaCR").item() == pytest.approx(1.0)


def test_build_qb_faced_defense_adjustments_weights_by_dropbacks() -> None:
    """Faced-defense schedule should weight each opponent by the QB's dropback volume."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB1", "QB1"],
            "qb_name": ["QB One", "QB One"],
            "team_abbr": ["DEN", "DEN"],
            "opponent_team": ["KC", "BUF"],
            "qb_dropbacks": [40.0, 10.0],
        }
    )
    defense_adjustments = pl.DataFrame(
        {
            "team": ["KC", "BUF"],
            "adj_def_qb_epa_per_dropback": [0.30, -0.10],
        }
    )

    faced_defense = main._build_qb_faced_defense_adjustments(qb_games, defense_adjustments)

    assert faced_defense.select("adj_def_qb_epa_per_dropback_faced").item() == pytest.approx(0.22)


def test_build_qb_faced_defense_adjustments_aggregates_multi_team_qb_seasons() -> None:
    """Season-level faced-defense context should cover every game a QB played across teams."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB1", "QB1", "QB1"],
            "qb_name": ["QB One", "QB One", "QB One"],
            "team_abbr": ["CLE", "CLE", "CIN"],
            "opponent_team": ["BAL", "GB", "PIT"],
            "qb_dropbacks": [20.0, 30.0, 50.0],
        }
    )
    defense_adjustments = pl.DataFrame(
        {
            "team": ["BAL", "GB", "PIT"],
            "adj_def_qb_epa_per_dropback": [-0.10, 0.20, -0.05],
        }
    )

    faced_defense = main._build_qb_faced_defense_adjustments(qb_games, defense_adjustments)

    assert faced_defense.height == 1
    assert faced_defense.select("qb_id").item() == "QB1"
    assert faced_defense.select("adj_def_qb_epa_per_dropback_faced").item() == pytest.approx(0.015)


def test_build_qb_faced_defense_adjustments_normalizes_team_aliases() -> None:
    """Opponent joins should use the shared team-alias normalization before weighting."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB1", "QB1"],
            "qb_name": ["QB One", "QB One"],
            "team_abbr": ["DEN", "DEN"],
            "opponent_team": ["LA", "BUF"],
            "qb_dropbacks": [30.0, 10.0],
        }
    )
    defense_adjustments = pl.DataFrame(
        {
            "team": ["LAR", "BUF"],
            "adj_def_qb_epa_per_dropback": [0.30, -0.10],
        }
    )

    faced_defense = main._build_qb_faced_defense_adjustments(qb_games, defense_adjustments)

    assert faced_defense.select("adj_def_qb_epa_per_dropback_faced").item() == pytest.approx(0.2)


def test_build_qb_faced_defense_adjustments_rejects_unmatched_opponents() -> None:
    """Unmatched opponent rows should fail loudly instead of diluting schedule strength."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB1"],
            "qb_name": ["QB One"],
            "team_abbr": ["DEN"],
            "opponent_team": ["XXX"],
            "qb_dropbacks": [25.0],
        }
    )
    defense_adjustments = pl.DataFrame(
        {
            "team": ["BUF"],
            "adj_def_qb_epa_per_dropback": [-0.10],
        }
    )

    with pytest.raises(ValueError, match="Missing QB defense adjustments"):
        main._build_qb_faced_defense_adjustments(qb_games, defense_adjustments)


def test_build_qb_faced_rush_defense_adjustments_weights_by_designed_carries() -> None:
    """Faced rush-defense context should weight opponents by designed carries."""
    qb_games = pl.DataFrame(
        {
            "qb_id": ["QB1", "QB1"],
            "qb_name": ["QB One", "QB One"],
            "team_abbr": ["DEN", "DEN"],
            "opponent_team": ["KC", "BUF"],
            "qb_designed_carries": [8.0, 2.0],
        }
    )
    team_adjustments = pl.DataFrame(
        {
            "team": ["KC", "BUF"],
            "adj_def_rushing_epa_per_offensive_snap": [0.30, -0.10],
        }
    )

    faced_rush_defense = main._build_qb_faced_rush_defense_adjustments(qb_games, team_adjustments)

    assert faced_rush_defense.select(
        "adj_def_rushing_epa_per_offensive_snap_faced"
    ).item() == pytest.approx(0.22)


def test_build_qb_adjusted_designed_rush_surface_uses_faced_rush_defense() -> None:
    """Harder run-defense schedules should lift the adjusted designed-rush value surface."""
    qb_combined = pl.DataFrame(
        {
            "qb_id": ["HARD", "SOFT"],
            "qb_name": ["Hard Runner", "Soft Runner"],
            "team": ["A", "B"],
            "qb_designed_epa_per_carry": [0.20, 0.20],
        }
    )
    faced_rush_defense = pl.DataFrame(
        {
            "qb_id": ["HARD", "SOFT"],
            "qb_name": ["Hard Runner", "Soft Runner"],
            "adj_def_rushing_epa_per_offensive_snap_faced": [0.30, -0.10],
        }
    )

    adjusted = main._build_qb_adjusted_designed_rush_surface(qb_combined, faced_rush_defense)

    assert adjusted.filter(pl.col("qb_id") == "HARD").select(
        "adj_qb_designed_rush_epa_per_carry"
    ).item() == pytest.approx(0.50)
    assert adjusted.filter(pl.col("qb_id") == "SOFT").select(
        "adj_qb_designed_rush_epa_per_carry"
    ).item() == pytest.approx(0.10)


def test_main_writes_qb_designed_rush_context_companions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Main should publish the adjusted designed-rush value and faced rush-defense lens."""
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "load_qb_stats",
        lambda season: pl.DataFrame(
            {
                "qb_id": ["qb-1"],
                "qb_name": ["QB One"],
                "team_abbr": ["DEN"],
                "week": [1],
                "qb_passer_rating": [100.0],
                "qb_designed_carries": [4],
                "qb_designed_rush_epa": [1.7],
                "qb_designed_epa_per_carry": [0.425],
            }
        ),
    )
    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    monkeypatch.setattr(
        main,
        "compute_qb_season_stats",
        lambda qb_df, weekly_df=None: pl.DataFrame(
            {
                "qb_id": ["qb-1"],
                "qb_name": ["QB One"],
                "team": ["DEN"],
                "qb_is_eligible": [True],
                "qb_attempts_total": [10],
                "qb_win_pct": [1.0],
                "qb_designed_carries_total": [4],
                "qb_designed_epa_per_carry": [0.425],
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "compute_qb_adjusted_stats",
        lambda qb_games, response_cols, ridge_lambda=1.0: (
            pl.DataFrame({"qb_id": ["qb-1"], "adj_qb_epa_per_dropback": [0.12]}),
            pl.DataFrame({"team": ["KC"], "adj_def_qb_epa_per_dropback": [-0.05]}),
        ),
    )
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

    qb_combined = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_combined.parquet")
    qb_ratings = pl.read_parquet(tmp_path / f"{main.SEASON}_qb_ratings.parquet")

    assert qb_combined.select(
        "adj_def_rushing_epa_per_offensive_snap_faced"
    ).item() == pytest.approx(0.30)
    assert qb_combined.select("adj_qb_designed_rush_epa_per_carry").item() == pytest.approx(0.725)
    assert qb_ratings.select(
        "adj_def_rushing_epa_per_offensive_snap_faced"
    ).item() == pytest.approx(0.30)
    assert qb_ratings.select("adj_qb_designed_rush_epa_per_carry").item() == pytest.approx(0.725)


def test_main_skips_historical_qb_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify main no longer runs historical QB calibration."""
    _patch_common(monkeypatch, tmp_path)

    monkeypatch.setattr(main, "compute_all_teams_qb_per_game", lambda qb_df: _qb_per_game())
    assert not hasattr(main, "calibrate_qb_model")
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
    assert (tmp_path / f"{main.SEASON}_qb_ratings.parquet").exists()


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

    opponents = pl.read_parquet(tmp_path / f"{main.SEASON}_opponent_profiles.parquet")
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

    combined = pl.read_parquet(tmp_path / f"{main.SEASON}_combined.parquet")
    assert "diff_qb_passer_rating" not in combined.columns
