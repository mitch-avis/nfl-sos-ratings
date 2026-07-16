"""Tests for the local analyst UI API."""

import io
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from nfl_sos_ratings.ui_api import create_app


def _write_table(path: Path, header: str, row: str) -> None:
    """Write a minimal Parquet file for API contract tests."""
    pl.read_csv(io.StringIO(f"{header}\n{row}\n")).write_parquet(path)


def _seed_season_contract(data_dir: Path, season: int) -> None:
    """Create the first-pass UI Parquet contract for one season."""
    _write_table(
        data_dir / f"{season}_team_per_game_stats.parquet",
        "team,points_for",
        "DET,510",
    )
    _write_table(
        data_dir / f"{season}_qb_per_game_stats.parquet",
        "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,qb_epa_per_dropback",
        "qb-1,Jared Goff,DET,605,35.6,0.18",
    )
    _write_table(
        data_dir / f"{season}_combined.parquet",
        "team,points_for,points_per_offensive_snap,opp_points_for,SaCR,SaOR,SaDR,SaSTR,SaOvR,SRS",
        "DET,510,0.42,390,1.2,1.1,0.9,0.3,1.0,7.4",
    )
    _write_table(data_dir / f"{season}_ratings.parquet", "team,SaCR,SaSTR", "DET,1.2,0.3")
    _write_table(
        data_dir / f"{season}_qb_combined.parquet",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback,opp_qb_any_a,QRaw,QSoS,QSaOR,QOutcome,QSaCR"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,1.0,0.2,1.1,0.4,1.3",
    )
    _write_table(data_dir / f"{season}_qb_ratings.parquet", "player_id,QSaCR", "qb-1,1.3")


def test_list_seasons_returns_complete_contracts_only(tmp_path: Path) -> None:
    """List only seasons with the full backend UI contract present."""
    _seed_season_contract(tmp_path, 2024)
    _write_table(tmp_path / "2025_combined.parquet", "team,SaCR", "KC,1.0")

    client = TestClient(create_app(tmp_path))

    response = client.get("/api/seasons")

    assert response.status_code == 200
    assert response.json() == {"seasons": [2024]}


def test_get_season_returns_grouped_team_and_qb_tables(tmp_path: Path) -> None:
    """Return the normalized season dataset for the requested UI season."""
    _seed_season_contract(tmp_path, 2024)

    client = TestClient(create_app(tmp_path))

    response = client.get("/api/seasons/2024")

    assert response.status_code == 200
    payload = response.json()
    assert payload["season"] == 2024
    assert payload["teams"]["rows"][0]["team"] == "DET"
    assert payload["qbs"]["rows"][0]["player_display_name"] == "Jared Goff"
    assert payload["teams"]["column_groups"]["ratings"] == [
        "SaCR",
        "SaOR",
        "SaDR",
        "SaSTR",
        "SaOvR",
        "SRS",
    ]
    assert payload["teams"]["column_groups"]["per_game_rates"] == ["points_for"]
    assert payload["qbs"]["column_groups"]["per_game_rates"] == ["qb_attempts_per_game"]


def test_get_missing_season_returns_not_found(tmp_path: Path) -> None:
    """Translate a missing contract error into a 404 response."""
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/seasons/2099")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("Season 2099 is missing UI contract files")


def test_create_app_allows_local_network_frontend_origins(tmp_path: Path) -> None:
    """Allow browser requests from a Vite dev server on the LAN."""
    client = TestClient(create_app(tmp_path))

    response = client.get(
        "/api/health",
        headers={"origin": "http://192.168.50.123:5173"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://192.168.50.123:5173"


def test_get_team_and_qb_game_logs_return_entity_specific_rows(tmp_path: Path) -> None:
    """Serve additive team and QB game-log payloads for one selected entity."""
    _seed_season_contract(tmp_path, 2024)
    _write_table(
        tmp_path / "2024_team_game_logs.parquet",
        (
            "game_id,week,team,opponent_team,points_for,points_allowed,point_margin,"
            "points_per_offensive_snap"
        ),
        "g1,1,DET,KC,24,17,7,0.42\ng2,2,DET,CHI,21,20,1,0.35\ng3,1,KC,DET,17,24,-7,0.31",
    )
    _write_table(
        tmp_path / "2024_qb_game_logs.parquet",
        (
            "game_id,week,team,opponent_team,qb_id,qb_name,qb_attempts,qb_pass_yards,"
            "qb_epa_per_dropback,qb_game_winning_drive"
        ),
        (
            "g1,1,DET,KC,qb-1,Jared Goff,34,280,0.18,1\n"
            "g2,2,DET,CHI,qb-1,Jared Goff,29,244,0.09,0\n"
            "g3,1,KC,DET,qb-2,Patrick Mahomes,37,305,0.21,1"
        ),
    )

    client = TestClient(create_app(tmp_path))

    team_response = client.get("/api/seasons/2024/teams/DET/game-logs")
    qb_response = client.get("/api/seasons/2024/qbs/qb-1/game-logs")

    assert team_response.status_code == 200
    assert [row["opponent_team"] for row in team_response.json()["rows"]] == ["KC", "CHI"]
    assert team_response.json()["column_groups"]["per_snap_rates"] == ["points_per_offensive_snap"]

    assert qb_response.status_code == 200
    assert [row["week"] for row in qb_response.json()["rows"]] == [1, 2]
    assert qb_response.json()["column_groups"]["per_dropback_rates"] == ["qb_epa_per_dropback"]


def test_get_metadata_returns_registry_payload(tmp_path: Path) -> None:
    """The metadata endpoint serves the full metric registry."""
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/metadata")

    assert response.status_code == 200
    payload = response.json()
    team_categories = [category["name"] for category in payload["entities"]["team"]["categories"]]
    qb_categories = [category["name"] for category in payload["entities"]["qb"]["categories"]]
    assert team_categories[0] == "Schedule-Adjusted Ratings"
    assert qb_categories[0] == "Schedule-Adjusted Ratings"
    assert payload["metrics"]["qb_sack_rate"]["polarity"] == "lower"
    assert "qb_primary" in payload["pools"]
