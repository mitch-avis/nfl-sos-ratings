"""Tests for the local analyst UI API."""

import io
from typing import TYPE_CHECKING

import polars as pl
import pytest
from fastapi.testclient import TestClient

from nfl_sos_ratings import ui_api
from nfl_sos_ratings.rating_ranges import TEAM_RANGE_COLUMNS, summarize_rank_ranges
from nfl_sos_ratings.ui_api import create_app

if TYPE_CHECKING:
    from pathlib import Path


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
        "team,points_for,points_per_offensive_snap,opp_points_for,team_rating,offense_rating,defense_rating,special_teams_rating,sos,SRS",
        "DET,510,0.42,390,7.1,4.0,2.6,0.5,0.4,7.4",
    )
    _write_table(data_dir / f"{season}_ratings.parquet", "team,team_rating", "DET,7.1")
    _write_table(
        data_dir / f"{season}_qb_combined.parquet",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback,opp_qb_any_a,adj_qb_epa_per_dropback,qb_faced_pass_defense"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,0.15,0.01",
    )
    _write_table(
        data_dir / f"{season}_qb_ratings.parquet", "player_id,adj_qb_epa_per_dropback", "qb-1,0.15"
    )


def test_list_seasons_returns_complete_contracts_only(tmp_path: Path) -> None:
    """List only seasons with the full backend UI contract present."""
    # Arrange
    _seed_season_contract(tmp_path, 2024)
    _write_table(tmp_path / "2025_combined.parquet", "team,team_rating", "KC,1.0")
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"seasons": [2024]}


def test_get_season_returns_grouped_team_and_qb_tables(tmp_path: Path) -> None:
    """Return the normalized season dataset for the requested UI season."""
    # Arrange
    _seed_season_contract(tmp_path, 2024)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024")

    # Assert
    assert response.status_code == 200
    payload = response.json()
    assert payload["season"] == 2024
    assert payload["teams"]["rows"][0]["team"] == "DET"
    assert payload["qbs"]["rows"][0]["player_display_name"] == "Jared Goff"
    assert payload["teams"]["column_groups"]["ratings"] == [
        "team_rating",
        "offense_rating",
        "defense_rating",
        "special_teams_rating",
        "sos",
        "SRS",
    ]
    assert payload["teams"]["column_groups"]["per_game_rates"] == ["points_for"]
    assert payload["qbs"]["column_groups"]["per_game_rates"] == ["qb_attempts_per_game"]


def test_get_missing_season_returns_not_found(tmp_path: Path) -> None:
    """Translate a missing contract error into a 404 response."""
    # Arrange
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2099")

    # Assert
    assert response.status_code == 404
    assert response.json()["detail"].startswith("Season 2099 is missing UI contract files")


def test_create_app_allows_local_network_frontend_origins(tmp_path: Path) -> None:
    """Allow browser requests from a Vite dev server on the LAN."""
    # Arrange
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get(
        "/api/health",
        headers={"origin": "http://192.168.50.123:5173"},
    )

    # Assert
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://192.168.50.123:5173"


def _seed_game_logs(data_dir: Path) -> None:
    """Write a season contract plus team and QB game logs for two teams."""
    _seed_season_contract(data_dir, 2024)
    _write_table(
        data_dir / "2024_team_game_logs.parquet",
        (
            "game_id,week,team,opponent_team,points_for,points_allowed,point_margin,"
            "points_per_offensive_snap"
        ),
        "g1,1,DET,KC,24,17,7,0.42\ng2,2,DET,CHI,21,20,1,0.35\ng3,1,KC,DET,17,24,-7,0.31",
    )
    _write_table(
        data_dir / "2024_qb_game_logs.parquet",
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


def test_get_team_game_logs_return_the_selected_team_rows(tmp_path: Path) -> None:
    """Serve the team game-log payload for one selected team."""
    # Arrange
    _seed_game_logs(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024/teams/DET/game-logs")

    # Assert
    assert response.status_code == 200
    assert [row["opponent_team"] for row in response.json()["rows"]] == ["KC", "CHI"]
    assert response.json()["column_groups"]["per_snap_rates"] == ["points_per_offensive_snap"]


def test_get_qb_game_logs_return_the_selected_qb_rows(tmp_path: Path) -> None:
    """Serve the QB game-log payload for one selected QB."""
    # Arrange
    _seed_game_logs(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024/qbs/qb-1/game-logs")

    # Assert
    assert response.status_code == 200
    assert [row["week"] for row in response.json()["rows"]] == [1, 2]
    assert response.json()["column_groups"]["per_dropback_rates"] == ["qb_epa_per_dropback"]


def _seed_rating_histories(data_dir: Path) -> None:
    """Write a season contract plus team and QB rating histories."""
    _seed_season_contract(data_dir, 2024)
    _write_table(
        data_dir / "2024_ratings_by_week.parquet",
        "week,team,games_played,team_rating",
        "1,DET,1,1.7\n2,DET,2,4.5\n1,KC,1,-0.4",
    )
    _write_table(
        data_dir / "2024_qb_ratings_by_week.parquet",
        "week,qb_id,qb_games_played,qb_dropbacks,adj_qb_epa_per_dropback",
        "1,qb-1,1,38,0.05\n2,qb-1,2,74,0.09",
    )


def test_get_team_rating_history_returns_the_teams_weeks(tmp_path: Path) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024/teams/DET/rating-history")

    # Assert
    assert response.status_code == 200
    assert [row["team_rating"] for row in response.json()["rows"]] == [1.7, 4.5]


def test_get_qb_rating_history_returns_the_qbs_weeks(tmp_path: Path) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024/qbs/qb-1/rating-history")

    # Assert
    assert response.status_code == 200
    assert response.json()["column_groups"]["ratings"] == ["adj_qb_epa_per_dropback"]


@pytest.mark.parametrize(
    "path",
    [
        "/api/seasons/2024/teams/NOPE/rating-history",
        "/api/seasons/2024/qbs/nope/rating-history",
        "/api/seasons/2023/teams/DET/rating-history",
    ],
)
def test_rating_history_for_an_unknown_entity_or_season_returns_not_found(
    tmp_path: Path, path: str
) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 404


def test_get_metadata_returns_registry_payload(tmp_path: Path) -> None:
    """The metadata endpoint serves the full metric registry."""
    # Arrange
    client = TestClient(create_app(data_dir=tmp_path))

    # Act
    response = client.get("/api/metadata")

    # Assert
    assert response.status_code == 200
    payload = response.json()
    team_categories = [category["name"] for category in payload["entities"]["team"]["categories"]]
    qb_categories = [category["name"] for category in payload["entities"]["qb"]["categories"]]
    assert team_categories[0] == "Schedule-Adjusted Ratings"
    assert qb_categories[0] == "Schedule-Adjusted Ratings"
    assert payload["metrics"]["qb_sack_rate"]["polarity"] == "lower"
    assert "points per game" in payload["metrics"]["team_rating"]["description"]
    assert "pools" not in payload


def _seed_rating_ranges(data_dir: Path) -> None:
    """Write a season contract plus a team rank-range file (no QB file)."""
    _seed_season_contract(data_dir, 2024)
    draws = pl.DataFrame({"draw": [0, 0], "team": ["DET", "KC"], "team_rating": [2.0, 1.0]})
    summarize_rank_ranges(
        draws, draws.select("team", "team_rating"), TEAM_RANGE_COLUMNS
    ).write_parquet(data_dir / "2024_rating_ranges.parquet")


def test_get_team_rating_ranges_returns_every_team(tmp_path: Path) -> None:
    # Arrange
    _seed_rating_ranges(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get("/api/seasons/2024/teams/rating-ranges")

    # Assert
    assert response.status_code == 200
    assert [row["team"] for row in response.json()["rows"]] == ["DET", "KC"]


@pytest.mark.parametrize(
    "path", ["/api/seasons/2024/qbs/rating-ranges", "/api/seasons/2023/teams/rating-ranges"]
)
def test_rating_ranges_without_the_file_return_not_found(tmp_path: Path, path: str) -> None:
    # Arrange
    _seed_rating_ranges(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 404
    assert "rating_ranges.parquet" in response.json()["detail"]


def _write_frontend_build(dist_dir: Path) -> None:
    """Create a minimal built single-page app."""
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist_dir / "assets" / "app.js").write_text("console.log('app');", encoding="utf-8")
    (dist_dir / "favicon.svg").write_text("<svg/>", encoding="utf-8")


@pytest.mark.parametrize("path", ["/", "/teams", "/qbs/00-0033106"])
def test_frontend_routes_fall_back_to_index_html(tmp_path: Path, path: str) -> None:
    # Arrange
    dist_dir = tmp_path / "dist"
    _write_frontend_build(dist_dir)
    client = TestClient(create_app(tmp_path, web_dist=dist_dir))

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert "<div id=root>" in response.text


@pytest.mark.parametrize(
    ("path", "body"),
    [("/assets/app.js", "console.log('app');"), ("/favicon.svg", "<svg/>")],
)
def test_frontend_serves_built_assets_and_root_files(tmp_path: Path, path: str, body: str) -> None:
    # Arrange
    dist_dir = tmp_path / "dist"
    _write_frontend_build(dist_dir)
    client = TestClient(create_app(tmp_path, web_dist=dist_dir))

    # Act
    response = client.get(path)

    # Assert
    assert response.text == body


def test_unknown_api_paths_return_json_404_not_the_app(tmp_path: Path) -> None:
    # Arrange
    dist_dir = tmp_path / "dist"
    _write_frontend_build(dist_dir)
    client = TestClient(create_app(tmp_path, web_dist=dist_dir))

    # Act
    response = client.get("/api/not-a-route")

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_frontend_does_not_serve_files_outside_the_build(tmp_path: Path) -> None:
    # Arrange
    dist_dir = tmp_path / "dist"
    _write_frontend_build(dist_dir)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    client = TestClient(create_app(tmp_path, web_dist=dist_dir))

    # Act
    response = client.get("/..%2Fsecret.txt")

    # Assert
    assert "secret" not in response.text


def test_missing_frontend_build_explains_how_to_build_it(tmp_path: Path) -> None:
    # Arrange
    client = TestClient(create_app(tmp_path, web_dist=tmp_path / "missing"))

    # Act
    response = client.get("/")

    # Assert
    assert response.status_code == 503
    assert "npm run build" in response.text


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("127.0.0.1", 8080, False)),
        (["--host", "0.0.0.0", "--port", "9001"], ("0.0.0.0", 9001, False)),  # noqa: S104 - flag
    ],
)
def test_web_command_starts_uvicorn_with_host_and_port(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: tuple[str, int, bool]
) -> None:
    # Arrange
    calls: list[tuple[str, int, bool]] = []

    def fake_run(app: object, *, host: str, port: int, **kwargs: object) -> None:
        calls.append((host, port, bool(kwargs.get("reload", False))))

    monkeypatch.setattr(ui_api.uvicorn, "run", fake_run)

    # Act
    ui_api.main(argv)

    # Assert
    assert calls == [expected]


@pytest.mark.parametrize(
    "path", ["/api/seasons/2024/teams/NOPE/game-logs", "/api/seasons/2024/qbs/nope/game-logs"]
)
def test_game_logs_for_an_unknown_entity_return_not_found(tmp_path: Path, path: str) -> None:
    # Arrange
    _seed_game_logs(tmp_path)
    client = TestClient(create_app(tmp_path))

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 404


def test_create_app_from_environment_reads_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    _seed_season_contract(tmp_path, 2024)
    monkeypatch.setenv(ui_api.DATA_DIR_ENV, str(tmp_path))
    client = TestClient(ui_api.create_app_from_environment())

    # Act
    response = client.get("/api/seasons")

    # Assert
    assert response.json() == {"seasons": [2024]}


def test_web_command_with_reload_serves_the_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(ui_api.uvicorn, "run", fake_run)
    monkeypatch.delenv(ui_api.DATA_DIR_ENV, raising=False)

    # Act
    ui_api.main(["--reload", "--data-dir", "elsewhere"])

    # Assert
    assert calls == [
        (
            "nfl_sos_ratings.ui_api:create_app_from_environment",
            {"factory": True, "host": "127.0.0.1", "port": 8080, "reload": True},
        )
    ]
    assert ui_api.os.environ[ui_api.DATA_DIR_ENV] == "elsewhere"
