"""FastAPI app serving the analyst UI data contract and the built web app."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from nfl_sos_ratings.config import DATA_DIR
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.ui_data import (
    MissingEntityRowsError,
    MissingSeasonContractError,
    SeasonDataset,
    TablePayload,
    discover_available_seasons,
    load_qb_game_log_payload,
    load_qb_rating_history_payload,
    load_season_ui_dataset,
    load_team_game_log_payload,
    load_team_rating_history_payload,
)

# The built single-page app: `cd web && npm run build` writes it here.
DEFAULT_WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
# `--data-dir` reaches the app factory through this variable when `--reload` is on.
DATA_DIR_ENV = "NFL_SOS_DATA_DIR"

_MISSING_BUILD_HTML = """<!doctype html><title>nfl-sos-ratings</title>
<h1>Frontend not built</h1>
<p>The API is running, but <code>web/dist</code> does not exist. Run <code>npm run build</code>
inside <code>web/</code> (or <code>npm run dev</code> for the dev server).</p>
"""


def _safe_child(base: Path, relative: str) -> Path | None:
    """Return ``base/relative`` when it stays inside ``base``, else ``None``."""
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Serve the built single-page app from ``dist_dir`` with a client-side-routing fallback.

    Any path outside ``/api`` returns the matching file from ``dist_dir`` when it exists and
    ``index.html`` otherwise, so deep links work. A path under ``/api`` that no API route
    matched gets the API's JSON 404, never the app. Without ``index.html`` a short page explains
    how to build the app, with status 503.
    """
    assets = dist_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    router = APIRouter()

    @router.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        """Serve a built file or fall back to ``index.html``."""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"No API route for /{path}")
        index = dist_dir / "index.html"
        if not index.is_file():
            return HTMLResponse(_MISSING_BUILD_HTML, status_code=503)
        candidate = _safe_child(dist_dir, path) if path else None
        if candidate is not None and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)

    app.include_router(router)


def _entity_router(data_dir: Path) -> APIRouter:
    """Return the per-team and per-QB routes: game logs and rating history for one season."""
    router = APIRouter()

    @router.get("/api/seasons/{season}/teams/{team}/game-logs")
    def get_team_game_logs(season: int, team: str) -> TablePayload:
        """Return additive team game logs for one team and season."""
        try:
            return load_team_game_log_payload(data_dir, season, team)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/{qb_id}/game-logs")
    def get_qb_game_logs(season: int, qb_id: str) -> TablePayload:
        """Return additive QB game logs for one quarterback and season."""
        try:
            return load_qb_game_log_payload(data_dir, season, qb_id)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/teams/{team}/rating-history")
    def get_team_rating_history(season: int, team: str) -> TablePayload:
        """Return one team's ratings as of each week, each fit on the games through that week."""
        try:
            return load_team_rating_history_payload(data_dir, season, team)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/{qb_id}/rating-history")
    def get_qb_rating_history(season: int, qb_id: str) -> TablePayload:
        """Return one quarterback's rating as of each week, fit on the games through that week."""
        try:
            return load_qb_rating_history_payload(data_dir, season, qb_id)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router


def create_app(data_dir: Path | None = None, *, web_dist: Path | None = None) -> FastAPI:
    """Create the analyst API, plus the built web app from ``web_dist`` (default ``web/dist``)."""
    resolved_data_dir = data_dir or Path(DATA_DIR)
    app = FastAPI(
        title="NFL SOS Ratings UI API",
        summary="Parquet-backed data service for the analyst-facing local UI.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^http://(localhost|127\.0\.0\.1|0\.0\.0\.0|\d{1,3}(?:\.\d{1,3}){3}):\d+$"
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Return a basic health payload for local development."""
        return {"status": "ok"}

    @app.get("/api/metadata")
    def get_metadata() -> dict[str, object]:
        """Return the metric registry: categories and metrics."""
        return get_registry().payload()

    @app.get("/api/seasons")
    def list_seasons() -> dict[str, list[int]]:
        """List seasons with a complete first-pass UI contract."""
        return {"seasons": discover_available_seasons(resolved_data_dir)}

    @app.get("/api/seasons/{season}")
    def get_season(season: int) -> SeasonDataset:
        """Return the normalized analyst UI dataset for one season."""
        try:
            return load_season_ui_dataset(resolved_data_dir, season)
        except MissingSeasonContractError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    app.include_router(_entity_router(resolved_data_dir))
    mount_frontend(app, web_dist or DEFAULT_WEB_DIST)
    return app


def create_app_from_environment() -> FastAPI:
    """Create the app for ``uvicorn --reload``, reading the data directory from the environment."""
    return create_app(Path(os.environ.get(DATA_DIR_ENV, DATA_DIR)))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``web`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings web",
        description="Serve the analyst web app (built into web/dist) and its API.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})."
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})."
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs to serve (default: {DATA_DIR})."
    )
    parser.add_argument(
        "--reload", action="store_true", help="Restart on Python file changes (development)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Serve the analyst web app and API with uvicorn."""
    args = _parse_args(argv)
    if args.reload:
        os.environ[DATA_DIR_ENV] = args.data_dir
        uvicorn.run(
            "nfl_sos_ratings.ui_api:create_app_from_environment",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return
    uvicorn.run(create_app(Path(args.data_dir)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["create_app", "create_app_from_environment", "main", "mount_frontend"]
