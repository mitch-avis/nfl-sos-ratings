"""FastAPI app serving the analyst UI data contract and the built web app."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from nfl_sos_ratings.config import DATA_DIR
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.refresh_runner import RefreshRunner, RefreshState
from nfl_sos_ratings.ui_data import (
    MissingEntityRowsError,
    MissingSeasonContractError,
    SeasonDataset,
    TablePayload,
    WpRatingsPayload,
    discover_available_seasons,
    load_qb_game_log_payload,
    load_qb_rank_history_payload,
    load_qb_rating_history_payload,
    load_qb_rating_pairs_payload,
    load_qb_rating_ranges_payload,
    load_qb_wp_ratings_payload,
    load_season_ui_dataset,
    load_team_game_log_payload,
    load_team_rank_history_payload,
    load_team_rating_history_payload,
    load_team_rating_pairs_payload,
    load_team_rating_ranges_payload,
    load_team_wp_ratings_payload,
)
from nfl_sos_ratings.wp_filter import MAX_WP_THRESHOLD

# The built single-page app: `cd web && npm run build` writes it here.
DEFAULT_WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
# `--data-dir` reaches the app factory through this variable when `--reload` is on.
DATA_DIR_ENV = "NFL_SOS_DATA_DIR"
# Set to 1 by ``web --reload --allow-refresh`` so the reloading app factory allows refreshes too.
ALLOW_REFRESH_ENV = "NFL_SOS_ALLOW_REFRESH"
REPO_ROOT = Path(__file__).resolve().parents[1]
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh-season.sh"
# The app sends this header with every refresh request; a cross-site form or image cannot.
REFRESH_HEADER = "X-Requested-With"
REFRESH_HEADER_VALUE = "nfl-sos-ratings"

# The garbage-time filter's query parameter: a whole percentage, 0 (no filter) to 30.
type WpThreshold = Annotated[
    int,
    Query(
        ge=0,
        le=MAX_WP_THRESHOLD,
        description=(
            "Keep plays whose win probability before the snap was at least this many percent "
            "and at most 100 minus it (0 keeps every play)."
        ),
    ),
]

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


def _rating_pairs_router(data_dir: Path) -> APIRouter:
    """Return the per-entity uncertainty routes: head-to-head pairs and weekly rank ranges."""
    router = APIRouter()

    @router.get("/api/seasons/{season}/teams/{team}/rating-pairs")
    def get_team_rating_pairs(season: int, team: str) -> TablePayload:
        """Return how often one team is rated above each other team across resamples."""
        try:
            return load_team_rating_pairs_payload(data_dir, season, team)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/teams/{team}/rank-history")
    def get_team_rank_history(season: int, team: str) -> TablePayload:
        """Return one team's rank range as of each week of a season in progress."""
        try:
            return load_team_rank_history_payload(data_dir, season, team)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/{qb_id}/rank-history")
    def get_qb_rank_history(season: int, qb_id: str) -> TablePayload:
        """Return one eligible quarterback's rank range as of each week of a season in progress."""
        try:
            return load_qb_rank_history_payload(data_dir, season, qb_id)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/{qb_id}/rating-pairs")
    def get_qb_rating_pairs(season: int, qb_id: str) -> TablePayload:
        """Return how often one quarterback is rated above each other eligible one."""
        try:
            return load_qb_rating_pairs_payload(data_dir, season, qb_id)
        except (MissingSeasonContractError, MissingEntityRowsError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router


def _rating_ranges_router(data_dir: Path) -> APIRouter:
    """Return the league-wide rank-range routes, one payload per season for teams and for QBs."""
    router = APIRouter()

    @router.get("/api/seasons/{season}/teams/rating-ranges")
    def get_team_rating_ranges(season: int) -> TablePayload:
        """Return every team's rating and rank ranges over game-bootstrap resamples."""
        try:
            return load_team_rating_ranges_payload(data_dir, season)
        except MissingSeasonContractError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/rating-ranges")
    def get_qb_rating_ranges(season: int) -> TablePayload:
        """Return the eligible quarterbacks' rating and rank ranges over bootstrap resamples."""
        try:
            return load_qb_rating_ranges_payload(data_dir, season)
        except MissingSeasonContractError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router


def _wp_ratings_router(data_dir: Path) -> APIRouter:
    """Return the garbage-time filter routes: one season's ratings at a chosen threshold."""
    router = APIRouter()

    @router.get("/api/seasons/{season}/teams/wp-ratings")
    def get_team_wp_ratings(season: int, threshold: WpThreshold = 0) -> WpRatingsPayload:
        """Return every team's ratings refit on the plays the filter keeps."""
        try:
            return load_team_wp_ratings_payload(data_dir, season, threshold)
        except MissingSeasonContractError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/api/seasons/{season}/qbs/wp-ratings")
    def get_qb_wp_ratings(season: int, threshold: WpThreshold = 0) -> WpRatingsPayload:
        """Return the qualifying quarterbacks' ratings refit on the dropbacks the filter keeps."""
        try:
            return load_qb_wp_ratings_payload(data_dir, season, threshold)
        except MissingSeasonContractError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router


class RefreshPayload(TypedDict):
    """The refresh button's view of the server: whether refreshing is on, and the runner's state."""

    allowed: bool
    state: RefreshState
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    summary: str | None
    log_tail: list[str]


def refresh_runner_for(data_dir: Path) -> RefreshRunner:
    """Return the runner for ``scripts/refresh-season.sh``, which rebuilds the repository data.

    Raises:
        ValueError: If ``data_dir`` is not the repository's ``data/``, which the script rebuilds; a
            refresh would then change files the server does not serve.

    """
    if data_dir.resolve() != (REPO_ROOT / "data").resolve():
        msg = (
            f"--allow-refresh rebuilds {REPO_ROOT / 'data'}, but the server serves {data_dir}; "
            "serve the repository's data/ to allow refreshes"
        )
        raise ValueError(msg)
    return RefreshRunner([str(REFRESH_SCRIPT)], cwd=REPO_ROOT)


def _same_origin(request: Request) -> bool:
    """Return whether the request has no ``Origin`` header or one naming this server's host."""
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("host")


def _refresh_router(refresh: RefreshRunner | None) -> APIRouter:
    """Return the refresh status and start routes (start is refused unless refreshing is on)."""
    router = APIRouter()

    def payload() -> RefreshPayload:
        """Return the runner's status, or an idle status that says refreshing is off."""
        if refresh is None:
            return {
                "allowed": False,
                "state": "idle",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "summary": None,
                "log_tail": [],
            }
        status = refresh.status()
        return {
            "allowed": True,
            "state": status.state,
            "started_at": status.started_at,
            "finished_at": status.finished_at,
            "exit_code": status.exit_code,
            "summary": status.summary,
            "log_tail": status.log_tail,
        }

    @router.get("/api/refresh")
    def refresh_status() -> RefreshPayload:
        """Return whether refreshing is on and the last or running refresh's progress."""
        return payload()

    @router.post("/api/refresh", status_code=202)
    def start_refresh(request: Request) -> RefreshPayload:
        """Start a refresh of the season in progress, unless one is running."""
        if refresh is None:
            raise HTTPException(
                status_code=403,
                detail="Refreshing is off; start the server with --allow-refresh.",
            )
        if request.headers.get(REFRESH_HEADER) != REFRESH_HEADER_VALUE or not _same_origin(request):
            raise HTTPException(status_code=403, detail="Refreshes start only from this app.")
        if not refresh.start():
            raise HTTPException(status_code=409, detail="A refresh is already running.")
        return payload()

    return router


def create_app(
    data_dir: Path | None = None,
    *,
    web_dist: Path | None = None,
    refresh: RefreshRunner | None = None,
) -> FastAPI:
    """Create the analyst API, plus the built web app from ``web_dist`` (default ``web/dist``).

    ``refresh`` turns on the refresh button's ``POST /api/refresh``; without it, the route refuses.
    """
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
    app.include_router(_rating_ranges_router(resolved_data_dir))
    app.include_router(_rating_pairs_router(resolved_data_dir))
    app.include_router(_wp_ratings_router(resolved_data_dir))
    app.include_router(_refresh_router(refresh))
    mount_frontend(app, web_dist or DEFAULT_WEB_DIST)
    return app


def create_app_from_environment() -> FastAPI:
    """Create the app for ``uvicorn --reload``, reading the data directory from the environment."""
    data_dir = Path(os.environ.get(DATA_DIR_ENV, DATA_DIR))
    allow = os.environ.get(ALLOW_REFRESH_ENV) == "1"
    return create_app(data_dir, refresh=refresh_runner_for(data_dir) if allow else None)


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
    parser.add_argument(
        "--allow-refresh",
        action="store_true",
        help=(
            "Let the app's refresh button run scripts/refresh-season.sh, which rebuilds the season "
            "in progress in the repository's data/ (anyone who can reach the server can start one)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Serve the analyst web app and API with uvicorn."""
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    try:
        refresh = refresh_runner_for(data_dir) if args.allow_refresh else None
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.reload:
        os.environ[DATA_DIR_ENV] = args.data_dir
        os.environ[ALLOW_REFRESH_ENV] = "1" if refresh is not None else "0"
        uvicorn.run(
            "nfl_sos_ratings.ui_api:create_app_from_environment",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return
    uvicorn.run(create_app(data_dir, refresh=refresh), host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = [
    "create_app",
    "create_app_from_environment",
    "main",
    "mount_frontend",
    "refresh_runner_for",
]
