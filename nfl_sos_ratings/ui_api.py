"""FastAPI app for serving the local analyst UI data contract."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nfl_sos_ratings.config import DATA_DIR
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.ui_data import (
    MissingEntityGameLogError,
    MissingSeasonContractError,
    SeasonDataset,
    TablePayload,
    discover_available_seasons,
    load_qb_game_log_payload,
    load_season_ui_dataset,
    load_team_game_log_payload,
)


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create the local analyst UI API application."""
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
        """Return the metric registry: categories, metrics, and rating pools."""
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

    @app.get("/api/seasons/{season}/teams/{team}/game-logs")
    def get_team_game_logs(season: int, team: str) -> TablePayload:
        """Return additive team game logs for one team and season."""
        try:
            return load_team_game_log_payload(resolved_data_dir, season, team)
        except (MissingSeasonContractError, MissingEntityGameLogError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/seasons/{season}/qbs/{qb_id}/game-logs")
    def get_qb_game_logs(season: int, qb_id: str) -> TablePayload:
        """Return additive QB game logs for one quarterback and season."""
        try:
            return load_qb_game_log_payload(resolved_data_dir, season, qb_id)
        except (MissingSeasonContractError, MissingEntityGameLogError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()


def main() -> None:
    """Run the local analyst UI API with Uvicorn."""
    import uvicorn

    uvicorn.run(
        "nfl_sos_ratings.ui_api:app",
        host="0.0.0.0",  # noqa: S104
        port=8080,
        reload=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "create_app", "main"]
