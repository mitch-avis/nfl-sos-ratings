"""Parquet-backed data contract helpers for the local analyst UI."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import polars as pl

from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.metrics.schema import Entity

SEASON_FILE_RE = re.compile(r"^(?P<season>\d{4})_(?P<suffix>[a-z0-9_]+)\.parquet$")
REQUIRED_CONTRACT_SUFFIXES = (
    "team_per_game_stats",
    "qb_per_game_stats",
    "combined",
    "qb_combined",
    "ratings",
    "qb_ratings",
)
TEAM_GAME_LOG_SUFFIX = "team_game_logs"
QB_GAME_LOG_SUFFIX = "qb_game_logs"
TEAM_RATING_COLUMNS = ("SaCR", "SaOR", "SaDR", "SaSTR", "SaOvR", "SRS", "sos")
QB_RATING_COLUMNS = ("QRaw", "QSoS", "faced_opp_SaCR", "QSaOR", "QOutcome", "QSaCR")
TEAM_EXCLUDED_PREFIXES = ("qb_", "opp_qb_", "Q", "diff_", "adj_")
QB_EXCLUDED_PREFIXES = ("diff_", "adj_")
QB_PER_DROPBACK_RATE_COLUMNS = (
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
)


class MissingSeasonContractError(FileNotFoundError):
    """Raised when a requested season does not have the complete UI contract."""


class MissingEntityGameLogError(LookupError):
    """Raised when a requested entity does not have game-log rows in the UI contract."""


class TablePayload(TypedDict):
    """Normalized table payload for a UI index view."""

    rows: list[dict[str, object]]
    visible_columns: list[str]
    column_groups: dict[str, list[str]]
    column_metadata: dict[str, dict[str, object]]


class SeasonDataset(TypedDict):
    """Normalized season payload for the analyst UI."""

    season: int
    teams: TablePayload
    qbs: TablePayload


def discover_available_seasons(data_dir: Path) -> list[int]:
    """Return seasons that have the complete first-pass UI Parquet contract."""
    discovered: dict[int, set[str]] = {}
    for file_path in data_dir.glob("*.parquet"):
        match = SEASON_FILE_RE.match(file_path.name)
        if match is None:
            continue
        season = int(match.group("season"))
        discovered.setdefault(season, set()).add(match.group("suffix"))

    available = [
        season
        for season, suffixes in discovered.items()
        if all(required in suffixes for required in REQUIRED_CONTRACT_SUFFIXES)
    ]
    return sorted(available, reverse=True)


def load_season_ui_dataset(data_dir: Path, season: int) -> SeasonDataset:
    """Load one season of normalized index data for the local analyst UI."""
    contract_paths = _build_contract_paths(data_dir, season)
    _validate_contract_paths(contract_paths, season)

    team_frame = pl.read_parquet(contract_paths["combined"])
    qb_frame = pl.read_parquet(contract_paths["qb_combined"])

    return {
        "season": season,
        "teams": _build_team_payload(team_frame),
        "qbs": _build_qb_payload(qb_frame),
    }


def load_team_game_log_payload(data_dir: Path, season: int, team: str) -> TablePayload:
    """Load additive team game logs for one team and season."""
    frame = _load_game_log_frame(data_dir, season, TEAM_GAME_LOG_SUFFIX)
    filtered = _filter_entity_game_logs(frame, "team", team, season, "team")
    return _build_team_game_log_payload(filtered)


def load_qb_game_log_payload(data_dir: Path, season: int, qb_id: str) -> TablePayload:
    """Load additive quarterback game logs for one QB and season."""
    frame = _load_game_log_frame(data_dir, season, QB_GAME_LOG_SUFFIX)
    filtered = _filter_entity_game_logs(frame, "qb_id", qb_id, season, "QB")
    return _build_qb_game_log_payload(filtered)


def _build_contract_paths(data_dir: Path, season: int) -> dict[str, Path]:
    """Return the required contract file paths for one season."""
    return {
        suffix: data_dir / f"{season}_{suffix}.parquet" for suffix in REQUIRED_CONTRACT_SUFFIXES
    }


def _validate_contract_paths(contract_paths: dict[str, Path], season: int) -> None:
    """Raise an explicit error when any required season contract files are missing."""
    missing_files = [path.name for path in contract_paths.values() if not path.exists()]
    if missing_files:
        missing_list = ", ".join(sorted(missing_files))
        raise MissingSeasonContractError(
            f"Season {season} is missing UI contract files: {missing_list}"
        )


def _order_columns_by_category(columns: list[str], entity: Entity) -> list[str]:
    """Order columns by the registry's category/subcategory display taxonomy."""
    registry = get_registry()
    categories = registry.categories(entity)
    category_rank = {category.name: index for index, category in enumerate(categories)}
    subcategory_rank = {
        (category.name, subcategory): index
        for category in categories
        for index, subcategory in enumerate(category.subcategories)
    }

    def sort_key(item: tuple[int, str]) -> tuple[int, int, int]:
        original_index, column = item
        resolved = registry.resolve_column(column)
        if resolved is None:
            return (len(category_rank), 0, original_index)
        category_index = category_rank.get(resolved.category, len(category_rank))
        subcategory_index = subcategory_rank.get(
            (resolved.category, resolved.subcategory or ""), -1
        )
        return (category_index, subcategory_index, original_index)

    return [column for _, column in sorted(enumerate(columns), key=sort_key)]


def _build_team_payload(frame: pl.DataFrame) -> TablePayload:
    """Return a grouped team index payload from the season combined data."""
    identity_columns = [column for column in ("team",) if column in frame.columns]
    rating_columns = _ordered_existing_columns(frame.columns, TEAM_RATING_COLUMNS)
    opponent_context = [
        column
        for column in frame.columns
        if column.startswith("opp_") and not column.startswith("opp_qb_")
    ]
    per_snap_rates = [
        column
        for column in frame.columns
        if _is_team_per_snap_column(column) and not column.startswith("opp_")
    ]
    excluded_columns = set(identity_columns + rating_columns + opponent_context + per_snap_rates)
    per_game_rates = _order_columns_by_category(
        [
            column
            for column in frame.columns
            if column not in excluded_columns
            and not _starts_with_any(column, TEAM_EXCLUDED_PREFIXES)
        ],
        "team",
    )
    per_snap_rates = _order_columns_by_category(per_snap_rates, "team")
    visible_columns = (
        identity_columns + rating_columns + per_game_rates + per_snap_rates + opponent_context
    )
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": {
            "identity": identity_columns,
            "ratings": rating_columns,
            "per_game_rates": per_game_rates,
            "per_snap_rates": per_snap_rates,
            "opponent_context": opponent_context,
        },
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _build_qb_payload(frame: pl.DataFrame) -> TablePayload:
    """Return a grouped QB index payload from the season QB combined data."""
    identity_columns = [
        column
        for column in ("qb_id", "qb_name", "player_id", "player_display_name", "team")
        if column in frame.columns
    ]
    rating_columns = _ordered_existing_columns(frame.columns, QB_RATING_COLUMNS)
    opponent_context = [
        column
        for column in frame.columns
        if column.startswith("opp_") or column.startswith("qopp_")
    ]
    per_game_rates = [column for column in frame.columns if column.endswith("_per_game")]
    per_dropback_rates = [
        column
        for column in frame.columns
        if not (column.startswith("opp_") or column.startswith("qopp_"))
        and (column in QB_PER_DROPBACK_RATE_COLUMNS or column.endswith("_per_dropback"))
    ]
    excluded_columns = set(
        identity_columns + rating_columns + opponent_context + per_game_rates + per_dropback_rates
    )
    raw_totals = _order_columns_by_category(
        [
            column
            for column in frame.columns
            if column not in excluded_columns and not _starts_with_any(column, QB_EXCLUDED_PREFIXES)
        ],
        "qb",
    )
    per_game_rates = _order_columns_by_category(per_game_rates, "qb")
    per_dropback_rates = _order_columns_by_category(per_dropback_rates, "qb")
    visible_columns = (
        identity_columns
        + rating_columns
        + raw_totals
        + per_game_rates
        + per_dropback_rates
        + opponent_context
    )
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": {
            "identity": identity_columns,
            "ratings": rating_columns,
            "raw_totals": raw_totals,
            "per_game_rates": per_game_rates,
            "per_dropback_rates": per_dropback_rates,
            "opponent_context": opponent_context,
        },
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _load_game_log_frame(data_dir: Path, season: int, suffix: str) -> pl.DataFrame:
    """Read one additive game-log Parquet file for the requested season."""
    file_path = data_dir / f"{season}_{suffix}.parquet"
    if not file_path.exists():
        raise MissingSeasonContractError(
            f"Season {season} is missing UI contract files: {file_path.name}"
        )
    return pl.read_parquet(file_path)


def _filter_entity_game_logs(
    frame: pl.DataFrame,
    column: str,
    entity_id: str,
    season: int,
    entity_label: str,
) -> pl.DataFrame:
    """Return game-log rows for the requested entity or raise a clear lookup error."""
    if column not in frame.columns:
        raise MissingEntityGameLogError(
            f"Season {season} {entity_label} game logs do not include the {column} column."
        )

    filtered = frame.filter(pl.col(column) == entity_id)
    if filtered.is_empty():
        raise MissingEntityGameLogError(
            f"Season {season} has no UI game-log rows for {entity_label} {entity_id}."
        )
    return filtered.sort([key for key in ("week", "game_id") if key in filtered.columns])


def _build_team_game_log_payload(frame: pl.DataFrame) -> TablePayload:
    """Return a grouped team game-log payload for one selected team."""
    identity_columns = _ordered_existing_columns(
        frame.columns,
        ("game_id", "week", "team", "opponent_team"),
    )
    result_columns = _ordered_existing_columns(
        frame.columns,
        ("points_for", "points_allowed", "point_margin", "win_value", "turnover_margin"),
    )
    per_snap_rates = [
        column
        for column in frame.columns
        if _is_team_per_snap_column(column) and not column.startswith("opp_")
    ]
    excluded_columns = set(
        identity_columns + result_columns + per_snap_rates + ["season", "season_type"]
    )
    raw_totals = [column for column in frame.columns if column not in excluded_columns]
    visible_columns = identity_columns + result_columns + raw_totals + per_snap_rates
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": {
            "identity": identity_columns,
            "results": result_columns,
            "raw_totals": raw_totals,
            "per_snap_rates": per_snap_rates,
        },
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _build_qb_game_log_payload(frame: pl.DataFrame) -> TablePayload:
    """Return a grouped QB game-log payload for one selected quarterback."""
    identity_columns = _ordered_existing_columns(
        frame.columns,
        ("game_id", "week", "team", "opponent_team", "qb_id", "qb_name"),
    )
    result_columns = _ordered_existing_columns(
        frame.columns,
        ("points_for", "points_allowed", "qb_fourth_quarter_comeback", "qb_game_winning_drive"),
    )
    per_dropback_rates = [
        column
        for column in frame.columns
        if column in QB_PER_DROPBACK_RATE_COLUMNS or column.endswith("_per_dropback")
    ]
    excluded_columns = set(
        identity_columns + result_columns + per_dropback_rates + ["season", "season_type"]
    )
    raw_totals = [column for column in frame.columns if column not in excluded_columns]
    visible_columns = identity_columns + result_columns + raw_totals + per_dropback_rates
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": {
            "identity": identity_columns,
            "results": result_columns,
            "raw_totals": raw_totals,
            "per_dropback_rates": per_dropback_rates,
        },
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _ordered_existing_columns(
    columns: Iterable[str], preferred_order: tuple[str, ...]
) -> list[str]:
    """Keep only the requested columns, preserving the provided preference order."""
    available = set(columns)
    return [column for column in preferred_order if column in available]


def _is_team_per_snap_column(column: str) -> bool:
    """Return whether a column belongs in the team per-snap group."""
    return column.endswith("_per_offensive_snap") or column.endswith("_per_defensive_snap")


def _starts_with_any(column: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether a column starts with any of the provided prefixes."""
    return any(column.startswith(prefix) for prefix in prefixes)
