"""Parquet-backed data contract helpers for the local analyst UI."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import polars as pl

from nfl_sos_ratings import config
from nfl_sos_ratings.data_loader import PBP_START_SEASON
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.qb_rating import fit_qb_ratings
from nfl_sos_ratings.rating_ranges import (
    PAIR_QUANTILES,
    QB_PAIR_COLUMNS,
    QB_RANGE_COLUMNS,
    RANGE_QUANTILES,
    TEAM_PAIR_COLUMNS,
    TEAM_RANGE_COLUMNS,
    TOP_RANKS,
    UNIT_RANGE_COLUMNS,
    PairColumns,
    RangeColumns,
    quantile_suffix,
)
from nfl_sos_ratings.team_rating import fit_team_ratings, fit_team_ratings_with_previous_penalties
from nfl_sos_ratings.wp_filter import MAX_WP_THRESHOLD, QbWpFilter, TeamWpFilter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

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
# The first season with 17 regular-season games per team.
FIRST_17_GAME_SEASON = 2021
TEAM_GAME_LOG_SUFFIX = "team_game_logs"
QB_GAME_LOG_SUFFIX = "qb_game_logs"
TEAM_RATING_HISTORY_SUFFIX = "ratings_by_week"
QB_RATING_HISTORY_SUFFIX = "qb_ratings_by_week"
TEAM_RATING_RANGES_SUFFIX = "rating_ranges"
QB_RATING_RANGES_SUFFIX = "qb_rating_ranges"
TEAM_RATING_PAIRS_SUFFIX = "rating_pairs"
QB_RATING_PAIRS_SUFFIX = "qb_rating_pairs"
TEAM_RANK_HISTORY_SUFFIX = "rating_ranges_by_week"
QB_RANK_HISTORY_SUFFIX = "qb_rating_ranges_by_week"
TEAM_RATINGS_SUFFIX = "ratings"
QB_RATINGS_SUFFIX = "qb_ratings"
TEAM_WP_BINS_SUFFIX = "team_wp_bins"
QB_WP_BINS_SUFFIX = "qb_wp_bins"
# Seasons whose filter models stay built, and filtered tables (one per season and threshold) kept.
_WP_MODEL_CACHE_SIZE = 8
_WP_TABLE_CACHE_SIZE = 256
_FILTERED_PREFIX = "filtered_"
# The kept-share columns describe the filter itself, so they keep their names.
_KEPT_SHARE_PREFIX = "wp_kept_"
TEAM_RATING_COLUMNS = (
    "team_rating",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "sos",
    "SRS",
)
QB_RATING_COLUMNS = ("adj_qb_epa_per_dropback", "qb_faced_pass_defense")
TEAM_EXCLUDED_PREFIXES: tuple[str, ...] = ()
QB_EXCLUDED_PREFIXES: tuple[str, ...] = ()
QB_PER_DROPBACK_RATE_COLUMNS = (
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
)


class MissingSeasonContractError(FileNotFoundError):
    """Raised when a requested season does not have the complete UI contract."""


class MissingEntityRowsError(LookupError):
    """Raised when a requested entity has no rows in a per-entity UI file (game logs, history)."""


class TablePayload(TypedDict):
    """Normalized table payload for a UI index view."""

    rows: list[dict[str, object]]
    visible_columns: list[str]
    column_groups: dict[str, list[str]]
    column_metadata: dict[str, dict[str, object]]


class WpRatingsPayload(TablePayload):
    """A garbage-time filter table: the threshold it used and the largest one allowed."""

    threshold: int
    max_threshold: int


class SeasonDataset(TypedDict):
    """Normalized season payload for the analyst UI.

    ``in_progress`` is true only for the season being played (``config.SEASON``) while a team still
    has regular-season games left, so a completed season with a missing or cancelled game is never
    shown as in progress.
    """

    season: int
    in_progress: bool
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


def regular_season_games(season: int) -> int:
    """Return how many regular-season games each team plays: 17 from 2021, 16 before."""
    return 17 if season >= FIRST_17_GAME_SEASON else 16


def season_in_progress(season: int, team_frame: pl.DataFrame) -> bool:
    """Return whether ``season`` is the one being played and a team still has games left.

    Only ``config.SEASON`` can be in progress; a completed season stays complete even when a team
    played fewer games (a cancelled game, or one missing from nflverse play-by-play). Without a
    ``games_played`` column the season being played counts as in progress.
    """
    if season != config.SEASON:
        return False
    if "games_played" not in team_frame.columns or team_frame.is_empty():
        return True
    fewest = team_frame.get_column("games_played").min()
    return not isinstance(fewest, int) or fewest < regular_season_games(season)


def load_season_ui_dataset(data_dir: Path, season: int) -> SeasonDataset:
    """Load one season of normalized index data for the local analyst UI."""
    contract_paths = _build_contract_paths(data_dir, season)
    _validate_contract_paths(contract_paths, season)

    team_frame = pl.read_parquet(contract_paths["combined"])
    qb_frame = pl.read_parquet(contract_paths["qb_combined"])

    return {
        "season": season,
        "in_progress": season_in_progress(season, team_frame),
        "teams": _build_team_payload(team_frame),
        "qbs": _build_qb_payload(qb_frame),
    }


def load_team_game_log_payload(data_dir: Path, season: int, team: str) -> TablePayload:
    """Load additive team game logs for one team and season."""
    frame = _load_season_file(data_dir, season, TEAM_GAME_LOG_SUFFIX)
    filtered = _filter_entity_rows(frame, "team", team, season, "team game-log")
    return _build_team_game_log_payload(filtered)


def load_qb_game_log_payload(data_dir: Path, season: int, qb_id: str) -> TablePayload:
    """Load additive quarterback game logs for one QB and season."""
    frame = _load_season_file(data_dir, season, QB_GAME_LOG_SUFFIX)
    filtered = _filter_entity_rows(frame, "qb_id", qb_id, season, "QB game-log")
    return _build_qb_game_log_payload(filtered)


def load_team_rating_history_payload(data_dir: Path, season: int, team: str) -> TablePayload:
    """Load one team's week-by-week rating history for a season."""
    frame = _load_season_file(data_dir, season, TEAM_RATING_HISTORY_SUFFIX)
    filtered = _filter_entity_rows(frame, "team", team, season, "team rating-history")
    return _build_rating_history_payload(
        filtered, ("week", "team"), ("games_played",), TEAM_RATING_COLUMNS
    )


def load_qb_rating_history_payload(data_dir: Path, season: int, qb_id: str) -> TablePayload:
    """Load one quarterback's week-by-week rating history for a season."""
    frame = _load_season_file(data_dir, season, QB_RATING_HISTORY_SUFFIX)
    filtered = _filter_entity_rows(frame, "qb_id", qb_id, season, "QB rating-history")
    return _build_rating_history_payload(
        filtered, ("week", "qb_id"), ("qb_games_played", "qb_dropbacks"), QB_RATING_COLUMNS
    )


def load_team_rating_ranges_payload(data_dir: Path, season: int) -> TablePayload:
    """Load every team's bootstrap rating and rank ranges for a season, by published rank."""
    frame = _load_season_file(data_dir, season, TEAM_RATING_RANGES_SUFFIX)
    return _build_rating_ranges_payload(frame, ("team",), TEAM_RANGE_COLUMNS)


def load_qb_rating_ranges_payload(data_dir: Path, season: int) -> TablePayload:
    """Load the eligible quarterbacks' bootstrap rating and rank ranges, by published rank."""
    frame = _load_season_file(data_dir, season, QB_RATING_RANGES_SUFFIX)
    return _build_rating_ranges_payload(frame, ("qb_id", "qb_name", "team"), QB_RANGE_COLUMNS)


def load_team_rank_history_payload(data_dir: Path, season: int, team: str) -> TablePayload:
    """Load one team's rank range as of each week (seasons in progress only), by week."""
    frame = _load_season_file(data_dir, season, TEAM_RANK_HISTORY_SUFFIX)
    rows = _filter_entity_rows(frame, "team", team, season, "team rank-history")
    return _build_rank_history_payload(rows, TEAM_RANGE_COLUMNS)


def load_qb_rank_history_payload(data_dir: Path, season: int, qb_id: str) -> TablePayload:
    """Load one eligible quarterback's rank range as of each week, by week."""
    frame = _load_season_file(data_dir, season, QB_RANK_HISTORY_SUFFIX)
    rows = _filter_entity_rows(frame, "qb_id", qb_id, season, "QB rank-history")
    return _build_rank_history_payload(rows, QB_RANGE_COLUMNS)


def load_team_rating_pairs_payload(data_dir: Path, season: int, team: str) -> TablePayload:
    """Load one team's head-to-head chances against every other team for a season."""
    frame = _load_season_file(data_dir, season, TEAM_RATING_PAIRS_SUFFIX)
    rows = _filter_entity_rows(frame, "team", team, season, "team rating-pairs")
    return _build_rating_pairs_payload(rows, TEAM_PAIR_COLUMNS)


def load_qb_rating_pairs_payload(data_dir: Path, season: int, qb_id: str) -> TablePayload:
    """Load one eligible quarterback's head-to-head chances against the other eligible ones."""
    frame = _load_season_file(data_dir, season, QB_RATING_PAIRS_SUFFIX)
    rows = _filter_entity_rows(frame, "qb_id", qb_id, season, "QB rating-pairs")
    return _build_rating_pairs_payload(rows, QB_PAIR_COLUMNS)


@dataclass(frozen=True, slots=True)
class _WpInputs:
    """The files a season's filter model reads, stamped so a rebuilt file means a new model.

    ``previous`` is the previous season's team game logs, whose cross-validated penalties the team
    fit reuses as the season command does; the first play-by-play season has none.
    """

    game_logs: Path
    bins: Path
    published: Path
    previous: Path | None
    stamp: tuple[tuple[str, int, int], ...]


def _wp_inputs(
    data_dir: Path, season: int, suffixes: tuple[str, str, str], *, reuse_previous: bool
) -> _WpInputs:
    """Return a season's filter inputs (game logs, bins, published ratings), checking each exists.

    Raises:
        MissingSeasonContractError: If a file, including the previous season's game logs when
            ``reuse_previous`` applies, is missing.

    """
    paths = [data_dir / f"{season}_{suffix}.parquet" for suffix in suffixes]
    previous = (
        data_dir / f"{season - 1}_{TEAM_GAME_LOG_SUFFIX}.parquet"
        if reuse_previous and season > PBP_START_SEASON
        else None
    )
    needed = [*paths, *([] if previous is None else [previous])]
    missing = [path.name for path in needed if not path.exists()]
    if missing:
        msg = f"Season {season} is missing garbage-time filter files: {', '.join(missing)}"
        raise MissingSeasonContractError(msg)
    stamp = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in needed)
    return _WpInputs(paths[0], paths[1], paths[2], previous, stamp)


@functools.lru_cache(maxsize=_WP_MODEL_CACHE_SIZE)
def _team_wp_model(inputs: _WpInputs) -> TeamWpFilter:
    """Build a season's team filter model with the penalties the season command used."""
    game_logs = pl.read_parquet(inputs.game_logs)
    previous = (
        None if inputs.previous is None else fit_team_ratings(pl.read_parquet(inputs.previous))
    )
    fit = fit_team_ratings_with_previous_penalties(game_logs, previous)
    return TeamWpFilter(game_logs, pl.read_parquet(inputs.bins), fit)


@functools.lru_cache(maxsize=_WP_MODEL_CACHE_SIZE)
def _qb_wp_model(inputs: _WpInputs) -> QbWpFilter:
    """Build a season's QB filter model with the penalty the season command cross-validated."""
    qb_games = pl.read_parquet(inputs.game_logs)
    return QbWpFilter(qb_games, pl.read_parquet(inputs.bins), fit_qb_ratings(qb_games))


def _with_rank(frame: pl.DataFrame, rating: str, rank: str) -> pl.DataFrame:
    """Add ``rank``: 1 for the highest ``rating``, ties sharing the better rank."""
    return frame.with_columns(
        pl.col(rating).rank(method="min", descending=True).cast(pl.Int64).alias(rank)
    )


def _filtered_table(
    published: pl.DataFrame,
    filtered: pl.DataFrame,
    unfiltered: pl.DataFrame,
    names: tuple[str, str, str],
) -> pl.DataFrame:
    """Join published, filtered, and unfiltered ratings, ranked among the published rows.

    ``names`` is the id, rating, and rank column. Filtered columns get the ``filtered_`` prefix
    (the kept share keeps its name); ``_change`` columns are filtered minus unfiltered.
    """
    key, rating, rank = names
    ids = published.select(key)
    filtered = _with_rank(ids.join(filtered, on=key, how="inner"), rating, rank)
    unfiltered = _with_rank(ids.join(unfiltered, on=key, how="inner"), rating, rank)
    renamed = filtered.rename(
        {
            column: f"{_FILTERED_PREFIX}{column}"
            for column in filtered.columns
            if column != key and not column.startswith(_KEPT_SHARE_PREFIX)
        }
    )
    return (
        _with_rank(published, rating, rank)
        .join(renamed, on=key, how="left")
        .join(
            unfiltered.select(key, pl.col(rating).alias("_rating"), pl.col(rank).alias("_rank")),
            on=key,
            how="left",
        )
        .with_columns(
            (pl.col(f"{_FILTERED_PREFIX}{rating}") - pl.col("_rating")).alias(
                f"{_FILTERED_PREFIX}{rating}_change"
            ),
            (pl.col(f"{_FILTERED_PREFIX}{rank}") - pl.col("_rank")).alias(
                f"{_FILTERED_PREFIX}{rank}_change"
            ),
        )
        .drop("_rating", "_rank")
        .sort(f"{_FILTERED_PREFIX}{rank}", nulls_last=True)
    )


@functools.lru_cache(maxsize=_WP_TABLE_CACHE_SIZE)
def _team_wp_table(inputs: _WpInputs, threshold: int) -> pl.DataFrame:
    """Return the team filter table for one season file set and threshold."""
    model = _team_wp_model(inputs)
    published = pl.read_parquet(inputs.published).select("team", "team_rating")
    return _filtered_table(
        published, model.ratings(threshold), model.ratings(0), ("team", "team_rating", "team_rank")
    )


@functools.lru_cache(maxsize=_WP_TABLE_CACHE_SIZE)
def _qb_wp_table(inputs: _WpInputs, threshold: int) -> pl.DataFrame:
    """Return the QB filter table for one season file set and threshold."""
    model = _qb_wp_model(inputs)
    published = pl.read_parquet(inputs.published)
    identity = [column for column in ("qb_id", "qb_name", "team") if column in published.columns]
    return _filtered_table(
        published.select(*identity, "adj_qb_epa_per_dropback"),
        model.ratings(threshold),
        model.ratings(0),
        ("qb_id", "adj_qb_epa_per_dropback", "qb_rank"),
    )


def _wp_payload(
    frame: pl.DataFrame, groups: dict[str, tuple[str, ...]], threshold: int
) -> WpRatingsPayload:
    """Return a filter payload with the columns of ``groups`` that ``frame`` has."""
    column_groups = {
        name: _ordered_existing_columns(frame.columns, columns) for name, columns in groups.items()
    }
    visible_columns = [column for columns in column_groups.values() for column in columns]
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": column_groups,
        "column_metadata": get_registry().column_metadata(visible_columns),
        "threshold": threshold,
        "max_threshold": MAX_WP_THRESHOLD,
    }


def load_team_wp_ratings_payload(data_dir: Path, season: int, threshold: int) -> WpRatingsPayload:
    """Return every team's ratings with garbage-time plays filtered at ``threshold`` percent.

    Teams are listed by filtered rank beside their published rating and rank; ``_change``
    columns are filtered minus unfiltered, which for teams equals the published values. Models and
    tables are cached in process and rebuilt when a season file changes.

    Raises:
        MissingSeasonContractError: If the season's game logs, bins, or ratings file, or the
            previous season's game logs, are missing.
        ValueError: If ``threshold`` is outside 0-20.

    """
    inputs = _wp_inputs(
        data_dir,
        season,
        (TEAM_GAME_LOG_SUFFIX, TEAM_WP_BINS_SUFFIX, TEAM_RATINGS_SUFFIX),
        reuse_previous=True,
    )
    groups = {
        "identity": ("team",),
        "published": ("team_rank", "team_rating"),
        "filtered": (
            "filtered_team_rank",
            "filtered_team_rating",
            "filtered_team_rating_change",
            "filtered_team_rank_change",
        ),
        "filtered_units": (
            "filtered_offense_rating",
            "filtered_defense_rating",
            "filtered_special_teams_rating",
            "filtered_sos",
        ),
        "kept": ("wp_kept_play_share",),
    }
    return _wp_payload(_team_wp_table(inputs, threshold), groups, threshold)


def load_qb_wp_ratings_payload(data_dir: Path, season: int, threshold: int) -> WpRatingsPayload:
    """Return the published quarterbacks' ratings with garbage-time dropbacks filtered out.

    Ranks are among the published (qualifying) quarterbacks. Filtered ratings use play-level EPA
    at every threshold, so ``_change`` columns compare with the same calculation at 0%; the
    published rating, on official EPA, sits beside them.

    Raises:
        MissingSeasonContractError: If the season's QB game logs, bins, or ratings file is missing.
        ValueError: If ``threshold`` is outside 0-20.

    """
    inputs = _wp_inputs(
        data_dir,
        season,
        (QB_GAME_LOG_SUFFIX, QB_WP_BINS_SUFFIX, QB_RATINGS_SUFFIX),
        reuse_previous=False,
    )
    groups = {
        "identity": ("qb_id", "qb_name", "team"),
        "published": ("qb_rank", "adj_qb_epa_per_dropback"),
        "filtered": (
            "filtered_qb_rank",
            "filtered_adj_qb_epa_per_dropback",
            "filtered_adj_qb_epa_per_dropback_change",
            "filtered_qb_rank_change",
        ),
        "filtered_context": (
            "filtered_qb_epa_per_dropback",
            "filtered_qb_faced_pass_defense",
            "filtered_qb_dropbacks",
        ),
        "kept": ("wp_kept_dropback_share",),
    }
    return _wp_payload(_qb_wp_table(inputs, threshold), groups, threshold)


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
        msg = f"Season {season} is missing UI contract files: {missing_list}"
        raise MissingSeasonContractError(msg)


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
    opponent_context = [column for column in frame.columns if column.startswith(("opp_", "qopp_"))]
    per_game_rates = [column for column in frame.columns if column.endswith("_per_game")]
    per_dropback_rates = [
        column
        for column in frame.columns
        if not (column.startswith(("opp_", "qopp_")))
        and column not in rating_columns
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


def _load_season_file(data_dir: Path, season: int, suffix: str) -> pl.DataFrame:
    """Read one per-entity Parquet file, such as the game logs, for the requested season."""
    file_path = data_dir / f"{season}_{suffix}.parquet"
    if not file_path.exists():
        msg = f"Season {season} is missing UI contract files: {file_path.name}"
        raise MissingSeasonContractError(msg)
    return pl.read_parquet(file_path)


def _filter_entity_rows(
    frame: pl.DataFrame,
    column: str,
    entity_id: str,
    season: int,
    rows_label: str,
) -> pl.DataFrame:
    """Return the requested entity's rows, sorted by week, or raise a clear lookup error.

    ``rows_label`` names the entity and the file in messages, for example ``"QB game-log"``.
    """
    if column not in frame.columns:
        msg = f"Season {season} {rows_label} rows do not include the {column} column."
        raise MissingEntityRowsError(msg)

    filtered = frame.filter(pl.col(column) == entity_id)
    if filtered.is_empty():
        msg = f"Season {season} has no UI {rows_label} rows for {entity_id}."
        raise MissingEntityRowsError(msg)
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


def _build_rating_history_payload(
    frame: pl.DataFrame,
    identity: tuple[str, ...],
    sample: tuple[str, ...],
    ratings: tuple[str, ...],
) -> TablePayload:
    """Return a rating-history payload: identity, sample size, then ratings, headline first."""
    groups = {
        "identity": _ordered_existing_columns(frame.columns, identity),
        "sample": _ordered_existing_columns(frame.columns, sample),
        "ratings": _ordered_existing_columns(frame.columns, ratings),
    }
    visible_columns = [column for columns in groups.values() for column in columns]
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": groups,
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _build_rating_ranges_payload(
    frame: pl.DataFrame, identity: tuple[str, ...], columns: RangeColumns
) -> TablePayload:
    """Return a rank-range payload: identity, published rank, the ranges, then rank chances."""
    suffixes = [quantile_suffix(level) for level in RANGE_QUANTILES]
    rank_chances = (
        *(f"{columns.rank}_top{top}_probability" for top in TOP_RANKS),
        f"{columns.rank}_missing_share",
        f"{columns.rank}_probabilities",
    )
    groups = {
        "identity": _ordered_existing_columns(frame.columns, identity),
        "published": _ordered_existing_columns(frame.columns, (columns.rank,)),
        "rating_range": _ordered_existing_columns(
            frame.columns, tuple(f"{columns.rating}{suffix}" for suffix in suffixes)
        ),
        "rank_range": _ordered_existing_columns(
            frame.columns, tuple(f"{columns.rank}{suffix}" for suffix in suffixes)
        ),
        "rank_chances": _ordered_existing_columns(frame.columns, rank_chances),
    }
    for unit in UNIT_RANGE_COLUMNS:
        unit_columns = _ordered_existing_columns(
            frame.columns,
            (
                unit.rank,
                *(f"{unit.rating}{suffix}" for suffix in suffixes),
                *(f"{unit.rank}{suffix}" for suffix in suffixes),
            ),
        )
        if unit_columns:
            groups[f"{unit.rank.removesuffix('_rank')}_range"] = unit_columns
    visible_columns = [column for group in groups.values() for column in group]
    return {
        "rows": frame.sort(columns.rank).select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": groups,
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _build_rank_history_payload(frame: pl.DataFrame, columns: RangeColumns) -> TablePayload:
    """Return a weekly rank-range payload: week and identity, the rank range, then the chances."""
    groups = {
        "identity": _ordered_existing_columns(frame.columns, ("week", columns.id)),
        "rank_range": _ordered_existing_columns(
            frame.columns,
            (
                columns.rank,
                *(f"{columns.rank}{quantile_suffix(level)}" for level in RANGE_QUANTILES),
            ),
        ),
        "rank_chances": _ordered_existing_columns(
            frame.columns,
            (
                *(f"{columns.rank}_top{top}_probability" for top in TOP_RANKS),
                f"{columns.rank}_missing_share",
            ),
        ),
    }
    visible_columns = [column for group in groups.values() for column in group]
    return {
        "rows": frame.select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": groups,
        "column_metadata": get_registry().column_metadata(visible_columns),
    }


def _build_rating_pairs_payload(frame: pl.DataFrame, columns: PairColumns) -> TablePayload:
    """Return a head-to-head payload: the pair, the chance, then the rating-gap percentiles."""
    groups = {
        "identity": _ordered_existing_columns(frame.columns, (columns.id, columns.other)),
        "chance": _ordered_existing_columns(frame.columns, (columns.above, columns.share)),
        "rating_gap": _ordered_existing_columns(
            frame.columns,
            tuple(f"{columns.gap}{quantile_suffix(level)}" for level in PAIR_QUANTILES),
        ),
    }
    visible_columns = [column for group in groups.values() for column in group]
    return {
        "rows": frame.sort(columns.other).select(visible_columns).to_dicts(),
        "visible_columns": visible_columns,
        "column_groups": groups,
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
    return column.endswith(("_per_offensive_snap", "_per_defensive_snap"))


def _starts_with_any(column: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether a column starts with any of the provided prefixes."""
    return any(column.startswith(prefix) for prefix in prefixes)
