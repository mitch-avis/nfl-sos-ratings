"""All-time companion ratings computed as a post-pass over written season files.

These companions are intentionally separate from the within-season flagship path.
They reuse the already-written season outputs, compute pooled reference z-scores
for the requested companion surfaces, and write the augmented files back out.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.metrics import get_registry

TEAM_RATINGS_ORDER = (
    "team",
    "games_played",
    "SaCR",
    "SaCR_alltime",
    "SaOvR",
    "SaOvR_alltime",
    "SaOR",
    "SaDR",
    "SaSTR",
    "SRS",
    "sos",
)

QB_RATINGS_ORDER = (
    "qb_id",
    "qb_name",
    "player_id",
    "player_display_name",
    "team",
    "QSaCR",
    "QSaCR_alltime",
    "QSaOR",
    "QSaOR_alltime",
    "QRaw",
    "QSoS",
    "faced_opp_SaCR",
    "adj_qb_designed_rush_epa_per_carry",
    "adj_def_rushing_epa_per_offensive_snap_faced",
    "QOutcome",
)


def apply_alltime_rating_companions(data_dir: Path, seasons: Sequence[int]) -> None:
    """Augment written rating outputs with pooled all-time companion columns."""
    _apply_team_alltime_companions(data_dir, seasons)
    _apply_qb_alltime_companions(data_dir, seasons)


def _apply_team_alltime_companions(data_dir: Path, seasons: Sequence[int]) -> None:
    """Write team all-time companion ratings into the combined and ratings outputs."""
    combined_frames: dict[int, pl.DataFrame] = {}
    ratings_frames: dict[int, pl.DataFrame] = {}
    sacr_sources: dict[int, list[float]] = {}
    saovr_sources: dict[int, list[float]] = {}

    for season in sorted(seasons):
        combined_path = data_dir / f"{season}_combined.parquet"
        ratings_path = data_dir / f"{season}_ratings.parquet"
        if not combined_path.exists() or not ratings_path.exists():
            continue

        combined = pl.read_parquet(combined_path)
        ratings = pl.read_parquet(ratings_path)
        if {"team", "SaCR", "SaOvR", "SaOR", "SaDR", "SaSTR"} - set(combined.columns):
            continue

        combined_frames[season] = combined
        ratings_frames[season] = ratings
        sacr_sources[season] = composite_weights.build_weighted_composite(
            combined,
            composite_weights.TEAM_SACR_FROZEN_SPEC,
            combined,
        ).tolist()
        saovr_sources[season] = (
            combined.select((pl.col("SaOR") + pl.col("SaDR") + pl.col("SaSTR")).alias("_raw"))
            .to_series()
            .cast(pl.Float64)
            .to_list()
        )

    if not combined_frames:
        return

    sacr_reference = [value for season in sorted(sacr_sources) for value in sacr_sources[season]]
    saovr_reference = [value for season in sorted(saovr_sources) for value in saovr_sources[season]]

    for season, combined in combined_frames.items():
        team_keys = combined.select("team").to_series().to_list()
        companion_frame = pl.DataFrame(
            {
                "team": team_keys,
                "SaCR_alltime": _score_against_reference(sacr_sources[season], sacr_reference),
                "SaOvR_alltime": _score_against_reference(saovr_sources[season], saovr_reference),
            }
        )
        combined_augmented = _augment_frame(combined, ["team"], companion_frame)
        ratings_augmented = _augment_frame(ratings_frames[season], ["team"], companion_frame)
        ratings_augmented = _reorder_columns(ratings_augmented, TEAM_RATINGS_ORDER)
        _validate_and_write(combined_augmented, data_dir / f"{season}_combined.parquet")
        _validate_and_write(ratings_augmented, data_dir / f"{season}_ratings.parquet")


def _apply_qb_alltime_companions(data_dir: Path, seasons: Sequence[int]) -> None:
    """Write QB all-time companion ratings into the combined and ratings outputs."""
    combined_frames: dict[int, pl.DataFrame] = {}
    ratings_frames: dict[int, pl.DataFrame] = {}
    qsaor_sources: dict[int, list[float]] = {}
    qsaor_masks: dict[int, list[bool]] = {}
    qsacr_sources: dict[int, list[float]] = {}
    qsacr_masks: dict[int, list[bool]] = {}

    for season in sorted(seasons):
        combined_path = data_dir / f"{season}_qb_combined.parquet"
        ratings_path = data_dir / f"{season}_qb_ratings.parquet"
        if not combined_path.exists() or not ratings_path.exists():
            continue

        combined = pl.read_parquet(combined_path)
        ratings = pl.read_parquet(ratings_path)
        if "adj_qb_epa_per_dropback" not in combined.columns:
            continue

        combined_frames[season] = combined
        ratings_frames[season] = ratings
        qsaor_sources[season] = (
            combined.select(pl.col("adj_qb_epa_per_dropback").cast(pl.Float64))
            .to_series()
            .to_list()
        )
        qsaor_masks[season] = _published_mask(combined, "QSaOR")

        qsacr_sources[season] = composite_weights.build_weighted_composite(
            combined,
            composite_weights.QB_QSACR_FROZEN_SPEC,
            combined,
        ).tolist()
        qsacr_masks[season] = _published_mask(combined, "QSaCR")

    if not combined_frames:
        return

    qsaor_reference = [
        value
        for season in sorted(qsaor_sources)
        for value, include in zip(qsaor_sources[season], qsaor_masks[season], strict=True)
        if include
    ]
    qsacr_reference = [
        value
        for season in sorted(qsacr_sources)
        for value, include in zip(qsacr_sources[season], qsacr_masks[season], strict=True)
        if include
    ]

    for season, combined in combined_frames.items():
        join_keys = _matching_qb_join_keys(combined, ratings_frames[season])
        if not join_keys:
            continue

        companion_frame = combined.select(join_keys).with_columns(
            pl.Series(
                "QSaOR_alltime",
                _masked_scores(qsaor_sources[season], qsaor_reference, qsaor_masks[season]),
                dtype=pl.Float64,
            ),
            pl.Series(
                "QSaCR_alltime",
                _masked_scores(qsacr_sources[season], qsacr_reference, qsacr_masks[season]),
                dtype=pl.Float64,
            ),
        )
        combined_augmented = _augment_frame(combined, join_keys, companion_frame)
        ratings_augmented = _augment_frame(ratings_frames[season], join_keys, companion_frame)
        ratings_augmented = _reorder_columns(ratings_augmented, QB_RATINGS_ORDER)
        _validate_and_write(combined_augmented, data_dir / f"{season}_qb_combined.parquet")
        _validate_and_write(ratings_augmented, data_dir / f"{season}_qb_ratings.parquet")


def _augment_frame(
    frame: pl.DataFrame, join_keys: list[str], companion_frame: pl.DataFrame
) -> pl.DataFrame:
    """Join new companion columns onto a frame after dropping stale copies."""
    companion_columns = [column for column in companion_frame.columns if column not in join_keys]
    existing_companions = [column for column in companion_columns if column in frame.columns]
    base_frame = frame.drop(existing_companions) if existing_companions else frame
    return base_frame.join(companion_frame, on=join_keys, how="left")


def _matching_qb_join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    """Return the shared QB identity keys for companion joins."""
    preferred_keys = ("qb_id", "player_id", "qb_name", "player_display_name", "team")
    return [key for key in preferred_keys if key in left.columns and key in right.columns]


def _published_mask(frame: pl.DataFrame, column: str) -> list[bool]:
    """Return whether each row publishes the named flagship metric."""
    if column not in frame.columns:
        return [False] * frame.height
    return (
        frame.select(pl.col(column).cast(pl.Float64).fill_nan(None).is_not_null())
        .to_series()
        .to_list()
    )


def _masked_scores(
    values: Sequence[float],
    reference_values: Sequence[float],
    mask: Sequence[bool],
) -> list[float | None]:
    """Return pooled z-scores only for rows whose published flagship is non-null."""
    scored_values = _score_against_reference(values, reference_values)
    return [score if include else None for score, include in zip(scored_values, mask, strict=True)]


def _score_against_reference(
    values: Sequence[float],
    reference_values: Sequence[float],
) -> list[float]:
    """Return one list of rounded pooled z-scores."""
    value_array = np.asarray(values, dtype=np.float64)
    reference_array = np.asarray(reference_values, dtype=np.float64)
    if reference_array.size == 0:
        return [round(float(value), 3) for value in value_array.tolist()]
    std = float(reference_array.std(ddof=1)) if reference_array.size > 1 else 0.0
    centered = value_array - float(reference_array.mean())
    scored = centered / std if std > 0.0 else centered
    return [round(float(value), 3) for value in scored.tolist()]


def _reorder_columns(frame: pl.DataFrame, preferred_order: Iterable[str]) -> pl.DataFrame:
    """Move preferred columns to the front while keeping the rest stable."""
    preferred = [column for column in preferred_order if column in frame.columns]
    trailing = [column for column in frame.columns if column not in set(preferred)]
    return frame.select(preferred + trailing)


def _validate_and_write(frame: pl.DataFrame, path: Path) -> None:
    """Validate output columns against the registry before overwriting one file."""
    unknown = get_registry().validate_columns(frame.columns)
    if unknown:
        raise ValueError(
            f"Output {path.name} contains columns missing from the metric registry: "
            + ", ".join(unknown)
        )
    frame.write_parquet(path)
