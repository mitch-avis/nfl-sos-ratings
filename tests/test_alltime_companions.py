"""Tests for the all-time rating companion post-pass."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.alltime_companions import apply_alltime_rating_companions


def _write_team_files(
    data_dir: Path, season: int, combined: pl.DataFrame, ratings: pl.DataFrame
) -> None:
    """Write one season of team combined and ratings files."""
    combined.write_parquet(data_dir / f"{season}_combined.parquet")
    ratings.write_parquet(data_dir / f"{season}_ratings.parquet")


def _write_qb_files(
    data_dir: Path,
    season: int,
    combined: pl.DataFrame,
    ratings: pl.DataFrame,
) -> None:
    """Write one season of QB combined and ratings files."""
    combined.write_parquet(data_dir / f"{season}_qb_combined.parquet")
    ratings.write_parquet(data_dir / f"{season}_qb_ratings.parquet")


def _zscore_against(values: list[float]) -> list[float]:
    """Return rounded pooled z-scores for assertion-friendly comparisons."""
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    centered = array - float(array.mean())
    scores = centered / std if std > 0.0 else centered
    return [round(float(value), 3) for value in scores]


def test_apply_alltime_rating_companions_uses_pooled_sources_without_touching_flagships(
    tmp_path: Path,
) -> None:
    """All-time companions should be a post-pass over pooled raw sources, not the flagship path."""
    team_2005_combined = pl.DataFrame(
        {
            "team": ["A", "B"],
            "SaCR": [1.0, -1.0],
            "SaOvR": [0.8, -0.8],
            "SaOR": [0.7, -0.7],
            "SaDR": [0.5, -0.5],
            "SaSTR": [0.1, -0.1],
            "adj_off_passing_epa_per_offensive_snap": [0.30, -0.30],
            "adj_off_rushing_epa_per_offensive_snap": [0.15, -0.15],
            "adj_def_passing_epa_per_offensive_snap": [0.22, -0.22],
            "adj_def_rushing_epa_per_offensive_snap": [0.10, -0.10],
            "st_rating": [0.06, -0.06],
        }
    )
    team_2005_ratings = pl.DataFrame(
        {
            "team": ["A", "B"],
            "games_played": [16, 16],
            "SaCR": [1.0, -1.0],
            "SaOvR": [0.8, -0.8],
            "SaOR": [0.7, -0.7],
            "SaDR": [0.5, -0.5],
            "SaSTR": [0.1, -0.1],
            "SRS": [6.0, -6.0],
            "sos": [0.3, -0.3],
        }
    )
    team_2006_combined = pl.DataFrame(
        {
            "team": ["C", "D"],
            "SaCR": [0.9, -0.9],
            "SaOvR": [0.6, -0.6],
            "SaOR": [0.6, -0.6],
            "SaDR": [0.3, -0.3],
            "SaSTR": [0.0, 0.0],
            "adj_off_passing_epa_per_offensive_snap": [0.20, -0.20],
            "adj_off_rushing_epa_per_offensive_snap": [0.12, -0.12],
            "adj_def_passing_epa_per_offensive_snap": [0.16, -0.16],
            "adj_def_rushing_epa_per_offensive_snap": [0.08, -0.08],
            "st_rating": [0.01, -0.01],
        }
    )
    team_2006_ratings = pl.DataFrame(
        {
            "team": ["C", "D"],
            "games_played": [16, 16],
            "SaCR": [0.9, -0.9],
            "SaOvR": [0.6, -0.6],
            "SaOR": [0.6, -0.6],
            "SaDR": [0.3, -0.3],
            "SaSTR": [0.0, 0.0],
            "SRS": [4.0, -4.0],
            "sos": [0.2, -0.2],
        }
    )

    qb_2005_combined = pl.DataFrame(
        {
            "qb_id": ["qb-2005"],
            "qb_name": ["Legacy QB"],
            "team": ["A"],
            "QSaCR": [None],
            "QSaOR": [0.4],
            "QRaw": [None],
            "QSoS": [0.1],
            "QOutcome": [0.0],
            "adj_qb_epa_per_dropback": [0.10],
            "adj_qb_completion_percentage_above_expectation": [None],
            "adj_qb_sack_rate": [0.05],
            "adj_qb_td_int_margin_rate": [0.02],
            "qb_is_eligible": [True],
        }
    )
    qb_2005_ratings = pl.DataFrame(
        {
            "qb_id": ["qb-2005"],
            "qb_name": ["Legacy QB"],
            "team": ["A"],
            "QSaCR": [None],
            "QSaOR": [0.4],
            "QRaw": [None],
            "QSoS": [0.1],
            "faced_opp_SaCR": [0.2],
            "adj_qb_designed_rush_epa_per_carry": [0.15],
            "adj_def_rushing_epa_per_offensive_snap_faced": [0.05],
            "QOutcome": [0.0],
        }
    )
    qb_2006_combined = pl.DataFrame(
        {
            "qb_id": ["qb-2006-a", "qb-2006-b"],
            "qb_name": ["Modern QB A", "Modern QB B"],
            "team": ["C", "D"],
            "QSaCR": [0.8, -0.8],
            "QSaOR": [0.6, -0.6],
            "QRaw": [0.5, -0.5],
            "QSoS": [0.2, -0.2],
            "QOutcome": [0.1, -0.1],
            "adj_qb_epa_per_dropback": [0.22, 0.04],
            "adj_qb_completion_percentage_above_expectation": [3.0, 1.0],
            "adj_qb_sack_rate": [0.03, 0.06],
            "adj_qb_td_int_margin_rate": [0.07, 0.02],
            "qb_is_eligible": [True, True],
        }
    )
    qb_2006_ratings = pl.DataFrame(
        {
            "qb_id": ["qb-2006-a", "qb-2006-b"],
            "qb_name": ["Modern QB A", "Modern QB B"],
            "team": ["C", "D"],
            "QSaCR": [0.8, -0.8],
            "QSaOR": [0.6, -0.6],
            "QRaw": [0.5, -0.5],
            "QSoS": [0.2, -0.2],
            "faced_opp_SaCR": [0.3, -0.3],
            "adj_qb_designed_rush_epa_per_carry": [0.35, -0.05],
            "adj_def_rushing_epa_per_offensive_snap_faced": [0.10, -0.10],
            "QOutcome": [0.1, -0.1],
        }
    )

    _write_team_files(tmp_path, 2005, team_2005_combined, team_2005_ratings)
    _write_team_files(tmp_path, 2006, team_2006_combined, team_2006_ratings)
    _write_qb_files(tmp_path, 2005, qb_2005_combined, qb_2005_ratings)
    _write_qb_files(tmp_path, 2006, qb_2006_combined, qb_2006_ratings)

    original_team_flagships = pl.read_parquet(tmp_path / "2005_ratings.parquet").select(
        ["SaCR", "SaOvR"]
    )
    original_qb_flagships = pl.read_parquet(tmp_path / "2006_qb_ratings.parquet").select(
        ["QSaCR", "QSaOR"]
    )

    apply_alltime_rating_companions(tmp_path, seasons=[2005, 2006])

    team_2005_after = pl.read_parquet(tmp_path / "2005_ratings.parquet")
    team_2006_after = pl.read_parquet(tmp_path / "2006_ratings.parquet")
    qb_2005_after = pl.read_parquet(tmp_path / "2005_qb_ratings.parquet")
    qb_2006_after = pl.read_parquet(tmp_path / "2006_qb_ratings.parquet")

    team_sources = [
        *composite_weights.build_weighted_composite(
            team_2005_combined,
            composite_weights.TEAM_SACR_FROZEN_SPEC,
            team_2005_combined,
        ).tolist(),
        *composite_weights.build_weighted_composite(
            team_2006_combined,
            composite_weights.TEAM_SACR_FROZEN_SPEC,
            team_2006_combined,
        ).tolist(),
    ]
    expected_team_sacr_alltime = _zscore_against(team_sources)
    observed_team_sacr_alltime = [
        *team_2005_after.select("SaCR_alltime").to_series().to_list(),
        *team_2006_after.select("SaCR_alltime").to_series().to_list(),
    ]
    assert observed_team_sacr_alltime == pytest.approx(expected_team_sacr_alltime)

    qb_qsaor_sources = [0.10, 0.22, 0.04]
    observed_qsaor_alltime = [
        qb_2005_after.select("QSaOR_alltime").item(),
        *qb_2006_after.select("QSaOR_alltime").to_series().to_list(),
    ]
    assert observed_qsaor_alltime == pytest.approx(_zscore_against(qb_qsaor_sources))

    expected_qsacr_sources = composite_weights.build_weighted_composite(
        qb_2006_combined,
        composite_weights.QB_QSACR_FROZEN_SPEC,
        qb_2006_combined,
    ).tolist()
    assert qb_2005_after.select("QSaCR_alltime").item() is None
    assert qb_2006_after.select("QSaCR_alltime").to_series().to_list() == pytest.approx(
        _zscore_against(expected_qsacr_sources)
    )

    assert team_2005_after.select(["SaCR", "SaOvR"]).equals(original_team_flagships)
    assert qb_2006_after.select(["QSaCR", "QSaOR"]).equals(original_qb_flagships)

    assert team_2005_after.columns == [
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
    ]
    assert qb_2006_after.columns[:10] == [
        "qb_id",
        "qb_name",
        "team",
        "QSaCR",
        "QSaCR_alltime",
        "QSaOR",
        "QSaOR_alltime",
        "QRaw",
        "QSoS",
        "faced_opp_SaCR",
    ]
