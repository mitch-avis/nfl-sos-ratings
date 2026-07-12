"""Tests for the CSV-backed UI data contract."""

from pathlib import Path

import pytest

from nfl_sos_ratings.ui_data import (
    MissingSeasonContractError,
    discover_available_seasons,
    load_qb_game_log_payload,
    load_season_ui_dataset,
    load_team_game_log_payload,
)


def _write_csv(path: Path, header: str, row: str) -> None:
    """Write a minimal CSV file for contract tests."""
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")


def test_discover_available_seasons_requires_complete_contract(tmp_path: Path) -> None:
    """Return only seasons that have the complete first-pass UI contract."""
    contract_files = {
        "team_per_game_stats": "team,points_for\nDET,31\n",
        "qb_per_game_stats": "player_id,player_display_name\nqb-1,Jared Goff\n",
        "combined": "team,points_for,opp_points_for,SaCR\nDET,31,20,1.2\n",
        "qb_combined": (
            "player_id,player_display_name,qb_attempts_total,opp_qb_any_a,QSaCR\n"
            "qb-1,Jared Goff,500,6.5,1.1\n"
        ),
        "ratings": "team,SaCR\nDET,1.2\n",
        "qb_ratings": "player_id,QSaCR\nqb-1,1.1\n",
    }

    for suffix, content in contract_files.items():
        (tmp_path / f"2024_{suffix}.csv").write_text(content, encoding="utf-8")

    incomplete_files = dict(contract_files)
    incomplete_files.pop("qb_ratings")
    for suffix, content in incomplete_files.items():
        (tmp_path / f"2025_{suffix}.csv").write_text(content, encoding="utf-8")

    assert discover_available_seasons(tmp_path) == [2024]


def test_load_season_ui_dataset_groups_team_and_qb_columns(tmp_path: Path) -> None:
    """Build one normalized payload with grouped index columns for teams and QBs."""
    _write_csv(
        tmp_path / "2024_team_per_game_stats.csv",
        "team,points_for,points_per_offensive_snap,games_played",
        "DET,510,0.42,17",
    )
    _write_csv(
        tmp_path / "2024_qb_per_game_stats.csv",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18",
    )
    _write_csv(
        tmp_path / "2024_combined.csv",
        (
            "team,points_for,total_yards,points_per_offensive_snap,"
            "opp_points_for,opp_points_allowed,SaCR,SaOR,SaDR,SaOvR,SRS"
        ),
        "DET,510,6800,0.42,390,315,1.2,1.1,0.9,1.0,7.4",
    )
    _write_csv(tmp_path / "2024_ratings.csv", "team,SaCR\n", "DET,1.2")
    _write_csv(
        tmp_path / "2024_qb_combined.csv",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback,opp_qb_any_a,QRaw,QSoS,QSaOR,QOutcome,QSaCR"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,1.0,0.2,1.1,0.4,1.3",
    )
    _write_csv(tmp_path / "2024_qb_ratings.csv", "player_id,QSaCR\n", "qb-1,1.3")

    dataset = load_season_ui_dataset(tmp_path, 2024)

    assert dataset["season"] == 2024
    assert dataset["teams"]["rows"][0]["team"] == "DET"
    assert dataset["qbs"]["rows"][0]["player_display_name"] == "Jared Goff"
    assert dataset["teams"]["column_groups"]["per_game_rates"] == ["points_for", "total_yards"]
    assert dataset["teams"]["column_groups"]["per_snap_rates"] == ["points_per_offensive_snap"]
    assert dataset["teams"]["column_groups"]["opponent_context"] == [
        "opp_points_for",
        "opp_points_allowed",
    ]
    assert dataset["teams"]["column_groups"]["ratings"] == [
        "SaCR",
        "SaOR",
        "SaDR",
        "SaOvR",
        "SRS",
    ]
    assert "raw_totals" not in dataset["teams"]["column_groups"]
    assert dataset["qbs"]["column_groups"]["raw_totals"] == ["qb_attempts_total"]
    assert dataset["qbs"]["column_groups"]["per_game_rates"] == ["qb_attempts_per_game"]
    assert dataset["qbs"]["column_groups"]["per_dropback_rates"] == ["qb_epa_per_dropback"]
    assert dataset["qbs"]["column_groups"]["opponent_context"] == ["opp_qb_any_a"]
    assert dataset["qbs"]["column_groups"]["ratings"] == [
        "QRaw",
        "QSoS",
        "QSaOR",
        "QOutcome",
        "QSaCR",
    ]


def test_load_season_ui_dataset_errors_for_incomplete_contract(tmp_path: Path) -> None:
    """Raise a clear error when a requested season is missing contract files."""
    _write_csv(tmp_path / "2024_combined.csv", "team,SaCR", "DET,1.2")

    with pytest.raises(MissingSeasonContractError):
        load_season_ui_dataset(tmp_path, 2024)


def test_load_season_ui_dataset_supports_current_qb_output_names(tmp_path: Path) -> None:
    """Support the explicit QB output schema used by the current generated CSVs."""
    _write_csv(tmp_path / "2024_team_per_game_stats.csv", "team,points_for", "DET,510")
    _write_csv(tmp_path / "2024_qb_per_game_stats.csv", "qb_id,qb_name,team", "qb-1,Jared Goff,DET")
    _write_csv(tmp_path / "2024_combined.csv", "team,SaCR", "DET,1.2")
    _write_csv(tmp_path / "2024_ratings.csv", "team,SaCR", "DET,1.2")
    _write_csv(
        tmp_path / "2024_qb_combined.csv",
        "qb_id,qb_name,team,qb_attempts_total,qb_attempts_per_game,qb_epa_per_dropback,qopp_qb_any_a,qopp_qb_epa_per_dropback,QRaw,QSoS,QSaOR,QOutcome,QSaCR",
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,0.05,1.0,0.2,1.1,0.4,1.3",
    )
    _write_csv(tmp_path / "2024_qb_ratings.csv", "qb_id,QSaCR", "qb-1,1.3")

    dataset = load_season_ui_dataset(tmp_path, 2024)

    assert dataset["qbs"]["rows"][0]["qb_name"] == "Jared Goff"
    assert dataset["qbs"]["column_groups"]["identity"] == ["qb_id", "qb_name", "team"]
    assert dataset["qbs"]["column_groups"]["opponent_context"] == [
        "qopp_qb_any_a",
        "qopp_qb_epa_per_dropback",
    ]


def test_load_team_and_qb_game_log_payloads_filter_rows_by_entity(tmp_path: Path) -> None:
    """Load additive team and QB game-log payloads for one selected entity."""
    _write_csv(
        tmp_path / "2024_team_game_logs.csv",
        (
            "game_id,week,team,opponent_team,points_for,points_allowed,point_margin,"
            "points_per_offensive_snap"
        ),
        "g1,1,DET,KC,24,17,7,0.42\ng2,2,DET,CHI,21,20,1,0.35\ng3,1,KC,DET,17,24,-7,0.31",
    )
    _write_csv(
        tmp_path / "2024_qb_game_logs.csv",
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

    team_payload = load_team_game_log_payload(tmp_path, 2024, "DET")
    qb_payload = load_qb_game_log_payload(tmp_path, 2024, "qb-1")

    assert [row["opponent_team"] for row in team_payload["rows"]] == ["KC", "CHI"]
    assert team_payload["column_groups"]["identity"] == ["game_id", "week", "team", "opponent_team"]
    assert team_payload["column_groups"]["results"] == [
        "points_for",
        "points_allowed",
        "point_margin",
    ]
    assert team_payload["column_groups"]["per_snap_rates"] == ["points_per_offensive_snap"]

    assert [row["week"] for row in qb_payload["rows"]] == [1, 2]
    assert qb_payload["rows"][0]["qb_name"] == "Jared Goff"
    assert qb_payload["column_groups"]["identity"] == [
        "game_id",
        "week",
        "team",
        "opponent_team",
        "qb_id",
        "qb_name",
    ]
    assert qb_payload["column_groups"]["per_dropback_rates"] == ["qb_epa_per_dropback"]
