"""Tests for the Parquet-backed UI data contract."""

import io
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_sos_ratings.ui_data import (
    MissingEntityRowsError,
    MissingSeasonContractError,
    discover_available_seasons,
    load_qb_game_log_payload,
    load_qb_rating_history_payload,
    load_season_ui_dataset,
    load_team_game_log_payload,
    load_team_rating_history_payload,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_table(path: Path, header: str, row: str) -> None:
    """Write a minimal Parquet contract file from CSV-style text."""
    pl.read_csv(io.StringIO(f"{header}\n{row}\n")).write_parquet(path)


def test_discover_available_seasons_requires_complete_contract(tmp_path: Path) -> None:
    """Return only seasons that have the complete first-pass UI contract."""
    # Arrange
    contract_files = {
        "team_per_game_stats": "team,points_for\nDET,31\n",
        "qb_per_game_stats": "player_id,player_display_name\nqb-1,Jared Goff\n",
        "combined": "team,points_for,opp_points_for,team_rating\nDET,31,20,1.2\n",
        "qb_combined": (
            "player_id,player_display_name,qb_attempts_total,opp_qb_any_a,adj_qb_epa_per_dropback\n"
            "qb-1,Jared Goff,500,6.5,0.1\n"
        ),
        "ratings": "team,team_rating\nDET,1.2\n",
        "qb_ratings": "player_id,adj_qb_epa_per_dropback\nqb-1,0.1\n",
    }

    for suffix, content in contract_files.items():
        pl.read_csv(io.StringIO(content)).write_parquet(tmp_path / f"2024_{suffix}.parquet")

    incomplete_files = dict(contract_files)
    incomplete_files.pop("qb_ratings")
    for suffix, content in incomplete_files.items():
        pl.read_csv(io.StringIO(content)).write_parquet(tmp_path / f"2025_{suffix}.parquet")

    # Act
    seasons = discover_available_seasons(tmp_path)

    # Assert
    assert seasons == [2024]


def test_load_season_ui_dataset_groups_team_and_qb_columns(tmp_path: Path) -> None:
    """Build one normalized payload with grouped index columns for teams and QBs."""
    # Arrange
    _write_table(
        tmp_path / "2024_team_per_game_stats.parquet",
        "team,points_for,points_per_offensive_snap,games_played",
        "DET,510,0.42,17",
    )
    _write_table(
        tmp_path / "2024_qb_per_game_stats.parquet",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18",
    )
    _write_table(
        tmp_path / "2024_combined.parquet",
        (
            "team,points_for,total_yards,points_per_offensive_snap,"
            "opp_points_for,opp_points_allowed,team_rating,offense_rating,defense_rating,"
            "special_teams_rating,sos,SRS"
        ),
        "DET,510,6800,0.42,390,315,7.1,4.0,2.6,0.5,0.8,7.4",
    )
    _write_table(tmp_path / "2024_ratings.parquet", "team,team_rating\n", "DET,7.1")
    _write_table(
        tmp_path / "2024_qb_combined.parquet",
        (
            "player_id,player_display_name,team,qb_attempts_total,qb_attempts_per_game,"
            "qb_epa_per_dropback,opp_qb_any_a,adj_qb_epa_per_dropback,qb_faced_pass_defense"
        ),
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,0.15,0.01",
    )
    _write_table(
        tmp_path / "2024_qb_ratings.parquet", "player_id,adj_qb_epa_per_dropback\n", "qb-1,0.15"
    )

    # Act
    dataset = load_season_ui_dataset(tmp_path, 2024)

    # Assert
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
        "team_rating",
        "offense_rating",
        "defense_rating",
        "special_teams_rating",
        "sos",
        "SRS",
    ]
    assert "raw_totals" not in dataset["teams"]["column_groups"]
    assert dataset["qbs"]["column_groups"]["raw_totals"] == ["qb_attempts_total"]
    assert dataset["qbs"]["column_groups"]["per_game_rates"] == ["qb_attempts_per_game"]
    assert dataset["qbs"]["column_groups"]["per_dropback_rates"] == ["qb_epa_per_dropback"]
    assert dataset["qbs"]["column_groups"]["opponent_context"] == ["opp_qb_any_a"]
    assert dataset["qbs"]["column_groups"]["ratings"] == [
        "adj_qb_epa_per_dropback",
        "qb_faced_pass_defense",
    ]


def test_load_season_ui_dataset_errors_for_incomplete_contract(tmp_path: Path) -> None:
    """Raise a clear error when a requested season is missing contract files."""
    # Arrange
    _write_table(tmp_path / "2024_combined.parquet", "team,team_rating", "DET,1.2")

    # Act & Assert
    with pytest.raises(MissingSeasonContractError):
        load_season_ui_dataset(tmp_path, 2024)


def test_load_season_ui_dataset_supports_current_qb_output_names(tmp_path: Path) -> None:
    """Support the explicit QB data schema used by the current generated Parquet files."""
    # Arrange
    _write_table(tmp_path / "2024_team_per_game_stats.parquet", "team,points_for", "DET,510")
    _write_table(
        tmp_path / "2024_qb_per_game_stats.parquet", "qb_id,qb_name,team", "qb-1,Jared Goff,DET"
    )
    _write_table(tmp_path / "2024_combined.parquet", "team,team_rating", "DET,1.2")
    _write_table(tmp_path / "2024_ratings.parquet", "team,team_rating", "DET,1.2")
    _write_table(
        tmp_path / "2024_qb_combined.parquet",
        "qb_id,qb_name,team,qb_attempts_total,qb_attempts_per_game,qb_epa_per_dropback,qopp_qb_any_a,qopp_qb_epa_per_dropback,adj_qb_epa_per_dropback",
        "qb-1,Jared Goff,DET,605,35.6,0.18,6.5,0.05,0.15",
    )
    _write_table(tmp_path / "2024_qb_ratings.parquet", "qb_id,adj_qb_epa_per_dropback", "qb-1,0.15")

    # Act
    dataset = load_season_ui_dataset(tmp_path, 2024)

    # Assert
    assert dataset["qbs"]["rows"][0]["qb_name"] == "Jared Goff"
    assert dataset["qbs"]["column_groups"]["identity"] == ["qb_id", "qb_name", "team"]
    assert dataset["qbs"]["column_groups"]["opponent_context"] == [
        "qopp_qb_any_a",
        "qopp_qb_epa_per_dropback",
    ]


def _seed_game_logs(data_dir: Path) -> None:
    """Write team and QB game logs for two teams."""
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


def test_load_team_game_log_payload_filters_rows_to_the_team(tmp_path: Path) -> None:
    """Load the team game-log payload for one selected team."""
    # Arrange
    _seed_game_logs(tmp_path)

    # Act
    team_payload = load_team_game_log_payload(tmp_path, 2024, "DET")

    # Assert
    assert [row["opponent_team"] for row in team_payload["rows"]] == ["KC", "CHI"]
    assert team_payload["column_groups"]["identity"] == ["game_id", "week", "team", "opponent_team"]
    assert team_payload["column_groups"]["results"] == [
        "points_for",
        "points_allowed",
        "point_margin",
    ]
    assert team_payload["column_groups"]["per_snap_rates"] == ["points_per_offensive_snap"]


def test_load_qb_game_log_payload_filters_rows_to_the_qb(tmp_path: Path) -> None:
    """Load the QB game-log payload for one selected QB."""
    # Arrange
    _seed_game_logs(tmp_path)

    # Act
    qb_payload = load_qb_game_log_payload(tmp_path, 2024, "qb-1")

    # Assert
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


def test_payloads_carry_registry_column_metadata(tmp_path: Path) -> None:
    """Every payload includes registry-resolved metadata for its columns."""
    # Arrange
    _write_table(tmp_path / "2024_team_per_game_stats.parquet", "team,points_for", "DET,510")
    _write_table(
        tmp_path / "2024_qb_per_game_stats.parquet", "qb_id,qb_name,team", "qb-1,Jared Goff,DET"
    )
    _write_table(
        tmp_path / "2024_combined.parquet",
        "team,points_for,opp_points_for,team_rating",
        "DET,510,390,1.2",
    )
    _write_table(tmp_path / "2024_ratings.parquet", "team,team_rating", "DET,1.2")
    _write_table(
        tmp_path / "2024_qb_combined.parquet",
        "qb_id,qb_name,team,qb_sack_rate,qopp_qb_sack_rate,adj_qb_epa_per_dropback",
        "qb-1,Jared Goff,DET,0.05,0.06,1.3",
    )
    _write_table(tmp_path / "2024_qb_ratings.parquet", "qb_id,adj_qb_epa_per_dropback", "qb-1,0.15")

    # Act
    dataset = load_season_ui_dataset(tmp_path, 2024)

    # Assert
    team_metadata = dataset["teams"]["column_metadata"]
    assert team_metadata["team_rating"]["polarity"] == "higher"
    assert team_metadata["team_rating"]["category"] == "Schedule-Adjusted Ratings"
    assert team_metadata["opp_points_for"]["contextual"] is True
    assert team_metadata["opp_points_for"]["category"] == "Overall"

    qb_metadata = dataset["qbs"]["column_metadata"]
    assert qb_metadata["qb_sack_rate"]["polarity"] == "lower"
    assert qb_metadata["qopp_qb_sack_rate"]["polarity"] == "higher"
    assert qb_metadata["qopp_qb_sack_rate"]["category"] == "Pressure, Sacks & Pocket"


def test_group_columns_follow_registry_category_order(tmp_path: Path) -> None:
    """Columns inside each group are ordered by registry category taxonomy."""
    # Arrange
    _write_table(tmp_path / "2024_team_per_game_stats.parquet", "team,points_for", "DET,510")
    _write_table(
        tmp_path / "2024_qb_per_game_stats.parquet", "qb_id,qb_name,team", "qb-1,Jared Goff,DET"
    )
    _write_table(
        tmp_path / "2024_combined.parquet",
        # Deliberately shuffled: Defense, Overall, Offense.
        "team,passing_yards_allowed,wins,passing_yards,def_sacks,team_rating",
        "DET,3300,12,4600,48,1.2",
    )
    _write_table(tmp_path / "2024_ratings.parquet", "team,team_rating", "DET,1.2")
    _write_table(
        tmp_path / "2024_qb_combined.parquet",
        "qb_id,qb_name,team,qb_sack_rate,qb_wins,adj_qb_epa_per_dropback",
        "qb-1,Jared Goff,DET,0.05,12,1.3",
    )
    _write_table(tmp_path / "2024_qb_ratings.parquet", "qb_id,adj_qb_epa_per_dropback", "qb-1,0.15")

    # Act
    dataset = load_season_ui_dataset(tmp_path, 2024)

    # Assert
    team_stats_columns = dataset["teams"]["column_groups"]["per_game_rates"]
    # Overall (wins) before Offense (passing_yards) before Defense columns.
    assert team_stats_columns.index("wins") < team_stats_columns.index("passing_yards")
    assert team_stats_columns.index("passing_yards") < team_stats_columns.index(
        "passing_yards_allowed"
    )
    assert team_stats_columns.index("passing_yards_allowed") < team_stats_columns.index("def_sacks")


def test_discover_available_seasons_ignores_files_without_a_season_prefix(tmp_path: Path) -> None:
    # Arrange
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "notes.parquet")

    # Act
    seasons = discover_available_seasons(tmp_path)

    # Assert
    assert seasons == []


def test_group_columns_put_unknown_columns_last(tmp_path: Path) -> None:
    # Arrange
    _write_table(tmp_path / "2024_team_per_game_stats.parquet", "team,points_for", "DET,510")
    _write_table(
        tmp_path / "2024_qb_per_game_stats.parquet", "qb_id,qb_name,team", "qb-1,Jared Goff,DET"
    )
    _write_table(
        tmp_path / "2024_combined.parquet",
        "team,zz_unregistered,points_for,team_rating",
        "DET,1,510,1.2",
    )
    _write_table(tmp_path / "2024_ratings.parquet", "team,team_rating", "DET,1.2")
    _write_table(tmp_path / "2024_qb_combined.parquet", "qb_id,qb_name,team", "qb-1,Jared Goff,DET")
    _write_table(tmp_path / "2024_qb_ratings.parquet", "qb_id,team", "qb-1,DET")

    # Act
    dataset = load_season_ui_dataset(tmp_path, 2024)

    # Assert
    assert dataset["teams"]["column_groups"]["per_game_rates"][-1] == "zz_unregistered"


def test_load_team_game_log_payload_without_the_file_raises_missing_contract(
    tmp_path: Path,
) -> None:
    # Act & Assert
    with pytest.raises(MissingSeasonContractError, match="team_game_logs"):
        load_team_game_log_payload(tmp_path, 2024, "DET")


def test_load_qb_game_log_payload_without_the_id_column_raises_lookup_error(
    tmp_path: Path,
) -> None:
    # Arrange
    _write_table(tmp_path / "2024_qb_game_logs.parquet", "game_id,week,team", "g1,1,DET")

    # Act & Assert
    with pytest.raises(MissingEntityRowsError, match="qb_id"):
        load_qb_game_log_payload(tmp_path, 2024, "qb-1")


def test_load_team_game_log_payload_unknown_team_raises_lookup_error(tmp_path: Path) -> None:
    # Arrange
    _seed_game_logs(tmp_path)

    # Act & Assert
    with pytest.raises(MissingEntityRowsError, match="NOPE"):
        load_team_game_log_payload(tmp_path, 2024, "NOPE")


def _seed_rating_histories(data_dir: Path) -> None:
    """Write team and QB rating histories for two teams over two weeks."""
    _write_table(
        data_dir / "2024_ratings_by_week.parquet",
        ("week,team,games_played,offense_rating,defense_rating,special_teams_rating,team_rating"),
        "2,DET,2,3.0,1.0,0.5,4.5\n1,DET,1,1.0,0.5,0.2,1.7\n1,KC,1,-0.5,0.0,0.1,-0.4",
    )
    _write_table(
        data_dir / "2024_qb_ratings_by_week.parquet",
        "week,qb_id,qb_games_played,qb_dropbacks,adj_qb_epa_per_dropback",
        "1,qb-1,1,38,0.05\n2,qb-1,2,74,0.09\n1,qb-2,1,41,0.02",
    )


def test_load_team_rating_history_payload_lists_the_teams_weeks_in_order(
    tmp_path: Path,
) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)

    # Act
    payload = load_team_rating_history_payload(tmp_path, 2024, "DET")

    # Assert
    assert [row["week"] for row in payload["rows"]] == [1, 2]


def test_load_team_rating_history_payload_puts_the_headline_rating_first(
    tmp_path: Path,
) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)

    # Act
    payload = load_team_rating_history_payload(tmp_path, 2024, "DET")

    # Assert
    assert payload["column_groups"] == {
        "identity": ["week", "team"],
        "sample": ["games_played"],
        "ratings": ["team_rating", "offense_rating", "defense_rating", "special_teams_rating"],
    }


def test_load_qb_rating_history_payload_groups_sample_and_rating_columns(
    tmp_path: Path,
) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)

    # Act
    payload = load_qb_rating_history_payload(tmp_path, 2024, "qb-1")

    # Assert
    assert [row["qb_dropbacks"] for row in payload["rows"]] == [38, 74]
    assert payload["column_groups"] == {
        "identity": ["week", "qb_id"],
        "sample": ["qb_games_played", "qb_dropbacks"],
        "ratings": ["adj_qb_epa_per_dropback"],
    }


def test_load_team_rating_history_payload_without_the_file_raises_missing_contract(
    tmp_path: Path,
) -> None:
    # Act & Assert
    with pytest.raises(MissingSeasonContractError, match="ratings_by_week"):
        load_team_rating_history_payload(tmp_path, 2024, "DET")


def test_load_qb_rating_history_payload_unknown_qb_raises_lookup_error(tmp_path: Path) -> None:
    # Arrange
    _seed_rating_histories(tmp_path)

    # Act & Assert
    with pytest.raises(MissingEntityRowsError, match="rating-history rows for qb-9"):
        load_qb_rating_history_payload(tmp_path, 2024, "qb-9")
