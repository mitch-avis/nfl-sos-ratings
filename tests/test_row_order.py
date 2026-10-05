"""Tests for the fixed row order of the pipeline's data files."""

from nfl_sos_ratings.row_order import row_identity_keys


def test_row_identity_keys_order_team_bins_by_unit_then_bin() -> None:
    # Arrange
    columns = ["game_id", "week", "team", "opponent_team", "wp_unit", "wp_bin", "wp_bin_plays"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("team", "week", "game_id", "wp_unit", "wp_bin")


def test_row_identity_keys_put_the_passer_before_week_and_bin() -> None:
    # Arrange
    columns = ["game_id", "week", "qb_id", "wp_bin", "qb_wp_bin_dropbacks"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("qb_id", "week", "game_id", "wp_bin")


def test_row_identity_keys_are_empty_without_an_identity_column() -> None:
    # Act
    keys = row_identity_keys(["week", "wp_bin"])

    # Assert
    assert keys == ()
