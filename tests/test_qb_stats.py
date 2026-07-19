"""Tests for nfl_sos_ratings.qb_stats."""

import polars as pl
import pytest

from nfl_sos_ratings import qb_stats


def test_compute_qb_season_stats_includes_volume_and_eligibility() -> None:
    """Verify QB season aggregation computes averages, volume, and eligibility flags."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B", "QB_C"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B", "QB C"],
            "qb_attempts": [30, 28, 20, 22, 18],
            "qb_passer_rating": [100.0, 102.0, 92.0, 95.0, 90.0],
            "qb_completion_percentage_above_expectation": [2.0, 3.0, 0.0, 0.5, -1.0],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "points_for": [24, 17, 21, 20, 14],
            "points_allowed": [17, 20, 14, 21, 14],
        }
    )

    result = qb_stats.compute_qb_season_stats(
        qb_df,
        weekly_df=weekly_df,
        min_games=2,
        min_attempts=55,
    )

    qb_a = result.filter(pl.col("qb_id") == "QB_A")
    qb_b = result.filter(pl.col("qb_id") == "QB_B")

    assert qb_a.select("qb_games_played").item() == 2
    assert qb_a.select("qb_attempts_total").item() == 58
    assert qb_a.select("qb_is_eligible").item() is True
    assert qb_a.select("qb_passer_rating").item() == 101.0
    assert qb_a.select("qb_win_pct").item() == 0.5

    assert qb_b.select("qb_games_played").item() == 2
    assert qb_b.select("qb_attempts_total").item() == 42
    assert qb_b.select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_assigns_results_to_primary_qb_only() -> None:
    """Verify QB wins are assigned only to the primary QB for each team-week."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 1],
            "qb_id": ["QB_A", "QB_B"],
            "qb_name": ["QB A", "QB B"],
            "qb_attempts": [24, 8],
            "qb_dropbacks": [26, 10],
            "qb_offense_snaps": [48, 12],
            "qb_passer_rating": [100.0, 80.0],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN"],
            "week": [1],
            "points_for": [24],
            "points_allowed": [17],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_win_pct").item() == 1.0
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_wins").item() == 1
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_win_pct").item() == 0.5
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_wins").item() == 0


def test_compute_qb_season_stats_sums_late_game_totals() -> None:
    """Verify season QB summaries expose summed 4QC and GWD totals."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [20, 22],
            "qb_fourth_quarter_comeback": [1, 0],
            "qb_game_winning_drive": [1, 0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_fourth_quarter_comebacks").item() == 1
    assert result.select("qb_game_winning_drives").item() == 1


def test_compute_qb_season_stats_exposes_explicit_per_game_and_total_columns() -> None:
    """Verify season summaries do not mix per-game averages under raw season column names."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [20, 30],
            "qb_completions": [10, 20],
            "qb_pass_yards": [120.0, 210.0],
            "qb_pass_touchdowns": [1.0, 2.0],
            "qb_interceptions": [0.0, 1.0],
            "qb_dropbacks": [22, 33],
            "qb_offense_snaps": [40, 50],
            "qb_sacks": [2.0, 3.0],
            "qb_sack_yards_lost": [10.0, 15.0],
            "qb_sack_fumbles_lost": [0.0, 1.0],
            "qb_passing_epa": [4.0, 6.0],
            "qb_fourth_quarter_comeback": [1, 0],
            "qb_game_winning_drive": [1, 0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert "qb_attempts" not in result.columns
    assert "qb_completions" not in result.columns
    assert "qb_pass_yards" not in result.columns
    assert result.select("qb_attempts_per_game").item() == 25.0
    assert result.select("qb_completions_per_game").item() == 15.0
    assert result.select("qb_pass_yards_per_game").item() == 165.0
    assert result.select("qb_completions_total").item() == 30
    assert result.select("qb_attempts_total").item() == 50
    assert result.select("qb_pass_yards_total").item() == 330.0


def test_compute_qb_season_stats_handles_missing_attempts_column() -> None:
    """Verify missing qb_attempts defaults attempts total to zero and marks ineligible."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_passer_rating": [100.0, 98.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(
        qb_df,
        min_games=2,
        min_attempts=1,
    )

    assert result.select("qb_attempts_total").item() == 0
    assert result.select("qb_is_eligible").item() is False
    assert result.select("qb_games_played").item() == 2


def test_compute_qb_season_stats_defaults_to_238_attempt_threshold() -> None:
    """Verify default QB eligibility requires a full-season 238-attempt threshold."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC"],
            "week": [1, 2, 1, 2],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B"],
            "qb_attempts": [119, 119, 119, 118],
            "qb_passer_rating": [100.0, 101.0, 99.0, 98.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df, min_games=2)

    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_is_eligible").item() is True
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_uses_16_game_qualifier_before_2021() -> None:
    """Verify pre-2021 seasons use the 16-game, 224-attempt qualifier.

    Weekly team data should drive the default season-length-based eligibility threshold.
    """
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["SF"] * 16 + ["JAX"] * 16,
            "week": list(range(1, 17)) + list(range(1, 17)),
            "qb_id": ["QB_A"] * 16 + ["QB_B"] * 16,
            "qb_name": ["QB A"] * 16 + ["QB B"] * 16,
            "qb_attempts": [14] * 16 + ([14] * 15) + [13],
            "qb_passer_rating": [95.0] * 32,
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["SF"] * 16 + ["JAX"] * 16,
            "week": list(range(1, 17)) + list(range(1, 17)),
            "points_for": [20] * 32,
            "points_allowed": [17] * 32,
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_attempts_total").item() == 224
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_is_eligible").item() is True
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_attempts_total").item() == 223
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_default_eligibility_is_attempt_based() -> None:
    """Verify default qualification does not add a separate games-played cutoff."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN"] * 7,
            "week": list(range(1, 8)),
            "qb_id": ["QB_A"] * 7,
            "qb_name": ["QB A"] * 7,
            "qb_attempts": [34] * 7,
            "qb_passer_rating": [100.0] * 7,
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_games_played").item() == 7
    assert result.select("qb_attempts_total").item() == 238
    assert result.select("qb_is_eligible").item() is True


def test_compute_qb_season_stats_derives_attempt_normalized_rates() -> None:
    """Verify season efficiency rates use total attempts instead of game-average volume."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [10, 30],
            "qb_pass_yards": [100.0, 150.0],
            "qb_pass_touchdowns": [2.0, 1.0],
            "qb_interceptions": [0.0, 2.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_yards_per_attempt").item() == 6.25
    assert result.select("qb_touchdown_rate").item() == 0.075
    assert result.select("qb_interception_rate").item() == 0.05


def test_compute_qb_season_stats_derives_dropback_metrics_and_totals() -> None:
    """Verify season QB summaries expose dropback-based totals and advanced rates."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [10, 30],
            "qb_dropbacks": [12, 34],
            "qb_offense_snaps": [60, 65],
            "qb_pass_yards": [100.0, 150.0],
            "qb_pass_touchdowns": [2.0, 1.0],
            "qb_interceptions": [0.0, 2.0],
            "qb_sacks": [2.0, 1.0],
            "qb_sack_yards_lost": [12.0, 7.0],
            "qb_passing_epa": [5.0, 3.0],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df)

    assert result.select("qb_dropbacks_total").item() == 46
    assert result.select("qb_offense_snaps_total").item() == 125
    assert result.select("qb_td_int_differential").item() == 1.0
    assert result.select("qb_epa_per_dropback").item() == pytest.approx(8.0 / 46.0)
    assert result.select("qb_any_a").item() == pytest.approx(201.0 / 43.0)
    assert result.select("qb_pass_yards_per_dropback").item() == pytest.approx(250.0 / 46.0)
    assert result.select("qb_sack_rate").item() == pytest.approx(3.0 / 46.0)
    assert result.select("qb_td_int_margin_rate").item() == pytest.approx(1.0 / 46.0)


def test_compute_qb_season_stats_defaults_win_pct_when_results_unavailable() -> None:
    """Verify qb_win_pct does not become NaN when team/week score joins fail."""
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["LAC", "LAC"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [30, 28],
            "qb_passer_rating": [90.0, 92.0],
        }
    )
    # Intentionally mismatched team code so joined points are unavailable.
    weekly_df = pl.DataFrame(
        {
            "team": ["SD", "SD"],
            "week": [1, 2],
            "points_for": [24, 21],
            "points_allowed": [20, 17],
        }
    )

    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    assert result.select("qb_win_pct").item() == 0.5


def test_compute_qb_game_volumes_from_pbp_combines_dropbacks_and_snap_counts() -> None:
    """Verify QB game volumes union PBP dropbacks with QB snap-count rows."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 4,
            "week": [1] * 4,
            "posteam": ["DEN", "DEN", "DEN", "KC"],
            "passer_player_id": ["GSIS_A", "GSIS_A", "GSIS_B", "GSIS_C"],
            "passer_player_name": ["Starter QB", "Starter QB", "Backup QB", "KC QB"],
            "qb_dropback": [1, 1, 1, 1],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": [
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
            ],
            "week": [1, 1, 1, 1],
            "team": ["DEN", "DEN", "DEN", "KC"],
            "player": ["Starter QB", "Backup QB", "Runner QB", "KC QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_D", "PFR_C"],
            "position": ["QB", "QB", "QB", "QB"],
            "offense_snaps": [42.0, 16.0, 3.0, 58.0],
        }
    )

    result = qb_stats.compute_qb_game_volumes_from_pbp(pbp, snap_counts).sort(
        ["team_abbr", "qb_name"]
    )

    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Backup QB",
            "qb_id": "GSIS_B",
            "snap_player_id": "PFR_B",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 16,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Runner QB",
            "qb_id": None,
            "snap_player_id": "PFR_D",
            "qb_dropbacks": 0,
            "qb_offense_snaps": 3,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Starter QB",
            "qb_id": "GSIS_A",
            "snap_player_id": "PFR_A",
            "qb_dropbacks": 2,
            "qb_offense_snaps": 42,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "KC",
            "qb_name": "KC QB",
            "qb_id": "GSIS_C",
            "snap_player_id": "PFR_C",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 58,
        },
    ]


def test_compute_qb_game_stats_from_pbp_derives_dropback_metrics() -> None:
    """Verify PBP-derived QB game stats carry dropbacks, snaps, and core passing metrics."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 4,
            "week": [1] * 4,
            "posteam": ["DEN", "DEN", "DEN", "DEN"],
            "passer_player_id": ["GSIS_A", "GSIS_A", "GSIS_A", "GSIS_B"],
            "passer_player_name": ["Starter QB", "Starter QB", "Starter QB", "Backup QB"],
            "qb_dropback": [1, 1, 1, 1],
            "pass": [1, 0, 1, 1],
            "complete_pass": [1, 0, 0, 0],
            "passing_yards": [20.0, 0.0, 0.0, 0.0],
            "yards_gained": [20.0, -7.0, 0.0, 0.0],
            "pass_touchdown": [1, 0, 0, 0],
            "interception": [0, 0, 0, 1],
            "sack": [0, 1, 0, 0],
            "fumble_lost": [0, 1, 0, 0],
            "qb_epa": [2.0, -1.0, -0.5, -2.0],
            "cpoe": [5.0, None, -3.0, -6.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "team": ["DEN", "DEN", "DEN"],
            "player": ["Starter QB", "Backup QB", "Runner QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_D"],
            "position": ["QB", "QB", "QB"],
            "offense_snaps": [42.0, 16.0, 3.0],
        }
    )

    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts).sort("qb_name")

    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Backup QB",
            "qb_id": "GSIS_B",
            "snap_player_id": "PFR_B",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 16,
            "qb_attempts": 1,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 1,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": -2.0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": -2.0,
            "qb_pass_yards_per_dropback": 0.0,
            "qb_td_int_margin_rate": -1.0,
            "qb_sack_rate": 0.0,
            "qb_any_a": -45.0,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": -6.0,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Runner QB",
            "qb_id": None,
            "snap_player_id": "PFR_D",
            "qb_dropbacks": 0,
            "qb_offense_snaps": 3,
            "qb_attempts": 0,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 0,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": 0.0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": None,
            "qb_pass_yards_per_dropback": None,
            "qb_td_int_margin_rate": None,
            "qb_sack_rate": None,
            "qb_any_a": None,
            "qb_scramble_rate": None,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": None,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Starter QB",
            "qb_id": "GSIS_A",
            "snap_player_id": "PFR_A",
            "qb_dropbacks": 3,
            "qb_offense_snaps": 42,
            "qb_attempts": 2,
            "qb_completions": 1,
            "qb_pass_yards": 20.0,
            "qb_pass_touchdowns": 1,
            "qb_interceptions": 0,
            "qb_sacks": 1,
            "qb_sack_yards_lost": 7.0,
            "qb_sack_fumbles_lost": 1,
            "qb_passing_epa": 0.5,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": 1 / 6,
            "qb_pass_yards_per_dropback": 20.0 / 3.0,
            "qb_td_int_margin_rate": 1 / 3,
            "qb_sack_rate": 1 / 3,
            "qb_any_a": 11.0,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": 1.0,
        },
    ]


def test_compute_qb_game_stats_from_pbp_splits_designed_runs_scrambles_and_kneels() -> None:
    """Verify QB rushing PBP splits exclude scrambles and kneels from designed-run value."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_04_BUF_MIA"] * 5,
            "week": [4] * 5,
            "posteam": ["BUF"] * 5,
            "passer_player_id": ["GSIS_A", "GSIS_A", None, "GSIS_A", "GSIS_A"],
            "passer_player_name": [
                "Dual Threat QB",
                "Dual Threat QB",
                None,
                "Dual Threat QB",
                "Dual Threat QB",
            ],
            "rusher_player_id": [None, None, "GSIS_A", "GSIS_A", "GSIS_A"],
            "qb_dropback": [1, 1, 0, 1, 0],
            "pass": [1, 0, 0, 0, 0],
            "complete_pass": [1, 0, 0, 0, 0],
            "passing_yards": [15.0, 0.0, 0.0, 0.0, 0.0],
            "yards_gained": [15.0, 0.0, 8.0, 12.0, -1.0],
            "pass_touchdown": [0, 0, 0, 0, 0],
            "interception": [0, 0, 0, 0, 0],
            "sack": [0, 0, 0, 0, 0],
            "fumble_lost": [0, 0, 0, 0, 0],
            "qb_epa": [1.5, 0.0, 1.2, 0.8, -0.9],
            "cpoe": [4.0, None, None, None, None],
            "rush": [0, 0, 1, 1, 1],
            "qb_scramble": [0, 0, 0, 1, 0],
            "qb_kneel": [0, 0, 0, 0, 1],
            "qb_spike": [0, 1, 0, 0, 0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_04_BUF_MIA"],
            "week": [4],
            "team": ["BUF"],
            "player": ["Dual Threat QB"],
            "pfr_player_id": ["PFR_A"],
            "position": ["QB"],
            "offense_snaps": [58.0],
        }
    )
    qb_identity = pl.DataFrame(
        {
            "qb_id": ["GSIS_A"],
            "snap_player_id": ["PFR_A"],
            "qb_name": ["Dual Threat QB"],
            "qb_position": ["QB"],
        }
    )

    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts, qb_identity)
    row = result.row(0, named=True)

    assert row["qb_dropbacks"] == 3
    assert row["qb_passing_epa"] == pytest.approx(2.3)
    assert row["qb_designed_carries"] == 1
    assert row["qb_designed_rush_yards"] == 8.0
    assert row["qb_designed_rush_epa"] == pytest.approx(1.2)
    assert row["qb_scrambles"] == 1
    assert row["qb_scramble_yards"] == 12.0
    assert row["qb_kneels"] == 1
    assert row["qb_scramble_rate"] == pytest.approx(1.0 / 3.0)
    assert row["qb_yards_per_scramble"] == pytest.approx(12.0)
    assert row["qb_designed_yards_per_carry"] == pytest.approx(8.0)
    assert row["qb_designed_epa_per_carry"] == pytest.approx(1.2)


def test_compute_qb_season_stats_reconciles_designed_rush_components_for_multi_team_qb() -> None:
    """Verify designed-run splits aggregate by QB across teams and reconcile to official carries."""
    qb_df = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "week": [1, 2, 3],
            "team_abbr": ["BAL", "BAL", "HOU"],
            "qb_id": ["qb-1", "qb-1", "qb-1"],
            "qb_name": ["Dual Threat QB", "Dual Threat QB", "Dual Threat QB"],
            "qb_dropbacks": [30, 25, 20],
            "qb_attempts": [28, 23, 18],
            "qb_carries": [5, 4, 6],
            "qb_rushing_yards": [31.0, 25.0, 40.0],
            "qb_designed_carries": [2, 1, 3],
            "qb_designed_rush_yards": [14.0, 7.0, 18.0],
            "qb_designed_rush_epa": [1.0, 0.4, 1.6],
            "qb_scrambles": [2, 2, 2],
            "qb_scramble_yards": [18.0, 19.0, 24.0],
            "qb_kneels": [1, 1, 1],
        }
    )

    season = qb_stats.compute_qb_season_stats(qb_df)
    row = season.to_dicts()[0]

    assert season.height == 1
    assert row["team"] == "BAL"
    assert row["qb_carries_total"] == 15
    assert row["qb_designed_carries_total"] == 6
    assert row["qb_scrambles_total"] == 6
    assert row["qb_kneels_total"] == 3
    assert (
        row["qb_designed_carries_total"] + row["qb_scrambles_total"] + row["qb_kneels_total"]
        == row["qb_carries_total"]
    )
    assert row["qb_designed_rush_yards_total"] == 39.0
    assert row["qb_designed_rush_epa_total"] == pytest.approx(3.0)
    assert row["qb_designed_carries_per_game"] == pytest.approx(2.0)
    assert row["qb_designed_rush_yards_per_game"] == pytest.approx(13.0)
    assert row["qb_designed_yards_per_carry"] == pytest.approx(6.5)
    assert row["qb_designed_epa_per_carry"] == pytest.approx(0.5)
    assert row["qb_scramble_rate"] == pytest.approx(6.0 / 75.0)
    assert row["qb_yards_per_scramble"] == pytest.approx(61.0 / 6.0)


def test_compute_qb_game_stats_from_pbp_assigns_4qc_and_gwd_to_primary_qb() -> None:
    """Verify late-game comeback flags are assigned only to the primary QB row."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "posteam": ["DEN", "DEN", "KC"],
            "passer_player_id": ["GSIS_B", "GSIS_A", "GSIS_C"],
            "passer_player_name": ["Backup QB", "Starter QB", "KC QB"],
            "qb_dropback": [1, 1, 1],
            "pass": [1, 1, 1],
            "complete_pass": [0, 1, 0],
            "passing_yards": [0.0, 30.0, 0.0],
            "yards_gained": [0.0, 30.0, 0.0],
            "pass_touchdown": [0, 1, 0],
            "interception": [0, 0, 0],
            "sack": [0, 0, 0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [-0.5, 4.0, -1.0],
            "cpoe": [-2.0, 7.0, -3.0],
            "qtr": [2, 4, 4],
            "score_differential": [0, -3, -4],
            "score_differential_post": [0, 4, -4],
            "posteam_score": [7, 17, 20],
            "posteam_score_post": [7, 24, 20],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "team": ["DEN", "DEN", "KC"],
            "player": ["Starter QB", "Backup QB", "KC QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_C"],
            "position": ["QB", "QB", "QB"],
            "offense_snaps": [50.0, 10.0, 60.0],
        }
    )

    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts).sort("qb_name")

    assert (
        result.filter(pl.col("qb_name") == "Starter QB").select("qb_fourth_quarter_comeback").item()
        == 1
    )
    assert (
        result.filter(pl.col("qb_name") == "Starter QB").select("qb_game_winning_drive").item() == 1
    )
    assert (
        result.filter(pl.col("qb_name") == "Backup QB").select("qb_fourth_quarter_comeback").item()
        == 0
    )
    assert (
        result.filter(pl.col("qb_name") == "Backup QB").select("qb_game_winning_drive").item() == 0
    )
    assert (
        result.filter(pl.col("qb_name") == "KC QB").select("qb_fourth_quarter_comeback").item() == 0
    )
    assert result.filter(pl.col("qb_name") == "KC QB").select("qb_game_winning_drive").item() == 0


def test_compute_qb_game_stats_from_pbp_does_not_assign_4qc_or_gwd_in_loss() -> None:
    """Verify late-game flags are cleared if the team ultimately loses after a late lead."""
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_03_NYJ_NE", "2025_03_NYJ_NE"],
            "week": [3, 3],
            "posteam": ["NYJ", "NE"],
            "passer_player_id": ["GSIS_TY", "GSIS_NE"],
            "passer_player_name": ["Tyrod Taylor", "NE QB"],
            "qb_dropback": [1, 1],
            "pass": [1, 1],
            "complete_pass": [1, 1],
            "passing_yards": [20.0, 18.0],
            "yards_gained": [20.0, 18.0],
            "pass_touchdown": [1, 1],
            "interception": [0, 0],
            "sack": [0, 0],
            "fumble_lost": [0, 0],
            "qb_epa": [2.0, 1.5],
            "cpoe": [4.0, 3.0],
            "qtr": [4, 4],
            "score_differential": [-3, -4],
            "score_differential_post": [4, 2],
            "posteam_score": [20, 23],
            "posteam_score_post": [27, 29],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_03_NYJ_NE", "2025_03_NYJ_NE"],
            "week": [3, 3],
            "team": ["NYJ", "NE"],
            "player": ["Tyrod Taylor", "NE QB"],
            "pfr_player_id": ["TaylTy00", "NEQb00"],
            "position": ["QB", "QB"],
            "offense_snaps": [60.0, 58.0],
        }
    )

    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts)
    tyrod_row = result.filter(pl.col("qb_name") == "Tyrod Taylor")

    assert tyrod_row.select("qb_fourth_quarter_comeback").item() == 0
    assert tyrod_row.select("qb_game_winning_drive").item() == 0


def test_compute_qb_season_stats_aggregates_rushing_and_completion_rates() -> None:
    """Verify season totals, per-game fields, and rates for the rushing family."""
    qb_df = pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "week": [1, 2],
            "team_abbr": ["DEN", "DEN"],
            "qb_id": ["qb-1", "qb-1"],
            "qb_name": ["John Doe", "John Doe"],
            "qb_dropbacks": [30, 34],
            "qb_attempts": [28, 32],
            "qb_completions": [20, 25],
            "qb_pass_yards": [220.0, 260.0],
            "qb_pass_touchdowns": [2, 1],
            "qb_interceptions": [0, 1],
            "qb_sacks": [1, 1],
            "qb_sack_yards_lost": [7.0, 6.0],
            "qb_passing_epa": [5.0, 4.0],
            "qb_carries": [6, 4],
            "qb_rushing_yards": [42.0, 18.0],
            "qb_rushing_tds": [1, 0],
            "qb_rushing_first_downs": [3, 1],
            "qb_rushing_epa": [2.0, 0.5],
            "qb_rushing_fumbles": [0, 1],
            "qb_rushing_fumbles_lost": [0, 1],
            "qb_rushing_2pt_conversions": [0, 0],
        }
    )

    season = qb_stats.compute_qb_season_stats(qb_df)
    row = season.to_dicts()[0]

    assert row["qb_carries_total"] == 10
    assert row["qb_rushing_yards_total"] == 60.0
    assert row["qb_rushing_tds_total"] == 1
    assert row["qb_carries_per_game"] == 5.0
    assert row["qb_rushing_yards_per_game"] == 30.0
    assert row["qb_completion_pct"] == 45.0 / 60.0
    assert row["qb_yards_per_carry"] == 6.0
    assert row["qb_epa_per_carry"] == 0.25
