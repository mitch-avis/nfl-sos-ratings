"""Regression anchors for the QSoS schedule-strength audit.

The default tests read a frozen slice of the 2025 published outputs checked in under
``tests/fixtures/qsos_anchor_2025/`` (rebuild it with ``build_fixture.py`` there), so they pass on
a fresh clone and do not move when nflverse revises upstream data. The ``published_data`` tests
repeat the checks against the live ``data/`` outputs; run them with ``pytest -m published_data``
after regenerating ``data/``.
"""

from pathlib import Path

import polars as pl
import pytest

from nfl_sos_ratings.validation.diagnostics import compute_qb_schedule_lens_anchor

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qsos_anchor_2025"
_DATA_DIR = Path("data")
_SEASON = 2025
_ANCHOR_QBS = ["Drake Maye", "Tyler Shough", "Joe Flacco", "J.J. McCarthy"]

# Equal-game opponent means for the frozen fixture, as printed by build_fixture.py.
_FIXTURE_ANCHOR: dict[str, dict[str, float | int]] = {
    "Drake Maye": {
        "games": 17,
        "avg_opp_SaCR": -0.701,
        "avg_opp_SaDR": -0.46499999999999997,
        "avg_opp_SRS": -4.2849441176470595,
    },
    "Tyler Shough": {
        "games": 11,
        "avg_opp_SaCR": -0.3287272727272727,
        "avg_opp_SaDR": -0.20190909090909093,
        "avg_opp_SRS": -1.5668308181818185,
    },
    "Joe Flacco": {
        "games": 13,
        "avg_opp_SaCR": -0.15484615384615388,
        "avg_opp_SaDR": -0.3684615384615384,
        "avg_opp_SRS": -1.695016076923077,
    },
    "J.J. McCarthy": {
        "games": 10,
        "avg_opp_SaCR": 0.12700000000000006,
        "avg_opp_SaDR": -0.5353,
        "avg_opp_SRS": -0.9946244999999999,
    },
}


def _column_mean(frame: pl.DataFrame, column: str) -> float:
    """Return the mean of one numeric column as a float."""
    return float(frame.select(pl.col(column).cast(pl.Float64).mean()).item())


def _independent_anchor(data_dir: Path) -> dict[str, dict[str, float | int]]:
    """Recompute the anchor with a plain join and mean, independent of the audit helper."""
    qb_logs = pl.read_parquet(data_dir / f"{_SEASON}_qb_game_logs.parquet")
    team_ratings = pl.read_parquet(data_dir / f"{_SEASON}_ratings.parquet").select(
        ["team", "SaCR", "SaDR", "SRS"]
    )
    anchor: dict[str, dict[str, float | int]] = {}
    for qb_name in _ANCHOR_QBS:
        joined = (
            qb_logs.filter(pl.col("qb_name") == qb_name)
            .select(["week", "opponent_team"])
            .join(team_ratings, left_on="opponent_team", right_on="team", how="left")
        )
        anchor[qb_name] = {
            "games": joined.height,
            "avg_opp_SaCR": _column_mean(joined, "SaCR"),
            "avg_opp_SaDR": _column_mean(joined, "SaDR"),
            "avg_opp_SRS": _column_mean(joined, "SRS"),
        }
    return anchor


def _helper_anchor(data_dir: Path) -> dict[str, dict[str, float | int]]:
    """Return the audit helper's anchor rows keyed by QB name."""
    frame = compute_qb_schedule_lens_anchor(data_dir, _SEASON, qb_names=_ANCHOR_QBS)
    return {
        str(row["qb_name"]): {key: value for key, value in row.items() if key != "qb_name"}
        for row in frame.iter_rows(named=True)
    }


def _assert_anchor_equal(
    observed: dict[str, dict[str, float | int]], expected: dict[str, dict[str, float | int]]
) -> None:
    """Assert two anchors agree on QBs, game counts, and opponent means."""
    assert sorted(observed) == sorted(expected)
    for qb_name, expected_row in expected.items():
        assert observed[qb_name]["games"] == expected_row["games"]
        for key in ("avg_opp_SaCR", "avg_opp_SaDR", "avg_opp_SRS"):
            assert observed[qb_name][key] == pytest.approx(expected_row[key])


def _flacco_teams(data_dir: Path) -> tuple[int, list[str]]:
    """Return Joe Flacco's game count and sorted teams from the QB game logs."""
    qb_logs = pl.read_parquet(data_dir / f"{_SEASON}_qb_game_logs.parquet")
    flacco_logs = qb_logs.filter(pl.col("qb_name") == "Joe Flacco")
    return flacco_logs.height, flacco_logs.select("team").unique().sort(
        "team"
    ).to_series().to_list()


def test_fixture_anchor_matches_pinned_opponent_quality() -> None:
    """An independent join over the frozen fixture reproduces the pinned four-QB anchor."""
    _assert_anchor_equal(_independent_anchor(_FIXTURE_DIR), _FIXTURE_ANCHOR)


def test_anchor_helper_matches_independent_aggregation_on_fixture() -> None:
    """The reusable audit helper agrees with the independent aggregation on the fixture."""
    _assert_anchor_equal(_helper_anchor(_FIXTURE_DIR), _independent_anchor(_FIXTURE_DIR))


def test_fixture_keeps_joe_flacco_games_from_both_teams() -> None:
    """Joe Flacco's 2025 row spans all 13 played games across both team stints."""
    assert _flacco_teams(_FIXTURE_DIR) == (13, ["CIN", "CLE"])


@pytest.mark.published_data
def test_published_anchor_helper_matches_independent_aggregation() -> None:
    """On the live outputs, the helper agrees with the independent aggregation and game counts."""
    live = _independent_anchor(_DATA_DIR)
    _assert_anchor_equal(_helper_anchor(_DATA_DIR), live)
    assert {qb: row["games"] for qb, row in live.items()} == {
        qb: row["games"] for qb, row in _FIXTURE_ANCHOR.items()
    }


@pytest.mark.published_data
def test_published_outputs_keep_joe_flacco_games_from_both_teams() -> None:
    """The live outputs still credit Joe Flacco with all 13 games across both teams."""
    assert _flacco_teams(_DATA_DIR) == (13, ["CIN", "CLE"])
