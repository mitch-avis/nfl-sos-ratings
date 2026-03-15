import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings import ratings


def test_rating_helpers_cover_edge_cases() -> None:
    assert np.allclose(ratings._zscore([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(ratings._zscore([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 0.0]))

    df = pl.DataFrame({"team": ["DEN"], "value": [1.5]})
    value_col = ratings._col(df, "value")
    assert value_col is not None
    assert np.allclose(value_col, np.array([1.5]))
    assert ratings._col(df, "missing") is None

    composite = ratings._build_composite(
        pl.DataFrame({"value": [1.0, 2.0, 3.0]}),
        [("missing", 0.5, True), ("value", 0.5, True)],
    )
    assert np.allclose(composite, np.array([-0.5, 0.0, 0.5]))


def test_derive_weights_builds_weighted_composite_and_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {
            "stat_a": [1.0, 2.0, 3.0, 4.0],
            "stat_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    win_pct = np.array([0.25, 0.5, 0.75, 1.0])

    weighted = ratings._derive_weights(
        df,
        [("stat_a", True), ("stat_b", False)],
        win_pct,
        "Offensive",
    )
    composite = ratings._build_composite(df, weighted)

    assert len(weighted) == 2
    assert np.allclose(composite, np.array([-1.161895, -0.387298, 0.387298, 1.161895]))
    assert "Offensive stat weights" in capsys.readouterr().out

    fallback = ratings._derive_weights(
        pl.DataFrame({"stat_c": [1.0, 2.0, 1.0]}),
        [("stat_c", True)],
        np.array([0.3, 0.5, 0.7]),
        "Defensive",
    )

    assert fallback == [("stat_c", 1.0, True)]
    assert "using equal weights" in capsys.readouterr().out


def test_compute_ratings_with_real_inputs() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "win_pct": [0.25, 0.5, 0.75, 1.0],
            "passing_epa": [0.0, 0.1, 0.2, 0.3],
            "rushing_epa": [0.0, 0.05, 0.1, 0.15],
            "points_for": [14, 20, 26, 32],
            "total_yards": [280, 320, 360, 400],
            "passing_yards": [180, 210, 240, 270],
            "rushing_yards": [100, 110, 120, 130],
            "passing_tds": [1, 2, 3, 4],
            "rushing_tds": [0, 1, 1, 2],
            "passing_first_downs": [8, 10, 12, 14],
            "rushing_first_downs": [4, 5, 6, 7],
            "passing_cpoe": [-1.0, 0.0, 1.0, 2.0],
            "sacks_suffered": [4, 3, 2, 1],
            "passing_interceptions": [2, 1, 1, 0],
            "sack_fumbles_lost": [1, 1, 0, 0],
            "rushing_fumbles_lost": [1, 0, 0, 0],
            "points_allowed": [30, 24, 18, 12],
            "def_sacks": [1, 2, 3, 4],
            "def_interceptions": [0, 1, 1, 2],
            "def_pass_defended": [3, 4, 5, 6],
            "def_tackles_for_loss": [4, 5, 6, 7],
            "def_qb_hits": [5, 6, 7, 8],
            "def_fumbles_forced": [0, 1, 1, 2],
            "def_safeties": [0, 0, 0, 1],
            "opp_points_allowed": [24, 22, 20, 18],
            "opp_points_for": [18, 21, 24, 27],
            "opp_passing_epa": [0.0, 0.05, 0.1, 0.15],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.columns == ["team", "SaOR", "SaDR", "SaCR"]
    assert result.select("team").to_series().to_list() == ["A", "B", "C", "D"]
    assert result.filter(pl.col("team") == "D").select("SaCR").item() > 0


def test_compute_ratings_without_win_pct_and_without_sos_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "unrelated_metric": [1.0, 2.0, 3.0],
        }
    )

    result = ratings.compute_ratings(df)

    assert result.height == 3
    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0, 0.0]
    assert "falling back to equal weights" in capsys.readouterr().out


def test_compute_ratings_uses_neutral_blend_when_correlations_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "B"],
            "win_pct": [0.4, 0.6],
            "points_for": [10, 20],
            "points_allowed": [20, 10],
        }
    )

    monkeypatch.setattr(
        ratings,
        "_derive_weights",
        lambda *args, **kwargs: [("points_for", 1.0, True)],
    )
    monkeypatch.setattr(ratings, "_build_composite", lambda *args, **kwargs: np.array([0.0, 0.0]))
    monkeypatch.setattr(ratings, "_zscore", lambda values: np.array(values, dtype=np.float64))
    monkeypatch.setattr(
        ratings.np,
        "corrcoef",
        lambda *args, **kwargs: np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
    )

    result = ratings.compute_ratings(df)

    assert result.select("SaCR").to_series().to_list() == [0.0, 0.0]
