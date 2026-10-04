"""Tests for rank ranges: bootstrap rating and rank quantiles per team or quarterback."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings.rating_ranges import RANGE_QUANTILES, RangeColumns, summarize_rank_ranges
from nfl_sos_ratings.team_rating import bootstrap_team_ratings, fit_team_ratings


def _draws(ratings: dict[int, dict[str, float]]) -> pl.DataFrame:
    """Return long draw rows from ``{draw: {unit: rating}}``."""
    return pl.DataFrame(
        [
            {"draw": draw, "unit": unit, "rating": rating}
            for draw, by_unit in ratings.items()
            for unit, rating in by_unit.items()
        ]
    )


_PUBLISHED = pl.DataFrame({"unit": ["A", "B", "C"], "rating": [3.0, 2.0, 1.0]})
_FOUR_DRAWS = {
    0: {"A": 3.0, "B": 2.0, "C": 1.0},
    1: {"A": 3.0, "B": 2.0, "C": 1.0},
    2: {"A": 1.0, "B": 3.0, "C": 2.0},
    3: {"A": 3.0, "B": 1.0, "C": 2.0},
}


def _summary(draws: dict[int, dict[str, float]], eligible: set[str] | None = None) -> pl.DataFrame:
    """Summarize ``draws`` against ``_PUBLISHED`` with plain column names."""
    return summarize_rank_ranges(
        _draws(draws),
        _PUBLISHED,
        RangeColumns(id="unit", rating="rating", rank="rank"),
        eligible=eligible,
    )


def _row(summary: pl.DataFrame, unit: str) -> dict[str, object]:
    """Return one unit's summary row."""
    return summary.filter(pl.col("unit") == unit).row(0, named=True)


def test_summarize_rank_ranges_gives_the_full_rank_distribution() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    assert _row(summary, "A")["rank_probabilities"] == pytest.approx([0.75, 0.0, 0.25])


def test_summarize_rank_ranges_reports_the_published_rank_and_top_probabilities() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    row = _row(summary, "C")
    assert (row["rank"], row["rank_top5_probability"]) == (3, pytest.approx(1.0))


def test_summarize_rank_ranges_takes_rank_quantiles_from_drawn_ranks() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    row = _row(summary, "B")
    assert (row["rank_q025"], row["rank_q500"], row["rank_q975"]) == (1, 2, 3)


def test_summarize_rank_ranges_has_every_quantile_column() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    suffixes = [f"_q{round(level * 1000):03d}" for level in RANGE_QUANTILES]
    assert {f"{base}{suffix}" for base in ("rating", "rank") for suffix in suffixes} <= set(
        summary.columns
    )


def test_summarize_rank_ranges_ranks_only_eligible_units_and_counts_missing_draws() -> None:
    # Arrange
    draws = {0: {"A": 3.0, "B": 2.0, "C": 9.0}, 1: {"A": 3.0, "C": 9.0}}

    # Act
    summary = _summary(draws, eligible={"A", "B"})

    # Assert
    assert summary.get_column("unit").sort().to_list() == ["A", "B"]
    row = _row(summary, "B")
    assert (row["rank_missing_share"], row["rank_probabilities"]) == (
        pytest.approx(0.5),
        pytest.approx([0.0, 0.5]),
    )


# A synthetic league for the calibration check: true effects average to zero.
_OFFENSE = dict(zip("ABCDEFGH", (0.12, 0.07, 0.03, 0.0, -0.01, -0.04, -0.07, -0.10), strict=True))
_DEFENSE = dict(zip("ABCDEFGH", (-0.02, 0.06, 0.09, -0.05, 0.03, -0.08, 0.01, -0.04), strict=True))
_SPECIAL = dict(zip("ABCDEFGH", (0.10, -0.05, 0.0, 0.05, -0.10, 0.0, 0.05, -0.05), strict=True))
_PLAYS, _SPECIAL_PLAYS = 60, 12


def _true_ratings() -> dict[str, float]:
    """Return each team's true points-per-game rating under the generating model."""
    return {
        team: (_OFFENSE[team] + _DEFENSE[team]) * _PLAYS + _SPECIAL[team] * _SPECIAL_PLAYS
        for team in _OFFENSE
    }


def _noisy_league(rng: np.random.Generator) -> pl.DataFrame:
    """Return one season: every pair meets four times, game EPA per play with noise."""
    rows: list[dict[str, object]] = []
    games = itertools.product(range(2), itertools.permutations(_OFFENSE, 2))
    for number, (_, (home, away)) in enumerate(games):
        for team, opponent, sign in ((home, away, 1.0), (away, home, -1.0)):
            epa = 0.02 + _OFFENSE[team] - _DEFENSE[opponent] + 0.01 * sign + rng.normal(0, 0.2)
            special = _SPECIAL[team] / 2 - _SPECIAL[opponent] / 2 + rng.normal(0, 0.3)
            rows.append(
                {
                    "game_id": f"g{number:03d}",
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": sign > 0,
                    "offensive_snaps": _PLAYS,
                    "offensive_epa": epa * _PLAYS,
                    "st_plays": _SPECIAL_PLAYS,
                    "st_epa": special * _SPECIAL_PLAYS,
                }
            )
    return pl.DataFrame(rows)


def test_team_rating_intervals_cover_the_true_ratings_near_the_nominal_rate() -> None:
    # Arrange
    rng = np.random.default_rng(0)
    truth = _true_ratings()
    covered = {0.80: 0, 0.95: 0}
    trials = 0

    # Act
    for replication in range(30):
        game_logs = _noisy_league(rng)
        draws = bootstrap_team_ratings(
            game_logs, fit_team_ratings(game_logs), resamples=200, seed=replication
        )
        for team, ratings in draws.group_by("team"):
            values = ratings.get_column("team_rating").to_numpy()
            for level in covered:
                low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
                covered[level] += int(low <= truth[str(team[0])] <= high)
            trials += 1

    # Assert
    # Tolerance fixed before the first run (.agents/rank-ranges-and-wp-filter-plan.md).
    assert 0.85 <= covered[0.95] / trials <= 1.0
    assert 0.65 <= covered[0.80] / trials <= 0.95
