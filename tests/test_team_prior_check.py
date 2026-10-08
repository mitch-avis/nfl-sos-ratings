"""Tests for the pre-registered walk-forward test of the preseason prior."""

import hashlib
import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from nfl_sos_ratings.ridge import UnitPrior
from nfl_sos_ratings.team_prior import (
    MIN_CARRYOVER_PAIRS,
    CarryoverSlopes,
    PriorHistory,
    SeasonPrior,
)
from nfl_sos_ratings.team_rating import fit_team_ratings, fit_team_ratings_with_previous_penalties
from nfl_sos_ratings.validation import team_prior_check
from nfl_sos_ratings.validation.team_prior_check import (
    WEEK_BANDS,
    PriorDecision,
    SnapshotAudit,
    baseline_name,
    build_prior_feature_rows,
    check_coverage,
    check_information_set,
    check_matching_rows,
    check_penalties,
    check_reproduction,
    compare_horizons,
    compare_horizons_by_game,
    decide,
    full_fade_weeks,
    input_fingerprint,
    main,
    report_decision,
    score_bands,
    season_bootstrap,
)
from nfl_sos_ratings.validation.walk_forward import build_team_rating_feature_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from pathlib import Path

_TEAMS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
_PLAYS = 60
_FIRST_SEASON = 1999


def _round_robin_weeks(teams: tuple[str, ...]) -> list[list[tuple[str, str]]]:
    """Return a double round robin by week (circle method): each team plays once a week."""
    rotation = list(teams)
    weeks: list[list[tuple[str, str]]] = []
    for _ in range(len(teams) - 1):
        half = len(rotation) // 2
        weeks.append(list(zip(rotation[:half], reversed(rotation[half:]), strict=True)))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return weeks + [[(away, home) for home, away in week] for week in weeks]


def _season_logs(
    offense: Mapping[str, float], defense: Mapping[str, float], *, seed: int, noise: float = 0.2
) -> pl.DataFrame:
    """Return a season of team-game rows with the columns the walk-forward harness reads."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for week_index, week in enumerate(_round_robin_weeks(_TEAMS)):
        for home, away in week:
            game_id = f"w{week_index + 1:02d}_{away}_{home}"
            rates = {
                home: 0.03 + offense[home] - defense[away] + noise * rng.standard_normal(),
                away: 0.01 + offense[away] - defense[home] + noise * rng.standard_normal(),
            }
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                margin = float(rates[team] - rates[opponent])
                rows.append(
                    {
                        "game_id": game_id,
                        "week": week_index + 1,
                        "team": team,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "offensive_snaps": _PLAYS,
                        "offensive_epa": float(rates[team]) * _PLAYS,
                        "st_plays": 12,
                        "st_epa": 0.0,
                        "point_margin": round(margin * _PLAYS),
                        "epa_margin_per_play": margin,
                    }
                )
    return pl.DataFrame(rows)


def _league(last_season: int, seed: int = 7) -> dict[int, pl.DataFrame]:
    """Return seasons from 1999 whose effects carry over from one season to the next."""
    rng = np.random.default_rng(seed)
    offense = rng.normal(0.0, 0.06, len(_TEAMS))
    defense = rng.normal(0.0, 0.06, len(_TEAMS))
    league: dict[int, pl.DataFrame] = {}
    for season in range(_FIRST_SEASON, last_season + 1):
        league[season] = _season_logs(
            dict(zip(_TEAMS, (offense - offense.mean()).tolist(), strict=True)),
            dict(zip(_TEAMS, (defense - defense.mean()).tolist(), strict=True)),
            seed=seed + season,
        )
        offense = 0.7 * offense + 0.7 * rng.normal(0.0, 0.06, len(_TEAMS))
        defense = 0.7 * defense + 0.7 * rng.normal(0.0, 0.06, len(_TEAMS))
    return league


def _write_league(data_dir: Path, league: dict[int, pl.DataFrame]) -> None:
    """Write each season's game logs and its published ratings, as the pipeline does."""
    previous = None
    for season in sorted(league):
        league[season].write_parquet(data_dir / f"{season}_team_game_logs.parquet")
        fit = fit_team_ratings_with_previous_penalties(league[season], previous)
        fit.ratings.write_parquet(data_dir / f"{season}_ratings.parquet")
        previous = fit_team_ratings(league[season])


def _reader(data_dir: Path) -> Callable[[int], pl.DataFrame]:
    """Return a loader of one season's game logs from ``data_dir``."""

    def load(season: int) -> pl.DataFrame:
        """Read one season's team game logs."""
        return pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")

    return load


def _zero_prior(season: int) -> SeasonPrior:
    """Return a prior whose every mean is zero."""
    zeros = dict.fromkeys(_TEAMS, 0.0)
    return SeasonPrior(season, zeros, zeros, CarryoverSlopes(1.0, 1.0, MIN_CARRYOVER_PAIRS))


def _rating_gaps(rows: pl.DataFrame) -> list[float]:
    """Return the rows' rating gaps in game order."""
    return rows.sort("season", "week", "game_id").get_column("rating_diff").to_list()


@pytest.mark.parametrize(
    ("horizon", "name"), [(3.0, "Prior3"), (17.0, "Prior17"), (math.inf, "PriorNoFade")]
)
def test_baseline_name_labels_each_horizon(horizon: float, name: str) -> None:
    # Act
    label = baseline_name(horizon)

    # Assert
    assert label == name


def test_prior_rows_without_a_prior_equal_the_published_rating_rows() -> None:
    # Arrange
    league = _league(2001)
    previous = fit_team_ratings(league[2000])
    expected = build_team_rating_feature_rows(league[2001], 2001, previous)

    # Act
    rows = build_prior_feature_rows(league[2001], 2001, 6.0, previous, None)

    # Assert
    assert _rating_gaps(rows) == pytest.approx(_rating_gaps(expected), abs=1e-12)


def test_prior_rows_with_a_zero_prior_equal_the_published_rating_rows() -> None:
    # Arrange
    league = _league(2001)
    previous = fit_team_ratings(league[2000])
    expected = build_team_rating_feature_rows(league[2001], 2001, previous)

    # Act
    rows = build_prior_feature_rows(league[2001], 2001, 6.0, previous, _zero_prior(2001))

    # Assert
    assert _rating_gaps(rows) == pytest.approx(_rating_gaps(expected), abs=1e-12)


def test_prior_rows_rate_week_one_games_at_zero() -> None:
    # Arrange
    league = _league(2003)
    history = PriorHistory(league.__getitem__)

    # Act
    rows = build_prior_feature_rows(
        league[2003], 2003, 6.0, history.penalties(2003), history.season_prior(2003)
    )

    # Assert
    assert set(rows.filter(pl.col("week") == 1).get_column("rating_diff").to_list()) == {0.0}


def test_prior_rows_rate_a_team_without_games_at_its_prior() -> None:
    # Arrange
    league = _league(2003)
    history = PriorHistory(league.__getitem__)
    skipped = league[2003].filter(~((pl.col("week") == 1) & pl.col("game_id").str.contains("AAA")))
    today = build_prior_feature_rows(skipped, 2003, 6.0, history.penalties(2003), None)

    # Act
    rows = build_prior_feature_rows(
        skipped, 2003, 6.0, history.penalties(2003), history.season_prior(2003)
    )

    # Assert
    week_two = rows.filter((pl.col("week") == 2) & pl.col("game_id").str.contains("AAA"))
    today_two = today.filter((pl.col("week") == 2) & pl.col("game_id").str.contains("AAA"))
    assert week_two.get_column("rating_diff").item() != today_two.get_column("rating_diff").item()


def test_snapshot_audit_finds_no_gap_on_a_correct_prior() -> None:
    # Arrange
    league = _league(2004)
    history = PriorHistory(league.__getitem__)
    audit = SnapshotAudit(residual_seasons=(2004,))

    # Act
    build_prior_feature_rows(
        league[2004], 2004, 6.0, history.penalties(2004), history.season_prior(2004), audit=audit
    )

    # Assert
    assert audit.snapshots > 0
    assert audit.residual_snapshots > 0
    assert audit.limit_snapshots > 0
    assert audit.means_gap < 1e-12
    assert audit.residual_gap < 1e-9
    assert audit.limit_gap < 1e-6


def test_full_fade_weeks_are_those_where_every_team_has_played_the_horizon() -> None:
    # Arrange
    logs = _league(1999)[1999]

    # Act
    weeks = full_fade_weeks(logs, 3.0)

    # Assert
    assert weeks == list(range(4, 11))


def test_check_matching_rows_returns_the_largest_gap() -> None:
    # Arrange
    rows = pl.DataFrame({"game_id": ["a", "b"], "rating_diff": [1.0, 2.0]})
    nearby = rows.with_columns(pl.col("rating_diff") + pl.Series([0.0, 1e-12]))

    # Act
    gap = check_matching_rows(rows, nearby, keys=["game_id"], column="rating_diff", label="test")

    # Assert
    assert gap == pytest.approx(1e-12, abs=1e-15)


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (pl.DataFrame({"game_id": ["a", "b"], "rating_diff": [1.0, 2.1]}), "differ by up to"),
        (pl.DataFrame({"game_id": ["a"], "rating_diff": [1.0]}), "same games"),
        (pl.DataFrame({"game_id": ["a", "b"], "rating_diff": [1.0, None]}), "same games"),
    ],
    ids=["different", "missing-game", "missing-value"],
)
def test_check_matching_rows_rejects_a_gap_or_a_missing_game(
    actual: pl.DataFrame, message: str
) -> None:
    # Arrange
    expected = pl.DataFrame({"game_id": ["a", "b"], "rating_diff": [1.0, 2.0]})

    # Act & Assert
    with pytest.raises(ValueError, match=message):
        check_matching_rows(expected, actual, keys=["game_id"], column="rating_diff", label="test")


def _differences(per_season: Mapping[int, list[float]]) -> pl.DataFrame:
    """Return paired differences in week 3 for each season's games."""
    return pl.DataFrame(
        {
            "season": [season for season, values in per_season.items() for _ in values],
            "week": [3 for values in per_season.values() for _ in values],
            "difference": [value for values in per_season.values() for value in values],
        }
    )


def test_season_bootstrap_resamples_whole_seasons() -> None:
    """With equal games per season and constant differences within each, resampling games by
    season equals resampling the season means with the same draws."""
    # Arrange
    per_season = {2003: [0.5, 0.5], 2004: [-1.0, -1.0], 2005: [2.0, 2.0]}
    draws = np.random.default_rng(0).integers(0, 3, size=(500, 3))
    means = np.array([0.5, -1.0, 2.0])[draws].mean(axis=1)

    # Act
    mean, lower, upper = season_bootstrap(_differences(per_season), [2003, 2004, 2005], draws, 0.9)

    # Assert
    assert mean == pytest.approx(0.5)
    assert (lower, upper) == pytest.approx(tuple(np.quantile(means, [0.05, 0.95])))


def _predictions(
    errors: Mapping[str, list[float]], seasons: list[int], weeks: list[int]
) -> pl.DataFrame:
    """Return prediction rows with the given absolute errors per baseline, one game per entry."""
    return pl.DataFrame(
        [
            {
                "baseline": baseline,
                "season": seasons[index],
                "week": weeks[index],
                "game_id": f"g{index}",
                "error": error,
                "fitted_k": 1.0,
            }
            for baseline, values in errors.items()
            for index, error in enumerate(values)
        ]
    )


def test_compare_horizons_pairs_each_candidate_with_today_by_band() -> None:
    # Arrange
    seasons = [2003, 2003, 2004, 2004]
    weeks = [2, 9, 2, 9]
    predictions = _predictions(
        {"TeamRating": [3.0, 3.0, 3.0, 3.0], "Prior3": [2.0, 3.0, 2.0, 3.0]}, seasons, weeks
    )

    # Act
    comparisons = compare_horizons(predictions, (3.0,), resamples=200, seed=0, confidence=0.9)

    # Assert
    by_band = {row["band"]: row for row in comparisons.iter_rows(named=True)}
    assert by_band["overall"]["mae_delta"] == pytest.approx(-0.5)
    assert by_band[WEEK_BANDS[0][0]]["mae_delta"] == pytest.approx(-1.0)
    assert by_band[WEEK_BANDS[2][0]]["mae_delta"] == pytest.approx(0.0)
    assert by_band["overall"]["games"] == 4


def test_compare_horizons_rejects_a_candidate_missing_a_game() -> None:
    # Arrange
    predictions = _predictions({"TeamRating": [3.0, 3.0], "Prior3": [2.0]}, [2003, 2003], [2, 3])

    # Act & Assert
    with pytest.raises(ValueError, match="same games"):
        compare_horizons(predictions, (3.0,), resamples=10, seed=0, confidence=0.9)


def _comparison_rows(rows: list[tuple[float, str, float, float]]) -> pl.DataFrame:
    """Return comparison rows: horizon, band, and the interval's ends around a midpoint."""
    return pl.DataFrame(
        [
            {
                "horizon": horizon,
                "band": band,
                "games": 10,
                "mae_delta": (lower + upper) / 2,
                "ci_lower": lower,
                "ci_upper": upper,
            }
            for horizon, band, lower, upper in rows
        ]
    )


def _all_bands(
    horizon: float, overall: tuple[float, float], bands: tuple[float, float]
) -> list[tuple[float, str, float, float]]:
    """Return one horizon's overall row and the same interval for every week band."""
    return [(horizon, "overall", *overall)] + [(horizon, band, *bands) for band, _, _ in WEEK_BANDS]


def _scores(mae: Mapping[str, float]) -> pl.DataFrame:
    """Return overall MAE rows per baseline."""
    return pl.DataFrame(
        {"baseline": list(mae), "band": ["overall"] * len(mae), "mae": list(mae.values())}
    )


def test_decide_without_a_qualifying_horizon_recommends_no_prior() -> None:
    # Arrange
    comparisons = _comparison_rows(
        _all_bands(3.0, (-0.1, 0.05), (-0.1, 0.1)) + _all_bands(6.0, (-0.2, 0.01), (-0.1, 0.1))
    )

    # Act
    decision = decide(comparisons, _scores({"TeamRating": 10.0, "Prior3": 9.99, "Prior6": 9.98}))

    # Assert
    assert decision == PriorDecision(None, (), (), ())


def test_decide_recommends_the_qualifying_horizon_with_the_lowest_mae() -> None:
    # Arrange
    comparisons = _comparison_rows(
        _all_bands(3.0, (-0.1, -0.01), (-0.1, 0.1)) + _all_bands(6.0, (-0.2, -0.02), (-0.2, 0.1))
    )

    # Act
    decision = decide(comparisons, _scores({"TeamRating": 10.0, "Prior3": 9.95, "Prior6": 9.9}))

    # Assert
    assert decision.recommended_horizon == 6.0
    assert decision.qualifying == (3.0, 6.0)
    assert decision.excluding_zero == ((3.0, "overall", "better"), (6.0, "overall", "better"))


def test_decide_disqualifies_a_horizon_that_is_worse_in_a_week_band() -> None:
    # Arrange
    rows = _all_bands(9.0, (-0.1, -0.01), (-0.1, 0.1))
    rows[2] = (9.0, WEEK_BANDS[1][0], 0.01, 0.2)

    # Act
    decision = decide(_comparison_rows(rows), _scores({"TeamRating": 10.0, "Prior9": 9.9}))

    # Assert
    assert decision.recommended_horizon is None
    assert decision.band_guard == (9.0,)
    assert (9.0, WEEK_BANDS[1][0], "worse") in decision.excluding_zero


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (
            PriorDecision(6.0, (3.0, 6.0), ((6.0, "overall", "better"),), ()),
            "recommendation is a 6-game",
        ),
        (PriorDecision(None, (), (), (9.0,)), "worse in a week band: 9"),
        (PriorDecision(None, (), (), ()), "recommendation is no prior"),
    ],
)
def test_report_decision_states_the_outcome(
    decision: PriorDecision, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    report_decision(decision)

    # Assert
    assert expected in capsys.readouterr().out


def test_input_fingerprint_matches_sha256sum_of_the_files_hashed_again(tmp_path: Path) -> None:
    # Arrange
    paths = [tmp_path / "b.parquet", tmp_path / "a.parquet"]
    for path, body in zip(paths, (b"second", b"first"), strict=True):
        path.write_bytes(body)
    listing = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in sorted(paths)
    )

    # Act
    fingerprint = input_fingerprint(paths)

    # Assert
    assert fingerprint == hashlib.sha256(listing.encode()).hexdigest()


def test_check_reproduction_matches_the_published_ratings(tmp_path: Path) -> None:
    # Arrange
    league = _league(2004)
    _write_league(tmp_path, league)
    history = PriorHistory(_reader(tmp_path))

    # Act
    gap = check_reproduction(history, tmp_path, [2003, 2004])

    # Assert
    assert gap < 1e-9


def test_check_reproduction_rejects_ratings_that_differ(tmp_path: Path) -> None:
    # Arrange
    league = _league(2004)
    _write_league(tmp_path, league)
    path = tmp_path / "2002_ratings.parquet"
    pl.read_parquet(path).with_columns(pl.col("defense_rating") + 0.5).write_parquet(path)
    history = PriorHistory(_reader(tmp_path))

    # Act & Assert
    with pytest.raises(ValueError, match="2002"):
        check_reproduction(history, tmp_path, [2003])


def test_check_coverage_rejects_a_team_without_a_previous_season(tmp_path: Path) -> None:
    # Arrange
    league = _league(2003)
    league[2003] = league[2003].with_columns(
        pl.col("team").replace("AAA", "NEW"), pl.col("opponent_team").replace("AAA", "NEW")
    )
    _write_league(tmp_path, league)

    # Act & Assert
    with pytest.raises(ValueError, match="NEW"):
        check_coverage(PriorHistory(_reader(tmp_path)), [2003])


def test_check_information_set_accepts_a_history_that_reads_only_earlier_seasons(
    tmp_path: Path,
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2004))

    # Act
    checked = check_information_set(PriorHistory(_reader(tmp_path)), tmp_path, [2003, 2004])

    # Assert
    assert checked == 2


def test_check_information_set_rejects_a_prior_that_differs_without_later_seasons(
    tmp_path: Path,
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2004))
    shifted = PriorHistory(
        lambda season: pl.read_parquet(tmp_path / f"{season}_team_game_logs.parquet").with_columns(
            pl.col("offensive_epa") * 1.5
        )
    )

    # Act & Assert
    with pytest.raises(ValueError, match="2003"):
        check_information_set(shifted, tmp_path, [2003])


def test_check_penalties_accepts_the_published_chain(tmp_path: Path) -> None:
    # Arrange
    _write_league(tmp_path, _league(2002))

    # Act
    checked = check_penalties(PriorHistory(_reader(tmp_path)), tmp_path, [1999, 2000, 2001, 2002])

    # Assert
    assert checked == 4


def test_check_penalties_rejects_a_different_penalty(tmp_path: Path) -> None:
    # Arrange
    league = _league(2002)
    _write_league(tmp_path, league)
    effects = dict.fromkeys(_TEAMS, 0.0) | {"AAA": 0.1, "BBB": -0.1}
    other = {**league, 2001: _season_logs(effects, effects, seed=1, noise=0.0)}

    # Act & Assert
    with pytest.raises(ValueError, match="2002"):
        check_penalties(PriorHistory(other.__getitem__), tmp_path, [2002])


def test_main_prints_the_integrity_checks_decision_and_extras(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2005))

    # Act
    main(
        [
            "--data-dir",
            str(tmp_path),
            "--start-season",
            "2003",
            "--end-season",
            "2004",
            "--spotlight-season",
            "2005",
            "--spotlight-team",
            "AAA",
        ]
    )

    # Assert
    output = capsys.readouterr().out
    assert "Input fingerprint" in output
    assert "Integrity checks passed" in output
    assert "Decision:" in output
    assert "Single-game bootstrap" in output
    assert "Prior17" in output
    assert "PriorNoFade" in output
    assert "Carryover slopes" in output
    assert "AAA in 2005 after week 4" in output
    assert "Week 1" in output


def test_main_says_when_the_spotlight_season_has_no_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2004))

    # Act
    main(["--data-dir", str(tmp_path), "--start-season", "2003", "--end-season", "2004"])

    # Assert
    assert "no 2026 game logs" in capsys.readouterr().out


def test_game_bootstrap_and_scores_skip_a_week_band_without_games() -> None:
    # Arrange
    predictions = _predictions(
        {"TeamRating": [3.0, 3.0], "Prior3": [2.0, 3.0]}, [2003, 2004], [2, 9]
    )

    # Act
    by_game = compare_horizons_by_game(predictions, (3.0,), resamples=50, seed=0, confidence=0.9)
    scores = score_bands(predictions)

    # Assert
    assert WEEK_BANDS[1][0] not in by_game.get_column("band").to_list()
    assert WEEK_BANDS[1][0] not in scores.get_column("band").to_list()
    assert by_game.filter(pl.col("band") == "overall").get_column("mae_delta").item() == -0.5


def test_check_coverage_skips_a_season_without_a_prior(tmp_path: Path) -> None:
    # Arrange
    _write_league(tmp_path, _league(2002))

    # Act
    result = check_coverage(PriorHistory(_reader(tmp_path)), [2002])

    # Assert
    assert result is None


def test_main_reports_seasons_without_a_prior_and_an_unrated_team(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2004))

    # Act
    main(
        [
            "--data-dir",
            str(tmp_path),
            "--start-season",
            "2002",
            "--end-season",
            "2003",
            "--spotlight-season",
            "2004",
            "--spotlight-team",
            "ZZZ",
        ]
    )

    # Assert
    output = capsys.readouterr().out
    assert "Carryover slopes (offense/defense) by season: 2003" in output
    assert "today unrated" in output
    assert "needs two seasons with a prior" in output


def test_main_stops_when_a_snapshot_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    _write_league(tmp_path, _league(2003))
    original = team_prior_check.prior_means

    def shifted(
        season_prior: SeasonPrior,
        games: Mapping[str, int],
        horizon: float,
        fitted: Collection[str],
    ) -> UnitPrior:
        """Return the prior means with every offense mean moved, as a broken prior would."""
        means = original(season_prior, games, horizon, fitted)
        moved = {team: value + 0.01 for team, value in means.offense.items()}
        return UnitPrior(offense=moved, defense=means.defense)

    monkeypatch.setattr(team_prior_check, "prior_means", shifted)

    # Act & Assert
    with pytest.raises(ValueError, match="integrity check failed"):
        main(["--data-dir", str(tmp_path), "--start-season", "2003", "--end-season", "2003"])
    assert "Decision" not in capsys.readouterr().out
