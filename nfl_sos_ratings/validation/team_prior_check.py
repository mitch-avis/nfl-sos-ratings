"""Walk-forward test of the preseason prior: does shrinking toward last season predict better?

The published team fit shrinks every scrimmage effect toward zero. A candidate shrinks it toward a
prior built from the previous season instead (``nfl_sos_ratings.team_prior``), faded out over the
first ``G`` games a team plays, for ``G`` = 3, 6, and 9. Each candidate runs through the
walk-forward harness from 1999 (so every margin model has the same warm-up) and is scored on
prediction weeks 2 and later of 2003-2025 against today's fit.

The protocol, written in ``.agents/roadmap.md`` before the first run, fixes the decision: a paired
bootstrap that resamples whole seasons (10,000 resamples, seed 0, the same draws for every
candidate and week band) gives 98.33% intervals for each candidate's MAE minus today's. A horizon
qualifies only if its overall interval lies below zero and none of its week-band intervals
(weeks 2-4, 5-8, and 9 on) lies above zero; the qualifying horizon with the lowest MAE is the
recommendation, and with none, no prior is.

Before any result is printed, the integrity checks run: the previous season's published ratings
are reproduced, today's rows equal the validation backtest's, a zero prior and a fully faded prior
equal today's fit, the prior means match an independent recomputation, the solver matches an
independent residual-form fit and the large-penalty limit, every team has a previous-season
effect, the prior never reads its own season or later, and every penalty is today's. Descriptive
extras, never decision inputs: a single-game bootstrap, the margin slope by band, Elo by band, a
17-game and a no-fade prior, the carryover slopes, one team's rating after week 4 of a season in
progress, and week 1 rated by the prior alone.

Run ``nfl-sos-ratings check-team-prior``; it only reads ``data/`` (the information-set check links
earlier seasons' files into a temporary directory it removes).

"Today's fit" is the team rating without a prior, as published when this test was registered; the
published rating now uses the 9-game prior this test recommended (``team_prior``), and the check
keeps its own baseline (``run_walk_forward_backtest(..., team_prior=False)``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_sos_ratings.config import DATA_DIR, END_YEAR
from nfl_sos_ratings.data_loader import PBP_START_SEASON
from nfl_sos_ratings.team_prior import (
    MIN_CARRYOVER_PAIRS,
    PriorHistory,
    SeasonPrior,
    games_played,
    prior_means,
    snapshot_fit,
    snapshot_ratings,
)
from nfl_sos_ratings.team_rating import (
    SCRIMMAGE_EPA_COLUMN,
    SCRIMMAGE_PLAYS_COLUMN,
    TeamRatingFit,
    fit_team_ratings,
    scrimmage_rows,
)
from nfl_sos_ratings.validation.walk_forward import (
    TEAM_RATING_BASELINE,
    build_home_game_frame,
    evaluate_feature_rows,
    previous_season_fit,
    run_walk_forward_backtest,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from nfl_sos_ratings.ridge import UnitPrior

CANDIDATE_HORIZONS: tuple[float, ...] = (3.0, 6.0, 9.0)
# Descriptive only: both leave a prior in completed seasons, so neither can be adopted.
EXTRA_HORIZONS: tuple[float, ...] = (17.0, math.inf)
HORIZONS: tuple[float, ...] = (*CANDIDATE_HORIZONS, *EXTRA_HORIZONS)
PREDICTION_START_WEEK = 2
VALIDATION_START_WEEK = 5
DEFAULT_START_SEASON = PBP_START_SEASON + 1 + MIN_CARRYOVER_PAIRS
WEEK_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("weeks 2-4", 2, 4),
    ("weeks 5-8", 5, 8),
    ("weeks 9+", 9, None),
)
OVERALL = "overall"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FAMILY_CONFIDENCE = 0.95
# Bonferroni over the three candidates: 98.33% intervals.
COMPARISON_CONFIDENCE = 1.0 - (1.0 - FAMILY_CONFIDENCE) / len(CANDIDATE_HORIZONS)
MATCH_TOLERANCE = 1e-9
# The prior's slopes come from fits at a 1e-6 penalty; the independent side solves exactly.
SLOPE_TOLERANCE = 1e-5
MEANS_TOLERANCE = 1e-12
LIMIT_TOLERANCE = 1e-6
LIMIT_LAMBDA = 1e12
DEFAULT_SPOTLIGHT_SEASON = 2026
DEFAULT_SPOTLIGHT_TEAM = "DEN"
# The spotlight shows ratings fit on the games before this week (after week 4).
SPOTLIGHT_WEEK = 5
_ROW_KEYS = ["season", "week", "game_id"]
_RESAMPLE_CHUNK = 500
_FEATURE_COLUMNS = [
    "season",
    "week",
    "baseline",
    "game_id",
    "home_team",
    "away_team",
    "rating_diff",
    "home_margin",
]


def baseline_name(horizon: float) -> str:
    """Return the walk-forward label of the prior faded over ``horizon`` games."""
    return "PriorNoFade" if math.isinf(horizon) else f"Prior{horizon:g}"


@dataclass(slots=True)
class SnapshotAudit:
    """Largest gaps the per-snapshot integrity checks found while building candidate rows.

    On every snapshot with a prior: ``means_gap`` compares the means the fit used with an
    independent recomputation from the season's games before the prediction week;
    ``residual_gap`` compares the fit's ratings with an ordinary fit on the residual response
    built from those independent means; ``unplayed_gap`` compares the ratings of teams without
    games with their independent prior means. Once per snapshot (on ``limit_horizon``'s pass),
    ``limit_gap`` compares each effect at a huge penalty with its unfaded independent mean.
    """

    limit_horizon: float = CANDIDATE_HORIZONS[0]
    snapshots: int = 0
    means_gap: float = 0.0
    residual_gap: float = 0.0
    unplayed_snapshots: int = 0
    unplayed_gap: float = 0.0
    limit_snapshots: int = 0
    limit_gap: float = 0.0


def _independent_means(
    game_logs: pl.DataFrame,
    week: int,
    season_prior: SeasonPrior,
    horizon: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Recompute the centered prior means from the full season, without ``team_prior``'s code."""
    earlier = game_logs.filter(pl.col("week") < week)
    teams = sorted(set(game_logs.get_column("team").to_list()))
    counts = dict.fromkeys(teams, 0)
    for team, _game in earlier.select("team", "game_id").unique().iter_rows():
        counts[team] += 1
    fitted = sorted(
        set(earlier.filter(pl.col(SCRIMMAGE_PLAYS_COLUMN) > 0).get_column("team").to_list())
    )
    centered: list[dict[str, float]] = []
    for slope, previous in (
        (season_prior.slopes.offense, season_prior.offense),
        (season_prior.slopes.defense, season_prior.defense),
    ):
        raw = {
            team: (1.0 if math.isinf(horizon) else max(0.0, 1.0 - counts[team] / horizon))
            * slope
            * previous[team]
            for team in teams
        }
        shift = sum(raw[team] for team in fitted) / len(fitted)
        centered.append({team: value - shift for team, value in raw.items()})
    return centered[0], centered[1]


def _rating(ratings: pl.DataFrame, column: str) -> dict[str, float]:
    """Return one rating column by team."""
    return dict(ratings.select("team", column).iter_rows())


def _gap(left: dict[str, float], right: dict[str, float], teams: Iterable[str]) -> float:
    """Return the largest absolute difference between two team mappings over ``teams``."""
    return max((abs(left[team] - right[team]) for team in teams), default=0.0)


def _audit_snapshot(  # noqa: PLR0913 - one snapshot's inputs, all needed
    audit: SnapshotAudit,
    *,
    game_logs: pl.DataFrame,
    week: int,
    penalties: TeamRatingFit,
    season_prior: SeasonPrior,
    horizon: float,
    ratings: pl.DataFrame,
    means: UnitPrior,
) -> None:
    """Run the per-snapshot integrity checks on one candidate snapshot and record the gaps."""
    prior_games = game_logs.filter(pl.col("week") < week)
    teams = sorted(set(game_logs.get_column("team").to_list()))
    rows = scrimmage_rows(prior_games)
    fitted = sorted(set(rows.get_column("team").to_list()))
    unplayed = [team for team in teams if team not in fitted]
    scale = float(rows.get_column("plays").sum()) / rows.height
    offense, defense = _independent_means(game_logs, week, season_prior, horizon)
    audit.snapshots += 1
    audit.means_gap = max(
        audit.means_gap,
        _gap(dict(means.offense), offense, teams),
        _gap(dict(means.defense), defense, teams),
    )
    shift = pl.col("team").replace_strict(offense) - pl.col("opponent_team").replace_strict(defense)
    residual = fit_team_ratings(
        prior_games.with_columns(
            (pl.col(SCRIMMAGE_EPA_COLUMN) - shift * pl.col(SCRIMMAGE_PLAYS_COLUMN)).alias(
                SCRIMMAGE_EPA_COLUMN
            )
        ),
        scrimmage_lambda=penalties.scrimmage_lambda,
        special_teams_lambda=penalties.special_teams_lambda,
    ).ratings
    for column, side in (("offense_rating", offense), ("defense_rating", defense)):
        solved, plain = _rating(ratings, column), _rating(residual, column)
        expected = {team: plain[team] + side[team] * scale for team in fitted}
        audit.residual_gap = max(audit.residual_gap, _gap(solved, expected, fitted))
    if unplayed:
        audit.unplayed_snapshots += 1
        expected_rows = {
            "offense_rating": {team: offense[team] * scale for team in unplayed},
            "defense_rating": {team: defense[team] * scale for team in unplayed},
            "special_teams_rating": dict.fromkeys(unplayed, 0.0),
            "team_rating": {team: (offense[team] + defense[team]) * scale for team in unplayed},
        }
        for column, expected in expected_rows.items():
            audit.unplayed_gap = max(
                audit.unplayed_gap, _gap(_rating(ratings, column), expected, unplayed)
            )
    if horizon != audit.limit_horizon:
        return
    unfaded = _independent_means(game_logs, week, season_prior, math.inf)
    limit = snapshot_ratings(
        prior_games,
        teams,
        dataclasses.replace(penalties, scrimmage_lambda=LIMIT_LAMBDA),
        season_prior,
        math.inf,
    )
    for column, side in zip(("offense_rating", "defense_rating"), unfaded, strict=True):
        pinned = {team: value / scale for team, value in _rating(limit, column).items()}
        audit.limit_gap = max(audit.limit_gap, _gap(pinned, side, fitted))
    audit.limit_snapshots += 1


def build_prior_feature_rows(  # noqa: PLR0913 - the season, its fit inputs, and the audit
    game_logs: pl.DataFrame,
    season: int,
    horizon: float,
    penalties: TeamRatingFit | None,
    season_prior: SeasonPrior | None,
    *,
    audit: SnapshotAudit | None = None,
) -> pl.DataFrame:
    """Build one candidate's walk-forward rows: each week rated on the games before it.

    Mirrors ``walk_forward.build_snapshot_feature_rows``: week-1 games get a gap of zero, and a
    team the snapshot does not rate counts as zero. With a prior, every team has a rating, those
    without games at their prior means.
    """
    home_games = build_home_game_frame(game_logs, season)
    teams = sorted(set(game_logs.get_column("team").to_list()))
    name = baseline_name(horizon)
    frames: list[pl.DataFrame] = []
    for week in sorted(set(home_games.get_column("week").to_list())):
        prior_games = game_logs.filter(pl.col("week") < week)
        if prior_games.is_empty():
            ratings = pl.DataFrame({"team": teams, "team_rating": [0.0] * len(teams)})
        else:
            ratings, means = snapshot_fit(prior_games, teams, penalties, season_prior, horizon)
            if audit is not None and season_prior is not None and penalties is not None:
                if means is None:
                    msg = "a fit with a prior reported no prior means"
                    raise ValueError(msg)
                _audit_snapshot(
                    audit,
                    game_logs=game_logs,
                    week=week,
                    penalties=penalties,
                    season_prior=season_prior,
                    horizon=horizon,
                    ratings=ratings,
                    means=means,
                )
        lookup = ratings.select("team", pl.col("team_rating").alias("rating"))
        frames.append(
            home_games.filter(pl.col("week") == week)
            .join(
                lookup.rename({"team": "home_team", "rating": "home"}), on="home_team", how="left"
            )
            .join(
                lookup.rename({"team": "away_team", "rating": "away"}), on="away_team", how="left"
            )
            .with_columns(
                pl.lit(name).alias("baseline"),
                (pl.col("home").fill_null(0.0) - pl.col("away").fill_null(0.0)).alias(
                    "rating_diff"
                ),
            )
            .select(_FEATURE_COLUMNS)
        )
    return pl.concat(frames)


def full_fade_weeks(game_logs: pl.DataFrame, horizon: float) -> list[int]:
    """Return the prediction weeks before which every team has played ``horizon`` games."""
    teams = sorted(set(game_logs.get_column("team").to_list()))
    weeks = sorted(set(game_logs.filter(pl.col("is_home")).get_column("week").to_list()))
    faded: list[int] = []
    for week in weeks:
        counts = games_played(game_logs.filter(pl.col("week") < week), teams)
        if min(counts.values()) >= horizon:
            faded.append(week)
    return faded


def check_matching_rows(
    expected: pl.DataFrame,
    actual: pl.DataFrame,
    *,
    keys: list[str],
    column: str,
    label: str,
) -> float:
    """Return the largest gap in ``column`` between two row sets keyed by ``keys``.

    Raises:
        ValueError: If the two do not cover the same games with values, or a gap exceeds
            ``MATCH_TOLERANCE``.

    """
    joined = expected.select(*keys, column).join(
        actual.select(*keys, pl.col(column).alias("_actual")), on=keys, how="inner"
    )
    gaps = joined.select(
        (pl.col(column).cast(pl.Float64) - pl.col("_actual").cast(pl.Float64)).abs().alias("gap")
    ).get_column("gap")
    # A missing value, or NaN on either side, is a mismatch: NaN never exceeds a tolerance.
    if (
        not joined.height == expected.height == actual.height
        or gaps.is_null().any()
        or (gaps.is_nan().any())
    ):
        msg = f"{label}: the two do not cover the same games with values"
        raise ValueError(msg)
    largest = float(gaps.to_numpy().max()) if gaps.len() else 0.0
    if largest > MATCH_TOLERANCE:
        msg = f"{label}: the rows differ by up to {largest:.3g}"
        raise ValueError(msg)
    return largest


def _read_logs(data_dir: Path) -> Callable[[int], pl.DataFrame]:
    """Return a loader of one season's team game logs from ``data_dir``."""

    def load(season: int) -> pl.DataFrame:
        """Read one season's team game logs."""
        return pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet")

    return load


def check_reproduction(history: PriorHistory, data_dir: Path, seasons: Iterable[int]) -> float:
    """Return the largest gap between each previous season's rebuilt ratings and its published ones.

    For each season ``s`` with a prior, the prior the candidates use (``history.season_prior(s)``)
    must rebuild all four rating columns of ``{s - 1}_ratings.parquet``: its offense and defense
    effects times season ``s - 1``'s scrimmage plays per game, special teams from season
    ``s - 1``'s published fit, and the team rating as their sum. A season without a prior checks
    the refit of season ``s - 1`` alone.

    Raises:
        ValueError: If the prior lacks a team, or a gap exceeds ``MATCH_TOLERANCE``, naming the
            seasons.

    """
    largest = 0.0
    columns = ["offense_rating", "defense_rating", "special_teams_rating", "team_rating"]
    for season in seasons:
        previous = season - 1
        penalties = history.penalties(previous)
        fit = fit_team_ratings(
            history.game_logs(previous),
            scrimmage_lambda=penalties.scrimmage_lambda,
            special_teams_lambda=penalties.special_teams_lambda,
        )
        published = pl.read_parquet(data_dir / f"{previous}_ratings.parquet").select(
            "team", *columns
        )
        rebuilt = fit.ratings
        label = f"{previous} published ratings"
        season_prior = history.season_prior(season)
        if season_prior is not None:
            special = dict(fit.ratings.select("team", "special_teams_rating").iter_rows())
            teams = sorted(special)
            missing = [
                team
                for team in teams
                if team not in season_prior.offense or team not in season_prior.defense
            ]
            if missing:
                msg = f"the {season} prior has no effect for {', '.join(missing)}"
                raise ValueError(msg)
            scale = fit.scrimmage_plays_per_game
            offense = [season_prior.offense[team] * scale for team in teams]
            defense = [season_prior.defense[team] * scale for team in teams]
            rebuilt = pl.DataFrame(
                {
                    "team": teams,
                    "offense_rating": offense,
                    "defense_rating": defense,
                    "special_teams_rating": [special[team] for team in teams],
                    "team_rating": [
                        o + d + special[team]
                        for o, d, team in zip(offense, defense, teams, strict=True)
                    ],
                }
            )
            label = f"{previous} published ratings from the {season} prior"
        for column in columns:
            largest = max(
                largest,
                check_matching_rows(
                    published, rebuilt, keys=["team"], column=column, label=f"{label} ({column})"
                ),
            )
    return largest


def _home_signs(rows: pl.DataFrame) -> np.ndarray:
    """Return +1 for home rows, -1 for away rows, and 0 where the site is unknown or neutral."""
    flags = rows.get_column("is_home").to_list()
    return np.array([0.0 if flag is None else (1.0 if flag else -1.0) for flag in flags])


def exact_scrimmage_effects(game_logs: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Return a season's scrimmage effects per play by exact weighted least squares, centered.

    Written with NumPy alone, apart from ``ridge`` and ``team_prior``, as the independent side of
    the carryover-slope check: the same model (intercept, home field, offense minus defense,
    weighted by plays), solved without a penalty, with each side's effects shifted to average zero.
    """
    rows = game_logs.filter(pl.col(SCRIMMAGE_PLAYS_COLUMN) > 0)
    teams = sorted(
        set(rows.get_column("team").to_list()) | set(rows.get_column("opponent_team").to_list())
    )
    index = {team: position for position, team in enumerate(teams)}
    count = len(teams)
    design = np.zeros((rows.height, 2 + 2 * count))
    design[:, 0] = 1.0
    design[:, 1] = _home_signs(rows)
    for row, (team, opponent) in enumerate(rows.select("team", "opponent_team").iter_rows()):
        design[row, 2 + index[team]] = 1.0
        design[row, 2 + count + index[opponent]] = -1.0
    plays = rows.get_column(SCRIMMAGE_PLAYS_COLUMN).cast(pl.Float64).to_numpy()
    response = rows.get_column(SCRIMMAGE_EPA_COLUMN).cast(pl.Float64).to_numpy() / plays
    root = np.sqrt(plays)
    solution, *_ = np.linalg.lstsq(design * root[:, np.newaxis], response * root, rcond=None)
    offense = solution[2 : 2 + count] - solution[2 : 2 + count].mean()
    defense = solution[2 + count :] - solution[2 + count :].mean()
    return (
        dict(zip(teams, offense.tolist(), strict=True)),
        dict(zip(teams, defense.tolist(), strict=True)),
    )


def _published_scrimmage_effects(
    data_dir: Path, season: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Return a season's published offense and defense ratings as per-play effects."""
    rows = pl.read_parquet(data_dir / f"{season}_team_game_logs.parquet").filter(
        pl.col(SCRIMMAGE_PLAYS_COLUMN) > 0
    )
    scale = float(rows.get_column(SCRIMMAGE_PLAYS_COLUMN).sum()) / rows.height
    ratings = pl.read_parquet(data_dir / f"{season}_ratings.parquet")
    return (
        {
            team: value / scale
            for team, value in ratings.select("team", "offense_rating").iter_rows()
        },
        {
            team: value / scale
            for team, value in ratings.select("team", "defense_rating").iter_rows()
        },
    )


def independent_slopes(data_dir: Path, season: int, pairs: int) -> tuple[float, float]:
    """Return the carryover slopes for ``season`` from the published files and exact fits.

    Pools the ``pairs`` season pairs ``(t - 1, t)`` before ``season``: season ``t``'s exact
    effects on season ``t - 1``'s published effects, through the origin, team by team.
    """
    sums = {"offense": [0.0, 0.0], "defense": [0.0, 0.0]}
    for later in range(season - pairs, season):
        before = _published_scrimmage_effects(data_dir, later - 1)
        after = exact_scrimmage_effects(
            pl.read_parquet(data_dir / f"{later}_team_game_logs.parquet")
        )
        for side, earlier, current in (
            ("offense", before[0], after[0]),
            ("defense", before[1], after[1]),
        ):
            for team in sorted(set(earlier) & set(current)):
                sums[side][0] += earlier[team] * current[team]
                sums[side][1] += earlier[team] * earlier[team]
    return sums["offense"][0] / sums["offense"][1], sums["defense"][0] / sums["defense"][1]


def check_slopes(history: PriorHistory, data_dir: Path, seasons: Iterable[int]) -> float:
    """Return the largest gap between each prior's carryover slopes and an independent recompute.

    Raises:
        ValueError: If a slope differs by more than ``SLOPE_TOLERANCE``, naming the season.

    """
    largest = 0.0
    for season in seasons:
        season_prior = history.season_prior(season)
        if season_prior is None:
            continue
        slopes = season_prior.slopes
        offense, defense = independent_slopes(data_dir, season, slopes.pairs)
        gap = max(abs(slopes.offense - offense), abs(slopes.defense - defense))
        if gap > SLOPE_TOLERANCE:
            msg = (
                f"{season}: the prior's carryover slopes ({slopes.offense:.6f}, "
                f"{slopes.defense:.6f}) differ from an independent exact fit's ({offense:.6f}, "
                f"{defense:.6f})"
            )
            raise ValueError(msg)
        largest = max(largest, gap)
    return largest


def check_coverage(history: PriorHistory, seasons: Iterable[int]) -> None:
    """Check that every team of each season with a prior has a previous-season effect.

    Raises:
        ValueError: If a team lacks one, naming the season and the teams.

    """
    for season in seasons:
        season_prior = history.season_prior(season)
        if season_prior is None:
            continue
        logs = history.game_logs(season)
        teams = set(logs.get_column("team").to_list()) | set(
            logs.get_column("opponent_team").to_list()
        )
        missing = sorted(
            team
            for team in teams
            if team not in season_prior.offense or team not in season_prior.defense
        )
        if missing:
            msg = f"{season}: no {season - 1} effect for {', '.join(missing)}"
            raise ValueError(msg)


def check_information_set(history: PriorHistory, data_dir: Path, seasons: Iterable[int]) -> int:
    """Check that each season's prior is the same when only earlier seasons exist; return a count.

    For each season ``s``, a fresh history over a directory holding only the game logs of the
    seasons before ``s`` must give the same prior for ``s`` as ``history``.

    Raises:
        ValueError: If the two priors differ, naming the season.

    """
    checked = 0
    for season in seasons:
        with tempfile.TemporaryDirectory(prefix="nfl-sos-prior-") as scratch:
            earlier = Path(scratch)
            for path in sorted(data_dir.glob("*_team_game_logs.parquet")):
                if int(path.name.split("_", 1)[0]) < season:
                    (earlier / path.name).symlink_to(path.resolve())
            restricted = PriorHistory(_read_logs(earlier)).season_prior(season)
        if restricted != history.season_prior(season):
            msg = f"{season}: the prior changes when the seasons from {season} on are absent"
            raise ValueError(msg)
        checked += 1
    return checked


def _snapshot_penalties(history: PriorHistory, season: int) -> TeamRatingFit | None:
    """Return the penalties the season's snapshots use, as the walk-forward harness does."""
    return history.cross_validated_fit(season - 1) if season > PBP_START_SEASON else None


def check_penalties(history: PriorHistory, data_dir: Path, seasons: Iterable[int]) -> int:
    """Check that every season's snapshot penalties equal today's exactly; return a count.

    Raises:
        ValueError: If a season's penalties differ from ``walk_forward.previous_season_fit``'s.

    """
    checked = 0
    for season in seasons:
        expected = previous_season_fit(data_dir, season)
        actual = _snapshot_penalties(history, season)
        pairs = [
            (fit.scrimmage_lambda, fit.special_teams_lambda) if fit is not None else None
            for fit in (expected, actual)
        ]
        if pairs[0] != pairs[1]:
            msg = f"{season}: the prior's penalties {pairs[1]} differ from today's {pairs[0]}"
            raise ValueError(msg)
        checked += 1
    return checked


def input_fingerprint(paths: Iterable[Path]) -> str:
    """Return a SHA-256 fingerprint of files: each file's ``sha256sum`` line, hashed again.

    Equals ``(cd <dir> && sha256sum <files in name order>) | sha256sum`` for files in one directory.
    """
    listing = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(paths, key=lambda path: path.name)
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def decision_input_paths(data_dir: Path, end_season: int) -> list[Path]:
    """Return the files the decision reads: game logs from 1999 and earlier seasons' ratings."""
    logs = [
        data_dir / f"{season}_team_game_logs.parquet"
        for season in range(PBP_START_SEASON, end_season + 1)
    ]
    ratings = [
        data_dir / f"{season}_ratings.parquet" for season in range(PBP_START_SEASON, end_season)
    ]
    return logs + ratings


def _band_rows(rows: pl.DataFrame, band: str) -> pl.DataFrame:
    """Return the rows of one week band, or every row for ``OVERALL``."""
    for name, first, last in WEEK_BANDS:
        if name == band:
            in_band = pl.col("week") >= first
            return rows.filter(in_band if last is None else in_band & (pl.col("week") <= last))
    return rows


def season_bootstrap(
    differences: pl.DataFrame,
    seasons: Sequence[int],
    draws: np.ndarray,
    confidence: float,
) -> tuple[float, float, float]:
    """Return the mean paired difference and its percentile interval from whole-season resamples.

    ``differences`` holds one ``difference`` per game with its ``season``; each row of ``draws``
    picks ``len(seasons)`` seasons (indexes into ``seasons``) with replacement, keeping each drawn
    season's games together, and the resampled statistic is the mean over the drawn games.
    """
    totals = dict(differences.group_by("season").agg(pl.col("difference").sum()).iter_rows())
    counts = dict(differences.group_by("season").len().iter_rows())
    sums = np.array([totals.get(season, 0.0) for season in seasons], dtype=np.float64)
    games = np.array([counts.get(season, 0) for season in seasons], dtype=np.float64)
    resampled = sums[draws].sum(axis=1) / games[draws].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(resampled, [tail, 1.0 - tail])
    return float(sums.sum() / games.sum()), float(lower), float(upper)


def _paired_differences(predictions: pl.DataFrame, horizon: float) -> pl.DataFrame:
    """Return each game's absolute error at ``horizon`` minus today's.

    Raises:
        ValueError: If the candidate and today's fit do not predict the same games.

    """
    paired = predictions.filter(
        pl.col("baseline").is_in([TEAM_RATING_BASELINE, baseline_name(horizon)])
    )
    if paired.select(pl.struct("baseline", *_ROW_KEYS).is_duplicated().any()).item():
        msg = f"{baseline_name(horizon)} or today's fit predicts a game more than once"
        raise ValueError(msg)
    errors = paired.with_columns(pl.col("error").abs()).pivot(
        on="baseline", index=_ROW_KEYS, values="error", aggregate_function="first"
    )
    columns = [TEAM_RATING_BASELINE, baseline_name(horizon)]
    if (
        not set(columns) <= set(errors.columns)
        or errors.select(pl.any_horizontal(pl.col(columns).is_null()).any()).item()
    ):
        msg = f"{baseline_name(horizon)} and today's fit do not predict the same games"
        raise ValueError(msg)
    return errors.select(
        "season", "week", (pl.col(columns[1]) - pl.col(columns[0])).alias("difference")
    )


def compare_horizons(
    predictions: pl.DataFrame,
    horizons: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = COMPARISON_CONFIDENCE,
) -> pl.DataFrame:
    """Return each horizon's paired MAE difference from today's fit, overall and by week band.

    ``mae_delta`` is the candidate's MAE minus today's, so a negative value favors the prior.
    Every horizon and band reuses one matrix of season draws.
    """
    today = predictions.filter(pl.col("baseline") == TEAM_RATING_BASELINE)
    seasons = sorted(set(today.get_column("season").to_list()))
    draws = np.random.default_rng(seed).integers(0, len(seasons), size=(resamples, len(seasons)))
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        differences = _paired_differences(predictions, horizon)
        for band in (OVERALL, *(name for name, _, _ in WEEK_BANDS)):
            band_differences = _band_rows(differences, band)
            if band_differences.is_empty():
                continue
            mean, lower, upper = season_bootstrap(band_differences, seasons, draws, confidence)
            rows.append(
                {
                    "horizon": horizon,
                    "band": band,
                    "games": band_differences.height,
                    "mae_delta": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pl.DataFrame(rows)


def compare_horizons_by_game(
    predictions: pl.DataFrame,
    horizons: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = COMPARISON_CONFIDENCE,
) -> pl.DataFrame:
    """Return the same comparisons from a bootstrap that resamples single games (descriptive)."""
    rows: list[dict[str, object]] = []
    tail = (1.0 - confidence) / 2.0
    for horizon in horizons:
        differences = _paired_differences(predictions, horizon)
        for band in (OVERALL, *(name for name, _, _ in WEEK_BANDS)):
            values = _band_rows(differences, band).get_column("difference").to_numpy()
            if values.size == 0:
                continue
            rng = np.random.default_rng(seed)
            means = np.empty(resamples)
            # In chunks, so the draws never sit in memory all at once (5,000 games by 10,000).
            for start in range(0, resamples, _RESAMPLE_CHUNK):
                size = min(_RESAMPLE_CHUNK, resamples - start)
                draws = rng.integers(0, values.size, (size, values.size))
                means[start : start + size] = values[draws].mean(axis=1)
            lower, upper = np.quantile(means, [tail, 1.0 - tail])
            rows.append(
                {
                    "horizon": horizon,
                    "band": band,
                    "games": int(values.size),
                    "mae_delta": float(values.mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pl.DataFrame(rows)


def score_bands(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return each baseline's games, MAE, and mean fitted margin slope, overall and by band."""
    rows: list[dict[str, object]] = []
    for baseline in sorted(set(predictions.get_column("baseline").to_list())):
        baseline_rows = predictions.filter(pl.col("baseline") == baseline)
        for band in (OVERALL, *(name for name, _, _ in WEEK_BANDS)):
            band_rows = _band_rows(baseline_rows, band)
            if band_rows.is_empty():
                continue
            rows.append(
                {
                    "baseline": baseline,
                    "band": band,
                    "games": band_rows.height,
                    "mae": float(np.abs(band_rows.get_column("error").to_numpy()).mean()),
                    "mean_k": float(band_rows.get_column("fitted_k").to_numpy().mean()),
                }
            )
    return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class PriorDecision:
    """The outcome of the pre-registered decision rule.

    ``excluding_zero`` lists every candidate interval that excludes zero as ``(horizon, band,
    "better" or "worse")``; ``band_guard`` lists horizons whose overall interval lay below zero
    but which a week band disqualified.
    """

    recommended_horizon: float | None
    qualifying: tuple[float, ...]
    excluding_zero: tuple[tuple[float, str, str], ...]
    band_guard: tuple[float, ...]


def decide(comparisons: pl.DataFrame, scores: pl.DataFrame) -> PriorDecision:
    """Apply the decision rule to the season-bootstrap comparisons and the overall MAEs."""
    rows = list(comparisons.iter_rows(named=True))
    horizons = sorted({float(row["horizon"]) for row in rows})
    qualifying: list[float] = []
    band_guard: list[float] = []
    for horizon in horizons:
        mine = [row for row in rows if row["horizon"] == horizon]
        below = any(row["band"] == OVERALL and row["ci_upper"] < 0.0 for row in mine)
        worse_band = any(row["band"] != OVERALL and row["ci_lower"] > 0.0 for row in mine)
        if below and not worse_band:
            qualifying.append(horizon)
        elif below:
            band_guard.append(horizon)
    band_order = [OVERALL, *(name for name, _, _ in WEEK_BANDS)]
    excluding = tuple(
        (float(row["horizon"]), str(row["band"]), "better" if row["ci_upper"] < 0.0 else "worse")
        for row in sorted(rows, key=lambda row: (row["horizon"], band_order.index(row["band"])))
        if row["ci_upper"] < 0.0 or row["ci_lower"] > 0.0
    )
    if not qualifying:
        return PriorDecision(None, (), excluding, tuple(band_guard))
    mae: dict[str, float] = dict(
        scores.filter(pl.col("band") == OVERALL).select("baseline", "mae").iter_rows()
    )
    best = min(qualifying, key=lambda horizon: mae[baseline_name(horizon)])
    return PriorDecision(best, tuple(qualifying), excluding, tuple(band_guard))


def _say(text: str) -> None:
    """Write one line of the report to standard output."""
    sys.stdout.write(f"{text}\n")


def report_decision(decision: PriorDecision) -> None:
    """Print the decision rule's outcome and every interval that excludes zero."""
    if decision.recommended_horizon is not None:
        qualifying = ", ".join(f"{horizon:g}" for horizon in decision.qualifying)
        _say(
            f"\nDecision: horizons qualifying: {qualifying} games; the recommendation is a "
            f"{decision.recommended_horizon:g}-game horizon (the lowest MAE among them)."
        )
    else:
        _say("\nDecision: no horizon qualified, so the recommendation is no prior.")
    if decision.band_guard:
        guarded = ", ".join(f"{horizon:g}" for horizon in decision.band_guard)
        _say(f"Qualified overall but worse in a week band: {guarded} games.")
    excluding = "; ".join(
        f"{baseline_name(horizon)} {band} ({direction})"
        for horizon, band, direction in decision.excluding_zero
    )
    _say(f"Candidate intervals excluding zero: {excluding or 'none'}.")
    _say("The decision goes to the maintainer either way.")


def _zero_prior(season_prior: SeasonPrior) -> SeasonPrior:
    """Return the same prior with every previous-season effect set to zero."""
    return dataclasses.replace(
        season_prior,
        offense=dict.fromkeys(season_prior.offense, 0.0),
        defense=dict.fromkeys(season_prior.defense, 0.0),
    )


@dataclass(frozen=True, slots=True)
class _Candidates:
    """Every horizon's walk-forward rows, plus what the row-level integrity checks found."""

    features: pl.DataFrame
    warm_up_gap: float
    zero_prior_gap: float
    fade_gap: float
    fade_rows: int
    audit: SnapshotAudit


def _build_candidates(
    history: PriorHistory, seasons: Sequence[int], window: Sequence[int], today: pl.DataFrame
) -> _Candidates:
    """Build every horizon's rows from ``seasons`` and check them against today's on ``window``.

    Raises:
        ValueError: If a zero prior or a fully faded prior differs from today's rows, or a
            snapshot check finds a gap beyond its tolerance.

    """
    audit = SnapshotAudit()
    frames: list[pl.DataFrame] = []
    warm_up_gap = zero_gap = fade_gap = 0.0
    fade_rows = 0
    for season in seasons:
        logs = history.game_logs(season)
        penalties = _snapshot_penalties(history, season)
        season_prior = history.season_prior(season)
        for horizon in HORIZONS:
            rows = build_prior_feature_rows(
                logs,
                season,
                horizon,
                penalties,
                season_prior,
                audit=audit if horizon in CANDIDATE_HORIZONS else None,
            )
            frames.append(rows)
            if season_prior is None:
                # Before a season has a prior, every candidate is today's fit.
                warm_up_gap = max(
                    warm_up_gap,
                    check_matching_rows(
                        today.filter(pl.col("season") == season),
                        rows.filter(pl.col("week") >= PREDICTION_START_WEEK),
                        keys=_ROW_KEYS,
                        column="rating_diff",
                        label=f"{season} {baseline_name(horizon)} before any prior",
                    ),
                )
            if season not in window or horizon not in CANDIDATE_HORIZONS:
                continue
            weeks = full_fade_weeks(logs, horizon)
            faded = rows.filter(pl.col("week").is_in(weeks))
            expected = today.filter((pl.col("season") == season) & pl.col("week").is_in(weeks))
            fade_rows += faded.height
            fade_gap = max(
                fade_gap,
                check_matching_rows(
                    expected,
                    faded,
                    keys=_ROW_KEYS,
                    column="rating_diff",
                    label=f"{season} {baseline_name(horizon)} after the fade",
                ),
            )
        if season in window and season_prior is not None:
            zero = build_prior_feature_rows(
                logs, season, CANDIDATE_HORIZONS[0], penalties, _zero_prior(season_prior)
            ).filter(pl.col("week") >= PREDICTION_START_WEEK)
            zero_gap = max(
                zero_gap,
                check_matching_rows(
                    today.filter(pl.col("season") == season),
                    zero,
                    keys=_ROW_KEYS,
                    column="rating_diff",
                    label=f"{season} zero prior",
                ),
            )
    for gap, tolerance, label in (
        (audit.means_gap, MEANS_TOLERANCE, "prior means against their recomputation"),
        (audit.residual_gap, MATCH_TOLERANCE, "solver against the residual-form fit"),
        (audit.unplayed_gap, MATCH_TOLERANCE, "teams without games against their prior"),
        (audit.limit_gap, LIMIT_TOLERANCE, "effects at a huge penalty against their means"),
    ):
        if gap > tolerance:
            msg = f"integrity check failed: {label} differ by up to {gap:.3g}"
            raise ValueError(msg)
    return _Candidates(pl.concat(frames), warm_up_gap, zero_gap, fade_gap, fade_rows, audit)


def _interval(row: dict[str, object]) -> str:
    """Format one comparison row as its difference and interval."""
    return f"{row['mae_delta']:+.3f} ({row['ci_lower']:+.3f} to {row['ci_upper']:+.3f})"


def _report_comparisons(comparisons: pl.DataFrame, horizons: Sequence[float]) -> None:
    """Print each horizon's paired difference from today's fit, overall and by band."""
    for horizon in horizons:
        rows = comparisons.filter(pl.col("horizon") == horizon).iter_rows(named=True)
        _say(
            f"  {baseline_name(horizon)}: "
            + "; ".join(f"{row['band']} {_interval(row)}" for row in rows)
        )


def _report_scores(scores: pl.DataFrame, baselines: Sequence[str]) -> None:
    """Print each baseline's MAE and mean fitted margin slope, overall and by band."""
    for baseline in baselines:
        rows = scores.filter(pl.col("baseline") == baseline).iter_rows(named=True)
        cells = "; ".join(
            f"{row['band']} {row['mae']:.3f} (k {row['mean_k']:.2f}, {row['games']} games)"
            for row in rows
        )
        _say(f"  {baseline}: {cells}")


def _report_slopes(history: PriorHistory, window: Sequence[int]) -> None:
    """Print the carryover slopes each season's prior uses."""
    cells: list[str] = []
    for season in window:
        slopes = history.slopes(season)
        if slopes is not None:
            cells.append(f"{season} {slopes.offense:.3f}/{slopes.defense:.3f}")
    _say("\nCarryover slopes (offense/defense) by season: " + ("; ".join(cells) or "none"))


def _report_spotlight(history: PriorHistory, data_dir: Path, season: int, team: str) -> None:
    """Print one team's rating and rank after week 4 of ``season``, today's and each prior's."""
    path = data_dir / f"{season}_team_game_logs.parquet"
    if not path.exists():
        _say(f"\nSpotlight: no {season} game logs in {data_dir}; skipped.")
        return
    logs = history.game_logs(season)
    early = logs.filter(pl.col("week") < SPOTLIGHT_WEEK)
    teams = sorted(set(logs.get_column("team").to_list()))
    penalties = _snapshot_penalties(history, season)
    season_prior = history.season_prior(season)
    cells: list[str] = []
    for label, prior, horizon in [
        ("today", None, math.inf),
        *((baseline_name(horizon), season_prior, horizon) for horizon in HORIZONS),
    ]:
        ratings = snapshot_ratings(early, teams, penalties, prior, horizon).sort(
            "team_rating", descending=True
        )
        order = ratings.get_column("team").to_list()
        if team not in order:
            cells.append(f"{label} unrated")
            continue
        value = float(ratings.filter(pl.col("team") == team).get_column("team_rating").item())
        cells.append(f"{label} {value:+.2f} ({order.index(team) + 1})")
    _say(
        f"\n{team} in {season} after week {SPOTLIGHT_WEEK - 1} (rating, rank): " + "; ".join(cells)
    )
    _say(f"  {season} game-log fingerprint: {input_fingerprint([path])}")


def _report_week_one(history: PriorHistory, window: Sequence[int]) -> None:
    """Print week 1's MAE with the prior alone against a home-edge-only prediction.

    The prior alone rates each team at its unfaded, centered means on the previous season's
    per-game scale; both predictions fit on earlier seasons' week-1 games only.
    """
    rows: list[dict[str, object]] = []
    for season in window:
        season_prior = history.season_prior(season)
        if season_prior is None:
            continue
        logs = history.game_logs(season)
        teams = sorted(set(logs.get_column("team").to_list()))
        means = prior_means(season_prior, dict.fromkeys(teams, 0), math.inf, teams)
        previous = scrimmage_rows(history.game_logs(season - 1))
        scale = float(previous.get_column("plays").sum()) / previous.height
        rating = {team: (means.offense[team] + means.defense[team]) * scale for team in teams}
        week_one = build_home_game_frame(logs, season).filter(pl.col("week") == 1)
        rows.extend(
            {
                "season": season,
                "gap": rating[game["home_team"]] - rating[game["away_team"]],
                "margin": game["home_margin"],
            }
            for game in week_one.iter_rows(named=True)
        )
    games = pl.DataFrame(rows, schema={"season": pl.Int64, "gap": pl.Float64, "margin": pl.Float64})
    prior_errors: list[float] = []
    home_errors: list[float] = []
    for season in sorted(set(games.get_column("season").to_list())):
        train = games.filter(pl.col("season") < season)
        if train.is_empty():
            continue
        design = np.column_stack((train.get_column("gap").to_numpy(), np.ones(train.height)))
        (slope, edge), *_ = np.linalg.lstsq(design, train.get_column("margin").to_numpy())
        home_edge = float(train.get_column("margin").to_numpy().mean())
        for gap, margin in (
            games.filter(pl.col("season") == season).select("gap", "margin").iter_rows()
        ):
            prior_errors.append(abs(slope * gap + edge - margin))
            home_errors.append(abs(home_edge - margin))
    if not prior_errors:
        _say("\nWeek 1 by the prior alone: needs two seasons with a prior; skipped.")
        return
    _say(
        f"\nWeek 1 by the prior alone: MAE {np.mean(prior_errors):.3f} against "
        f"{np.mean(home_errors):.3f} for the home edge alone ({len(prior_errors)} games)."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``check-team-prior`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings check-team-prior",
        description=(
            "Test whether shrinking the team fit toward a faded previous-season prior predicts "
            "game margins better in the walk-forward check (reads data/; writes only temporary "
            "files it removes)."
        ),
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Parquet outputs (default: {DATA_DIR})."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=DEFAULT_START_SEASON,
        help=f"First scored season (default: {DEFAULT_START_SEASON}).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=END_YEAR,
        help=f"Last scored season (default: {END_YEAR}).",
    )
    parser.add_argument(
        "--spotlight-season",
        type=int,
        default=DEFAULT_SPOTLIGHT_SEASON,
        help="Season whose ratings after week 4 are shown for each horizon.",
    )
    parser.add_argument(
        "--spotlight-team", default=DEFAULT_SPOTLIGHT_TEAM, help="Team shown by horizon."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the preseason prior test: integrity checks first, then the results and extras.

    Raises:
        ValueError: If an integrity check fails; nothing about the candidates is printed then.

    """
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    window = list(range(args.start_season, args.end_season + 1))
    seasons = list(range(PBP_START_SEASON, args.end_season + 1))
    inputs = decision_input_paths(data_dir, args.end_season)
    fingerprint = input_fingerprint(inputs)
    history = PriorHistory(_read_logs(data_dir))

    reproduction_gap = check_reproduction(history, data_dir, window)
    slope_gap = check_slopes(history, data_dir, window)
    check_coverage(history, window)
    information = check_information_set(history, data_dir, window)
    penalty_seasons = check_penalties(history, data_dir, seasons)
    # The test's baseline is the team rating without a prior, as published when it ran.
    backtest = run_walk_forward_backtest(
        data_dir, seasons, start_week=PREDICTION_START_WEEK, team_prior=False
    )
    today = backtest.filter(pl.col("baseline") == TEAM_RATING_BASELINE)
    validation = run_walk_forward_backtest(
        data_dir, seasons, start_week=VALIDATION_START_WEEK, team_prior=False
    ).filter(pl.col("baseline") == TEAM_RATING_BASELINE)
    validation_gap = max(
        check_matching_rows(
            validation,
            today.filter(pl.col("week") >= VALIDATION_START_WEEK),
            keys=_ROW_KEYS,
            column=column,
            label=f"today's fit against the validation backtest ({column})",
        )
        for column in ("rating_diff", "predicted_margin")
    )
    candidates = _build_candidates(history, seasons, window, today)
    in_window = pl.col("season").is_in(window)
    predictions = pl.concat(
        [
            today.select(*_FEATURE_COLUMNS, "error", "fitted_k"),
            evaluate_feature_rows(candidates.features, start_week=PREDICTION_START_WEEK).select(
                *_FEATURE_COLUMNS, "error", "fitted_k"
            ),
        ]
    ).filter(in_window)
    scores = score_bands(predictions)
    comparisons = compare_horizons(predictions, CANDIDATE_HORIZONS)
    decision = decide(comparisons, scores)

    _say(
        f"Preseason prior test, {args.start_season}-{args.end_season}, prediction weeks "
        f"{PREDICTION_START_WEEK} and later: horizons "
        + ", ".join(f"{horizon:g}" for horizon in CANDIDATE_HORIZONS)
        + " games against today's fit (no prior)"
    )
    _say(f"Input fingerprint: {fingerprint} ({len(inputs)} files)")
    audit = candidates.audit
    _say(
        "Integrity checks passed:\n"
        f"  previous seasons' published ratings rebuilt from each prior (largest gap "
        f"{reproduction_gap:.1e});\n"
        f"  carryover slopes match an independent exact fit (largest gap {slope_gap:.1e});\n"
        f"  today's rows equal the validation backtest's (largest gap {validation_gap:.1e});\n"
        f"  every candidate equals today's fit in the seasons before a prior (largest gap "
        f"{candidates.warm_up_gap:.1e});\n"
        f"  a zero prior equals today's fit (largest gap {candidates.zero_prior_gap:.1e});\n"
        f"  fully faded priors equal today's fit on {candidates.fade_rows} rows (largest gap "
        f"{candidates.fade_gap:.1e});\n"
        f"  on {audit.snapshots} snapshots, the means each fit used match an independent "
        f"recomputation (largest gap {audit.means_gap:.1e}) and its ratings match a residual-form "
        f"fit on those means (largest gap {audit.residual_gap:.1e});\n"
        f"  teams without games rate at their prior on {audit.unplayed_snapshots} snapshots "
        f"(largest gap {audit.unplayed_gap:.1e}); the huge-penalty limit holds on "
        f"{audit.limit_snapshots} snapshots (largest gap {audit.limit_gap:.1e});\n"
        f"  every team has a previous-season effect; {information} seasons' priors are unchanged "
        f"without later seasons; penalties equal today's in {penalty_seasons} seasons."
    )
    _say("\nMAE of predicted home margins (mean fitted slope k) by week band:")
    _report_scores(
        scores, [TEAM_RATING_BASELINE, *(baseline_name(horizon) for horizon in CANDIDATE_HORIZONS)]
    )
    _say(
        f"\nPaired MAE difference from today's fit (negative favors the prior), "
        f"{COMPARISON_CONFIDENCE:.2%} intervals, {BOOTSTRAP_RESAMPLES:,} season resamples:"
    )
    _report_comparisons(comparisons, CANDIDATE_HORIZONS)
    report_decision(decision)

    _say("\nDescriptive extras (not decision inputs):")
    _say("Single-game bootstrap, same level:")
    _report_comparisons(compare_horizons_by_game(predictions, HORIZONS), HORIZONS)
    _say("Not adoptable (a prior left in completed seasons), season bootstrap:")
    _report_comparisons(compare_horizons(predictions, EXTRA_HORIZONS), EXTRA_HORIZONS)
    _say("Their MAE (mean fitted slope k) by week band:")
    _report_scores(scores, [baseline_name(horizon) for horizon in EXTRA_HORIZONS])
    _say("Elo, the existing carry-over reference:")
    _report_scores(score_bands(backtest.filter((pl.col("baseline") == "Elo") & in_window)), ["Elo"])
    _report_slopes(history, window)
    _report_spotlight(history, data_dir, args.spotlight_season, args.spotlight_team)
    _report_week_one(history, window)


__all__ = [
    "CANDIDATE_HORIZONS",
    "COMPARISON_CONFIDENCE",
    "EXTRA_HORIZONS",
    "WEEK_BANDS",
    "PriorDecision",
    "SnapshotAudit",
    "baseline_name",
    "build_prior_feature_rows",
    "check_coverage",
    "check_information_set",
    "check_matching_rows",
    "check_penalties",
    "check_reproduction",
    "compare_horizons",
    "compare_horizons_by_game",
    "decide",
    "decision_input_paths",
    "full_fade_weeks",
    "input_fingerprint",
    "main",
    "report_decision",
    "score_bands",
    "season_bootstrap",
]
