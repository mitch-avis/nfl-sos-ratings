"""Split a team's in-season ``team_rating`` into its own play, schedule credit, and shrinkage.

Written for the 2026 Broncos question (roadmap, "Data notes"): refits the season's team ratings
three ways from the files in ``data/`` and prints, for one team, the unadjusted per-game EPA
margin, the rating with no penalty (fully adjusted for opponents), the published rating, each
rating's rank, the scrimmage shrink factors, and the published rank range. It also refits with a
prior centered on a share of the previous season's per-play effects (the preseason-prior sketch),
with and without the team's own prior. Reads ``data/`` only; writes to stdout.

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/den_rating_split.py \
        --season 2026 --team DEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_sos_ratings.ridge import UnitFit, fit_unit_ridge
from nfl_sos_ratings.team_rating import (
    TEAM_UNIT_COLUMNS,
    fit_team_ratings,
    scrimmage_rows,
    special_teams_rows,
)

DATA = Path("data")
NO_PENALTY = 1e-6
PRIOR_SHARES = (0.33, 0.45, 0.67)


def _logs(season: int) -> pl.DataFrame:
    """Return one season's team game logs from ``data/``."""
    return pl.read_parquet(DATA / f"{season}_team_game_logs.parquet")


def _ranked(ratings: dict[str, float]) -> dict[str, tuple[float, int]]:
    """Return each team's rating and its rank (1 = best; tied teams share the better rank)."""
    return {
        team: (rating, 1 + sum(other > rating for other in ratings.values()))
        for team, rating in ratings.items()
    }


def _unit_fits(
    logs: pl.DataFrame,
    penalties: tuple[float, float],
    priors: tuple[UnitFit, UnitFit] | None = None,
    share: float = 0.0,
    skip: str | None = None,
) -> dict[str, float]:
    """Return per-game team ratings from scrimmage and special-teams fits, optionally with a prior.

    With ``priors``, each unit's effects are pulled toward ``share`` times the previous season's
    per-play effects (none for ``skip``) instead of toward zero: the ordinary ridge on the residual
    response, with the prior means added back.
    """
    ratings: dict[str, float] = {}
    for rows, penalty, prior in zip(
        (scrimmage_rows(logs), special_teams_rows(logs)),
        penalties,
        priors or (None, None),
        strict=True,
    ):
        offense_prior = {} if prior is None else {t: share * v for t, v in prior.offense.items()}
        defense_prior = {} if prior is None else {t: share * v for t, v in prior.defense.items()}
        if skip is not None:
            offense_prior.pop(skip, None)
            defense_prior.pop(skip, None)
        residual = rows.with_columns(
            pl.col("epa_per_play")
            - pl.col("team").replace_strict(offense_prior, default=0.0)
            + pl.col("opponent_team").replace_strict(defense_prior, default=0.0)
        )
        fit = fit_unit_ridge(residual, TEAM_UNIT_COLUMNS, ridge_lambda=penalty)
        plays_per_game = float(rows.get_column("plays").sum()) / rows.height
        for team in set(fit.offense) | set(fit.defense):
            offense = fit.offense.get(team, 0.0) + offense_prior.get(team, 0.0)
            defense = fit.defense.get(team, 0.0) + defense_prior.get(team, 0.0)
            ratings[team] = ratings.get(team, 0.0) + (offense + defense) * plays_per_game
    return ratings


def _raw_margins(logs: pl.DataFrame, scale: tuple[float, float]) -> dict[str, float]:
    """Return each team's unadjusted per-game EPA margin against the league average."""
    league_scrimmage = logs["offensive_epa"].sum() / logs["offensive_snaps"].sum()
    league_special = logs["st_epa"].sum() / logs["st_plays"].sum()
    own = logs.group_by("team").agg(
        pl.col("offensive_epa", "offensive_snaps", "st_epa", "st_plays").sum()
    )
    faced = (
        logs.group_by("opponent_team")
        .agg(pl.col("offensive_epa", "offensive_snaps", "st_epa", "st_plays").sum())
        .rename({"opponent_team": "team"})
    )
    margins: dict[str, float] = {}
    for row in own.join(faced, on="team", suffix="_faced").iter_rows(named=True):
        offense = row["offensive_epa"] / row["offensive_snaps"] - league_scrimmage
        defense = league_scrimmage - row["offensive_epa_faced"] / row["offensive_snaps_faced"]
        special = (row["st_epa"] / row["st_plays"] - league_special) + (
            league_special - row["st_epa_faced"] / row["st_plays_faced"]
        )
        margins[row["team"]] = (offense + defense) * scale[0] + special * scale[1]
    return margins


def main() -> None:
    """Print the split for ``--team`` in ``--season``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--team", default="DEN")
    args = parser.parse_args()
    season, team = args.season, args.team
    logs = _logs(season)
    previous_fit = fit_team_ratings(_logs(season - 1))
    penalties = (previous_fit.scrimmage_lambda, previous_fit.special_teams_lambda)
    published = fit_team_ratings(
        logs, scrimmage_lambda=penalties[0], special_teams_lambda=penalties[1]
    )
    scale = (published.scrimmage_plays_per_game, published.special_teams_plays_per_game)
    out = sys.stdout.write
    out(f"{season} {team}; penalties from {season - 1} cross-validation: {penalties}\n")
    scoreboard = {
        row["team"]: row["margin"]
        for row in logs.group_by("team")
        .agg(((pl.col("points_for") - pl.col("points_allowed")).sum() / pl.len()).alias("margin"))
        .iter_rows(named=True)
    }
    for label, ratings in (
        ("scoreboard margin", scoreboard),
        ("unadjusted EPA margin", _raw_margins(logs, scale)),
        ("no penalty (fully adjusted)", _unit_fits(logs, (NO_PENALTY, NO_PENALTY))),
        ("published", _unit_fits(logs, penalties)),
    ):
        rating, rank = _ranked(ratings)[team]
        out(f"  {label}: {rating:+.2f} per game, rank {rank}\n")
    plays = logs.filter(pl.col("team") == team)["offensive_snaps"].sum()
    faced = logs.filter(pl.col("opponent_team") == team)["offensive_snaps"].sum()
    out(
        f"  scrimmage shrink factors: offense {plays / (plays + penalties[0]):.3f}, "
        f"defense {faced / (faced + penalties[0]):.3f}\n"
    )
    previous_logs = _logs(season - 1)
    two_back = fit_team_ratings(_logs(season - 2))
    priors = (
        fit_unit_ridge(
            scrimmage_rows(previous_logs), TEAM_UNIT_COLUMNS, ridge_lambda=two_back.scrimmage_lambda
        ),
        fit_unit_ridge(
            special_teams_rows(previous_logs),
            TEAM_UNIT_COLUMNS,
            ridge_lambda=two_back.special_teams_lambda,
        ),
    )
    for share in PRIOR_SHARES:
        rating, rank = _ranked(_unit_fits(logs, penalties, priors, share))[team]
        out(f"  prior at {share:.0%} of {season - 1}: {rating:+.2f}, rank {rank}\n")
    rating, rank = _ranked(_unit_fits(logs, penalties, priors, PRIOR_SHARES[1], skip=team))[team]
    out(f"  prior at {PRIOR_SHARES[1]:.0%}, none for {team} itself: {rating:+.2f}, rank {rank}\n")
    ranges = pl.read_parquet(DATA / f"{season}_rating_ranges.parquet").filter(
        pl.col("team") == team
    )
    if ranges.height:
        row = ranges.row(0, named=True)
        out(
            f"  rank range 95%: {row['team_rank_q025']:.0f}-{row['team_rank_q975']:.0f}, "
            f"top-10 chance {row['team_rank_top10_probability']:.3f}\n"
        )


if __name__ == "__main__":
    main()
