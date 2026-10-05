"""A small synthetic league with known effects and win-probability bins, shared by WP tests.

Six teams play nine home-and-away pairs. Every team-game has a close bin and a lopsided bin of
scrimmage plays, and special-teams plays in a close bin and without a bin; only DDD's offense piles
up EPA in the lopsided bin. The four passers AAA-DDD play the games among themselves, with DDD
again padding its numbers in a lopsided bin.
"""

from typing import TYPE_CHECKING

import polars as pl

from nfl_sos_ratings.qb_rating import fit_qb_ratings
from nfl_sos_ratings.team_rating import fit_team_ratings, fit_team_ratings_with_previous_penalties

if TYPE_CHECKING:
    from pathlib import Path

OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": -0.10, "EEE": 0.03, "FFF": -0.02}
DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.00, "DDD": -0.04, "EEE": 0.02, "FFF": 0.06}
# Only DDD's offense piles up EPA when the game is out of reach, so a filter pulls DDD's offense
# down and lifts the others as the league average it inflated falls.
GARBAGE_EPA_PER_PLAY = {"DDD": 0.60}
PAIRS = (
    ("AAA", "BBB"),
    ("BBB", "CCC"),
    ("CCC", "DDD"),
    ("DDD", "EEE"),
    ("EEE", "FFF"),
    ("FFF", "AAA"),
    ("AAA", "CCC"),
    ("DDD", "FFF"),
    ("BBB", "EEE"),
)
GAMES = tuple(pair for home, away in PAIRS for pair in ((home, away), (away, home)))
CLOSE_BIN = 40
LOPSIDED_BIN = 2
CLOSE_PLAYS = 48
LOPSIDED_PLAYS = 12
ST_CLOSE_PLAYS = 10
ST_UNBINNED_PLAYS = 2
# A moderate penalty, so tests show the season's penalty is the one reused.
LAMBDA = 5.0


def _bin_row(
    game: tuple[str, int, str, str, bool], unit: str, wp_bin: int | None, plays: int, epa: float
) -> dict[str, object]:
    """Return one team bin row for ``game`` = (game_id, week, team, opponent, is_home).

    ``is_home`` is not a bins column; it rides along so ``team_game_logs`` can rebuild the rows.
    """
    game_id, week, team, opponent, is_home = game
    return {
        "game_id": game_id,
        "week": week,
        "team": team,
        "opponent_team": opponent,
        "is_home": is_home,
        "wp_unit": unit,
        "wp_bin": wp_bin,
        "wp_bin_plays": plays,
        "wp_bin_epa": epa,
    }


def team_bins() -> pl.DataFrame:
    """Return every team-game's plays in a close bin, a lopsided bin, and (special teams) none."""
    rows: list[dict[str, object]] = []
    for number, (home, away) in enumerate(GAMES):
        for team, opponent, sign in ((home, away, 1.0), (away, home, -1.0)):
            game = (f"g{number:02d}", number + 1, team, opponent, sign > 0)
            close = 0.02 + OFFENSE[team] - DEFENSE[opponent] + 0.01 * sign
            garbage = GARBAGE_EPA_PER_PLAY.get(team, close)
            st = 0.05 * sign
            rows.extend(
                [
                    _bin_row(game, "scrimmage", CLOSE_BIN, CLOSE_PLAYS, close * CLOSE_PLAYS),
                    _bin_row(
                        game, "scrimmage", LOPSIDED_BIN, LOPSIDED_PLAYS, garbage * LOPSIDED_PLAYS
                    ),
                    _bin_row(game, "special_teams", CLOSE_BIN, ST_CLOSE_PLAYS, st * 10),
                    _bin_row(game, "special_teams", None, ST_UNBINNED_PLAYS, -0.3),
                ]
            )
    return pl.DataFrame(rows, schema_overrides={"wp_bin": pl.Int64})


def team_game_logs(bins: pl.DataFrame, threshold: int = 0) -> pl.DataFrame:
    """Return team-game rows summed from the bins a filter at ``threshold`` keeps."""
    kept = bins.filter(pl.col("wp_bin").is_null() | (pl.col("wp_bin") >= threshold))
    unit = pl.col("wp_unit")
    return (
        kept.group_by("game_id", "week", "team", "opponent_team", "is_home")
        .agg(
            pl.col("wp_bin_plays").filter(unit == "scrimmage").sum().alias("offensive_snaps"),
            pl.col("wp_bin_epa").filter(unit == "scrimmage").sum().alias("offensive_epa"),
            pl.col("wp_bin_plays").filter(unit == "special_teams").sum().alias("st_plays"),
            pl.col("wp_bin_epa").filter(unit == "special_teams").sum().alias("st_epa"),
        )
        .sort("game_id", "team")
    )


PASSER = {"AAA": 0.15, "BBB": 0.05, "CCC": -0.05, "DDD": -0.15}
PASS_DEFENSE = {"AAA": 0.04, "BBB": -0.02, "CCC": 0.06, "DDD": -0.08}


def qb_bins() -> pl.DataFrame:
    """Return each passer-game's dropbacks in a close and a lopsided bin; DDD pads the latter."""
    rows: list[dict[str, object]] = []
    for number, (home, away) in enumerate(
        (home, away) for home, away in GAMES if {home, away} <= set(PASSER)
    ):
        for team, opponent in ((home, away), (away, home)):
            close = 0.03 + PASSER[team] - PASS_DEFENSE[opponent]
            garbage = 0.8 if team == "DDD" else close
            for wp_bin, dropbacks, epa in ((40, 30, close * 30), (1, 8, garbage * 8)):
                rows.append(
                    {
                        "game_id": f"q{number:02d}",
                        "week": number + 1,
                        "qb_id": f"qb-{team}",
                        "wp_bin": wp_bin,
                        "qb_wp_bin_dropbacks": dropbacks,
                        "qb_wp_bin_epa": epa,
                        "opponent_team": opponent,
                    }
                )
    return pl.DataFrame(rows)


def qb_games(bins: pl.DataFrame, threshold: int = 0) -> pl.DataFrame:
    """Return passer-game rows summed from the bins a filter at ``threshold`` keeps.

    ``qb_epa_per_dropback`` is shifted by a constant, as the official weekly EPA the published
    rating uses differs from the play-level sum the bins carry.
    """
    kept = bins.filter(pl.col("wp_bin").is_null() | (pl.col("wp_bin") >= threshold))
    return (
        kept.group_by("game_id", "week", "qb_id", "opponent_team")
        .agg(
            pl.col("qb_wp_bin_dropbacks").sum().alias("qb_dropbacks"),
            pl.col("qb_wp_bin_epa").sum().alias("_epa"),
        )
        .with_columns((pl.col("_epa") / pl.col("qb_dropbacks")).alias("qb_epa_per_dropback"))
        .drop("_epa")
        .sort("game_id", "qb_id")
    )


def write_wp_season(data_dir: Path, season: int) -> None:
    """Write a season's game logs, bins, and published ratings, plus the previous season's logs.

    The published ratings come from the same fits the season command runs, so a 0% filter must
    reproduce them.
    """
    bins = team_bins()
    logs = team_game_logs(bins)
    logs.write_parquet(data_dir / f"{season - 1}_team_game_logs.parquet")
    logs.write_parquet(data_dir / f"{season}_team_game_logs.parquet")
    bins.drop("is_home").write_parquet(data_dir / f"{season}_team_wp_bins.parquet")
    fit = fit_team_ratings_with_previous_penalties(logs, fit_team_ratings(logs))
    fit.ratings.select("team", "team_rating").write_parquet(data_dir / f"{season}_ratings.parquet")
    passer_bins = qb_bins()
    official = qb_games(passer_bins).with_columns(pl.col("qb_epa_per_dropback") + 0.01)
    official.write_parquet(data_dir / f"{season}_qb_game_logs.parquet")
    passer_bins.drop("opponent_team").write_parquet(data_dir / f"{season}_qb_wp_bins.parquet")
    fit_qb_ratings(official).ratings.with_columns(
        pl.col("qb_id").str.replace("qb-", "").alias("team"),
        pl.col("qb_id").str.replace("qb-", "Passer ").alias("qb_name"),
    ).write_parquet(data_dir / f"{season}_qb_ratings.parquet")
