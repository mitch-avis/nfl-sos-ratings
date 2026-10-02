"""Rebuild the frozen 2025 QSoS anchor fixture from the published outputs in ``data/``.

Run from the repo root after regenerating ``data/``:

    .venv/bin/python tests/fixtures/qsos_anchor_2025/build_fixture.py

It writes the four anchor QBs' game logs and the league's team ratings (only the columns the
anchor reads) next to this script, then prints the equal-game opponent means to pin in
``tests/test_qsos_audit.py``.
"""

from pathlib import Path
from pprint import pformat

import polars as pl

SEASON = 2025
ANCHOR_QBS = ["Drake Maye", "Tyler Shough", "Joe Flacco", "J.J. McCarthy"]
SOURCE_DIR = Path("data")
FIXTURE_DIR = Path(__file__).parent


def main() -> None:
    """Write the fixture Parquet files and print the anchor values to pin."""
    qb_logs = (
        pl.read_parquet(SOURCE_DIR / f"{SEASON}_qb_game_logs.parquet")
        .filter(pl.col("qb_name").is_in(ANCHOR_QBS))
        .select(["qb_name", "team", "week", "opponent_team"])
        .sort(["qb_name", "week"])
    )
    team_ratings = (
        pl.read_parquet(SOURCE_DIR / f"{SEASON}_ratings.parquet")
        .select(["team", "SaCR", "SaDR", "SRS"])
        .sort("team")
    )
    qb_logs.write_parquet(FIXTURE_DIR / f"{SEASON}_qb_game_logs.parquet")
    team_ratings.write_parquet(FIXTURE_DIR / f"{SEASON}_ratings.parquet")

    anchor: dict[str, dict[str, float | int]] = {}
    for qb_name in ANCHOR_QBS:
        joined = qb_logs.filter(pl.col("qb_name") == qb_name).join(
            team_ratings, left_on="opponent_team", right_on="team", how="left"
        )
        anchor[qb_name] = {"games": joined.height} | {
            f"avg_opp_{column}": float(joined.select(pl.col(column).mean()).item())
            for column in ("SaCR", "SaDR", "SRS")
        }
    print(pformat(anchor, sort_dicts=False))


if __name__ == "__main__":
    main()
