"""Per season, how much of each early-season-gap field nflverse fills (roadmap, "Data notes").

Evidence for the seasons the loaders treat as having no value for a field (``_PBP_FIELD_GAPS`` and
``_PLAYER_STAT_GAPS`` in ``data_loader``). Reads nflverse's raw regular-season play-by-play and
weekly player stats through nflreadpy (cached), before the loaders blank anything, and prints one
row per season:

- pass attempts (no sacks or two-point tries) and how many carry air yards, air EPA, and a pass
  depth; completions and how many carry yards after catch, YAC EPA, and expected YAC;
- plays flagged as QB hits, and how many of them are sacks;
- plays with drive penalty yards (non-null);
- scrimmage snaps flagged no-huddle, and the flagged share of a trailing offense's snaps in the
  last two minutes of a half (a hurry-up offense, which almost never huddles), so a season that
  records the flag less often shows a lower share;
- plays flagged as kneel-downs;
- tackles for loss and QB hits credited in the weekly player stats (season totals).

Run from the repository root:

    POLARS_MAX_THREADS=1 .venv/bin/python .agents/findings_2026_10_08/source_coverage.py 1999 2025
"""

from __future__ import annotations

import sys

import nflreadpy as nfl
import polars as pl

from nfl_sos_ratings.data_loader import use_disk_cache_unless_configured

PASS_FIELDS = ("air_yards", "air_epa", "pass_length")
CATCH_FIELDS = ("yards_after_catch", "yac_epa", "xyac_mean_yardage")
PLAYER_FIELDS = ("def_tackles_for_loss", "def_qb_hits")
# The end of a half: a trailing offense's snaps in its last two minutes.
TWO_MINUTES = 120


def _non_null(frame: pl.DataFrame, column: str) -> int:
    """Return how many rows of ``frame`` have a value in ``column`` (0 when it is absent)."""
    return int(frame.get_column(column).is_not_null().sum()) if column in frame.columns else 0


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def season_row(season: int) -> dict[str, int]:
    """Return one season's coverage counts."""
    pbp = nfl.load_pbp([season]).filter(pl.col("season_type") == "REG")
    passes = pbp.filter(_flag("pass_attempt") & ~_flag("sack") & ~_flag("two_point_attempt"))
    completions = pbp.filter(_flag("complete_pass"))
    hits = pbp.filter(_flag("qb_hit"))
    players = nfl.load_player_stats([season], summary_level="week").filter(
        pl.col("season_type") == "REG"
    )
    row = {"season": season, "pass_attempts": passes.height}
    row |= {field: _non_null(passes, field) for field in PASS_FIELDS}
    row["completions"] = completions.height
    row |= {field: _non_null(completions, field) for field in CATCH_FIELDS}
    row["qb_hit_plays"] = hits.height
    row["qb_hit_sacks"] = hits.filter(_flag("sack")).height
    row["drive_penalty_plays"] = _non_null(pbp, "drive_yards_penalized")
    snaps = pbp.filter((_flag("qb_dropback") | _flag("rush")) & ~_flag("qb_kneel"))
    hurry_up = snaps.filter(
        pl.col("qtr").cast(pl.Int64).is_in([2, 4])
        & (pl.col("half_seconds_remaining") <= TWO_MINUTES)
        & (pl.col("score_differential") < 0)
    )
    row["no_huddle_snaps"] = snaps.filter(_flag("no_huddle")).height
    row["two_minute_no_huddle_pct"] = round(
        100 * hurry_up.filter(_flag("no_huddle")).height / max(hurry_up.height, 1), 1
    )
    row["kneel_plays"] = pbp.filter(_flag("qb_kneel")).height
    row |= {f"credited_{field}": int(players.get_column(field).sum()) for field in PLAYER_FIELDS}
    return row


def main() -> None:
    """Print the coverage table for the seasons from the first argument to the second."""
    use_disk_cache_unless_configured()
    first, last = (int(argument) for argument in sys.argv[1:3])
    table = pl.DataFrame([season_row(season) for season in range(first, last + 1)])
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f"{table}\n")


if __name__ == "__main__":
    main()
