# NFL Strength of Schedule Ratings

`nfl-sos-ratings` measures how good NFL teams and quarterbacks actually were, relative to the
opponents they actually faced that season. It does not treat wins and losses as ground truth.
Instead, it builds PBP-driven offensive, defensive, and quarterback profiles, removes head-to-head
leakage from opponent profiles, and compares subjects to the adjusted quality of their schedules.

It currently produces:

- Team ratings: `SaOR`, `SaDR`, `SaOvR`, `SaCR`, and `SRS`
- QB ratings: `QRaw`, `QSoS`, `QSaOR`, `QOutcome`, and `QSaCR`
- One-hop diff-based comparison outputs
- Simultaneous-adjustment outputs for teams and QBs
- A machine-readable metric registry (`nfl_sos_ratings/metrics/`) — the single source of truth
  for every published stat's label, layman description, polarity, category, source, and
  rating-pool eligibility, served to the web UI at `/api/metadata`
- Intermediate Parquet artifacts for auditability (convert any file to CSV with
  `pl.read_parquet(...).write_csv(...)` for spreadsheet inspection)
- Team and QB plots under `data/plots/`

The registry-backed analyst surfaces use a six-view model:

- `Ratings`
- `Raw Total Stats`
- `Per-Game Rates`
- `Per-Play Rates`
- `Opponent Per-Game Rates`
- `Opponent Per-Play Rates`

`Ratings` is its own schedule-adjusted view.
The other five reuse the same team/QB taxonomies, and opponent context is expressed through the
two opponent views rather than through a standalone `Opponent Context` category.

## Table of Contents

- [NFL Strength of Schedule Ratings](#nfl-strength-of-schedule-ratings)
  - [Table of Contents](#table-of-contents)
  - [What It Does](#what-it-does)
  - [Current Methodology](#current-methodology)
    - [Data Inputs](#data-inputs)
    - [Team Pipeline](#team-pipeline)
    - [QB Pipeline](#qb-pipeline)
    - [Opponent Profiling Rules](#opponent-profiling-rules)
    - [Simultaneous Adjustment](#simultaneous-adjustment)
    - [Derived Formulas](#derived-formulas)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [How to Run](#how-to-run)
  - [Data Files](#data-files)
    - [Team outputs](#team-outputs)
    - [QB outputs](#qb-outputs)
    - [Plot outputs](#plot-outputs)
  - [Project Structure](#project-structure)
  - [Development Commands](#development-commands)
  - [Troubleshooting](#troubleshooting)
    - [Import path issues](#import-path-issues)
    - [Missing plot files](#missing-plot-files)
    - [Missing QB opponent rows](#missing-qb-opponent-rows)
  - [Data Sources](#data-sources)

## What It Does

Traditional strength-of-schedule models lean on opponent records. This project instead asks:

- How did a team perform per game and per snap?
- How did a QB perform per game and per dropback?
- How strong were the exact opponents and defenses they faced?
- What changes after you exclude head-to-head leakage and compare against that adjusted opponent
  context?

There are two separate rating systems:

- Teams: offense, defense, overall, and composite ratings
- Quarterbacks: primary passing-performance ratings plus schedule and outcome context

## Current Methodology

### Data Inputs

The live pipeline is PBP-first.

- `load_pbp()` is the source of record for team-game and QB-game production, snap counts via play
  counts, EPA, CPOE, sacks, sack-yard losses, and most mirrored defensive stats.
- `load_snap_counts()` supplements QB participation and helps identify the primary QB for a
  team-week.
- `load_players()` and `load_rosters_weekly()` provide the GSIS/PFR identity crosswalk used to merge
  PBP and snap-count QB rows onto one canonical QB identity.
- `load_player_stats(summary_level="week")` is the authoritative source for official QB
  attempt-based passing totals and published QB passing stats, and it is also used for defense-only
  team stats that are not easily summarized directly from PBP, such as tackles for loss, QB hits,
  passes defended, and safeties.
- `load_team_stats(summary_level="week")` is the authoritative source for the published team offense
  surface, including yards, TDs, first-down splits, EPA splits, CPOE, sacks suffered, interceptions,
  and fumble-loss splits.
- `load_schedules()` supplies official scores for team outcomes.

### Team Pipeline

The team path is:

1. Build one PBP-derived row per team-game.
2. Derive both per-game totals and per-snap rates.
3. Build opponent profiles from each unique opponent's non-head-to-head games.
4. Join team and opponent profiles and emit `diff_*` columns.
5. Produce equal-weight team ratings:
   - `SaOR`: offense
   - `SaDR`: defense
   - `SaOvR`: overall, built from `win_value` and `turnover_margin`
   - `SaCR`: equal blend of offense, defense, and overall
6. Produce `SRS` from point margin as a simultaneous-adjustment reference.

Team rating pools now use per-snap and rate-like fields rather than raw season totals.

### QB Pipeline

The QB path is:

1. Build one PBP-derived row per QB-game.
2. Canonicalize each QB row to a GSIS-based identity, using PFR IDs and roster/player metadata to
   merge abbreviated PBP names with full-name snap-count rows.
3. Replace attempt-based game fields with authoritative weekly `player_stats` values for attempts,
   completions, passing yards, passing TDs, interceptions, sacks, sack yards lost, passing EPA, and
   passing CPOE.
4. Derive dropbacks, snaps, EPA/dropback, ANY/A, sack rate, yards per dropback, TD-INT margin rate,
   and passer rating from the corrected official/PBP inputs.
5. Derive late-game secondary stats from PBP score state:
   - fourth-quarter comebacks
   - game-winning drives
6. Assign wins, 4QC, and GWD only to the primary QB for the team-week, chosen by snaps, then
   dropbacks, then attempts.
7. Build QB opponent profiles from only the primary-QB games each QB actually played.
8. Deduplicate faced defenses before profiling and remove the old scheduled-opponent fallback.
9. Produce equal-weight QB ratings with a fixed outcome blend.

The primary QB stat pool is now centered on:

- `qb_epa_per_dropback`
- `qb_any_a`
- `qb_completion_percentage_above_expectation`
- `qb_td_int_margin_rate`
- `qb_sack_rate`

Secondary QB context remains available through fields such as passer rating, pass yards per
dropback, wins, fourth-quarter comebacks, and game-winning drives.

### Opponent Profiling Rules

Both pipelines follow the same core rules:

- Regular season only
- Normalize team abbreviations before joins
- Exclude all head-to-head games when profiling an opponent
- Deduplicate opponent lists before averaging
- Compare on per-game and per-play rates.
  For teams that often means per-snap; for QBs it usually means per-dropback, per-attempt, or
  per-carry depending on the subcategory.

### Simultaneous Adjustment

The repo now includes `nfl_sos_ratings/simultaneous_adjustment.py`.

It currently provides:

- `solve_srs()` for point-differential SRS
- `solve_team_stat_ridge()` for team offense/defense latent ratings
- `solve_qb_stat_ridge()` for QB offense vs defense-allowed latent ratings
- wrapper helpers that emit multi-stat adjusted tables for teams and QBs

The main pipeline writes these simultaneous-adjustment outputs alongside the existing diff-based
outputs so the two approaches can be compared directly.

### Derived Formulas

Key self-computed metrics use the following formulas:

- Team per-snap rates: game total divided by offensive or defensive snaps
- QB EPA per dropback: `qb_passing_epa / qb_dropbacks`
- QB pass yards per dropback: `qb_pass_yards / qb_dropbacks`
- QB TD-INT margin rate: `(qb_pass_touchdowns - qb_interceptions) / qb_dropbacks`
- QB sack rate: `qb_sacks / qb_dropbacks`
- QB ANY/A: `(qb_pass_yards + 20 * qb_pass_touchdowns - 45 * qb_interceptions - qb_sack_yards_lost)
/ (qb_attempts + qb_sacks)`
- Fourth-quarter comeback: primary QB on the eventual game winner, where the offense had at least
  one quarter-4-or-later snap while trailing and the team's final score exceeded the opponent's
  final score
- Game-winning drive: primary QB on the eventual game winner, where the offense had a quarter-4-or-
  later scoring play that moved the score from tied/trailing to leading and the team's final score
  exceeded the opponent's final score

Opponent-allowed QB rate fields and defensive mirror stats reuse the same formulas after applying
the head-to-head exclusion rule.

## Requirements

- Python 3.14+
- Linux, macOS, or Windows
- Local virtual environment at `.venv`

Runtime dependencies are pinned in `requirements.txt`.

## Installation

```bash
cd nfl-sos-ratings
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Configuration

Edit `nfl_sos_ratings/config.py`.

```python
SEASON: int = 2025
DATA_DIR: str = "data"
```

- `SEASON` selects the target season for `main.py`
- `DATA_DIR` selects where Parquet outputs and plots are written

## How to Run

Primary single-season pipeline:

```bash
python -m nfl_sos_ratings.main
```

Full multi-season pipeline:

```bash
python -m nfl_sos_ratings.pipeline
```

Visualization pass:

```bash
python -m nfl_sos_ratings.visualize
```

Local analyst UI backend:

```bash
python -m nfl_sos_ratings.ui_api
```

Local analyst UI frontend:

```bash
cd ui/web
npm install
npm run dev
```

Detailed frontend usage and troubleshooting notes live in `ui/web/README.md`.

## Data Files

All files are written under `DATA_DIR` with a `{SEASON}_` prefix.

### Team outputs

- `{SEASON}_team_per_game_stats.parquet` PBP-derived team-game profile rolled to one season row per
  team. Includes per-game totals, per-snap rates, `win_value`, and `turnover_margin`.
- `{SEASON}_opponent_profiles.parquet` Averaged opponent profile rows built from unique non-head-to-head
  opponents.
- `{SEASON}_combined.parquet` Team rows joined to opponent rows, `diff_*` columns, team ratings, `SRS`,
  and simultaneous-adjustment team columns.
- `{SEASON}_ratings.parquet` Compact team ratings summary with `SaCR`, `SaOR`, `SaDR`, `SaOvR`, and
  `SRS`.
- `{SEASON}_simultaneous_team_adjustments.parquet` Multi-stat simultaneous-adjustment output with
  `adj_off_*` and `adj_def_*` columns.

### QB outputs

- `{SEASON}_qb_per_game_stats.parquet` QB season summary keyed by canonical QB identity and team
  context. Includes explicit season totals such as `qb_attempts_total`, `qb_completions_total`, and
  `qb_pass_yards_total`; explicit per-game fields such as `qb_attempts_per_game`,
  `qb_completions_per_game`, and `qb_pass_yards_per_game`; dropback and snap totals; EPA per
  dropback; ANY/A; sack rate; yards per dropback; TD-INT differential fields; wins; and 4QC/GWD
  totals.
- `{SEASON}_qb_opponent_profiles.parquet` QB opponent context built from only the primary-QB games each
  QB actually played, with unique faced defenses and no fabricated schedule fallback.
- `{SEASON}_qb_combined.parquet` QB season rows joined to opponent context, `diff_qb_*` columns,
  simultaneous QB adjustment columns, and final QB ratings.
- `{SEASON}_qb_ratings.parquet` Compact QB ratings summary for qualified passers.
- `{SEASON}_simultaneous_qb_adjustments.parquet` Multi-stat simultaneous-adjustment QB output with
  `adj_*` columns.

### Plot outputs

Under `data/plots/`:

- `{SEASON}_adjusted_ratings_offense.png`
- `{SEASON}_adjusted_ratings_defense.png`
- `{SEASON}_adjusted_ratings_overall.png`
- `{SEASON}_sos_composite_ranking.png`
- `{SEASON}_qb_adjusted_ratings.png` when QB combined data exists
- `{SEASON}_qb_raw_vs_schedule.png` when QB combined data exists

## Project Structure

```text
nfl-sos-ratings/
├── nfl_sos_ratings/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── main.py
│   ├── opponent_stats.py
│   ├── pipeline.py
│   ├── qb_opponent_stats.py
│   ├── qb_ratings.py
│   ├── qb_stats.py
│   ├── ratings.py
│   ├── simultaneous_adjustment.py
│   ├── team_stats.py
│   └── visualize.py
├── tests/
├── ui/
│   └── web/
├── data/
├── .agents/
├── pyproject.toml
└── README.md
```

The active implementation handoff document for the repo's current state and backlog is in
`.agents/current-status.md`.

## Development Commands

From repository root:

```bash
ruff format .
ruff check .
ty check .
pyright .
pytest
```

Frontend build check:

```bash
cd ui/web
npm run build
```

## Troubleshooting

### Import path issues

Use the module form when possible:

```bash
python -m nfl_sos_ratings.main
```

### Missing plot files

Run the data pipeline first, then the visualization pass:

```bash
python -m nfl_sos_ratings.main
python -m nfl_sos_ratings.visualize
```

### Missing QB opponent rows

QBs with no reconstructable faced-opponent list are now skipped instead of receiving a fabricated
schedule. If that happens, inspect the underlying PBP and snap-count rows for missing QB identity or
participation context.

## Data Sources

All data is loaded through nflreadpy from nflverse sources.

Current live inputs:

- Play-by-play data
- Weekly player stats
- Snap counts
- Schedules and scores

nflverse: <https://github.com/nflverse> nflreadpy: <https://github.com/nflverse/nflreadpy>
