# NFL Strength of Schedule Ratings

This project computes schedule-strength-adjusted NFL team and quarterback ratings from nflverse via
nflreadpy and Polars.

It produces:

- Team-level schedule-adjusted ratings (`SaOR`, `SaDR`, `SaCR`)
- Quarterback-level schedule-adjusted ratings (`QRaw`, `QSoS`, `QSaOR`, `QSaCR`)
- Intermediate CSV artifacts for auditing each stage
- Visualization plots for team and QB views

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Method Summary](#method-summary)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Visualization Output](#visualization-output)
- [Project Structure](#project-structure)
- [Development Commands](#development-commands)
- [Troubleshooting](#troubleshooting)
- [Data Source](#data-source)

## What This Project Does

Traditional strength-of-schedule methods use opponent win/loss records. This project instead uses
opponent statistical profiles while explicitly removing head-to-head games between the evaluated
team and each opponent.

At a high level:

1. Build each team's per-game statistical profile.
2. For each team, build opponent profiles from opponents' non-head-to-head games.
3. Compare team profile vs opponent profile via `diff_*` features.
4. Produce team schedule-adjusted ratings.
5. Build QB season profiles from individual quarterback game rows.
6. Build QB opponent context and QB allowed-by-defense context (`qopp_*`).
7. Produce QB schedule-adjusted ratings with historical calibration.

## Method Summary

### Team ratings (`SaOR`, `SaDR`, `SaCR`)

Team ratings are computed in `nfl_sos_ratings/ratings.py` from the combined team/opponent dataset:

- Offensive and defensive stat pools are predefined.
- Stat weights are derived from correlation with `win_pct` (thresholded by minimum correlation).
- Offensive and defensive schedule signals are built from opponent context columns.
- Final offense and defense ratings are z-scored (`SaOR`, `SaDR`).
- Composite `SaCR` blends offense and defense based on their observed correlation with `win_pct`.

### Quarterback ratings (`QRaw`, `QSoS`, `QSaOR`, `QSaCR`)

QB ratings are computed in `nfl_sos_ratings/qb_ratings.py`:

- Input is qualified individual QB season rows (not one QB per team).
- Qualification defaults to at least 14 pass attempts per scheduled team game: 238 attempts in a
  17-game regular season.
- `QRaw` uses weighted conventional production-efficiency metrics as the baseline QB performance
  view: passer rating, completion percentage over expectation, yards per attempt, touchdown rate,
  and interception rate.
- `QSaOR` and `QSaCR` primarily use paired QB/opponent production context. Each matched QB stat and
  opponent-allowed stat is standardized separately before adjustment, so weak or strong schedules
  have comparable z-score impact instead of being washed out by raw stat scale.
- `diff_qb_*` columns are still emitted for auditability and fallback cases where paired opponent
  columns are unavailable.
- `QSoS` uses opponent defensive and allowed-QB production context (`qopp_*` columns), excluding
  style-only QB fields such as aggressiveness and air-yards depth. Repeated opponents are counted by
  the QB games actually faced; the unique team-schedule fallback is used only when QB game opponent
  rows are unavailable.
- `QSaOR` is the schedule-adjusted passing-performance signal.
- `QOutcome` is the z-scored QB game-result signal from `qb_win_pct` when available.
- `QSaCR` is the final QB composite (z-scored), blending schedule-adjusted passing performance,
  schedule difficulty, and the calibrated outcome signal.
- Percentile columns are also emitted.

### Historical QB calibration

`main.py` calibrates QB model constants before rating the target season:

- Historical seasons: previous 5 years (`SEASON - 5` to `SEASON - 1`, bounded at 2006+)
- Historical calibration uses the same QB opponent-context and differential construction as the
  target season when source data is available.
- Grid search over:
  - minimum correlation threshold
  - material schedule weight, constrained so schedule context cannot be washed out by a near-zero
    value
  - outcome weight, capped so `qb_win_pct` informs `QSaCR` without replacing passing context
- Objective: maximize correlation between calibrated QB composite and QB outcome target.
- Fallback defaults if historical calibration data is unavailable:
  - `min_correlation = 0.1`
  - `sos_weight = 2.0`
  - `outcome_weight = 0.75`

## Requirements

- Python 3.12+
- Linux/macOS/Windows
- A local virtual environment at `.venv`

Runtime dependencies are in `requirements.txt`.

## Installation

```bash
cd nfl-sos-ratings
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional developer tooling install:

```bash
uv pip install -r requirements-dev.txt
```

## Configuration

Edit `nfl_sos_ratings/config.py`:

```python
SEASON: int = 2025
OUTPUT_DIR: str = "output"
```

- `SEASON` controls the target season for the pipeline.
- `OUTPUT_DIR` controls where CSV and plot artifacts are written.

## How to Run

### Primary pipeline

Recommended module form:

```bash
python -m nfl_sos_ratings.main
```

Direct script path also works:

```bash
python nfl_sos_ratings/main.py
```

### Visualization pipeline

Recommended module form:

```bash
python -m nfl_sos_ratings.visualize
```

Direct script path also works:

```bash
python nfl_sos_ratings/visualize.py
```

## Output Files

All outputs are prefixed with `SEASON` in `OUTPUT_DIR`.

### Team artifacts

#### `{SEASON}_team_per_game_stats.csv`

Team per-game profile, including:

- Team offensive and defensive game stats
- Team QB aggregate stats (team view)
- Win totals and `win_pct`

#### `{SEASON}_opponent_profiles.csv`

Opponent profile averages for each team, built from opponents' non-head-to-head games.

#### `{SEASON}_combined.csv`

Team and opponent profile join with derived differentials:

- Team columns: `<stat>`
- Opponent columns: `opp_<stat>`
- Differential columns: `diff_<stat> = <stat> - opp_<stat>`

Includes final team ratings:

- `SaOR`
- `SaDR`
- `SaCR`

#### `{SEASON}_ratings.csv`

Compact team ratings summary with:

- `team`
- `games_played`
- `SaCR`, `SaOR`, `SaDR`

### QB artifacts

#### `{SEASON}_qb_per_game_stats.csv`

QB season summary table keyed by QB identity (`qb_id`, `qb_name`) and team context.

Includes:

- `qb_games_played`
- `qb_attempts_total`
- `qb_is_eligible` (default threshold: 238 attempts)
- `qb_win_pct` (when weekly points columns are available)
- Derived attempt-normalized efficiency rates: `qb_yards_per_attempt`, `qb_touchdown_rate`, and
  `qb_interception_rate`
- Mean `qb_*` metrics

#### `{SEASON}_qb_opponent_profiles.csv`

QB opponent context. Each row uses the opponents from the individual quarterback's actual game rows,
including repeat matchups. It does not use every opponent on that quarterback's team schedule unless
QB game/opponent rows are unavailable and a sparse-data fallback is needed.

Includes:

- Defensive context (e.g. `qopp_points_allowed`, `qopp_def_sacks`)
- Allowed-QB context from opposing defenses (e.g. `qopp_qb_passer_rating`)

All values exclude head-to-head leakage via opponent-exclusion logic.

#### `{SEASON}_qb_combined.csv`

QB season stats joined with QB opponent profile context.

Includes:

- `diff_qb_*` columns where matched QB and opponent-allowed pairs exist
- `QSaOR` from standardized paired QB/opponent production-efficiency context
- `QRaw`, `QSoS`, `QSaOR`, `QOutcome`, `QSaCR`
- Percentiles: `QRaw_pct`, `QSoS_pct`, `QSaOR_pct`, `QOutcome_pct`, `QSaCR_pct`

#### `{SEASON}_qb_ratings.csv`

Compact qualified-QB ratings table with identity columns, rating columns, and summary fields.

## Visualization Output

Generated under `output/plots/`.

### Team plots

- `{SEASON}_adjusted_ratings_offense.png`
- `{SEASON}_adjusted_ratings_defense.png`
- `{SEASON}_adjusted_ratings_overall.png`
- `{SEASON}_sos_composite_ranking.png`

### QB plots

If `{SEASON}_qb_combined.csv` exists:

- `{SEASON}_qb_adjusted_ratings.png`
- `{SEASON}_qb_raw_vs_schedule.png`

QB plots label individual players as `QB Name (TEAM)` and filter to qualified quarterbacks when the
eligibility column is available.

## Project Structure

```text
nfl-sos-ratings/
├── nfl_sos_ratings/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── team_stats.py
│   ├── opponent_stats.py
│   ├── ratings.py
│   ├── qb_stats.py
│   ├── qb_opponent_stats.py
│   ├── qb_ratings.py
│   ├── main.py
│   └── visualize.py
├── tests/
├── output/
├── pyproject.toml
├── requirements.in
├── requirements.txt
├── requirements-dev.in
├── requirements-dev.txt
└── README.md
```

## Development Commands

From repository root:

```bash
ruff format .
ruff check .
pyright .
pytest
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'nfl_sos_ratings'`

Use either:

- `python -m nfl_sos_ratings.main` (recommended)
- `python nfl_sos_ratings/main.py` (supported)

The script includes bootstrapping for direct path execution.

### Missing output files for plots

Run the primary pipeline first:

```bash
python -m nfl_sos_ratings.main
python -m nfl_sos_ratings.visualize
```

### No QB opponent profile rows for some entries

QB opponent profile generation requires enough matching weekly/team context for each QB season row.
Sparse or partial input data can produce fewer QB opponent rows than QB season rows.

## Data Source

All data is loaded via nflreadpy from nflverse datasets:

- Weekly team stats
- Schedules and game scores
- Next Gen Stats passing data

nflverse: <https://github.com/nflverse> nflreadpy: <https://github.com/nflverse/nflreadpy>
