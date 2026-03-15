# NFL Strength of Schedule Ratings

A Python project that calculates a statistical strength-of-schedule metric for all 32 NFL teams
using [nflreadpy](https://github.com/nflverse/nflreadpy). Rather than relying on traditional
win-loss-based strength of schedule, this tool builds a comprehensive statistical profile of each
team's opponents based on how those opponents performed against the rest of the league -- excluding
head-to-head matchups.

## Table of Contents

- [NFL Strength of Schedule Ratings](#nfl-strength-of-schedule-ratings)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [How It Works](#how-it-works)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Output](#output)
    - [`team_per_game_stats.csv`](#team_per_game_statscsv)
    - [`opponent_profiles.csv`](#opponent_profilescsv)
    - [`combined.csv`](#combinedcsv)
  - [Project Structure](#project-structure)
    - [Module Descriptions](#module-descriptions)
  - [Data Sources](#data-sources)
  - [Example](#example)

## Overview

Traditional strength of schedule looks at opponents' win-loss records. This project takes a
different approach: for every team, it collects all of their regular season opponents' per-game
statistics against every other team (removing the selected team from each opponent's dataset), then
averages those profiles together. The result is a detailed picture of what kind of opponents each
team actually faced, measured across 90+ statistical categories.

## How It Works

Using the Denver Broncos as an example:

1. **Collect team stats**: Gather all of the Broncos' 2025 regular season game-by-game statistics
   and compute per-game averages.
2. **Identify opponents**: Find the Broncos' 14 unique regular season opponents (each NFL team plays
   17 games against 14 unique opponents; division rivals are played twice).
3. **Build opponent profiles**: For each of those 14 opponents, gather all of their regular season
   statistics *excluding* games played against the Broncos:
   - **Division opponents** (KC, LAC, LV): Remove 2 games vs. DEN, leaving 15 games of data.
   - **Non-division opponents**: Remove 1 game vs. DEN, leaving 16 games of data.
4. **Average opponent profiles**: Take a simple average across all 14 opponent profiles (equal
   weight per opponent), producing a single "opponent strength" profile for the Broncos.
5. **Repeat for all 32 teams** to enable league-wide comparison.

This approach removes circular bias (a team's own performance doesn't inflate or deflate their
opponents' stats) and provides a granular, multi-dimensional view of schedule difficulty.

## Installation

Requires Python 3.10+.

```bash
# Clone or navigate to the project directory
cd nfl-strength-of-schedule

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# Windows:
source .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
| ------- | ------- |
| [nflreadpy](https://github.com/nflverse/nflreadpy) | NFL data loading from nflverse |
| [Polars](https://pola.rs/) | High-performance DataFrame library (used throughout) |
| [Matplotlib](https://matplotlib.org/) | Plotting and visualization |
| [Seaborn](https://seaborn.pydata.org/) | Statistical data visualization |

## Usage

```bash
# Run the full analysis pipeline
python main.py

# Generate visualizations (requires main.py to have run first)
python visualize.py
```

`main.py` will:

1. Load game-by-game team stats, the schedule, and QB Next Gen Stats for the configured season.
2. Compute per-game averages for all 32 teams.
3. Build opponent strength profiles for all 32 teams.
4. Compute diff columns (`diff_*`) for every paired team/opponent stat.
5. Save three CSV files to the `output/` directory.
6. Print a summary table to the console.

`visualize.py` will:

1. Read `output/combined.csv`.
2. Generate five PNG charts in `output/plots/`:
   - `diffs_offense.png` -- offensive stat differentials (team minus opponent avg)
   - `diffs_defense_qb.png` -- defensive and QB stat differentials
   - `sos_opponent_strength.png` -- raw opponent-strength profiles
   - `sos_composite_ranking.png` -- composite schedule-difficulty ranking
   - `heatmap_diffs.png` -- z-scored heatmap of all diff stats across all 32 teams

## Configuration

All configuration is centralized in `config.py`:

```python
# Change this to analyze any NFL regular season
SEASON = 2025

# Output directory for CSV files
OUTPUT_DIR = "output"
```

To run the analysis for a different season, simply change the `SEASON` value and re-run `main.py`.
The NFL division mapping and QB stat columns are also defined in `config.py` and can be adjusted as
needed.

## Output

Three CSV files are generated in the `output/` directory:

### `team_per_game_stats.csv`

Each team's own per-game averages across all statistical categories, including:

- **Offensive stats**: passing yards, rushing yards, total yards, touchdowns, EPA, first downs, air
  yards, YAC, etc.
- **Defensive stats**: sacks, interceptions, tackles for loss, QB hits, passes defended, etc.
- **Special teams**: field goal percentages by distance, punt/kickoff returns, etc.
- **Scoring**: points for, points allowed (derived from game scores)
- **QB Next Gen Stats**: time to throw, CPOE, aggressiveness, air distance, passer rating, etc.

### `opponent_profiles.csv`

Each team's averaged opponent strength profile -- the same stat categories as above, but
representing the average performance of the team's 14 unique opponents (excluding
head-to-head games).

### `combined.csv`

All three datasets merged side-by-side with three column groups per stat:

- **`stat`** -- team's own per-game average (e.g. `passing_yards`)
- **`opp_stat`** -- opponents' average for the same stat (e.g. `opp_passing_yards`)
- **`diff_stat`** -- difference: team minus opponent average (e.g. `diff_passing_yards`)

Positive diff values mean the team outperformed its opponents in that category; negative values
mean the opponents were stronger. For stats where lower is better (e.g. `points_allowed`,
`sacks`), a negative diff is the advantageous direction.

## Project Structure

```text
nfl-strength-of-schedule/
├── config.py           # Season, divisions, QB stat columns, output path
├── data_loader.py      # nflreadpy wrappers; loads team stats, schedule, QB NGS data
├── team_stats.py       # Per-game stat aggregation (team-level and QB-level)
├── opponent_stats.py   # Core SoS logic: opponent profile computation
├── main.py             # Pipeline orchestrator: load, compute, save, summarize
├── visualize.py        # Visualization script: generates plots from combined.csv
├── requirements.txt    # Python dependencies
├── output/             # Generated CSV files
│   └── plots/          # Generated PNG charts (created by visualize.py)
└── README.md
```

### Module Descriptions

- **`config.py`** -- Central configuration. Change `SEASON` here to analyze any year. Contains NFL
  division mappings and the list of QB Next Gen Stats columns to extract.
- **`data_loader.py`** -- Thin wrappers around `nflreadpy` functions. Loads weekly team stats
  (enriched with `total_yards` and `points_for`/`points_allowed`), the regular season schedule, and
  QB-level Next Gen Stats (reduced to one primary QB per team per week by most pass attempts).
- **`team_stats.py`** -- Computes per-game stat averages. Includes functions for all-team
  aggregation as well as single-team aggregation with an opponent exclusion filter (used by the
  opponent profile computation).
- **`opponent_stats.py`** -- The core strength-of-schedule logic. For each team, identifies their 14
  unique opponents, computes each opponent's per-game stats excluding head-to-head matchups, and
  averages the 14 profiles together.
- **`main.py`** -- Orchestrates the full pipeline: data loading, team stat computation, opponent
  profile computation, diff column generation, CSV export, and console summary output.
- **`visualize.py`** -- Standalone visualization script. Reads `combined.csv` and generates five
  PNG charts in `output/plots/`: offensive diffs, defensive & QB diffs, raw opponent strength
  profiles, a composite schedule-difficulty ranking, and a z-scored diff heatmap.

## Data Sources

All data is sourced from the [nflverse](https://github.com/nflverse) ecosystem via `nflreadpy`:

| Function | Data | Details |
| -------- | ---- | ------- |
| `load_team_stats()` | Team game stats | ~95 statistical columns per team per game |
| `load_schedules()` | Game schedule & scores | Used for opponent identification and point totals |
| `load_nextgen_stats()` | QB Next Gen Stats | Player-level passing metrics (AWS-tracked) |

## Example

After running `python main.py`, the console output includes a summary table and opponent
detail breakdown:

```text
Opponent detail sample (DEN):
  CIN: 16 games
  DAL: 16 games
  GB: 16 games
  HOU: 16 games
  IND: 16 games
  JAX: 16 games
  KC (DIV): 15 games
  LAC (DIV): 15 games
  LV (DIV): 15 games
  NYG: 16 games
  NYJ: 16 games
  PHI: 16 games
  TEN: 16 games
  WAS: 16 games
```

Division opponents (KC, LAC, LV) correctly show 15 games of data (17 total minus 2 head-to-head
games against DEN), while non-division opponents show 16 games (17 total minus 1 head-to-head game).
