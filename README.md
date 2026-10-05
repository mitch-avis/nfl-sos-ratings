# NFL Strength of Schedule Ratings

`nfl-sos-ratings` answers one question for every NFL team and quarterback since 1999: how good
were they, relative to the opponents they actually faced? It does not treat wins and losses as the
answer. It measures play-by-play efficiency (expected points added, or EPA), adjusts it for every
opponent on the schedule and for who those opponents played, and reports the result in points.

## Table of Contents

- [The Ratings](#the-ratings)
- [How the Adjustment Works](#how-the-adjustment-works)
- [Installation](#installation)
- [Commands](#commands)
- [Data Files](#data-files)
- [Analyst Web App](#analyst-web-app)
- [Development](#development)
- [Data Sources](#data-sources)

## The Ratings

Teams, all in points per game against an average team on a neutral field:

- `team_rating`: overall quality, the sum of the three unit ratings below.
- `offense_rating`, `defense_rating`, `special_teams_rating`: what each unit added or prevented,
  adjusted for the opposing units it faced.
- `sos`: the average `team_rating` of the opponents played, one entry per game, with each opponent
  rated without its games against this team. Positive means a harder schedule.
- `SRS`: the classic point-margin rating, kept beside the EPA-based rating as a score-based
  reference.

Quarterbacks, in EPA per dropback:

- `adj_qb_epa_per_dropback`: EPA per dropback after adjusting for the pass defenses faced, on the
  same scale as the raw `qb_epa_per_dropback`.
- `qb_faced_pass_defense`: how good those defenses were, weighted by the quarterback's dropbacks,
  with each defense rated without its games against him. Positive means tougher defenses.

Wins, comebacks, turnover margin, and the rest of the box score stay in the outputs as context.
None of them feed a rating.

## How the Adjustment Works

The full write-up, including its limits, is [docs/methodology.md]. In short:

1. Every offensive play has an EPA value: how much it changed the offense's expected points.
2. One least-squares fit explains each team-game's EPA per play as league average, plus that
   offense's strength, minus that defense's strength, plus home field. Because every team is
   solved at once, an offense is judged against the defenses it faced, each of those defenses is
   judged against every offense it faced, and so on through the whole schedule.
3. A ridge penalty pulls small samples toward average. For teams its strength is the one
   cross-validation chose for the previous season, which stays reliable a few weeks into a season;
   quarterbacks cross-validate their own season.
4. Per-play strengths become points per game by multiplying by the league's average plays per
   game. Special teams gets the same treatment on kicks, punts, and returns.
5. Strength of schedule refits the model once per team without that team's games, so a team that
   beats up on an opponent cannot make that opponent look weaker in its own schedule.

Quarterbacks use the same fit with passers in place of offenses, weighted by dropbacks.

The validation harness ([docs/validation-report.md]) rebuilds the team rating each week from
earlier games only and checks how well it predicts the next games' margins against SRS, raw EPA,
and Elo.

## Installation

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). Node.js and npm are needed only for
the web app in `web/`.

```bash
uv venv .venv
uv sync
```

Dependencies live in `pyproject.toml` and `uv.lock`. Add or remove one with `uv add <pkg>`,
`uv add --group dev <pkg>`, or `uv remove <pkg>`. `./update_requirements.sh`, run from an activated
`.venv`, upgrades everything.

## Commands

Everything runs through one entry point; `--help` on it or on any command prints usage.

```bash
.venv/bin/nfl-sos-ratings season [--season N]   # one season into data/ (default: SEASON)
.venv/bin/nfl-sos-ratings pipeline              # every season START_YEAR..END_YEAR
.venv/bin/nfl-sos-ratings schedules [--team NE --season 2025]  # all-time schedule ranks
.venv/bin/nfl-sos-ratings diff-data --before DIR --after DIR  # compare two data dirs (read-only)
.venv/bin/nfl-sos-ratings validate              # rewrite docs/validation-report.md
.venv/bin/nfl-sos-ratings check-additivity      # strong units vs weak opponents (read-only)
.venv/bin/nfl-sos-ratings check-passer --name "Drake Maye"  # later games vs QB model (read-only)
.venv/bin/nfl-sos-ratings check-in-season-penalty  # early-season penalty test (read-only)
.venv/bin/nfl-sos-ratings check-wp-filter      # garbage-time filter test (read-only)
.venv/bin/nfl-sos-ratings catalog               # regenerate the stats catalogs in docs/
.venv/bin/nfl-sos-ratings web                   # analyst web app and its API
```

To see what a rebuild changed, copy `data/` first (`cp -r data /tmp/data-before`), rebuild, then
run `diff-data --before /tmp/data-before --after data`. It reports each file as unchanged, row
order only, values changed (with the changed columns, how many rows changed, and the largest
absolute difference), schema changed, added, or removed. Rows are matched by `qb_id` or `team`, then
week and game, unless `--keys` names other columns; `--season N` limits the comparison to one
season and `--tolerance X` ignores numeric differences up to `X`.

`START_YEAR`, `END_YEAR`, `SEASON`, and `DATA_DIR` live in `nfl_sos_ratings/config.py`. The full
validation run is `validate --data-dir data --start-season 1999 --end-season 2025 --start-week 5
--report-path docs/validation-report.md`. `nfl-sos` and `nfl-sos-pipeline` remain as shortcuts for
`season` and `pipeline`.

`season` and `pipeline` download from nflverse. They cache downloads on disk for a day (nflreadpy's
filesystem cache) unless `NFLREADPY_CACHE` is set to `memory`, `filesystem`, or `off`. A full
pipeline run with fresh downloads took about 13 minutes on 2026-10-04 (`time
.venv/bin/nfl-sos-ratings pipeline`).

Every command, the `nfl-sos` and `nfl-sos-pipeline` shortcuts included, runs NumPy's BLAS on one
thread unless `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, or `MKL_NUM_THREADS` is already set. The
rating fits are many small solves, where BLAS threads cost far more CPU than they save: on
2026-10-04, `time .venv/bin/nfl-sos-ratings season --season 2025` (download cache warm) took 46.5 s
wall and 8 min 3 s of user CPU with default threading on 24 cores, and 30.0 s and 39 s with one
thread, with identical output.

## Data Files

Each season writes Parquet files named `{season}_{name}.parquet` under `DATA_DIR`. Convert any of
them for a spreadsheet with `pl.read_parquet(path).write_csv(...)`. Rows come in a fixed order, so
two builds from the same inputs give identical files: `ratings`, `qb_ratings`, and the two rank-range
files best first, every other file by `qb_id` (or `team` when it has no `qb_id`), then week and
game (`row_order.data_file_row_order`).

- `ratings`: one row per team with `team_rating`, the three unit ratings, `sos`, and `SRS`.
- `qb_ratings`: one row per qualifying quarterback (14 pass attempts per game his team has played,
  in `qb_attempt_qualifier`) with raw and adjusted EPA per dropback and faced pass defense.
- `combined`: one row per team with season stats, the head-to-head-excluded opponent profile
  (`opp_` columns), and the ratings.
- `qb_combined`: one row per quarterback with season stats, the faced-defense profile (`qopp_`
  columns), and the ratings.
- `team_game_logs` and `qb_game_logs`: one row per team-game and per quarterback-game.
- `ratings_by_week` and `qb_ratings_by_week`: the rating history, one row per team (or
  quarterback) per week, each fit on the games through that week.
- `rating_ranges` and `qb_rating_ranges`: rank ranges, one row per team (or qualifying
  quarterback) with rating and rank percentiles (`_q025` through `_q975`), the chance of a top-5
  and top-10 rank, and the chance of each rank, over 1000 game-bootstrap resamples of the season;
  team rows add each unit's rank and its rating and rank percentiles.
- `rating_pairs` and `qb_rating_pairs`: head-to-head chances, one row per ordered pair of teams
  (or qualifying quarterbacks) with how often the first was rated above the second and the
  percentiles of their rating difference, from the same resamples.
- `team_wp_bins` and `qb_wp_bins`: each team-game's scrimmage and special-teams plays and EPA,
  and each quarterback-game's dropbacks and play-level passing EPA, split into 1% win-probability
  bins (`wp_bin` = `floor(100 * min(wp, 1 - wp))`, null when play-by-play has no `wp`). Summed
  over every bin they equal the totals the ratings use; they are the inputs to the garbage-time
  filter in the analyst app, which keeps the bins at or above a chosen threshold.
- `team_per_game_stats`, `qb_per_game_stats`, `opponent_profiles`, `qb_opponent_profiles`: the
  intermediate tables behind `combined` and `qb_combined`.

Every column is defined in the metric registry (`nfl_sos_ratings/metrics/`), and the pipeline
refuses to write a column the registry does not define. The human-readable lists are
[docs/stats-catalog.md] and [docs/qb-stats-catalog.md], both generated from the registry.

## Analyst Web App

The app in `web/` (React, TypeScript, Vite, Tailwind) browses the Parquet outputs. Build it once,
then serve the app and its JSON API on one port:

```bash
cd web && npm ci && npm run build && cd ..
.venv/bin/nfl-sos-ratings web   # http://127.0.0.1:8080
```

For hot reload, keep `nfl-sos-ratings web` running and start `npm run dev` in `web/`
(<http://127.0.0.1:5280>, proxying `/api` to port 8080). Details are in [web/README.md].

## Development

`scripts/gate.sh` is the full check: lock and sync checks, ruff format and lint, ty, pyright,
pytest, a `--help` check of every command, and markdownlint. `--quick` skips the tests, and `--web`
adds the frontend checks (npm ci, lint, typecheck, Vitest, build). CI
(`.github/workflows/validation.yml`) runs the gate and the frontend checks on every push and pull
request; `.venv/bin/pre-commit install` adds the commit and pre-push hooks.

pytest skips tests marked `published_data`, which read the generated files in `data/`; run them
with `.venv/bin/pytest -m published_data` after a data refresh. Agent working rules live in
[AGENTS.md].

## Data Sources

All inputs are [nflverse] data loaded through [nflreadpy]: play-by-play, weekly player and team
stats, snap counts, schedules and scores, and player and roster metadata for quarterback identity.
ESPN QBR, which nflreadpy does not expose, is downloaded from the nflverse release assets and used
only as a validation reference.

[AGENTS.md]: AGENTS.md
[docs/methodology.md]: docs/methodology.md
[docs/qb-stats-catalog.md]: docs/qb-stats-catalog.md
[docs/stats-catalog.md]: docs/stats-catalog.md
[docs/validation-report.md]: docs/validation-report.md
[nflreadpy]: https://github.com/nflverse/nflreadpy
[nflverse]: https://github.com/nflverse
[web/README.md]: web/README.md
