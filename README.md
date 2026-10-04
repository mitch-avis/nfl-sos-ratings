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
.venv/bin/nfl-sos-ratings validate              # rewrite docs/validation-report.md
.venv/bin/nfl-sos-ratings check-additivity      # strong units vs weak opponents (read-only)
.venv/bin/nfl-sos-ratings check-passer --name "Drake Maye"  # later games vs QB model (read-only)
.venv/bin/nfl-sos-ratings check-in-season-penalty  # early-season penalty test (read-only)
.venv/bin/nfl-sos-ratings catalog               # regenerate the stats catalogs in docs/
.venv/bin/nfl-sos-ratings web                   # analyst web app and its API
```

`START_YEAR`, `END_YEAR`, `SEASON`, and `DATA_DIR` live in `nfl_sos_ratings/config.py`. The full
validation run is `validate --data-dir data --start-season 1999 --end-season 2025 --start-week 5
--report-path docs/validation-report.md`. `nfl-sos` and `nfl-sos-pipeline` remain as shortcuts for
`season` and `pipeline`.

`season` and `pipeline` download from nflverse. They cache downloads on disk for a day (nflreadpy's
filesystem cache) unless `NFLREADPY_CACHE` is set to `memory`, `filesystem`, or `off`. A full
pipeline run with fresh downloads took about 13 minutes on 2026-10-04 (`time
.venv/bin/nfl-sos-ratings pipeline`).

## Data Files

Each season writes Parquet files named `{season}_{name}.parquet` under `DATA_DIR`. Convert any of
them for a spreadsheet with `pl.read_parquet(path).write_csv(...)`.

- `ratings`: one row per team with `team_rating`, the three unit ratings, `sos`, and `SRS`.
- `qb_ratings`: one row per qualifying quarterback (14 attempts per team game) with raw and
  adjusted EPA per dropback and faced pass defense.
- `combined`: one row per team with season stats, the head-to-head-excluded opponent profile
  (`opp_` columns), and the ratings.
- `qb_combined`: one row per quarterback with season stats, the faced-defense profile (`qopp_`
  columns), and the ratings.
- `team_game_logs` and `qb_game_logs`: one row per team-game and per quarterback-game.
- `ratings_by_week` and `qb_ratings_by_week`: the rating history, one row per team (or
  quarterback) per week, each fit on the games through that week.
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
