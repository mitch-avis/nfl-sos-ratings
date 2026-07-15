# AGENTS.md

Guidance for AI coding agents working on `nfl-sos-ratings`. Human-facing docs live in
[`README.md`](README.md); this file covers what an agent needs to work effectively and safely.

## Project overview

`nfl-sos-ratings` computes schedule-strength-adjusted NFL ratings for teams and quarterbacks from
nflverse data. It answers "how good was this team or QB, relative to the opponents they actually
faced?" — not "what was their record?" Wins and losses are treated as noisy outcomes the ratings are
meant to see past, not as ground truth.

There are two independent rating systems:

- **Teams** — offense, defense, and overall profiles compared against the profiles of every opponent
  faced that season.
- **Quarterbacks** — QB profiles, built only from stats a QB controls, compared against the defenses
  faced that season.

## Stack

- Python 3.14+ (`requires-python = ">=3.14"`).
- [Polars](https://pola.rs) for all dataframes. This project does **not** use pandas.
- [nflreadpy](https://github.com/nflverse/nflreadpy) as the sole data source (nflverse datasets).
- NumPy for the rating and linear-algebra math.
- matplotlib and seaborn for plots.

## Environment and commands

Set up a local virtual environment at `.venv` with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

Run these from the repo root. All four must pass before a task is complete:

```bash
ruff format .        # format
ruff check .         # lint (strict; config in pyproject.toml)
ty check .           # type-check (strict on nfl_sos_ratings/, standard on tests/)
pyright .            # type-check (strict on nfl_sos_ratings/, standard on tests/)
pytest               # tests plus branch coverage (config in pyproject.toml)
```

When a change touches repository-owned Markdown, run `markdownlint` on the Markdown files you
touched as part of the normal validation flow and fix issues as you go. Do not lint vendored or
generated directories such as `.venv/`, `ui/web/node_modules/`, or build outputs.

Run the pipeline and the visualizations:

```bash
python -m nfl_sos_ratings.main        # writes Parquet outputs to data/
python -m nfl_sos_ratings.visualize   # writes plots to data/plots/
```

Dependencies are compiled from `.in` files to pinned `.txt` files via `update_requirements.sh`. Edit
the `.in` files and recompile; never hand-edit the pinned `.txt` files.

## Repository layout

The package is `nfl_sos_ratings/`; tests mirror it under `tests/`. The team pipeline flows through
`data_loader`/`team_stats` → `opponent_stats` → `ratings`; the QB pipeline through
`data_loader`/`qb_stats` → `qb_opponent_stats` → `qb_ratings`. The simultaneous-adjustment path
lives in `simultaneous_adjustment.py`. `main` orchestrates. Read the module you are changing rather
than assuming its shape.

## Active plan documents

The active implementation and handoff plans live in `.agents/`:

- `.agents/current-status.md` — consolidated repo status, recent work summary, active backlog,
  and next-agent guidance.
- `.agents/metric-expansion-plan.md` — the remaining planned-metric ETL work and category-sectioned
  UI follow-up.
- `.agents/frontend-ui-kickoff-plan.md` — the analyst web UI.

Agents working on any of those efforts must:

- Read the relevant plan document before making substantive changes.
- Update it in the same change set whenever progress, scope, decisions, blockers, validation status,
  or next steps change.
- Keep it accurate enough for a new agent to resume work without relying on chat history.

Completed one-off workstreams should be folded into `.agents/current-status.md` or the still-active
workstream plan instead of leaving stale plan files behind.

Treat a stale plan document as a repo bug.

## Domain rules that are easy to get wrong

These are correctness invariants specific to this project. Linters will not catch violations.

- **Regular season only.** Filter to `season_type == "REG"` (or `game_type == "REG"`) on every load.
- **Normalize team abbreviations before joining.** nflverse sources disagree (for example `LA`
  versus `LAR`). Route abbreviations through the existing normalization first, or joins silently
  drop rows.
- **Exclude head-to-head games when profiling an opponent.** An opponent's (or defense's) profile
  must be built from their games against the rest of the league, excluding games against the team or
  QB being evaluated. This is what makes the opponent side independent of the evaluated subject. Do
  not remove or weaken this exclusion.
- **Compare on rates, never on raw totals.** Division opponents play one fewer non-head-to-head game
  than non-division opponents, so raw season totals conflate rate with games played. All comparisons
  and all averaged opponent profiles use per-game and per-play rates.
  For teams that often means per-snap; for QBs it usually means per-dropback, per-attempt, or
  per-carry depending on the subcategory. Keep raw totals only as display columns on a subject's
  own profile.
- **Views are not categories.** `Ratings` is a top-level view, not part of the team/QB stat
  taxonomies.
  The other five views (`Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`, `Opponent Per-Game
  Rates`, `Opponent Per-Play Rates`) reuse the same team categories or QB subcategories.
  `Opponent Context` is expressed through those two opponent views, not as a standalone category.
- **Deduplicate the opponent list; weight each unique opponent equally.** A division rival played
  twice is profiled once (head-to-head exclusion makes the two profiles identical) and counts once
  in the averaged opponent profile.
- **Verify every self-computed metric.** Stats derived from play-by-play (especially defensive
  mirrors of offensive stats) must cite the formula used and be covered by a test that checks the
  computation against a known value or an independent aggregation. Do not ship an unverified metric.
- **Do not assume a column exists.** nflverse schemas differ across seasons and datasets. Check for
  a column before using it and handle its absence, as the existing loaders do.
- **The metric registry is the single source of truth.** Every column the pipeline writes must
  resolve against `nfl_sos_ratings/metrics/` (base metric name, or a registered prefix/suffix of
  one) — `main.py` fails the write otherwise. New metrics get a registry entry first (label,
  layman description, shape, denominator, polarity, source, `ratings_eligible`, `duplicate_of`).
  Rating-pool membership also lives in the registry (`catalog.py`); changing it changes published
  ratings and needs explicit sign-off.
- **The two docs catalogs are companion references, not a second spec.** Keep
  `docs/stats-catalog.md` and `docs/qb-stats-catalog.md` aligned with the metrics registry, but
  treat `nfl_sos_ratings/metrics/` as authoritative when they diverge.
- **Guard source floors in the loader layer.** When an nflverse source starts later than 1999,
  handle that in `data_loader.py` and return a typed empty frame rather than relying on season-loop
  exception handling higher up.

## Code conventions

Style and types are enforced by ruff and pyright with strict settings in `pyproject.toml`. Make code
pass both rather than restating their rules here. Beyond that:

- Follow **test-driven development**: write a failing test first, then implement until it passes.
  Tests use pytest and live alongside the existing ones in `tests/`.
- Prefer **pure functions that take and return Polars frames**. That is the established pattern and
  it keeps pipeline stages independently testable.
- Every module, class, and function needs a docstring (ruff enforces presence; write ones that
  explain intent, not just the signature).
- Keep tunable model constants named and grouped at module top (as in `ratings.py`), not inlined as
  magic numbers.

## Boundaries

- **Always:** run format, lint, type-check, and tests before finishing; add or update tests for the
  code you change; keep coverage above 90% on logic-bearing code.
- **Always:** when you edit Markdown docs or plan files, run `markdownlint` on those repo-owned
  Markdown files before finishing.
- **Dependencies:** adding a new dependency is fine when there is a good reason (no adequate
  stdlib or existing-dependency option, actively maintained, pulls its weight). Add it to the
  appropriate `.in` file, recompile with `update_requirements.sh`, and say in the change what it
  is for. Do not add heavyweight or overlapping dependencies for marginal convenience.
- **Ask first:** before changing the rating methodology or published rating outputs, or
  altering the strict ruff, pyright, or coverage configuration.
- **Never:** weaken lint or type settings to make a check pass; commit secrets or credentials; edit
  files under `data/` as if they were source (they are generated artifacts); hand-edit the pinned
  `requirements*.txt` files.

## When adding new methodology

Parts of this project are actively being extended, including play-by-play-based metrics and a
simultaneous opponent-adjustment system. When a task introduces a pattern the codebase does not yet
have, follow the specification in the task prompt over any existing pattern described here, and
update this file and `README.md` if the change makes either stale.
