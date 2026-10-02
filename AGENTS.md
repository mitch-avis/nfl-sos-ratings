# AGENTS.md

Guidance for AI coding agents working on `nfl-sos-ratings`. Human-facing docs live in
[`README.md`](README.md). Claude Code loads this file through [`CLAUDE.md`](CLAUDE.md).

## Project overview

`nfl-sos-ratings` computes schedule-strength-adjusted NFL ratings for teams and quarterbacks from
nflverse data. It answers "how good was this team or QB, relative to the opponents they actually
faced?", not "what was their record?". Wins and losses are noisy outcomes the ratings are meant to
see past, not ground truth.

There are two independent rating systems:

- **Teams**: offense, defense, special-teams, and overall profiles compared against the profiles
  of every opponent faced that season.
- **Quarterbacks**: QB profiles, built only from stats a QB controls, compared against the
  defenses faced that season.

## Stack

- Python 3.14+, managed with uv (`pyproject.toml` plus `uv.lock`).
- [Polars](https://pola.rs) for all dataframes; this project does **not** use pandas.
- nflverse data only, through [nflreadpy](https://github.com/nflverse/nflreadpy). The one
  exception is ESPN QBR, which nflreadpy lacks: `data_loader.load_espn_qbr` downloads it from the
  nflverse release assets.
- NumPy for the rating and linear-algebra math; FastAPI plus uvicorn for the local analyst API.
- React, TypeScript, Vite, Tailwind, and shadcn/ui for the analyst web app in `web/` (see
  `web/README.md`; `web/AGENTS.md` holds the frontend rules).

## Commands

Agents run Python tools as `.venv/bin/<tool>`, never bare `python`, `pytest`, or `pip`, and not
`uv run`: `uv run` re-syncs the environment before each command and can rewrite `uv.lock` after a
`pyproject.toml` edit. The `uv run` forms in `README.md` are equivalent for humans.

```bash
uv venv .venv && uv sync   # one-time setup
scripts/gate.sh            # the gate: lock/sync checks, ruff format, ruff, ty, pyright, pytest,
                           # markdownlint
scripts/gate.sh --quick    # static checks only, for iteration
scripts/gate.sh --web      # also build web/; use when web/ or the API payloads change
```

- `scripts/gate.sh` defines "checks pass". No task is reported done until it exits 0 on the final
  tree; `--quick` is for iteration, never for the report. Report each step as passed, failed (with
  the key error), or not run (with the reason). Single tools while iterating: `.venv/bin/ruff`,
  `.venv/bin/ty check .`, `.venv/bin/pyright .`, `.venv/bin/pytest`.
- pytest deselects tests marked `published_data`, which read the generated Parquet files in
  `data/`. Run them with `.venv/bin/pytest -m published_data` after a data refresh.
- After any `pyproject.toml` edit, even a comment, run `uv sync`: uv rebuilds the project
  package, and until then the gate's `uv sync --check` step fails.
- If the gate fails on something your change did not touch, check "Validation snapshot" in
  `.agents/current-status.md` for known failures, and say so in the report instead of quietly
  fixing or ignoring it.

Entry points all go through one front door, `nfl-sos-ratings <command>`; `--help` on the front
door or any command prints usage without running anything (commands are listed in
`nfl_sos_ratings/cli.py`, `COMMANDS`):

```bash
.venv/bin/nfl-sos-ratings season [--season N]  # one season (default SEASON in config.py)
.venv/bin/nfl-sos-ratings pipeline             # every season START_YEAR..END_YEAR, rewrites data/
.venv/bin/nfl-sos-ratings validate ...         # regenerates docs/validation-report.md
.venv/bin/nfl-sos-ratings weights              # composite-weight fit report
.venv/bin/nfl-sos-ratings qsos-audit           # QB schedule-strength audit
.venv/bin/nfl-sos-ratings web [--port 8080]    # analyst web app (web/dist) plus its API
```

The full validation run is `validate --data-dir data --start-season 1999 --end-season 2025
--start-week 5 --report-path docs/validation-report.md`. `nfl-sos` and `nfl-sos-pipeline` remain
as shortcuts for `season` and `pipeline`. `web` serves the built app, so run `npm run build` in
`web/` first; the Vite dev server (`npm run dev`) runs on 5280 and proxies `/api` to 8080.

The pipeline and validation commands download from nflverse and can outlive a 10-minute shell
timeout: run them detached (`nohup setsid <cmd> > run.log 2>&1 &`) and only after asking (see
Boundaries).

**Dependencies.** Adding one is fine when there is a good reason (no adequate stdlib or
existing-dependency option, actively maintained, pulls its weight); say in the change what it is
for. Use the targeted commands, which touch only that package: `uv add <pkg>`,
`uv add --group dev <pkg>`, `uv remove <pkg>` (or edit `pyproject.toml`, then `uv lock` and
`uv sync`). `./update_requirements.sh` (from an activated `.venv`) is the upgrade-everything
refresh: it runs `uv lock --upgrade`, so ask before running it and commit it on its own as
`chore: update deps`. Remove unused direct dependencies rather than letting them linger. Never
hand-edit `uv.lock` or any generated `requirements*.txt` export.

## Repository layout

- `nfl_sos_ratings/`: the package; tests mirror it under `tests/`. The team pipeline flows
  `data_loader`/`team_stats` → `opponent_stats` → `ratings`; the QB pipeline `data_loader`/`qb_stats`
  → `qb_opponent_stats` → `qb_ratings`. The ridge solves live in `simultaneous_adjustment.py`;
  `main` runs one season, `pipeline` runs them all, and `cli` is the `nfl-sos-ratings` front
  door. `ui_data` and `ui_api` serve the web app's JSON API and its built files.
- `nfl_sos_ratings/metrics/`: the metric registry; `nfl_sos_ratings/validation/`: walk-forward
  validation, diagnostics, and audits. Read the module you are changing rather than assuming it.
- `docs/`: `methodology.md` (reader-facing), `validation-report.md` (generated), two catalogs.
- `tests/stubs.py`: shared test doubles; `tests/fixtures/`: checked-in fixtures, each with the
  script that rebuilds it.
- `web/`: the analyst frontend; `data/`: generated Parquet outputs (gitignored); `.agents/`:
  plans and handoff (below); `scripts/gate.sh`: the gate.

## Plans, handoff, and scope

The active plans live in `.agents/`. Read the relevant one before substantive changes, and update
it in the same change set whenever progress, scope, decisions, blockers, validation status, or
next steps change. A stale plan document is a repo bug.

- `.agents/current-status.md`: repo status, validation snapshot, active backlog, next-agent
  guidance. Every session that lands work leaves it accurate enough to resume without chat history.
- `.agents/metric-expansion-plan.md`: remaining planned-metric ETL work.
- `.agents/frontend-ui-kickoff-plan.md`: the analyst web UI.

Fold completed one-off workstreams into `current-status.md` or the still-active plan instead of
leaving stale plan files behind. When a task introduces a pattern the codebase does not have yet,
follow the task's specification over the patterns described here, and update this file and
`README.md` if the change makes either stale.

**Narrowing is never a checkbox.** If what landed differs from the task or plan (smaller scope, a
substitute method, a skipped acceptance criterion), record the difference, the reason, and where
the remainder lives in the plan, and tell the user. Never rewrite the task text to fit the delivery.

**Correcting is not narrowing.** A stale path, command or flag name, broken pointer, or typo may be
fixed without asking when the fix is a fact with one plausible answer, changes no behavior,
methodology, or output, and is small. Keep it in the change set that noticed it and list it in
the final report.

## Domain rules that are easy to get wrong

These are correctness invariants specific to this project. Linters will not catch violations.

- **Regular season only.** Filter to `season_type == "REG"` (or `game_type == "REG"`) on every load.
- **Normalize team abbreviations before joining.** nflverse sources disagree (for example `LA`
  versus `LAR`). Route abbreviations through the existing normalization first, or joins silently
  drop rows.
- **Exclude head-to-head games when profiling an opponent.** An opponent's (or defense's) profile
  is built from their games against the rest of the league, excluding games against the team or
  QB being evaluated. This keeps the opponent side independent of the subject. Do not remove or
  weaken this exclusion.
- **Compare on rates, never on raw totals.** Division opponents play one fewer non-head-to-head
  game than non-division opponents, so raw season totals conflate rate with games played. All
  comparisons and averaged opponent profiles use per-game and per-play rates (teams: often
  per-snap; QBs: per-dropback, per-attempt, or per-carry by subcategory). Keep raw totals only as
  display columns on a subject's own profile.
- **Views are not categories.** `Ratings` is a top-level view, not part of the team/QB stat
  taxonomies. The other five views (`Raw Total Stats`, `Per-Game Rates`, `Per-Play Rates`,
  `Opponent Per-Game Rates`, `Opponent Per-Play Rates`) reuse the same team categories or QB
  subcategories. `Opponent Context` is expressed through the two opponent views, not as a category.
- **Deduplicate the opponent list; weight each unique opponent equally.** A division rival played
  twice is profiled once (head-to-head exclusion makes the two profiles identical) and counts once
  in the averaged opponent profile.
- **Verify every self-computed metric.** Stats derived from play-by-play (especially defensive
  mirrors of offensive stats) cite the formula used and are covered by a test that checks the
  computation against a known value or an independent aggregation. Do not ship an unverified
  metric.
- **Do not assume a column exists.** nflverse schemas differ across seasons and datasets. Check
  for a column before using it and handle its absence, as the existing loaders do.
- **The metric registry is the single source of truth.** Every column the pipeline writes resolves
  against `nfl_sos_ratings/metrics/` (base metric name, or a registered prefix/suffix of one);
  `main.py` fails the write otherwise. New metrics get a registry entry first (label, layman
  description, shape, denominator, polarity, source, `ratings_eligible`, `duplicate_of`).
  Rating-pool membership also lives in the registry (`catalog.py`); changing it changes published
  ratings and needs explicit sign-off.
- **The docs catalogs are companions, not a second spec.** Keep `docs/stats-catalog.md` and
  `docs/qb-stats-catalog.md` aligned with the registry; the registry wins when they diverge.
- **Guard source floors in the loader layer.** When an nflverse source starts later than 1999,
  handle that in `data_loader.py` and return a typed empty frame rather than relying on
  season-loop exception handling higher up.
- **Postseason data never feeds published ratings.** `load_playoff_pbp_data()` exists only for
  playoff validation checks.

## Code conventions

Ruff, pyright, and ty enforce style and types with the settings in `pyproject.toml`: ruff selects
`ALL` rules (with the per-file ignores listed there), and pyright runs in strict mode over the
package and the tests. Beyond that:

- **Test-driven development** for new behavior: write a failing test in `tests/` first, then
  implement until it passes. Small targeted fixes need no test-first, but the suite stays green.
- Prefer **pure functions that take and return Polars frames**; that keeps stages testable.
- Every module, class, and function has a docstring that explains intent, not just the signature.
- Keep tunable model constants named and grouped at module top (as in `ratings.py`), not inlined.
- Suppressions are one line, one rule, reason inline (`# noqa: S310 - fixed https URL`); no
  blanket `noqa`, `type: ignore`, or `pragma: no cover`.
- When a module you are changing grows past about 2000 lines, propose a split before adding more.

## Naming and provenance

- Campaign and process vocabulary (stage numbers and letters, block labels, experiment codes, plan
  task numbers) belongs only in `.agents/` files and in the historical sections of
  `docs/validation-report.md` (via `nfl_sos_ratings/validation/history_strings.py`).
- Everywhere else (identifiers, docstrings, comments, registry labels and descriptions, published
  column names, CLI labels, README, catalogs), use timeless names that say what something is or
  does. `tests/test_methodology_language_policy.py` guards part of this.
- Provenance for fitted metrics (fit windows, commands, weight snapshots, refit policy) belongs in
  structured registry metadata fields, not in descriptions or tooltips.

## Numbers, validation, and methodology experiments

- Every number in docs or plans (validation tables, composite weights, inventories, benchmark
  claims) comes from a repo command or checked-in helper, and the command sits beside it.
  Regenerate the source artifact instead of hand-editing numbers; never retype a number from
  memory or from a subagent's summary, which is input to verify, not a source to copy.
- An inventory or audit of code (columns, metrics, call sites) is produced by a script, not
  written by hand. A search or tool call that fails or returns nothing where matches must exist is
  reported as a failure; never fill the gap by inference.
- When frozen weights, published rating definitions, or validation baselines change, update the
  registry metadata, `README.md`, `docs/methodology.md`, and the active `.agents/` docs in the same
  change set.
- Before a comparative methodology experiment, write the falsifiable hypothesis, the allowed
  information set, the decision rule (which metric, which window, what result changes what), and
  the generating command in the relevant `.agents/` plan or validation note.
- Apply the decision rule as written. Report every paired interval that excludes zero, in either
  direction. The incumbent gets no benefit of the doubt: an inconclusive result is a tie, the
  simpler option is the recommendation, and the decision goes to the user as a question.

## Boundaries

- **Always:** run `scripts/gate.sh` before finishing; add or update tests for the code you change.
  pytest enforces a 75% coverage floor (`fail_under` in `pyproject.toml`); the goal for
  logic-bearing code is 90% or higher. Don't lower the total; raise the floor as coverage grows.
- **Ask first, then stop and wait:**
  - changing the rating methodology, published rating definitions or outputs, frozen composite
    weights, or rating-pool membership;
  - regenerating `data/` (`nfl-sos`, `nfl-sos-pipeline`) or rerunning the walk-forward validation,
    composite-weight fit, or audit commands;
  - altering ruff, pyright, ty, or coverage configuration;
  - adding or changing CI, pre-commit hooks, or `.claude/` settings and hooks;
  - pushing, merging to `main`, or anything destructive (force-push, history rewrites, deleting
    branches or files you didn't create);
  - touching a neighboring repo (`../nfl-predictor` or any other);
  - anything the task says to decide with the user.
- **Never:** weaken lint, type, or coverage settings, or skip or delete failing tests, to get a
  pass; commit secrets; edit `data/`, `uv.lock`, `requirements*.txt`, or
  `docs/validation-report.md` by hand (they are generated); write plan labels outside `.agents/`.

**Questions and decisions for the user.** Every question or pending decision carries enough plain,
factual context for the user to recognize what it is about and decide without digging, plus the
agent's recommendation unless the answer is plain from the question itself. The user may not
remember the details of something built long ago, or may not know the area well, so write for a
reader who cannot check it without digging: one line on what the component does today and why the
decision comes up, from the code or docs (name the file), not from memory; the recommended option
first, with its reason and its main cost or risk; how sure the agent is and what would change its
mind. If the evidence is too thin to recommend, say so and name what would settle it. A
recommendation is advice, never consent: the agent still waits for the answer on everything under
"Ask first", and it never bends a written decision rule toward its own preference.

## Git

- Conventional Commits: `type(scope): lowercase imperative summary`, at most 72 characters, no
  trailing period, with a body that says what changed and why. Types: `feat`, `fix`, `docs`,
  `test`, `refactor`, `perf`, `build`, `ci`, `chore`; `!` marks a breaking change. Test-only
  commits are `test`; dependency refreshes are `chore: update deps`.
- One logical change per commit; keep formatting churn out of behavior commits. Branch from
  `main` as `<type>/<kebab-topic>`. Commit only when asked.

## Neighboring repos

`../nfl-predictor` ports this repo's head-to-head-excluded opponent profiling and simultaneous
ridge, reading `README.md`, `AGENTS.md`, and `docs/` here as read-only reference; it never imports
this package. Keep `docs/methodology.md` and `docs/validation-report.md` accurate, since another
project's agents rely on them. Do not edit a neighboring repo from here.
