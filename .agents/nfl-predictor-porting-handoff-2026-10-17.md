# nfl-predictor Porting Handoff (2026-10-17)

Purpose: summarize the `nfl-sos-ratings` follow-ups worth checking against the ongoing
`nfl-predictor` port now that this repo is no longer in its temporary "locked" state.

## Scope

This note covers the post-lock follow-ups on this branch plus the recent repo state that another
agent would otherwise need to reconstruct from `.agents/current-status.md`, `.agents/roadmap.md`,
and the latest config/test changes.

## What Changed Here

### Refresh Path

- `nfl_sos_ratings/refresh_runner.py` now closes `process.stdout` inside `_follow()` and records
  the process result from a `finally` block. Under Python 3.14, the previous code leaked the pipe
  wrapper long enough to trip `ResourceWarning` / `PytestUnraisableExceptionWarning` in the
  refresh-runner tests.
- `scripts/refresh-season.sh` did **not** change scope. It still runs `nfl-sos-ratings season`,
  then `.venv/bin/pytest -m published_data -q`, then `nfl-sos-ratings diff-data`. Do not widen it
  to the full gate in the port unless there is a separate product decision.

### published_data Pytest Behavior

- The repo keeps generated-data checks opt-in on the default suite. `pyproject.toml` now again
  carries `-m not published_data` in pytest `addopts`.
- `tests/conftest.py` already lifted the coverage floor when every selected test carried the
  `published_data` marker. A 2026-10-17 follow-up added a `pytest_configure` fast path for the
  exact marker expression `published_data`, because the refresh script hits pytest through the
  default `-n=auto` xdist path and needed the coverage decision before collection finished.
- Net effect here: the default `pytest` run still keeps generated-data tests out of CI and other
  non-data environments, while `.venv/bin/pytest -m published_data -q` works directly for refresh
  validation without `--no-cov`.

### Pyright and Tooling

- `pyproject.toml` now defines separate pyright execution environments for `.` and `tests`, with
  `reportMissingTypeStubs = false` in both. That keeps `nflreadpy` imports from failing only in the
  test environment.
- The branch also carries the current tooling floor: `uv >=0.13`, `uv_build >=0.13,<0.14`,
  `polars >=2.0`, `fastapi >=0.143`, `ruff >=0.17`, and `ty >=0.0.89`, with `uv.lock` refreshed.
- Pytest defaults now run in parallel (`-n=auto`) and treat warnings as errors.

### Test Fallout

- Most changed test files outside the refresh path are import-order cleanups or small expectation
  adjustments needed by the stricter tooling refresh.
- `tests/test_pipeline.py` also hardens logged-exception collection for typing safety and carries a
  small `Record` round-trip test.
- `tests/test_project_metadata.py` now pins both the pyright missing-stub policy and the default
  pytest marker deselection.

## Broader Repo State Since The Earlier Frozen Snapshot

- UI and UX work already landed here: team palettes, the palette grid, glossary/tooltips/copy
  cleanup, and the refresh button backed by `refresh_runner.py`.
- Data/rating work already landed here: the preseason prior at 9 games, the scramble/QB-role/spike
  fixes, and the Polars 2.0 parity validation.
- The active open items here remain Q1 (QB rating from all QB plays), F1/F2 (frontend follow-ups),
  and M1 (retired stats), as recorded in `.agents/current-status.md` and `.agents/roadmap.md`.

## Porting Advice For nfl-predictor

1. If `nfl-predictor` has the same refresh button or background runner pattern, port the stdout
   closing and `finally` cleanup from `nfl_sos_ratings/refresh_runner.py`.
2. If it keeps generated-data tests behind a marker while pytest runs with xdist by default, port
   both the configure-time and collection-time coverage-floor lifts from `tests/conftest.py`.
3. If it uses `nflreadpy` and split pyright execution environments, mirror the missing-stub policy
   in each environment rather than relying on a single global pyright setting.
4. Keep the refresh script contract narrow unless the maintainer explicitly wants the port's
   refresh action to run the full gate.

## Relevant Files In This Repo

- `nfl_sos_ratings/refresh_runner.py`
- `scripts/refresh-season.sh`
- `tests/conftest.py`
- `tests/test_conftest.py`
- `pyproject.toml`
- `tests/test_project_metadata.py`
- `.agents/current-status.md`
- `.agents/roadmap.md`

## Useful Validation Commands

```bash
./scripts/gate.sh
.venv/bin/pytest -m published_data -q
./scripts/refresh-season.sh --dry-run
```

For a real refresh run in this repo, keep the existing ask-first rule because the script rewrites
`data/`.
