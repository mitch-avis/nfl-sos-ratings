# Project Instructions

## Overview

This repository computes NFL schedule-strength metrics from nflverse data using Polars and produces
season CSV outputs plus visualization plots.

## Environment and Dependencies

- Use Python 3.12.
- Use a local virtual environment at .venv (project-root).
- Install runtime dependencies with: uv pip install -r requirements.txt
- Install test extras with: uv pip install -e .[test]
- Never add dependencies without updating pyproject.toml and (if runtime) requirements.txt.

## Development Commands

- Format: ruff format .
- Lint: ruff check .
- Type check: pyright .
- Test: pytest
- Coverage-focused test run: pytest --cov=nfl_sos_ratings --cov-report=term-missing

Prefer running commands from the repository root.

## Repository Structure

- nfl_sos_ratings/config.py: season and constants (divisions, QB stat column list, output path).
- nfl_sos_ratings/data_loader.py: nflreadpy wrappers; regular-season filtering and base enrichment.
- nfl_sos_ratings/team_stats.py: per-game aggregation helpers and opponent-exclusion helpers.
- nfl_sos_ratings/opponent_stats.py: opponent profile computation, including head-to-head
  exclusions.
- nfl_sos_ratings/ratings.py: schedule-adjusted rating logic (SaOR/SaDR/SaCR).
- nfl_sos_ratings/main.py: end-to-end CSV pipeline orchestration.
- nfl_sos_ratings/visualize.py: plot generation from combined output.
- tests/: pytest suite for main flow, stats logic, ratings, and visualization behavior.
- output/: generated season artifacts and plots.

## Coding Conventions

- Keep functions small and data-transform oriented; prefer pure helpers where possible.
- Prefer Polars expressions/aggregations over Python row loops for data operations.
- Use explicit, descriptive stat prefixes consistently:
  - team columns: raw stat names (example: points_for)
  - opponent columns: opp_prefix (example: opp_points_for)
  - differential columns: diff_prefix (example: diff_points_for)
- Preserve current typing style (modern builtins and union syntax, e.g., list[str], A | None).
- Keep module docstrings and function docstrings concise and factual.
- Maintain 100-character line length unless unavoidable for readability.

## Data and Pipeline Rules

- Restrict analysis data to regular season rows unless a change explicitly targets other season
  types.
- Opponent profiles must exclude head-to-head games against the team being evaluated.
- Keep season-scoped outputs named with the SEASON prefix.
- Treat missing columns defensively (skip unavailable stats instead of failing when feasible).
- Keep output schemas stable unless a schema change is part of the requested task.

## Error Handling and Logging

- Validate required inputs/columns at function boundaries when adding new logic.
- Use clear stdout status messages in pipeline scripts (main.py, visualize.py) for long-running
  steps.
- For non-fatal missing-data cases, prefer warnings and graceful skips.

## Testing Expectations

- Add or update pytest coverage for every behavior change.
- Follow existing test style:
  - small synthetic Polars DataFrames
  - monkeypatch for I/O-heavy paths
  - direct assertions on computed columns and file outputs
- Cover both success and guard/fallback paths.
- Keep tests deterministic and free of network calls.

## Import and Execution Patterns

- Preserve the repository's existing import style within each module.
- Keep scripts directly executable with if __name__ == "__main__": main().
- If refactoring imports, ensure both pytest and script execution still work.

## Security and Safety

- Do not execute untrusted code from downloaded data.
- Avoid writing outside output/ for generated artifacts unless explicitly requested.
- Keep file operations path-safe and explicit.
