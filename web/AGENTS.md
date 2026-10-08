# AGENTS.md (web/)

Rules for agents working in `web/`, the analyst web app. The repository-root
[`AGENTS.md`](../AGENTS.md) still applies; this file adds what is specific to the frontend.
Human-facing docs live in [`README.md`](README.md).

## Commands

Run from `web/` with pnpm 12 (CI pins the version in `.github/workflows/validation.yml`). If `node`
is not on `PATH`, `source ~/.nvm/nvm.sh` first.

```bash
pnpm install --frozen-lockfile   # install exactly what pnpm-lock.yaml pins
pnpm run lint                    # oxlint
pnpm run typecheck               # tsc -b
pnpm exec vitest run             # tests, once (pnpm test starts watch mode)
pnpm run build                   # tsc -b plus the Vite production build into dist/
```

All four checks pass before work in `web/` is reported done, plus `scripts/gate.sh --web` from the
repository root. The one known oxlint warning about TanStack `useReactTable` is inherent to the
library; do not add new warnings.

To look at the app, build it and run `.venv/bin/nfl-sos-ratings web --port 8090` from the root
(a spare port avoids a server the user may have running on 8080).

## Rules

- **The backend defines meaning.** The app reads the JSON from `nfl_sos_ratings/ui_api.py`; it
  never recomputes ratings or redefines a column. Labels, descriptions, and polarity come from
  `/api/metadata` (the metric registry). If a payload is wrong, fix the Python side and its tests.
- **Pure logic lives in `src/domain/`** with a `*.test.ts` beside it. Components stay thin.
- **TDD applies here too:** new domain logic or behavior gets a failing Vitest test first.
- **shadcn/ui primitives in `src/components/ui/`** are generated code; prefer composing them over
  editing them, and keep any edit minimal.
- **Dependencies:** `pnpm add <pkg>` (or `pnpm add -D <pkg>`) updates `pnpm-lock.yaml`; never
  hand-edit the lockfile. Say in the change what a new package is for.
- **Ports:** the API and built app on 8080, the Vite dev server on 5280 (`vite.config.ts`), so
  the app can run beside nfl-predictor (8000 and 5173). Don't change them without asking.
- `dist/` and `node_modules/` are build output; never commit them.
