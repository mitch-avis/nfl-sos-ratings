# CLAUDE.md

@AGENTS.md

## Claude Code

- `.claude/settings.json` blocks hand edits to generated files (`data/`, `uv.lock`,
  `requirements*.txt`, `docs/validation-report.md`, `ui/web/package-lock.json`) and prompts before
  `git push`, `./update_requirements.sh`, and the commands that rewrite `data/` or the validation
  report. Regenerate a blocked file with the command that owns it; never route around a block.
- Bash calls time out at 10 minutes. `scripts/gate.sh` fits in the foreground; launch the
  pipeline and validation commands with `nohup setsid` (after asking), never with a promise to
  "keep monitoring" and nothing actually watching.
- There is no `.pre-commit-config.yaml`, so no hook runs the gate for you: run `scripts/gate.sh`
  yourself before reporting done.
- Skills carry general defaults. Where one disagrees with `AGENTS.md` (for example `uv run` versus
  `.venv/bin/<tool>`, or which checks form the gate), `AGENTS.md` wins.
- Run `/doctor prompt-audit` after large edits to `AGENTS.md` or this file to catch stale or
  conflicting instructions.
