#!/usr/bin/env bash
# The one validation gate. A task is not done until this script exits 0 on the final tree.
# It uses the repo's own .venv (never `uv run`, which re-syncs the environment first) and runs
# every selected step before exiting, so one run reports everything that is wrong rather than
# only the first failure.
#
# Usage:
#   scripts/gate.sh          lock and sync checks, ruff format, ruff, ty, pyright, pytest,
#                            CLI --help smoke check, markdownlint
#   scripts/gate.sh --quick  skip pytest and the CLI smoke check (static checks only; for
#                            iteration, never the report)
#   scripts/gate.sh --web    also check the frontend in web/ (pnpm install, lint, typecheck,
#                            vitest, build)
#
# Exit status is 0 only when every selected step passed.

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 2

RUN_TESTS=1
RUN_WEB=0
for arg in "$@"; do
	case "$arg" in
	--quick) RUN_TESTS=0 ;;
	--web) RUN_WEB=1 ;;
	-h | --help)
		sed -n '2,15p' "$0"
		exit 0
		;;
	*)
		echo "gate: unknown option: $arg" >&2
		exit 2
		;;
	esac
done

if [[ ! -x .venv/bin/python ]]; then
	echo "gate: .venv/bin/python not found; create it with 'uv venv .venv && uv sync'" >&2
	exit 2
fi

declare -a NAMES=()
declare -a STATUSES=()
FAILED=0

run_step() {
	# run_step <name> <command...>
	local name="$1"
	shift
	echo
	echo "==> ${name}"
	NAMES+=("$name")
	if "$@"; then
		STATUSES+=("ok")
	else
		STATUSES+=("FAIL")
		FAILED=1
	fi
}

markdownlint_step() {
	# Lint every tracked or untracked-but-not-ignored Markdown file, so .gitignore keeps .venv/,
	# data/, and web/node_modules/ out. markdownlint-cli2 picks up .markdownlint.json itself.
	local -a files
	if ! command -v markdownlint-cli2 >/dev/null 2>&1; then
		echo "markdownlint-cli2 is not installed" >&2
		return 1
	fi
	mapfile -t files < <(git ls-files --cached --others --exclude-standard -- '*.md')
	if [[ ${#files[@]} -eq 0 ]]; then
		return 0
	fi
	markdownlint-cli2 "${files[@]}"
}

web_step() {
	if ! command -v node >/dev/null 2>&1 && [[ -s "$HOME/.nvm/nvm.sh" ]]; then
		# shellcheck disable=SC1091  # nvm lives outside the repo
		. "$HOME/.nvm/nvm.sh"
	fi
	local pnpm_home="${PNPM_HOME:-$HOME/.local/share/pnpm}"
	if ! command -v pnpm >/dev/null 2>&1 && [[ -x "$pnpm_home/pnpm" ]]; then
		PATH="$pnpm_home:$PATH"
	fi
	(
		cd web &&
			pnpm install --frozen-lockfile &&
			pnpm run lint &&
			pnpm run typecheck &&
			pnpm exec vitest run &&
			pnpm run build
	)
}

cli_help_step() {
	# Every front-door command must print its usage and exit 0 without running anything. The
	# command names come from nfl_sos_ratings.cli.COMMANDS, so a new command is covered here too.
	local -a commands
	mapfile -t commands < <(
		.venv/bin/python -c \
			'from nfl_sos_ratings.cli import COMMANDS; print("\n".join(c.name for c in COMMANDS))'
	) || return 1
	if [[ ${#commands[@]} -eq 0 ]]; then
		echo "no commands found in nfl_sos_ratings.cli.COMMANDS" >&2
		return 1
	fi
	.venv/bin/nfl-sos-ratings --help >/dev/null || return 1
	local command
	for command in "${commands[@]}"; do
		if ! .venv/bin/nfl-sos-ratings "$command" --help >/dev/null; then
			echo "nfl-sos-ratings ${command} --help failed" >&2
			return 1
		fi
	done
}

run_step "uv lock --check" uv lock --check
run_step "uv sync --check" uv sync --check
run_step "ruff format --check" .venv/bin/ruff format --check .
run_step "ruff check" .venv/bin/ruff check .
run_step "ty check" .venv/bin/ty check .
run_step "pyright" .venv/bin/pyright .
if [[ "$RUN_TESTS" -eq 1 ]]; then
	run_step "pytest" .venv/bin/pytest -q -p no:sugar
	run_step "cli --help" cli_help_step
fi
run_step "markdownlint" markdownlint_step
if [[ "$RUN_WEB" -eq 1 ]]; then
	run_step "web" web_step
fi

echo
echo "==> gate summary"
for i in "${!NAMES[@]}"; do
	printf '  %-24s %s\n' "${NAMES[$i]}" "${STATUSES[$i]}"
done
if [[ "$FAILED" -ne 0 ]]; then
	echo "gate: FAILED"
	exit 1
fi
echo "gate: all steps passed"
