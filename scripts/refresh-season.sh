#!/usr/bin/env bash
# Weekly refresh of the season in progress: rebuild it, check it, and report what changed.
#
# Usage:
#   scripts/refresh-season.sh [--season N] [--dry-run]
#
#   --season N   season to rebuild (default: SEASON in nfl_sos_ratings/config.py)
#   --dry-run    print the steps without running them, and write no log
#   -h, --help   show this help
#
# Steps, from the repo root: copy data/ to a temporary directory, run `nfl-sos-ratings season`,
# run the published_data tests, then `nfl-sos-ratings diff-data` against the copy. Output goes to
# the terminal and to logs/refresh-YYYYMMDD.log (gitignored). Any failing step stops the run with
# a non-zero exit and keeps the copy of data/ for inspection; a clean run removes it.
#
# Needs the project's .venv (uv venv .venv && uv sync) and a built data/.

set -Eeuo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
readonly REPO_ROOT SCRIPT_NAME

season=""
dry_run=false
backup=""
finished=false

usage() {
	sed -n '2,17p' "${REPO_ROOT}/scripts/${SCRIPT_NAME}"
}

die() {
	echo "${SCRIPT_NAME}: $*" >&2
	exit 2
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--season)
			[[ $# -ge 2 ]] || die "--season needs a value"
			[[ "$2" =~ ^[0-9]{4}$ ]] || die "--season must be a four-digit year, got '$2'"
			season="$2"
			shift 2
			;;
		--dry-run)
			dry_run=true
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			die "unknown option: $1 (see --help)"
			;;
		esac
	done
}

# Print one step, then run it unless this is a dry run.
run() {
	printf '+ %s\n' "$*"
	if [[ "${dry_run}" == false ]]; then
		"$@"
	fi
}

# On exit, remove the copy of data/ after a clean run and keep it after a failure.
cleanup() {
	if [[ -z "${backup}" || "${dry_run}" == true ]]; then
		return
	fi
	if [[ "${finished}" == true ]]; then
		rm -rf -- "${backup}"
	else
		echo "${SCRIPT_NAME}: a step failed; the copy of data/ from before the rebuild is kept at ${backup}" >&2
	fi
}

main() {
	parse_args "$@"
	cd -- "${REPO_ROOT}"
	[[ -x .venv/bin/nfl-sos-ratings ]] || die ".venv/bin/nfl-sos-ratings not found; run 'uv venv .venv && uv sync'"
	[[ -d data ]] || die "data/ not found; build it first with 'nfl-sos-ratings pipeline'"

	if [[ "${dry_run}" == false ]]; then
		mkdir -p logs
		local -r log_file="logs/refresh-$(date +%Y%m%d).log"
		exec > >(tee -a "${log_file}") 2> >(tee -a "${log_file}" >&2)
		backup="$(mktemp -d "${TMPDIR:-/tmp}/nfl-sos-data-before.XXXXXX")"
	else
		backup="${TMPDIR:-/tmp}/nfl-sos-data-before.XXXXXX"
	fi
	trap cleanup EXIT

	local season_args=()
	local diff_args=(--before "${backup}" --after data)
	if [[ -n "${season}" ]]; then
		season_args=(--season "${season}")
		diff_args+=(--season "${season}")
	fi

	echo "== Refresh started $(date '+%Y-%m-%d %H:%M:%S')"
	run cp -a data/. "${backup}/"
	run .venv/bin/nfl-sos-ratings season "${season_args[@]}"
	run .venv/bin/pytest -m published_data -q --no-cov
	run .venv/bin/nfl-sos-ratings diff-data "${diff_args[@]}"
	echo "== Refresh finished $(date '+%Y-%m-%d %H:%M:%S')"
	finished=true
}

main "$@"
