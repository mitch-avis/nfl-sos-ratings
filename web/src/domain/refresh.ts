/**
 * The refresh button's logic: when to poll the server, when a run has just finished (so every
 * page refetches), and the one-line status the button's panel shows.
 */
import type { RefreshState, RefreshStatus } from '@/api/types'

/** How often the app asks the server about a running refresh. */
export const REFRESH_POLL_MS = 2000
/** How many of the run's last output lines the panel shows after a failure. */
export const FAILED_LOG_LINES = 8

const SUMMARY_PREFIX = 'Summary: '

/** Whether a refresh that was running has just ended, either way; the data may have changed. */
export function refreshFinished(previous: RefreshState | undefined, next: RefreshState | undefined): boolean {
  return previous === 'running' && (next === 'succeeded' || next === 'failed')
}

/** Poll every `REFRESH_POLL_MS` while a refresh runs; otherwise don't poll. */
export function refreshPollInterval(status: RefreshStatus | undefined): number | false {
  return status?.state === 'running' ? REFRESH_POLL_MS : false
}

/** One sentence on the last or running refresh; `clock` turns an ISO time into a local time. */
export function refreshMessage(status: RefreshStatus, clock: (iso: string) => string): string {
  switch (status.state) {
    case 'idle':
      return 'No refresh has run since the server started.'
    case 'running':
      return `Refreshing since ${clock(status.started_at ?? '')}.`
    case 'succeeded': {
      const done = `Refreshed at ${clock(status.finished_at ?? '')}; the data checks passed.`
      if (status.summary === null) return done
      return `${done} Files: ${status.summary.replace(SUMMARY_PREFIX, '')}.`
    }
    case 'failed':
      return (
        `The refresh failed at ${clock(status.finished_at ?? '')} (exit code ${status.exit_code}). ` +
        'The last lines of its output are below.'
      )
  }
}
