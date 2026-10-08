import { describe, expect, it } from 'vitest'

import type { RefreshStatus } from '@/api/types'

import { REFRESH_POLL_MS, refreshFinished, refreshMessage, refreshPollInterval } from './refresh'

const IDLE: RefreshStatus = {
  allowed: true,
  state: 'idle',
  started_at: null,
  finished_at: null,
  exit_code: null,
  summary: null,
  log_tail: [],
}

/** A stand-in clock that shows the hours and minutes of an ISO time, independent of time zone. */
const clock = (iso: string) => iso.slice(11, 16)

describe('refreshFinished', () => {
  it.each([
    ['running', 'succeeded', true],
    ['running', 'failed', true],
    ['running', 'running', false],
    ['idle', 'running', false],
    [undefined, 'succeeded', false],
    ['succeeded', 'succeeded', false],
  ] as const)('from %s to %s is %s', (previous, next, expected) => {
    // Act
    const finished = refreshFinished(previous, next)

    // Assert
    expect(finished).toBe(expected)
  })
})

describe('refreshPollInterval', () => {
  it('polls while a refresh runs', () => {
    // Act
    const interval = refreshPollInterval({ ...IDLE, state: 'running' })

    // Assert
    expect(interval).toBe(REFRESH_POLL_MS)
  })

  it.each([undefined, IDLE, { ...IDLE, state: 'succeeded' as const }])('stops otherwise', (status) => {
    // Act
    const interval = refreshPollInterval(status)

    // Assert
    expect(interval).toBe(false)
  })
})

describe('refreshMessage', () => {
  it('says nothing has run yet', () => {
    // Act
    const message = refreshMessage(IDLE, clock)

    // Assert
    expect(message).toBe('No refresh has run since the server started.')
  })

  it('says when a running refresh started', () => {
    // Act
    const message = refreshMessage({ ...IDLE, state: 'running', started_at: '2026-10-08T14:05:09+00:00' }, clock)

    // Assert
    expect(message).toBe('Refreshing since 14:05.')
  })

  it('gives the finish time and what changed', () => {
    // Arrange
    const status: RefreshStatus = {
      ...IDLE,
      state: 'succeeded',
      finished_at: '2026-10-08T14:09:40+00:00',
      exit_code: 0,
      summary: 'Summary: 12 unchanged, 6 values changed, 0 added, 0 removed',
    }

    // Act
    const message = refreshMessage(status, clock)

    // Assert
    expect(message).toBe(
      'Refreshed at 14:09; the data checks passed. Files: 12 unchanged, 6 values changed, 0 added, 0 removed.',
    )
  })

  it('gives the finish time alone when the run printed no summary', () => {
    // Act
    const message = refreshMessage({ ...IDLE, state: 'succeeded', finished_at: '2026-10-08T14:09:40+00:00', exit_code: 0 }, clock)

    // Assert
    expect(message).toBe('Refreshed at 14:09; the data checks passed.')
  })

  it('says a refresh failed, when, and with what exit code', () => {
    // Act
    const message = refreshMessage({ ...IDLE, state: 'failed', finished_at: '2026-10-08T14:07:00+00:00', exit_code: 2 }, clock)

    // Assert
    expect(message).toBe('The refresh failed at 14:07 (exit code 2). The last lines of its output are below.')
  })
})
