import { ApiError } from '@/api/client'
import type { EntityKind, RowValue, TablePayload } from '@/api/types'

import type { TrendReference } from './trend'

/** What the "Rating by week" chart draws for one team or QB. */
export interface RatingHistoryChart {
  rows: Array<Record<string, RowValue>>
  columns: string[]
  reference: TrendReference | null
}

/**
 * The rating-history chart for one team or QB, or null when fewer than two weeks are rated.
 *
 * Team ratings are points per game against an average team, so 0 is drawn as the reference. The
 * QB rating's league average moves week to week and is not in the payload, so QBs get no line.
 */
export function buildRatingHistoryChart(kind: EntityKind, payload: TablePayload): RatingHistoryChart | null {
  const columns = payload.column_groups.ratings ?? []
  const weeks = new Set(payload.rows.map((row) => row.week).filter((week) => typeof week === 'number'))
  if (columns.length === 0 || weeks.size < 2) return null
  return {
    rows: payload.rows,
    columns,
    reference: kind === 'teams' ? { value: 0, caption: 'Dashed line: an average team (0).' } : null,
  }
}

/** Whether a rating-history request failed because the season has no history file (a 404). */
export function isMissingRatingHistory(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}
