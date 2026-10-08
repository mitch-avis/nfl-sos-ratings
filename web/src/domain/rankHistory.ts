import { ApiError } from '@/api/client'
import type { EntityKind, RowValue, TablePayload } from '@/api/types'

import { ordinal } from './rankRanges'

/**
 * Rank ranges by week for a season in progress: the rank when the games through each week are
 * redrawn at random and refit. The backend (`main.build_team_rank_ranges_by_week`) computes every
 * number; this module reads the payload and shapes it for the band chart and its text alternative.
 */

/** One week's median rank and its 50% and 95% bands, rank 1 the best. */
export interface RankHistoryPoint {
  week: number
  published: number | null
  median: number | null
  low50: number | null
  high50: number | null
  low95: number | null
  high95: number | null
}

const RANK_COLUMN: Record<EntityKind, string> = { teams: 'team_rank', qbs: 'qb_rank' }

function numberOrNull(value: RowValue | undefined): number | null {
  return typeof value === 'number' ? value : null
}

/** Read every week of one team's or QB's rank history, in week order. */
export function parseRankHistory(kind: EntityKind, payload: TablePayload): RankHistoryPoint[] {
  const rank = RANK_COLUMN[kind]
  return payload.rows
    .map((row) => ({
      week: numberOrNull(row.week) ?? 0,
      published: numberOrNull(row[rank]),
      median: numberOrNull(row[`${rank}_q500`]),
      low50: numberOrNull(row[`${rank}_q250`]),
      high50: numberOrNull(row[`${rank}_q750`]),
      low95: numberOrNull(row[`${rank}_q025`]),
      high95: numberOrNull(row[`${rank}_q975`]),
    }))
    .sort((left, right) => left.week - right.week)
}

function band(low: number | null, high: number | null): [number, number] | null {
  return low === null || high === null ? null : [low, high]
}

/** The rows the band chart draws: the median line and the two range areas as `[low, high]`. */
export function rankBandData(
  points: readonly RankHistoryPoint[],
): Array<{ week: number; median: number | null; band50: [number, number] | null; band95: [number, number] | null }> {
  return points.map((point) => ({
    week: point.week,
    median: point.median,
    band50: band(point.low50, point.high50),
    band95: band(point.low95, point.high95),
  }))
}

function weekText(point: RankHistoryPoint): string {
  const median = point.median === null ? 'n/a' : ordinal(point.median)
  const range =
    point.low95 === null || point.high95 === null ? 'n/a' : `${ordinal(point.low95)}–${ordinal(point.high95)}`
  return `${median} after week ${point.week} (95%: ${range})`
}

/** A one-sentence text alternative: the first and the latest week's median rank and 95% range. */
export function describeRankHistory(points: readonly RankHistoryPoint[]): string {
  const first = points[0]
  const last = points.at(-1)
  if (!first || !last) return ''
  if (first === last) return `Median rank ${weekText(first)}.`
  return `Median rank ${weekText(first)} and ${weekText(last)}.`
}

/** The bottom of the rank axis: the worst rank any week's 95% band reaches, at least 1. */
export function rankAxisMax(points: readonly RankHistoryPoint[]): number {
  return Math.max(1, ...points.map((point) => point.high95 ?? point.median ?? 1))
}

/** Whether the request failed only because the season has no weekly rank ranges (a 404). */
export function isMissingRankHistory(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}
