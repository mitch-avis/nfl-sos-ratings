import { ApiError } from '@/api/client'
import type { EntityKind, RowValue, TablePayload } from '@/api/types'

import { formatFixed } from './format'

/** One compared team or QB: how often the subject was rated above it, and by how much. */
export interface RatingPair {
  otherId: string
  aboveChance: number | null
  gapLow: number | null
  gapMid: number | null
  gapHigh: number | null
  share: number | null
}

/** The head-to-head payload's columns for one kind (`nfl_sos_ratings/rating_ranges.py`). */
const PAIR_COLUMNS: Record<EntityKind, { other: string; above: string; gap: string; share: string }> = {
  teams: {
    other: 'other_team',
    above: 'team_rated_above_probability',
    gap: 'team_rating_gap',
    share: 'team_pair_share',
  },
  qbs: { other: 'other_qb_id', above: 'qb_rated_above_probability', gap: 'qb_rating_gap', share: 'qb_pair_share' },
}

/** Decimals and unit of a rating gap: points per game for teams, EPA per dropback for QBs. */
const GAP_FORMAT: Record<EntityKind, { decimals: number; unit: string }> = {
  teams: { decimals: 1, unit: 'points' },
  qbs: { decimals: 3, unit: 'EPA per dropback' },
}

function asNumber(value: RowValue | undefined): number | null {
  return typeof value === 'number' ? value : null
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function signed(value: number | null, decimals: number): string {
  if (value === null) return '—'
  const text = formatFixed(value, decimals)
  return value > 0 && Number(value.toFixed(decimals)) !== 0 ? `+${text}` : text
}

/** Turn one subject's head-to-head payload into typed pairs, in the payload's order. */
export function parseRatingPairs(kind: EntityKind, payload: TablePayload): RatingPair[] {
  const columns = PAIR_COLUMNS[kind]
  return payload.rows.map((row) => ({
    otherId: String(row[columns.other]),
    aboveChance: asNumber(row[columns.above]),
    gapLow: asNumber(row[`${columns.gap}_q025`]),
    gapMid: asNumber(row[`${columns.gap}_q500`]),
    gapHigh: asNumber(row[`${columns.gap}_q975`]),
    share: asNumber(row[columns.share]),
  }))
}

/**
 * Say in one plain sentence how often `subject` was rated above `other` across resampled
 * seasons and the median and 95% range of the rating difference. QB chances count only the
 * resamples with both passers, so the sentence says how many those were when it is not all.
 */
export function describeRatingPair(kind: EntityKind, subject: string, other: string, pair: RatingPair): string {
  if (pair.aboveChance === null) return `${subject} and ${other} never appeared in the same resampled season.`
  const { decimals, unit } = GAP_FORMAT[kind]
  const among = kind === 'qbs' ? 'the resampled seasons with both' : 'resampled seasons'
  const sentence =
    `${subject} rated above ${other} in ${percent(pair.aboveChance)} of ${among}; difference ` +
    `${signed(pair.gapMid, decimals)} ${unit}, 95%: ${signed(pair.gapLow, decimals)} to ${signed(pair.gapHigh, decimals)}.`
  if (pair.share === null || pair.share >= 1) return sentence
  return `${sentence} Both appeared in ${percent(pair.share)} of resampled seasons.`
}

/**
 * The unit to compare with by default: the one ranked just above, or just below the leader.
 * `rankedIds` runs best first; null when the unit is alone or not ranked.
 */
export function neighborId(rankedIds: readonly string[], entityId: string): string | null {
  const index = rankedIds.indexOf(entityId)
  if (index === -1 || rankedIds.length < 2) return null
  return rankedIds[index === 0 ? 1 : index - 1] ?? null
}

/** Whether the head-to-head request failed only because the season was built without pairs. */
export function isMissingRatingPairs(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}
