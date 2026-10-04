import { ApiError } from '@/api/client'
import type { EntityKind, RankRangesPayload, RowValue } from '@/api/types'

/**
 * Rank ranges: how much a team's or QB's rank moves when the season's games are resampled.
 *
 * The backend (`rating_ranges.summarize_rank_ranges`) computes every number; this module only reads
 * the payload, orders it, and turns it into text and chart geometry.
 */

export const QUANTILE_KEYS = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975'] as const
export type QuantileKey = (typeof QUANTILE_KEYS)[number]
type Quantiles = Record<QuantileKey, number | null>

/** One team's or QB's rank range, read from one payload row. */
export interface RankRange {
  id: string
  label: string
  team: string | null
  publishedRank: number
  rank: Quantiles
  rating: Quantiles
  top5: number
  top10: number
  missingShare: number
  probabilities: number[]
}

const COLUMNS: Record<EntityKind, { id: string; label: string; rank: string; rating: string }> = {
  teams: { id: 'team', label: 'team', rank: 'team_rank', rating: 'team_rating' },
  qbs: { id: 'qb_id', label: 'qb_name', rank: 'qb_rank', rating: 'adj_qb_epa_per_dropback' },
}

function numberOrNull(value: RowValue | number[] | undefined): number | null {
  return typeof value === 'number' ? value : null
}

function quantiles(row: Record<string, RowValue | number[]>, base: string): Quantiles {
  return Object.fromEntries(QUANTILE_KEYS.map((key) => [key, numberOrNull(row[`${base}_${key}`])])) as Quantiles
}

/** Read every row of a rank-range payload. */
export function parseRankRanges(kind: EntityKind, payload: RankRangesPayload): RankRange[] {
  const columns = COLUMNS[kind]
  return payload.rows.map((row) => {
    const probabilities = row[`${columns.rank}_probabilities`]
    return {
      id: String(row[columns.id] ?? ''),
      label: String(row[columns.label] ?? row[columns.id] ?? ''),
      team: kind === 'qbs' && typeof row.team === 'string' ? row.team : null,
      publishedRank: numberOrNull(row[columns.rank]) ?? 0,
      rank: quantiles(row, columns.rank),
      rating: quantiles(row, columns.rating),
      top5: numberOrNull(row[`${columns.rank}_top5_probability`]) ?? 0,
      top10: numberOrNull(row[`${columns.rank}_top10_probability`]) ?? 0,
      missingShare: numberOrNull(row[`${columns.rank}_missing_share`]) ?? 0,
      probabilities: Array.isArray(probabilities) ? probabilities : [],
    }
  })
}

/** Order ranges for the league chart: median rank first, published rank breaking ties. */
export function rankRangesByMedian(ranges: RankRange[]): RankRange[] {
  const median = (range: RankRange) => range.rank.q500 ?? Number.POSITIVE_INFINITY
  return [...ranges].sort((a, b) => median(a) - median(b) || a.publishedRank - b.publishedRank)
}

/** `1st`, `2nd`, `3rd`, `4th`, ..., `11th`, ..., `21st`. */
export function ordinal(rank: number): string {
  const lastTwo = rank % 100
  if (lastTwo >= 11 && lastTwo <= 13) return `${rank}th`
  const suffix = { 1: 'st', 2: 'nd', 3: 'rd' }[rank % 10] ?? 'th'
  return `${rank}${suffix}`
}

function rankRangeText(low: number | null, high: number | null): string {
  if (low === null || high === null) return 'n/a'
  return low === high ? ordinal(low) : `${ordinal(low)}–${ordinal(high)}`
}

/** The detail-page headline, for example `6th; middle 50%: 4th–8th; 95%: 1st–16th`. */
export function rankRangeHeadline(range: RankRange): string {
  const published = ordinal(range.publishedRank)
  if (range.rank.q500 === null) return `${published}; not ranked in any resample`
  return [
    published,
    `middle 50%: ${rankRangeText(range.rank.q250, range.rank.q750)}`,
    `95%: ${rankRangeText(range.rank.q025, range.rank.q975)}`,
  ].join('; ')
}

/** A one-line text alternative for one interval in the league chart or the table. */
export function rankRangeSummary(range: RankRange): string {
  const median = range.rank.q500 === null ? 'n/a' : ordinal(range.rank.q500)
  return `${range.label}: published rank ${ordinal(range.publishedRank)}, median ${median}; middle 50%: ${rankRangeText(range.rank.q250, range.rank.q750)}; 95%: ${rankRangeText(range.rank.q025, range.rank.q975)}`
}

/** A share of resamples as a percent that never rounds a possible outcome to 0% or 100%. */
export function formatChance(share: number): string {
  if (share > 0 && share < 0.005) return '<1%'
  if (share < 1 && share > 0.995) return '>99%'
  return `${Math.round(share * 100)}%`
}

/** The top-5 and top-10 chances, plus how often a QB had no dropbacks when that happened. */
export function rankChanceText(kind: EntityKind, range: RankRange): string {
  const text = `Top 5 in ${formatChance(range.top5)} of resamples, top 10 in ${formatChance(range.top10)}`
  if (range.missingShare <= 0) return text
  const missing = kind === 'qbs' ? 'no dropbacks' : 'no games'
  return `${text}; ${missing} in ${formatChance(range.missingShare)}`
}

/** Left offset and width, in percent of the track, of ranks `low` through `high` out of `count`. */
export function rankSpan(low: number, high: number, count: number): { left: number; width: number } {
  return { left: ((low - 1) / count) * 100, width: ((high - low + 1) / count) * 100 }
}

/** The center of one rank's cell, in percent of the track. */
export function rankCenter(rank: number, count: number): number {
  return ((rank - 0.5) / count) * 100
}

/**
 * Axis ticks for `count` ranks: 1, then every fifth rank, then the last rank. A step within three
 * ranks of the last is dropped so the labels stay apart on a phone-width track.
 */
export function rankTicks(count: number): number[] {
  const ticks = [1]
  for (let rank = 5; rank < count; rank += 5) {
    if (count - rank >= 4) ticks.push(rank)
  }
  if (count > 1) ticks.push(count)
  return ticks
}

/** The P(rank = k) bars for the detail page, the published rank marked. */
export function rankHistogram(range: RankRange): Array<{ rank: number; probability: number; published: boolean }> {
  return range.probabilities.map((probability, index) => ({
    rank: index + 1,
    probability,
    published: index + 1 === range.publishedRank,
  }))
}

/** Whether a rank-range request failed because the season has no rank-range file (a 404). */
export function isMissingRankRanges(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}
