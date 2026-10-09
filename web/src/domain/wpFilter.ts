import type { EntityKind, RowValue, TablePayload, WpRatingsPayload } from '@/api/types'

import { formatFixed } from './format'

/** The largest threshold the API accepts: plays with a win probability of 20% to 80% remain. */
export const MAX_WP_THRESHOLD = 20
/** The query-string key that carries the threshold, so a filtered view can be shared. */
export const WP_QUERY_KEY = 'wp'

/** The rating, rank, and kept-share columns of one entity's filter payload. */
const WP_COLUMNS: Record<EntityKind, { id: string; label: string; rating: string; rank: string; kept: string }> = {
  teams: { id: 'team', label: 'team', rating: 'team_rating', rank: 'team_rank', kept: 'wp_kept_play_share' },
  qbs: {
    id: 'qb_id',
    label: 'qb_name',
    rating: 'adj_qb_epa_per_dropback',
    rank: 'qb_rank',
    kept: 'wp_kept_dropback_share',
  },
}

/** The published rating, rank, and kept-share column names for one entity's filter payload. */
export function wpColumns(kind: EntityKind): { rating: string; rank: string; kept: string } {
  const { rating, rank, kept } = WP_COLUMNS[kind]
  return { rating, rank, kept }
}

/** One team or QB in the filter view: published values beside the filtered ones. */
export interface WpRatingRow {
  id: string
  label: string
  team: string | null
  publishedRank: number | null
  publishedRating: number | null
  filteredRank: number | null
  filteredRating: number | null
  ratingChange: number | null
  rankChange: number | null
  keptShare: number | null
}

/** Read `?wp=`: a whole percentage from 0 to 20; anything else means no filter. */
export function parseWpThreshold(raw: string | null): number {
  if (raw === null || !/^\d+$/.test(raw)) return 0
  const value = Number(raw)
  return value <= MAX_WP_THRESHOLD ? value : 0
}

/** Say in plain words which plays a threshold keeps. */
export function describeWpThreshold(threshold: number): string {
  if (threshold === 0) return 'Off: every play counts, as in the published ratings.'
  return `Keeps plays that began with the offense's chance of winning between ${threshold}% and ${100 - threshold}%; plays from more lopsided moments are left out.`
}

function asNumber(value: RowValue | undefined): number | null {
  return typeof value === 'number' ? value : null
}

/** Turn a filter payload into typed rows, in the payload's order (filtered rank). */
export function parseWpRatings(kind: EntityKind, payload: WpRatingsPayload): WpRatingRow[] {
  const columns = WP_COLUMNS[kind]
  return payload.rows.map((row) => ({
    id: String(row[columns.id]),
    label: String(row[columns.label] ?? row[columns.id]),
    team: kind === 'qbs' && typeof row.team === 'string' ? row.team : null,
    publishedRank: asNumber(row[columns.rank]),
    publishedRating: asNumber(row[columns.rating]),
    filteredRank: asNumber(row[`filtered_${columns.rank}`]),
    filteredRating: asNumber(row[`filtered_${columns.rating}`]),
    ratingChange: asNumber(row[`filtered_${columns.rating}_change`]),
    rankChange: asNumber(row[`filtered_${columns.rank}_change`]),
    keptShare: asNumber(row[columns.kept]),
  }))
}

/** Describe a rank change: negative means the row moved up the table. */
export function formatRankChange(change: number | null): string {
  if (change === null) return '—'
  if (change === 0) return 'same'
  return change < 0 ? `up ${-change}` : `down ${change}`
}

/** Carry a non-zero threshold into a link that already has a query string. */
export function withWpThreshold(path: string, threshold: number): string {
  if (threshold === 0) return path
  return `${path}${path.includes('?') ? '&' : '?'}${WP_QUERY_KEY}=${threshold}`
}

/** Show a change with an explicit sign (+1.20, -0.50), using a column's fixed decimals. */
export function formatSignedChange(value: number | null, decimals: number | null): string {
  if (value === null) return '—'
  const text = formatFixed(value, decimals)
  return value > 0 && Number(value.toFixed(decimals ?? 2)) !== 0 ? `+${text}` : text
}

/**
 * The main table with the filter's rating and rank beside the published rating: each row gains
 * `filtered_<rating>` and `filtered_<rank>` from `payload` (null for a row the filter view lacks),
 * placed right after the rating column. A view without the rating column, or no payload (the
 * filter off or still loading), leaves the table and columns as they are.
 */
export function withWpColumns(
  kind: EntityKind,
  table: TablePayload,
  selectedColumns: string[],
  payload: WpRatingsPayload | undefined,
): { table: TablePayload; selectedColumns: string[] } {
  const { id, rating, rank } = WP_COLUMNS[kind]
  const position = selectedColumns.indexOf(rating)
  if (payload === undefined || position < 0) return { table, selectedColumns }
  const added = [`filtered_${rating}`, `filtered_${rank}`]
  const filtered = new Map(payload.rows.map((row) => [String(row[id]), row]))
  const rows = table.rows.map((row) => {
    const source = filtered.get(String(row[id]))
    return { ...row, ...Object.fromEntries(added.map((column) => [column, source?.[column] ?? null])) }
  })
  return {
    table: { ...table, rows, visible_columns: [...table.visible_columns, ...added] },
    selectedColumns: [...selectedColumns.slice(0, position + 1), ...added, ...selectedColumns.slice(position + 1)],
  }
}
