/**
 * The detail page's rating summary: the published ratings with their ranks, headline first, and
 * the headline's value before the schedule adjustment, so the adjustment's effect is visible.
 */
import type { EntityKind, RowValue } from '@/api/types'

type Row = Record<string, RowValue>
type Polarity = 'higher' | 'lower' | 'neutral'

export interface Rank {
  rank: number
  count: number
}

export interface SummaryTile {
  column: string
  value: RowValue
  rank: Rank | null
  /** The headline rating's counterpart before the schedule adjustment; on the headline tile only. */
  unadjusted: { column: string; value: RowValue; rank: Rank | null } | null
}

/** How the summary reads a column's direction and whether it is context rather than quality. */
export interface SummaryMetadata {
  polarity: (column: string) => Polarity
  contextual: (column: string) => boolean
}

/** The summary's columns per page, headline first, and the headline's unadjusted counterpart. */
const SUMMARY: Record<EntityKind, { columns: string[]; unadjusted: string }> = {
  teams: {
    columns: ['team_rating', 'offense_rating', 'defense_rating', 'special_teams_rating', 'sos', 'SRS'],
    unadjusted: 'epa_margin_per_play',
  },
  qbs: {
    columns: ['adj_qb_epa_per_dropback', 'qb_faced_pass_defense', 'qb_dropbacks_total'],
    unadjusted: 'qb_epa_per_dropback',
  },
}

/**
 * `value`'s rank among `values` (1 = best by `polarity`), with tied values sharing a rank and the
 * places after them skipped; missing values are left out. No rank for a neutral column.
 */
export function rankAmong(values: RowValue[], value: RowValue, polarity: Polarity): Rank | null {
  if (polarity === 'neutral' || typeof value !== 'number' || !Number.isFinite(value)) return null
  const numbers = values.filter((candidate): candidate is number => typeof candidate === 'number' && Number.isFinite(candidate))
  const better = numbers.filter((candidate) => (polarity === 'higher' ? candidate > value : candidate < value)).length
  return { rank: better + 1, count: numbers.length }
}

/**
 * The summary tiles for one team or QB row among the season's `rows`. A QB is ranked among the
 * qualifiers, as on the index, and a QB below the qualifier is not ranked; context columns (such
 * as schedule strength) are never ranked as better or worse.
 */
export function summaryTiles(kind: EntityKind, row: Row, rows: Row[], metadata: SummaryMetadata): SummaryTile[] {
  const config = SUMMARY[kind]
  const ranked = kind === 'qbs' ? rows.filter((candidate) => candidate.qb_is_eligible === true) : rows
  const rankable = kind !== 'qbs' || row.qb_is_eligible === true
  const rankOf = (column: string): Rank | null =>
    rankable && !metadata.contextual(column)
      ? rankAmong(
          ranked.map((candidate) => candidate[column] ?? null),
          row[column] ?? null,
          metadata.polarity(column),
        )
      : null
  return config.columns
    .filter((column) => column in row)
    .map((column) => ({
      column,
      value: row[column] ?? null,
      rank: rankOf(column),
      unadjusted:
        column === config.columns[0] && config.unadjusted in row
          ? { column: config.unadjusted, value: row[config.unadjusted] ?? null, rank: rankOf(config.unadjusted) }
          : null,
    }))
}
