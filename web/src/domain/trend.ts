import type { RowValue } from '@/api/types'

export interface TrendPoint {
  week: number
  value: number
  opponent: string
}

/** Week-by-week points for one numeric column, skipping games without a value. */
export function buildTrendPoints(rows: Array<Record<string, RowValue>>, column: string): TrendPoint[] {
  return rows
    .filter((row) => typeof row.week === 'number' && typeof row[column] === 'number')
    .map((row) => ({
      week: row.week as number,
      value: row[column] as number,
      opponent: String(row.opponent_team ?? ''),
    }))
    .sort((left, right) => left.week - right.week)
}
