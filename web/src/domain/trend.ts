import type { RowValue } from '@/api/types'

import { formatValue } from './format'

export interface TrendPoint {
  week: number
  value: number
  opponent: string
}

/** A dashed horizontal line on a trend chart, with the caption that explains it. */
export interface TrendReference {
  value: number
  caption: string
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

/** A reference line at the mean of the games shown, or none when there are no points. */
export function meanReference(points: TrendPoint[]): TrendReference | null {
  if (points.length === 0) return null
  const mean = points.reduce((total, point) => total + point.value, 0) / points.length
  return { value: mean, caption: `Dashed line: the mean of the games shown (${formatValue(mean)}).` }
}
