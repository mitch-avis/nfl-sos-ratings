import { useMemo, useState } from 'react'

import { useTheme } from '@/app/ThemeProvider'
import { SortableHeader } from '@/components/common/SortableHeader'
import type { OpponentBreakdownTable as Breakdown } from '@/domain/detailAnalytics'
import { compareDetailCellValues, formatDetailCellValue } from '@/domain/detailUi'
import { getMetricMetadata, getMetricTooltip } from '@/domain/metricMetadata'
import { buildColumnDecimals, buildColumnStats, getHeatCellStyle } from '@/domain/tableState'

interface SortState {
  column: string
  desc: boolean
}

const DEFAULT_SORT: SortState = { column: 'opponent_team', desc: false }

/** One row per unique opponent, sortable, with heat shading by column. */
export function OpponentBreakdownTable({ breakdown }: { breakdown: Breakdown }) {
  const { resolved: theme, palette } = useTheme()
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT)
  const activeSort = breakdown.columns.some((column) => column.id === sort.column) ? sort : DEFAULT_SORT

  const rows = useMemo(() => {
    const sorted = [...breakdown.rows]
    sorted.sort((left, right) => {
      const comparison = compareDetailCellValues(
        activeSort.column,
        left[activeSort.column] ?? null,
        right[activeSort.column] ?? null,
      )
      return activeSort.desc ? -comparison : comparison
    })
    return sorted
  }, [activeSort, breakdown.rows])
  const decimals = useMemo(
    () => buildColumnDecimals(rows, breakdown.columns.map((column) => column.id)),
    [breakdown.columns, rows],
  )
  const stats = useMemo(
    () => buildColumnStats(rows, breakdown.columns.map((column) => column.id)),
    [breakdown.columns, rows],
  )

  const nextSort = (columnId: string): SortState => {
    if (activeSort.column === columnId) return { column: columnId, desc: !activeSort.desc }
    if (columnId === 'opp_schedule_bucket') return { column: columnId, desc: true }
    const firstValue = rows[0]?.[columnId]
    return {
      column: columnId,
      desc: typeof firstValue === 'string' ? false : getMetricMetadata(columnId).polarity !== 'lower',
    }
  }

  return (
    <div className="max-h-[70vh] overflow-auto rounded-md border">
      <table className="w-max min-w-full text-sm tabular">
        <thead className="sticky top-0 z-10 bg-muted">
          <tr>
            {breakdown.columns.map((column) => {
              const sorted = activeSort.column === column.id
              return (
                <th
                  key={column.id}
                  scope="col"
                  aria-sort={sorted ? (activeSort.desc ? 'descending' : 'ascending') : undefined}
                  className="h-10 border-b px-2 text-left font-medium whitespace-nowrap text-muted-foreground"
                >
                  <SortableHeader
                    label={column.label ?? getMetricMetadata(column.id).label}
                    hint={column.tooltip ?? getMetricTooltip(column.id)}
                    direction={sorted ? (activeSort.desc ? 'desc' : 'asc') : false}
                    onSort={() => setSort(nextSort(column.id))}
                  />
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.opponent_team ?? 'unknown-opponent')} className="border-b last:border-0">
              {breakdown.columns.map((column) => {
                const value = row[column.id] ?? null
                return (
                  <td
                    key={column.id}
                    className="px-2 py-1.5 whitespace-nowrap"
                    style={getHeatCellStyle(column.id, value, stats, theme, palette)}
                  >
                    {formatDetailCellValue(column.id, value, decimals[column.id] ?? null)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
