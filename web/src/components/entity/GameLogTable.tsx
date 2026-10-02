import type { RowValue } from '@/api/types'
import { MetricLabel } from '@/components/common/MetricLabel'
import { detailHeaderLabel } from '@/domain/detailSections'
import { buildGameOverviewUrl, formatDetailCellValue } from '@/domain/detailUi'

/** One row per game: result context first, then the columns of the current view. */
export function GameLogTable({ rows, columns }: { rows: Array<Record<string, RowValue>>; columns: string[] }) {
  return (
    <div className="max-h-[70vh] overflow-auto rounded-md border">
      <table className="w-max min-w-full text-sm tabular">
        <thead className="sticky top-0 z-10 bg-muted">
          <tr>
            {columns.map((column) => (
              <th key={column} scope="col" className="h-10 border-b px-2 text-left font-medium whitespace-nowrap text-muted-foreground">
                <MetricLabel column={column} label={detailHeaderLabel(column)} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((gameRow, index) => (
            <tr key={`${String(gameRow.game_id ?? index)}-${index}`} className="border-b last:border-0 hover:bg-muted/40">
              {columns.map((column) => {
                const value = gameRow[column] ?? null
                return (
                  <td key={column} className="px-2 py-1.5 whitespace-nowrap">
                    {column === 'game_id' && typeof value === 'string' ? (
                      <a
                        className="text-primary hover:underline"
                        href={buildGameOverviewUrl(value)}
                        rel="noreferrer"
                        target="_blank"
                      >
                        {value}
                      </a>
                    ) : (
                      formatDetailCellValue(column, value)
                    )}
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
