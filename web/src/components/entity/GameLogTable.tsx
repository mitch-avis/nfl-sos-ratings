import type { RowValue } from '@/api/types'
import { MetricLabel } from '@/components/common/MetricLabel'
import { RankIntervalTrack } from '@/components/entity/RankInterval'
import { detailHeaderLabel } from '@/domain/detailSections'
import { buildGameOverviewUrl, formatDetailCellValue } from '@/domain/detailUi'
import { middleRankText, type RankRange } from '@/domain/rankRanges'
import { buildColumnDecimals } from '@/domain/tableState'

/** Season-long rank ranges by team, shown beside each opponent, on a track of `count` ranks. */
export interface OpponentRanges {
  ranges: Map<string, Pick<RankRange, 'publishedRank' | 'rank'>>
  count: number
}

function OpponentCell({ team, opponents }: { team: string; opponents?: OpponentRanges }) {
  const range = opponents?.ranges.get(team)
  if (!opponents || !range) return team
  return (
    <span className="inline-flex items-center gap-2">
      {team}
      <span className="text-xs text-muted-foreground">{middleRankText(range)}</span>
      <span className="inline-block w-14">
        <RankIntervalTrack range={range} count={opponents.count} size="mini" />
      </span>
    </span>
  )
}

/**
 * One row per game: result context first, then the columns of the current view. With
 * `opponents`, each opponent shows its season-long middle-50% rank range and a mini interval.
 */
export function GameLogTable({
  rows,
  columns,
  opponents,
}: {
  rows: Array<Record<string, RowValue>>
  columns: string[]
  opponents?: OpponentRanges
}) {
  const decimals = buildColumnDecimals(rows, columns)
  return (
    <div className="max-h-[70vh] overflow-auto rounded-md border">
      <table className="w-max min-w-full text-sm tabular">
        <caption className="sr-only">Game by game</caption>
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
                    {column === 'opponent_team' && typeof value === 'string' ? (
                      <OpponentCell team={value} opponents={opponents} />
                    ) : column === 'game_id' && typeof value === 'string' ? (
                      <a
                        className="text-primary hover:underline"
                        href={buildGameOverviewUrl(value)}
                        rel="noreferrer"
                        target="_blank"
                      >
                        {value}
                      </a>
                    ) : (
                      formatDetailCellValue(column, value, decimals[column] ?? null)
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
