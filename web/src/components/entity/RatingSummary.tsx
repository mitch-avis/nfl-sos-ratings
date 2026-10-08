import type { EntityKind, RowValue } from '@/api/types'
import { MetricLabel } from '@/components/common/MetricLabel'
import { getMetricMetadata } from '@/domain/metricMetadata'
import { ordinal } from '@/domain/rankRanges'
import { summaryTiles, type Rank } from '@/domain/ratingSummary'
import { buildColumnDecimals, formatColumnValue } from '@/domain/tableState'
import { cn } from '@/utils/cn'

type Row = Record<string, RowValue>

const METADATA = {
  polarity: (column: string) => getMetricMetadata(column).polarity,
  contextual: (column: string) => Boolean(getMetricMetadata(column).contextual),
}

function rankText(rank: Rank | null): string | null {
  return rank ? `${ordinal(rank.rank)} of ${rank.count}` : null
}

/**
 * The detail page's lead section: each published rating with its rank, headline first, and the
 * headline's value before the schedule adjustment, so the adjustment's size is visible.
 */
export function RatingSummary({ kind, row, rows }: { kind: EntityKind; row: Row; rows: Row[] }) {
  const tiles = summaryTiles(kind, row, rows, METADATA)
  const decimals = buildColumnDecimals(rows, [
    ...tiles.map((tile) => tile.column),
    ...tiles.flatMap((tile) => (tile.unadjusted ? [tile.unadjusted.column] : [])),
  ])
  return (
    <section aria-label="Season Ratings" className="flex flex-col gap-2">
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-7">
        {tiles.map((tile, index) => (
          <div
            key={tile.column}
            className={cn('rounded-md border bg-muted/30 px-3 py-2', index === 0 && 'col-span-2 border-primary/40 bg-primary/5')}
          >
            <dt className="text-xs text-muted-foreground">
              <MetricLabel column={tile.column} />
            </dt>
            <dd className={cn('tabular font-semibold', index === 0 ? 'text-2xl' : 'text-lg')}>
              {formatColumnValue(tile.column, tile.value, decimals[tile.column] ?? null)}
            </dd>
            {rankText(tile.rank) ? <dd className="text-xs text-muted-foreground">{rankText(tile.rank)}</dd> : null}
            {tile.unadjusted ? (
              <dd className="mt-1 text-xs text-muted-foreground">
                Before the schedule adjustment: {getMetricMetadata(tile.unadjusted.column).label}{' '}
                <span className="tabular font-medium text-foreground">
                  {formatColumnValue(tile.unadjusted.column, tile.unadjusted.value, decimals[tile.unadjusted.column] ?? null)}
                </span>
                {rankText(tile.unadjusted.rank) ? ` (${rankText(tile.unadjusted.rank)})` : ''}
              </dd>
            ) : null}
          </div>
        ))}
      </dl>
    </section>
  )
}
