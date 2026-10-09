import { useWpRatings } from '@/api/queries'
import type { EntityKind } from '@/api/types'
import { useWpThreshold } from '@/app/useWpThreshold'
import { Skeleton } from '@/components/ui/skeleton'
import { columnDecimals, formatFixed } from '@/domain/format'
import { getMetricMetadata } from '@/domain/metricMetadata'
import { formatRankChange, formatSignedChange, parseWpRatings, wpColumns } from '@/domain/wpFilter'

import { WpFilterControl } from './WpFilterControl'

/**
 * A detail page's garbage-time filter: the compact control, and at a non-zero threshold one line
 * with this team's or QB's filtered rating and rank, each beside how far it moved.
 */
export function WpEntityLine({ kind, season, entityId }: { kind: EntityKind; season: number; entityId: string }) {
  const [threshold] = useWpThreshold()
  const query = useWpRatings(kind, season, threshold)
  const rows = threshold > 0 && query.data ? parseWpRatings(kind, query.data) : []
  const row = rows.find((candidate) => candidate.id === entityId)
  const columns = wpColumns(kind)
  const decimals = columnDecimals(
    rows.flatMap((candidate) => [candidate.filteredRating, candidate.publishedRating]),
    getMetricMetadata(`filtered_${columns.rating}`).shape,
  )

  let detail = null
  if (threshold > 0 && query.data === undefined && !query.isError) {
    detail = <Skeleton className="h-5 w-64 bg-muted" />
  } else if (threshold > 0 && query.data !== undefined) {
    detail = row ? (
      <span className="text-muted-foreground">
        At {threshold}%: {getMetricMetadata(`filtered_${columns.rating}`).label}{' '}
        <span className="tabular font-medium text-foreground">{formatFixed(row.filteredRating, decimals)}</span> (
        {formatSignedChange(row.ratingChange, decimals)}), rank{' '}
        <span className="tabular font-medium text-foreground">{row.filteredRank ?? '—'}</span> (
        {formatRankChange(row.rankChange)})
      </span>
    ) : (
      <span className="text-muted-foreground">
        {kind === 'qbs'
          ? 'Below the qualifier, so the filter does not rank him.'
          : 'No filtered rating for this team this season.'}
      </span>
    )
  }

  return (
    <section aria-label="Garbage-time filter" className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
      <WpFilterControl kind={kind} season={season} where="here, beside how far each moved" />
      {detail}
    </section>
  )
}
