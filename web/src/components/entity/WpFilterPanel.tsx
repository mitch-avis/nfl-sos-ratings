import { useEffect, useId, useState } from 'react'
import { Link } from 'react-router'

import { useWpRatings } from '@/api/queries'
import type { EntityKind, WpRatingsPayload } from '@/api/types'
import { useWpThreshold } from '@/app/useWpThreshold'
import { ErrorState } from '@/components/common/ErrorState'
import { SortableHeader } from '@/components/common/SortableHeader'
import { StatTile } from '@/components/common/StatTile'
import { TeamChip } from '@/components/common/TeamChip'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Slider } from '@/components/ui/slider'
import { columnDecimals, formatFixed } from '@/domain/format'
import { getMetricMetadata, getMetricTooltip } from '@/domain/metricMetadata'
import {
  describeWpThreshold,
  formatRankChange,
  formatSignedChange,
  MAX_WP_THRESHOLD,
  parseWpRatings,
  sortWpRows,
  withWpThreshold,
  wpColumns,
  type WpRatingRow,
  type WpSortKey,
} from '@/domain/wpFilter'
import { useDebouncedValue } from '@/hooks/use-debounced-value'

// Wait this long after the slider stops before asking the API for a new threshold.
const WP_DEBOUNCE_MS = 250

function formatShare(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

/** Decimals for the rating and its change, from the filtered rating's registry shape. */
function ratingDecimals(kind: EntityKind, rows: WpRatingRow[]): number | null {
  const shape = getMetricMetadata(`filtered_${wpColumns(kind).rating}`).shape
  return columnDecimals(
    rows.flatMap((row) => [row.filteredRating, row.publishedRating]),
    shape,
  )
}

function ExplorationNote({ kind }: { kind: EntityKind }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Badge variant="outline" className="w-fit border-amber-500/60 text-amber-700 dark:text-amber-300">
        Unvalidated exploration view
      </Badge>
      <p className="max-w-prose text-sm text-muted-foreground">
        The published ratings use every play. Rank ranges and the rest of this page count every play
        too. In a walk-forward test, no threshold predicted team game margins better than every play,
        and 20% predicted them worse.
        {kind === 'qbs'
          ? ' Filtered QB ratings use play-by-play EPA, which differs slightly from the official EPA in the published rating, so changes compare with the same calculation at 0%.'
          : null}
      </p>
    </div>
  )
}

function WpRatingsTable({
  kind,
  season,
  threshold,
  rows,
}: {
  kind: EntityKind
  season: number
  threshold: number
  rows: WpRatingRow[]
}) {
  const [sort, setSort] = useState<{ key: WpSortKey; descending: boolean }>({ key: 'filteredRank', descending: false })
  const columns = wpColumns(kind)
  const decimals = ratingDecimals(kind, rows)
  const sorted = sortWpRows(rows, sort)
  const header = (key: WpSortKey, column: string, descendingFirst: boolean, label?: string) => (
    <SortableHeader
      label={label ?? getMetricMetadata(column).label}
      hint={getMetricTooltip(column)}
      direction={sort.key === key ? (sort.descending ? 'desc' : 'asc') : false}
      onSort={() =>
        setSort((current) =>
          current.key === key ? { key, descending: !current.descending } : { key, descending: descendingFirst },
        )
      }
    />
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] text-sm">
        <caption className="sr-only">
          {kind === 'teams' ? 'Teams' : 'Qualifying quarterbacks'} filtered at {threshold}%
        </caption>
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="py-2 pr-3 font-medium">{header('filteredRank', `filtered_${columns.rank}`, false)}</th>
            <th className="py-2 pr-3 font-medium">{kind === 'teams' ? 'Team' : 'Quarterback'}</th>
            {kind === 'qbs' ? <th className="py-2 pr-3 font-medium">Team</th> : null}
            <th className="py-2 pr-3 text-right font-medium">{getMetricMetadata(`filtered_${columns.rating}`).label}</th>
            <th className="py-2 pr-3 text-right font-medium">
              {header('ratingChange', `filtered_${columns.rating}_change`, true)}
            </th>
            <th className="py-2 pr-3 font-medium">{header('rankChange', `filtered_${columns.rank}_change`, false)}</th>
            <th className="py-2 pr-3 font-medium">
              {/* The cell shows the published rank and rating together, so it is labeled for both. */}
              {header('publishedRank', columns.rank, false, 'Published')}
            </th>
            <th className="py-2 text-right font-medium">{header('keptShare', columns.kept, false)}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.id} className="border-b last:border-0">
              <td className="py-1.5 pr-3 tabular font-semibold">{row.filteredRank ?? '—'}</td>
              <td className="py-1.5 pr-3">
                <Link
                  className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
                  to={withWpThreshold(`/${kind}/${encodeURIComponent(row.id)}?season=${season}`, threshold)}
                >
                  {kind === 'teams' ? <TeamChip team={row.id} /> : null}
                  {row.label}
                </Link>
              </td>
              {kind === 'qbs' ? (
                <td className="py-1.5 pr-3 text-muted-foreground">
                  <span className="inline-flex items-center gap-1.5">
                    {row.team ? <TeamChip team={row.team} /> : null}
                    {row.team ?? '—'}
                  </span>
                </td>
              ) : null}
              <td className="py-1.5 pr-3 text-right tabular">{formatFixed(row.filteredRating, decimals)}</td>
              <td className="py-1.5 pr-3 text-right tabular">{formatSignedChange(row.ratingChange, decimals)}</td>
              <td className="py-1.5 pr-3 text-muted-foreground">{formatRankChange(row.rankChange)}</td>
              <td className="py-1.5 pr-3 tabular text-muted-foreground">
                {row.publishedRank ?? '—'} · {formatFixed(row.publishedRating, decimals)}
              </td>
              <td className="py-1.5 text-right tabular text-muted-foreground">{formatShare(row.keptShare)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function WpEntitySummary({ kind, row, rows }: { kind: EntityKind; row: WpRatingRow; rows: WpRatingRow[] }) {
  const decimals = ratingDecimals(kind, rows)
  const columns = wpColumns(kind)
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatTile
        label={getMetricMetadata(`filtered_${columns.rank}`).label}
        value={row.filteredRank ?? '—'}
        footnote={formatRankChange(row.rankChange)}
      />
      <StatTile
        label={getMetricMetadata(`filtered_${columns.rating}`).label}
        value={formatFixed(row.filteredRating, decimals)}
        footnote={`${formatSignedChange(row.ratingChange, decimals)} from no filter`}
      />
      <StatTile
        label="Published"
        value={`${row.publishedRank ?? '—'} · ${formatFixed(row.publishedRating, decimals)}`}
        footnote="Rank and rating, every play"
      />
      <StatTile label={getMetricMetadata(columns.kept).label} value={formatShare(row.keptShare)} />
    </div>
  )
}

function FilteredView({
  kind,
  season,
  threshold,
  entityId,
  payload,
}: {
  kind: EntityKind
  season: number
  threshold: number
  entityId?: string
  payload: WpRatingsPayload
}) {
  const rows = parseWpRatings(kind, payload)
  if (entityId === undefined) return <WpRatingsTable kind={kind} season={season} threshold={threshold} rows={rows} />
  const row = rows.find((candidate) => candidate.id === entityId)
  if (!row) {
    return (
      <p className="text-sm text-muted-foreground">
        {kind === 'qbs'
          ? 'This quarterback is below the qualifier, so the filter view does not rank him.'
          : 'This team has no filtered rating for this season.'}
      </p>
    )
  }
  return <WpEntitySummary kind={kind} row={row} rows={rows} />
}

/**
 * The garbage-time filter: a 0-20% slider kept in `?wp=`, and at a non-zero threshold an
 * exploration view of the ratings refit without plays in lopsided game states. On an index page
 * it lists every team or qualifying QB; with `entityId` it shows that one row.
 */
export function WpFilterPanel({ kind, season, entityId }: { kind: EntityKind; season: number; entityId?: string }) {
  const [threshold, setThreshold] = useWpThreshold()
  const [draft, setDraft] = useState(threshold)
  const [synced, setSynced] = useState(threshold)
  // Follow a threshold that changes from outside (navigation, a shared link).
  if (threshold !== synced) {
    setSynced(threshold)
    setDraft(threshold)
  }
  const settled = useDebouncedValue(draft, WP_DEBOUNCE_MS)
  useEffect(() => {
    // Commit only a value the user has stopped moving, never a stale one after a navigation.
    if (settled === draft && settled !== threshold) setThreshold(settled)
  }, [draft, settled, setThreshold, threshold])
  const query = useWpRatings(kind, season, threshold)
  const labelId = useId()

  return (
    <Card role="region" aria-labelledby={labelId} className="gap-3 px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div id={labelId} className="font-semibold">
          Garbage-time filter
        </div>
        <div className="text-sm font-medium tabular" aria-hidden>
          {draft === 0 ? 'Off' : `${draft}%`}
        </div>
      </div>
      <Slider
        aria-label="Garbage-time filter"
        className="py-3"
        min={0}
        max={MAX_WP_THRESHOLD}
        step={1}
        value={[draft]}
        onValueChange={([value]) => setDraft(value ?? 0)}
      />
      <div className="-mt-2 flex justify-between text-xs text-muted-foreground">
        <span>Off</span>
        <span>{MAX_WP_THRESHOLD}%</span>
      </div>
      <p className="text-sm">{describeWpThreshold(draft)}</p>
      {threshold === 0 ? (
        <p className="max-w-prose text-sm text-muted-foreground">
          Move the slider to leave out plays from lopsided game states and see how much the{' '}
          {kind === 'teams' ? 'team' : 'quarterback'} ratings depend on them.
        </p>
      ) : (
        <>
          <ExplorationNote kind={kind} />
          {query.isError ? (
            <ErrorState error={query.error} title="Could not load the filtered ratings" />
          ) : query.data ? (
            <FilteredView
              kind={kind}
              season={season}
              threshold={query.data.threshold}
              entityId={entityId}
              payload={query.data}
            />
          ) : (
            <Skeleton className="h-32 w-full" />
          )}
        </>
      )}
    </Card>
  )
}
