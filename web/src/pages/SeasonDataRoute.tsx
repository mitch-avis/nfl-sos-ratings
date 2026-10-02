import { useEffect, type ReactNode } from 'react'
import { useParams } from 'react-router'

import { useMetricRegistry, useSeasonDataset, useSeasons } from '@/api/queries'
import type { EntityKind, SeasonDataset } from '@/api/types'
import { useSeason } from '@/app/useSeason'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { LoadingState } from '@/components/common/LoadingState'
import { getEntityLabel, getEntityRow } from '@/domain/entityConfig'

const BASE_TITLE = 'NFL SOS Ratings'

/** `NFL SOS Ratings - 2025 - Teams - DEN`, matching the route. */
function useDocumentTitle(kind: EntityKind, dataset: SeasonDataset | undefined) {
  const { entityId } = useParams()
  useEffect(() => {
    const segments = [BASE_TITLE]
    if (dataset) segments.push(String(dataset.season))
    segments.push(kind === 'teams' ? 'Teams' : 'QBs')
    const row = dataset && entityId ? getEntityRow(dataset[kind], kind, decodeURIComponent(entityId)) : undefined
    if (row) segments.push(getEntityLabel(kind, row))
    document.title = segments.join(' - ')
  }, [dataset, entityId, kind])
}

/**
 * Loads the metric registry and the selected season, then renders `children` with the dataset.
 * The registry hydrates the labels and tooltips every table reads, so pages wait for it.
 */
export function SeasonDataRoute({
  kind,
  children,
}: {
  kind: EntityKind
  children: (dataset: SeasonDataset) => ReactNode
}) {
  const seasons = useSeasons()
  const registry = useMetricRegistry()
  const { season } = useSeason()
  const dataset = useSeasonDataset(season)
  useDocumentTitle(kind, dataset.data)

  if (seasons.isError) return <ErrorState error={seasons.error} title="Could not load the season list" />
  if (seasons.isSuccess && season === null) {
    return (
      <EmptyState
        title="No seasons with data"
        description="Build outputs with `nfl-sos-ratings pipeline` (or `season`), then reload."
      />
    )
  }
  if (dataset.isError) return <ErrorState error={dataset.error} title={`Could not load ${season ?? 'the'} season`} />
  // A registry failure is tolerable: the season payload carries its own column metadata.
  if (registry.isPending || !dataset.data) return <LoadingState label="Loading season data…" />
  return <>{children(dataset.data)}</>
}
