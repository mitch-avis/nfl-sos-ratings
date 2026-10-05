import { useQuery } from '@tanstack/react-query'

import { hydrateColumnMetadata, hydrateMetricRegistry } from '@/domain/metricMetadata'

import { apiFetch } from './client'
import type {
  EntityKind,
  MetricRegistryPayload,
  RankRangesPayload,
  SeasonDataset,
  SeasonsResponse,
  TablePayload,
} from './types'

/** Seasons that have a complete Parquet contract under `data/`, newest first. */
export function useSeasons() {
  return useQuery({
    queryKey: ['seasons'],
    queryFn: ({ signal }) => apiFetch<SeasonsResponse>('/api/seasons', signal),
    staleTime: Infinity,
  })
}

/**
 * The metric registry: labels, descriptions, polarity, and categories for every column.
 *
 * Fetching it hydrates the module-level metadata the tables read synchronously.
 */
export function useMetricRegistry() {
  return useQuery({
    queryKey: ['metadata'],
    queryFn: async ({ signal }) => {
      const registry = await apiFetch<MetricRegistryPayload>('/api/metadata', signal)
      hydrateMetricRegistry(registry)
      return registry
    },
    staleTime: Infinity,
  })
}

/** One season's team and QB tables. */
export function useSeasonDataset(season: number | null) {
  return useQuery({
    queryKey: ['season', season],
    enabled: season !== null,
    queryFn: async ({ signal }) => {
      const dataset = await apiFetch<SeasonDataset>(`/api/seasons/${season}`, signal)
      hydrateColumnMetadata(dataset.teams.column_metadata)
      hydrateColumnMetadata(dataset.qbs.column_metadata)
      return dataset
    },
  })
}

/** Game-by-game rows for one team or QB. */
export function useEntityGameLogs(kind: EntityKind, season: number, entityId: string) {
  return useQuery({
    queryKey: ['game-logs', kind, season, entityId],
    enabled: entityId !== '',
    queryFn: async ({ signal }) => {
      const payload = await apiFetch<TablePayload>(
        `/api/seasons/${season}/${kind}/${encodeURIComponent(entityId)}/game-logs`,
        signal,
      )
      hydrateColumnMetadata(payload.column_metadata)
      return payload
    },
  })
}

/** One team's or QB's rating as of each week; seasons built before rating histories answer 404. */
export function useRatingHistory(kind: EntityKind, season: number, entityId: string) {
  return useQuery({
    queryKey: ['rating-history', kind, season, entityId],
    enabled: entityId !== '',
    queryFn: async ({ signal }) => {
      const payload = await apiFetch<TablePayload>(
        `/api/seasons/${season}/${kind}/${encodeURIComponent(entityId)}/rating-history`,
        signal,
      )
      hydrateColumnMetadata(payload.column_metadata)
      return payload
    },
  })
}

/** Every team's or QB's bootstrap rank range for a season; seasons built without them answer 404. */
export function useRankRanges(kind: EntityKind, season: number) {
  return useQuery({
    queryKey: ['rating-ranges', kind, season],
    queryFn: async ({ signal }) => {
      const payload = await apiFetch<RankRangesPayload>(`/api/seasons/${season}/${kind}/rating-ranges`, signal)
      hydrateColumnMetadata(payload.column_metadata)
      return payload
    },
  })
}
