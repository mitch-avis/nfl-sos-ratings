import { useQuery } from '@tanstack/react-query'

import { hydrateColumnMetadata, hydrateMetricRegistry } from '@/domain/metricMetadata'

import { apiFetch } from './client'
import type {
  EntityKind,
  MetricRegistryPayload,
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
