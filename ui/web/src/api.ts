import { hydrateColumnMetadata, hydrateMetricRegistry } from './metricMetadata';
import type {
  EntityKind,
  MetricRegistryPayload,
  SeasonDataset,
  SeasonsResponse,
  TablePayload,
} from './types';

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchSeasons(): Promise<SeasonsResponse> {
  return readJson<SeasonsResponse>('/api/seasons');
}

export async function fetchMetricRegistry(): Promise<MetricRegistryPayload> {
  const registry = await readJson<MetricRegistryPayload>('/api/metadata');
  hydrateMetricRegistry(registry);
  return registry;
}

export async function fetchSeasonDataset(season: number): Promise<SeasonDataset> {
  const dataset = await readJson<SeasonDataset>(`/api/seasons/${season}`);
  hydrateColumnMetadata(dataset.teams.column_metadata);
  hydrateColumnMetadata(dataset.qbs.column_metadata);
  return dataset;
}

export async function fetchEntityGameLogs(
  kind: EntityKind,
  season: number,
  entityId: string,
): Promise<TablePayload> {
  const payload = await readJson<TablePayload>(
    `/api/seasons/${season}/${kind}/${entityId}/game-logs`,
  );
  hydrateColumnMetadata(payload.column_metadata);
  return payload;
}
