import type { EntityKind, SeasonDataset, SeasonsResponse, TablePayload } from './types';

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

export function fetchSeasonDataset(season: number): Promise<SeasonDataset> {
  return readJson<SeasonDataset>(`/api/seasons/${season}`);
}

export function fetchEntityGameLogs(
  kind: EntityKind,
  season: number,
  entityId: string,
): Promise<TablePayload> {
  return readJson<TablePayload>(`/api/seasons/${season}/${kind}/${entityId}/game-logs`);
}