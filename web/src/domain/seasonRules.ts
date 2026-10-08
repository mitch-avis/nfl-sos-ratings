import type { SeasonDataset } from '@/api/types'

/**
 * The most games any team has played, for the season-in-progress notice; null unless the API marks
 * the season as the one being played (a completed season with a missing or cancelled game is never
 * in progress). Null too when the rows carry no `games_played` values.
 */
export function getInProgressGames(dataset: Pick<SeasonDataset, 'in_progress' | 'teams'>): number | null {
  if (!dataset.in_progress) return null
  const played = dataset.teams.rows
    .map((row) => row.games_played)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  return played.length === 0 ? null : Math.max(...played)
}
