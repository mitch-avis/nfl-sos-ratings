/** The first season with 17 regular-season games per team. */
const FIRST_17_GAME_SEASON = 2021

/** Pass attempts per team game a QB needs to qualify for the season ratings. */
export const QUALIFIER_ATTEMPTS_PER_GAME = 14

/** Regular-season games per team: 17 from 2021 on, 16 before. */
export function getRegularSeasonGameCount(season: number): number {
  return season >= FIRST_17_GAME_SEASON ? 17 : 16
}

/** The season's QB rating qualifier: 14 pass attempts per team game. */
export function getQuarterbackQualifierAttempts(season: number): number {
  return getRegularSeasonGameCount(season) * QUALIFIER_ATTEMPTS_PER_GAME
}

/**
 * Games played so far when the season is still in progress, or null once it is complete.
 *
 * A season is in progress while no team has played a full regular season. Returns null when the
 * rows carry no `games_played` values.
 */
export function getInProgressGames(
  season: number,
  teamRows: ReadonlyArray<Record<string, unknown>>,
): number | null {
  const played = teamRows
    .map((row) => row.games_played)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (played.length === 0) return null
  const mostGames = Math.max(...played)
  return mostGames < getRegularSeasonGameCount(season) ? mostGames : null
}
