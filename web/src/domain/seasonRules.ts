/** The first season with 17 regular-season games per team. */
const FIRST_17_GAME_SEASON = 2021

/** Regular-season games per team: 17 from 2021 on, 16 before. */
export function getRegularSeasonGameCount(season: number): number {
  return season >= FIRST_17_GAME_SEASON ? 17 : 16
}

/**
 * The most games any team has played while the season is still in progress, or null once it is
 * complete.
 *
 * A season is in progress until every team has played a full regular season, so the notice stays
 * through the last week. The value comes from the loaded rows, so it follows each data refresh.
 * Returns null when the rows carry no `games_played` values.
 */
export function getInProgressGames(
  season: number,
  teamRows: ReadonlyArray<Record<string, unknown>>,
): number | null {
  const played = teamRows
    .map((row) => row.games_played)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (played.length === 0) return null
  return Math.min(...played) < getRegularSeasonGameCount(season) ? Math.max(...played) : null
}
