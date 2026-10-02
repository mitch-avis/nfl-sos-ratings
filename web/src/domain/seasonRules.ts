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
