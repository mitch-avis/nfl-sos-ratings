import { useLayoutEffect } from 'react'

import { useTheme } from './ThemeProvider'

/**
 * Shows `team`'s palette while the calling component is mounted (when team colors are on), and the
 * chosen palette again once it unmounts; `null` leaves the palette alone. Runs before paint. Team
 * pages call it from their route, outside the season loader, so a team page keeps its colors while
 * another season loads; QB pages know their team only once the season's data is in.
 */
export function useTeamPageColors(team: string | null): void {
  const { setPageTeam } = useTheme()
  useLayoutEffect(() => {
    if (team === null) return undefined
    setPageTeam(team)
    return () => setPageTeam(null)
  }, [setPageTeam, team])
}
