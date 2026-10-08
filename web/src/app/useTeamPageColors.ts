import { useLayoutEffect } from 'react'

import { useTheme } from './ThemeProvider'

/**
 * Shows `team`'s palette while the calling page is mounted (when team colors are on), and the
 * chosen palette again once it unmounts. Runs before paint, so the page never flashes the other
 * palette.
 */
export function useTeamPageColors(team: string | null): void {
  const { setPageTeam } = useTheme()
  useLayoutEffect(() => {
    setPageTeam(team)
    return () => setPageTeam(null)
  }, [setPageTeam, team])
}
