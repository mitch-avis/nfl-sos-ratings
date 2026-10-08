import type { ReactNode } from 'react'
import { useParams } from 'react-router'

import { useTeamPageColors } from './useTeamPageColors'

/**
 * Shows the team page's team palette from the URL, before and while the season's data loads, so
 * the page keeps its colors through a season change.
 */
export function TeamPageColors({ children }: { children: ReactNode }) {
  const team = decodeURIComponent(useParams().entityId ?? '')
  useTeamPageColors(team === '' ? null : team)
  return children
}
