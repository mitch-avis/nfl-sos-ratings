import { useCallback } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

import { useSeasons } from '@/api/queries'

/**
 * The season in the `?season=` query string, falling back to the newest season with data.
 *
 * `season` is null until the season list loads (or when no season has data). `unavailable` is the
 * requested season when the list has loaded without it, so the page can say what it showed instead.
 */
export function useSeason(): {
  season: number | null
  seasons: number[]
  unavailable: number | null
  setSeason: (season: number) => void
} {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const navigate = useNavigate()
  const seasonsQuery = useSeasons()
  const seasons = seasonsQuery.data?.seasons ?? []
  const requestedText = searchParams.get('season')
  const requested = Number(requestedText)
  const season = seasons.includes(requested) ? requested : (seasons[0] ?? null)
  const unavailable =
    requestedText !== null && seasons.length > 0 && !seasons.includes(requested) && Number.isInteger(requested)
      ? requested
      : null

  const setSeason = useCallback(
    (next: number) => {
      const params = new URLSearchParams(searchParams)
      params.set('season', String(next))
      params.delete('compare')
      navigate(`${location.pathname}?${params.toString()}`)
    },
    [location.pathname, navigate, searchParams],
  )

  return { season, seasons, unavailable, setSeason }
}

/** A link target that keeps the current season. */
export function withSeason(path: string, season: number | null): string {
  return season === null ? path : `${path}?season=${season}`
}
