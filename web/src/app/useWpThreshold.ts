import { useCallback } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

import { parseWpThreshold, WP_QUERY_KEY } from '@/domain/wpFilter'

/**
 * The garbage-time filter threshold in the `?wp=` query string (0, the default, means off), and a
 * setter that replaces the history entry so dragging the slider does not flood the back button.
 */
export function useWpThreshold(): [number, (next: number) => void] {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const navigate = useNavigate()
  const threshold = parseWpThreshold(searchParams.get(WP_QUERY_KEY))

  const setThreshold = useCallback(
    (next: number) => {
      const params = new URLSearchParams(location.search)
      if (next === 0) params.delete(WP_QUERY_KEY)
      else params.set(WP_QUERY_KEY, String(next))
      navigate(`${location.pathname}?${params.toString()}`, { replace: true })
    },
    [location.pathname, location.search, navigate],
  )

  return [threshold, setThreshold]
}
