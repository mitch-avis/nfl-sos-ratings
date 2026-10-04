import { useSyncExternalStore } from 'react'

// Phones and most tablets report no hover; a mouse or trackpad (including an iPad's) reports hover.
const TOUCH_ONLY_QUERY = '(hover: none)'

function subscribe(onChange: () => void): () => void {
  const query = window.matchMedia(TOUCH_ONLY_QUERY)
  query.addEventListener('change', onChange)
  return () => query.removeEventListener('change', onChange)
}

/** Whether the primary pointer can hover; false on touch screens, where hints open on tap. */
export function useHasHover(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => !window.matchMedia(TOUCH_ONLY_QUERY).matches,
    () => true,
  )
}
