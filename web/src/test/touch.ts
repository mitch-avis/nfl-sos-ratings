import { vi } from 'vitest'

/** Make `(hover: none)` match, as on a phone, until `vi.restoreAllMocks()`. */
export function stubTouchScreen(): void {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (query: string) =>
      ({
        matches: query === '(hover: none)',
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  )
}
