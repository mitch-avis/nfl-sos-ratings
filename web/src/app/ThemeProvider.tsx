import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import type { PaletteMode, ThemeMode } from '@/api/types'
import { normalizePalette, PALETTE_CSS_VARIABLES, paletteCssVariables } from '@/domain/teamPalettes'

export type Theme = ThemeMode | 'system'

interface ThemeContextValue {
  theme: Theme
  resolved: ThemeMode
  palette: PaletteMode
  setTheme: (theme: Theme) => void
  setPalette: (palette: PaletteMode) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)
const THEME_KEY = 'nfl-sos-theme'
const PALETTE_KEY = 'nfl-sos-palette'

function readStored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const stored = window.localStorage.getItem(key)
    if (stored !== null && (allowed as readonly string[]).includes(stored)) return stored as T
  } catch {
    /* storage unavailable */
  }
  return fallback
}

function readStoredPalette(): PaletteMode {
  try {
    return normalizePalette(window.localStorage.getItem(PALETTE_KEY))
  } catch {
    return 'classic'
  }
}

function store(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    /* storage unavailable */
  }
}

function systemPrefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches
}

/**
 * Applies the `dark` class and `data-palette` attribute to `<html>` and remembers both choices.
 * A team palette (`teamPalettes.ts`) sets its accent and chart colors as CSS variables on `<html>`
 * for the current mode; the default palette clears them so the stylesheet's values apply. The old
 * stored `broncos` choice reads as the Denver palette.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() =>
    readStored(THEME_KEY, ['light', 'dark', 'system'] as const, 'system'),
  )
  const [palette, setPaletteState] = useState<PaletteMode>(readStoredPalette)
  const [systemDark, setSystemDark] = useState(systemPrefersDark)

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!media) return
    const onChange = (event: MediaQueryListEvent) => setSystemDark(event.matches)
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  const resolved: ThemeMode = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme

  useEffect(() => {
    document.documentElement.classList.toggle('dark', resolved === 'dark')
    document.documentElement.style.colorScheme = resolved
  }, [resolved])

  useEffect(() => {
    const root = document.documentElement
    root.dataset.palette = palette
    for (const variable of PALETTE_CSS_VARIABLES) root.style.removeProperty(variable)
    for (const [variable, value] of Object.entries(paletteCssVariables(palette, resolved))) {
      root.style.setProperty(variable, value)
    }
  }, [palette, resolved])

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next)
    store(THEME_KEY, next)
  }, [])

  const setPalette = useCallback((next: PaletteMode) => {
    setPaletteState(next)
    store(PALETTE_KEY, next)
  }, [])

  const value = useMemo(
    () => ({ theme, resolved, palette, setTheme, setPalette }),
    [theme, resolved, palette, setTheme, setPalette],
  )
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext)
  if (!value) throw new Error('useTheme must be used inside ThemeProvider')
  return value
}
