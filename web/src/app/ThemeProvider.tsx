import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import type { PaletteMode, ThemeMode } from '@/api/types'

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
 * The Broncos palette swaps the accent and the heat-map colors for an orange-and-navy scale.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() =>
    readStored(THEME_KEY, ['light', 'dark', 'system'] as const, 'system'),
  )
  const [palette, setPaletteState] = useState<PaletteMode>(() =>
    readStored(PALETTE_KEY, ['classic', 'broncos'] as const, 'classic'),
  )
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
    document.documentElement.dataset.palette = palette
  }, [palette])

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
