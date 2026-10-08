import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import type { PaletteMode, ThemeMode } from '@/api/types'
import { activePalette, normalizePalette, PALETTE_CSS_VARIABLES, paletteCssVariables } from '@/domain/teamPalettes'

export type Theme = ThemeMode | 'system'

interface ThemeContextValue {
  theme: Theme
  resolved: ThemeMode
  /** The chosen palette, for the Teams, Quarterbacks, and Glossary pages and the palette menu. */
  palette: PaletteMode
  /** The palette the current page shows: a team or QB page's team, or the chosen one. */
  activePalette: PaletteMode
  /** Whether team and QB pages show their team's palette. */
  teamPageColors: boolean
  setTheme: (theme: Theme) => void
  setPalette: (palette: PaletteMode) => void
  setTeamPageColors: (on: boolean) => void
  /** Set by a team or QB page for its team; null elsewhere. */
  setPageTeam: (team: string | null) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)
const THEME_KEY = 'nfl-sos-theme'
const PALETTE_KEY = 'nfl-sos-palette'
const TEAM_PAGE_COLORS_KEY = 'nfl-sos-team-page-colors'

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
 * Applies the `dark` class and `data-palette` attribute to `<html>` and remembers the theme, the
 * chosen palette, and whether team pages use their team's colors. The palette shown is the active
 * one: a team or QB page's team (set through `useTeamPageColors`) while team colors are on,
 * otherwise the chosen palette. A team palette (`teamPalettes.ts`) sets its accent, chart, and hint
 * colors as CSS variables on `<html>` for the current mode, before paint; the default palette clears
 * them so the stylesheet's values apply. The old stored `broncos` choice reads as the Denver
 * palette.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() =>
    readStored(THEME_KEY, ['light', 'dark', 'system'] as const, 'system'),
  )
  const [palette, setPaletteState] = useState<PaletteMode>(readStoredPalette)
  const [teamPageColors, setTeamPageColorsState] = useState(
    () => readStored(TEAM_PAGE_COLORS_KEY, ['on', 'off'] as const, 'on') === 'on',
  )
  const [pageTeam, setPageTeam] = useState<string | null>(null)
  const [systemDark, setSystemDark] = useState(systemPrefersDark)

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!media) return
    const onChange = (event: MediaQueryListEvent) => setSystemDark(event.matches)
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  const resolved: ThemeMode = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme

  useLayoutEffect(() => {
    document.documentElement.classList.toggle('dark', resolved === 'dark')
    document.documentElement.style.colorScheme = resolved
  }, [resolved])

  const shown = activePalette(palette, pageTeam, teamPageColors)

  useLayoutEffect(() => {
    const root = document.documentElement
    root.dataset.palette = shown
    for (const variable of PALETTE_CSS_VARIABLES) root.style.removeProperty(variable)
    for (const [variable, value] of Object.entries(paletteCssVariables(shown, resolved))) {
      root.style.setProperty(variable, value)
    }
  }, [shown, resolved])

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next)
    store(THEME_KEY, next)
  }, [])

  const setPalette = useCallback((next: PaletteMode) => {
    setPaletteState(next)
    store(PALETTE_KEY, next)
  }, [])

  const setTeamPageColors = useCallback((on: boolean) => {
    setTeamPageColorsState(on)
    store(TEAM_PAGE_COLORS_KEY, on ? 'on' : 'off')
  }, [])

  const value = useMemo(
    () => ({
      theme,
      resolved,
      palette,
      activePalette: shown,
      teamPageColors,
      setTheme,
      setPalette,
      setTeamPageColors,
      setPageTeam,
    }),
    [theme, resolved, palette, shown, teamPageColors, setTheme, setPalette, setTeamPageColors],
  )
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext)
  if (!value) throw new Error('useTheme must be used inside ThemeProvider')
  return value
}
