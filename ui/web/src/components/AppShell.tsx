import { NavLink } from 'react-router-dom';

import type { ReactElement, ReactNode } from 'react';

import type { PaletteMode, ThemeMode } from '../types';

interface AppShellProps {
  seasons: number[];
  season: number;
  onSeasonChange: (season: number) => void;
  palette: PaletteMode;
  theme: ThemeMode;
  onPaletteChange: (palette: PaletteMode) => void;
  onThemeChange: (theme: ThemeMode) => void;
  children: ReactNode;
}

export function AppShell({
  seasons,
  season,
  onSeasonChange,
  palette,
  theme,
  onPaletteChange,
  onThemeChange,
  children,
}: AppShellProps): ReactElement {
  return (
    <div className="app-frame">
      <aside className="masthead">
        <section className="masthead-brand panel feature-card">
          <h1>NFL SOS Ratings</h1>
          <p className="brand-description">
            Schedule-adjusted team and QB tables pulled directly from the generated CSV outputs.
          </p>
        </section>

        <section className="masthead-section">
          <p className="eyebrow">Season</p>
          <div className="season-box">
            <select
              id="season-select"
              aria-label="Season"
              value={season}
              onChange={(event) => onSeasonChange(Number(event.target.value))}
            >
              {seasons.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </div>
        </section>

        <section className="masthead-section">
          <p className="eyebrow">Views</p>
          <nav className="nav-pills">
            <NavLink to={`/teams?season=${season}`}>Teams</NavLink>
            <NavLink to={`/qbs?season=${season}`}>QBs</NavLink>
            <NavLink to={`/glossary?season=${season}`}>Glossary</NavLink>
          </nav>
        </section>

        <section className="masthead-section">
          <p className="eyebrow">Theme</p>
          <div className="segmented-control">
            <button
              type="button"
              className={theme === 'light' ? 'active' : ''}
              onClick={() => onThemeChange('light')}
            >
              Light
            </button>
            <button
              type="button"
              className={theme === 'dark' ? 'active' : ''}
              onClick={() => onThemeChange('dark')}
            >
              Dark
            </button>
          </div>
        </section>

        <section className="masthead-section">
          <p className="eyebrow">Palette</p>
          <div className="segmented-control">
            <button
              type="button"
              className={palette === 'classic' ? 'active' : ''}
              onClick={() => onPaletteChange('classic')}
            >
              Classic
            </button>
            <button
              type="button"
              className={palette === 'broncos' ? 'active' : ''}
              onClick={() => onPaletteChange('broncos')}
            >
              Broncos
            </button>
          </div>
        </section>
      </aside>
      <main className="workspace">{children}</main>
    </div>
  );
}