import { NavLink } from 'react-router-dom';

import { useEffect, useState, type ReactElement, type ReactNode } from 'react';

import { getPageJumpSlots, type ScrollPositionState } from '../detailUi';
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

const SCROLL_EDGE_THRESHOLD = 8;

function readScrollPositionState(): ScrollPositionState {
  if (typeof window === 'undefined') {
    return { atBottom: false, atTop: true, canScroll: false };
  }

  const documentHeight = Math.max(
    document.body.scrollHeight,
    document.documentElement.scrollHeight,
  );
  const maxScrollTop = Math.max(documentHeight - window.innerHeight, 0);
  const scrollTop = Math.max(
    window.scrollY,
    window.pageYOffset,
    document.documentElement.scrollTop,
    document.body.scrollTop,
    0,
  );
  const canScroll = maxScrollTop > SCROLL_EDGE_THRESHOLD;

  if (!canScroll) {
    return { atBottom: true, atTop: true, canScroll: false };
  }

  return {
    atBottom: maxScrollTop - scrollTop <= SCROLL_EDGE_THRESHOLD,
    atTop: scrollTop <= SCROLL_EDGE_THRESHOLD,
    canScroll,
  };
}

function scrollToPageEdge(edge: 'top' | 'bottom'): void {
  if (typeof window === 'undefined') {
    return;
  }

  window.scrollTo({
    top: edge === 'top' ? 0 : document.documentElement.scrollHeight,
    behavior: 'smooth',
  });
}

function PageJumpButtons(): ReactElement | null {
  const [scrollPosition, setScrollPosition] = useState<ScrollPositionState>(readScrollPositionState);
  const slots = getPageJumpSlots(scrollPosition);

  useEffect(() => {
    const updateScrollPosition = (): void => {
      setScrollPosition((current) => {
        const next = readScrollPositionState();
        return current.atBottom === next.atBottom
          && current.atTop === next.atTop
          && current.canScroll === next.canScroll
          ? current
          : next;
      });
    };

    updateScrollPosition();
    window.addEventListener('scroll', updateScrollPosition, { passive: true });
    window.addEventListener('resize', updateScrollPosition);

    return () => {
      window.removeEventListener('scroll', updateScrollPosition);
      window.removeEventListener('resize', updateScrollPosition);
    };
  }, []);

  if (slots.length === 0) {
    return null;
  }

  return (
    <div className="page-jump-controls" aria-label="Page navigation shortcuts">
      {slots.map((slot) => (
        <button
          key={slot.direction}
          type="button"
          aria-hidden={!slot.visible}
          className={slot.visible ? 'page-jump-button' : 'page-jump-button is-hidden'}
          disabled={!slot.visible}
          aria-label={slot.direction === 'up' ? 'Scroll to top' : 'Scroll to bottom'}
          onClick={() => scrollToPageEdge(slot.direction === 'up' ? 'top' : 'bottom')}
          tabIndex={slot.visible ? 0 : -1}
        >
          {slot.direction === 'up' ? '↑' : '↓'}
        </button>
      ))}
    </div>
  );
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
            Analyst-facing team and QB views that keep rankings, opponent context, and weekly
            season detail tied to the same schedule-adjusted stat surface.
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
      <PageJumpButtons />
    </div>
  );
}