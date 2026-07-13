import type { SortingState } from '@tanstack/react-table';
import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from 'react-router-dom';

import { fetchEntityGameLogs, fetchSeasonDataset, fetchSeasons } from './api';
import { ComparisonPanel } from './components/ComparisonPanel';
import { AppShell } from './components/AppShell';
import { DataTable } from './components/DataTable';
import { EntityDetail } from './components/EntityDetail';
import { GlossaryPage } from './components/GlossaryPage';
import { OverviewCards } from './components/OverviewCards';
import { getEntityConfig, getEntityLabel, getEntityRow } from './entityConfig';
import { getMetricMetadata } from './metricMetadata';
import { buildEnabledGroups, getSelectedColumns } from './tableState';
import type { EntityKind, PaletteMode, SeasonDataset, TablePayload, ThemeMode } from './types';

function readStoredTheme(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'light';
  }
  return window.localStorage.getItem('nfl-sos-theme') === 'dark' ? 'dark' : 'light';
}

function readStoredPalette(): PaletteMode {
  if (typeof window === 'undefined') {
    return 'classic';
  }
  return window.localStorage.getItem('nfl-sos-palette') === 'broncos' ? 'broncos' : 'classic';
}

interface EntityPageViewState {
  activeDetailGroup?: string;
  compareIds?: string[];
  enabledGroups?: Record<string, boolean>;
  query?: string;
  showUnratedRows?: boolean;
  sorting?: SortingState;
}

function getRouteEntityKind(pathname: string): EntityKind | null {
  if (pathname === '/teams' || pathname.startsWith('/teams/')) {
    return 'teams';
  }
  if (pathname === '/qbs' || pathname.startsWith('/qbs/')) {
    return 'qbs';
  }
  return null;
}

function isIndexRoute(pathname: string): boolean {
  return pathname === '/teams' || pathname === '/qbs';
}

function buildDefaultSorting(kind: EntityKind): SortingState {
  const config = getEntityConfig(kind);
  return [
    {
      id: config.defaultSortColumn,
      desc: getMetricMetadata(config.defaultSortColumn).polarity !== 'lower',
    },
  ];
}

function buildResolvedEnabledGroups(
  kind: EntityKind,
  table: TablePayload,
  enabledGroups?: Record<string, boolean>,
): Record<string, boolean> {
  const defaults = buildEnabledGroups(table.column_groups, getEntityConfig(kind).defaultGroups);
  return Object.fromEntries(
    Object.keys(defaults).map((group) => [group, enabledGroups?.[group] ?? defaults[group]]),
  );
}

function buildAllEnabledGroups(table: TablePayload): Record<string, boolean> {
  return Object.fromEntries(Object.keys(table.column_groups).map((group) => [group, true]));
}

function buildResolvedPageViewState(
  kind: EntityKind,
  table: TablePayload,
  current?: EntityPageViewState,
): {
  activeDetailGroup?: string;
  compareIds: string[];
  enabledGroups: Record<string, boolean>;
  query: string;
  showUnratedRows: boolean;
  sorting: SortingState;
} {
  const config = getEntityConfig(kind);
  const availableDetailGroups = config.detailGroups.filter(
    (group) => (table.column_groups[group] ?? []).length > 0,
  );
  return {
    activeDetailGroup: availableDetailGroups.includes(current?.activeDetailGroup ?? '')
      ? current?.activeDetailGroup
      : availableDetailGroups[0],
    compareIds: current?.compareIds ?? [],
    enabledGroups: buildResolvedEnabledGroups(kind, table, current?.enabledGroups),
    query: current?.query ?? '',
    showUnratedRows: current?.showUnratedRows ?? false,
    sorting: current?.sorting?.length ? current.sorting : buildDefaultSorting(kind),
  };
}

function reconcileCompareIds(kind: EntityKind, table: TablePayload, compareIds: string[]): string[] {
  const identityKey = getEntityConfig(kind).identityKey;
  const availableIds = new Set(table.rows.map((row) => String(row[identityKey] ?? '')));
  return compareIds.filter((entityId) => availableIds.has(entityId));
}

function enabledGroupsEqual(
  left: Record<string, boolean>,
  right: Record<string, boolean>,
): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  return Array.from(keys).every((key) => Boolean(left[key]) === Boolean(right[key]));
}

function sortingStatesEqual(left: SortingState, right: SortingState): boolean {
  return (
    left.length === right.length
    && left.every(
      (entry, index) => entry.id === right[index]?.id && entry.desc === right[index]?.desc,
    )
  );
}

function stringArraysEqual(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function getRegularSeasonGameCount(season: number): number {
  return season >= 2021 ? 17 : 16;
}

function getQuarterbackQualifierAttempts(season: number): number {
  return getRegularSeasonGameCount(season) * 14;
}

function buildDocumentTitle(
  pathname: string,
  season: number | null,
  dataset: SeasonDataset | null,
): string {
  const baseTitle = 'NFL SOS Analyst UI';
  const routeKind = getRouteEntityKind(pathname);
  if (pathname === '/glossary') {
    return `${baseTitle} - Glossary`;
  }
  if (!routeKind) {
    return baseTitle;
  }

  const seasonPart = season ? `${season}` : null;
  const routeLabel = routeKind === 'teams' ? 'Teams' : 'QBs';
  const segments = [baseTitle];
  if (seasonPart) {
    segments.push(seasonPart);
  }
  segments.push(routeLabel);

  const pathParts = pathname.split('/').filter(Boolean);
  const entityId = pathParts[1];
  if (entityId && dataset) {
    const row = getEntityRow(dataset[routeKind], routeKind, entityId);
    if (row) {
      segments.push(getEntityLabel(routeKind, row));
    }
  }

  return segments.join(' - ');
}

function canResetPageView(
  kind: EntityKind,
  table: TablePayload,
  viewState: ReturnType<typeof buildResolvedPageViewState>,
  compareIds: string[],
): boolean {
  const defaults = buildResolvedPageViewState(kind, table);
  return (
    !enabledGroupsEqual(viewState.enabledGroups, defaults.enabledGroups)
    || !sortingStatesEqual(viewState.sorting, defaults.sorting)
    || compareIds.length > 0
    || viewState.query.trim().length > 0
    || (kind === 'qbs' && viewState.showUnratedRows)
  );
}

function EntityPage({
  canReset,
  compareIds,
  dataset,
  enabledGroups,
  kind,
  onEnableAllGroups,
  onQueryChange,
  onResetView,
  onShowUnratedRowsChange,
  onSortingChange,
  onToggleCompare,
  onToggleGroup,
  palette,
  query,
  season,
  showUnratedRows,
  sorting,
  theme,
}: {
  canReset: boolean;
  compareIds: string[];
  dataset: SeasonDataset;
  enabledGroups: Record<string, boolean>;
  kind: EntityKind;
  onEnableAllGroups: () => void;
  onQueryChange: (query: string) => void;
  onResetView: () => void;
  onShowUnratedRowsChange?: (showUnratedRows: boolean) => void;
  onSortingChange: (sorting: SortingState) => void;
  onToggleCompare: (entityId: string) => void;
  onToggleGroup: (group: string) => void;
  palette: PaletteMode;
  query: string;
  season: number;
  showUnratedRows: boolean;
  sorting: SortingState;
  theme: ThemeMode;
}): ReactElement {
  const config = getEntityConfig(kind);
  const table = dataset[kind];
  const basePath = `/${kind}`;
  const identityColumns = config.identityColumns;
  const showUnratedQbs = showUnratedRows;

  const selectedColumns = useMemo(
    () => getSelectedColumns(table.column_groups, enabledGroups, table.visible_columns),
    [enabledGroups, table.column_groups, table.visible_columns],
  );

  const displayedRows = useMemo(() => {
    if (kind !== 'qbs' || showUnratedQbs) {
      return table.rows;
    }
    return table.rows.filter(
      (row) => row.QSaCR !== null || row.QSaOR !== null || row.QRaw !== null,
    );
  }, [kind, showUnratedQbs, table.rows]);

  const displayTable = useMemo(
    () => ({ ...table, rows: displayedRows }),
    [displayedRows, table],
  );

  const compareColumns = useMemo(() => {
    const requested = selectedColumns.filter((column) => column !== config.identityKey);
    return requested.length > 0 ? requested : config.compareColumns;
  }, [config.compareColumns, config.identityKey, selectedColumns]);
  const regularSeasonGames = getRegularSeasonGameCount(season);
  const qualifierAttempts = getQuarterbackQualifierAttempts(season);

  return (
    <>
      <section className="hero panel">
        <p className="eyebrow">{dataset.season} Regular Season</p>
        <h2>{config.title}</h2>
        <p>
          Sortable, filterable contract views with deep-linked detail pages and a first comparison
          strip, so the subject row, schedule context, and rating outputs stay inspectable in one
          place.
        </p>
      </section>
      <section className="panel guidance-panel">
        <div className="guidance-grid">
          <article>
            <p className="eyebrow">Use First</p>
            <strong>{config.primaryRankingLabel}</strong>
            <p>{config.primaryRankingDescription}</p>
          </article>
          {config.pageNotes.map((note) => (
            <article key={note}>
              <p className="eyebrow">Reading Note</p>
              <p>{note}</p>
            </article>
          ))}
          {kind === 'qbs' ? (
            <article>
              <p className="eyebrow">Row Filter</p>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showUnratedQbs}
                  onChange={(event) => onShowUnratedRowsChange?.(event.target.checked)}
                />
                <span>Show unrated or empty QB rows</span>
              </label>
              <p>
                Includes quarterbacks who played at least one offensive snap but finished below
                the season rating threshold of {qualifierAttempts} pass attempts, which follows
                the standard 14 attempts per team game qualifier for this {regularSeasonGames}-game
                season.
              </p>
            </article>
          ) : null}
        </div>
      </section>
      <OverviewCards
        displayCount={displayTable.rows.length}
        table={table}
        entityLabel={config.singularLabel}
        kind={kind}
        totalCount={table.rows.length}
      />
      <ComparisonPanel
        basePath={basePath}
        compareColumns={compareColumns}
        compareIds={compareIds}
        config={config}
        palette={palette}
        season={season}
        table={displayTable}
        theme={theme}
        onRemove={onToggleCompare}
      />
      <DataTable
        key={`${kind}-${season}`}
        basePath={basePath}
        compareIds={compareIds}
        defaultSortColumn={config.defaultSortColumn}
        detailColumn={config.labelKey}
        enabledGroups={enabledGroups}
        entityKind={kind}
        identityKey={config.identityKey}
        identityColumns={identityColumns}
        onEnableAllGroups={onEnableAllGroups}
        onQueryChange={onQueryChange}
        onResetView={onResetView}
        onSortingChange={onSortingChange}
        onToggleGroup={onToggleGroup}
        onToggleCompare={onToggleCompare}
        palette={palette}
        query={query}
        canReset={canReset}
        season={season}
        selectedColumns={selectedColumns}
        sorting={sorting}
        theme={theme}
        title={config.title}
        table={displayTable}
      />
    </>
  );
}

function EntityDetailPage({
  activeGroup,
  dataset,
  kind,
  onActiveGroupChange,
  palette,
  season,
  theme,
}: {
  activeGroup?: string;
  dataset: SeasonDataset;
  kind: EntityKind;
  onActiveGroupChange: (group: string) => void;
  palette: PaletteMode;
  season: number;
  theme: ThemeMode;
}): ReactElement {
  const navigate = useNavigate();
  const params = useParams();
  const config = getEntityConfig(kind);
  const table = dataset[kind];
  const entityId = params.entityId ?? '';
  const row = getEntityRow(table, kind, entityId);
  const [gameLogs, setGameLogs] = useState<TablePayload | null>(null);
  const [gameLogError, setGameLogError] = useState<string | null>(null);
  const [gameLogsLoading, setGameLogsLoading] = useState(true);

  useEffect(() => {
    if (row) {
      return;
    }

    navigate(`/${kind}?season=${season}`, { replace: true });
  }, [kind, navigate, row, season]);

  useEffect(() => {
    if (!row) {
      return;
    }

    let cancelled = false;
    setGameLogsLoading(true);
    setGameLogError(null);

    async function loadGameLogs(): Promise<void> {
      try {
        const payload = await fetchEntityGameLogs(kind, season, entityId);
        if (!cancelled) {
          setGameLogs(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setGameLogs(null);
          setGameLogError(
            loadError instanceof Error ? loadError.message : 'Could not load weekly game logs.',
          );
        }
      } finally {
        if (!cancelled) {
          setGameLogsLoading(false);
        }
      }
    }

    void loadGameLogs();
    return () => {
      cancelled = true;
    };
  }, [entityId, kind, row, season]);

  if (!row) {
    return <section className="panel loading-panel">Returning to {config.title.toLowerCase()}...</section>;
  }

  return (
    <EntityDetail
      activeGroup={activeGroup}
      basePath={`/${kind}`}
      config={config}
      gameLogError={gameLogError}
      gameLogs={gameLogs}
      gameLogsLoading={gameLogsLoading}
      onActiveGroupChange={onActiveGroupChange}
      opponentRatingsTable={dataset.teams}
      palette={palette}
      row={row}
      season={season}
      table={table}
      theme={theme}
    />
  );
}

export function App(): ReactElement {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const [seasons, setSeasons] = useState<number[]>([]);
  const [dataset, setDataset] = useState<SeasonDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState<ThemeMode>(readStoredTheme);
  const [palette, setPalette] = useState<PaletteMode>(readStoredPalette);
  const [pageViewStates, setPageViewStates] = useState<Record<EntityKind, EntityPageViewState>>({
    teams: {},
    qbs: {},
  });
  const hydratedCompareQueryKeys = useRef<Partial<Record<EntityKind, string>>>({});

  const selectedSeason = Number(searchParams.get('season'));
  const compareIdsFromQuery = searchParams
    .get('compare')
    ?.split(',')
    .map((value) => value.trim())
    .filter(Boolean) ?? [];
  const routeKind = getRouteEntityKind(location.pathname);

  const resolvedViewStates = useMemo(
    () =>
      dataset
        ? {
            teams: buildResolvedPageViewState('teams', dataset.teams, pageViewStates.teams),
            qbs: buildResolvedPageViewState('qbs', dataset.qbs, pageViewStates.qbs),
          }
        : null,
    [dataset, pageViewStates],
  );

  const compareIdsByKind = useMemo(() => {
    if (!dataset || !resolvedViewStates) {
      return { teams: [] as string[], qbs: [] as string[] };
    }
    return {
      teams: reconcileCompareIds('teams', dataset.teams, resolvedViewStates.teams.compareIds),
      qbs: reconcileCompareIds('qbs', dataset.qbs, resolvedViewStates.qbs.compareIds),
    };
  }, [dataset, resolvedViewStates]);

  const compareHydrationState = useMemo(() => {
    if (!dataset || !resolvedViewStates || !routeKind || !isIndexRoute(location.pathname)) {
      return null;
    }

    const fromQuery = reconcileCompareIds(routeKind, dataset[routeKind], compareIdsFromQuery);
    const key = `${routeKind}:${selectedSeason || dataset.season}:${fromQuery.join(',')}`;
    return {
      compareIds: fromQuery,
      currentState: resolvedViewStates[routeKind].compareIds,
      key,
      kind: routeKind,
    };
  }, [compareIdsFromQuery, dataset, location.pathname, resolvedViewStates, routeKind, selectedSeason]);

  useEffect(() => {
    let cancelled = false;

    async function loadSeasons(): Promise<void> {
      try {
        const response = await fetchSeasons();
        if (cancelled) {
          return;
        }
        setSeasons(response.seasons);

        const fallbackSeason = response.seasons[0];
        if (!selectedSeason && fallbackSeason) {
          navigate(`${location.pathname}?season=${fallbackSeason}`, { replace: true });
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Could not load seasons.');
        }
      }
    }

    void loadSeasons();
    return () => {
      cancelled = true;
    };
  }, [location.pathname, navigate, selectedSeason]);

  useEffect(() => {
    if (!selectedSeason) {
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    async function loadDataset(): Promise<void> {
      try {
        const nextDataset = await fetchSeasonDataset(selectedSeason);
        if (!cancelled) {
          setDataset(nextDataset);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Could not load season data.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadDataset();
    return () => {
      cancelled = true;
    };
  }, [selectedSeason]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('nfl-sos-theme', theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.dataset.palette = palette;
    window.localStorage.setItem('nfl-sos-palette', palette);
  }, [palette]);

  useEffect(() => {
    document.title = buildDocumentTitle(location.pathname, selectedSeason || dataset?.season || null, dataset);
  }, [dataset, location.pathname, selectedSeason]);

  useEffect(() => {
    if (!compareHydrationState || compareHydrationState.compareIds.length === 0) {
      return;
    }
    if (hydratedCompareQueryKeys.current[compareHydrationState.kind] === compareHydrationState.key) {
      return;
    }

    hydratedCompareQueryKeys.current[compareHydrationState.kind] = compareHydrationState.key;

    if (stringArraysEqual(compareHydrationState.compareIds, compareHydrationState.currentState)) {
      return;
    }

    updatePageViewState(compareHydrationState.kind, {
      compareIds: compareHydrationState.compareIds,
    });
  }, [compareHydrationState]);

  useEffect(() => {
    if (!dataset || !resolvedViewStates) {
      return;
    }

    const nextPatches: Partial<Record<EntityKind, string[]>> = {};
    for (const kind of ['teams', 'qbs'] as const) {
      const reconciled = reconcileCompareIds(kind, dataset[kind], resolvedViewStates[kind].compareIds);
      if (!stringArraysEqual(reconciled, resolvedViewStates[kind].compareIds)) {
        nextPatches[kind] = reconciled;
      }
    }

    if (Object.keys(nextPatches).length === 0) {
      return;
    }

    setPageViewStates((current) => ({
      ...current,
      ...(nextPatches.teams
        ? {
            teams: {
              ...current.teams,
              compareIds: nextPatches.teams,
            },
          }
        : {}),
      ...(nextPatches.qbs
        ? {
            qbs: {
              ...current.qbs,
              compareIds: nextPatches.qbs,
            },
          }
        : {}),
    }));
  }, [dataset, resolvedViewStates]);

  useEffect(() => {
    if (!dataset || !routeKind || !isIndexRoute(location.pathname)) {
      return;
    }
    if (
      compareHydrationState
      && compareHydrationState.compareIds.length > 0
      && hydratedCompareQueryKeys.current[compareHydrationState.kind] !== compareHydrationState.key
    ) {
      return;
    }

    const compareIds = routeKind === 'teams' ? compareIdsByKind.teams : compareIdsByKind.qbs;
    const nextSearchParams = new URLSearchParams(searchParams);
    nextSearchParams.set('season', String(selectedSeason || dataset.season));
    if (compareIds.length > 0) {
      nextSearchParams.set('compare', compareIds.join(','));
    } else {
      nextSearchParams.delete('compare');
    }

    if (nextSearchParams.toString() !== searchParams.toString()) {
      navigate(`${location.pathname}?${nextSearchParams.toString()}`, { replace: true });
    }
  }, [
    compareIdsByKind.qbs,
    compareIdsByKind.teams,
    dataset,
    location.pathname,
    navigate,
    routeKind,
    searchParams,
    selectedSeason,
    compareHydrationState,
  ]);

  const handleSeasonChange = (season: number): void => {
    navigate(`${location.pathname}?season=${season}`);
  };

  const updatePageViewState = (
    kind: EntityKind,
    patch: Partial<EntityPageViewState>,
  ): void => {
    setPageViewStates((current) => ({
      ...current,
      [kind]: {
        ...current[kind],
        ...patch,
      },
    }));
  };

  const resetPageViewState = (kind: EntityKind): void => {
    setPageViewStates((current) => ({
      ...current,
      [kind]: {
        ...current[kind],
        compareIds: [],
        enabledGroups: undefined,
        query: '',
        showUnratedRows: false,
        sorting: undefined,
      },
    }));
  };

  const handleToggleCompare = (
    kind: EntityKind,
    compareIds: string[],
    entityId: string,
  ): void => {
    const nextIds = compareIds.includes(entityId)
      ? compareIds.filter((value) => value !== entityId)
      : [...compareIds, entityId].slice(0, 4);
    updatePageViewState(kind, { compareIds: nextIds });
  };

  return (
    <AppShell
      seasons={seasons}
      season={selectedSeason || seasons[0] || 0}
      onSeasonChange={handleSeasonChange}
      palette={palette}
      theme={theme}
      onPaletteChange={setPalette}
      onThemeChange={setTheme}
    >
      {error ? <section className="panel error-panel">{error}</section> : null}
      {loading || !dataset ? <section className="panel loading-panel">Loading season data...</section> : null}
      {!loading && dataset && resolvedViewStates ? (
        <Routes>
          <Route path="/" element={<Navigate to={`/teams?season=${dataset.season}`} replace />} />
          <Route
            path="/teams"
            element={
              <EntityPage
                canReset={canResetPageView(
                  'teams',
                  dataset.teams,
                  resolvedViewStates.teams,
                  compareIdsByKind.teams,
                )}
                compareIds={compareIdsByKind.teams}
                dataset={dataset}
                enabledGroups={resolvedViewStates.teams.enabledGroups}
                kind="teams"
                onEnableAllGroups={() =>
                  updatePageViewState('teams', {
                    enabledGroups: buildAllEnabledGroups(dataset.teams),
                  })
                }
                onQueryChange={(query) => updatePageViewState('teams', { query })}
                onResetView={() => resetPageViewState('teams')}
                onSortingChange={(sorting) => updatePageViewState('teams', { sorting })}
                onToggleCompare={(entityId) =>
                  handleToggleCompare('teams', compareIdsByKind.teams, entityId)
                }
                onToggleGroup={(group) =>
                  updatePageViewState('teams', {
                    enabledGroups: {
                      ...resolvedViewStates.teams.enabledGroups,
                      [group]: !resolvedViewStates.teams.enabledGroups[group],
                    },
                  })
                }
                palette={palette}
                query={resolvedViewStates.teams.query}
                season={selectedSeason || dataset.season}
                showUnratedRows={resolvedViewStates.teams.showUnratedRows}
                sorting={resolvedViewStates.teams.sorting}
                theme={theme}
              />
            }
          />
          <Route
            path="/teams/:entityId"
            element={
              <EntityDetailPage
                activeGroup={resolvedViewStates.teams.activeDetailGroup}
                dataset={dataset}
                kind="teams"
                onActiveGroupChange={(group) => updatePageViewState('teams', { activeDetailGroup: group })}
                palette={palette}
                season={selectedSeason || dataset.season}
                theme={theme}
              />
            }
          />
          <Route
            path="/qbs"
            element={
              <EntityPage
                canReset={canResetPageView(
                  'qbs',
                  dataset.qbs,
                  resolvedViewStates.qbs,
                  compareIdsByKind.qbs,
                )}
                compareIds={compareIdsByKind.qbs}
                dataset={dataset}
                enabledGroups={resolvedViewStates.qbs.enabledGroups}
                kind="qbs"
                onEnableAllGroups={() =>
                  updatePageViewState('qbs', {
                    enabledGroups: buildAllEnabledGroups(dataset.qbs),
                  })
                }
                onQueryChange={(query) => updatePageViewState('qbs', { query })}
                onResetView={() => resetPageViewState('qbs')}
                onShowUnratedRowsChange={(showUnratedRows) =>
                  updatePageViewState('qbs', { showUnratedRows })
                }
                onSortingChange={(sorting) => updatePageViewState('qbs', { sorting })}
                onToggleCompare={(entityId) =>
                  handleToggleCompare('qbs', compareIdsByKind.qbs, entityId)
                }
                onToggleGroup={(group) =>
                  updatePageViewState('qbs', {
                    enabledGroups: {
                      ...resolvedViewStates.qbs.enabledGroups,
                      [group]: !resolvedViewStates.qbs.enabledGroups[group],
                    },
                  })
                }
                palette={palette}
                query={resolvedViewStates.qbs.query}
                season={selectedSeason || dataset.season}
                showUnratedRows={resolvedViewStates.qbs.showUnratedRows}
                sorting={resolvedViewStates.qbs.sorting}
                theme={theme}
              />
            }
          />
          <Route
            path="/qbs/:entityId"
            element={
              <EntityDetailPage
                activeGroup={resolvedViewStates.qbs.activeDetailGroup}
                dataset={dataset}
                kind="qbs"
                onActiveGroupChange={(group) => updatePageViewState('qbs', { activeDetailGroup: group })}
                palette={palette}
                season={selectedSeason || dataset.season}
                theme={theme}
              />
            }
          />
          <Route path="/glossary" element={<GlossaryPage />} />
        </Routes>
      ) : null}
    </AppShell>
  );
}