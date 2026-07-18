import { useMemo, useState, type ReactElement } from 'react';

import {
  buildOpponentBreakdown,
  buildWeeklyHighlights,
  enrichGameLogsWithOpponentRatings,
  orderedExisting,
} from '../detailAnalytics';
import {
  buildGameOverviewUrl,
  compareDetailCellValues,
  formatDetailCellValue,
} from '../detailUi';
import { formatValue, humanizeGroup } from '../format';
import { getColumnSection, getGroupDescription, getMetricMetadata, getMetricTooltip } from '../metricMetadata';
import { buildColumnStats, getHeatCellStyle } from '../tableState';
import { TooltipLabel } from './TooltipLabel';
import type {
  EntityConfig,
  PaletteMode,
  PrimaryView,
  RowValue,
  TablePayload,
  ThemeMode,
} from '../types';
import {
  buildGameLogColumnSelection,
  deriveLegacyDetailSurfaceId,
  type ResolvedEntityViewState,
} from '../viewModel';
import { getFullTeamName } from '../entityConfig';
import { ViewControls } from './ViewControls';

interface EntityDetailProps {
  canReset: boolean;
  config: EntityConfig;
  gameLogError?: string | null;
  gameLogs?: TablePayload | null;
  gameLogsLoading?: boolean;
  onResetView: () => void;
  onSelectTeamCategory: (category: string) => void;
  onSelectView: (view: PrimaryView) => void;
  onToggleSubcategory: (subcategory: string) => void;
  opponentRatingsTable: TablePayload;
  palette: PaletteMode;
  row: Record<string, RowValue>;
  season: number;
  table: TablePayload;
  theme: ThemeMode;
  viewState: ResolvedEntityViewState;
}

const TEAM_RATING_ORDER = ['SaCR', 'sos', 'SRS', 'SaOvR', 'SaOR', 'SaDR'];
const QB_RATING_ORDER = ['QSaCR', 'QSaOR', 'QSoS', 'faced_opp_SaCR', 'QOutcome', 'QRaw'];

function buildRatingsColumns(kind: EntityConfig['kind'], columns: string[]): string[] {
  const preferredOrder = kind === 'teams' ? TEAM_RATING_ORDER : QB_RATING_ORDER;
  const orderedColumns = orderedExisting(columns, preferredOrder);
  const remainingColumns = columns.filter((column) => !orderedColumns.includes(column));
  return [...orderedColumns, ...remainingColumns];
}

function bucketColumnsByRegistry(
  kind: EntityConfig['kind'],
  columns: string[],
): Array<{ title: string; columns: string[] }> | null {
  const entity = kind === 'teams' ? 'team' : 'qb';
  const sections = new Map<string, { title: string; rank: number; columns: string[] }>();
  for (const column of columns) {
    const section = getColumnSection(entity, column);
    if (!section) {
      return null;
    }
    const current = sections.get(section.title) ?? { ...section, columns: [] };
    current.columns.push(column);
    sections.set(section.title, current);
  }
  return Array.from(sections.values())
    .sort((left, right) => left.rank - right.rank)
    .map(({ title, columns: sectionColumns }) => ({ title, columns: sectionColumns }));
}

function bucketColumns(
  kind: EntityConfig['kind'],
  isRatingsView: boolean,
  columns: string[],
): Array<{ title: string; columns: string[] }> {
  if (isRatingsView) {
    return [{ title: 'Season Ratings', columns: buildRatingsColumns(kind, columns) }];
  }

  // Section by the metric registry's category taxonomy when every column has
  // hydrated registry metadata; fall back to the legacy heuristics otherwise.
  const registrySections = bucketColumnsByRegistry(kind, columns);
  if (registrySections) {
    return registrySections;
  }
  return [{ title: 'Metrics', columns }];
}

export function EntityDetail({
  canReset,
  config,
  gameLogError,
  gameLogs,
  gameLogsLoading,
  onResetView,
  onSelectTeamCategory,
  onSelectView,
  onToggleSubcategory,
  opponentRatingsTable,
  palette,
  row,
  season,
  table,
  theme,
  viewState,
}: EntityDetailProps): ReactElement {
  const rawLabel = String(row[config.labelKey] ?? row[config.identityKey] ?? '');
  const label = config.kind === 'teams'
    ? getFullTeamName(rawLabel)
    : `${rawLabel} - ${getFullTeamName(String(row.team ?? ''))}`;
  const [opponentBreakdownSort, setOpponentBreakdownSort] = useState<{ column: string; desc: boolean }>({
    column: 'opponent_team',
    desc: false,
  });
  const metricColumns = useMemo(
    () => table.visible_columns.filter((column) => !config.identityColumns.includes(column)),
    [config.identityColumns, table.visible_columns],
  );
  const groupedSections = useMemo(
    () => bucketColumns(config.kind, viewState.primaryView === 'ratings', metricColumns),
    [config.kind, metricColumns, viewState.primaryView],
  );
  const enrichedGameLogs = useMemo(
    () => (gameLogs ? enrichGameLogsWithOpponentRatings(gameLogs, opponentRatingsTable) : null),
    [gameLogs, opponentRatingsTable],
  );
  const weeklyHighlights = useMemo(
    () => (enrichedGameLogs ? buildWeeklyHighlights(config.kind, row, enrichedGameLogs) : []),
    [config.kind, enrichedGameLogs, row],
  );
  const gameLogSelection = useMemo(
    () => (enrichedGameLogs ? buildGameLogColumnSelection(config.kind, enrichedGameLogs, viewState) : null),
    [config.kind, enrichedGameLogs, viewState],
  );
  const gameLogColumns = useMemo(
    () => gameLogSelection?.columns ?? [],
    [gameLogSelection],
  );
  const legacySurfaceId = useMemo(
    () => deriveLegacyDetailSurfaceId(config.kind, viewState),
    [config.kind, viewState],
  );
  const opponentBreakdown = useMemo(
    () =>
      enrichedGameLogs
        ? buildOpponentBreakdown(config.kind, row, enrichedGameLogs, legacySurfaceId)
        : null,
    [config.kind, enrichedGameLogs, legacySurfaceId, row],
  );
  const resolvedOpponentBreakdownSort = useMemo(() => {
    if (!opponentBreakdown) {
      return opponentBreakdownSort;
    }

    const availableColumns = new Set(opponentBreakdown.columns.map((column) => column.id));
    return availableColumns.has(opponentBreakdownSort.column)
      ? opponentBreakdownSort
      : { column: 'opponent_team', desc: false };
  }, [opponentBreakdown, opponentBreakdownSort]);
  const sortedOpponentBreakdownRows = useMemo(() => {
    if (!opponentBreakdown) {
      return [];
    }

    const rows = [...opponentBreakdown.rows];
    rows.sort((left, right) => {
      const comparison = compareDetailCellValues(
        resolvedOpponentBreakdownSort.column,
        left[resolvedOpponentBreakdownSort.column] ?? null,
        right[resolvedOpponentBreakdownSort.column] ?? null,
      );
      return resolvedOpponentBreakdownSort.desc ? -comparison : comparison;
    });
    return rows;
  }, [opponentBreakdown, resolvedOpponentBreakdownSort]);
  const opponentBreakdownColumnStats = useMemo(
    () =>
      opponentBreakdown
        ? buildColumnStats(sortedOpponentBreakdownRows, opponentBreakdown.columns.map((column) => column.id))
        : {},
    [opponentBreakdown, sortedOpponentBreakdownRows],
  );
  const weeklyHasOpponentContext = useMemo(
    () => (gameLogSelection?.contextColumns.length ?? 0) > 0,
    [gameLogSelection],
  );

  const detailHeaderLabel = (column: string): string => {
    if (column === 'win_value') {
      return 'Outcome';
    }
    if (column === 'turnover_margin') {
      return 'T/O Margin';
    }
    return getMetricMetadata(column).label;
  };

  return (
    <>
      <section className="panel detail-header-pane">
        <div className="detail-hero">
          <p className="eyebrow">{config.singularLabel} Detail</p>
          <h2>{label}</h2>
        </div>
        <ViewControls
          canReset={canReset}
          kind={config.kind}
          onReset={onResetView}
          onSelectTeamCategory={onSelectTeamCategory}
          onSelectView={onSelectView}
          onToggleSubcategory={onToggleSubcategory}
          state={viewState}
        />
      </section>

      <section className="panel detail-section">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Current View</p>
            <h2>{humanizeGroup(viewState.primaryView)}</h2>
            <p className="group-description">{getGroupDescription(config.kind, viewState.primaryView)}</p>
          </div>
          <div className="panel-stats">
            <span>{metricColumns.length} columns</span>
          </div>
        </div>

        <div className="detail-subsection-stack">
          {groupedSections.map((section) => (
            <div key={section.title} className="detail-subsection">
              <div className="detail-subsection-header">
                {viewState.primaryView !== 'ratings' ? <p className="eyebrow">Category</p> : null}
                <h3>{section.title}</h3>
              </div>
              <dl className={viewState.primaryView === 'ratings' ? 'metric-grid compact-metric-grid' : 'metric-grid'}>
                {section.columns.map((column) => (
                  <div key={column} className={viewState.primaryView === 'ratings' ? 'metric-cell compact-metric-cell' : 'metric-cell'}>
                    <dt>
                      <TooltipLabel
                        label={detailHeaderLabel(column)}
                        tooltip={getMetricTooltip(column)}
                      />
                    </dt>
                    <dd>{formatValue(row[column] ?? null)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </section>

      <section className="panel detail-section">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Weekly Log</p>
            <h2>Game-by-Game Details</h2>
            <p className="group-description">
              See every game for this {season} season in one place, with base result context plus
              the currently selected surface and matched opponent season context side by side.
            </p>
          </div>
          {gameLogs ? (
            <div className="panel-stats">
              <span>{gameLogs.rows.length} games</span>
              <span>{gameLogColumns.length} columns</span>
            </div>
          ) : null}
        </div>

        {gameLogsLoading ? <div className="compare-selected-wrap">Loading weekly logs...</div> : null}
        {!gameLogsLoading && gameLogError ? (
          <div className="compare-selected-wrap">{gameLogError}</div>
        ) : null}
        {!gameLogsLoading && !gameLogError && enrichedGameLogs && gameLogSelection ? (
          <>
            {weeklyHighlights.length > 0 ? (
              <div className="weekly-highlight-grid overview-grid">
                {weeklyHighlights.map((highlight) => (
                  <article key={highlight.eyebrow} className="panel stat-card weekly-highlight-card">
                    <p className="eyebrow">{highlight.eyebrow}</p>
                    <h3>{highlight.value}</h3>
                    <strong>{highlight.title}</strong>
                    <p>{highlight.context}</p>
                  </article>
                ))}
              </div>
            ) : null}
            <div className="compare-selected-wrap weekly-log-controls">
              <p className="group-description">
                {humanizeGroup(gameLogSelection.effectiveView)} surface keyed to the current page
                controls, with base game context pinned first.
              </p>
              {weeklyHasOpponentContext ? (
                <p className="group-description">
                  Opponent rating columns here describe the full-season strength of that opponent.
                  They add schedule context to each row, but they are not single-game grades.
                </p>
              ) : null}
            </div>
            <div className="table-shell">
              <table>
                <thead>
                  <tr>
                    {gameLogColumns.map((column) => (
                      <th key={column}>
                        <TooltipLabel
                          label={detailHeaderLabel(column)}
                          tooltip={getMetricTooltip(column)}
                        />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {enrichedGameLogs.rows.map((gameRow, index) => (
                    <tr key={`${String(gameRow.game_id ?? index)}-${index}`}>
                      {gameLogColumns.map((column) => {
                        const value = gameRow[column] ?? null;
                        return (
                          <td key={column}>
                            {column === 'game_id' && typeof value === 'string' ? (
                              <a
                                className="detail-link"
                                href={buildGameOverviewUrl(value)}
                                rel="noreferrer"
                                target="_blank"
                              >
                                {value}
                              </a>
                            ) : (
                              formatDetailCellValue(column, value)
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </section>

      {!gameLogsLoading && !gameLogError && opponentBreakdown ? (
        <section className="panel detail-section">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Opponent Breakdown</p>
              <h2>Unique Opponents</h2>
              <p className="group-description">{opponentBreakdown.description}</p>
            </div>
            <div className="panel-stats">
              <span>{opponentBreakdown.rows.length} opponents</span>
              <span>{opponentBreakdown.columns.length} columns</span>
            </div>
          </div>
          <div className="compare-selected-wrap weekly-log-controls">
            <p className="group-description">
              This opponent ledger now follows the same top-of-page control state as the weekly
              table instead of maintaining its own local surface toggle.
            </p>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  {opponentBreakdown.columns.map((column) => (
                    <th key={column.id}>
                      <button
                        type="button"
                        className="sort-button"
                        onClick={() =>
                          setOpponentBreakdownSort((current) =>
                            current.column === column.id
                              ? { column: column.id, desc: !current.desc }
                              : column.id === 'opp_schedule_bucket'
                                ? { column: column.id, desc: true }
                              : {
                                  column: column.id,
                                  desc:
                                    typeof sortedOpponentBreakdownRows[0]?.[column.id] === 'string'
                                      ? false
                                      : getMetricMetadata(column.id).polarity !== 'lower',
                                },
                          )
                        }
                      >
                        <TooltipLabel
                          label={column.label ?? getMetricMetadata(column.id).label}
                          tooltip={column.tooltip ?? getMetricTooltip(column.id)}
                        />
                        <span className="sort-indicator">
                          {resolvedOpponentBreakdownSort.column === column.id
                            ? resolvedOpponentBreakdownSort.desc
                              ? '↓'
                              : '↑'
                            : '·'}
                        </span>
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedOpponentBreakdownRows.map((breakdownRow) => (
                  <tr key={String(breakdownRow.opponent_team ?? 'unknown-opponent')}>
                    {opponentBreakdown.columns.map((column) => {
                      const value = breakdownRow[column.id] ?? null;
                      const heatStyle = getHeatCellStyle(
                        column.id,
                        value,
                        opponentBreakdownColumnStats,
                        theme,
                        palette,
                      );

                      return (
                        <td key={column.id} style={heatStyle}>
                          {formatDetailCellValue(column.id, value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </>
  );
}