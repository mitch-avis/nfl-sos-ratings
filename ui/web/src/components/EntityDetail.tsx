import { useMemo, useState, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import {
  buildGameLogGroups,
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
import { getGroupDescription, getMetricMetadata, getMetricTooltip } from '../metricMetadata';
import { buildColumnStats, getHeatCellStyle } from '../tableState';
import { TooltipLabel } from './TooltipLabel';
import type { EntityConfig, PaletteMode, RowValue, TablePayload, ThemeMode } from '../types';

interface EntityDetailProps {
  activeGroup?: string;
  basePath: string;
  config: EntityConfig;
  gameLogError?: string | null;
  gameLogs?: TablePayload | null;
  gameLogsLoading?: boolean;
  onActiveGroupChange?: (group: string) => void;
  opponentRatingsTable: TablePayload;
  palette: PaletteMode;
  row: Record<string, RowValue>;
  season: number;
  table: TablePayload;
  theme: ThemeMode;
}

const TEAM_RATING_ORDER = ['SaCR', 'SRS', 'SaOvR', 'SaOR', 'SaDR'];
const QB_RATING_ORDER = ['QSaCR', 'QSaOR', 'QOutcome', 'QRaw', 'QSoS'];

function buildRatingsColumns(kind: EntityConfig['kind'], columns: string[]): string[] {
  const preferredOrder = kind === 'teams' ? TEAM_RATING_ORDER : QB_RATING_ORDER;
  const orderedColumns = orderedExisting(columns, preferredOrder);
  const remainingColumns = columns.filter((column) => !orderedColumns.includes(column));
  return [...orderedColumns, ...remainingColumns];
}

function isTeamDefenseMetric(column: string): boolean {
  return (
    column === 'points_allowed'
    || column === 'defensive_snaps'
    || column.startsWith('def_')
    || column.includes('allowed')
  );
}

function isTeamOutcomeMetric(column: string): boolean {
  return ['games_played', 'games', 'point_margin', 'win_value', 'turnover_margin'].includes(column);
}

function bucketColumns(kind: EntityConfig['kind'], group: string, columns: string[]): Array<{ title: string; columns: string[] }> {
  const buckets = new Map<string, string[]>();

  if (group === 'ratings') {
    return [{ title: 'Season Ratings', columns: buildRatingsColumns(kind, columns) }];
  }

  const push = (title: string, column: string): void => {
    const current = buckets.get(title) ?? [];
    current.push(column);
    buckets.set(title, current);
  };

  for (const column of columns) {
    if (kind === 'teams' && group === 'opponent_context') {
      const baseColumn = column.replace(/^opp_/, '');
      if (isTeamDefenseMetric(baseColumn)) {
        push('Opponent Defense', column);
      } else if (isTeamOutcomeMetric(baseColumn)) {
        push('Opponent Outcomes', column);
      } else {
        push('Opponent Offense', column);
      }
      continue;
    }

    if (kind === 'teams') {
      if (column.startsWith('opp_')) {
        push('Opponent Context', column);
      } else if (
        column.includes('allowed')
        || column.startsWith('def_')
        || column.includes('defensive_snap')
        || column === 'points_allowed'
      ) {
        push('Defense', column);
      } else if (column === 'games_played' || column === 'point_margin' || column === 'win_value' || column === 'turnover_margin') {
        push('Overall / Outcomes', column);
      } else {
        push('Offense', column);
      }
      continue;
    }

    if (group === 'opponent_context') {
      if (column.startsWith('qopp_def_') || column === 'qopp_points_allowed') {
        push('Opponent Defense', column);
      } else if (column.startsWith('qopp_qb_')) {
        push('Opponent Allowed QB Rates', column);
      } else {
        push('Other Context', column);
      }
      continue;
    }

    if (group === 'per_dropback_rates') {
      if (column.includes('sack')) {
        push('Negative Plays', column);
      } else {
        push('Passing Efficiency', column);
      }
      continue;
    }

    if (group === 'per_game_rates' || group === 'raw_totals') {
      if (column.includes('wins') || column.includes('losses') || column.includes('ties') || column.includes('comeback') || column.includes('winning_drive')) {
        push('Results and Outcomes', column);
      } else if (column.includes('attempt') || column.includes('completion') || column.includes('dropback') || column.includes('snap')) {
        push('Volume', column);
      } else if (column.includes('interception') || column.includes('sack')) {
        push('Negative Plays', column);
      } else {
        push('Passing Production', column);
      }
      continue;
    }

    push('Metrics', column);
  }

  return Array.from(buckets.entries()).map(([title, groupedColumns]) => ({
    title,
    columns: groupedColumns,
  }));
}

export function EntityDetail({
  activeGroup,
  basePath,
  config,
  gameLogError,
  gameLogs,
  gameLogsLoading,
  onActiveGroupChange,
  opponentRatingsTable,
  palette,
  row,
  season,
  table,
  theme,
}: EntityDetailProps): ReactElement {
  const label = String(row[config.labelKey] ?? row[config.identityKey] ?? '');
  const availableGroups = config.detailGroups.filter((group) => (table.column_groups[group] ?? []).length > 0);
  const [activeGameLogGroup, setActiveGameLogGroup] = useState<string>('results');
  const [opponentBreakdownSort, setOpponentBreakdownSort] = useState<{ column: string; desc: boolean }>({
    column: 'opponent_team',
    desc: false,
  });
  const resolvedActiveGroup = availableGroups.includes(activeGroup ?? '')
    ? (activeGroup ?? 'ratings')
    : (availableGroups[0] ?? 'ratings');
  const groupedSections = useMemo(
    () => bucketColumns(config.kind, resolvedActiveGroup, table.column_groups[resolvedActiveGroup] ?? []),
    [config.kind, resolvedActiveGroup, table.column_groups],
  );
  const enrichedGameLogs = useMemo(
    () => (gameLogs ? enrichGameLogsWithOpponentRatings(gameLogs, opponentRatingsTable) : null),
    [gameLogs, opponentRatingsTable],
  );
  const gameLogGroups = useMemo(
    () => (enrichedGameLogs ? buildGameLogGroups(config.kind, enrichedGameLogs) : []),
    [config.kind, enrichedGameLogs],
  );
  const weeklyHighlights = useMemo(
    () => (enrichedGameLogs ? buildWeeklyHighlights(config.kind, row, enrichedGameLogs) : []),
    [config.kind, enrichedGameLogs, row],
  );
  const resolvedGameLogGroup = gameLogGroups.find((group) => group.id === activeGameLogGroup) ?? gameLogGroups[0] ?? null;
  const gameLogIdentityColumns = useMemo(
    () => (enrichedGameLogs ? orderedExisting(enrichedGameLogs.visible_columns, ['week', 'opponent_team', 'game_id']) : []),
    [enrichedGameLogs],
  );
  const gameLogColumns = useMemo(
    () => (resolvedGameLogGroup ? [...gameLogIdentityColumns, ...resolvedGameLogGroup.columns] : []),
    [gameLogIdentityColumns, resolvedGameLogGroup],
  );
  const opponentBreakdown = useMemo(
    () =>
      enrichedGameLogs && resolvedGameLogGroup
        ? buildOpponentBreakdown(config.kind, row, enrichedGameLogs, resolvedGameLogGroup.id)
        : null,
    [config.kind, enrichedGameLogs, resolvedGameLogGroup, row],
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
    () => gameLogColumns.some((column) => column.startsWith('opp_')),
    [gameLogColumns],
  );

  return (
    <>
      <section className="hero panel detail-hero">
        <p className="eyebrow">{config.singularLabel} Detail</p>
        <h2>{label}</h2>
        <p>
          Direct view into the current {season} contract row, grouped by metric family so the
          rating outputs stay adjacent to their supporting stat surfaces.
        </p>
        <div className="hero-actions">
          <Link className="detail-link" to={`${basePath}?season=${season}`}>
            Back to index
          </Link>
          <Link className="detail-link" to={`/glossary?season=${season}`}>
            Open glossary
          </Link>
        </div>
      </section>

      <section className="overview-grid">
        {config.compareColumns.map((column) => {
          const metadata = getMetricMetadata(column);
          return (
            <article key={column} className="panel stat-card metric-spotlight">
              <p className="eyebrow">{metadata.label}</p>
              <h3>{formatValue(row[column] ?? null)}</h3>
            </article>
          );
        })}
      </section>

      <section className="panel guidance-panel">
        <div className="group-toggle-wrap detail-group-toggle-wrap">
          {availableGroups.map((group) => (
            <button
              key={group}
              type="button"
              className={resolvedActiveGroup === group ? 'group-toggle active' : 'group-toggle'}
              onClick={() => onActiveGroupChange?.(group)}
            >
              <span>{humanizeGroup(group)}</span>
              <strong>{(table.column_groups[group] ?? []).length}</strong>
            </button>
          ))}
        </div>
      </section>

      <section className="panel detail-section">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Metric Family</p>
            <h2>{humanizeGroup(resolvedActiveGroup)}</h2>
            <p className="group-description">{getGroupDescription(config.kind, resolvedActiveGroup)}</p>
          </div>
          <div className="panel-stats">
            <span>{(table.column_groups[resolvedActiveGroup] ?? []).length} columns</span>
          </div>
        </div>

        <div className="detail-subsection-stack">
          {groupedSections.map((section) => (
            <div key={section.title} className="detail-subsection">
              <div className="detail-subsection-header">
                {resolvedActiveGroup !== 'ratings' ? <p className="eyebrow">Category</p> : null}
                <h3>{section.title}</h3>
              </div>
              <dl className={resolvedActiveGroup === 'ratings' ? 'metric-grid compact-metric-grid' : 'metric-grid'}>
                {section.columns.map((column) => (
                  <div key={column} className={resolvedActiveGroup === 'ratings' ? 'metric-cell compact-metric-cell' : 'metric-cell'}>
                    <dt>
                      <TooltipLabel
                        label={getMetricMetadata(column).label}
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
              See every game for this {season} season in one place, with the selected stat family,
              result context, and season-long opponent ratings kept side by side.
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
        {!gameLogsLoading && !gameLogError && enrichedGameLogs && resolvedGameLogGroup ? (
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
              <div className="group-toggle-wrap detail-group-toggle-wrap">
                {gameLogGroups.map((group) => (
                  <button
                    key={group.id}
                    type="button"
                    className={resolvedGameLogGroup.id === group.id ? 'group-toggle active' : 'group-toggle'}
                    onClick={() => setActiveGameLogGroup(group.id)}
                  >
                    <span>{group.label}</span>
                    <strong>{group.columns.length}</strong>
                  </button>
                ))}
              </div>
              <p className="group-description">{resolvedGameLogGroup.description}</p>
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
                          label={getMetricMetadata(column).label}
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
            <div className="group-toggle-wrap detail-group-toggle-wrap">
              {gameLogGroups.map((group) => (
                <button
                  key={`breakdown-${group.id}`}
                  type="button"
                  className={resolvedGameLogGroup.id === group.id ? 'group-toggle active' : 'group-toggle'}
                  onClick={() => setActiveGameLogGroup(group.id)}
                >
                  <span>{group.label}</span>
                  <strong>{group.columns.length}</strong>
                </button>
              ))}
            </div>
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