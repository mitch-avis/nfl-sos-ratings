import { useMemo, useState, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { formatValue, humanizeGroup } from '../format';
import { getGroupDescription, getMetricMetadata, getMetricTooltip } from '../metricMetadata';
import { TooltipLabel } from './TooltipLabel';
import type { EntityConfig, RowValue, TablePayload } from '../types';

interface EntityDetailProps {
  activeGroup?: string;
  basePath: string;
  config: EntityConfig;
  gameLogError?: string | null;
  gameLogs?: TablePayload | null;
  gameLogsLoading?: boolean;
  onActiveGroupChange?: (group: string) => void;
  opponentRatingsTable: TablePayload;
  row: Record<string, RowValue>;
  season: number;
  table: TablePayload;
}

interface GameLogGroup {
  id: string;
  label: string;
  description: string;
  columns: string[];
}

const TEAM_RATING_ORDER = ['SaCR', 'SRS', 'SaOvR', 'SaOR', 'SaDR'];
const QB_RATING_ORDER = ['QSaCR', 'QSaOR', 'QOutcome', 'QRaw', 'QSoS'];
const OPPONENT_RATING_COLUMNS = ['opp_SaCR', 'opp_SRS', 'opp_SaOvR', 'opp_SaOR', 'opp_SaDR'];
const TEAM_RESULT_COLUMNS = ['points_for', 'points_allowed', 'point_margin', 'win_value', 'turnover_margin'];
const QB_RESULT_COLUMNS = [
  'points_for',
  'points_allowed',
  'point_margin',
  'win_value',
  'turnover_margin',
  'qb_fourth_quarter_comeback',
  'qb_game_winning_drive',
];
const QB_EFFICIENCY_COLUMNS = [
  'qb_completion_percentage_above_expectation',
  'qb_passer_rating',
  'qb_any_a',
];

function orderedExisting(columns: string[], preferredColumns: string[]): string[] {
  const available = new Set(columns);
  return preferredColumns.filter((column) => available.has(column));
}

function compareValues(left: RowValue, right: RowValue): number {
  if (left === right) {
    return 0;
  }
  if (left === null) {
    return 1;
  }
  if (right === null) {
    return -1;
  }
  if (typeof left === 'number' && typeof right === 'number') {
    return left - right;
  }
  if (typeof left === 'boolean' && typeof right === 'boolean') {
    return Number(left) - Number(right);
  }
  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' });
}

function buildRatingsColumns(kind: EntityConfig['kind'], columns: string[]): string[] {
  const preferredOrder = kind === 'teams' ? TEAM_RATING_ORDER : QB_RATING_ORDER;
  const orderedColumns = orderedExisting(columns, preferredOrder);
  const remainingColumns = columns.filter((column) => !orderedColumns.includes(column));
  return [...orderedColumns, ...remainingColumns];
}

function enrichGameLogsWithOpponentRatings(
  gameLogs: TablePayload,
  opponentRatingsTable: TablePayload,
): TablePayload {
  const ratingColumns = OPPONENT_RATING_COLUMNS.map((column) => column.replace(/^opp_/, '')).filter(
    (column) => opponentRatingsTable.visible_columns.includes(column),
  );
  if (ratingColumns.length === 0) {
    return gameLogs;
  }

  const ratingsByTeam = new Map(
    opponentRatingsTable.rows.map((ratingRow) => [String(ratingRow.team ?? ''), ratingRow]),
  );
  const rows = gameLogs.rows.map((gameRow) => {
    const opponentTeam = String(gameRow.opponent_team ?? '');
    const ratingRow = ratingsByTeam.get(opponentTeam);
    if (!ratingRow) {
      return gameRow;
    }
    return {
      ...gameRow,
      ...Object.fromEntries(
        ratingColumns.map((column) => [`opp_${column}`, ratingRow[column] ?? null]),
      ),
    };
  });
  const visibleColumns = [...new Set([...gameLogs.visible_columns, ...ratingColumns.map((column) => `opp_${column}`)])];
  return { ...gameLogs, rows, visible_columns: visibleColumns };
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

function buildGameLogGroups(kind: EntityConfig['kind'], gameLogs: TablePayload): GameLogGroup[] {
  const identityColumns = orderedExisting(gameLogs.visible_columns, ['week', 'opponent_team', 'game_id']);
  const nonIdentityColumns = gameLogs.visible_columns.filter(
    (column) => !identityColumns.includes(column) && column !== 'qb_id' && column !== 'team' && column !== 'qb_name',
  );
  const opponentRatings = orderedExisting(nonIdentityColumns, OPPONENT_RATING_COLUMNS);

  if (kind === 'teams') {
    const resultColumns = orderedExisting(nonIdentityColumns, TEAM_RESULT_COLUMNS);
    const perSnapRates = nonIdentityColumns.filter(
      (column) => column.endsWith('_per_offensive_snap') || column.endsWith('_per_defensive_snap'),
    );
    const defenseColumns = nonIdentityColumns.filter(
      (column) =>
        !resultColumns.includes(column)
        && !perSnapRates.includes(column)
        && !opponentRatings.includes(column)
        && isTeamDefenseMetric(column),
    );
    const offenseColumns = nonIdentityColumns.filter(
      (column) =>
        !resultColumns.includes(column)
        && !perSnapRates.includes(column)
        && !opponentRatings.includes(column)
        && !defenseColumns.includes(column)
        && !isTeamOutcomeMetric(column),
    );

    return [
      {
        id: 'results',
        label: 'Results',
        description: 'Single-game outcomes and scoreboard context for each opponent.',
        columns: resultColumns,
      },
      {
        id: 'offense',
        label: 'Offense',
        description: 'Single-game team offensive production and efficiency stats.',
        columns: offenseColumns,
      },
      {
        id: 'defense',
        label: 'Defense',
        description: 'Single-game team defensive production and allowed-stat surface.',
        columns: defenseColumns,
      },
      {
        id: 'per_snap_rates',
        label: 'Per-Snap Rates',
        description: 'Single-game snap-normalized rates for faster cross-game comparisons.',
        columns: perSnapRates,
      },
      {
        id: 'opponent_ratings',
        label: 'Opponent Ratings',
        description: 'Season-long opponent rating context joined onto each weekly row.',
        columns: opponentRatings,
      },
      {
        id: 'all',
        label: 'All Stats',
        description: 'Full game-log row with every available single-game stat and rating context column.',
        columns: nonIdentityColumns,
      },
    ].filter((group) => group.columns.length > 0);
  }

  const resultColumns = orderedExisting(nonIdentityColumns, QB_RESULT_COLUMNS);
  const perDropbackRates = nonIdentityColumns.filter(
    (column) => column.endsWith('_per_dropback') || column === 'qb_td_int_margin_rate' || column === 'qb_sack_rate',
  );
  const efficiencyColumns = orderedExisting(nonIdentityColumns, QB_EFFICIENCY_COLUMNS).filter(
    (column) => !perDropbackRates.includes(column),
  );
  const volumeColumns = nonIdentityColumns.filter(
    (column) =>
      !resultColumns.includes(column)
      && !perDropbackRates.includes(column)
      && !efficiencyColumns.includes(column)
      && !opponentRatings.includes(column),
  );

  return [
    {
      id: 'results',
      label: 'Results',
      description: 'Single-game outcomes, late-game events, and scoreboard context.',
      columns: resultColumns,
    },
    {
      id: 'volume',
      label: 'Volume',
      description: 'Single-game QB volume, dropback, and raw passing totals.',
      columns: volumeColumns,
    },
    {
      id: 'efficiency',
      label: 'Efficiency',
      description: 'Single-game passing efficiency metrics that are not already dropback-normalized.',
      columns: efficiencyColumns,
    },
    {
      id: 'per_dropback_rates',
      label: 'Per-Dropback Rates',
      description: 'Single-game dropback-normalized passing rates.',
      columns: perDropbackRates,
    },
    {
      id: 'opponent_ratings',
      label: 'Opponent Ratings',
      description: 'Season-long opponent rating context joined onto each weekly QB row.',
      columns: opponentRatings,
    },
    {
      id: 'all',
      label: 'All Stats',
      description: 'Full game-log row with every available single-game stat and rating context column.',
      columns: nonIdentityColumns,
    },
  ].filter((group) => group.columns.length > 0);
}

function buildOpponentBreakdown(
  gameLogs: TablePayload,
  gameLogColumns: string[],
): { columns: string[]; rows: Array<Record<string, RowValue>> } {
  const summaryColumns = gameLogColumns.filter(
    (column) => !['week', 'game_id', 'opponent_team', 'games', 'weeks'].includes(column),
  );
  const groupedRows = new Map<string, Array<Record<string, RowValue>>>();

  for (const row of gameLogs.rows) {
    const opponentTeam = String(row.opponent_team ?? 'Unknown');
    const currentRows = groupedRows.get(opponentTeam) ?? [];
    currentRows.push(row);
    groupedRows.set(opponentTeam, currentRows);
  }

  const rows = Array.from(groupedRows.entries()).map(([opponentTeam, rowsForOpponent]) => {
    const numericSummary = Object.fromEntries(
      summaryColumns.map((column) => {
        const numericValues = rowsForOpponent
          .map((row) => row[column])
          .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));

        if (numericValues.length === rowsForOpponent.length && numericValues.length > 0) {
          return [column, numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length];
        }

        const firstValue = rowsForOpponent.find((row) => row[column] !== null)?.[column] ?? null;
        return [column, firstValue];
      }),
    );

    return {
      games: rowsForOpponent.length,
      opponent_team: opponentTeam,
      weeks: rowsForOpponent
        .map((row) => row.week)
        .filter((week): week is number => typeof week === 'number')
        .sort((left, right) => left - right)
        .join(', '),
      ...numericSummary,
    };
  });

  rows.sort((left, right) => String(left.opponent_team).localeCompare(String(right.opponent_team)));
  return {
    columns: ['opponent_team', 'games', 'weeks', ...summaryColumns],
    rows,
  };
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
  row,
  season,
  table,
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
    () => (enrichedGameLogs && gameLogColumns.length > 0 ? buildOpponentBreakdown(enrichedGameLogs, gameLogColumns) : null),
    [enrichedGameLogs, gameLogColumns],
  );
  const sortedOpponentBreakdownRows = useMemo(() => {
    if (!opponentBreakdown) {
      return [];
    }

    const rows = [...opponentBreakdown.rows];
    rows.sort((left, right) => {
      const comparison = compareValues(
        left[opponentBreakdownSort.column] ?? null,
        right[opponentBreakdownSort.column] ?? null,
      );
      return opponentBreakdownSort.desc ? -comparison : comparison;
    });
    return rows;
  }, [opponentBreakdown, opponentBreakdownSort]);

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
            <h2>Game-by-Game Surface</h2>
            <p className="group-description">
              Additive game-log contract rows for the current {season} season, intended as the
              first detail-page enrichment layer before charts and grouped opponent breakdowns.
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
                      {gameLogColumns.map((column) => (
                        <td key={column}>{formatValue(gameRow[column] ?? null)}</td>
                      ))}
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
              <h2>Grouped Opponents</h2>
              <p className="group-description">
                Repeat opponents are grouped into one summary row so division rivals do not clutter
                the detail page with duplicated weekly context.
              </p>
            </div>
            <div className="panel-stats">
              <span>{opponentBreakdown.rows.length} opponents</span>
              <span>{opponentBreakdown.columns.length} columns</span>
            </div>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  {opponentBreakdown.columns.map((column) => (
                    <th key={column}>
                      <button
                        type="button"
                        className="sort-button"
                        onClick={() =>
                          setOpponentBreakdownSort((current) =>
                            current.column === column
                              ? { column, desc: !current.desc }
                              : {
                                  column,
                                  desc:
                                    typeof sortedOpponentBreakdownRows[0]?.[column] === 'string'
                                      ? false
                                      : getMetricMetadata(column).polarity !== 'lower',
                                },
                          )
                        }
                      >
                        <TooltipLabel
                          label={getMetricMetadata(column).label}
                          tooltip={getMetricTooltip(column)}
                        />
                        <span className="sort-indicator">
                          {opponentBreakdownSort.column === column
                            ? opponentBreakdownSort.desc
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
                    {opponentBreakdown.columns.map((column) => (
                      <td key={column}>{formatValue(breakdownRow[column] ?? null)}</td>
                    ))}
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