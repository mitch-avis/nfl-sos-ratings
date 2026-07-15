import { formatValue } from './format.js';
import { getMetricMetadata } from './metricMetadata.js';
import type { EntityKind, RowValue, TablePayload } from './types.js';

type DataRow = Record<string, RowValue>;

export interface GameLogGroup {
  id: string;
  label: string;
  description: string;
  columns: string[];
}

export interface WeeklyHighlight {
  context: string;
  eyebrow: string;
  title: string;
  value: string;
}

export interface OpponentBreakdownColumn {
  id: string;
  label?: string;
  tooltip?: string;
}

export interface OpponentBreakdownTable {
  columns: OpponentBreakdownColumn[];
  description: string;
  rows: DataRow[];
}

interface NumericGameRow {
  row: DataRow;
  value: number;
}

interface OpponentLedgerSpec {
  description: string;
  difficultyMetric?: string;
  groupColumns: string[];
  deltaMetric?: string;
}

const OPPONENT_RATING_COLUMNS = ['opp_SaCR', 'opp_SRS', 'opp_SaOvR', 'opp_SaOR', 'opp_SaDR'];
const TEAM_RESULT_COLUMNS = [
  'points_for',
  'points_allowed',
  'point_margin',
  'win_value',
  'turnover_margin',
];
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
const TEAM_TREND_COLUMNS = ['points_per_offensive_snap', 'passing_epa', 'point_margin'];
const QB_TREND_COLUMNS = ['qb_epa_per_dropback', 'qb_any_a', 'qb_passer_rating'];

export function orderedExisting(columns: string[], preferredColumns: string[]): string[] {
  const available = new Set(columns);
  return preferredColumns.filter((column) => available.has(column));
}

function uniqueColumns(columns: Array<string | undefined>): string[] {
  return Array.from(new Set(columns.filter((column): column is string => Boolean(column))));
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function roundNumber(value: number): number {
  return Number(value.toFixed(6));
}

function getWeekNumber(row: DataRow): number | null {
  return typeof row.week === 'number' ? row.week : null;
}

function formatWeekContext(row: DataRow): string {
  const parts: string[] = [];
  const week = getWeekNumber(row);
  if (week !== null) {
    parts.push(`Week ${week}`);
  }
  if (typeof row.opponent_team === 'string' && row.opponent_team.length > 0) {
    parts.push(`vs ${row.opponent_team}`);
  }
  return parts.join(' ');
}

function formatWeekSpan(rows: NumericGameRow[]): string {
  const weeks = rows
    .map((entry) => getWeekNumber(entry.row))
    .filter((week): week is number => week !== null)
    .sort((left, right) => left - right);

  if (weeks.length === 0) {
    return 'Recent games';
  }
  if (weeks.length === 1) {
    return `Week ${weeks[0]}`;
  }

  const isContiguous = weeks.every((week, index) => index === 0 || week === weeks[index - 1] + 1);
  if (isContiguous) {
    return `Weeks ${weeks[0]}-${weeks[weeks.length - 1]}`;
  }
  return `Weeks ${weeks.join(', ')}`;
}

function formatSignedNumber(value: number): string {
  const prefix = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${prefix}${formatValue(Math.abs(roundNumber(value)))}`;
}

function getNumericRows(gameLogs: TablePayload, column: string): NumericGameRow[] {
  return gameLogs.rows
    .map((row) => ({ row, value: row[column] }))
    .filter(
      (entry): entry is NumericGameRow =>
        typeof entry.value === 'number' && Number.isFinite(entry.value),
    );
}

function findFirstAvailableMetric(gameLogs: TablePayload, candidates: string[]): string | null {
  return candidates.find((column) => gameLogs.visible_columns.includes(column)) ?? null;
}

function resolveSeasonBaseline(
  seasonRow: DataRow,
  gameLogs: TablePayload,
  metric: string,
): number | null {
  const seasonValue = seasonRow[metric];
  if (typeof seasonValue === 'number' && Number.isFinite(seasonValue)) {
    return seasonValue;
  }

  const numericRows = getNumericRows(gameLogs, metric);
  return numericRows.length > 0 ? roundNumber(average(numericRows.map((entry) => entry.value))) : null;
}

function buildPeakHighlight(gameLogs: TablePayload, metric: string): WeeklyHighlight | null {
  const numericRows = getNumericRows(gameLogs, metric);
  if (numericRows.length === 0) {
    return null;
  }

  const bestEntry = numericRows.reduce((best, current) =>
    current.value > best.value ? current : best,
  );

  return {
    context: formatWeekContext(bestEntry.row),
    eyebrow: 'Peak Week',
    title: getMetricMetadata(metric).label,
    value: formatValue(bestEntry.value),
  };
}

function buildTrendHighlight(gameLogs: TablePayload, metric: string): WeeklyHighlight | null {
  const numericRows = getNumericRows(gameLogs, metric).sort((left, right) => {
    const leftWeek = getWeekNumber(left.row) ?? Number.MAX_SAFE_INTEGER;
    const rightWeek = getWeekNumber(right.row) ?? Number.MAX_SAFE_INTEGER;
    return leftWeek - rightWeek;
  });

  if (numericRows.length < 2) {
    return null;
  }

  const windowSize = Math.min(3, Math.max(1, Math.floor(numericRows.length / 2)));
  const openingWindow = numericRows.slice(0, windowSize);
  const closingWindow = numericRows.slice(-windowSize);
  const openingAverage = roundNumber(average(openingWindow.map((entry) => entry.value)));
  const closingAverage = roundNumber(average(closingWindow.map((entry) => entry.value)));
  const delta = roundNumber(closingAverage - openingAverage);

  return {
    context:
      `Last ${closingWindow.length} avg ${formatValue(closingAverage)} vs first `
      + `${openingWindow.length} avg ${formatValue(openingAverage)} on `
      + `${getMetricMetadata(metric).label}.`,
    eyebrow: 'Closing Form',
    title: getMetricMetadata(metric).fullName,
    value: formatSignedNumber(delta),
  };
}

function buildRollingAverageHighlight(
  seasonRow: DataRow,
  gameLogs: TablePayload,
  metric: string,
): WeeklyHighlight | null {
  const numericRows = getNumericRows(gameLogs, metric).sort((left, right) => {
    const leftWeek = getWeekNumber(left.row) ?? Number.MAX_SAFE_INTEGER;
    const rightWeek = getWeekNumber(right.row) ?? Number.MAX_SAFE_INTEGER;
    return leftWeek - rightWeek;
  });

  if (numericRows.length < 2) {
    return null;
  }

  const windowSize = Math.min(3, numericRows.length);
  const recentWindow = numericRows.slice(-windowSize);
  const recentAverage = roundNumber(average(recentWindow.map((entry) => entry.value)));
  const baseline = resolveSeasonBaseline(seasonRow, gameLogs, metric);
  const baselineContext =
    baseline === null
      ? ''
      : ` vs season ${formatValue(baseline)} (${formatSignedNumber(recentAverage - baseline)})`;

  return {
    context:
      `${formatWeekSpan(recentWindow)} avg ${formatValue(recentAverage)}`
      + `${baselineContext} on ${getMetricMetadata(metric).label}.`,
    eyebrow: `Recent ${windowSize}-Game`,
    title: getMetricMetadata(metric).fullName,
    value: formatValue(recentAverage),
  };
}

function buildToughestOpponentHighlight(gameLogs: TablePayload): WeeklyHighlight | null {
  const toughestOpponents = getNumericRows(gameLogs, 'opp_SaCR');
  if (toughestOpponents.length === 0) {
    return null;
  }

  const toughestGame = toughestOpponents.reduce((best, current) =>
    current.value > best.value ? current : best,
  );
  const opponentLabel = String(toughestGame.row.opponent_team ?? 'Unknown');

  return {
    context: `${formatWeekContext(toughestGame.row)} · Opp SaCR ${formatValue(toughestGame.value)}`,
    eyebrow: 'Schedule Edge',
    title: 'Toughest Opponent',
    value: opponentLabel,
  };
}

export function buildWeeklyHighlights(
  kind: EntityKind,
  seasonRow: DataRow,
  gameLogs: TablePayload,
): WeeklyHighlight[] {
  const coreMetric = findFirstAvailableMetric(
    gameLogs,
    kind === 'teams' ? TEAM_TREND_COLUMNS : QB_TREND_COLUMNS,
  );

  return [
    coreMetric ? buildPeakHighlight(gameLogs, coreMetric) : null,
    coreMetric ? buildRollingAverageHighlight(seasonRow, gameLogs, coreMetric) : null,
    coreMetric ? buildTrendHighlight(gameLogs, coreMetric) : null,
    buildToughestOpponentHighlight(gameLogs),
  ].filter((highlight): highlight is WeeklyHighlight => highlight !== null);
}

export function enrichGameLogsWithOpponentRatings(
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
  const visibleColumns = [
    ...new Set([...gameLogs.visible_columns, ...ratingColumns.map((column) => `opp_${column}`)]),
  ];
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

export function buildGameLogGroups(kind: EntityKind, gameLogs: TablePayload): GameLogGroup[] {
  const identityColumns = orderedExisting(gameLogs.visible_columns, ['week', 'opponent_team', 'game_id']);
  const nonIdentityColumns = gameLogs.visible_columns.filter(
    (column) =>
      !identityColumns.includes(column)
      && column !== 'qb_id'
      && column !== 'team'
      && column !== 'qb_name',
  );
  const opponentRatings = orderedExisting(nonIdentityColumns, OPPONENT_RATING_COLUMNS);

  if (kind === 'teams') {
    const resultColumns = orderedExisting(nonIdentityColumns, TEAM_RESULT_COLUMNS);
    const perSnapRates = nonIdentityColumns.filter(
      (column) =>
        column.endsWith('_per_offensive_snap') || column.endsWith('_per_defensive_snap'),
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
        description: 'Final score, margin, and result context for each game.',
        columns: resultColumns,
      },
      {
        id: 'offense',
        label: 'Offense',
        description: 'What the selected team did with the ball in each game.',
        columns: offenseColumns,
      },
      {
        id: 'defense',
        label: 'Defense',
        description: 'What the selected team allowed or created on defense in each game.',
        columns: defenseColumns,
      },
      {
        id: 'per_snap_rates',
        label: 'Per-Snap Rates',
        description: 'Per-snap versions of the weekly stats so fast and slow games are easier to compare.',
        columns: perSnapRates,
      },
      {
        id: 'opponent_ratings',
        label: 'Opponent Ratings',
        description: 'Season-long opponent ratings attached to each game for context.',
        columns: opponentRatings,
      },
      {
        id: 'all',
        label: 'All Stats',
        description: 'Every available weekly field in one table.',
        columns: nonIdentityColumns,
      },
    ].filter((group) => group.columns.length > 0);
  }

  const resultColumns = orderedExisting(nonIdentityColumns, QB_RESULT_COLUMNS);
  const perDropbackRates = nonIdentityColumns.filter(
    (column) =>
      column.endsWith('_per_dropback')
      || column === 'qb_td_int_margin_rate'
      || column === 'qb_sack_rate',
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
      description: 'Final score, margin, and late-game result context for each start.',
      columns: resultColumns,
    },
    {
      id: 'volume',
      label: 'Volume',
      description: 'Attempts, yards, dropbacks, and other weekly passing volume stats.',
      columns: volumeColumns,
    },
    {
      id: 'efficiency',
      label: 'Efficiency',
      description: 'Weekly passing-efficiency measures that are not already dropback-normalized.',
      columns: efficiencyColumns,
    },
    {
      id: 'per_dropback_rates',
      label: 'Per-Dropback Rates',
      description: 'Per-dropback passing rates for cleaner game-to-game efficiency comparisons.',
      columns: perDropbackRates,
    },
    {
      id: 'opponent_ratings',
      label: 'Opponent Ratings',
      description: 'Season-long opponent ratings attached to each weekly QB row for context.',
      columns: opponentRatings,
    },
    {
      id: 'all',
      label: 'All Stats',
      description: 'Every available weekly QB field in one table.',
      columns: nonIdentityColumns,
    },
  ].filter((group) => group.columns.length > 0);
}

function pickFirstAvailable(availableColumns: string[], candidates: string[]): string | undefined {
  return candidates.find((column) => availableColumns.includes(column));
}

function buildLedgerDescription(surface: string): string {
  return (
    `These rows are against unique opponents for the ${surface} view. Division opponents are averaged together here when they were faced more than once. `
    + 'Opponent rating columns are season-long context, not single-game grades.'
  );
}

function getSelectedGroupColumns(
  kind: EntityKind,
  gameLogs: TablePayload,
  activeGroupId: string,
): string[] {
  return buildGameLogGroups(kind, gameLogs).find((group) => group.id === activeGroupId)?.columns ?? [];
}

function pickDeltaMetric(groupColumns: string[], candidates: string[]): string | undefined {
  return pickFirstAvailable(groupColumns, candidates) ?? groupColumns.find((column) => !column.startsWith('opp_'));
}

function buildTeamLedgerSpec(
  activeGroupId: string,
  availableColumns: string[],
  groupColumns: string[],
): OpponentLedgerSpec {
  const overallDifficulty = pickFirstAvailable(availableColumns, ['opp_SaCR', 'opp_SRS']);
  const offenseContext = pickFirstAvailable(availableColumns, ['opp_SaDR']);
  const defenseContext = pickFirstAvailable(availableColumns, ['opp_SaOR']);

  switch (activeGroupId) {
    case 'offense':
      return {
        description: buildLedgerDescription('selected team offense'),
        difficultyMetric: offenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'passing_epa',
          'points_for',
          'passing_yards',
          'total_yards',
        ]),
      };
    case 'defense':
      return {
        description: buildLedgerDescription('selected team defense'),
        difficultyMetric: defenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'passing_epa_allowed',
          'points_allowed',
          'passing_yards_allowed',
          'total_yards_allowed',
        ]),
      };
    case 'results':
      return {
        description: buildLedgerDescription('selected team results'),
        difficultyMetric: overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['point_margin']),
      };
    case 'per_snap_rates':
      return {
        description: buildLedgerDescription('selected team per-snap view'),
        difficultyMetric: overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'points_per_offensive_snap',
          'passing_epa_per_offensive_snap',
          'points_allowed_per_defensive_snap',
          'passing_epa_allowed_per_defensive_snap',
        ]),
      };
    case 'opponent_ratings':
    case 'all':
    default:
      return {
        description: buildLedgerDescription('selected team weekly summary'),
        difficultyMetric: overallDifficulty ?? offenseContext ?? defenseContext,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'point_margin',
          'points_per_offensive_snap',
          'points_allowed_per_defensive_snap',
        ]),
      };
  }
}

function buildQbLedgerSpec(
  activeGroupId: string,
  availableColumns: string[],
  groupColumns: string[],
): OpponentLedgerSpec {
  const passDefenseContext = pickFirstAvailable(availableColumns, ['opp_SaDR']);
  const overallDifficulty = pickFirstAvailable(availableColumns, ['opp_SaCR']);

  switch (activeGroupId) {
    case 'results':
      return {
        description: buildLedgerDescription('selected quarterback results'),
        difficultyMetric: overallDifficulty ?? passDefenseContext,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['point_margin', 'win_value']),
      };
    case 'volume':
      return {
        description: buildLedgerDescription('selected quarterback volume'),
        difficultyMetric: passDefenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['qb_pass_yards', 'qb_attempts', 'qb_dropbacks']),
      };
    case 'efficiency':
      return {
        description: buildLedgerDescription('selected quarterback efficiency'),
        difficultyMetric: passDefenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'qb_passer_rating',
          'qb_any_a',
          'qb_completion_percentage_above_expectation',
        ]),
      };
    case 'per_dropback_rates':
      return {
        description: buildLedgerDescription('selected quarterback per-dropback view'),
        difficultyMetric: passDefenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'qb_epa_per_dropback',
          'qb_any_a',
          'qb_pass_yards_per_dropback',
        ]),
      };
    case 'opponent_ratings':
    case 'all':
    default:
      return {
        description: buildLedgerDescription('selected quarterback weekly summary'),
        difficultyMetric: overallDifficulty ?? passDefenseContext,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, [
          'qb_epa_per_dropback',
          'qb_passer_rating',
          'point_margin',
        ]),
      };
  }
}

function summarizeGroupValue(rowsForOpponent: DataRow[], column: string): RowValue {
  const numericValues = rowsForOpponent
    .map((row) => row[column])
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));

  if (numericValues.length === rowsForOpponent.length && numericValues.length > 0) {
    return roundNumber(average(numericValues));
  }

  return rowsForOpponent.find((row) => row[column] !== null)?.[column] ?? null;
}

function buildScheduleBuckets(rows: DataRow[], difficultyMetric: string): Map<string, string> {
  // The difficulty metric is already a league-wide z-score (opp_SaCR / opp_SaDR /
  // opp_SaOR), so the thresholds apply to the raw value directly: +0.5 or higher
  // is Tougher, -0.5 or lower is Softer, everything between is Middle.
  const bucketByOpponent = new Map<string, string>();
  rows.forEach((row) => {
    const opponentTeam = String(row.opponent_team ?? '');
    const value = row[difficultyMetric];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return;
    }
    let bucket = 'Middle';
    if (value >= 0.5) {
      bucket = 'Tougher';
    } else if (value <= -0.5) {
      bucket = 'Softer';
    }
    bucketByOpponent.set(opponentTeam, bucket);
  });
  return bucketByOpponent;
}

export function buildOpponentBreakdown(
  kind: EntityKind,
  seasonRow: DataRow,
  gameLogs: TablePayload,
  activeGroupId: string,
): OpponentBreakdownTable {
  const groupColumns = getSelectedGroupColumns(kind, gameLogs, activeGroupId);
  const spec =
    kind === 'teams'
      ? buildTeamLedgerSpec(activeGroupId, gameLogs.visible_columns, groupColumns)
      : buildQbLedgerSpec(activeGroupId, gameLogs.visible_columns, groupColumns);
  const groupedRows = new Map<string, DataRow[]>();

  for (const row of gameLogs.rows) {
    const opponentTeam = String(row.opponent_team ?? 'Unknown');
    const currentRows = groupedRows.get(opponentTeam) ?? [];
    currentRows.push(row);
    groupedRows.set(opponentTeam, currentRows);
  }

  const summaryColumns = uniqueColumns([spec.difficultyMetric, ...spec.groupColumns]);
  const deltaColumn = spec.deltaMetric ? `season_delta_${spec.deltaMetric}` : undefined;
  const rows: DataRow[] = Array.from(groupedRows.entries()).map(
    ([opponentTeam, rowsForOpponent]) => {
      const summary = Object.fromEntries(
        summaryColumns.map((column) => [column, summarizeGroupValue(rowsForOpponent, column)]),
      );
    const deltaValue =
      spec.deltaMetric && typeof summary[spec.deltaMetric] === 'number'
        ? (() => {
            const baseline = resolveSeasonBaseline(seasonRow, gameLogs, spec.deltaMetric!);
            return baseline === null
              ? null
              : roundNumber((summary[spec.deltaMetric] as number) - baseline);
          })()
        : null;

      return {
        games: rowsForOpponent.length,
        opponent_team: opponentTeam,
        weeks: rowsForOpponent
          .map((row) => row.week)
          .filter((week): week is number => typeof week === 'number')
          .sort((left, right) => left - right)
          .join(', '),
        ...summary,
        ...(deltaColumn ? { [deltaColumn]: deltaValue } : {}),
      };
    },
  );

  if (spec.difficultyMetric) {
    const bucketByOpponent = buildScheduleBuckets(rows, spec.difficultyMetric);
    rows.forEach((row) => {
      row.opp_schedule_bucket = bucketByOpponent.get(String(row.opponent_team ?? '')) ?? 'Middle';
    });
  }

  rows.sort((left, right) =>
    String(left.opponent_team).localeCompare(String(right.opponent_team)),
  );

  const reservedIds = new Set(
    ['opponent_team', 'games', 'weeks', 'opp_schedule_bucket', spec.difficultyMetric].filter(
      (id): id is string => Boolean(id),
    ),
  );
  const columns: OpponentBreakdownColumn[] = [
    { id: 'opponent_team' },
    { id: 'games' },
    { id: 'weeks' },
    ...(spec.difficultyMetric
      ? [
          { id: spec.difficultyMetric },
          {
            id: 'opp_schedule_bucket',
            label: 'Sched Tier',
            tooltip:
              'Quick difficulty label based on the selected opponent rating column for this view. '
              + 'The rating is a league-wide z-score: +0.5 or higher reads Tougher, -0.5 or lower '
              + 'reads Softer, and everything between reads Middle.',
          },
        ]
      : []),
    ...uniqueColumns(spec.groupColumns.filter((column) => !reservedIds.has(column))).map(
      (column) => ({ id: column }),
    ),
    ...(deltaColumn ? [{ id: deltaColumn }] : []),
  ];

  return {
    columns,
    description: spec.description,
    rows,
  };
}