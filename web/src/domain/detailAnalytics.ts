import { SEASON_DELTA_PREFIX } from './metricMetadata';
import type { EntityKind, RowValue, TablePayload } from '@/api/types';

type DataRow = Record<string, RowValue>;

export interface GameLogGroup {
  id: string;
  label: string;
  description: string;
  columns: string[];
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

const OPPONENT_RATING_COLUMNS = [
  'opp_team_rating',
  'opp_SRS',
  'opp_offense_rating',
  'opp_defense_rating',
];
// Opponents ranked this high or low on the tier metric read Tougher or Softer.
const SCHEDULE_TIER_SIZE = 10;
const LEAGUE_SIZE_FIELD = 'opp_league_size';
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

function getNumericRows(gameLogs: TablePayload, column: string): NumericGameRow[] {
  return gameLogs.rows
    .map((row) => ({ row, value: row[column] }))
    .filter(
      (entry): entry is NumericGameRow =>
        typeof entry.value === 'number' && Number.isFinite(entry.value),
    );
}

/**
 * The season value a per-game number is compared with: `seasonRow` as the API serves it (per game),
 * never a view's row whose counts were turned into season totals, else the mean of the game values.
 */
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
  const ranksByColumn = new Map(
    ratingColumns.map((column) => [column, rankTeams(opponentRatingsTable.rows, column)]),
  );
  const leagueSize = opponentRatingsTable.rows.length;
  const rows = gameLogs.rows.map((gameRow) => {
    const opponentTeam = String(gameRow.opponent_team ?? '');
    const ratingRow = ratingsByTeam.get(opponentTeam);
    if (!ratingRow) {
      return gameRow;
    }
    return {
      ...gameRow,
      ...Object.fromEntries(
        ratingColumns.flatMap((column) => [
          [`opp_${column}`, ratingRow[column] ?? null],
          [`opp_${column}_rank`, ranksByColumn.get(column)?.get(opponentTeam) ?? null],
        ]),
      ),
      [LEAGUE_SIZE_FIELD]: leagueSize,
    };
  });
  const visibleColumns = [
    ...new Set([...gameLogs.visible_columns, ...ratingColumns.map((column) => `opp_${column}`)]),
  ];
  return { ...gameLogs, rows, visible_columns: visibleColumns };
}

/** Rank teams 1..n on one rating column, best (highest) first; teams without a value are skipped. */
function rankTeams(rows: DataRow[], column: string): Map<string, number> {
  const ranked = rows
    .filter((row) => typeof row[column] === 'number' && Number.isFinite(row[column]))
    .sort((a, b) => (b[column] as number) - (a[column] as number));
  return new Map(ranked.map((row, index) => [String(row.team ?? ''), index + 1]));
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

function buildLedgerDescription(): string {
  return (
    'One row per opponent: an opponent played twice, such as a division rival, gets one row that '
    + 'averages both games. Opponent rating and tier columns describe the opponent\'s full season, '
    + 'not these games; a vs Season column compares these games with the season average.'
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
  const overallDifficulty = pickFirstAvailable(availableColumns, ['opp_team_rating', 'opp_SRS']);
  // A team's offense is judged against the opposing defense, and its defense against the
  // opposing offense.
  const offenseContext = pickFirstAvailable(availableColumns, ['opp_defense_rating']);
  const defenseContext = pickFirstAvailable(availableColumns, ['opp_offense_rating']);

  switch (activeGroupId) {
    case 'offense':
      return {
        description: buildLedgerDescription(),
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
        description: buildLedgerDescription(),
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
        description: buildLedgerDescription(),
        difficultyMetric: overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['point_margin']),
      };
    case 'per_snap_rates':
      return {
        description: buildLedgerDescription(),
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
        description: buildLedgerDescription(),
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
  const passDefenseContext = pickFirstAvailable(availableColumns, ['opp_defense_rating']);
  const overallDifficulty = pickFirstAvailable(availableColumns, ['opp_team_rating']);

  switch (activeGroupId) {
    case 'results':
      return {
        description: buildLedgerDescription(),
        difficultyMetric: overallDifficulty ?? passDefenseContext,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['point_margin', 'win_value']),
      };
    case 'volume':
      return {
        description: buildLedgerDescription(),
        difficultyMetric: passDefenseContext ?? overallDifficulty,
        groupColumns,
        deltaMetric: pickDeltaMetric(groupColumns, ['qb_pass_yards', 'qb_attempts', 'qb_dropbacks']),
      };
    case 'efficiency':
      return {
        description: buildLedgerDescription(),
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
        description: buildLedgerDescription(),
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
        description: buildLedgerDescription(),
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
  // Tiers come from the opponent's league rank on the difficulty metric (1 = best), attached
  // when the game log is enriched: top 10 is Tougher, bottom 10 is Softer, the rest Middle.
  const bucketByOpponent = new Map<string, string>();
  rows.forEach((row) => {
    const opponentTeam = String(row.opponent_team ?? '');
    const rank = row[`${difficultyMetric}_rank`];
    const leagueSize = row[LEAGUE_SIZE_FIELD];
    if (typeof rank !== 'number' || typeof leagueSize !== 'number') {
      return;
    }
    let bucket = 'Middle';
    if (rank <= SCHEDULE_TIER_SIZE) {
      bucket = 'Tougher';
    } else if (rank > leagueSize - SCHEDULE_TIER_SIZE) {
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
  const deltaColumn = spec.deltaMetric ? `${SEASON_DELTA_PREFIX}${spec.deltaMetric}` : undefined;
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
    const bucketByOpponent = buildScheduleBuckets(gameLogs.rows, spec.difficultyMetric);
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
    { id: 'weeks', label: 'Weeks', tooltip: 'The weeks of the games against this opponent.' },
    ...(spec.difficultyMetric
      ? [
          { id: spec.difficultyMetric },
          {
            id: 'opp_schedule_bucket',
            label: 'Opp Tier',
            tooltip:
              'Where this opponent ranked in the league on the rating to the left, over its full '
              + 'season: Tougher is top 10, Softer is bottom 10, and Middle is the rest.',
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
