import type {
  ColumnMetadataPayload,
  EntityKind,
  PrimaryView,
  RowValue,
  TablePayload,
} from './types.js';

type DataRow = Record<string, RowValue>;

export interface EntityViewState {
  primaryView?: PrimaryView;
  teamCategory?: string;
  teamSubcategories?: Record<string, Record<string, boolean>>;
  qbSubcategories?: Record<string, boolean>;
}

export interface ResolvedEntityViewState {
  primaryView: PrimaryView;
  teamCategories: string[];
  teamCategory: string;
  teamSubcategories: Record<string, Record<string, boolean>>;
  qbSubcategories: Record<string, boolean>;
  activeSubcategoryOptions: string[];
  activeSubcategories: Record<string, boolean>;
}

export interface DerivedTableView {
  metricColumns: string[];
  selectedColumns: string[];
  table: TablePayload;
}

export interface GameLogColumnSelection {
  columns: string[];
  contextColumns: string[];
  effectiveView: PrimaryView;
  metricColumns: string[];
}

export const PRIMARY_VIEWS: PrimaryView[] = [
  'ratings',
  'raw_total_stats',
  'per_game_rates',
  'per_play_rates',
  'opponent_per_game_rates',
  'opponent_per_play_rates',
];

const TEAM_CATEGORIES = ['Overall', 'Offense', 'Defense', 'Special Teams'];

const TEAM_SUBCATEGORIES: Record<string, string[]> = {
  Overall: [],
  Offense: [
    'Total',
    'Passing',
    'Rushing',
    'Receiving',
    'Scoring',
    'Downs & Conversions',
    'Drives & Field Position',
    'Turnovers',
    'Penalties',
  ],
  Defense: [
    'Total',
    'Passing',
    'Rushing',
    'Receiving',
    'Scoring',
    'Downs & Conversions',
    'Drives & Field Position',
    'Turnovers',
    'Pressure & Playmaking',
    'Penalties',
  ],
  'Special Teams': [
    'Kicking',
    'Kickoffs & Coverage',
    'Kick Returns',
    'Punting & Coverage',
    'Punt Returns',
    'ST Scoring & Blocks',
  ],
};

const QB_SUBCATEGORIES = [
  'Identity & Availability',
  'Passing Volume',
  'Passing Efficiency',
  'Advanced & Expected',
  'Pressure, Sacks & Pocket',
  'Rushing',
  'Scoring, Clutch & Outcomes',
  'Turnovers & Ball Security',
];

const PLAY_SUFFIXES = [
  '_per_offensive_snap',
  '_per_defensive_snap',
  '_per_dropback',
  '_per_attempt',
  '_per_carry',
  '_per_drive',
  '_per_series',
  '_per_target',
  '_per_reception',
  '_per_return',
  '_per_punt',
  '_per_kickoff',
  '_per_fg_att',
];

const TEAM_RAW_TOTAL_ALREADY_TOTAL = new Set(['games_played', 'games', 'wins', 'losses', 'ties']);
const OPPONENT_RATING_COLUMNS = ['opp_SaDR', 'opp_SaOR', 'opp_SaCR', 'opp_SRS'];
const DETAIL_IDENTITY_COLUMNS = ['week', 'opponent_team', 'game_id'];
const DETAIL_RESULT_COLUMNS = [
  'points_for',
  'points_allowed',
  'point_margin',
  'win_value',
  'turnover_margin',
];

export function resolveEntityViewState(
  kind: EntityKind,
  current?: EntityViewState,
): ResolvedEntityViewState {
  const primaryView = current?.primaryView ?? 'ratings';
  const defaultTeamSubcategories = Object.fromEntries(
    TEAM_CATEGORIES.map((category) => [
      category,
      Object.fromEntries(TEAM_SUBCATEGORIES[category].map((subcategory) => [subcategory, true])),
    ]),
  );
  const mergedTeamSubcategories = {
    ...defaultTeamSubcategories,
    ...current?.teamSubcategories,
  };

  const teamCategory = TEAM_CATEGORIES.includes(current?.teamCategory ?? '')
    ? (current?.teamCategory ?? 'Offense')
    : 'Offense';
  const activeSubcategoryOptions = kind === 'teams'
    ? TEAM_SUBCATEGORIES[teamCategory]
    : QB_SUBCATEGORIES;
  const activeSubcategories = kind === 'teams'
    ? Object.fromEntries(
        activeSubcategoryOptions.map((subcategory) => [
          subcategory,
          mergedTeamSubcategories[teamCategory]?.[subcategory] ?? true,
        ]),
      )
    : Object.fromEntries(
        QB_SUBCATEGORIES.map((subcategory) => [subcategory, current?.qbSubcategories?.[subcategory] ?? true]),
      );

  return {
    primaryView,
    teamCategories: [...TEAM_CATEGORIES],
    teamCategory,
    teamSubcategories: mergedTeamSubcategories,
    qbSubcategories: Object.fromEntries(
      QB_SUBCATEGORIES.map((subcategory) => [subcategory, current?.qbSubcategories?.[subcategory] ?? true]),
    ),
    activeSubcategoryOptions,
    activeSubcategories,
  };
}

export function getEffectiveStatView(primaryView: PrimaryView): PrimaryView {
  return primaryView === 'ratings' ? 'per_game_rates' : primaryView;
}

export function buildSeasonViewTable(
  kind: EntityKind,
  sourceTable: TablePayload,
  state: ResolvedEntityViewState,
): DerivedTableView {
  const identityColumns = sourceTable.column_groups.identity ?? [];
  const ratingColumns = sourceTable.column_groups.ratings ?? [];
  const metricColumns =
    state.primaryView === 'ratings'
      ? ratingColumns
      : sourceTable.visible_columns.filter(
          (column) =>
            !identityColumns.includes(column)
            && !ratingColumns.includes(column)
            && matchesTaxonomy(kind, sourceTable.column_metadata?.[column], state)
            && matchesSeasonView(column, sourceTable.column_metadata?.[column], state.primaryView),
        );
  const selectedColumns = [...identityColumns, ...metricColumns];
  const rows = sourceTable.rows.map((row) =>
    transformSeasonRow(kind, row, metricColumns, state.primaryView, sourceTable.column_metadata ?? {}),
  );

  return {
    metricColumns,
    selectedColumns,
    table: {
      ...sourceTable,
      rows,
      visible_columns: selectedColumns,
    },
  };
}

export function buildGameLogColumnSelection(
  kind: EntityKind,
  gameLogs: TablePayload,
  state: ResolvedEntityViewState,
): GameLogColumnSelection {
  const effectiveView = getEffectiveStatView(state.primaryView);
  const identityColumns = orderedExisting(gameLogs.visible_columns, DETAIL_IDENTITY_COLUMNS);
  const resultColumns = orderedExisting(gameLogs.visible_columns, DETAIL_RESULT_COLUMNS);
  const matchedContextColumns = getMatchedOpponentContextColumns(kind, gameLogs.visible_columns, state);

  const metricColumns = effectiveView.startsWith('opponent_')
    ? matchedContextColumns
    : gameLogs.visible_columns.filter(
        (column) =>
          !identityColumns.includes(column)
          && !resultColumns.includes(column)
          && !OPPONENT_RATING_COLUMNS.includes(column)
          && matchesTaxonomy(kind, gameLogs.column_metadata?.[column], state)
          && matchesSeasonView(column, gameLogs.column_metadata?.[column], effectiveView),
      );

  const contextColumns = effectiveView.startsWith('opponent_') ? [] : matchedContextColumns;

  return {
    columns: uniqueColumns([...identityColumns, ...resultColumns, ...contextColumns, ...metricColumns]),
    contextColumns,
    effectiveView,
    metricColumns,
  };
}

export function deriveLegacyDetailSurfaceId(
  kind: EntityKind,
  state: ResolvedEntityViewState,
): string {
  const effectiveView = getEffectiveStatView(state.primaryView);

  if (kind === 'teams') {
    if (effectiveView === 'per_play_rates') {
      return 'per_snap_rates';
    }
    if (state.teamCategory === 'Defense') {
      return 'defense';
    }
    if (state.teamCategory === 'Offense') {
      return 'offense';
    }
    return 'results';
  }

  if (effectiveView === 'per_play_rates') {
    return 'per_dropback_rates';
  }
  if (state.activeSubcategories['Passing Volume'] || state.activeSubcategories['Identity & Availability']) {
    return 'volume';
  }
  if (
    state.activeSubcategories['Passing Efficiency']
    || state.activeSubcategories['Advanced & Expected']
    || state.activeSubcategories['Pressure, Sacks & Pocket']
  ) {
    return 'efficiency';
  }
  return 'results';
}

function transformSeasonRow(
  kind: EntityKind,
  row: DataRow,
  metricColumns: string[],
  primaryView: PrimaryView,
  columnMetadata: Record<string, ColumnMetadataPayload>,
): DataRow {
  if (!(kind === 'teams' && primaryView === 'raw_total_stats')) {
    return row;
  }

  const gamesPlayed = typeof row.games_played === 'number'
    ? row.games_played
    : typeof row.games === 'number'
      ? row.games
      : null;
  if (gamesPlayed === null) {
    return row;
  }

  const transformed: DataRow = { ...row };
  for (const column of metricColumns) {
    const metadata = columnMetadata[column];
    const value = row[column];
    if (
      metadata?.shape === 'count'
      && typeof value === 'number'
      && Number.isFinite(value)
      && !TEAM_RAW_TOTAL_ALREADY_TOTAL.has(column)
      && !column.endsWith('_total')
      && !column.endsWith('_per_game')
      && !isPlayColumn(column)
    ) {
      transformed[column] = Number((value * gamesPlayed).toFixed(6));
    }
  }
  return transformed;
}

function matchesTaxonomy(
  kind: EntityKind,
  metadata: ColumnMetadataPayload | undefined,
  state: ResolvedEntityViewState,
): boolean {
  if (!metadata || metadata.category === 'Schedule-Adjusted Ratings') {
    return false;
  }

  if (kind === 'teams') {
    if (metadata.category !== state.teamCategory) {
      return false;
    }
    if (state.activeSubcategoryOptions.length === 0) {
      return true;
    }
    if (!metadata.subcategory) {
      return true;
    }
    return state.activeSubcategories[metadata.subcategory] ?? false;
  }

  return state.activeSubcategories[metadata.category] ?? false;
}

function matchesSeasonView(
  column: string,
  metadata: ColumnMetadataPayload | undefined,
  view: PrimaryView,
): boolean {
  if (!metadata) {
    return false;
  }

  const isContextual = metadata.contextual;
  if (view.startsWith('opponent_') !== isContextual) {
    return false;
  }

  switch (view) {
    case 'raw_total_stats':
      if (isTotalColumn(column)) {
        return true;
      }
      if (isPerGameColumn(column) || isPlayColumn(column)) {
        return false;
      }
      return metadata.shape !== 'score' && metadata.shape !== 'id';
    case 'per_game_rates':
    case 'opponent_per_game_rates':
      if (isTotalColumn(column) || isPlayColumn(column)) {
        return false;
      }
      return metadata.shape !== 'score' && metadata.shape !== 'id';
    case 'per_play_rates':
    case 'opponent_per_play_rates':
      if (isPlayColumn(column)) {
        return true;
      }
      if (isTotalColumn(column) || isPerGameColumn(column)) {
        return false;
      }
      return metadata.shape === 'rate' || metadata.shape === 'avg' || metadata.shape === 'flag';
    case 'ratings':
    default:
      return false;
  }
}

function getMatchedOpponentContextColumns(
  kind: EntityKind,
  availableColumns: string[],
  state: ResolvedEntityViewState,
): string[] {
  const candidates = kind === 'teams'
    ? state.teamCategory === 'Offense'
      ? ['opp_SaDR', 'opp_SaCR']
      : state.teamCategory === 'Defense'
        ? ['opp_SaOR', 'opp_SaCR']
        : ['opp_SaCR', 'opp_SRS']
    : ['opp_SaDR', 'opp_SaCR'];

  return orderedExisting(availableColumns, candidates);
}

function orderedExisting(columns: string[], preferredColumns: string[]): string[] {
  const available = new Set(columns);
  return preferredColumns.filter((column) => available.has(column));
}

function uniqueColumns(columns: string[]): string[] {
  return Array.from(new Set(columns));
}

function isTotalColumn(column: string): boolean {
  return column.endsWith('_total');
}

function isPerGameColumn(column: string): boolean {
  return column.endsWith('_per_game');
}

function isPlayColumn(column: string): boolean {
  return PLAY_SUFFIXES.some((suffix) => column.endsWith(suffix));
}