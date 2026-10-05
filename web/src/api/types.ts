export type RowValue = string | number | boolean | null;
export type ThemeMode = 'light' | 'dark';
export type PaletteMode = 'classic' | 'broncos';
export type MetricShape = 'count' | 'rate' | 'avg' | 'flag' | 'id' | 'score';
export type PrimaryView =
  | 'ratings'
  | 'raw_total_stats'
  | 'per_game_rates'
  | 'per_play_rates'
  | 'opponent_per_game_rates'
  | 'opponent_per_play_rates';

export interface ColumnMetadataPayload {
  label: string;
  full_name: string;
  description: string;
  polarity: 'higher' | 'lower' | 'neutral';
  contextual: boolean;
  category: string;
  subcategory: string | null;
  shape: MetricShape;
  denominator: string | null;
  source: string;
  base_name: string;
}

export interface RegistryCategoryPayload {
  name: string;
  description: string;
  subcategories: string[];
}

export interface MetricRegistryPayload {
  entities: Record<'team' | 'qb', { categories: RegistryCategoryPayload[] }>;
  metrics: Record<string, ColumnMetadataPayload>;
}

export interface TablePayload {
  rows: Array<Record<string, RowValue>>;
  visible_columns: string[];
  column_groups: Record<string, string[]>;
  column_metadata?: Record<string, ColumnMetadataPayload>;
}

/** `/api/seasons/{season}/{kind}/rating-ranges`: one row per team or QB, with list columns. */
export interface RankRangesPayload extends Omit<TablePayload, 'rows'> {
  rows: Array<Record<string, RowValue | number[]>>;
}

/** `/api/seasons/{season}/{kind}/wp-ratings?threshold=X`: the garbage-time filter view. */
export interface WpRatingsPayload extends TablePayload {
  threshold: number;
  max_threshold: number;
}

export interface SeasonDataset {
  season: number;
  teams: TablePayload;
  qbs: TablePayload;
}

export interface SeasonsResponse {
  seasons: number[];
}

export type EntityKind = 'teams' | 'qbs';

export interface EntityConfig {
  kind: EntityKind;
  title: string;
  singularLabel: string;
  identityKey: string;
  labelKey: string;
  defaultSortColumn: string;
  defaultGroups: string[];
  compareColumns: string[];
  detailGroups: string[];
  identityColumns: string[];
  primaryRankingLabel: string;
  primaryRankingDescription: string;
  pageNotes: string[];
}
