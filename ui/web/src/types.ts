export type RowValue = string | number | boolean | null;
export type ThemeMode = 'light' | 'dark';
export type PaletteMode = 'classic' | 'broncos';

export interface TablePayload {
  rows: Array<Record<string, RowValue>>;
  visible_columns: string[];
  column_groups: Record<string, string[]>;
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