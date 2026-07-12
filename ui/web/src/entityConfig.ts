import type { EntityConfig, EntityKind, RowValue, TablePayload } from './types';

const ENTITY_CONFIG: Record<EntityKind, EntityConfig> = {
  teams: {
    kind: 'teams',
    title: 'Team Ratings Index',
    singularLabel: 'Team',
    identityKey: 'team',
    labelKey: 'team',
    defaultSortColumn: 'SaCR',
    defaultGroups: ['identity', 'ratings', 'per_game_rates'],
    compareColumns: ['SaCR', 'SRS', 'SaOvR', 'SaOR', 'SaDR'],
    detailGroups: ['ratings', 'per_game_rates', 'per_snap_rates', 'opponent_context'],
    identityColumns: ['team'],
    primaryRankingLabel: 'Primary overall team rank: SaCR',
    primaryRankingDescription:
      'Use SaCR for the final schedule-adjusted team ranking. Use SRS as a point-margin '
      + 'reference and SaOR, SaDR, and SaOvR as supporting component views.',
    pageNotes: [
      'SaCR is the default team sort and the best current one-number summary of the schedule-adjusted team model.',
      'SRS remains a useful point-margin reference when you want a simpler whole-team baseline beside the composite.',
      'SaCR remains the multi-stat composite, while SaOR and SaDR are better offense-only and defense-only slices.',
    ],
  },
  qbs: {
    kind: 'qbs',
    title: 'Quarterback Ratings Index',
    singularLabel: 'QB',
    identityKey: 'qb_id',
    labelKey: 'qb_name',
    defaultSortColumn: 'QSaCR',
    defaultGroups: ['identity', 'ratings', 'per_game_rates'],
    compareColumns: ['QSaCR', 'QSaOR', 'QSoS', 'QOutcome', 'QRaw'],
    detailGroups: ['ratings', 'per_dropback_rates', 'per_game_rates', 'raw_totals', 'opponent_context'],
    identityColumns: ['qb_id', 'qb_name', 'team'],
    primaryRankingLabel: 'Primary overall QB rank: QSaCR',
    primaryRankingDescription:
      'Use QSaCR for the final overall QB ranking. Use QSaOR for the cleaner opponent-adjusted '
      + 'passing-performance view when you want less outcome signal.',
    pageNotes: [
      'QSaCR is the final QB composite after opponent adjustment and a smaller outcome layer.',
      'QSaOR is the better choice when you want QB passing performance with opponent adjustment but without the extra outcome blend.',
      'QSoS is schedule difficulty. Higher means the QB faced tougher defenses, not that he played better.',
    ],
  },
};

export function getEntityConfig(kind: EntityKind): EntityConfig {
  return ENTITY_CONFIG[kind];
}

export function getEntityId(kind: EntityKind, row: Record<string, RowValue>): string {
  const config = getEntityConfig(kind);
  const value = row[config.identityKey];
  return String(value ?? '');
}

export function getEntityLabel(kind: EntityKind, row: Record<string, RowValue>): string {
  const config = getEntityConfig(kind);
  const label = row[config.labelKey];
  if (label !== null && label !== undefined && label !== '') {
    return String(label);
  }
  return getEntityId(kind, row);
}

export function getEntityRow(
  table: TablePayload,
  kind: EntityKind,
  entityId: string,
): Record<string, RowValue> | undefined {
  return table.rows.find((row) => getEntityId(kind, row) === entityId);
}