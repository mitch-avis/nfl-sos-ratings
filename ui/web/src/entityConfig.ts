import type { EntityConfig, EntityKind, RowValue, TablePayload } from './types';

const TEAM_FULL_NAMES: Record<string, string> = {
  ARI: 'Arizona Cardinals',
  ATL: 'Atlanta Falcons',
  BAL: 'Baltimore Ravens',
  BUF: 'Buffalo Bills',
  CAR: 'Carolina Panthers',
  CHI: 'Chicago Bears',
  CIN: 'Cincinnati Bengals',
  CLE: 'Cleveland Browns',
  DAL: 'Dallas Cowboys',
  DEN: 'Denver Broncos',
  DET: 'Detroit Lions',
  GB: 'Green Bay Packers',
  HOU: 'Houston Texans',
  IND: 'Indianapolis Colts',
  JAX: 'Jacksonville Jaguars',
  KC: 'Kansas City Chiefs',
  LA: 'Los Angeles Rams',
  LAC: 'Los Angeles Chargers',
  LV: 'Las Vegas Raiders',
  MIA: 'Miami Dolphins',
  MIN: 'Minnesota Vikings',
  NE: 'New England Patriots',
  NO: 'New Orleans Saints',
  NYG: 'New York Giants',
  NYJ: 'New York Jets',
  PHI: 'Philadelphia Eagles',
  PIT: 'Pittsburgh Steelers',
  SEA: 'Seattle Seahawks',
  SF: 'San Francisco 49ers',
  TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans',
  WAS: 'Washington Commanders',
};

const ENTITY_CONFIG: Record<EntityKind, EntityConfig> = {
  teams: {
    kind: 'teams',
    title: 'Team Ratings Index',
    singularLabel: 'Team',
    identityKey: 'team',
    labelKey: 'team',
    defaultSortColumn: 'SaCR',
    defaultGroups: ['identity', 'ratings', 'per_game_rates'],
    compareColumns: ['SaCR', 'SaCR_alltime', 'SaOvR', 'SaOvR_alltime', 'sos', 'SRS'],
    detailGroups: ['ratings', 'per_game_rates', 'per_snap_rates', 'opponent_context'],
    identityColumns: ['team'],
    primaryRankingLabel: 'Primary overall team rank: SaCR',
    primaryRankingDescription:
      'Use SaCR for the final schedule-adjusted team ranking. Use SRS as a point-margin '
      + 'reference and SaOR, SaDR, and SaOvR as supporting component views.',
    pageNotes: [
      'SaCR is the default team sort and the best current one-number summary of the schedule-adjusted team model.',
      'SoS is the played-game mean opponent SaCR, which surfaces how hard the overall team slate was without changing the team grade itself.',
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
    compareColumns: ['QSaCR', 'QSaCR_alltime', 'QSaOR', 'QSaOR_alltime', 'QSoS', 'faced_opp_SaCR'],
    detailGroups: ['ratings', 'per_dropback_rates', 'per_game_rates', 'raw_totals', 'opponent_context'],
    identityColumns: ['qb_id', 'qb_name', 'team'],
    primaryRankingLabel: 'Primary overall QB rank: QSaCR',
    primaryRankingDescription:
      'Use QSaCR for the final overall QB ranking. Use QSaOR for the cleaner opponent-adjusted '
      + 'passing-performance view when you want less outcome signal.',
    pageNotes: [
      'QSaCR is the final QB composite after opponent adjustment and a smaller outcome layer.',
      'QSaOR is the better choice when you want QB passing performance with opponent adjustment but without the extra outcome blend.',
      'QSoS is dropback-weighted pass-defense difficulty. Higher means the QB faced tougher pass defenses, not that he played better.',
      'Opp SaCR is the equal-game mean opponent team quality companion, so users can compare overall slate strength against the pass-defense-only QSoS lens.',
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

export function getFullTeamName(teamAbbreviation: string): string {
  return TEAM_FULL_NAMES[teamAbbreviation] ?? teamAbbreviation;
}