import type { EntityConfig, EntityKind, RowValue, TablePayload } from '@/api/types';

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
  LAR: 'Los Angeles Rams',
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
    defaultSortColumn: 'team_rating',
    defaultGroups: ['identity', 'ratings', 'per_game_rates'],
    compareColumns: [
      'team_rating',
      'offense_rating',
      'defense_rating',
      'special_teams_rating',
      'sos',
      'SRS',
    ],
    detailGroups: ['ratings', 'per_game_rates', 'per_snap_rates', 'opponent_context'],
    identityColumns: ['team'],
    primaryRankingLabel: 'Primary overall team rank: Team Rating',
    primaryRankingDescription:
      'Team Rating is points per game better than an average team on a neutral field, after '
      + 'adjusting for every opponent faced. It is the sum of the offense, defense, and '
      + 'special-teams ratings.',
    pageNotes: [
      'Team Rating is built from expected points added (EPA) per play, adjusted for each opponent and for who those opponents played, then converted to points per game.',
      'Offense, defense, and special-teams ratings are on the same points-per-game scale, so they add up exactly to Team Rating.',
      'SoS is the average Team Rating of the opponents played, with each opponent rated without its games against this team. Positive means a harder schedule.',
      'SRS is the classic point-margin rating, kept as a score-based reference beside the EPA-based Team Rating.',
    ],
  },
  qbs: {
    kind: 'qbs',
    title: 'Quarterback Ratings Index',
    singularLabel: 'QB',
    identityKey: 'qb_id',
    labelKey: 'qb_name',
    defaultSortColumn: 'adj_qb_epa_per_dropback',
    defaultGroups: ['identity', 'ratings', 'per_game_rates'],
    compareColumns: ['adj_qb_epa_per_dropback', 'qb_epa_per_dropback', 'qb_faced_pass_defense'],
    detailGroups: ['ratings', 'per_dropback_rates', 'per_game_rates', 'raw_totals', 'opponent_context'],
    // qb_id stays the row key and the link target; a raw player ID is noise in the table.
    identityColumns: ['qb_name', 'team'],
    primaryRankingLabel: 'Primary overall QB rank: Adjusted EPA per Dropback',
    primaryRankingDescription:
      'Adjusted EPA per dropback is the quarterback\'s expected points added per dropback after '
      + 'adjusting for the pass defenses he faced, on the same scale as raw EPA per dropback.',
    pageNotes: [
      'Adjusted EPA per dropback compares each passer with the defenses he actually faced, and each defense with every passer it faced.',
      'Faced Pass D is the dropback-weighted quality of those defenses (positive means tougher), with each defense rated without its games against this quarterback.',
      'Small samples are pulled toward the league average, so a backup with a few big plays does not top the table.',
      'Wins, comebacks, and other outcomes do not feed the rating; they stay available as context stats.',
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
