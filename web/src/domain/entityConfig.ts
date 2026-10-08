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
      'Team Rating is points per game better (+) or worse (−) than an average team on a neutral '
      + 'field, after adjusting for every opponent faced; 0 is average. It is the sum of the '
      + 'offense, defense, and special-teams ratings.',
    pageNotes: [
      'Team Rating is built from expected points added (EPA), which scores each play by how much it changed the offense\'s expected points. Each team is judged against its opponents, and those opponents against theirs, all at once, then put in points per game.',
      'Offense, defense, and special-teams ratings are on the same points-per-game scale, so they add up exactly to Team Rating.',
      'SoS (schedule strength) is the average Team Rating of the opponents played, counted once per game, with each opponent rated without its games against this team. Positive means a tougher schedule.',
      'SRS is the classic score-based rating: average point margin, adjusted for the opponents played. It sits beside Team Rating as a check built from final scores instead of plays.',
      'Shading marks better or worse within the season; gray shading marks context instead, such as schedule strength or the opponents faced, deeper for tougher.',
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
      'Adjusted EPA per dropback is the quarterback\'s expected points added per dropback (EPA: how '
      + 'much each play changed his team\'s expected points), after adjusting for the pass defenses '
      + 'he faced. It reads on the same scale as raw EPA per dropback, whose league average is '
      + 'usually a little above 0.',
    pageNotes: [
      'Adjusted EPA per dropback compares each passer with the defenses he actually faced, and each defense with every passer it faced.',
      'Pass Defense Faced is the average strength of those pass defenses, weighted by his dropbacks against each, with each defense rated only on its plays against other passers. Positive means tougher.',
      'Small samples are pulled toward the league average, so a backup with a few big plays does not top the table.',
      'Wins, comebacks, and other outcomes do not feed the rating; they stay available as context stats.',
      'EPA also reflects his line, receivers, and play-calling, which public play-by-play cannot separate, so the rating describes the passing offense he led.',
      'Shading marks better or worse within the season; gray shading marks context instead, such as schedule strength or the opponents faced, deeper for tougher.',
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
