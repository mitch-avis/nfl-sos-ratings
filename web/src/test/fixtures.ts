import type {
  ColumnMetadataPayload,
  MetricRegistryPayload,
  RankRangesPayload,
  RowValue,
  SeasonDataset,
  TablePayload,
} from '@/api/types'

export function columnMeta(
  label: string,
  overrides: Partial<ColumnMetadataPayload> = {},
): ColumnMetadataPayload {
  return {
    label,
    full_name: label,
    description: `${label} description.`,
    polarity: 'higher',
    contextual: false,
    category: 'Schedule-Adjusted Ratings',
    subcategory: null,
    shape: 'score',
    denominator: null,
    source: 'D',
    base_name: label,
    ...overrides,
  }
}

function table(
  rows: Array<Record<string, RowValue>>,
  identity: string[],
  ratings: string[],
  metadata: Record<string, ColumnMetadataPayload>,
): TablePayload {
  return {
    rows,
    visible_columns: [...identity, ...ratings],
    column_groups: { identity, ratings },
    column_metadata: metadata,
  }
}

const TEAM_RATINGS = [
  'team_rating',
  'offense_rating',
  'defense_rating',
  'special_teams_rating',
  'sos',
  'SRS',
]
const QB_RATINGS = ['adj_qb_epa_per_dropback', 'qb_faced_pass_defense']
const RATING_LABELS: Record<string, string> = {
  team_rating: 'Team Rating',
  offense_rating: 'Off Rating',
  defense_rating: 'Def Rating',
  special_teams_rating: 'ST Rating',
  sos: 'SoS',
  SRS: 'SRS',
  adj_qb_epa_per_dropback: 'Adj EPA/DB',
  qb_faced_pass_defense: 'Faced Pass D',
}

export const SEASON_2025: SeasonDataset = {
  season: 2025,
  teams: table(
    [
      {
        team: 'DEN',
        team_rating: 8.4,
        offense_rating: 4.1,
        defense_rating: 3.9,
        special_teams_rating: 0.4,
        sos: 0.2,
        SRS: 8.1,
      },
      {
        team: 'KC',
        team_rating: 3.5,
        offense_rating: 2.6,
        defense_rating: 0.7,
        special_teams_rating: 0.2,
        sos: -0.1,
        SRS: 4.0,
      },
      {
        team: 'LV',
        team_rating: -9.0,
        offense_rating: -5.5,
        defense_rating: -3.1,
        special_teams_rating: -0.4,
        sos: 0.4,
        SRS: -9.5,
      },
    ],
    ['team'],
    TEAM_RATINGS,
    {
      team: columnMeta('Team', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      ...Object.fromEntries(
        TEAM_RATINGS.map((column) => [column, columnMeta(RATING_LABELS[column])]),
      ),
    },
  ),
  qbs: table(
    [
      {
        qb_id: 'qb-1',
        qb_name: 'Bo Nix',
        team: 'DEN',
        adj_qb_epa_per_dropback: 0.12,
        qb_faced_pass_defense: 0.01,
      },
      {
        qb_id: 'qb-2',
        qb_name: 'Backup Arm',
        team: 'LV',
        adj_qb_epa_per_dropback: null,
        qb_faced_pass_defense: null,
      },
    ],
    ['qb_id', 'qb_name', 'team'],
    QB_RATINGS,
    {
      qb_id: columnMeta('QB ID', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      qb_name: columnMeta('QB', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      team: columnMeta('Team', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      ...Object.fromEntries(QB_RATINGS.map((column) => [column, columnMeta(RATING_LABELS[column])])),
    },
  ),
}

export const DEN_GAME_LOGS: TablePayload = {
  rows: [
    { game_id: '2025_01_TEN_DEN', week: 1, opponent_team: 'TEN', points_for: 20, points_allowed: 12, point_margin: 8, win_value: 1 },
    { game_id: '2025_02_DEN_IND', week: 2, opponent_team: 'IND', points_for: 28, points_allowed: 29, point_margin: -1, win_value: 0 },
    { game_id: '2025_03_DEN_LAC', week: 3, opponent_team: 'LAC', points_for: 20, points_allowed: 23, point_margin: -3, win_value: 0 },
  ],
  visible_columns: ['game_id', 'week', 'opponent_team', 'points_for', 'points_allowed', 'point_margin', 'win_value'],
  column_groups: {},
  column_metadata: {
    points_for: columnMeta('Points For', { category: 'Offense', subcategory: 'Scoring', shape: 'count' }),
    points_allowed: columnMeta('Points Allowed', { category: 'Defense', polarity: 'lower', shape: 'count' }),
    point_margin: columnMeta('Point Margin', { category: 'Overall', shape: 'count' }),
  },
}

export const REGISTRY: MetricRegistryPayload = {
  entities: { team: { categories: [] }, qb: { categories: [] } },
  metrics: {},
}

/** A fetch stub that answers each known API path with its JSON payload. */
export function stubApi(routes: Record<string, unknown>): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url
    const path = url.replace(/^https?:\/\/[^/]+/, '')
    if (!(path in routes)) {
      return new Response(JSON.stringify({ detail: `No route for ${path}` }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      })
    }
    return new Response(JSON.stringify(routes[path]), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  }) as typeof fetch
}

export const DEN_RATING_HISTORY: TablePayload = {
  rows: [
    { week: 1, team: 'DEN', games_played: 1, team_rating: 1.2, offense_rating: 0.6 },
    { week: 2, team: 'DEN', games_played: 2, team_rating: 3.4, offense_rating: 1.5 },
    { week: 3, team: 'DEN', games_played: 3, team_rating: 5.1, offense_rating: 2.2 },
  ],
  visible_columns: ['week', 'team', 'games_played', 'team_rating', 'offense_rating'],
  column_groups: {
    identity: ['week', 'team'],
    sample: ['games_played'],
    ratings: ['team_rating', 'offense_rating'],
  },
  column_metadata: {
    team_rating: columnMeta('Team Rating'),
    offense_rating: columnMeta('Off Rating'),
  },
}

const RANGE_QUANTILES = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']

function teamRankRange(team: string, published: number, ranks: number[], probabilities: number[]) {
  return {
    team,
    team_rank: published,
    ...Object.fromEntries(RANGE_QUANTILES.map((key, index) => [`team_rank_${key}`, ranks[index]])),
    ...Object.fromEntries(RANGE_QUANTILES.map((key, index) => [`team_rating_${key}`, 8 - published * 4 + index])),
    team_rank_top5_probability: 1,
    team_rank_top10_probability: 1,
    team_rank_missing_share: 0,
    team_rank_probabilities: probabilities,
  }
}

export const TEAM_RANK_RANGES: RankRangesPayload = {
  rows: [
    teamRankRange('DEN', 1, [1, 1, 1, 1, 2, 2, 3], [0.62, 0.3, 0.08]),
    teamRankRange('KC', 2, [1, 1, 2, 2, 2, 3, 3], [0.3, 0.52, 0.18]),
    teamRankRange('LV', 3, [2, 2, 3, 3, 3, 3, 3], [0.08, 0.18, 0.74]),
  ],
  visible_columns: [],
  column_groups: {},
  column_metadata: {},
}
