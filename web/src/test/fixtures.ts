import type {
  ColumnMetadataPayload,
  MetricRegistryPayload,
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

const TEAM_RATINGS = ['SaCR', 'sos', 'SRS', 'SaOvR', 'SaOR', 'SaDR']
const QB_RATINGS = ['QSaCR', 'QSaOR', 'QSoS', 'QRaw']

export const SEASON_2025: SeasonDataset = {
  season: 2025,
  teams: table(
    [
      { team: 'DEN', SaCR: 1.4, sos: 0.2, SRS: 8.1, SaOvR: 1.2, SaOR: 0.9, SaDR: 1.5 },
      { team: 'KC', SaCR: 0.6, sos: -0.1, SRS: 4.0, SaOvR: 0.5, SaOR: 0.8, SaDR: 0.1 },
      { team: 'LV', SaCR: -1.3, sos: 0.4, SRS: -9.5, SaOvR: -1.1, SaOR: -1.2, SaDR: -0.7 },
    ],
    ['team'],
    TEAM_RATINGS,
    {
      team: columnMeta('Team', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      ...Object.fromEntries(TEAM_RATINGS.map((column) => [column, columnMeta(column)])),
    },
  ),
  qbs: table(
    [
      { qb_id: 'qb-1', qb_name: 'Bo Nix', team: 'DEN', QSaCR: 0.8, QSaOR: 0.7, QSoS: 0.1, QRaw: 0.6 },
      { qb_id: 'qb-2', qb_name: 'Backup Arm', team: 'LV', QSaCR: null, QSaOR: null, QSoS: null, QRaw: null },
    ],
    ['qb_id', 'qb_name', 'team'],
    QB_RATINGS,
    {
      qb_id: columnMeta('QB ID', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      qb_name: columnMeta('QB', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      team: columnMeta('Team', { category: 'Identity', shape: 'id', polarity: 'neutral' }),
      ...Object.fromEntries(QB_RATINGS.map((column) => [column, columnMeta(column)])),
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
