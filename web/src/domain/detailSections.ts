import type { EntityKind } from '@/api/types'

import { orderedExisting } from './detailAnalytics'
import { getColumnSection, getMetricMetadata } from './metricMetadata'

const TEAM_RATING_ORDER = [
  'team_rating',
  'offense_rating',
  'defense_rating',
  'special_teams_rating',
  'sos',
  'SRS',
]
const QB_RATING_ORDER = ['adj_qb_epa_per_dropback', 'qb_epa_per_dropback', 'qb_faced_pass_defense']

interface Section {
  title: string
  columns: string[]
}

/** Ratings first in their preferred order, then any other rating columns. */
function buildRatingsColumns(kind: EntityKind, columns: string[]): string[] {
  const ordered = orderedExisting(columns, kind === 'teams' ? TEAM_RATING_ORDER : QB_RATING_ORDER)
  return [...ordered, ...columns.filter((column) => !ordered.includes(column))]
}

/** Group columns by the registry's category taxonomy, or return null if any column is unknown. */
function bucketColumnsByRegistry(kind: EntityKind, columns: string[]): Section[] | null {
  const entity = kind === 'teams' ? 'team' : 'qb'
  const sections = new Map<string, { title: string; rank: number; columns: string[] }>()
  for (const column of columns) {
    const section = getColumnSection(entity, column)
    if (!section) return null
    const current = sections.get(section.title) ?? { ...section, columns: [] }
    current.columns.push(column)
    sections.set(section.title, current)
  }
  return Array.from(sections.values())
    .sort((left, right) => left.rank - right.rank)
    .map(({ title, columns: sectionColumns }) => ({ title, columns: sectionColumns }))
}

export function bucketColumns(kind: EntityKind, isRatingsView: boolean, columns: string[]): Section[] {
  if (isRatingsView) return [{ title: 'Season Ratings', columns: buildRatingsColumns(kind, columns) }]
  return bucketColumnsByRegistry(kind, columns) ?? [{ title: 'Metrics', columns }]
}

/** A detail-page label, with short names for the two result columns. */
export function detailHeaderLabel(column: string): string {
  if (column === 'win_value') return 'Outcome'
  if (column === 'turnover_margin') return 'T/O Margin'
  return getMetricMetadata(column).label
}
