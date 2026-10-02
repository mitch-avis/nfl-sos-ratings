import type { SortingState } from '@tanstack/react-table'

import type { EntityKind, TablePayload } from '@/api/types'

import { getEntityConfig } from './entityConfig'
import { getMetricMetadata } from './metricMetadata'
import { resolveEntityViewState, type EntityViewState, type ResolvedEntityViewState } from './viewModel'

/** The most rows the comparison panel holds at once. */
export const MAX_COMPARED = 4

/** Everything a Teams or QBs page remembers while the user moves between routes. */
export interface EntityPageViewState extends EntityViewState {
  compareIds?: string[]
  query?: string
  showUnratedRows?: boolean
  sorting?: SortingState
}

export interface ResolvedPageViewState {
  compareIds: string[]
  query: string
  showUnratedRows: boolean
  sorting: SortingState
  viewState: ResolvedEntityViewState
}

/** Sort by the entity's headline rating, best first. */
export function buildDefaultSorting(kind: EntityKind): SortingState {
  const config = getEntityConfig(kind)
  return [
    {
      id: config.defaultSortColumn,
      desc: getMetricMetadata(config.defaultSortColumn).polarity !== 'lower',
    },
  ]
}

export function resolvePageViewState(
  kind: EntityKind,
  current?: EntityPageViewState,
): ResolvedPageViewState {
  return {
    compareIds: current?.compareIds ?? [],
    query: current?.query ?? '',
    showUnratedRows: current?.showUnratedRows ?? false,
    sorting: current?.sorting?.length ? current.sorting : buildDefaultSorting(kind),
    viewState: resolveEntityViewState(kind, current),
  }
}

/** Keep only compared IDs that exist in this season's table, in their original order. */
export function reconcileCompareIds(
  kind: EntityKind,
  table: TablePayload,
  compareIds: string[],
): string[] {
  const identityKey = getEntityConfig(kind).identityKey
  const availableIds = new Set(table.rows.map((row) => String(row[identityKey] ?? '')))
  return compareIds.filter((entityId) => availableIds.has(entityId))
}

/** Add or remove one ID, keeping at most `MAX_COMPARED`. */
export function toggleCompareId(compareIds: string[], entityId: string): string[] {
  return compareIds.includes(entityId)
    ? compareIds.filter((value) => value !== entityId)
    : [...compareIds, entityId].slice(0, MAX_COMPARED)
}

function nestedBooleanMapsEqual(
  left: Record<string, Record<string, boolean>>,
  right: Record<string, Record<string, boolean>>,
): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)])
  return Array.from(keys).every((key) => flatBooleanMapsEqual(left[key] ?? {}, right[key] ?? {}))
}

function flatBooleanMapsEqual(left: Record<string, boolean>, right: Record<string, boolean>): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)])
  return Array.from(keys).every((key) => Boolean(left[key]) === Boolean(right[key]))
}

export function viewStatesEqual(left: ResolvedEntityViewState, right: ResolvedEntityViewState): boolean {
  return (
    left.primaryView === right.primaryView &&
    left.teamCategory === right.teamCategory &&
    nestedBooleanMapsEqual(left.teamSubcategories, right.teamSubcategories) &&
    flatBooleanMapsEqual(left.qbSubcategories, right.qbSubcategories)
  )
}

export function sortingStatesEqual(left: SortingState, right: SortingState): boolean {
  return (
    left.length === right.length &&
    left.every((entry, index) => entry.id === right[index]?.id && entry.desc === right[index]?.desc)
  )
}

/** Whether any control differs from the page defaults, so Reset has something to undo. */
export function canResetPageView(kind: EntityKind, state: ResolvedPageViewState): boolean {
  const defaults = resolvePageViewState(kind)
  return (
    !viewStatesEqual(state.viewState, defaults.viewState) ||
    !sortingStatesEqual(state.sorting, defaults.sorting) ||
    state.compareIds.length > 0 ||
    state.query.trim().length > 0 ||
    (kind === 'qbs' && state.showUnratedRows)
  )
}

/** The patch that flips one subcategory in the current view (per category for teams). */
export function toggleSubcategoryPatch(
  kind: EntityKind,
  viewState: ResolvedEntityViewState,
  subcategory: string,
): EntityPageViewState {
  const enabled = !viewState.activeSubcategories[subcategory]
  if (kind === 'teams') {
    return {
      teamSubcategories: {
        ...viewState.teamSubcategories,
        [viewState.teamCategory]: {
          ...viewState.teamSubcategories[viewState.teamCategory],
          [subcategory]: enabled,
        },
      },
    }
  }
  return { qbSubcategories: { ...viewState.qbSubcategories, [subcategory]: enabled } }
}

/** The patch that returns every control to its default. */
export const RESET_PATCH: EntityPageViewState = {
  compareIds: [],
  primaryView: undefined,
  teamCategory: undefined,
  teamSubcategories: undefined,
  qbSubcategories: undefined,
  query: '',
  showUnratedRows: false,
  sorting: undefined,
}
