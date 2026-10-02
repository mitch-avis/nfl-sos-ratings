import { describe, expect, it } from 'vitest'

import { SEASON_2025 } from '@/test/fixtures'

import {
  MAX_COMPARED,
  RESET_PATCH,
  canResetPageView,
  reconcileCompareIds,
  resolvePageViewState,
  toggleCompareId,
  toggleSubcategoryPatch,
} from './pageViewState'
import { getQuarterbackQualifierAttempts, getRegularSeasonGameCount } from './seasonRules'
import { buildTrendPoints } from './trend'

describe('page view state', () => {
  it('defaults to the ratings view sorted by the headline rating, best first', () => {
    const state = resolvePageViewState('teams')
    expect(state.viewState.primaryView).toBe('ratings')
    expect(state.sorting).toEqual([{ id: 'SaCR', desc: true }])
    expect(canResetPageView('teams', state)).toBe(false)
  })

  it('caps the comparison and toggles IDs off again', () => {
    let ids: string[] = []
    for (const id of ['A', 'B', 'C', 'D', 'E']) ids = toggleCompareId(ids, id)
    expect(ids).toHaveLength(MAX_COMPARED)
    expect(toggleCompareId(ids, 'B')).toEqual(['A', 'C', 'D'])
  })

  it('drops compared IDs missing from the season and keeps the order', () => {
    expect(reconcileCompareIds('teams', SEASON_2025.teams, ['LV', 'NOPE', 'DEN'])).toEqual(['LV', 'DEN'])
  })

  it('lets reset undo any changed control', () => {
    expect(canResetPageView('teams', resolvePageViewState('teams', { query: 'den' }))).toBe(true)
    expect(canResetPageView('qbs', resolvePageViewState('qbs', { showUnratedRows: true }))).toBe(true)
    expect(canResetPageView('teams', resolvePageViewState('teams', { primaryView: 'per_game_rates' }))).toBe(true)
    expect(canResetPageView('teams', resolvePageViewState('teams', RESET_PATCH))).toBe(false)
  })

  it('toggles a subcategory within the active team category only', () => {
    const viewState = resolvePageViewState('teams', { teamCategory: 'Offense' }).viewState
    const subcategory = viewState.activeSubcategoryOptions[0]
    const patch = toggleSubcategoryPatch('teams', viewState, subcategory)
    expect(patch.teamSubcategories?.Offense?.[subcategory]).toBe(false)
    expect(patch.teamSubcategories?.Defense).toEqual(viewState.teamSubcategories.Defense)
  })
})

describe('season rules', () => {
  it('uses 17 games from 2021 and 14 qualifying attempts per game', () => {
    expect(getRegularSeasonGameCount(2020)).toBe(16)
    expect(getRegularSeasonGameCount(2021)).toBe(17)
    expect(getQuarterbackQualifierAttempts(2025)).toBe(238)
  })
})

describe('weekly trend points', () => {
  it('orders numeric values by week and skips missing ones', () => {
    const points = buildTrendPoints(
      [
        { week: 3, value: 1.5, opponent_team: 'LAC' },
        { week: 1, value: 0.5, opponent_team: 'TEN' },
        { week: 2, value: null, opponent_team: 'IND' },
      ],
      'value',
    )
    expect(points).toEqual([
      { week: 1, value: 0.5, opponent: 'TEN' },
      { week: 3, value: 1.5, opponent: 'LAC' },
    ])
  })
})
