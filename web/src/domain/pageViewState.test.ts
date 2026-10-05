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
import {
  getInProgressGames,
  getRegularSeasonGameCount,
} from './seasonRules'
import { buildTrendPoints } from './trend'

describe('page view state', () => {
  it('defaults to the ratings view sorted by the headline rating, best first', () => {
    // Act
    const state = resolvePageViewState('teams')

    // Assert
    expect(state.viewState.primaryView).toBe('ratings')
    expect(state.sorting).toEqual([{ id: 'team_rating', desc: true }])
    expect(canResetPageView('teams', state)).toBe(false)
  })

  it('caps the comparison at the maximum', () => {
    // Arrange
    let ids: string[] = []

    // Act
    for (const id of ['A', 'B', 'C', 'D', 'E']) ids = toggleCompareId(ids, id)

    // Assert
    expect(ids).toHaveLength(MAX_COMPARED)
  })

  it('toggles a compared ID off again', () => {
    // Act
    const ids = toggleCompareId(['A', 'B', 'C', 'D'], 'B')

    // Assert
    expect(ids).toEqual(['A', 'C', 'D'])
  })

  it('drops compared IDs missing from the season and keeps the order', () => {
    // Act
    const ids = reconcileCompareIds('teams', SEASON_2025.teams, ['LV', 'NOPE', 'DEN'])

    // Assert
    expect(ids).toEqual(['LV', 'DEN'])
  })

  it.each([
    ['teams', { query: 'den' }, true],
    ['qbs', { showUnratedRows: true }, true],
    ['teams', { primaryView: 'per_game_rates' }, true],
    ['teams', RESET_PATCH, false],
  ] as const)('reports whether %s view %o can be reset', (kind, patch, expected) => {
    // Arrange
    const state = resolvePageViewState(kind, patch)

    // Act
    const resettable = canResetPageView(kind, state)

    // Assert
    expect(resettable).toBe(expected)
  })

  it('toggles a subcategory within the active team category only', () => {
    // Arrange
    const viewState = resolvePageViewState('teams', { teamCategory: 'Offense' }).viewState
    const subcategory = viewState.activeSubcategoryOptions[0]

    // Act
    const patch = toggleSubcategoryPatch('teams', viewState, subcategory)

    // Assert
    expect(patch.teamSubcategories?.Offense?.[subcategory]).toBe(false)
    expect(patch.teamSubcategories?.Defense).toEqual(viewState.teamSubcategories.Defense)
  })
})

describe('season rules', () => {
  it.each([
    [2020, 16],
    [2021, 17],
  ])('counts %i regular-season games as %i', (season, games) => {
    // Act
    const count = getRegularSeasonGameCount(season)

    // Assert
    expect(count).toBe(games)
  })

  it('reports games played so far while a season is in progress', () => {
    // Arrange
    const rows = [{ team: 'NE', games_played: 3 }, { team: 'DEN', games_played: 4 }]

    // Act
    const games = getInProgressGames(2026, rows)

    // Assert
    expect(games).toBe(4)
  })

  it('reports nothing once every team has a full season', () => {
    // Arrange
    const rows = [{ team: 'NE', games_played: 17 }, { team: 'DEN', games_played: 17 }]

    // Act
    const games = getInProgressGames(2025, rows)

    // Assert
    expect(games).toBeNull()
  })

  it('reports nothing when games played is unknown', () => {
    // Act
    const games = getInProgressGames(2026, [{ team: 'NE' }])

    // Assert
    expect(games).toBeNull()
  })

  it('keeps a season in progress until the last team finishes', () => {
    // Arrange
    const rows = [{ games_played: 17 }, { games_played: 16 }]

    // Act
    const games = getInProgressGames(2025, rows)

    // Assert
    expect(games).toBe(17)
  })
})

describe('weekly trend points', () => {
  it('orders numeric values by week and skips missing ones', () => {
    // Arrange
    const rows = [
      { week: 3, value: 1.5, opponent_team: 'LAC' },
      { week: 1, value: 0.5, opponent_team: 'TEN' },
      { week: 2, value: null, opponent_team: 'IND' },
    ]

    // Act
    const points = buildTrendPoints(rows, 'value')

    // Assert
    expect(points).toEqual([
      { week: 1, value: 0.5, opponent: 'TEN' },
      { week: 3, value: 1.5, opponent: 'LAC' },
    ])
  })
})
