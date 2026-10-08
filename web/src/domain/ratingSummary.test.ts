import { describe, expect, it } from 'vitest'

import type { RowValue } from '@/api/types'

import { rankAmong, summaryTiles } from './ratingSummary'

type Row = Record<string, RowValue>

describe('rankAmong', () => {
  it('ranks the highest value first when higher is better', () => {
    // Act
    const rank = rankAmong([3, 1, 2], 2, 'higher')

    // Assert
    expect(rank).toEqual({ rank: 2, count: 3 })
  })

  it('ranks the lowest value first when lower is better', () => {
    // Act
    const rank = rankAmong([3, 1, 2], 1, 'lower')

    // Assert
    expect(rank).toEqual({ rank: 1, count: 3 })
  })

  it('gives tied values the same rank and skips the places they share', () => {
    // Act
    const ranks = [rankAmong([3, 3, 1], 3, 'higher'), rankAmong([3, 3, 1], 1, 'higher')]

    // Assert
    expect(ranks).toEqual([
      { rank: 1, count: 3 },
      { rank: 3, count: 3 },
    ])
  })

  it('leaves missing values out of the count', () => {
    // Act
    const rank = rankAmong([3, null, 2, 'x'], 2, 'higher')

    // Assert
    expect(rank).toEqual({ rank: 2, count: 2 })
  })

  it.each([
    ['a neutral column', 2, 'neutral'],
    ['a missing value', null, 'higher'],
  ] as const)('has no rank for %s', (_case, value, polarity) => {
    // Act
    const rank = rankAmong([3, 1, 2], value, polarity)

    // Assert
    expect(rank).toBeNull()
  })
})

const TEAMS: Row[] = [
  { team: 'DEN', team_rating: 8.4, offense_rating: 3, defense_rating: 5, special_teams_rating: 0.4, sos: 1.2, SRS: 9, epa_margin_per_play: 0.12 },
  { team: 'KC', team_rating: 4, offense_rating: 4, defense_rating: 0, special_teams_rating: 0, sos: -0.5, SRS: 5, epa_margin_per_play: 0.15 },
  { team: 'LV', team_rating: -6, offense_rating: -3, defense_rating: -3, special_teams_rating: 0, sos: 0, SRS: -7, epa_margin_per_play: -0.2 },
]

const QBS: Row[] = [
  { qb_id: 'qb-1', adj_qb_epa_per_dropback: 0.12, qb_epa_per_dropback: 0.1, qb_faced_pass_defense: 0.01, qb_dropbacks_total: 600, qb_is_eligible: true },
  { qb_id: 'qb-3', adj_qb_epa_per_dropback: 0.2, qb_epa_per_dropback: 0.3, qb_faced_pass_defense: 0, qb_dropbacks_total: 30, qb_is_eligible: false },
  { qb_id: 'qb-4', adj_qb_epa_per_dropback: 0.05, qb_epa_per_dropback: 0.02, qb_faced_pass_defense: -0.01, qb_dropbacks_total: 550, qb_is_eligible: true },
]

const polarity = (column: string) => (column === 'sos' || column === 'qb_faced_pass_defense' || column === 'qb_dropbacks_total' ? 'neutral' : 'higher')
const contextual = (column: string) => column === 'sos' || column === 'qb_faced_pass_defense'

describe('summaryTiles', () => {
  it("lists a team's ratings with their ranks and the rating before the adjustment", () => {
    // Act
    const tiles = summaryTiles('teams', TEAMS[0], TEAMS, { polarity, contextual })

    // Assert
    expect(tiles.map((tile) => [tile.column, tile.rank?.rank ?? null])).toEqual([
      ['team_rating', 1],
      ['offense_rating', 2],
      ['defense_rating', 1],
      ['special_teams_rating', 1],
      ['sos', null],
      ['SRS', 1],
    ])
    expect(tiles[0].unadjusted).toEqual({ column: 'epa_margin_per_play', value: 0.12, rank: { rank: 2, count: 3 } })
  })

  it('ranks a quarterback among the qualifiers only', () => {
    // Act
    const tiles = summaryTiles('qbs', QBS[0], QBS, { polarity, contextual })

    // Assert
    expect(tiles[0]).toMatchObject({ column: 'adj_qb_epa_per_dropback', rank: { rank: 1, count: 2 } })
    expect(tiles[0].unadjusted).toEqual({ column: 'qb_epa_per_dropback', value: 0.1, rank: { rank: 1, count: 2 } })
    expect(tiles.map((tile) => tile.column)).toEqual(['adj_qb_epa_per_dropback', 'qb_faced_pass_defense', 'qb_dropbacks_total'])
  })

  it('leaves a quarterback below the qualifier unranked', () => {
    // Act
    const tiles = summaryTiles('qbs', QBS[1], QBS, { polarity, contextual })

    // Assert
    expect(tiles[0].rank).toBeNull()
    expect(tiles[0].unadjusted?.rank).toBeNull()
  })

  it('skips a tile whose column the season does not have', () => {
    // Act
    const tiles = summaryTiles('teams', { team: 'DEN', team_rating: 8.4 }, [{ team: 'DEN', team_rating: 8.4 }], {
      polarity,
      contextual,
    })

    // Assert
    expect(tiles.map((tile) => tile.column)).toEqual(['team_rating'])
    expect(tiles[0].unadjusted).toBeNull()
  })
})
