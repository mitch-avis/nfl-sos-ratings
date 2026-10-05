import { describe, expect, it } from 'vitest'

import type { WpRatingsPayload } from '@/api/types'

import {
  describeWpThreshold,
  formatRankChange,
  formatSignedChange,
  parseWpRatings,
  parseWpThreshold,
  sortWpRows,
  withWpThreshold,
  wpColumns,
} from './wpFilter'

const TEAM_PAYLOAD: WpRatingsPayload = {
  threshold: 10,
  max_threshold: 30,
  rows: [
    {
      team: 'KC',
      team_rank: 2,
      team_rating: 4.1,
      filtered_team_rank: 1,
      filtered_team_rating: 5.3,
      filtered_team_rating_change: 1.2,
      filtered_team_rank_change: -1,
      wp_kept_play_share: 0.78,
    },
    {
      team: 'DEN',
      team_rank: 1,
      team_rating: 6.0,
      filtered_team_rank: 2,
      filtered_team_rating: 4.9,
      filtered_team_rating_change: -1.1,
      filtered_team_rank_change: 1,
      wp_kept_play_share: 0.74,
    },
  ],
  visible_columns: [],
  column_groups: {},
}

describe('parseWpThreshold', () => {
  it.each([
    ['10', 10],
    ['0', 0],
    ['30', 30],
  ])('reads %s as %d', (raw, expected) => {
    // Act
    const threshold = parseWpThreshold(raw)

    // Assert
    expect(threshold).toBe(expected)
  })

  it.each([null, '', '31', '-1', '5.5', 'abc'])('treats %s as no filter', (raw) => {
    // Act
    const threshold = parseWpThreshold(raw)

    // Assert
    expect(threshold).toBe(0)
  })
})

describe('describeWpThreshold', () => {
  it('says the filter is off at zero', () => {
    // Act
    const description = describeWpThreshold(0)

    // Assert
    expect(description).toMatch(/every play/)
  })

  it('names the kept win-probability band', () => {
    // Act
    const description = describeWpThreshold(10)

    // Assert
    expect(description).toMatch(/between 10% and 90%/)
  })
})

describe('parseWpRatings', () => {
  it('reads team rows with published and filtered values', () => {
    // Act
    const rows = parseWpRatings('teams', TEAM_PAYLOAD)

    // Assert
    expect(rows[0]).toEqual({
      id: 'KC',
      label: 'KC',
      team: null,
      publishedRank: 2,
      publishedRating: 4.1,
      filteredRank: 1,
      filteredRating: 5.3,
      ratingChange: 1.2,
      rankChange: -1,
      keptShare: 0.78,
    })
  })

  it('reads QB rows by passer id with the name as the label', () => {
    // Arrange
    const payload: WpRatingsPayload = {
      ...TEAM_PAYLOAD,
      rows: [
        {
          qb_id: '00-1',
          qb_name: 'Drake Maye',
          team: 'NE',
          qb_rank: 1,
          adj_qb_epa_per_dropback: 0.21,
          filtered_qb_rank: 2,
          filtered_adj_qb_epa_per_dropback: 0.196,
          filtered_adj_qb_epa_per_dropback_change: -0.014,
          filtered_qb_rank_change: 1,
          wp_kept_dropback_share: 0.88,
        },
      ],
    }

    // Act
    const rows = parseWpRatings('qbs', payload)

    // Assert
    expect(rows[0]).toMatchObject({ id: '00-1', label: 'Drake Maye', team: 'NE', filteredRank: 2 })
  })
})

describe('sortWpRows', () => {
  it('orders by the biggest rating gain first', () => {
    // Arrange
    const rows = parseWpRatings('teams', TEAM_PAYLOAD)

    // Act
    const sorted = sortWpRows(rows, { key: 'ratingChange', descending: true })

    // Assert
    expect(sorted.map((row) => row.id)).toEqual(['KC', 'DEN'])
  })

  it('orders by published rank, best first', () => {
    // Arrange
    const rows = parseWpRatings('teams', TEAM_PAYLOAD)

    // Act
    const sorted = sortWpRows(rows, { key: 'publishedRank', descending: false })

    // Assert
    expect(sorted.map((row) => row.id)).toEqual(['DEN', 'KC'])
  })
})

describe('formatRankChange', () => {
  it.each([
    [-2, 'up 2'],
    [1, 'down 1'],
    [0, 'same'],
    [null, '—'],
  ])('describes a rank change of %s as %s', (change, expected) => {
    // Act
    const text = formatRankChange(change)

    // Assert
    expect(text).toBe(expected)
  })
})

describe('withWpThreshold', () => {
  it('adds the threshold to a link', () => {
    // Act
    const link = withWpThreshold('/teams/NE?season=2025', 10)

    // Assert
    expect(link).toBe('/teams/NE?season=2025&wp=10')
  })

  it('leaves a link alone with the filter off', () => {
    // Act
    const link = withWpThreshold('/teams/NE?season=2025', 0)

    // Assert
    expect(link).toBe('/teams/NE?season=2025')
  })
})

describe('formatSignedChange', () => {
  it.each([
    [1.234, 2, '+1.23'],
    [-0.5, 2, '-0.50'],
    [0, 2, '0.00'],
    [null, 2, '—'],
  ])('shows %s with %d decimals as %s', (value, decimals, expected) => {
    // Act
    const text = formatSignedChange(value, decimals)

    // Assert
    expect(text).toBe(expected)
  })
})

describe('wpColumns', () => {
  it('names the QB rating, rank, and kept-share columns', () => {
    // Act
    const columns = wpColumns('qbs')

    // Assert
    expect(columns).toMatchObject({
      rating: 'adj_qb_epa_per_dropback',
      rank: 'qb_rank',
      kept: 'wp_kept_dropback_share',
    })
  })
})
