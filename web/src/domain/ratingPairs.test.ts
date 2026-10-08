import { describe, expect, it } from 'vitest'

import { ApiError } from '@/api/client'
import type { TablePayload } from '@/api/types'

import {
  describeRatingPair,
  isMissingRatingPairs,
  neighborId,
  parseRatingPairs,
  type RatingPair,
} from './ratingPairs'

const TEAM_PAYLOAD: TablePayload = {
  rows: [
    {
      team: 'NE',
      other_team: 'BUF',
      team_rated_above_probability: 0.38,
      team_pair_share: 1,
      team_rating_gap_q025: -5.0,
      team_rating_gap_q500: -1.2,
      team_rating_gap_q975: 2.8,
    },
  ],
  visible_columns: [],
  column_groups: {},
  column_metadata: {},
}

const QB_PAIR: RatingPair = {
  otherId: 'qb-2',
  aboveChance: 0.82,
  gapLow: -0.01,
  gapMid: 0.045,
  gapHigh: 0.101,
  share: 0.87,
}

describe('parseRatingPairs', () => {
  it('reads the compared unit, the chance, the gap percentiles, and the share', () => {
    // Act
    const pairs = parseRatingPairs('teams', TEAM_PAYLOAD)

    // Assert
    expect(pairs).toEqual([
      { otherId: 'BUF', aboveChance: 0.38, gapLow: -5.0, gapMid: -1.2, gapHigh: 2.8, share: 1 },
    ])
  })
})

describe('describeRatingPair', () => {
  it('says how often a team was rated above another and by how much', () => {
    // Arrange
    const [pair] = parseRatingPairs('teams', TEAM_PAYLOAD)

    // Act
    const sentence = describeRatingPair('teams', 'NE', 'BUF', pair)

    // Assert
    expect(sentence).toBe(
      'NE rated above BUF in 38% of redraws. Typical gap (NE minus BUF): -1.2 points per game; 95% of redraws: -5.0 to +2.8.',
    )
  })

  it('counts only the resamples with both quarterbacks and says how many those were', () => {
    // Act
    const sentence = describeRatingPair('qbs', 'Drake Maye', 'Josh Allen', QB_PAIR)

    // Assert
    expect(sentence).toBe(
      'Drake Maye rated above Josh Allen in 82% of the redraws with both. Typical gap (Drake Maye minus Josh Allen): +0.045 EPA per dropback; 95% of redraws: -0.010 to +0.101. Both appeared in 87% of redraws.',
    )
  })

  it('says when two units never shared a resampled season', () => {
    // Arrange
    const pair: RatingPair = { ...QB_PAIR, aboveChance: null, gapLow: null, gapMid: null, gapHigh: null, share: 0 }

    // Act
    const sentence = describeRatingPair('qbs', 'One', 'Two', pair)

    // Assert
    expect(sentence).toBe('One and Two never appeared in the same redraw.')
  })
})

describe('neighborId', () => {
  it('picks the unit ranked just above', () => {
    // Act
    const neighbor = neighborId(['A', 'B', 'C'], 'B')

    // Assert
    expect(neighbor).toBe('A')
  })

  it('picks the unit ranked just below the leader', () => {
    // Act
    const neighbor = neighborId(['A', 'B', 'C'], 'A')

    // Assert
    expect(neighbor).toBe('B')
  })

  it('has no neighbor for a unit ranked alone', () => {
    // Act
    const neighbor = neighborId(['A'], 'A')

    // Assert
    expect(neighbor).toBeNull()
  })

  it('has no neighbor for a unit missing from the ranking', () => {
    // Act
    const neighbor = neighborId(['A', 'B'], 'Z')

    // Assert
    expect(neighbor).toBeNull()
  })
})

describe('isMissingRatingPairs', () => {
  it('treats only a not-found answer as a season without pairs', () => {
    // Arrange
    const errors = [new ApiError(404, 'missing'), new ApiError(500, 'broken'), new Error('network')]

    // Act
    const missing = errors.map(isMissingRatingPairs)

    // Assert
    expect(missing).toEqual([true, false, false])
  })
})
