import { describe, expect, it } from 'vitest'

import { ApiError } from '@/api/client'
import type { TablePayload } from '@/api/types'

import {
  describeRankHistory,
  isMissingRankHistory,
  parseRankHistory,
  rankAxisMax,
  rankBandData,
} from './rankHistory'

function week(number: number, ranks: [number, number, number, number, number, number, number], published: number) {
  const keys = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']
  return {
    week: number,
    team: 'NE',
    team_rank: published,
    ...Object.fromEntries(keys.map((key, index) => [`team_rank_${key}`, ranks[index]])),
  }
}

const PAYLOAD: TablePayload = {
  rows: [week(2, [4, 5, 6, 9, 13, 17, 20], 8), week(1, [3, 5, 8, 12, 18, 24, 28], 11)],
  visible_columns: [],
  column_groups: {},
  column_metadata: {},
}

describe('parseRankHistory', () => {
  it('reads each week in order with the median and both bands', () => {
    // Act
    const points = parseRankHistory('teams', PAYLOAD)

    // Assert
    expect(points).toEqual([
      { week: 1, published: 11, median: 12, low50: 8, high50: 18, low95: 3, high95: 28 },
      { week: 2, published: 8, median: 9, low50: 6, high50: 13, low95: 4, high95: 20 },
    ])
  })
})

describe('rankBandData', () => {
  it('turns each week into the range pairs the band areas draw', () => {
    // Arrange
    const points = parseRankHistory('teams', PAYLOAD)

    // Act
    const data = rankBandData(points)

    // Assert
    expect(data[0]).toEqual({ week: 1, median: 12, band50: [8, 18], band95: [3, 28] })
  })
})

describe('describeRankHistory', () => {
  it('gives the first and latest week as a text alternative', () => {
    // Arrange
    const points = parseRankHistory('teams', PAYLOAD)

    // Act
    const text = describeRankHistory(points)

    // Assert
    expect(text).toBe('Median rank 12th in week 1 (95%: 3rd–28th) and 9th in week 2 (95%: 4th–20th).')
  })

  it('describes a single week on its own', () => {
    // Arrange
    const [first] = parseRankHistory('teams', PAYLOAD)

    // Act
    const text = describeRankHistory([first])

    // Assert
    expect(text).toBe('Median rank 12th in week 1 (95%: 3rd–28th).')
  })
})

describe('rankAxisMax', () => {
  it('reaches the worst rank any week could take', () => {
    // Act
    const max = rankAxisMax(parseRankHistory('teams', PAYLOAD))

    // Assert
    expect(max).toBe(28)
  })
})

describe('isMissingRankHistory', () => {
  it('treats only a not-found answer as a season without weekly ranges', () => {
    // Arrange
    const errors = [new ApiError(404, 'missing'), new ApiError(500, 'broken')]

    // Act
    const missing = errors.map(isMissingRankHistory)

    // Assert
    expect(missing).toEqual([true, false])
  })
})
