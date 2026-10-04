import { assert, test } from 'vitest'

import { ApiError } from '@/api/client'
import type { TablePayload } from '@/api/types'

import { buildRatingHistoryChart, isMissingRatingHistory } from './ratingHistory'

function history(weeks: number[], ratings: string[]): TablePayload {
  return {
    rows: weeks.map((week) => ({ week, team: 'DEN', team_rating: week * 1.5 })),
    visible_columns: ['week', 'team', ...ratings],
    column_groups: { identity: ['week', 'team'], sample: [], ratings },
  }
}

test('buildRatingHistoryChart charts the rating columns against an average team', () => {
  // Arrange
  const payload = history([1, 2, 3], ['team_rating', 'offense_rating'])

  // Act
  const chart = buildRatingHistoryChart('teams', payload)

  // Assert
  assert.deepEqual(chart?.columns, ['team_rating', 'offense_rating'])
  assert.deepEqual(chart?.reference, { value: 0, caption: 'Dashed line: an average team (0).' })
})

test('buildRatingHistoryChart draws no reference line for quarterbacks', () => {
  // Arrange
  const payload = history([1, 2], ['adj_qb_epa_per_dropback'])

  // Act
  const chart = buildRatingHistoryChart('qbs', payload)

  // Assert
  assert.isNull(chart?.reference)
})

test('buildRatingHistoryChart needs at least two weeks to draw a line', () => {
  // Arrange
  const payload = history([1], ['team_rating'])

  // Act
  const chart = buildRatingHistoryChart('teams', payload)

  // Assert
  assert.isNull(chart)
})

test.each([
  [new ApiError(404, 'Season 2001 is missing UI contract files: 2001_ratings_by_week.parquet'), true],
  [new ApiError(500, 'Internal Server Error'), false],
  [new Error('network down'), false],
])('isMissingRatingHistory(%s) is %s', (error, expected) => {
  // Act
  const missing = isMissingRatingHistory(error)

  // Assert
  assert.strictEqual(missing, expected)
})
