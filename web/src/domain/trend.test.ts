import { assert, test } from 'vitest'

import { meanReference, niceTicks, trendColumns } from './trend'

test('meanReference draws the line at the mean of the points shown', () => {
  // Arrange
  const points = [
    { week: 1, value: 1, opponent: 'KC' },
    { week: 2, value: 2.5, opponent: 'LV' },
    { week: 3, value: 6, opponent: 'LAC' },
  ]

  // Act
  const reference = meanReference(points)

  // Assert
  assert.deepEqual(reference, { value: 3.1666666666666665, caption: 'Dashed line: the average of the games shown (3.167).' })
})

test("meanReference states the mean in the column's own format", () => {
  // Arrange
  const points = [
    { week: 1, value: 0.4, opponent: 'KC' },
    { week: 2, value: 0.5, opponent: 'LV' },
  ]

  // Act
  const reference = meanReference(points, (value) => `${(value * 100).toFixed(1)}%`)

  // Assert
  assert.strictEqual(reference?.caption, 'Dashed line: the average of the games shown (45.0%).')
})

test('meanReference draws no line without points', () => {
  // Act
  const reference = meanReference([])

  // Assert
  assert.isNull(reference)
})

test('niceTicks covers the values with round steps', () => {
  // Act
  const ticks = [niceTicks(-3.46, 0), niceTicks(0.12, 0.48), niceTicks(150, 290)]

  // Assert
  assert.deepEqual(ticks, [
    [-4, -3, -2, -1, 0],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [150, 200, 250, 300],
  ])
})

test('niceTicks spreads a flat line around its value', () => {
  // Act
  const ticks = niceTicks(2, 2)

  // Assert
  assert.deepEqual(ticks, [1, 1.5, 2, 2.5, 3])
})

test('trendColumns puts the preferred column first when the rows have it', () => {
  // Arrange
  const rows = [{ week: 1, passing_yards: 210, epa_margin_per_play: 0.12 }]

  // Act
  const columns = trendColumns(rows, ['passing_yards'], ['epa_margin_per_play', 'point_margin'])

  // Assert
  assert.deepEqual(columns, ['epa_margin_per_play', 'passing_yards'])
})

test('trendColumns keeps the view order when no preferred column has values', () => {
  // Arrange
  const rows = [{ week: 1, passing_yards: 210, rushing_yards: 95 }]

  // Act
  const columns = trendColumns(rows, ['passing_yards', 'rushing_yards', 'week'], ['epa_margin_per_play'])

  // Assert
  assert.deepEqual(columns, ['passing_yards', 'rushing_yards'])
})
