import { assert, test } from 'vitest'

import { meanReference } from './trend'

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
  assert.deepEqual(reference, { value: 3.1666666666666665, caption: 'Dashed line: the mean of the games shown (3.167).' })
})

test('meanReference draws no line without points', () => {
  // Act
  const reference = meanReference([])

  // Assert
  assert.isNull(reference)
})
