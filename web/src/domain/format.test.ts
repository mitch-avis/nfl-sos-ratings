import { describe, expect, it } from 'vitest'

import { columnDecimals, countLabel, formatFixed } from './format'

describe('columnDecimals', () => {
  it.each([
    { name: 'points-per-game scores', values: [9.944, -0.75, 3.5], shape: 'score' as const, expected: 2 },
    { name: 'per-play rates below 1', values: [0.21, -0.05], shape: 'rate' as const, expected: 3 },
    { name: 'values below 100', values: [7.12, 24.5], shape: 'avg' as const, expected: 2 },
    { name: 'large values', values: [245.3, 198.75], shape: 'count' as const, expected: 1 },
    { name: 'whole numbers', values: [12, 3, null], shape: 'count' as const, expected: 0 },
  ])('gives $name $expected decimals', ({ values, shape, expected }) => {
    // Act
    const decimals = columnDecimals(values, shape)

    // Assert
    expect(decimals).toBe(expected)
  })

  it('has no rule for a column without numbers', () => {
    // Act
    const decimals = columnDecimals(['DEN', null], 'id')

    // Assert
    expect(decimals).toBeNull()
  })
})

describe('formatFixed', () => {
  it.each([
    { value: 0.75, decimals: 3, expected: '0.750' },
    { value: 1234.56, decimals: 1, expected: '1,234.6' },
    { value: -0.0004, decimals: 3, expected: '0.000' },
    { value: null, decimals: 2, expected: '—' },
    { value: 'DEN', decimals: null, expected: 'DEN' },
  ])('formats $value with $decimals decimals as $expected', ({ value, decimals, expected }) => {
    // Act
    const text = formatFixed(value, decimals)

    // Assert
    expect(text).toBe(expected)
  })
})

describe('percentages', () => {
  it.each([
    { name: 'proportions with tenths of a percent', values: [0.653, 0.6, null], expected: 1 },
    { name: 'whole percentages', values: [0.29, 0.5, 1, 0], expected: 0 },
  ])('gives $name $expected decimals', ({ values, expected }) => {
    // Act
    const decimals = columnDecimals(values, 'rate', true)

    // Assert
    expect(decimals).toBe(expected)
  })

  it.each([
    { value: 0.653, decimals: 1, expected: '65.3%' },
    { value: 1, decimals: 0, expected: '100%' },
    { value: -0.0004, decimals: 1, expected: '0.0%' },
    { value: -0.032, decimals: 1, expected: '-3.2%' },
    { value: 0.0153, decimals: null, expected: '1.5%' },
    { value: null, decimals: 1, expected: '—' },
  ])('shows the proportion $value as $expected', ({ value, decimals, expected }) => {
    // Act
    const text = formatFixed(value, decimals, true)

    // Assert
    expect(text).toBe(expected)
  })
})

describe('countLabel', () => {
  it.each([
    [1, 'game', undefined, '1 game'],
    [4, 'game', undefined, '4 games'],
    [0, 'opponent', undefined, '0 opponents'],
    [1, 'match', 'matches', '1 match'],
    [2, 'match', 'matches', '2 matches'],
  ])('labels %i %s', (count, singular, plural, expected) => {
    // Act
    const label = countLabel(count, singular, plural)

    // Assert
    expect(label).toBe(expected)
  })
})
