import { describe, expect, it } from 'vitest'

import { columnMeta } from '@/test/fixtures'

import { hydrateColumnMetadata } from './metricMetadata'
import { buildColumnDecimals, formatColumnValue } from './tableState'

describe('formatColumnValue', () => {
  it('shows a proportion the registry marks as a percentage as a percentage', () => {
    // Arrange
    hydrateColumnMetadata({ third_down_pct: columnMeta('3rd Down %', { shape: 'rate', percent: true }) })
    const decimals = buildColumnDecimals([{ third_down_pct: 0.4567 }, { third_down_pct: 0.5 }], ['third_down_pct'])

    // Act
    const text = formatColumnValue('third_down_pct', 0.4567, decimals.third_down_pct)

    // Assert
    expect(text).toBe('45.7%')
  })

  it('leaves any other rate a plain number', () => {
    // Arrange
    hydrateColumnMetadata({ yards_per_attempt: columnMeta('Y/A', { shape: 'rate' }) })
    const decimals = buildColumnDecimals([{ yards_per_attempt: 7.123 }, { yards_per_attempt: 6.5 }], ['yards_per_attempt'])

    // Act
    const text = formatColumnValue('yards_per_attempt', 7.123, decimals.yards_per_attempt)

    // Assert
    expect(text).toBe('7.12')
  })
})
