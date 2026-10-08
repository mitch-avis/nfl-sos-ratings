import { beforeAll, describe, expect, it } from 'vitest'

import { columnMeta } from '@/test/fixtures'

import { hydrateColumnMetadata } from './metricMetadata'
import { buildColumnDecimals, buildColumnStats, formatColumnValue, getHeatCellStyle, shadedColumns } from './tableState'

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

const ROWS = [
  { team_rating: 8, sos: 2, completions: 30, completion_pct: 0.7 },
  { team_rating: 0, sos: 0, completions: 20, completion_pct: 0.6 },
  { team_rating: -8, sos: -2, completions: 10, completion_pct: 0.5 },
]
const STATS = buildColumnStats(ROWS, ['team_rating', 'sos', 'completions', 'completion_pct'])

describe('getHeatCellStyle', () => {
  beforeAll(() => {
    hydrateColumnMetadata({
      team_rating: columnMeta('Team Rating'),
      sos: columnMeta('SoS', { contextual: true }),
      completions: columnMeta('Comp', { shape: 'count' }),
      completion_pct: columnMeta('Comp %', { shape: 'rate', denominator: 'attempts' }),
    })
  })

  it('shades a context column in one neutral hue, darkest at its tougher end', () => {
    // Act
    const [hardest, middle, easiest] = [2, 0, -2].map((value) => getHeatCellStyle('sos', value, STATS, 'light', 'classic'))

    // Assert
    expect(hardest?.backgroundColor).toBe('rgba(100, 116, 139, 0.3)')
    expect(middle?.backgroundColor).toBe('rgba(100, 116, 139, 0.15)')
    expect(easiest).toBeUndefined()
  })

  it('never paints a context column with the good or bad colors', () => {
    // Act
    const context = getHeatCellStyle('sos', 2, STATS, 'dark', 'KC')
    const grade = getHeatCellStyle('team_rating', 8, STATS, 'dark', 'KC')

    // Assert
    expect(context?.backgroundColor).toBe('rgba(148, 163, 184, 0.3)')
    expect(grade?.backgroundColor).not.toBe(context?.backgroundColor)
  })

  it.each([
    ['Tougher', 'rgba(100, 116, 139, 0.3)'],
    ['Middle', 'rgba(100, 116, 139, 0.15)'],
  ])('shades the %s schedule bucket as context', (bucket, color) => {
    // Act
    const style = getHeatCellStyle('opp_schedule_bucket', bucket, STATS, 'light', 'classic')

    // Assert
    expect(style?.backgroundColor).toBe(color)
  })

  it('leaves the softer schedule bucket unshaded', () => {
    // Act
    const style = getHeatCellStyle('opp_schedule_bucket', 'Softer', STATS, 'light', 'classic')

    // Assert
    expect(style).toBeUndefined()
  })
})

describe('shadedColumns', () => {
  it('keeps rates and drops raw counts, which one opponent cannot compare fairly', () => {
    // Arrange
    hydrateColumnMetadata({
      completions: columnMeta('Comp', { shape: 'count' }),
      completion_pct: columnMeta('Comp %', { shape: 'rate', denominator: 'attempts' }),
      team_rating: columnMeta('Team Rating'),
    })

    // Act
    const columns = shadedColumns(['completions', 'completion_pct', 'team_rating'])

    // Assert
    expect(columns).toEqual(['completion_pct', 'team_rating'])
  })
})
