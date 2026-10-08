import { describe, expect, it } from 'vitest'

import type { MetricRegistryPayload } from '@/api/types'
import { columnMeta } from '@/test/fixtures'

import {
  directionText,
  getMetricFormula,
  getMetricMetadata,
  hydrateColumnMetadata,
  hydrateMetricRegistry,
} from './metricMetadata'

const SEASON_DELTA_RULE = {
  prefix: 'season_delta_',
  label_template: '{label} vs Season',
  full_name_template: '{full_name} vs. Season Baseline',
  description_note: 'This compares the average in these matchups with the full-season average.',
  contextual: false,
  invert_polarity_for_qb: false,
}

describe('getMetricMetadata', () => {
  it('describes a season_delta_ column with the registry rule and its base metric', () => {
    // Arrange
    const registry: MetricRegistryPayload = {
      entities: { team: { categories: [] }, qb: { categories: [] } },
      metrics: {},
      prefix_rules: [SEASON_DELTA_RULE],
    }
    hydrateMetricRegistry(registry)
    hydrateColumnMetadata({
      qb_attempts: columnMeta('Att', {
        full_name: 'Pass Attempts',
        description: 'Official pass attempts.',
        polarity: 'neutral',
        category: 'Passing Volume',
        shape: 'count',
      }),
    })

    // Act
    const metadata = getMetricMetadata('season_delta_qb_attempts')

    // Assert
    expect(metadata).toMatchObject({
      label: 'Att vs Season',
      fullName: 'Pass Attempts vs. Season Baseline',
      detail: 'Official pass attempts. This compares the average in these matchups with the full-season average.',
      polarity: 'neutral',
      contextual: false,
      category: 'Passing Volume',
      shape: 'count',
    })
  })
})

describe('directionText', () => {
  it.each([
    [{ polarity: 'higher' as const }, 'Higher is better.'],
    [{ polarity: 'lower' as const }, 'Lower is better.'],
    [{ polarity: 'higher' as const, contextual: true }, 'Context, not a grade: it describes the opposition, not this team or QB.'],
    [{ polarity: 'neutral' as const }, null],
  ])('reads %o as %s', (overrides, expected) => {
    // Arrange
    hydrateColumnMetadata({ direction_probe: columnMeta('Probe', overrides) })

    // Act
    const text = directionText('direction_probe')

    // Assert
    expect(text).toBe(expected)
  })
})

describe('getMetricFormula', () => {
  it("returns a registry metric's formula", () => {
    // Arrange
    hydrateMetricRegistry({
      entities: { team: { categories: [] }, qb: { categories: [] } },
      metrics: { formula_probe: { ...columnMeta('Probe'), formula: 'yards / plays' } },
      prefix_rules: [],
    } as MetricRegistryPayload)

    // Act
    const formula = getMetricFormula('formula_probe')

    // Assert
    expect(formula).toBe('yards / plays')
  })

  it('has no formula for a column the registry does not list', () => {
    // Act
    const formula = getMetricFormula('opp_formula_probe')

    // Assert
    expect(formula).toBeNull()
  })
})
