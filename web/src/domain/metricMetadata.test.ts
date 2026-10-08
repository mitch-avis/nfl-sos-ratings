import { describe, expect, it } from 'vitest'

import type { MetricRegistryPayload } from '@/api/types'
import { columnMeta } from '@/test/fixtures'

import { getMetricMetadata, hydrateColumnMetadata, hydrateMetricRegistry } from './metricMetadata'

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
