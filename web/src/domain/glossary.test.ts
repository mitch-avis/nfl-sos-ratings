import { describe, expect, it } from 'vitest'

import type { MetricRegistryPayload, RegistryMetricPayload } from '@/api/types'
import { columnMeta } from '@/test/fixtures'

import { buildGlossary, filterGlossary } from './glossary'

function metric(label: string, overrides: Partial<RegistryMetricPayload> = {}): RegistryMetricPayload {
  return { ...columnMeta(label), entity: 'team', formula: null, note: null, since: 1999, duplicate_of: null, ...overrides }
}

const REGISTRY: MetricRegistryPayload = {
  entities: {
    team: {
      categories: [
        { name: 'Passing', description: 'Throwing the ball.', subcategories: [] },
        { name: 'Rushing', description: 'Running the ball.', subcategories: [] },
      ],
    },
    qb: { categories: [] },
  },
  metrics: {
    team_rating: metric('Team Rating', { category: 'Schedule-Adjusted Ratings', description: 'Points per game better than average.' }),
    rush_yards: metric('Rush Yds', { full_name: 'Rushing Yards', category: 'Rushing', description: 'Yards on carries.' }),
    sacks_taken: metric('Sacks', { full_name: 'Sacks Taken', category: 'Passing', polarity: 'lower', description: 'Times the passer was sacked.', since: 2006 }),
    completion_pct: metric('Comp %', { full_name: 'Completion Percentage', category: 'Passing', description: 'Completions per attempt.', formula: 'completions / attempts' }),
    receptions: metric('Rec', { full_name: 'Receptions', category: 'Passing', description: 'Catches.', duplicate_of: 'completions' }),
    completions: metric('Comp', { full_name: 'Completions', category: 'Passing', description: 'Passes caught.' }),
    qb_sack_rate: metric('Sack Rate', { entity: 'qb', category: 'Pressure, Sacks & Pocket', polarity: 'lower', description: 'Sacks per dropback.' }),
  },
  prefix_rules: [],
}

describe('buildGlossary', () => {
  it('starts with the ideas behind the pages and the headline ratings', () => {
    // Act
    const [startHere] = buildGlossary(REGISTRY)

    // Assert
    expect(startHere.title).toBe('Start here')
    expect(startHere.entries.map((entry) => entry.title)).toEqual(
      expect.arrayContaining(['Rank range', 'Head-to-head chance', 'Schedule strength', 'Team Rating']),
    )
  })

  it("groups the other metrics by entity and the registry's category order, unknown categories last", () => {
    // Act
    const sections = buildGlossary(REGISTRY).slice(1)

    // Assert
    expect(sections.map((section) => section.title)).toEqual([
      'Teams: Passing',
      'Teams: Rushing',
      'Quarterbacks: Pressure, Sacks & Pocket',
    ])
    expect(sections[0].description).toBe('Throwing the ball.')
  })

  it('lists each headline rating once, in Start here only', () => {
    // Act
    const titles = buildGlossary(REGISTRY).flatMap((section) => section.entries.map((entry) => entry.title))

    // Assert
    expect(titles.filter((title) => title === 'Team Rating')).toHaveLength(1)
  })

  it('carries the table label, direction, formula, first season, and duplicate', () => {
    // Act
    const passing = buildGlossary(REGISTRY).find((section) => section.title === 'Teams: Passing')

    // Assert
    const byTitle = new Map(passing?.entries.map((entry) => [entry.title, entry]))
    expect(byTitle.get('Completion Percentage')).toMatchObject({ shownAs: 'Comp %', formula: 'completions / attempts', direction: 'Higher is better.' })
    expect(byTitle.get('Sacks Taken')).toMatchObject({ direction: 'Lower is better.', since: 2006 })
    expect(byTitle.get('Receptions')?.sameAs).toBe('Completions')
    expect(byTitle.get('Completions')?.since).toBeNull()
  })
})

describe('filterGlossary', () => {
  it('keeps the entries whose name, label, or description matches, in any case', () => {
    // Arrange
    const glossary = buildGlossary(REGISTRY)

    // Act
    const filtered = filterGlossary(glossary, 'SACK')

    // Assert
    expect(filtered.flatMap((section) => section.entries.map((entry) => entry.title))).toEqual(['Sacks Taken', 'Sack Rate'])
  })

  it('returns every section for an empty query', () => {
    // Arrange
    const glossary = buildGlossary(REGISTRY)

    // Act
    const filtered = filterGlossary(glossary, '  ')

    // Assert
    expect(filtered).toEqual(glossary)
  })
})
