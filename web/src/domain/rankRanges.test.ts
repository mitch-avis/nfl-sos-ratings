import { assert, test } from 'vitest'

import { ApiError } from '@/api/client'
import type { RankRangesPayload } from '@/api/types'

import {
  formatChance,
  isMissingRankRanges,
  ordinal,
  parseRankRanges,
  rankCenter,
  rankChanceText,
  rankHistogram,
  rankRangeHeadline,
  rankRangeSummary,
  rankRangesByMedian,
  rankSpan,
  rankTicks,
  type RankRange,
} from './rankRanges'

const QUANTILES = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975'] as const

function teamRow(team: string, published: number, ranks: number[], probabilities: number[]) {
  return {
    team,
    team_rank: published,
    ...Object.fromEntries(QUANTILES.map((key, index) => [`team_rank_${key}`, ranks[index]])),
    ...Object.fromEntries(QUANTILES.map((key, index) => [`team_rating_${key}`, index - 3])),
    team_rank_top5_probability: 0.42,
    team_rank_top10_probability: 0.81,
    team_rank_missing_share: 0,
    team_rank_probabilities: probabilities,
  }
}

const TEAMS: RankRangesPayload = {
  rows: [
    teamRow('DEN', 1, [1, 1, 1, 2, 2, 3, 3], [0.5, 0.3, 0.2]),
    teamRow('KC', 2, [1, 1, 2, 2, 3, 3, 3], [0.3, 0.4, 0.3]),
    teamRow('LV', 3, [1, 2, 2, 3, 3, 3, 3], [0.2, 0.3, 0.5]),
  ],
  visible_columns: [],
  column_groups: {},
}

function range(overrides: Partial<RankRange> = {}): RankRange {
  return { ...parseRankRanges('teams', TEAMS)[0], ...overrides }
}

test('parseRankRanges reads the published rank, quantiles, and rank chances', () => {
  // Act
  const [den] = parseRankRanges('teams', TEAMS)

  // Assert
  assert.deepEqual(
    [den.id, den.label, den.publishedRank, den.rank.q250, den.rank.q975, den.rating.q500, den.top5],
    ['DEN', 'DEN', 1, 1, 3, 0, 0.42],
  )
  assert.deepEqual(den.probabilities, [0.5, 0.3, 0.2])
})

test('parseRankRanges labels quarterbacks by name and keeps their team', () => {
  // Arrange
  const payload: RankRangesPayload = {
    rows: [{ qb_id: 'qb-1', qb_name: 'B.Nix', team: 'DEN', qb_rank: 4, qb_rank_q500: 5, qb_rank_missing_share: 0.1 }],
    visible_columns: [],
    column_groups: {},
  }

  // Act
  const [nix] = parseRankRanges('qbs', payload)

  // Assert
  assert.deepEqual([nix.id, nix.label, nix.team, nix.rank.q500, nix.rank.q025, nix.missingShare], [
    'qb-1',
    'B.Nix',
    'DEN',
    5,
    null,
    0.1,
  ])
  assert.deepEqual(nix.probabilities, [])
})

test('ordinal uses th for the teens and st, nd, rd otherwise', () => {
  // Act
  const ordinals = [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 32].map(ordinal)

  // Assert
  assert.deepEqual(ordinals, ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '32nd'])
})

test('rankRangeHeadline gives the published rank and the middle 50% and 95% ranges', () => {
  // Arrange
  const den = range({ publishedRank: 6, rank: { ...range().rank, q025: 1, q250: 4, q750: 8, q975: 16 } })

  // Act
  const headline = rankRangeHeadline(den)

  // Assert
  assert.equal(headline, '6th; middle 50%: 4th–8th; 95%: 1st–16th')
})

test('rankRangeHeadline names a single rank once', () => {
  // Arrange
  const den = range({ rank: { ...range().rank, q025: 1, q250: 1, q750: 1, q975: 2 } })

  // Act
  const headline = rankRangeHeadline(den)

  // Assert
  assert.equal(headline, '1st; middle 50%: 1st; 95%: 1st–2nd')
})

test('rankRangeHeadline says when no resample ranked the subject', () => {
  // Arrange
  const empty = range({ rank: Object.fromEntries(QUANTILES.map((key) => [key, null])) as RankRange['rank'] })

  // Act
  const headline = rankRangeHeadline(empty)

  // Assert
  assert.equal(headline, '1st; not ranked in any resample')
})

test('rankRangeSummary names the subject and its median for screen readers', () => {
  // Act
  const summary = rankRangeSummary(range())

  // Assert
  assert.equal(summary, 'DEN: published rank 1st, median 2nd; middle 50%: 1st–2nd; 95%: 1st–3rd')
})

test('formatChance never rounds a possible outcome to 0% or 100%', () => {
  // Act
  const chances = [0, 0.004, 0.42, 0.996, 1].map(formatChance)

  // Assert
  assert.deepEqual(chances, ['0%', '<1%', '42%', '>99%', '100%'])
})

test('rankChanceText adds how often a quarterback was missing', () => {
  // Arrange
  const qb = range({ missingShare: 0.12 })

  // Act
  const text = rankChanceText('qbs', qb)

  // Assert
  assert.equal(text, 'Top 5 in 42% of resamples, top 10 in 81%; no dropbacks in 12%')
})

test('rankChanceText leaves the missing share out when nothing was missing', () => {
  // Act
  const text = rankChanceText('teams', range())

  // Assert
  assert.equal(text, 'Top 5 in 42% of resamples, top 10 in 81%')
})

test('rankSpan covers whole rank cells from the low rank to the high rank', () => {
  // Act
  const span = rankSpan(2, 3, 4)

  // Assert
  assert.deepEqual(span, { left: 25, width: 50 })
})

test('rankCenter puts a rank in the middle of its cell', () => {
  // Act
  const center = rankCenter(1, 4)

  // Assert
  assert.equal(center, 12.5)
})

test('rankTicks steps by five and ends at the last rank', () => {
  // Act
  const ticks = rankTicks(37)

  // Assert
  assert.deepEqual(ticks, [1, 5, 10, 15, 20, 25, 30, 37])
})

test('rankTicks drops a step within three ranks of the last rank', () => {
  // Act
  const ticks = rankTicks(32)

  // Assert
  assert.deepEqual(ticks, [1, 5, 10, 15, 20, 25, 32])
})

test('rankRangesByMedian orders by median rank, then published rank', () => {
  // Arrange
  const ranges = parseRankRanges('teams', TEAMS)

  // Act
  const ordered = rankRangesByMedian([ranges[2], ranges[1], ranges[0]])

  // Assert
  assert.deepEqual(
    ordered.map((item) => item.id),
    ['DEN', 'KC', 'LV'],
  )
})

test('rankHistogram lists every rank with its probability and marks the published one', () => {
  // Act
  const bars = rankHistogram(range())

  // Assert
  assert.deepEqual(bars, [
    { rank: 1, probability: 0.5, published: true },
    { rank: 2, probability: 0.3, published: false },
    { rank: 3, probability: 0.2, published: false },
  ])
})

test('isMissingRankRanges is true only for a 404', () => {
  // Act
  const results = [new ApiError(404, 'missing'), new ApiError(500, 'broken'), new Error('x')].map(isMissingRankRanges)

  // Assert
  assert.deepEqual(results, [true, false, false])
})
