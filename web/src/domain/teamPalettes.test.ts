import { describe, expect, it } from 'vitest'

import {
  heatPaletteFor,
  normalizePalette,
  paletteCssVariables,
  paletteGroups,
  paletteName,
} from './teamPalettes'

describe('paletteGroups', () => {
  it('lists every team under its division, in division order', () => {
    // Act
    const groups = paletteGroups()

    // Assert
    expect(groups.map((group) => group.division)[0]).toBe('AFC East')
    expect(groups.flatMap((group) => group.teams)).toHaveLength(32)
    expect(groups.find((group) => group.division === 'AFC West')?.teams.map((team) => team.id)).toEqual([
      'DEN',
      'KC',
      'LV',
      'LAC',
    ])
  })
})

describe('normalizePalette', () => {
  it.each([
    ['broncos', 'DEN'],
    ['KC', 'KC'],
    ['classic', 'classic'],
    ['nonsense', 'classic'],
    [null, 'classic'],
  ])('reads a stored %s as %s', (stored, expected) => {
    // Act
    const palette = normalizePalette(stored)

    // Assert
    expect(palette).toBe(expected)
  })
})

describe('paletteCssVariables', () => {
  it('keeps the hand-tuned Broncos accent', () => {
    // Act
    const variables = paletteCssVariables('DEN', 'light')

    // Assert
    expect(variables['--primary']).toBe('oklch(0.66 0.2 40)')
    expect(variables['--chart-2']).toBe('oklch(0.35 0.09 255)')
    expect('--sidebar-primary-foreground' in variables).toBe(false)
  })

  it('sets the sidebar text color for generated palettes', () => {
    // Act
    const variables = paletteCssVariables('KC', 'dark')

    // Assert
    expect(variables['--sidebar-primary-foreground']).toBe(variables['--primary-foreground'])
  })

  it('sets nothing for the default palette', () => {
    // Act
    const variables = paletteCssVariables('classic', 'light')

    // Assert
    expect(variables).toEqual({})
  })
})

describe('heatPaletteFor', () => {
  it("returns a team's heat scale, or null when it falls back to the default", () => {
    // Act
    const scales = [heatPaletteFor('DEN', 'dark'), heatPaletteFor('LV', 'light'), heatPaletteFor('classic', 'light')]

    // Assert
    expect(scales[0]).toEqual({ good: [124, 51, 24], bad: [15, 48, 84], mid: [22, 27, 34] })
    expect(scales.slice(1)).toEqual([null, null])
  })
})

describe('paletteName', () => {
  it('names the default palette and team palettes', () => {
    // Act
    const names = [paletteName('classic'), paletteName('DEN')]

    // Assert
    expect(names).toEqual(['Default', 'Denver Broncos'])
  })
})
