import { describe, expect, it } from 'vitest'

import {
  activePalette,
  brandMark,
  heatPaletteFor,
  normalizePalette,
  paletteCssVariables,
  paletteGroups,
  paletteName,
  teamColors,
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
  it('builds the Broncos palette from nflverse colors like every other team', () => {
    // Act
    const variables = paletteCssVariables('DEN', 'light')

    // Assert
    expect(variables['--primary']).toBe('oklch(0.554 0.188 36.5)')
    expect(variables['--sidebar-primary-foreground']).toBe(variables['--primary-foreground'])
  })

  it('colors the hint cards and hover backgrounds but leaves the page surfaces alone', () => {
    // Act
    const variables = paletteCssVariables('GB', 'dark')

    // Assert
    expect(Object.keys(variables)).toEqual(
      expect.arrayContaining(['--accent', '--accent-foreground', '--sidebar-accent', '--hint', '--hint-border']),
    )
    expect(Object.keys(variables)).not.toEqual(expect.arrayContaining(['--background']))
    expect(Object.keys(variables)).not.toEqual(expect.arrayContaining(['--card']))
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
  it('gives every team its own heat scale in both modes', () => {
    // Arrange
    const teams = paletteGroups().flatMap((group) => group.teams.map((team) => team.id))

    // Act
    const missing = teams.flatMap((team) =>
      (['light', 'dark'] as const).filter((mode) => heatPaletteFor(team, mode) === null).map((mode) => `${team} ${mode}`),
    )

    // Assert
    expect(missing).toEqual([])
  })

  it('leaves the default palette on the default heat scale', () => {
    // Act
    const scale = heatPaletteFor('classic', 'light')

    // Assert
    expect(scale).toBeNull()
  })
})

describe('teamColors', () => {
  it("returns a team's two main colors", () => {
    // Act
    const colors = teamColors('KC')

    // Assert
    expect(colors).toEqual(['#E31837', '#FFB612'])
  })

  it('returns null for a team without colors', () => {
    // Act
    const colors = teamColors('XYZ')

    // Assert
    expect(colors).toBeNull()
  })
})

describe('brandMark', () => {
  it('draws the default logo for the default palette', () => {
    // Act
    const mark = brandMark('classic')

    // Assert
    expect(mark).toEqual({ background: '#1f3a8a', line: '#fb923c' })
  })

  it("draws a team palette's logo in the team's colors", () => {
    // Act
    const mark = brandMark('DEN')

    // Assert
    expect(mark).toEqual({ background: '#002244', line: '#FB4F14' })
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

describe('activePalette', () => {
  it.each([
    ['KC', 'DEN', true, 'DEN'],
    ['classic', 'DEN', true, 'DEN'],
    ['KC', 'DEN', false, 'KC'],
    ['KC', null, true, 'KC'],
    ['KC', 'XYZ', true, 'KC'],
  ] as const)('with %s chosen, a page for %s, and team colors %s, shows %s', (chosen, pageTeam, teamColors, expected) => {
    // Act
    const palette = activePalette(chosen, pageTeam, teamColors)

    // Assert
    expect(palette).toBe(expected)
  })
})
