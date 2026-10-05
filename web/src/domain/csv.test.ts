import { describe, expect, it } from 'vitest'

import { csvFileName, toCsv } from './csv'

describe('toCsv', () => {
  it('writes a header of column keys and one line per row, in order', () => {
    // Arrange
    const rows = [
      { team: 'DEN', team_rating: 8.4123456, sos: 0.2 },
      { team: 'KC', team_rating: 3.5, sos: -0.1 },
    ]

    // Act
    const csv = toCsv(['team', 'team_rating'], rows)

    // Assert
    expect(csv).toBe('team,team_rating\r\nDEN,8.4123456\r\nKC,3.5\r\n')
  })

  it('quotes text with commas, quotes, or line breaks', () => {
    // Arrange
    const rows = [{ qb_name: 'Smith, Jr.', note: 'the "wildcat"\nformation' }]

    // Act
    const csv = toCsv(['qb_name', 'note'], rows)

    // Assert
    expect(csv).toBe('qb_name,note\r\n"Smith, Jr.","the ""wildcat""\nformation"\r\n')
  })

  it('leaves missing values empty and spells out booleans', () => {
    // Arrange
    const rows = [{ team: 'LV', qb_is_eligible: true, sos: null }]

    // Act
    const csv = toCsv(['team', 'qb_is_eligible', 'sos', 'absent'], rows)

    // Assert
    expect(csv).toBe('team,qb_is_eligible,sos,absent\r\nLV,true,,\r\n')
  })
})

describe('csvFileName', () => {
  it('names the file after the kind and season', () => {
    // Act
    const name = csvFileName('qbs', 2025)

    // Assert
    expect(name).toBe('nfl-sos-ratings-qbs-2025.csv')
  })
})
