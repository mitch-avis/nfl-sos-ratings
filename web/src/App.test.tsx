import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DEN_GAME_LOGS, REGISTRY, SEASON_2025, stubApi } from '@/test/fixtures'
import { renderApp } from '@/test/renderApp'

const API = {
  '/api/seasons': { seasons: [2025, 2024] },
  '/api/metadata': REGISTRY,
  '/api/seasons/2025': SEASON_2025,
  '/api/seasons/2025/teams/DEN/game-logs': DEN_GAME_LOGS,
}

beforeEach(() => {
  vi.stubGlobal('fetch', stubApi(API))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function bodyRows(): HTMLElement[] {
  const tables = screen.getAllByRole('table')
  const indexTable = tables[tables.length - 1]
  return within(indexTable).getAllByRole('row').slice(1)
}

describe('team index', () => {
  it('lists teams best SaCR first with ranks and detail links', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    const rows = bodyRows()
    expect(rows.map((row) => within(row).getAllByRole('cell')[2].textContent)).toEqual(['DEN', 'KC', 'LV'])
    expect(within(rows[0]).getAllByRole('cell')[1]).toHaveTextContent('1')
    expect(within(rows[0]).getByRole('link', { name: 'DEN' })).toHaveAttribute('href', '/teams/DEN?season=2025')
  })

  it('filters rows by the search box', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    const search = await screen.findByRole('searchbox', { name: 'Search visible columns' })

    // Act
    await user.type(search, 'kc')

    // Assert
    expect(bodyRows()).toHaveLength(1)
    expect(bodyRows()[0]).toHaveTextContent('KC')
  })

  it('adds compared teams to the panel and the URL', async () => {
    // Arrange
    const user = userEvent.setup()
    const { router } = renderApp('/teams?season=2025')
    const compareKc = await screen.findByRole('checkbox', { name: 'Compare KC' })

    // Act
    await user.click(compareKc)

    // Assert
    expect(await screen.findByText('Team comparison')).toBeInTheDocument()
    await waitFor(() => expect(router.state.location.search).toContain('compare=KC'))
  })

  it('restores a shared comparison from the URL', async () => {
    // Act
    renderApp('/teams?season=2025&compare=LV,DEN')

    // Assert
    expect(await screen.findByText('Team comparison')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Compare LV' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Compare DEN' })).toBeChecked()
  })

  it('starts on the Ratings view with reset disabled', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('button', { name: 'Reset' })).toBeDisabled()
    expect(screen.queryByRole('group', { name: 'Category' })).not.toBeInTheDocument()
  })

  it('switching views shows the category filter and enables reset', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    const reset = await screen.findByRole('button', { name: 'Reset' })

    // Act
    await user.click(screen.getByRole('button', { name: 'Per-Game Rates' }))

    // Assert
    expect(screen.getByRole('button', { name: 'Per-Game Rates' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('group', { name: 'Category' })).toBeInTheDocument()
    expect(reset).toBeEnabled()
  })

  it('reset returns to the Ratings view', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    const reset = await screen.findByRole('button', { name: 'Reset' })
    await user.click(screen.getByRole('button', { name: 'Per-Game Rates' }))

    // Act
    await user.click(reset)

    // Assert
    expect(screen.getByRole('button', { name: 'Ratings' })).toHaveAttribute('aria-pressed', 'true')
  })
})

describe('QB index', () => {
  it('hides unrated QBs by default', async () => {
    // Act
    renderApp('/qbs?season=2025')

    // Assert
    expect(await screen.findByRole('link', { name: 'Bo Nix' })).toBeInTheDocument()
    expect(screen.queryByText('Backup Arm')).not.toBeInTheDocument()
  })

  it('shows unrated QBs once the switch is on', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/qbs?season=2025')
    await screen.findByRole('link', { name: 'Bo Nix' })

    // Act
    await user.click(screen.getByRole('switch', { name: 'Show unrated or empty QB rows' }))

    // Assert
    expect(screen.getByText('Backup Arm')).toBeInTheDocument()
  })
})

describe('team detail', () => {
  it('shows the full team name, ratings, and the weekly log', async () => {
    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: 'Denver Broncos' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Season Ratings' })).toHaveTextContent('SaCR1.40')
    expect(await screen.findByRole('link', { name: '2025_02_DEN_IND' })).toHaveAttribute('target', '_blank')
  })

  it('returns to the index for an unknown team', async () => {
    // Act
    const { router } = renderApp('/teams/NOPE?season=2025')

    // Assert
    await waitFor(() => expect(router.state.location.pathname).toBe('/teams'))
  })
})

describe('seasons and glossary', () => {
  it('defaults to the newest season when none is given', async () => {
    // Act
    renderApp('/teams')

    // Assert
    expect(await screen.findByRole('heading', { name: /· 2025/ })).toBeInTheDocument()
  })

  it('reports a backend error instead of loading forever', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ '/api/seasons': { seasons: [2025] }, '/api/metadata': REGISTRY }))

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByText('Could not load 2025 season')).toBeInTheDocument()
    expect(screen.getByText('No route for /api/seasons/2025')).toBeInTheDocument()
  })

  it('renders the glossary without season data', async () => {
    // Act
    renderApp('/glossary')

    // Assert
    expect(await screen.findByRole('heading', { name: 'Glossary' })).toBeInTheDocument()
    expect(screen.getByText('Primary overall team rank: SaCR')).toBeInTheDocument()
  })
})
