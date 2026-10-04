import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DEN_GAME_LOGS, DEN_RATING_HISTORY, REGISTRY, SEASON_2025, stubApi, TEAM_RANK_RANGES } from '@/test/fixtures'
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
  it('lists teams best Team Rating first with ranks and detail links', async () => {
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

describe('season in progress', () => {
  it('flags a partial season above the team index', async () => {
    // Arrange
    const partial = {
      ...SEASON_2025,
      season: 2026,
      teams: {
        ...SEASON_2025.teams,
        rows: SEASON_2025.teams.rows.map((row) => ({ ...row, games_played: 3 })),
      },
    }
    vi.stubGlobal(
      'fetch',
      stubApi({ '/api/seasons': { seasons: [2026] }, '/api/metadata': REGISTRY, '/api/seasons/2026': partial }),
    )

    // Act
    renderApp('/teams?season=2026')

    // Assert
    expect(await screen.findByText(/Season in progress/)).toHaveTextContent('3 games')
  })

  it('states the QB qualifier for the games played so far', async () => {
    // Arrange
    const partial = {
      ...SEASON_2025,
      season: 2026,
      teams: {
        ...SEASON_2025.teams,
        rows: SEASON_2025.teams.rows.map((row) => ({ ...row, games_played: 3 })),
      },
    }
    vi.stubGlobal(
      'fetch',
      stubApi({ '/api/seasons': { seasons: [2026] }, '/api/metadata': REGISTRY, '/api/seasons/2026': partial }),
    )

    // Act
    renderApp('/qbs?season=2026')

    // Assert
    expect(await screen.findByText(/rating threshold of/)).toHaveTextContent(
      '42 pass attempts (14 per team game over the 3 games played so far)',
    )
  })

  it('shows no notice for a completed season', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    expect(screen.queryByText(/Season in progress/)).not.toBeInTheDocument()
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

  it('leaves the raw player ID out of the table', async () => {
    // Act
    renderApp('/qbs?season=2025')

    // Assert
    expect(await screen.findByRole('link', { name: 'Bo Nix' })).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /QB ID/ })).not.toBeInTheDocument()
  })
})

describe('index header', () => {
  it('keeps the reading notes folded away until asked for', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByText('Reading notes')).toBeVisible()
    expect(screen.getByText(/SRS is the classic point-margin rating/)).not.toBeVisible()
  })

  it('leaves out the summary tiles that repeat the table header', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    expect(screen.queryByText('Rows shown')).not.toBeInTheDocument()
  })
})

describe('phone layout', () => {
  afterEach(() => {
    window.innerWidth = 1024
  })

  it('pins only the name column so the stats have room', async () => {
    // Arrange
    window.innerWidth = 402

    // Act
    renderApp('/qbs?season=2025')

    // Assert
    await screen.findByRole('link', { name: 'Bo Nix' })
    const pinned = screen.getAllByRole('columnheader').filter((header) => header.classList.contains('sticky'))
    expect(pinned.map((header) => header.textContent)).toEqual(['QB'])
  })
})

describe('team detail', () => {
  it('shows the full team name, ratings, and the weekly log', async () => {
    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: 'Denver Broncos' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Season Ratings' })).toHaveTextContent('Team Rating8.40')
    expect(await screen.findByRole('link', { name: '2025_02_DEN_IND' })).toHaveAttribute('target', '_blank')
  })

  it('charts the rating by week when the season has a rating history', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/teams/DEN/rating-history': DEN_RATING_HISTORY }))

    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    expect(await screen.findByText('Rating by week')).toBeInTheDocument()
    expect(screen.getByText('Team Rating by week')).toBeInTheDocument()
    expect(screen.getByText('Dashed line: an average team (0).')).toBeInTheDocument()
  })

  it('leaves the rating chart out for a season without a rating history', async () => {
    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    expect(await screen.findByRole('link', { name: '2025_02_DEN_IND' })).toBeInTheDocument()
    expect(screen.queryByText('Rating by week')).not.toBeInTheDocument()
    expect(screen.queryByText('Could not load the rating history')).not.toBeInTheDocument()
  })

  it('reports a rating-history failure other than a missing file', async () => {
    // Arrange
    const api = stubApi(API)
    vi.stubGlobal('fetch', (async (input: RequestInfo | URL) =>
      String(input).endsWith('/rating-history')
        ? new Response(JSON.stringify({ detail: 'disk read failed' }), {
            status: 500,
            headers: { 'content-type': 'application/json' },
          })
        : api(input)) as typeof fetch)

    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    expect(await screen.findByText('Could not load the rating history')).toBeInTheDocument()
    expect(screen.getByText('disk read failed')).toBeInTheDocument()
  })

  it('returns to the index for an unknown team', async () => {
    // Act
    const { router } = renderApp('/teams/NOPE?season=2025')

    // Assert
    await waitFor(() => expect(router.state.location.pathname).toBe('/teams'))
  })
})

const RANGES_PATH = '/api/seasons/2025/teams/rating-ranges'

describe('rank ranges', () => {
  it('charts every team on the index, each row linking to its detail page', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES }))

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByText('Rank ranges')).toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'DEN: published rank 1st, median 1st; middle 50%: 1st–2nd; 95%: 1st–3rd' }),
    ).toHaveAttribute('href', '/teams/DEN?season=2025')
  })

  it('adds a rank range column beside the team rating', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES }))

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('columnheader', { name: /Rank range/ })).toBeInTheDocument()
    expect(within(bodyRows()[2]).getByText('3rd')).toBeInTheDocument()
  })

  it('leaves the rank ranges out for a season without range files', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    expect(screen.queryByText('Rank ranges')).not.toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /Rank range/ })).not.toBeInTheDocument()
    expect(screen.queryByText('Could not load the rank ranges')).not.toBeInTheDocument()
  })

  it('reports a rank-range failure other than a missing file', async () => {
    // Arrange
    const api = stubApi(API)
    vi.stubGlobal('fetch', (async (input: RequestInfo | URL) =>
      String(input).endsWith('/rating-ranges')
        ? new Response(JSON.stringify({ detail: 'disk read failed' }), {
            status: 500,
            headers: { 'content-type': 'application/json' },
          })
        : api(input)) as typeof fetch)

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByText('Could not load the rank ranges')).toBeInTheDocument()
  })

  it('headlines the rank range and its chances on the detail page', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES }))

    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    expect(await screen.findByText('2nd; middle 50%: 2nd; 95%: 1st–3rd')).toBeInTheDocument()
    expect(screen.getByText('Top 5 in 100% of resamples, top 10 in 100%')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Chance of each rank' })).toHaveTextContent('2nd52%')
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
    expect(screen.getByText('Primary overall team rank: Team Rating')).toBeInTheDocument()
  })
})
