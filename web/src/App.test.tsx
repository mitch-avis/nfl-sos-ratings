import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DEN_GAME_LOGS,
  DEN_RATING_HISTORY,
  KC_RANK_HISTORY,
  KC_RATING_PAIRS,
  columnMeta,
  QB_RANK_RANGES,
  REGISTRY,
  SEASON_2025,
  QB_WP_RATINGS,
  stubApi,
  TEAM_RANK_RANGES,
  TEAM_WP_RATINGS,
} from '@/test/fixtures'
import { PALETTE_CSS_VARIABLES, paletteCssVariables } from '@/domain/teamPalettes'
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

  it('lays out compared teams side by side with their rank ranges', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES }))

    // Act
    renderApp('/teams?season=2025&compare=DEN,KC')

    // Assert
    const table = await screen.findByRole('table', { name: 'Team comparison' })
    const headers = within(table).getAllByRole('columnheader')
    expect(headers.map((header) => header.textContent?.slice(0, 3))).toEqual(['Met', 'DEN', 'KC'])
    await waitFor(() => expect(headers[1]).toHaveTextContent('1st–2nd'))
    expect(within(table).getByRole('rowheader', { name: /Team Rating/ })).toBeInTheDocument()
  })

  it('removes a team from the side-by-side comparison', async () => {
    // Arrange
    const user = userEvent.setup()
    const { router } = renderApp('/teams?season=2025&compare=DEN,KC')
    const remove = await screen.findByRole('button', { name: 'Remove KC from comparison' })

    // Act
    await user.click(remove)

    // Assert
    await waitFor(() => expect(router.state.location.search).toContain('compare=DEN'))
    expect(router.state.location.search).not.toContain('KC')
  })

  it('exports the table as shown to CSV', async () => {
    // Arrange
    const user = userEvent.setup()
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => 'blob:table')
    Object.assign(URL, { createObjectURL, revokeObjectURL: vi.fn() })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    renderApp('/teams?season=2025')
    const exportButton = await screen.findByRole('button', { name: 'Export the table as CSV' })

    // Act
    await user.click(exportButton)

    // Assert
    expect(click).toHaveBeenCalledOnce()
    const blob = createObjectURL.mock.calls[0]?.[0]
    const [header, first] = (await blob?.text())?.split('\r\n') ?? []
    expect(header?.startsWith('team,')).toBe(true)
    expect(first?.startsWith('DEN,')).toBe(true)
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
      in_progress: true,
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
      in_progress: true,
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
    expect(await screen.findByText(/14 pass attempts per game/)).toHaveTextContent(
      'at least 14 pass attempts per game his team has played',
    )
  })

  it('keeps the notice up until every team has finished its season', async () => {
    // Arrange
    const lastWeek = {
      ...SEASON_2025,
      in_progress: true,
      teams: {
        ...SEASON_2025.teams,
        rows: SEASON_2025.teams.rows.map((row, index) => ({ ...row, games_played: index === 0 ? 16 : 17 })),
      },
    }
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025': lastWeek }))

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByText(/Season in progress/)).toHaveTextContent('up to 17 games')
  })

  it('shows no notice for a completed season with a team short of a full season', async () => {
    // Arrange
    const missingGame = {
      ...SEASON_2025,
      teams: {
        ...SEASON_2025.teams,
        rows: SEASON_2025.teams.rows.map((row, index) => ({ ...row, games_played: index === 0 ? 16 : 17 })),
      },
    }
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025': missingGame }))

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    expect(screen.queryByText(/Season in progress/)).not.toBeInTheDocument()
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
  it('lists only qualifying QBs by default', async () => {
    // Act
    renderApp('/qbs?season=2025')

    // Assert
    expect(await screen.findByRole('link', { name: 'Bo Nix' })).toBeInTheDocument()
    expect(screen.queryByText('Short Sample')).not.toBeInTheDocument()
    expect(screen.queryByText('Backup Arm')).not.toBeInTheDocument()
  })

  it('adds the QBs below the qualifier once the switch is on', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/qbs?season=2025')
    await screen.findByRole('link', { name: 'Bo Nix' })

    // Act
    await user.click(screen.getByRole('switch', { name: 'Show QBs below the qualifier' }))

    // Assert
    expect(screen.getByText('Short Sample')).toBeInTheDocument()
    expect(screen.getByText('Backup Arm')).toBeInTheDocument()
  })

  it('says a QB below the qualifier has no rank range, and why', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/qbs/rating-ranges': QB_RANK_RANGES }))
    const user = userEvent.setup()
    renderApp('/qbs?season=2025')
    await screen.findByRole('columnheader', { name: /Rank range/ })

    // Act
    await user.click(screen.getByRole('switch', { name: 'Show QBs below the qualifier' }))

    // Assert
    expect(screen.getByRole('button', { name: 'Below the qualifier: 25 of 238 pass attempts' })).toHaveTextContent(
      'Below qualifier',
    )
  })

  it('leaves the raw player ID out of the table', async () => {
    // Act
    renderApp('/qbs?season=2025')

    // Assert
    expect(await screen.findByRole('link', { name: 'Bo Nix' })).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /QB ID/ })).not.toBeInTheDocument()
  })
})

describe('qb detail', () => {
  it('says why a QB below the qualifier has no rank range', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/qbs/rating-ranges': QB_RANK_RANGES }))

    // Act
    renderApp('/qbs/qb-3?season=2025')

    // Assert
    expect(await screen.findByText(/Not ranked: below the qualifier/)).toHaveTextContent(
      'Not ranked: below the qualifier (25 of 238 pass attempts), so no rank range.',
    )
  })
})

describe('rank column', () => {
  it('explains that Rank follows the current sort', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')

    // Act
    await user.hover(await screen.findByRole('button', { name: 'About the rank' }))

    // Assert
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      "Each row's position in the current sort (Team Rating).",
    )
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

  it('says which team was not found on returning to the index', async () => {
    // Act
    renderApp('/teams/NOPE?season=2025')

    // Assert
    expect(await screen.findByText('No team NOPE in 2025.')).toHaveAttribute('role', 'status')
  })

  it('says when the requested season is not available', async () => {
    // Act
    renderApp('/teams?season=1990')

    // Assert
    expect(await screen.findByRole('heading', { name: /Team Ratings Index · 2025/ })).toBeInTheDocument()
    expect(screen.getByText('Season 1990 is not available; showing 2025.')).toHaveAttribute('role', 'status')
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

  it("shows each opponent's season-long rank range in the weekly log", async () => {
    // Arrange
    const kcGames = {
      ...DEN_GAME_LOGS,
      rows: DEN_GAME_LOGS.rows.slice(0, 2).map((row, index) => ({ ...row, opponent_team: index === 0 ? 'DEN' : 'LV' })),
    }
    vi.stubGlobal(
      'fetch',
      stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES, '/api/seasons/2025/teams/KC/game-logs': kcGames }),
    )

    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    const table = await screen.findByRole('table', { name: 'Game by game' })
    const den = await within(table).findByRole('cell', { name: /^DEN/ })
    expect(den).toHaveTextContent('DEN1st–2nd')
  })

  it('breaks the detail-page rank range down by unit', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, [RANGES_PATH]: TEAM_RANK_RANGES }))

    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    const table = await screen.findByRole('table', { name: 'Rank range by unit' })
    const offense = within(table).getByRole('row', { name: /Offense/ })
    expect(offense).toHaveTextContent('1st; middle 50%: 1st–2nd; 95%: 1st–3rd')
    expect(within(table).getAllByRole('row')).toHaveLength(4)
  })
})

describe('head-to-head comparison', () => {
  it('compares a team with the one ranked just above it', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/teams/KC/rating-pairs': KC_RATING_PAIRS }))

    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    const card = await screen.findByRole('region', { name: 'Head to head' })
    expect(
      await within(card).findByText(
        'KC rated above DEN in 21% of resampled seasons; difference -4.9 points, 95%: -9.8 to +0.3.',
      ),
    ).toBeInTheDocument()
    expect(within(card).getByRole('combobox', { name: 'Compare with' })).toHaveTextContent('DEN')
  })

  it('states the head-to-head chance when exactly two teams are compared', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/teams/KC/rating-pairs': KC_RATING_PAIRS }))

    // Act
    renderApp('/teams?season=2025&compare=KC,DEN')

    // Assert
    expect(
      await screen.findByText(
        'KC rated above DEN in 21% of resampled seasons; difference -4.9 points, 95%: -9.8 to +0.3.',
      ),
    ).toBeInTheDocument()
  })

  it('leaves the card out for a season built without head-to-head chances', async () => {
    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Kansas City/ })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Head to head' })).not.toBeInTheDocument())
  })
})

describe('rank by week', () => {
  it('charts the rank range week by week for a season in progress', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/teams/KC/rank-history': KC_RANK_HISTORY }))

    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    const card = await screen.findByRole('region', { name: 'Rank by week' })
    expect(within(card).getByText('Median rank 2nd in week 1 (95%: 1st–3rd) and 1st in week 2 (95%: 1st–3rd).')).toBeInTheDocument()
    expect(within(card).getByRole('table', { name: 'Rank by week' })).toBeInTheDocument()
  })

  it('leaves the chart out for a season without weekly rank ranges', async () => {
    // Act
    renderApp('/teams/KC?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: /Kansas City/ })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Rank by week' })).not.toBeInTheDocument())
  })
})

describe('team palettes', () => {
  afterEach(() => {
    window.localStorage.removeItem('nfl-sos-palette')
    window.localStorage.removeItem('nfl-sos-team-page-colors')
    document.documentElement.removeAttribute('style')
  })

  it("shows a team page in that team's colors", async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')

    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    await screen.findByRole('heading', { name: 'Denver Broncos' })
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('DEN'))
    expect(document.documentElement.style.getPropertyValue('--primary')).toBe(
      paletteCssVariables('DEN', 'light')['--primary'],
    )
  })

  it("shows a QB page in his team's colors", async () => {
    // Act
    renderApp('/qbs/qb-1?season=2025')

    // Assert
    await screen.findByRole('heading', { name: /Bo Nix/ })
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('DEN'))
  })

  it('returns to the chosen palette on leaving a team page', async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')
    const { router } = renderApp('/teams/DEN?season=2025')
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('DEN'))

    // Act
    await router.navigate('/teams?season=2025')

    // Assert
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('KC'))
  })

  it('keeps the chosen palette on team pages when team colors are switched off', async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')
    window.localStorage.setItem('nfl-sos-team-page-colors', 'off')

    // Act
    renderApp('/teams/DEN?season=2025')

    // Assert
    await screen.findByRole('heading', { name: 'Denver Broncos' })
    expect(document.documentElement.dataset.palette).toBe('KC')
  })

  it('switches team-page colors off from the palette menu', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    await user.click(await screen.findByRole('button', { name: 'Palette: Default' }))

    // Act
    await user.click(await screen.findByRole('menuitemcheckbox', { name: "Use each team's colors on its page" }))

    // Assert
    expect(window.localStorage.getItem('nfl-sos-team-page-colors')).toBe('off')
  })

  it('opens the palette menu at the chosen team', async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'SEA')
    const user = userEvent.setup()
    renderApp('/teams?season=2025')

    // Act
    await user.click(await screen.findByRole('button', { name: 'Palette: Seattle Seahawks' }))

    // Assert
    const chosen = await screen.findByRole('menuitemradio', { name: 'Seattle Seahawks' })
    await waitFor(() => expect(chosen).toHaveFocus())
  })

  it('switches to a team palette from the palette menu', async () => {
    // Arrange
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    await user.click(await screen.findByRole('button', { name: 'Palette: Default' }))

    // Act
    await user.click(await screen.findByRole('menuitemradio', { name: 'Kansas City Chiefs' }))

    // Assert
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('KC'))
    expect(document.documentElement.style.getPropertyValue('--primary')).toMatch(/^oklch\(/)
    expect(window.localStorage.getItem('nfl-sos-palette')).toBe('KC')
  })

  it('reads the old stored Broncos choice as the Denver palette', async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'broncos')

    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('button', { name: 'Palette: Denver Broncos' })).toBeInTheDocument()
    expect(document.documentElement.style.getPropertyValue('--primary')).toBe(paletteCssVariables('DEN', 'light')['--primary'])
  })

  it("marks the top of the header with a team palette's colors", async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')

    // Act
    const { container } = renderApp('/teams?season=2025')

    // Assert
    await screen.findByRole('button', { name: 'Palette: Kansas City Chiefs' })
    expect(container.querySelector('header [data-palette-stripe]')).not.toBeNull()
  })

  it('leaves the header unmarked with the default palette', async () => {
    // Act
    const { container } = renderApp('/teams?season=2025')

    // Assert
    await screen.findByRole('button', { name: 'Palette: Default' })
    expect(container.querySelector('[data-palette-stripe]')).toBeNull()
  })

  it("clears every team palette color on switching back to the default", async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')
    const user = userEvent.setup()
    renderApp('/teams?season=2025')
    await user.click(await screen.findByRole('button', { name: 'Palette: Kansas City Chiefs' }))

    // Act
    await user.click(await screen.findByRole('menuitemradio', { name: 'Default' }))

    // Assert
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('classic'))
    expect(PALETTE_CSS_VARIABLES.filter((name) => document.documentElement.style.getPropertyValue(name) !== '')).toEqual([])
  })

  it('returns a team page to the chosen palette when team colors are switched off there', async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')
    const user = userEvent.setup()
    renderApp('/teams/DEN?season=2025')
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('DEN'))
    await user.click(await screen.findByRole('button', { name: 'Palette: Kansas City Chiefs' }))

    // Act
    await user.click(await screen.findByRole('menuitemcheckbox', { name: "Use each team's colors on its page" }))

    // Assert
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('KC'))
  })

  it("keeps a team page in its team's colors while another season loads", async () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'KC')
    const loaded = stubApi({ ...API, '/api/seasons': { seasons: [2025, 2024] } })
    vi.stubGlobal('fetch', ((input: RequestInfo | URL) =>
      String(input).endsWith('/api/seasons/2024') ? new Promise<Response>(() => {}) : loaded(input)) as typeof fetch)
    const { router } = renderApp('/teams/DEN?season=2025')
    await waitFor(() => expect(document.documentElement.dataset.palette).toBe('DEN'))

    // Act
    await router.navigate('/teams/DEN?season=2024')

    // Assert
    await screen.findByText(/Loading season data/)
    expect(document.documentElement.dataset.palette).toBe('DEN')
  })
})

describe('team colors', () => {
  it("shows each team's colors beside its name in the index", async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    const link = await screen.findByRole('link', { name: 'DEN' })
    expect(link.querySelector('[data-team-chip="DEN"]')).not.toBeNull()
  })

  it("shows each quarterback's team colors in the QB index", async () => {
    // Act
    const { container } = renderApp('/qbs?season=2025')

    // Assert
    await screen.findByRole('link', { name: 'Bo Nix' })
    expect(container.querySelector('tbody [data-team-chip="DEN"]')).not.toBeNull()
  })

  it("shows the team's colors beside the detail page title and each opponent", async () => {
    // Act
    const { container } = renderApp('/teams/DEN?season=2025')

    // Assert
    const heading = await screen.findByRole('heading', { name: 'Denver Broncos' })
    expect(heading.querySelector('[data-team-chip="DEN"]')).not.toBeNull()
    await screen.findByRole('link', { name: '2025_02_DEN_IND' })
    expect(container.querySelector('tbody [data-team-chip="IND"]')).not.toBeNull()
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

  it('explains each metric from the registry when opened directly', async () => {
    // Arrange
    const registry = {
      ...REGISTRY,
      metrics: {
        qb_sack_rate: {
          ...columnMeta('Sack Rate', { full_name: 'Sack Rate', shape: 'rate', category: 'Pressure, Sacks & Pocket' }),
          description: 'Sacks taken per dropback.',
          entity: 'qb',
        },
      },
    }
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/metadata': registry }))

    // Act
    renderApp('/glossary')

    // Assert
    const term = await screen.findByText('Sack Rate')
    expect(term.closest('div')).toHaveTextContent('Sack RateSacks taken per dropback.')
  })
})

describe('garbage-time filter', () => {
  it('starts off, with every play counted and no filtered table', async () => {
    // Act
    renderApp('/teams?season=2025')

    // Assert
    expect(await screen.findByRole('slider', { name: 'Garbage-time filter' })).toHaveAttribute('aria-valuenow', '0')
    expect(screen.getByText(/Off: every play counts/)).toBeInTheDocument()
    expect(screen.queryByText('Unvalidated exploration view')).not.toBeInTheDocument()
  })

  it('lists teams by filtered rank beside their published rank and rating', async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/teams/wp-ratings?threshold=10': TEAM_WP_RATINGS }))

    // Act
    renderApp('/teams?season=2025&wp=10')

    // Assert
    const table = await screen.findByRole('table', { name: /filtered at 10%/ })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((row) => within(row).getByRole('link').textContent)).toEqual(['KC', 'DEN', 'LV'])
    expect(within(rows[0]).getByText('up 1')).toBeInTheDocument()
    expect(within(rows[0]).getByRole('link')).toHaveAttribute('href', '/teams/KC?season=2025&wp=10')
    expect(within(table).getByRole('button', { name: /Published/ })).toBeInTheDocument()
    expect(within(rows[0]).getByText('2 · 5.40')).toBeInTheDocument()
    expect(screen.getByText('Unvalidated exploration view')).toBeInTheDocument()
    expect(screen.getByText(/Rank ranges and the rest of this page count every play/)).toBeInTheDocument()
    expect(screen.getByText(/no threshold predicted team game margins better than every play/)).toBeInTheDocument()
  })

  it("lists each filtered quarterback's team in its own column", async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/qbs/wp-ratings?threshold=10': QB_WP_RATINGS }))

    // Act
    renderApp('/qbs?season=2025&wp=10')

    // Assert
    const table = await screen.findByRole('table', { name: /filtered at 10%/ })
    const [header, row] = within(table).getAllByRole('row')
    expect(within(header).getByRole('columnheader', { name: 'Team' })).toBeInTheDocument()
    expect(within(row).getByRole('cell', { name: 'DEN' })).toBeInTheDocument()
  })

  it('puts the chosen threshold in the address after the slider settles', async () => {
    // Arrange
    const user = userEvent.setup()
    const { router } = renderApp('/teams?season=2025')
    const slider = await screen.findByRole('slider', { name: 'Garbage-time filter' })

    // Act
    slider.focus()
    await user.keyboard('{ArrowRight}{ArrowRight}')

    // Assert
    await waitFor(() => expect(router.state.location.search).toContain('wp=2'))
    expect(screen.getByText(/between 2% and 98%/)).toBeInTheDocument()
  })

  it("shows a QB's filtered rank and rating on his detail page", async () => {
    // Arrange
    vi.stubGlobal('fetch', stubApi({ ...API, '/api/seasons/2025/qbs/wp-ratings?threshold=10': QB_WP_RATINGS }))

    // Act
    renderApp('/qbs/qb-1?season=2025&wp=10')

    // Assert
    const panel = await screen.findByRole('region', { name: 'Garbage-time filter' })
    expect(await within(panel).findByText('0.15')).toBeInTheDocument()
    expect(within(panel).getByText('Unvalidated exploration view')).toBeInTheDocument()
  })

  it('opens a detail page from the filtered table without losing the threshold', async () => {
    // Arrange
    vi.stubGlobal(
      'fetch',
      stubApi({
        ...API,
        '/api/seasons/2025/teams/wp-ratings?threshold=10': TEAM_WP_RATINGS,
      }),
    )
    const user = userEvent.setup()
    const { router } = renderApp('/teams?season=2025&wp=10')
    const table = await screen.findByRole('table', { name: /filtered at 10%/ })

    // Act
    await user.click(within(table).getByRole('link', { name: 'DEN' }))

    // Assert
    await waitFor(() => expect(router.state.location.pathname).toBe('/teams/DEN'))
    expect(router.state.location.search).toContain('wp=10')
    expect(await screen.findByRole('region', { name: 'Garbage-time filter' })).toBeInTheDocument()
  })
})
