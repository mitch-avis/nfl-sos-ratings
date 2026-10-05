import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { parseRankRanges } from '@/domain/rankRanges'
import { TEAM_RANK_RANGES } from '@/test/fixtures'
import { stubTouchScreen } from '@/test/touch'

import { RankRangeChart } from './RankRangeChart'

function renderChart() {
  const ranges = parseRankRanges('teams', TEAM_RANK_RANGES)
  const router = createMemoryRouter([
    { path: '/', element: <RankRangeChart kind="teams" season={2025} ranges={ranges} /> },
  ])
  return render(
    <TooltipProvider delayDuration={0}>
      <RouterProvider router={router} />
    </TooltipProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('RankRangeChart', () => {
  it('reads out the hovered row above the chart', async () => {
    // Arrange
    const user = userEvent.setup()
    renderChart()

    // Act
    await user.hover(screen.getByRole('link', { name: /^KC:/ }))

    // Assert
    expect(screen.getByRole('status')).toHaveTextContent('KC: 2nd; middle 50%: 2nd; 95%: 1st–3rd')
  })

  it('selects a tapped row on touch screens and links to its page from the readout', async () => {
    // Arrange
    stubTouchScreen()
    const user = userEvent.setup()
    renderChart()

    // Act
    await user.click(screen.getByRole('button', { name: /^LV:/ }))

    // Assert
    const readout = screen.getByRole('status')
    expect(readout).toHaveTextContent('LV: 3rd')
    expect(within(readout).getByRole('link', { name: 'Open LV' })).toHaveAttribute('href', '/teams/LV?season=2025')
  })
})
