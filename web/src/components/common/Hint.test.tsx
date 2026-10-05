import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { stubTouchScreen } from '@/test/touch'

import { Hint } from './Hint'

function renderHint() {
  return render(
    <TooltipProvider delayDuration={0}>
      <Hint content="Points per game better than an average team.">
        <button type="button">Team Rating</button>
      </Hint>
      <p>Elsewhere</p>
    </TooltipProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Hint', () => {
  it('shows the hint card on hover with a mouse', async () => {
    // Arrange
    const user = userEvent.setup()
    renderHint()

    // Act
    await user.hover(screen.getByRole('button', { name: 'Team Rating' }))

    // Assert
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Points per game better than an average team.')
  })

  it('opens the hint card on tap on a touch screen', async () => {
    // Arrange
    stubTouchScreen()
    const user = userEvent.setup()
    renderHint()

    // Act
    await user.click(screen.getByRole('button', { name: 'Team Rating' }))

    // Assert
    expect(await screen.findByRole('dialog')).toHaveTextContent('Points per game better than an average team.')
  })

  it('closes a tapped hint card when the reader taps elsewhere', async () => {
    // Arrange
    stubTouchScreen()
    const user = userEvent.setup()
    renderHint()
    await user.click(screen.getByRole('button', { name: 'Team Rating' }))
    await screen.findByRole('dialog')

    // Act
    await user.click(screen.getByText('Elsewhere'))

    // Assert
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })
})
