import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { stubTouchScreen } from '@/test/touch'

import { SortableHeader } from './SortableHeader'

function renderHeader(onSort: () => void) {
  return render(
    <TooltipProvider delayDuration={0}>
      <SortableHeader label="Team Rating" hint="Points per game above average." direction={false} onSort={onSort} />
    </TooltipProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('SortableHeader', () => {
  it('explains the column when a mouse hovers the header', async () => {
    // Arrange
    const user = userEvent.setup()
    renderHeader(vi.fn())

    // Act
    await user.hover(screen.getByRole('button', { name: /Team Rating/ }))

    // Assert
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Points per game above average.')
  })

  it('sorts when the header is clicked', async () => {
    // Arrange
    const user = userEvent.setup()
    const onSort = vi.fn()
    renderHeader(onSort)

    // Act
    await user.click(screen.getByRole('button', { name: /Team Rating/ }))

    // Assert
    expect(onSort).toHaveBeenCalledOnce()
  })

  it('keeps sorting on the header and puts the explanation behind an info button on touch', async () => {
    // Arrange
    stubTouchScreen()
    const user = userEvent.setup()
    const onSort = vi.fn()
    renderHeader(onSort)

    // Act
    await user.click(screen.getByRole('button', { name: 'About Team Rating' }))

    // Assert
    expect(await screen.findByRole('dialog')).toHaveTextContent('Points per game above average.')
    expect(onSort).not.toHaveBeenCalled()
  })
})
