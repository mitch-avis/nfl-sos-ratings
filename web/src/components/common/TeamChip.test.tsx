import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { TeamChip } from './TeamChip'

describe('TeamChip', () => {
  it("shows the team's two main colors, hidden from screen readers", () => {
    // Act
    const { container } = render(<TeamChip team="KC" />)

    // Assert
    const chip = container.querySelector('[data-team-chip]')
    expect(chip).toHaveAttribute('aria-hidden', 'true')
    expect(chip).toHaveAttribute('data-team-chip', 'KC')
    expect((chip as HTMLElement).style.background).toContain('linear-gradient')
  })

  it('renders nothing for a team without colors', () => {
    // Act
    const { container } = render(<TeamChip team="XYZ" />)

    // Assert
    expect(container.querySelector('[data-team-chip]')).toBeNull()
  })
})
