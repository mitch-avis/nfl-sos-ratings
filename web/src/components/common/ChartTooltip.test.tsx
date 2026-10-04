import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ChartTooltipCard } from './ChartTooltip'

describe('ChartTooltipCard', () => {
  it('shows the title and each value in text color beside its series swatch', () => {
    // Act
    render(<ChartTooltipCard title="Week 9" rows={[{ label: 'Team Rating', value: '2.547', color: 'var(--chart-1)' }]} />)

    // Assert
    expect(screen.getByText('Week 9')).toBeInTheDocument()
    expect(screen.getByText('2.547')).toHaveClass('text-popover-foreground')
    expect(screen.getByText('Team Rating').previousElementSibling).toHaveStyle({ background: 'var(--chart-1)' })
  })
})
