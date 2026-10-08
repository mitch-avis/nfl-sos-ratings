import { render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ThemeProvider } from '@/app/ThemeProvider'

import { BrandMark } from './BrandMark'

afterEach(() => {
  window.localStorage.clear()
})

function tileFill(container: HTMLElement): string | null {
  return container.querySelector('rect')?.getAttribute('fill') ?? null
}

describe('BrandMark', () => {
  it('draws the default logo with the default palette', () => {
    // Act
    const { container } = render(
      <ThemeProvider>
        <BrandMark />
      </ThemeProvider>,
    )

    // Assert
    expect(tileFill(container)).toBe('#1f3a8a')
  })

  it("draws the logo in the chosen team's colors", () => {
    // Arrange
    window.localStorage.setItem('nfl-sos-palette', 'GB')

    // Act
    const { container } = render(
      <ThemeProvider>
        <BrandMark />
      </ThemeProvider>,
    )

    // Assert
    expect(tileFill(container)).toBe('#203731')
    expect(container.querySelector('path')).toHaveAttribute('stroke', '#FFB612')
  })
})
