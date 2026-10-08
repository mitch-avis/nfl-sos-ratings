import { useTheme } from '@/app/ThemeProvider'
import { brandMark } from '@/domain/teamPalettes'
import { cn } from '@/utils/cn'

/** The app logo (the trend line of `public/favicon.svg`), drawn in the colors of the palette shown. */
export function BrandMark({ className }: { className?: string }) {
  const { activePalette } = useTheme()
  const { background, line } = brandMark(activePalette)
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn('size-8 shrink-0', className)}>
      <rect width="32" height="32" rx="7" fill={background} />
      <path
        d="M7 22 L13 14 L18 18 L25 9"
        fill="none"
        stroke={line}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="25" cy="9" r="2.4" fill="#ffffff" />
    </svg>
  )
}
