import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react'
import type { MouseEvent, ReactNode } from 'react'

import { useHasHover } from '@/hooks/use-has-hover'

import { Hint } from './Hint'
import { InfoTooltip } from './InfoTooltip'

type SortDirection = false | 'asc' | 'desc'

/** The arrow beside a sortable column label. */
export function SortIcon({ direction }: { direction: SortDirection }) {
  if (direction === 'asc') return <ArrowUp className="size-3.5" aria-label="sorted ascending" />
  if (direction === 'desc') return <ArrowDown className="size-3.5" aria-label="sorted descending" />
  return <ArrowUpDown className="size-3.5 opacity-40" aria-hidden />
}

/**
 * A sortable column header that also explains the column. With a mouse the header button itself
 * shows the hint on hover or focus; on touch screens a tap sorts, so the hint sits behind a
 * separate info button beside it.
 */
export function SortableHeader({
  label,
  hint,
  direction,
  onSort,
}: {
  label: string
  hint: ReactNode
  direction: SortDirection
  onSort: (event: MouseEvent<HTMLButtonElement>) => void
}) {
  const hasHover = useHasHover()
  const sortButton = (
    <button type="button" className="inline-flex items-center gap-1 hover:text-foreground" onClick={onSort}>
      <span className={hasHover ? 'underline decoration-muted-foreground/40 decoration-dotted underline-offset-4' : undefined}>
        {label}
      </span>
      <SortIcon direction={direction} />
    </button>
  )
  if (hasHover) return <Hint content={hint}>{sortButton}</Hint>
  return (
    <span className="inline-flex items-center gap-1.5">
      {sortButton}
      <InfoTooltip content={hint} label={`About ${label}`} />
    </span>
  )
}
