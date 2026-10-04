import type { ReactNode } from 'react'

import { cn } from '@/utils/cn'

import { HINT_CARD_CLASS } from './hintStyles'

export interface ChartTooltipRow {
  label: string
  value: string
  /** The series color, drawn as a swatch; values and labels stay in text colors. */
  color?: string
}

/**
 * The card a Recharts `<Tooltip content={...}>` renders: the same hint card as every other
 * tooltip, with a title and one row per series.
 */
export function ChartTooltipCard({ title, rows }: { title: ReactNode; rows: ChartTooltipRow[] }) {
  return (
    <div className={cn(HINT_CARD_CLASS, 'min-w-36')}>
      <div className="font-medium text-popover-foreground">{title}</div>
      {rows.map((row) => (
        <div key={row.label} className="mt-1 flex items-center gap-2">
          {row.color ? (
            <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: row.color }} />
          ) : null}
          <span className="text-muted-foreground">{row.label}</span>
          <span className="ml-auto pl-3 font-medium text-popover-foreground tabular">{row.value}</span>
        </div>
      ))}
    </div>
  )
}
