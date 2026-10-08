import { ChevronRight } from 'lucide-react'
import { useState } from 'react'

import type { EntityKind } from '@/api/types'
import { useWpThreshold } from '@/app/useWpThreshold'

import { WpFilterPanel } from './WpFilterPanel'

/**
 * The garbage-time filter as a folded exploration section: closed by default, open when the
 * address already carries a threshold, so a shared link shows what it was shared for.
 */
export function WpExploration({ kind, season }: { kind: EntityKind; season: number }) {
  const [threshold] = useWpThreshold()
  const [open, setOpen] = useState(threshold > 0)
  return (
    <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)} className="group flex flex-col gap-3">
      <summary className="flex w-fit cursor-pointer list-none items-center gap-2 rounded-sm py-1 font-medium focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-4 text-muted-foreground transition-transform group-open:rotate-90" aria-hidden />
        Explore ratings without garbage time
        <span className="text-sm font-normal text-muted-foreground">
          {threshold > 0 ? `filter at ${threshold}%` : 'every play counts'}
        </span>
      </summary>
      <WpFilterPanel kind={kind} season={season} />
    </details>
  )
}
