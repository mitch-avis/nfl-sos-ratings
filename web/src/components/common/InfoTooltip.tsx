import { Info } from 'lucide-react'
import type { ReactNode } from 'react'

import { Hint } from './Hint'

/** A small info button that reveals `content` on hover or focus, or on tap on touch screens. */
export function InfoTooltip({ content, label = 'More information' }: { content: ReactNode; label?: string }) {
  return (
    <Hint content={content}>
      <button
        type="button"
        aria-label={label}
        className="-m-1 inline-flex size-6 items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus-visible:outline-2"
      >
        <Info className="size-3.5" />
      </button>
    </Hint>
  )
}
