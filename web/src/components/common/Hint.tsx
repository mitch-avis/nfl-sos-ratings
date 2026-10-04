import type { ReactElement, ReactNode } from 'react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useHasHover } from '@/hooks/use-has-hover'
import { cn } from '@/utils/cn'

import { HINT_CARD_CLASS } from './hintStyles'

type Side = 'top' | 'right' | 'bottom' | 'left'

/**
 * Explains `children` in a hint card: on hover or keyboard focus with a mouse, and on tap on touch
 * screens, where tapping anywhere else closes it. `children` must be one focusable element, such
 * as a button.
 */
export function Hint({
  content,
  children,
  side = 'top',
  className,
}: {
  content: ReactNode
  children: ReactElement
  side?: Side
  className?: string
}) {
  const hasHover = useHasHover()
  if (hasHover) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent side={side} className={className}>
          {content}
        </TooltipContent>
      </Tooltip>
    )
  }
  return (
    <Popover>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent side={side} sideOffset={6} className={cn(HINT_CARD_CLASS, className)}>
        {content}
      </PopoverContent>
    </Popover>
  )
}
