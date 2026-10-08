import { BookOpen } from 'lucide-react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/utils/cn'

import { HINT_CARD_CLASS } from './hintStyles'

/**
 * "How to read this page": a page's reading notes in a card that opens on click or tap. A popover
 * rather than a hover hint, because the notes are a few sentences long.
 */
export function ReadingNotes({ notes }: { notes: string[] }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-ring"
        >
          <BookOpen className="size-3.5" aria-hidden />
          How to read this page
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className={cn(HINT_CARD_CLASS, 'max-w-sm')}>
        <ul className="list-disc space-y-1.5 pl-4">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  )
}
