import type { ReactNode } from 'react'

/** A short, polite status line near the top of a page: something the reader should know. */
export function Notice({ children }: { children: ReactNode }) {
  return (
    <p role="status" className="rounded-lg border bg-card px-4 py-2 text-sm text-muted-foreground">
      {children}
    </p>
  )
}
