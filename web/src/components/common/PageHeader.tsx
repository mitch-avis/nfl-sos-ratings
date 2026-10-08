import type { ReactNode } from 'react'

/** Page title (with an optional decorative mark before it), description, and right-aligned actions. */
export function PageHeader({
  title,
  titleMark,
  description,
  actions,
}: {
  title: string
  titleMark?: ReactNode
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h1 className="flex items-center gap-2.5 text-xl font-semibold tracking-tight sm:text-2xl">
          {titleMark}
          {title}
        </h1>
        {description ? <p className="mt-1 max-w-prose text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  )
}
