import { Skeleton } from '@/components/ui/skeleton'

/** Placeholder blocks while a page's data loads. */
export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-8 w-64 bg-muted" />
      <div className="grid gap-3 sm:grid-cols-3">
        <Skeleton className="h-20 bg-muted" />
        <Skeleton className="h-20 bg-muted" />
        <Skeleton className="h-20 bg-muted" />
      </div>
      <Skeleton className="h-96 bg-muted" />
    </div>
  )
}
