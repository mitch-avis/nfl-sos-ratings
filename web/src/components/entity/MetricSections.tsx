import type { EntityKind, RowValue } from '@/api/types'
import { MetricLabel } from '@/components/common/MetricLabel'
import { bucketColumns, detailHeaderLabel } from '@/domain/detailSections'
import { formatValue } from '@/domain/format'
import { cn } from '@/utils/cn'

/** The subject's season values for the current view, grouped by category. */
export function MetricSections({
  kind,
  row,
  columns,
  isRatingsView,
}: {
  kind: EntityKind
  row: Record<string, RowValue>
  columns: string[]
  isRatingsView: boolean
}) {
  const sections = bucketColumns(kind, isRatingsView, columns)
  return (
    <div className="flex flex-col gap-5">
      {sections.map((section) => (
        <section key={section.title} aria-label={section.title} className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold">{section.title}</h3>
          <dl
            className={cn(
              'grid gap-2',
              isRatingsView
                ? 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'
                : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6',
            )}
          >
            {section.columns.map((column) => (
              <div key={column} className="rounded-md border bg-muted/30 px-3 py-2">
                <dt className="text-xs text-muted-foreground">
                  <MetricLabel column={column} label={detailHeaderLabel(column)} />
                </dt>
                <dd className="tabular text-lg font-semibold">{formatValue(row[column] ?? null)}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  )
}
