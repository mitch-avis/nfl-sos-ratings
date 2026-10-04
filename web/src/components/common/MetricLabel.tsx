import { getMetricMetadata, getMetricTooltip } from '@/domain/metricMetadata'
import { useHasHover } from '@/hooks/use-has-hover'
import { cn } from '@/utils/cn'

import { Hint } from './Hint'

/**
 * A metric label that explains the metric from the metric registry: on hover or focus with a
 * mouse, on tap on touch screens. Sortable headers use `SortableHeader` instead, so a tap there can
 * still sort.
 *
 * `label` and `tooltip` override the registry text for derived columns.
 */
export function MetricLabel({
  column,
  label,
  tooltip,
  className,
}: {
  column: string
  label?: string
  tooltip?: string
  className?: string
}) {
  const hasHover = useHasHover()
  const classes = cn(
    'cursor-help text-left underline decoration-muted-foreground/40 decoration-dotted underline-offset-4',
    className,
  )
  const text = label ?? getMetricMetadata(column).label
  return (
    <Hint content={tooltip ?? getMetricTooltip(column)} className="max-w-sm">
      {hasHover ? (
        <span tabIndex={0} className={classes}>
          {text}
        </span>
      ) : (
        <button type="button" className={classes}>
          {text}
        </button>
      )}
    </Hint>
  )
}
