import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { getMetricMetadata, getMetricTooltip } from '@/domain/metricMetadata'
import { cn } from '@/utils/cn'

/**
 * A column label that explains the metric on hover or focus, from the metric registry.
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
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'cursor-help underline decoration-muted-foreground/40 decoration-dotted underline-offset-4',
            className,
          )}
        >
          {label ?? getMetricMetadata(column).label}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm text-pretty">{tooltip ?? getMetricTooltip(column)}</TooltipContent>
    </Tooltip>
  )
}
