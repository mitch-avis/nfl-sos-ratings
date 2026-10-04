import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { RowValue } from '@/api/types'
import { ChartTooltipCard } from '@/components/common/ChartTooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { formatValue } from '@/domain/format'
import { getMetricMetadata } from '@/domain/metricMetadata'
import { buildTrendPoints, meanReference, type TrendPoint, type TrendReference } from '@/domain/trend'
import { useHasHover } from '@/hooks/use-has-hover'

/**
 * A line chart of one metric by week. The metric picker offers the numeric `columns`.
 *
 * `reference` is the dashed line: `'mean'` (the default) draws the mean of the points shown, a
 * `TrendReference` draws a fixed value, and `null` draws none.
 */
export function WeeklyTrendChart({
  rows,
  columns,
  reference = 'mean',
}: {
  rows: Array<Record<string, RowValue>>
  columns: string[]
  reference?: 'mean' | TrendReference | null
}) {
  const numericColumns = useMemo(
    () => columns.filter((column) => column !== 'week' && rows.some((row) => typeof row[column] === 'number')),
    [columns, rows],
  )
  const [picked, setPicked] = useState<string | null>(null)
  const hasHover = useHasHover()
  const column = picked !== null && numericColumns.includes(picked) ? picked : numericColumns[0]
  const points = useMemo(() => (column ? buildTrendPoints(rows, column) : []), [column, rows])
  if (!column || points.length < 2) return null

  const line = reference === 'mean' ? meanReference(points) : reference
  const label = getMetricMetadata(column).label

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-medium">{label} by week</div>
        <Select value={column} onValueChange={setPicked}>
          <SelectTrigger size="sm" className="w-56" aria-label="Metric to chart">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {numericColumns.map((candidate) => (
              <SelectItem key={candidate} value={candidate}>
                {getMetricMetadata(candidate).label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="week" tickLine={false} className="text-xs" />
            <YAxis tickLine={false} width={56} className="text-xs" tickFormatter={(value: number) => formatValue(value)} />
            <Tooltip
              trigger={hasHover ? 'hover' : 'click'}
              content={({ active, payload }) => {
                const point = payload?.[0]?.payload as TrendPoint | undefined
                if (!active || !point) return null
                return (
                  <ChartTooltipCard
                    title={point.opponent ? `Week ${point.week} vs ${point.opponent}` : `Week ${point.week}`}
                    rows={[{ label, value: formatValue(point.value), color: 'var(--chart-1)' }]}
                  />
                )
              }}
            />
            {line ? <ReferenceLine y={line.value} stroke="var(--muted-foreground)" strokeDasharray="4 4" /> : null}
            <Line type="monotone" dataKey="value" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {line ? <p className="text-xs text-muted-foreground">{line.caption}</p> : null}
    </div>
  )
}
