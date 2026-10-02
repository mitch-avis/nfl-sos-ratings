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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { formatValue } from '@/domain/format'
import { getMetricMetadata } from '@/domain/metricMetadata'
import { buildTrendPoints, type TrendPoint } from '@/domain/trend'

/**
 * A line chart of one game-log metric by week, with the season mean as a reference line.
 * The metric picker offers the numeric columns of the current view.
 */
export function WeeklyTrendChart({
  rows,
  columns,
}: {
  rows: Array<Record<string, RowValue>>
  columns: string[]
}) {
  const numericColumns = useMemo(
    () => columns.filter((column) => column !== 'week' && rows.some((row) => typeof row[column] === 'number')),
    [columns, rows],
  )
  const [picked, setPicked] = useState<string | null>(null)
  const column = picked !== null && numericColumns.includes(picked) ? picked : numericColumns[0]
  const points = useMemo(() => (column ? buildTrendPoints(rows, column) : []), [column, rows])
  if (!column || points.length < 2) return null

  const mean = points.reduce((total, point) => total + point.value, 0) / points.length
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
              formatter={(value) => [formatValue(Number(value)), label]}
              labelFormatter={(week, payload) => {
                const opponent = (payload?.[0]?.payload as TrendPoint | undefined)?.opponent
                return opponent ? `Week ${week} vs ${opponent}` : `Week ${week}`
              }}
              contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8 }}
            />
            <ReferenceLine y={mean} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="value" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">Dashed line: the mean of the games shown ({formatValue(mean)}).</p>
    </div>
  )
}
