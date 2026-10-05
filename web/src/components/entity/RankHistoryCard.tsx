import { useId, useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { useRankHistory } from '@/api/queries'
import type { EntityKind } from '@/api/types'
import { ChartTooltipCard } from '@/components/common/ChartTooltip'
import { ErrorState } from '@/components/common/ErrorState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  describeRankHistory,
  isMissingRankHistory,
  parseRankHistory,
  rankAxisMax,
  rankBandData,
  type RankHistoryPoint,
} from '@/domain/rankHistory'
import { ordinal, rankTicks } from '@/domain/rankRanges'
import { useHasHover } from '@/hooks/use-has-hover'

function rangeText(low: number | null, high: number | null): string {
  return low === null || high === null ? 'n/a' : `${ordinal(low)}–${ordinal(high)}`
}

function medianText(point: RankHistoryPoint): string {
  return point.median === null ? 'n/a' : ordinal(point.median)
}

/**
 * The rank range week by week for a season in progress: the median rank as a line with the middle
 * 50% and 95% of redraws as bands, rank 1 at the top. Hidden for seasons without weekly ranges.
 */
export function RankHistoryCard({
  kind,
  season,
  entityId,
  count = 0,
}: {
  kind: EntityKind
  season: number
  entityId: string
  /** How many teams or QBs are ranked, so the axis runs from 1 to the last rank. */
  count?: number
}) {
  const query = useRankHistory(kind, season, entityId)
  const titleId = useId()
  const hasHover = useHasHover()
  const points = useMemo(() => (query.data ? parseRankHistory(kind, query.data) : []), [kind, query.data])

  if (query.isError) {
    return isMissingRankHistory(query.error) ? null : (
      <ErrorState error={query.error} title="Could not load the rank by week" />
    )
  }
  if (points.length === 0) return null
  const byWeek = new Map(points.map((point) => [point.week, point]))
  const axisMax = Math.max(count, rankAxisMax(points))

  return (
    <Card role="region" aria-labelledby={titleId} className="gap-4">
      <CardHeader>
        <CardTitle id={titleId} className="text-base">
          Rank by week
        </CardTitle>
        <CardDescription>
          The rank when the {season} games through each week are redrawn at random and the ratings
          refit, rank 1 at the top. The line is the median redraw, the darker band the middle 50%,
          and the lighter band the middle 95%. In the first weeks, with one or two games per team, a
          redraw can only repeat or drop a team&apos;s games, never change their results, so those
          bands understate the uncertainty.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <p className="text-sm">{describeRankHistory(points)}</p>
        <div className="h-56 w-full" aria-hidden>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rankBandData(points)} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="week" tickLine={false} className="text-xs" />
              <YAxis
                reversed
                domain={[1, axisMax]}
                ticks={rankTicks(axisMax)}
                interval={0}
                allowDecimals={false}
                tickLine={false}
                width={44}
                className="text-xs"
                tickFormatter={(value: number) => ordinal(value)}
              />
              <Tooltip
                trigger={hasHover ? 'hover' : 'click'}
                content={({ active, label }) => {
                  const point = typeof label === 'number' ? byWeek.get(label) : undefined
                  if (!active || !point) return null
                  return (
                    <ChartTooltipCard
                      title={`Week ${point.week}`}
                      rows={[
                        { label: 'Median rank', value: medianText(point), color: 'var(--chart-1)' },
                        { label: 'Middle 50%', value: rangeText(point.low50, point.high50) },
                        { label: 'Middle 95%', value: rangeText(point.low95, point.high95) },
                      ]}
                    />
                  )
                }}
              />
              <Area dataKey="band95" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} isAnimationActive={false} />
              <Area dataKey="band50" stroke="none" fill="var(--chart-1)" fillOpacity={0.35} isAnimationActive={false} />
              <Line dataKey="median" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <table className="sr-only">
          <caption>Rank by week</caption>
          <thead>
            <tr>
              <th scope="col">Week</th>
              <th scope="col">Median rank</th>
              <th scope="col">Middle 50%</th>
              <th scope="col">Middle 95%</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={point.week}>
                <th scope="row">{point.week}</th>
                <td>{medianText(point)}</td>
                <td>{rangeText(point.low50, point.high50)}</td>
                <td>{rangeText(point.low95, point.high95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
