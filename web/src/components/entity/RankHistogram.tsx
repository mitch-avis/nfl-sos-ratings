import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { ChartTooltipCard } from '@/components/common/ChartTooltip'
import { formatChance, ordinal, rankHistogram, type RankRange } from '@/domain/rankRanges'
import { useHasHover } from '@/hooks/use-has-hover'

/**
 * P(rank = k) across resamples: one column per rank, the published rank in the full series color
 * and the others lighter. A visually hidden table carries the same numbers for screen readers.
 */
export function RankHistogram({ range }: { range: RankRange }) {
  const hasHover = useHasHover()
  const bars = rankHistogram(range)
  if (bars.length === 0) return null
  // Plot whole percents so the axis picks clean ticks (0, 4, 8, ...) rather than 3.5% steps.
  const points = bars.map((bar) => ({ ...bar, percent: bar.probability * 100 }))
  return (
    <div className="flex flex-col gap-2">
      <div className="h-48 w-full" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={points} margin={{ top: 8, right: 8, bottom: 4, left: 4 }} barCategoryGap={2}>
            <CartesianGrid vertical={false} className="stroke-border" />
            <XAxis dataKey="rank" tickLine={false} interval="preserveStartEnd" className="text-xs" />
            <YAxis
              tickLine={false}
              width={40}
              className="text-xs"
              allowDecimals={false}
              tickFormatter={(value: number) => `${value}%`}
            />
            <Tooltip
              trigger={hasHover ? 'hover' : 'click'}
              cursor={{ fill: 'var(--muted)', fillOpacity: 0.6 }}
              content={({ active, payload }) => {
                const bar = payload?.[0]?.payload as (typeof points)[number] | undefined
                if (!active || !bar) return null
                return (
                  <ChartTooltipCard
                    title={bar.published ? `Ranked ${ordinal(bar.rank)} (published rank)` : `Ranked ${ordinal(bar.rank)}`}
                    rows={[{ label: 'Share of redraws', value: formatChance(bar.probability), color: 'var(--chart-1)' }]}
                  />
                )
              }}
            />
            <Bar dataKey="percent" maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
              {points.map((bar) => (
                <Cell key={bar.rank} fill="var(--chart-1)" fillOpacity={bar.published ? 1 : 0.45} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        How often each rank came up across the 1,000 redraws. Rank 1 is at the left; the highlighted
        column is the published rank ({ordinal(range.publishedRank)}).
      </p>
      <table className="sr-only">
        <caption>Chance of each rank</caption>
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Chance</th>
          </tr>
        </thead>
        <tbody>
          {bars.map((bar) => (
            <tr key={bar.rank}>
              <th scope="row">{ordinal(bar.rank)}</th>
              <td>{formatChance(bar.probability)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
