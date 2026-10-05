import { RankIntervalTrack } from '@/components/entity/RankInterval'
import { rankRangeHeadline, type UnitRankRange } from '@/domain/rankRanges'

/**
 * A team's offense, defense, and special-teams rank ranges beside its overall one: the published
 * rank, the middle 50% and 95% of resampled ranks, and a mini interval on a track of `count` ranks.
 */
export function UnitRankRangeTable({ ranges, count }: { ranges: UnitRankRange[]; count: number }) {
  return (
    <div className="flex flex-col gap-1.5">
      <table className="w-full text-sm">
        <caption className="sr-only">Rank range by unit</caption>
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="py-1.5 pr-3 font-medium">Unit</th>
            <th className="py-1.5 pr-3 font-medium">Rank range</th>
            <th className="w-1/4 py-1.5 font-medium">
              <span className="sr-only">Interval</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {ranges.map((range) => (
            <tr key={range.unit} className="border-b last:border-0">
              <th scope="row" className="py-1.5 pr-3 text-left font-medium">
                {range.label}
              </th>
              <td className="py-1.5 pr-3 tabular">{rankRangeHeadline(range)}</td>
              <td className="py-1.5">
                <RankIntervalTrack range={range} count={count} size="mini" showPublished />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted-foreground">
        Unit percentiles do not add up to the team&apos;s: the median of a sum is not the sum of the
        medians.
      </p>
    </div>
  )
}
