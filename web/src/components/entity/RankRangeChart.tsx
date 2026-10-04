import { Link } from 'react-router'

import type { EntityKind } from '@/api/types'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  rankCenter,
  rankChanceText,
  rankRangeHeadline,
  rankRangeSummary,
  rankRangesByMedian,
  rankTicks,
  type RankRange,
} from '@/domain/rankRanges'

import { RankIntervalKey, RankIntervalTrack } from './RankInterval'

// One label width per kind keeps every row's track aligned: team codes are short, QB names long.
const ROW_GRID: Record<EntityKind, string> = {
  teams: 'grid grid-cols-[2.75rem_minmax(0,1fr)] items-center gap-x-2',
  qbs: 'grid grid-cols-[6.5rem_minmax(0,1fr)] items-center gap-x-2 sm:grid-cols-[9rem_minmax(0,1fr)]',
}

/**
 * The league's rank ranges, one row per team or QB ordered by median rank, rank 1 at the left.
 * Each row links to the detail page; its accessible name is the interval in words.
 */
export function RankRangeChart({ kind, season, ranges }: { kind: EntityKind; season: number; ranges: RankRange[] }) {
  const count = ranges.length
  const ticks = rankTicks(count)
  return (
    <div className="flex flex-col gap-3">
      <RankIntervalKey />
      <div className={ROW_GRID[kind]} aria-hidden>
        <span className="text-xs text-muted-foreground">Rank</span>
        <div className="relative h-4 text-xs text-muted-foreground tabular">
          {ticks.map((tick) => (
            <span key={tick} className="absolute -translate-x-1/2" style={{ left: `${rankCenter(tick, count)}%` }}>
              {tick}
            </span>
          ))}
        </div>
      </div>
      <ol className="flex flex-col">
        {rankRangesByMedian(ranges).map((range) => (
          <li key={range.id}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to={`/${kind}/${encodeURIComponent(range.id)}?season=${season}`}
                  aria-label={rankRangeSummary(range)}
                  className={`${ROW_GRID[kind]} rounded-sm px-1 py-1 hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring`}
                >
                  <span className="truncate text-sm font-medium">{range.label}</span>
                  <RankIntervalTrack range={range} count={count} ticks={ticks} />
                </Link>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-pretty">
                <div className="font-medium">
                  {range.label}: {rankRangeHeadline(range)}
                </div>
                <div>{rankChanceText(kind, range)}</div>
              </TooltipContent>
            </Tooltip>
          </li>
        ))}
      </ol>
    </div>
  )
}
