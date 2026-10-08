import { ArrowRight } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

import type { EntityKind } from '@/api/types'
import { HINT_CARD_CLASS } from '@/components/common/hintStyles'
import {
  rankCenter,
  rankChanceText,
  rankRangeHeadline,
  rankRangeSummary,
  rankRangesByMedian,
  rankTicks,
  type RankRange,
} from '@/domain/rankRanges'
import { useHasHover } from '@/hooks/use-has-hover'
import { TeamChip } from '@/components/common/TeamChip'
import { cn } from '@/utils/cn'

import { RankIntervalKey, RankIntervalTrack } from './RankInterval'

// One label width per kind keeps every row's track aligned: team codes are short, QB names long.
const ROW_GRID: Record<EntityKind, string> = {
  teams: 'grid grid-cols-[3.5rem_minmax(0,1fr)] items-center gap-x-2',
  qbs: 'grid grid-cols-[6.5rem_minmax(0,1fr)] items-center gap-x-2 sm:grid-cols-[9rem_minmax(0,1fr)]',
}
const ROW_CLASS = 'w-full rounded-sm px-1 py-1 text-left hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring'

/** The pinned card above the rows: the active row's ranges in words, or how to pick a row. */
function Readout({ kind, season, range, hasHover }: { kind: EntityKind; season: number; range: RankRange | undefined; hasHover: boolean }) {
  return (
    <div role="status" aria-live="polite" className={cn(HINT_CARD_CLASS, 'sticky top-16 z-10 w-full max-w-none')}>
      {range ? (
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <div>
            <div className="font-medium">
              {range.label}: {rankRangeHeadline(range)}
            </div>
            <div className="text-muted-foreground">{rankChanceText(kind, range)}</div>
          </div>
          {hasHover ? null : (
            <Link
              to={`/${kind}/${encodeURIComponent(range.id)}?season=${season}`}
              className="inline-flex items-center gap-1 font-medium text-primary"
            >
              Open {range.label}
              <ArrowRight className="size-3.5" aria-hidden />
            </Link>
          )}
        </div>
      ) : (
        <div className="text-muted-foreground">
          {hasHover ? 'Hover or focus a row for its numbers; click it to open the detail page.' : 'Tap a row for its numbers.'}
        </div>
      )}
    </div>
  )
}

/**
 * The league's rank ranges, one row per team or QB ordered by median rank, rank 1 at the left.
 * The readout above the rows describes the hovered, focused, or tapped row, so no floating card
 * covers the neighboring rows. With a mouse each row links to its detail page; on touch screens a
 * tap selects the row and the readout carries the link. Each row's accessible name is its interval
 * in words.
 */
export function RankRangeChart({ kind, season, ranges }: { kind: EntityKind; season: number; ranges: RankRange[] }) {
  const hasHover = useHasHover()
  const [activeId, setActiveId] = useState<string | null>(null)
  const count = ranges.length
  const ticks = rankTicks(count)
  const active = ranges.find((range) => range.id === activeId)
  return (
    <div className="flex flex-col gap-3">
      <RankIntervalKey />
      <Readout kind={kind} season={season} range={active} hasHover={hasHover} />
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
        {rankRangesByMedian(ranges).map((range) => {
          const row = (
            <>
              <span className="flex min-w-0 items-center gap-1.5 text-sm font-medium">
                {kind === 'teams' ? <TeamChip team={range.id} /> : null}
                <span className="truncate">{range.label}</span>
              </span>
              <RankIntervalTrack range={range} count={count} ticks={ticks} />
            </>
          )
          const className = cn(ROW_GRID[kind], ROW_CLASS, range.id === activeId && 'bg-muted/60')
          return (
            <li key={range.id}>
              {hasHover ? (
                <Link
                  to={`/${kind}/${encodeURIComponent(range.id)}?season=${season}`}
                  aria-label={rankRangeSummary(range)}
                  className={className}
                  onMouseEnter={() => setActiveId(range.id)}
                  onFocus={() => setActiveId(range.id)}
                >
                  {row}
                </Link>
              ) : (
                <button
                  type="button"
                  aria-label={rankRangeSummary(range)}
                  aria-pressed={range.id === activeId}
                  className={className}
                  onClick={() => setActiveId(range.id === activeId ? null : range.id)}
                >
                  {row}
                </button>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
