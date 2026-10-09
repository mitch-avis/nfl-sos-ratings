import { CalendarClock } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

import { useRankRanges, useWpRatings } from '@/api/queries'
import type { EntityKind, SeasonDataset } from '@/api/types'
import { useEntityPageState } from '@/app/EntityViewStateProvider'
import { useWpThreshold } from '@/app/useWpThreshold'
import { ErrorState } from '@/components/common/ErrorState'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { Notice } from '@/components/common/Notice'
import { PageHeader } from '@/components/common/PageHeader'
import { ReadingNotes } from '@/components/common/ReadingNotes'
import { ComparisonPanel } from '@/components/entity/ComparisonPanel'
import { EntityTable } from '@/components/entity/EntityTable'
import { RankRangeChart } from '@/components/entity/RankRangeChart'
import { ViewControls } from '@/components/entity/ViewControls'
import { WpFilterControl } from '@/components/entity/WpFilterControl'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { getEntityConfig } from '@/domain/entityConfig'
import {
  canResetPageView,
  reconcileCompareIds,
  seasonRedirectState,
  toggleCompareId,
  toggleSubcategoryPatch,
} from '@/domain/pageViewState'
import { isMissingRankRanges, parseRankRanges } from '@/domain/rankRanges'
import { getInProgressGames } from '@/domain/seasonRules'
import { buildSeasonViewTable } from '@/domain/viewModel'
import { withWpColumns } from '@/domain/wpFilter'

function sameIds(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

/**
 * Keep the compared rows in the `?compare=` query string so a comparison can be shared: a new
 * `?compare=` value is read into the page state once, and after that the state drives the URL.
 */
function useCompareQuerySync(
  kind: EntityKind,
  dataset: SeasonDataset,
  compareIds: string[],
  setCompareIds: (ids: string[]) => void,
) {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const navigate = useNavigate()
  const hydratedKey = useRef<string | null>(null)

  useEffect(() => {
    const fromQuery = reconcileCompareIds(
      kind,
      dataset[kind],
      (searchParams.get('compare') ?? '').split(',').map((value) => value.trim()).filter(Boolean),
    )
    const queryKey = `${kind}:${dataset.season}:${fromQuery.join(',')}`
    if (fromQuery.length > 0 && hydratedKey.current !== queryKey) {
      hydratedKey.current = queryKey
      if (!sameIds(fromQuery, compareIds)) {
        setCompareIds(fromQuery)
        return
      }
    }
    const params = new URLSearchParams(searchParams)
    params.set('season', String(dataset.season))
    if (compareIds.length > 0) params.set('compare', compareIds.join(','))
    else params.delete('compare')
    if (params.toString() !== searchParams.toString()) {
      hydratedKey.current = `${kind}:${dataset.season}:${compareIds.join(',')}`
      navigate(`${location.pathname}?${params.toString()}`, {
        replace: true,
        state: seasonRedirectState(location.state, searchParams.get('season'), dataset.season),
      })
    }
  }, [compareIds, dataset, kind, location.pathname, location.state, navigate, searchParams, setCompareIds])
}

/** The id a team or QB page could not find, passed along when it sent the reader back here. */
function notFoundId(state: unknown): string | null {
  if (typeof state !== 'object' || state === null || !('notFound' in state)) return null
  return typeof state.notFound === 'string' ? state.notFound : null
}

// How to use the table, ahead of the page's own reading notes.
const TABLE_NOTE = 'Sort any column, search, and switch views; open a row for its game-by-game detail.'

/** One line under the title while a season is under way: how far it is, and what that means. */
function SeasonProgress({ kind, games }: { kind: EntityKind; games: number }) {
  return (
    <p className="-mt-6 flex max-w-prose items-start gap-2 text-sm text-muted-foreground">
      <CalendarClock className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
      <span>
        <span className="font-medium text-foreground">Season in progress: up to {games} games per team.</span>{' '}
        {kind === 'teams'
          ? "Until a team has played 9 games, its rating also leans on its rating last season, a little less after each game, so ratings will move as the season goes on."
          : 'Ratings use only the games played so far, so they lean toward the league average and will move as the season goes on.'}
      </span>
    </p>
  )
}

/** The toolbar's note of the picked rows: how many, a jump to the comparison, and a reset. */
function SelectionBar({ count, onView, onClear }: { count: number; onView: () => void; onClear: () => void }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="font-medium">{count} selected</span>
      <Button size="sm" variant="secondary" onClick={onView}>
        View comparison
      </Button>
      <Button size="sm" variant="ghost" onClick={onClear}>
        Clear selection
      </Button>
    </div>
  )
}

/** The Teams or QBs index: the ranking line, the table with its filter, comparison, and rank ranges. */
export function EntityIndexPage({ kind, dataset }: { kind: EntityKind; dataset: SeasonDataset }) {
  const config = getEntityConfig(kind)
  const notFound = notFoundId(useLocation().state)
  const state = useEntityPageState(kind)
  const table = dataset[kind]
  const compareIds = useMemo(() => reconcileCompareIds(kind, table, state.compareIds), [kind, state.compareIds, table])
  const { update } = state

  useEffect(() => {
    if (!sameIds(compareIds, state.compareIds)) update({ compareIds })
  }, [compareIds, state.compareIds, update])
  const setCompareIds = useCallback((ids: string[]) => update({ compareIds: ids }), [update])
  // Stable across unrelated re-renders (another query settling, for example), so the table does not
  // rebuild its columns and remount every checkbox.
  const toggleCompare = useCallback(
    (entityId: string) => update({ compareIds: toggleCompareId(compareIds, entityId) }),
    [compareIds, update],
  )
  useCompareQuerySync(kind, dataset, compareIds, setCompareIds)
  const rankRangesQuery = useRankRanges(kind, dataset.season)
  const rankRanges = useMemo(
    () => (rankRangesQuery.data ? parseRankRanges(kind, rankRangesQuery.data) : undefined),
    [kind, rankRangesQuery.data],
  )

  const seasonView = useMemo(() => buildSeasonViewTable(kind, table, state.viewState), [kind, state.viewState, table])
  const displayTable = useMemo(() => {
    if (kind !== 'qbs' || state.showUnratedRows) return seasonView.table
    return {
      ...seasonView.table,
      rows: seasonView.table.rows.filter((row) => row.qb_is_eligible === true),
    }
  }, [kind, seasonView.table, state.showUnratedRows])
  const [wpThreshold] = useWpThreshold()
  const wpQuery = useWpRatings(kind, dataset.season, wpThreshold)
  // At a non-zero threshold the filtered rating and rank sit beside the published rating.
  const tableView = useMemo(
    () => withWpColumns(kind, displayTable, seasonView.selectedColumns, wpThreshold > 0 ? wpQuery.data : undefined),
    [displayTable, kind, seasonView.selectedColumns, wpQuery.data, wpThreshold],
  )
  const compareColumns = useMemo(() => {
    // The first column already names each row, so only a QB's team stays from the identity columns.
    const requested = seasonView.selectedColumns.filter(
      (column) => !config.identityColumns.includes(column) || (kind === 'qbs' && column === 'team'),
    )
    return requested.length > 0 ? requested : config.compareColumns
  }, [config.compareColumns, config.identityColumns, kind, seasonView.selectedColumns])
  const { viewState } = state
  const season = dataset.season
  const gamesSoFar = getInProgressGames(dataset)
  const comparisonRef = useRef<HTMLDivElement>(null)

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={`${config.title} · ${season}`}
        description={
          <>
            {config.primaryRankingDescription} <ReadingNotes notes={[TABLE_NOTE, ...config.pageNotes]} />
          </>
        }
      />

      {gamesSoFar !== null ? <SeasonProgress kind={kind} games={gamesSoFar} /> : null}

      {notFound !== null ? (
        <Notice>
          No {kind === 'teams' ? 'team' : 'quarterback'} {notFound} in {season}.
        </Notice>
      ) : null}

      <EntityTable
        key={`${kind}-${season}`}
        compareIds={compareIds}
        config={config}
        controls={
          <ViewControls
            canReset={canResetPageView(kind, { ...state, compareIds })}
            kind={kind}
            onReset={state.reset}
            onSelectTeamCategory={(teamCategory) => update({ teamCategory })}
            onSelectView={(primaryView) => update({ primaryView })}
            onToggleSubcategory={(subcategory) => update(toggleSubcategoryPatch(kind, viewState, subcategory))}
            state={viewState}
          />
        }
        onQueryChange={(query) => update({ query })}
        onSortingChange={(sorting) => update({ sorting })}
        onToggleCompare={toggleCompare}
        query={state.query}
        rankRanges={rankRanges}
        season={season}
        selectedColumns={tableView.selectedColumns}
        sorting={state.sorting}
        table={tableView.table}
        toolbar={
          <>
            <WpFilterControl kind={kind} season={season} where="beside the published rating in the Ratings view" />
            {kind === 'qbs' ? (
              <div className="flex items-center gap-2">
                <Switch
                  id="show-unrated"
                  checked={state.showUnratedRows}
                  onCheckedChange={(checked) => update({ showUnratedRows: checked })}
                />
                <Label htmlFor="show-unrated">Show QBs below the qualifier</Label>
                <InfoTooltip
                  label="About the qualifier"
                  content={`A quarterback is ranked once they have 14 pass attempts for every game their team has played${gamesSoFar !== null ? ' so far, so a team\'s bye lowers its quarterbacks\' mark by 14' : ''}. Switch on to list the passers below that mark too; they have no rank range. Each quarterback's mark is the Att to Qualify column in Raw Total Stats.`}
                />
              </div>
            ) : null}
            {compareIds.length > 0 ? (
              <SelectionBar
                count={compareIds.length}
                onView={() => comparisonRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                onClear={() => update({ compareIds: [] })}
              />
            ) : null}
          </>
        }
      />

      <div ref={comparisonRef} className="scroll-mt-20">
        <ComparisonPanel
          compareColumns={compareColumns}
          compareIds={compareIds}
          config={config}
          season={season}
          table={displayTable}
          rankRanges={rankRanges}
          onRemove={(entityId) => update({ compareIds: toggleCompareId(compareIds, entityId) })}
        />
      </div>

      {rankRangesQuery.isError && !isMissingRankRanges(rankRangesQuery.error) ? (
        <ErrorState error={rankRangesQuery.error} title="Could not load the rank ranges" />
      ) : null}
      {rankRanges && rankRanges.length > 0 ? (
        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">Rank ranges</CardTitle>
            <CardDescription>
              Where each {kind === 'teams' ? 'team' : 'qualifying QB'} ranks across 1,000 redraws of
              the {season} season: its games drawn at random, with repeats, and everyone re-rated each
              time. The spread shows how much a rank depends on which games happened to be played, not
              whether the model is right. Seasons in progress show wider ranges.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RankRangeChart kind={kind} season={season} ranges={rankRanges} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}
