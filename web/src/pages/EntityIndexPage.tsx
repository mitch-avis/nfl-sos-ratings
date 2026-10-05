import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

import { useRankRanges } from '@/api/queries'
import type { EntityKind, SeasonDataset } from '@/api/types'
import { useEntityPageState } from '@/app/EntityViewStateProvider'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { ComparisonPanel } from '@/components/entity/ComparisonPanel'
import { EntityTable } from '@/components/entity/EntityTable'
import { RankRangeChart } from '@/components/entity/RankRangeChart'
import { ViewControls } from '@/components/entity/ViewControls'
import { WpFilterPanel } from '@/components/entity/WpFilterPanel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { getEntityConfig } from '@/domain/entityConfig'
import {
  canResetPageView,
  reconcileCompareIds,
  toggleCompareId,
  toggleSubcategoryPatch,
} from '@/domain/pageViewState'
import { isMissingRankRanges, parseRankRanges } from '@/domain/rankRanges'
import { getInProgressGames } from '@/domain/seasonRules'
import { buildSeasonViewTable } from '@/domain/viewModel'

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
      navigate(`${location.pathname}?${params.toString()}`, { replace: true })
    }
  }, [compareIds, dataset, kind, location.pathname, navigate, searchParams, setCompareIds])
}

/** The Teams or QBs index: guidance, summary tiles, comparison, and the main table. */
export function EntityIndexPage({ kind, dataset }: { kind: EntityKind; dataset: SeasonDataset }) {
  const config = getEntityConfig(kind)
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
  const compareColumns = useMemo(() => {
    // The first column already names each row, so only a QB's team stays from the identity columns.
    const requested = seasonView.selectedColumns.filter(
      (column) => !config.identityColumns.includes(column) || (kind === 'qbs' && column === 'team'),
    )
    return requested.length > 0 ? requested : config.compareColumns
  }, [config.compareColumns, config.identityColumns, kind, seasonView.selectedColumns])
  const { viewState } = state
  const season = dataset.season
  const gamesSoFar = getInProgressGames(season, dataset.teams.rows)

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={`${config.title} · ${season}`}
        description="Every rating and stat compares each subject with the opponents it actually faced. Sort, filter, and switch views; open a row for its game-by-game detail."
      />

      {gamesSoFar !== null ? (
        <Alert>
          <AlertTitle>Season in progress: up to {gamesSoFar} games per team</AlertTitle>
          <AlertDescription>
            Ratings use only the games played so far. With this few games they are pulled strongly
            toward the league average and will move as the season goes on.
          </AlertDescription>
        </Alert>
      ) : null}

      <Card className="gap-2 px-4 py-3">
        <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Use first</div>
        <div className="font-semibold">{config.primaryRankingLabel}</div>
        <p className="max-w-prose text-sm text-muted-foreground">{config.primaryRankingDescription}</p>
        <details className="text-sm">
          <summary className="w-fit cursor-pointer py-1 font-medium text-muted-foreground hover:text-foreground">
            Reading notes
          </summary>
          <ul className="mt-1 max-w-prose list-disc space-y-1 pl-5 text-muted-foreground">
            {config.pageNotes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      </Card>

      {kind === 'qbs' ? (
        <Card className="px-4 py-3">
          <CardContent className="flex flex-col gap-1 px-0">
            <div className="flex items-center gap-2">
              <Switch
                id="show-unrated"
                checked={state.showUnratedRows}
                onCheckedChange={(checked) => update({ showUnratedRows: checked })}
              />
              <Label htmlFor="show-unrated">Show QBs below the qualifier</Label>
            </div>
            <p className="text-sm text-muted-foreground">
              The table ranks quarterbacks with at least 14 pass attempts per game his team has
              played{gamesSoFar !== null ? ' so far, so teams that have had a bye need fewer' : ''}.
              Switch on to list the passers below that mark too; they have no rank range. Each
              quarterback&apos;s number is the Qualifier Att column in Raw Total Stats.
            </p>
          </CardContent>
        </Card>
      ) : null}

      <WpFilterPanel kind={kind} season={season} />

      <ComparisonPanel
        compareColumns={compareColumns}
        compareIds={compareIds}
        config={config}
        season={season}
        table={displayTable}
        onRemove={(entityId) => update({ compareIds: toggleCompareId(compareIds, entityId) })}
      />

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
        selectedColumns={seasonView.selectedColumns}
        sorting={state.sorting}
        table={displayTable}
      />

      {rankRangesQuery.isError && !isMissingRankRanges(rankRangesQuery.error) ? (
        <ErrorState error={rankRangesQuery.error} title="Could not load the rank ranges" />
      ) : null}
      {rankRanges && rankRanges.length > 0 ? (
        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">Rank ranges</CardTitle>
            <CardDescription>
              Where each {kind === 'teams' ? 'team' : 'qualifying QB'} ranks when the {season} games are
              redrawn at random, with repeats, and the ratings are refit on every redraw. The spread
              shows how much a rank depends on which games happened to be played, not whether the
              model is right. Seasons in progress show very wide ranges.
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
