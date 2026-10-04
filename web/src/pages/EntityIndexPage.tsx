import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

import type { EntityKind, SeasonDataset } from '@/api/types'
import { useEntityPageState } from '@/app/EntityViewStateProvider'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { ComparisonPanel } from '@/components/entity/ComparisonPanel'
import { EntityTable } from '@/components/entity/EntityTable'
import { ViewControls } from '@/components/entity/ViewControls'
import { Card, CardContent } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { getEntityConfig } from '@/domain/entityConfig'
import { humanizeGroup } from '@/domain/format'
import {
  canResetPageView,
  reconcileCompareIds,
  toggleCompareId,
  toggleSubcategoryPatch,
} from '@/domain/pageViewState'
import { getQuarterbackQualifierAttempts, getRegularSeasonGameCount } from '@/domain/seasonRules'
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
  useCompareQuerySync(kind, dataset, compareIds, setCompareIds)

  const seasonView = useMemo(() => buildSeasonViewTable(kind, table, state.viewState), [kind, state.viewState, table])
  const displayTable = useMemo(() => {
    if (kind !== 'qbs' || state.showUnratedRows) return seasonView.table
    return {
      ...seasonView.table,
      rows: seasonView.table.rows.filter((row) => row.adj_qb_epa_per_dropback != null),
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
  const enabledSubcategories = Object.entries(viewState.activeSubcategories)
    .filter(([, enabled]) => enabled)
    .map(([label]) => label)
  const selectedSlice =
    viewState.primaryView === 'ratings'
      ? 'Rating columns'
      : [kind === 'teams' ? viewState.teamCategory : null, enabledSubcategories.join(', ')]
          .filter(Boolean)
          .join(': ')
  const totalCount = seasonView.table.rows.length
  const displayCount = displayTable.rows.length
  const season = dataset.season

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={`${config.title} · ${season}`}
        description="Every rating and stat compares each subject with the opponents it actually faced. Sort, filter, and switch views; open a row for its game-by-game detail."
      />

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,2fr)]">
        <Card className="gap-2 px-4 py-3">
          <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Use first</div>
          <div className="font-semibold">{config.primaryRankingLabel}</div>
          <p className="text-sm text-muted-foreground">{config.primaryRankingDescription}</p>
        </Card>
        <Card className="gap-2 px-4 py-3">
          <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Reading notes</div>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {config.pageNotes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Card>
      </div>

      {kind === 'qbs' ? (
        <Card className="px-4 py-3">
          <CardContent className="flex flex-col gap-1 px-0">
            <div className="flex items-center gap-2">
              <Switch
                id="show-unrated"
                checked={state.showUnratedRows}
                onCheckedChange={(checked) => update({ showUnratedRows: checked })}
              />
              <Label htmlFor="show-unrated">Show unrated or empty QB rows</Label>
            </div>
            <p className="text-sm text-muted-foreground">
              Includes quarterbacks who played at least one offensive snap but finished below the
              season rating threshold of {getQuarterbackQualifierAttempts(season)} pass attempts (14
              per team game in this {getRegularSeasonGameCount(season)}-game season).
            </p>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-3">
        <StatTile
          label="Rows shown"
          value={displayCount}
          footnote={
            displayCount === totalCount
              ? `${config.singularLabel} rows available for this season.`
              : `${displayCount} of ${totalCount} ${config.singularLabel} rows.`
          }
        />
        <StatTile label="Current view" value={humanizeGroup(viewState.primaryView)} footnote="One stat view at a time." />
        <StatTile label="Metrics in view" value={seasonView.metricColumns.length} footnote={selectedSlice} />
      </div>

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
        onToggleCompare={(entityId) => update({ compareIds: toggleCompareId(compareIds, entityId) })}
        query={state.query}
        season={season}
        selectedColumns={seasonView.selectedColumns}
        sorting={state.sorting}
        table={displayTable}
      />
    </div>
  )
}
