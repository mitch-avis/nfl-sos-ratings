import { ArrowLeft } from 'lucide-react'
import { useMemo } from 'react'
import { Link, Navigate, useParams } from 'react-router'

import { useEntityGameLogs } from '@/api/queries'
import type { EntityKind, SeasonDataset } from '@/api/types'
import { useEntityPageState } from '@/app/EntityViewStateProvider'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { GameLogTable } from '@/components/entity/GameLogTable'
import { MetricSections } from '@/components/entity/MetricSections'
import { OpponentBreakdownTable } from '@/components/entity/OpponentBreakdownTable'
import { ViewControls } from '@/components/entity/ViewControls'
import { WeeklyTrendChart } from '@/components/entity/WeeklyTrendChart'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  buildOpponentBreakdown,
  buildWeeklyHighlights,
  enrichGameLogsWithOpponentRatings,
} from '@/domain/detailAnalytics'
import { getEntityConfig, getEntityRow, getFullTeamName } from '@/domain/entityConfig'
import { humanizeGroup } from '@/domain/format'
import { getGroupDescription } from '@/domain/metricMetadata'
import { canResetPageView, toggleSubcategoryPatch } from '@/domain/pageViewState'
import {
  buildGameLogColumnSelection,
  buildSeasonViewTable,
  deriveLegacyDetailSurfaceId,
} from '@/domain/viewModel'

/** One team's or QB's season: current-view values, weekly log, trend, and opponent ledger. */
export function EntityDetailPage({ kind, dataset }: { kind: EntityKind; dataset: SeasonDataset }) {
  const config = getEntityConfig(kind)
  const state = useEntityPageState(kind)
  const { viewState, update } = state
  const season = dataset.season
  const entityId = decodeURIComponent(useParams().entityId ?? '')
  const seasonView = useMemo(() => buildSeasonViewTable(kind, dataset[kind], viewState), [dataset, kind, viewState])
  const row = getEntityRow(seasonView.table, kind, entityId)
  const gameLogsQuery = useEntityGameLogs(kind, season, row ? entityId : '')

  const enrichedGameLogs = useMemo(
    () => (gameLogsQuery.data ? enrichGameLogsWithOpponentRatings(gameLogsQuery.data, dataset.teams) : null),
    [dataset.teams, gameLogsQuery.data],
  )
  const weeklyHighlights = useMemo(
    () => (enrichedGameLogs && row ? buildWeeklyHighlights(kind, row, enrichedGameLogs) : []),
    [enrichedGameLogs, kind, row],
  )
  const gameLogSelection = useMemo(
    () => (enrichedGameLogs ? buildGameLogColumnSelection(kind, enrichedGameLogs, viewState) : null),
    [enrichedGameLogs, kind, viewState],
  )
  const opponentBreakdown = useMemo(
    () =>
      enrichedGameLogs && row
        ? buildOpponentBreakdown(kind, row, enrichedGameLogs, deriveLegacyDetailSurfaceId(kind, viewState))
        : null,
    [enrichedGameLogs, kind, row, viewState],
  )

  if (!row) return <Navigate to={`/${kind}?season=${season}`} replace />

  const rawLabel = String(row[config.labelKey] ?? row[config.identityKey] ?? '')
  const label =
    kind === 'teams' ? getFullTeamName(rawLabel) : `${rawLabel} - ${getFullTeamName(String(row.team ?? ''))}`
  const metricColumns = seasonView.table.visible_columns.filter((column) => !config.identityColumns.includes(column))

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={label}
        description={`${config.singularLabel} detail · ${season} regular season`}
        actions={
          <Button asChild variant="outline" size="sm">
            <Link to={`/${kind}?season=${season}`}>
              <ArrowLeft />
              All {kind === 'teams' ? 'teams' : 'QBs'}
            </Link>
          </Button>
        }
      />

      <Card className="sticky top-14 z-10 gap-0 px-4 py-3 shadow-sm">
        <ViewControls
          canReset={canResetPageView(kind, state)}
          kind={kind}
          onReset={state.reset}
          onSelectTeamCategory={(teamCategory) => update({ teamCategory })}
          onSelectView={(primaryView) => update({ primaryView })}
          onToggleSubcategory={(subcategory) => update(toggleSubcategoryPatch(kind, viewState, subcategory))}
          state={viewState}
        />
      </Card>

      <Card className="gap-4">
        <CardHeader className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base">{humanizeGroup(viewState.primaryView)}</CardTitle>
            <CardDescription>{getGroupDescription(kind, viewState.primaryView)}</CardDescription>
          </div>
          <Badge variant="secondary">{metricColumns.length} columns</Badge>
        </CardHeader>
        <CardContent>
          <MetricSections
            kind={kind}
            row={row}
            columns={metricColumns}
            isRatingsView={viewState.primaryView === 'ratings'}
          />
        </CardContent>
      </Card>

      <Card className="gap-4">
        <CardHeader className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base">Game by game</CardTitle>
            <CardDescription>
              Every {season} game, with result context first and the current view&apos;s columns after
              it. Opponent rating columns describe that opponent&apos;s full season, not a
              single-game grade.
            </CardDescription>
          </div>
          {gameLogsQuery.data && gameLogSelection ? (
            <div className="flex gap-1.5">
              <Badge variant="secondary">{gameLogsQuery.data.rows.length} games</Badge>
              <Badge variant="secondary">{gameLogSelection.columns.length} columns</Badge>
            </div>
          ) : null}
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {gameLogsQuery.isLoading ? <Skeleton className="h-64" /> : null}
          {gameLogsQuery.isError ? <ErrorState error={gameLogsQuery.error} title="Could not load the weekly log" /> : null}
          {enrichedGameLogs && gameLogSelection ? (
            <>
              {weeklyHighlights.length > 0 ? (
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {weeklyHighlights.map((highlight) => (
                    <StatTile
                      key={highlight.eyebrow}
                      label={highlight.eyebrow}
                      value={highlight.value}
                      footnote={
                        <>
                          <span className="font-medium text-foreground">{highlight.title}</span> · {highlight.context}
                        </>
                      }
                    />
                  ))}
                </div>
              ) : null}
              <WeeklyTrendChart rows={enrichedGameLogs.rows} columns={gameLogSelection.metricColumns} />
              <GameLogTable rows={enrichedGameLogs.rows} columns={gameLogSelection.columns} />
            </>
          ) : null}
        </CardContent>
      </Card>

      {opponentBreakdown ? (
        <Card className="gap-4">
          <CardHeader className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <CardTitle className="text-base">Unique opponents</CardTitle>
              <CardDescription>{opponentBreakdown.description}</CardDescription>
            </div>
            <div className="flex gap-1.5">
              <Badge variant="secondary">{opponentBreakdown.rows.length} opponents</Badge>
              <Badge variant="secondary">{opponentBreakdown.columns.length} columns</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <OpponentBreakdownTable breakdown={opponentBreakdown} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}
