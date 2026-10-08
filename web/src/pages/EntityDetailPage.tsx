import { ArrowLeft } from 'lucide-react'
import { useMemo } from 'react'
import { Link, Navigate, useParams } from 'react-router'

import { useEntityGameLogs, useRankRanges, useRatingHistory } from '@/api/queries'
import type { EntityKind, SeasonDataset } from '@/api/types'
import { useEntityPageState } from '@/app/EntityViewStateProvider'
import { useTeamPageColors } from '@/app/useTeamPageColors'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { TeamChip } from '@/components/common/TeamChip'
import { StatTile } from '@/components/common/StatTile'
import { GameLogTable } from '@/components/entity/GameLogTable'
import { HeadToHeadCard } from '@/components/entity/HeadToHeadCard'
import { MetricSections } from '@/components/entity/MetricSections'
import { OpponentBreakdownTable } from '@/components/entity/OpponentBreakdownTable'
import { RankHistogram } from '@/components/entity/RankHistogram'
import { RankHistoryCard } from '@/components/entity/RankHistoryCard'
import { ViewControls } from '@/components/entity/ViewControls'
import { UnitRankRangeTable } from '@/components/entity/UnitRankRanges'
import { WeeklyTrendChart } from '@/components/entity/WeeklyTrendChart'
import { WpFilterPanel } from '@/components/entity/WpFilterPanel'
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
import { countLabel, humanizeGroup } from '@/domain/format'
import { getGroupDescription } from '@/domain/metricMetadata'
import { canResetPageView, toggleSubcategoryPatch } from '@/domain/pageViewState'
import {
  belowQualifierText,
  isMissingRankRanges,
  parseRankRanges,
  opponentRankRanges,
  parseUnitRankRanges,
  rankChanceText,
  rankRangeHeadline,
} from '@/domain/rankRanges'
import { buildRatingHistoryChart, isMissingRatingHistory } from '@/domain/ratingHistory'
import {
  buildGameLogColumnSelection,
  buildSeasonViewTable,
  deriveLegacyDetailSurfaceId,
} from '@/domain/viewModel'
import { buildColumnDecimals } from '@/domain/tableState'

/** One team's or QB's season: current-view values, rating by week, weekly log, and opponents. */
export function EntityDetailPage({ kind, dataset }: { kind: EntityKind; dataset: SeasonDataset }) {
  const config = getEntityConfig(kind)
  const state = useEntityPageState(kind)
  const { viewState, update } = state
  const season = dataset.season
  const entityId = decodeURIComponent(useParams().entityId ?? '')
  const seasonView = useMemo(() => buildSeasonViewTable(kind, dataset[kind], viewState), [dataset, kind, viewState])
  const row = getEntityRow(seasonView.table, kind, entityId)
  // A QB page shows his team's palette (team pages set theirs from the route, `router.tsx`).
  const qbTeam = kind === 'qbs' && row ? String(row.team ?? '') : ''
  useTeamPageColors(qbTeam === '' ? null : qbTeam)
  const gameLogsQuery = useEntityGameLogs(kind, season, row ? entityId : '')
  const ratingHistoryQuery = useRatingHistory(kind, season, row ? entityId : '')
  const rankRangesQuery = useRankRanges(kind, season)
  const rankRange = useMemo(
    () =>
      rankRangesQuery.data
        ? parseRankRanges(kind, rankRangesQuery.data).find((range) => range.id === entityId)
        : undefined,
    [entityId, kind, rankRangesQuery.data],
  )
  const teamRankRangesQuery = useRankRanges('teams', season)
  const opponentRanges = useMemo(
    () =>
      teamRankRangesQuery.data
        ? { ranges: opponentRankRanges(kind, teamRankRangesQuery.data), count: teamRankRangesQuery.data.rows.length }
        : undefined,
    [kind, teamRankRangesQuery.data],
  )
  const unitRankRanges = useMemo(
    () => (kind === 'teams' && rankRangesQuery.data ? parseUnitRankRanges(rankRangesQuery.data, entityId) : []),
    [entityId, kind, rankRangesQuery.data],
  )

  const enrichedGameLogs = useMemo(
    () => (gameLogsQuery.data ? enrichGameLogsWithOpponentRatings(gameLogsQuery.data, dataset.teams) : null),
    [dataset.teams, gameLogsQuery.data],
  )
  const ratingHistoryChart = useMemo(
    () => (ratingHistoryQuery.data ? buildRatingHistoryChart(kind, ratingHistoryQuery.data) : null),
    [kind, ratingHistoryQuery.data],
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

  if (!row) return <Navigate to={`/${kind}?season=${season}`} replace state={{ notFound: entityId }} />

  const rawLabel = String(row[config.labelKey] ?? row[config.identityKey] ?? '')
  const label = kind === 'teams' ? getFullTeamName(rawLabel) : rawLabel
  const subtitle =
    kind === 'teams'
      ? `Team detail · ${season} regular season`
      : `Quarterback · ${getFullTeamName(String(row.team ?? ''))} · ${season} regular season`
  const metricColumns = seasonView.table.visible_columns.filter((column) => !config.identityColumns.includes(column))

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={label}
        titleMark={<TeamChip team={kind === 'teams' ? rawLabel : String(row.team ?? '')} className="size-4" />}
        description={subtitle}
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
          <Badge variant="secondary">{countLabel(metricColumns.length, 'column')}</Badge>
        </CardHeader>
        <CardContent>
          <MetricSections
            kind={kind}
            row={row}
            columns={metricColumns}
            isRatingsView={viewState.primaryView === 'ratings'}
            decimals={buildColumnDecimals(seasonView.table.rows, metricColumns)}
          />
        </CardContent>
      </Card>

      {rankRangesQuery.isError && !isMissingRankRanges(rankRangesQuery.error) ? (
        <ErrorState error={rankRangesQuery.error} title="Could not load the rank ranges" />
      ) : null}
      {rankRangesQuery.data && !rankRange && belowQualifierText(row) ? (
        <Card className="gap-2 px-4 py-3">
          <div className="text-sm font-medium">Rank range</div>
          <p className="text-sm text-muted-foreground">{belowQualifierText(row)}</p>
        </Card>
      ) : null}
      {rankRange ? (
        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">Rank range</CardTitle>
            <CardDescription>
              The rank when the {season} games are redrawn at random, with repeats, and the ratings are
              refit on every redraw. It shows how much the rank depends on which games happened to be
              played, not whether the model is right.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div>
              <p className="text-lg font-semibold tabular">{rankRangeHeadline(rankRange)}</p>
              <p className="text-sm text-muted-foreground">{rankChanceText(kind, rankRange)}</p>
            </div>
            <RankHistogram range={rankRange} />
            {unitRankRanges.length > 0 ? (
              <UnitRankRangeTable ranges={unitRankRanges} count={rankRangesQuery.data?.rows.length ?? 0} />
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      <HeadToHeadCard kind={kind} season={season} entityId={entityId} rows={dataset[kind].rows} />

      <WpFilterPanel kind={kind} season={season} entityId={entityId} />

      {ratingHistoryQuery.isError && !isMissingRatingHistory(ratingHistoryQuery.error) ? (
        <ErrorState error={ratingHistoryQuery.error} title="Could not load the rating history" />
      ) : null}
      {ratingHistoryChart ? (
        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">Rating by week</CardTitle>
            <CardDescription>
              Each point is the rating fit on the {season} games through that week, so early weeks sit
              near average and spread out as games accumulate. The last point is the rating above.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <WeeklyTrendChart
              rows={ratingHistoryChart.rows}
              columns={ratingHistoryChart.columns}
              reference={ratingHistoryChart.reference}
            />
          </CardContent>
        </Card>
      ) : null}

      <RankHistoryCard kind={kind} season={season} entityId={entityId} count={rankRangesQuery.data?.rows.length} />

      <Card className="gap-4">
        <CardHeader className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base">Game by game</CardTitle>
            <CardDescription>
              Every {season} game, with result context first and the current view&apos;s columns after
              it. Opponent rating columns describe that opponent&apos;s full season, not a
              single-game grade, and so does the rank range beside each opponent: the middle 50% of
              its {kind === 'qbs' ? 'defense rank' : 'rank'} when the season is redrawn at random.
            </CardDescription>
          </div>
          {gameLogsQuery.data && gameLogSelection ? (
            <div className="flex gap-1.5">
              <Badge variant="secondary">{countLabel(gameLogsQuery.data.rows.length, 'game')}</Badge>
              <Badge variant="secondary">{countLabel(gameLogSelection.columns.length, 'column')}</Badge>
            </div>
          ) : null}
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {gameLogsQuery.isLoading ? <Skeleton className="h-64 bg-muted" /> : null}
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
              <GameLogTable rows={enrichedGameLogs.rows} columns={gameLogSelection.columns} opponents={opponentRanges} />
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
              <Badge variant="secondary">{countLabel(opponentBreakdown.rows.length, 'opponent')}</Badge>
              <Badge variant="secondary">{countLabel(opponentBreakdown.columns.length, 'column')}</Badge>
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
