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
import { GameLogTable } from '@/components/entity/GameLogTable'
import { HeadToHeadCard } from '@/components/entity/HeadToHeadCard'
import { MetricSections } from '@/components/entity/MetricSections'
import { OpponentBreakdownTable } from '@/components/entity/OpponentBreakdownTable'
import { RankHistogram } from '@/components/entity/RankHistogram'
import { RankHistoryCard } from '@/components/entity/RankHistoryCard'
import { RatingSummary } from '@/components/entity/RatingSummary'
import { ViewControls } from '@/components/entity/ViewControls'
import { UnitRankRangeTable } from '@/components/entity/UnitRankRanges'
import { WeeklyTrendChart } from '@/components/entity/WeeklyTrendChart'
import { WpExploration } from '@/components/entity/WpExploration'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { buildOpponentBreakdown, enrichGameLogsWithOpponentRatings } from '@/domain/detailAnalytics'
import { getEntityConfig, getEntityRow, getFullTeamName } from '@/domain/entityConfig'
import { humanizeGroup } from '@/domain/format'
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
  getEffectiveStatView,
  PRIMARY_VIEWS,
} from '@/domain/viewModel'
import { buildColumnDecimals } from '@/domain/tableState'

/** The game-by-game chart opens on the per-game form of what the rating measures. */
const TREND_PREFERRED: Record<EntityKind, readonly string[]> = {
  teams: ['epa_margin_per_play', 'point_margin'],
  qbs: ['qb_epa_per_dropback'],
}
// The stats section's views: the rating summary at the top of the page stands in for Ratings.
const STAT_VIEWS = PRIMARY_VIEWS.filter((view) => view !== 'ratings')

/** One team's or QB's season: current-view values, rating by week, weekly log, and opponents. */
export function EntityDetailPage({ kind, dataset }: { kind: EntityKind; dataset: SeasonDataset }) {
  const config = getEntityConfig(kind)
  const state = useEntityPageState(kind)
  const { viewState, update } = state
  const season = dataset.season
  const entityId = decodeURIComponent(useParams().entityId ?? '')
  const seasonView = useMemo(() => buildSeasonViewTable(kind, dataset[kind], viewState), [dataset, kind, viewState])
  const row = getEntityRow(seasonView.table, kind, entityId)
  // The season row as the API serves it, per game in every view: the baseline the game-by-game
  // tiles and the unique-opponent "vs Season" column compare per-game values with. The view's row
  // above turns counts into season totals in Raw Total Stats.
  const seasonRow = useMemo(() => getEntityRow(dataset[kind], kind, entityId), [dataset, entityId, kind])
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
  // The stats section shows a stat view; a Ratings choice carried over from the index reads as
  // Per-Game Rates there, as the game log already does.
  const statState = useMemo(
    () => ({ ...viewState, primaryView: getEffectiveStatView(viewState.primaryView) }),
    [viewState],
  )
  const statView = useMemo(() => buildSeasonViewTable(kind, dataset[kind], statState), [dataset, kind, statState])
  const gameLogSelection = useMemo(
    () => (enrichedGameLogs ? buildGameLogColumnSelection(kind, enrichedGameLogs, viewState) : null),
    [enrichedGameLogs, kind, viewState],
  )
  const opponentBreakdown = useMemo(
    () =>
      enrichedGameLogs && seasonRow
        ? buildOpponentBreakdown(kind, seasonRow, enrichedGameLogs, deriveLegacyDetailSurfaceId(kind, viewState))
        : null,
    [enrichedGameLogs, kind, seasonRow, viewState],
  )

  if (!row) return <Navigate to={`/${kind}?season=${season}`} replace state={{ notFound: entityId }} />

  const rawLabel = String(row[config.labelKey] ?? row[config.identityKey] ?? '')
  const label = kind === 'teams' ? getFullTeamName(rawLabel) : rawLabel
  const subtitle =
    kind === 'teams'
      ? `Team detail · ${season} regular season`
      : `Quarterback · ${getFullTeamName(String(row.team ?? ''))} · ${season} regular season`
  const statRow = getEntityRow(statView.table, kind, entityId) ?? row
  const statColumns = statView.table.visible_columns.filter((column) => !config.identityColumns.includes(column))
  const hasRating = typeof seasonRow?.[config.defaultSortColumn] === 'number'

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

      <Card className="gap-4">
        <CardContent>
          <RatingSummary kind={kind} row={seasonRow ?? row} rows={dataset[kind].rows} />
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

      <div className="flex flex-col gap-5">
        {/* Sticky only while this section scrolls by: the tabs drive these cards alone. */}
        <Card className="sticky top-14 z-10 gap-0 px-4 py-3 shadow-sm">
          <ViewControls
            canReset={canResetPageView(kind, state)}
            kind={kind}
            onReset={state.reset}
            onSelectTeamCategory={(teamCategory) => update({ teamCategory })}
            onSelectView={(primaryView) => update({ primaryView })}
            onToggleSubcategory={(subcategory) => update(toggleSubcategoryPatch(kind, viewState, subcategory))}
            state={statState}
            views={STAT_VIEWS}
          />
        </Card>

        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">{humanizeGroup(statState.primaryView)}</CardTitle>
            <CardDescription>{getGroupDescription(kind, statState.primaryView)}</CardDescription>
          </CardHeader>
          <CardContent>
            <MetricSections
              kind={kind}
              row={statRow}
              columns={statColumns}
              isRatingsView={false}
              decimals={buildColumnDecimals(statView.table.rows, statColumns)}
            />
          </CardContent>
        </Card>

        <Card className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">Game by game</CardTitle>
            <CardDescription>
              Every {season} game, with result context first and the current view&apos;s columns after
              it. Opponent rating columns describe that opponent&apos;s full season, not a
              single-game grade, and so does the rank range beside each opponent: the middle 50% of
              its {kind === 'qbs' ? 'defense rank' : 'rank'} when the season is redrawn at random.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {gameLogsQuery.isLoading ? <Skeleton className="h-64 bg-muted" /> : null}
            {gameLogsQuery.isError ? <ErrorState error={gameLogsQuery.error} title="Could not load the weekly log" /> : null}
            {enrichedGameLogs && gameLogSelection ? (
              <>
                <WeeklyTrendChart
                  rows={enrichedGameLogs.rows}
                  columns={gameLogSelection.metricColumns}
                  preferred={TREND_PREFERRED[kind]}
                />
                <GameLogTable rows={enrichedGameLogs.rows} columns={gameLogSelection.columns} opponents={opponentRanges} />
              </>
            ) : null}
          </CardContent>
        </Card>

        {opponentBreakdown ? (
          <Card className="gap-4">
            <CardHeader>
              <CardTitle className="text-base">Unique opponents</CardTitle>
              <CardDescription>{opponentBreakdown.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <OpponentBreakdownTable breakdown={opponentBreakdown} />
            </CardContent>
          </Card>
        ) : null}
      </div>

      {hasRating ? <WpExploration kind={kind} season={season} entityId={entityId} /> : null}
    </div>
  )
}
