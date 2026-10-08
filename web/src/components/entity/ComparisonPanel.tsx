import { ExternalLink, X } from 'lucide-react'
import { useId } from 'react'
import { Link } from 'react-router'

import type { EntityConfig, RowValue, TablePayload } from '@/api/types'
import { useTheme } from '@/app/ThemeProvider'
import { MetricLabel } from '@/components/common/MetricLabel'
import { TeamChip } from '@/components/common/TeamChip'
import { HeadToHeadSentence } from '@/components/entity/HeadToHeadCard'
import { RankIntervalTrack } from '@/components/entity/RankInterval'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { getEntityLabel } from '@/domain/entityConfig'
import { middleRankText, ordinal, type RankRange } from '@/domain/rankRanges'
import { buildColumnDecimals, buildColumnStats, formatColumnValue, getHeatCellStyle } from '@/domain/tableState'

interface ComparisonPanelProps {
  compareColumns: string[]
  compareIds: string[]
  config: EntityConfig
  season: number
  table: TablePayload
  /** The season's rank ranges, when it has them; each compared column shows its own. */
  rankRanges?: RankRange[]
  onRemove: (entityId: string) => void
}

/**
 * Up to four selected teams or QBs side by side: one column each, headed by the name, the published
 * rank and middle-50% rank range with a mini interval, and a remove button; one row per metric,
 * heat-mapped across the compared columns. The metric names stay pinned while the table scrolls
 * sideways, and two compared rows add the head-to-head sentence.
 */
export function ComparisonPanel({
  compareColumns,
  compareIds,
  config,
  season,
  table,
  rankRanges,
  onRemove,
}: ComparisonPanelProps) {
  const { resolved: theme, activePalette: palette } = useTheme()
  const titleId = useId()
  const compareRows = compareIds
    .map((entityId) => table.rows.find((row) => String(row[config.identityKey] ?? '') === entityId))
    .filter((row): row is Record<string, RowValue> => row !== undefined)
  if (compareRows.length === 0) return null
  // Shaded against the whole season, so two picks are not painted as each other's extremes.
  const compareStats = buildColumnStats(table.rows, compareColumns)
  // The same decimals as the season table below, from every row of the season.
  const compareDecimals = buildColumnDecimals(table.rows, compareColumns)
  const rangesById = new Map((rankRanges ?? []).map((range) => [range.id, range]))
  const rangeCount = rankRanges?.length ?? 0
  const entities = compareRows.map((row) => ({
    id: String(row[config.identityKey] ?? ''),
    label: getEntityLabel(config.kind, row),
    row,
  }))
  const [first, second] = entities
  const title = `${config.singularLabel} comparison`

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle id={titleId} className="text-base">
          {title}
        </CardTitle>
        <CardDescription>
          {compareRows.length} selected, side by side. The page URL keeps the selection, so the
          comparison can be shared or bookmarked.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {entities.length === 2 && first && second ? (
          <HeadToHeadSentence
            kind={config.kind}
            season={season}
            entityId={first.id}
            otherId={second.id}
            labels={{ subject: first.label, other: second.label }}
          />
        ) : null}
        <div className="max-h-[70vh] overflow-auto rounded-md border">
          <table aria-labelledby={titleId} className="w-full min-w-max text-sm tabular">
            <thead className="sticky top-0 z-20 bg-muted">
              <tr>
                <th scope="col" className="sticky left-0 z-30 bg-muted px-3 py-2 text-left font-medium text-muted-foreground">
                  Metric
                </th>
                {entities.map((entity) => {
                  const range = rangesById.get(entity.id)
                  return (
                    <th key={entity.id} scope="col" className="min-w-40 px-3 py-2 text-left align-top font-medium">
                      <div className="flex items-center justify-between gap-2">
                        <Link
                          to={`/${config.kind}/${encodeURIComponent(entity.id)}?season=${season}`}
                          className="inline-flex items-center gap-1.5 text-primary hover:underline"
                        >
                          <TeamChip team={config.kind === 'teams' ? entity.id : String(entity.row.team ?? '')} />
                          {entity.label}
                          <ExternalLink className="size-3" />
                        </Link>
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`Remove ${entity.label} from comparison`}
                          onClick={() => onRemove(entity.id)}
                        >
                          <X />
                        </Button>
                      </div>
                      {range ? (
                        <div className="mt-1 flex flex-col gap-1 text-xs font-normal text-muted-foreground">
                          <span>
                            {ordinal(range.publishedRank)} · middle 50%: {middleRankText(range)}
                          </span>
                          <RankIntervalTrack range={range} count={rangeCount} size="mini" showPublished />
                        </div>
                      ) : null}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {compareColumns.map((column) => (
                <tr key={column} className="border-t">
                  <th scope="row" className="sticky left-0 z-10 bg-card px-3 py-1.5 text-left font-medium whitespace-nowrap">
                    <MetricLabel column={column} />
                  </th>
                  {entities.map((entity) => (
                    <td
                      key={entity.id}
                      className="px-3 py-1.5"
                      style={getHeatCellStyle(column, entity.row[column] ?? null, compareStats, theme, palette)}
                    >
                      {formatColumnValue(column, entity.row[column] ?? null, compareDecimals[column] ?? null)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
