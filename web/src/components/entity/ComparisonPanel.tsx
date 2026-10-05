import { ExternalLink, X } from 'lucide-react'
import { Link } from 'react-router'

import type { EntityConfig, RowValue, TablePayload } from '@/api/types'
import { useTheme } from '@/app/ThemeProvider'
import { MetricLabel } from '@/components/common/MetricLabel'
import { HeadToHeadSentence } from '@/components/entity/HeadToHeadCard'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { getEntityLabel } from '@/domain/entityConfig'
import { formatFixed } from '@/domain/format'
import { buildColumnDecimals, buildColumnStats, getHeatCellStyle } from '@/domain/tableState'

interface ComparisonPanelProps {
  compareColumns: string[]
  compareIds: string[]
  config: EntityConfig
  season: number
  table: TablePayload
  onRemove: (entityId: string) => void
}

/** Side-by-side rows for up to four selected teams or QBs, heat-mapped against each other. */
export function ComparisonPanel({
  compareColumns,
  compareIds,
  config,
  season,
  table,
  onRemove,
}: ComparisonPanelProps) {
  const { resolved: theme, palette } = useTheme()
  const compareRows = compareIds
    .map((entityId) => table.rows.find((row) => String(row[config.identityKey] ?? '') === entityId))
    .filter((row): row is Record<string, RowValue> => row !== undefined)
  if (compareRows.length === 0) return null
  const compareStats = buildColumnStats(compareRows, compareColumns)
  // The same decimals as the season table below, from every row of the season.
  const compareDecimals = buildColumnDecimals(table.rows, compareColumns)
  // With exactly two rows, the first is compared head to head with the second.
  const [first, second] = compareRows
  const headToHead =
    compareRows.length === 2 && first && second
      ? {
          subject: String(first[config.identityKey] ?? ''),
          other: String(second[config.identityKey] ?? ''),
          labels: { subject: getEntityLabel(config.kind, first), other: getEntityLabel(config.kind, second) },
        }
      : null

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="text-base">{config.singularLabel} comparison</CardTitle>
        <CardDescription>
          {compareRows.length} selected. The page URL keeps the selection, so the comparison can be
          shared or bookmarked.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-2">
          {compareRows.map((row) => {
            const entityId = String(row[config.identityKey] ?? '')
            const label = getEntityLabel(config.kind, row)
            return (
              <Badge key={entityId} variant="outline" className="gap-1 py-1 pr-1 pl-2 text-sm">
                <Link
                  to={`/${config.kind}/${encodeURIComponent(entityId)}?season=${season}`}
                  className="inline-flex items-center gap-1 hover:underline"
                >
                  {label}
                  <ExternalLink className="size-3" />
                </Link>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Remove ${label} from comparison`}
                  onClick={() => onRemove(entityId)}
                >
                  <X />
                </Button>
              </Badge>
            )
          })}
        </div>
        {headToHead ? (
          <HeadToHeadSentence
            kind={config.kind}
            season={season}
            entityId={headToHead.subject}
            otherId={headToHead.other}
            labels={headToHead.labels}
          />
        ) : null}
        <Table className="tabular">
          <TableHeader>
            <TableRow>
              <TableHead>{config.singularLabel}</TableHead>
              {compareColumns.map((column) => (
                <TableHead key={column}>
                  <MetricLabel column={column} />
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {compareRows.map((row) => (
              <TableRow key={String(row[config.identityKey] ?? '')}>
                <TableCell className="font-medium">{getEntityLabel(config.kind, row)}</TableCell>
                {compareColumns.map((column) => (
                  <TableCell
                    key={column}
                    style={getHeatCellStyle(column, row[column] ?? null, compareStats, theme, palette)}
                  >
                    {formatFixed(row[column] ?? null, compareDecimals[column] ?? null)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
