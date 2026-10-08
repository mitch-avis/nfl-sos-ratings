import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { GitCompareArrows, Search } from 'lucide-react'
import { useMemo, type CSSProperties, type ReactNode } from 'react'
import { Link } from 'react-router'

import type { EntityConfig, EntityKind, RowValue, TablePayload } from '@/api/types'
import { useTheme } from '@/app/ThemeProvider'
import { CsvExportButton } from '@/components/common/CsvExportButton'
import { Hint } from '@/components/common/Hint'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { MetricLabel } from '@/components/common/MetricLabel'
import { SortableHeader } from '@/components/common/SortableHeader'
import { TeamChip } from '@/components/common/TeamChip'
import { Card, CardContent } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { csvFileName, toCsv } from '@/domain/csv'
import { formatValue } from '@/domain/format'
import { getMetricMetadata, getMetricTooltip } from '@/domain/metricMetadata'
import {
  belowQualifierDetail,
  belowQualifierText,
  ordinal,
  rankChanceText,
  rankRangeHeadline,
  rankRangeSummary,
  type RankRange,
} from '@/domain/rankRanges'
import {
  buildColumnDecimals,
  buildColumnStats,
  buildColumnWidths,
  formatColumnValue,
  getHeatCellStyle,
  sanitizeSorting,
} from '@/domain/tableState'
import { useIsMobile } from '@/hooks/use-mobile'
import { cn } from '@/utils/cn'

import { RankIntervalTrack } from './RankInterval'

type Row = Record<string, RowValue>

interface EntityTableProps {
  compareIds: string[]
  config: EntityConfig
  controls: ReactNode
  onQueryChange: (query: string) => void
  onSortingChange: (sorting: SortingState) => void
  onToggleCompare: (entityId: string) => void
  query: string
  /** Bootstrap rank ranges; when given, a "Rank range" column follows the headline rating. */
  rankRanges?: RankRange[]
  season: number
  selectedColumns: string[]
  sorting: SortingState
  table: TablePayload
  /** Page-specific controls for the toolbar beside the search box. */
  toolbar?: ReactNode
}

const CONTROL_COLUMNS = ['compare', 'rank']
const RANK_RANGE_COLUMN = 'rank_range'
const PHONE_PINNED_MAX_WIDTH = 120
// The rank text plus a mini interval track.
const RANK_RANGE_COLUMN_WIDTH = 168

/** The cell for a QB below the qualifier, who is not ranked: why, with his attempts. */
function BelowQualifierCell({ row }: { row: Row }) {
  const detail = belowQualifierDetail(row) ?? 'too few pass attempts'
  return (
    <Hint content={belowQualifierText(row)}>
      <button
        type="button"
        aria-label={`Below the qualifier: ${detail}`}
        className="cursor-help text-muted-foreground underline decoration-muted-foreground/40 decoration-dotted underline-offset-4"
      >
        Below qualifier
      </button>
    </Hint>
  )
}

function RankRangeCell({
  kind,
  range,
  count,
  row,
}: {
  kind: EntityKind
  range: RankRange | undefined
  count: number
  row: Row
}) {
  const { q250, q750 } = range?.rank ?? { q250: null, q750: null }
  if (!range && kind === 'qbs' && row.qb_is_eligible === false) return <BelowQualifierCell row={row} />
  if (!range || q250 === null || q750 === null) return <span className="text-muted-foreground">-</span>
  return (
    <Hint
      content={
        <>
          <div className="font-medium">
            {range.label}: {rankRangeHeadline(range)}
          </div>
          <div className="text-muted-foreground">{rankChanceText(kind, range)}</div>
        </>
      }
    >
      <button type="button" aria-label={rankRangeSummary(range)} className="flex items-center gap-2 rounded-sm text-left">
        <span className="w-16 text-right">{q250 === q750 ? ordinal(q250) : `${ordinal(q250)}–${ordinal(q750)}`}</span>
        <span className="w-20">
          <RankIntervalTrack range={range} count={count} size="mini" />
        </span>
      </button>
    </Hint>
  )
}

/** The column holding a team abbreviation, which gets the team's color chip. */
const TEAM_COLUMN = 'team'

/**
 * The index table: search, view controls, sortable heat-mapped columns, sticky identity columns,
 * a rank column, and a compare checkbox per row.
 */
export function EntityTable({
  compareIds,
  config,
  controls,
  onQueryChange,
  onSortingChange,
  onToggleCompare,
  query,
  rankRanges,
  season,
  selectedColumns,
  sorting,
  table,
  toolbar,
}: EntityTableProps) {
  const { resolved: theme, activePalette: palette } = useTheme()
  const basePath = `/${config.kind}`
  const availableColumnIds = useMemo(() => [...CONTROL_COLUMNS, ...selectedColumns], [selectedColumns])
  const fallbackSorting = useMemo<SortingState>(
    () =>
      availableColumnIds.includes(config.defaultSortColumn)
        ? [
            {
              id: config.defaultSortColumn,
              desc: getMetricMetadata(config.defaultSortColumn).polarity !== 'lower',
            },
          ]
        : [],
    [availableColumnIds, config.defaultSortColumn],
  )
  const safeSorting = useMemo(
    () => sanitizeSorting(sorting, availableColumnIds, fallbackSorting),
    [availableColumnIds, fallbackSorting, sorting],
  )

  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return table.rows
    return table.rows.filter((row) =>
      selectedColumns.some((column) => String(row[column] ?? '').toLowerCase().includes(normalized)),
    )
  }, [query, selectedColumns, table.rows])

  const columnStats = useMemo(() => buildColumnStats(filteredRows, selectedColumns), [filteredRows, selectedColumns])
  // Decimals come from the whole season, so filtering or searching never changes them.
  const columnDecimals = useMemo(() => buildColumnDecimals(table.rows, selectedColumns), [selectedColumns, table.rows])
  const columnWidths = useMemo(
    () => buildColumnWidths(table.rows, selectedColumns, config.identityColumns),
    [config.identityColumns, selectedColumns, table.rows],
  )
  const isPhone = useIsMobile()
  // On a phone only the name stays pinned, at a capped width, so the stats keep most of the screen.
  const stickyOffsets = useMemo(() => {
    const offsets: Record<string, number> = {}
    let left = 0
    for (const id of isPhone ? [config.labelKey] : [...CONTROL_COLUMNS, ...config.identityColumns]) {
      offsets[id] = left
      left += columnWidths[id] ?? 120
    }
    return offsets
  }, [columnWidths, config.identityColumns, config.labelKey, isPhone])

  const rankRangeColumn = useMemo<ColumnDef<Row> | null>(() => {
    if (!rankRanges || !selectedColumns.includes(config.defaultSortColumn)) return null
    const byId = new Map(rankRanges.map((range) => [range.id, range]))
    return {
      id: RANK_RANGE_COLUMN,
      header: () => (
        <span className="inline-flex items-center gap-1">
          Rank range
          <InfoTooltip
            label="About the rank range"
            content="The middle 50% of ranks across resampled seasons (the season's games redrawn at random). Thick bar: middle 50%; thin bar: middle 95%; dot: median; diamond: the published rank when it differs. Rank 1 is at the left."
          />
        </span>
      ),
      size: RANK_RANGE_COLUMN_WIDTH,
      enableSorting: false,
      cell: ({ row }) => (
        <RankRangeCell
          kind={config.kind}
          range={byId.get(String(row.original[config.identityKey] ?? ''))}
          count={rankRanges.length}
          row={row.original}
        />
      ),
    }
  }, [config.defaultSortColumn, config.identityKey, config.kind, rankRanges, selectedColumns])

  const columns = useMemo<ColumnDef<Row>[]>(
    () => [
      {
        id: 'compare',
        header: () => (
          <Hint content="Tick rows to compare them side by side, below the table.">
            <button type="button" aria-label="Compare" className="inline-flex size-6 items-center justify-center rounded-sm">
              <GitCompareArrows className="size-4" aria-hidden />
            </button>
          </Hint>
        ),
        size: columnWidths.compare ?? 44,
        enableSorting: false,
        cell: ({ row }) => {
          const entityId = String(row.original[config.identityKey] ?? '')
          const label = String(row.original[config.labelKey] ?? entityId)
          return (
            <Checkbox
              aria-label={`Compare ${label}`}
              checked={compareIds.includes(entityId)}
              onCheckedChange={() => onToggleCompare(entityId)}
            />
          )
        },
      },
      {
        id: 'rank',
        header: ({ table: reactTable }) => {
          const sortedId = reactTable.getState().sorting[0]?.id
          const sortedLabel = sortedId ? getMetricMetadata(sortedId).label : null
          const headline = getMetricMetadata(config.defaultSortColumn).label
          return (
            <span className="inline-flex items-center gap-1.5">
              Rank
              <InfoTooltip
                label="About the rank"
                content={`Each row's position in the current sort${sortedLabel ? ` (${sortedLabel})` : ''}. Sort by ${headline} for the published ranking; the Rank range column is always about ${headline}.`}
              />
            </span>
          )
        },
        size: columnWidths.rank ?? 76,
        enableSorting: false,
        cell: () => null,
      },
      ...selectedColumns.flatMap<ColumnDef<Row>>((column) => {
        const metricColumn: ColumnDef<Row> = {
          id: column,
          accessorFn: (row) => row[column],
          header: () => <MetricLabel column={column} />,
          size: columnWidths[column] ?? 128,
          sortDescFirst: (() => {
            const sample = filteredRows.find((row) => row[column] !== null)?.[column]
            if (typeof sample === 'string') return false
            return getMetricMetadata(column).polarity !== 'lower'
          })(),
          cell: ({ getValue, row }) => {
            const value = getValue() as RowValue
            const chip = column === TEAM_COLUMN && typeof value === 'string' ? <TeamChip team={value} /> : null
            if (column !== config.labelKey) {
              const text = formatColumnValue(column, value, columnDecimals[column] ?? null)
              return chip ? (
                <span className="inline-flex items-center gap-1.5">
                  {chip}
                  {text}
                </span>
              ) : (
                text
              )
            }
            const entityId = String(row.original[config.identityKey] ?? '')
            return (
              <Link
                className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
                to={`${basePath}/${encodeURIComponent(entityId)}?season=${season}`}
              >
                {chip}
                {formatValue(value)}
              </Link>
            )
          },
        }
        return column === config.defaultSortColumn && rankRangeColumn ? [metricColumn, rankRangeColumn] : [metricColumn]
      }),
    ],
    [
      basePath,
      columnDecimals,
      columnWidths,
      compareIds,
      config,
      filteredRows,
      onToggleCompare,
      rankRangeColumn,
      season,
      selectedColumns,
    ],
  )

  const reactTable = useReactTable({
    data: filteredRows,
    columns,
    state: { sorting: safeSorting },
    onSortingChange: (updater) => onSortingChange(typeof updater === 'function' ? updater(safeSorting) : updater),
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  const cellStyle = (columnId: string, width: number): CSSProperties => {
    const left = stickyOffsets[columnId]
    if (left === undefined) return { minWidth: width, width }
    const pinnedWidth = isPhone ? Math.min(width, PHONE_PINNED_MAX_WIDTH) : width
    return { left, minWidth: pinnedWidth, width: pinnedWidth, maxWidth: pinnedWidth }
  }

  return (
    <Card className="gap-4">
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
          <div className="relative w-full max-w-sm">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="search"
              aria-label="Search visible columns"
              placeholder="Search visible columns"
              className="pl-8"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
            />
          </div>
          {toolbar}
          <div className="ml-auto">
            <CsvExportButton
              fileName={csvFileName(config.kind, season)}
              build={() => toCsv(selectedColumns, reactTable.getRowModel().rows.map((row) => row.original))}
            />
          </div>
        </div>
        {controls}
        {/* The box fills the screen below the app header, so once the page scrolls to it, it reads as
            one full-height sheet with a sticky header row rather than a small window inside the page. */}
        <div className="max-h-[calc(100dvh-4.5rem)] overflow-auto rounded-md border">
          <table aria-label={config.title} className="w-max min-w-full text-sm tabular">
            <thead className="sticky top-0 z-20 bg-muted">
              {reactTable.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    const sticky = stickyOffsets[header.column.id] !== undefined
                    const sorted = header.column.getIsSorted()
                    return (
                      <th
                        key={header.id}
                        scope="col"
                        aria-sort={sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : undefined}
                        style={cellStyle(header.column.id, header.getSize())}
                        className={cn(
                          'h-10 border-b bg-muted px-2 text-left align-middle font-medium whitespace-nowrap text-muted-foreground',
                          sticky && 'sticky z-30',
                        )}
                      >
                        {header.column.getCanSort() ? (
                          <SortableHeader
                            label={getMetricMetadata(header.column.id).label}
                            hint={getMetricTooltip(header.column.id)}
                            direction={sorted}
                            onSort={(event) => header.column.getToggleSortingHandler()?.(event)}
                          />
                        ) : (
                          flexRender(header.column.columnDef.header, header.getContext())
                        )}
                      </th>
                    )
                  })}
                </tr>
              ))}
            </thead>
            <tbody>
              {reactTable.getRowModel().rows.map((row, rowIndex) => (
                <tr key={row.id} className="border-b last:border-0 hover:bg-muted/40">
                  {row.getVisibleCells().map((cell) => {
                    const columnId = cell.column.id
                    const sticky = stickyOffsets[columnId] !== undefined
                    const heat =
                      CONTROL_COLUMNS.includes(columnId) || columnId === RANK_RANGE_COLUMN
                        ? undefined
                      : getHeatCellStyle(columnId, (cell.getValue() as RowValue) ?? null, columnStats, theme, palette)
                    return (
                      <td
                        key={cell.id}
                        style={{ ...cellStyle(columnId, cell.column.getSize()), ...heat }}
                        className={cn('px-2 py-1.5 whitespace-nowrap', sticky && 'sticky z-10 truncate bg-card')}
                      >
                        {columnId === 'rank' ? rowIndex + 1 : flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
