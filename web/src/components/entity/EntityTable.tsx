import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ArrowUpDown, Search } from 'lucide-react'
import { useMemo, type CSSProperties, type ReactNode } from 'react'
import { Link } from 'react-router'

import type { EntityConfig, RowValue, TablePayload } from '@/api/types'
import { useTheme } from '@/app/ThemeProvider'
import { MetricLabel } from '@/components/common/MetricLabel'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { formatValue } from '@/domain/format'
import { getMetricMetadata } from '@/domain/metricMetadata'
import { buildColumnStats, buildColumnWidths, getHeatCellStyle, sanitizeSorting } from '@/domain/tableState'
import { cn } from '@/utils/cn'

type Row = Record<string, RowValue>

interface EntityTableProps {
  compareIds: string[]
  config: EntityConfig
  controls: ReactNode
  onQueryChange: (query: string) => void
  onSortingChange: (sorting: SortingState) => void
  onToggleCompare: (entityId: string) => void
  query: string
  season: number
  selectedColumns: string[]
  sorting: SortingState
  table: TablePayload
}

const CONTROL_COLUMNS = ['compare', 'rank']

function SortIcon({ direction }: { direction: false | 'asc' | 'desc' }) {
  if (direction === 'asc') return <ArrowUp className="size-3.5" aria-label="sorted ascending" />
  if (direction === 'desc') return <ArrowDown className="size-3.5" aria-label="sorted descending" />
  return <ArrowUpDown className="size-3.5 opacity-40" aria-hidden />
}

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
  season,
  selectedColumns,
  sorting,
  table,
}: EntityTableProps) {
  const { resolved: theme, palette } = useTheme()
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
  const columnWidths = useMemo(
    () => buildColumnWidths(table.rows, selectedColumns, config.identityColumns),
    [config.identityColumns, selectedColumns, table.rows],
  )
  const stickyOffsets = useMemo(() => {
    const offsets: Record<string, number> = {}
    let left = 0
    for (const id of [...CONTROL_COLUMNS, ...config.identityColumns]) {
      offsets[id] = left
      left += columnWidths[id] ?? 120
    }
    return offsets
  }, [columnWidths, config.identityColumns])

  const columns = useMemo<ColumnDef<Row>[]>(
    () => [
      {
        id: 'compare',
        header: () => 'Compare',
        size: columnWidths.compare ?? 108,
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
      { id: 'rank', header: () => 'Rank', size: columnWidths.rank ?? 76, enableSorting: false, cell: () => null },
      ...selectedColumns.map<ColumnDef<Row>>((column) => ({
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
          if (column !== config.labelKey) return formatValue(value)
          const entityId = String(row.original[config.identityKey] ?? '')
          return (
            <Link className="font-medium text-primary hover:underline" to={`${basePath}/${encodeURIComponent(entityId)}?season=${season}`}>
              {formatValue(value)}
            </Link>
          )
        },
      })),
    ],
    [basePath, columnWidths, compareIds, config, filteredRows, onToggleCompare, season, selectedColumns],
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
    return left === undefined
      ? { minWidth: width, width }
      : { left, minWidth: width, width, maxWidth: width }
  }

  return (
    <Card className="gap-4">
      <CardHeader className="flex flex-wrap items-start justify-between gap-3">
        <CardTitle className="text-base">{config.title}</CardTitle>
        <div className="flex flex-wrap gap-1.5 text-xs">
          <Badge variant="secondary">{filteredRows.length} rows</Badge>
          <Badge variant="secondary">{selectedColumns.length} columns</Badge>
          <Badge variant="secondary">{compareIds.length} compared</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="relative max-w-sm">
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
        {controls}
        <div className="max-h-[75vh] overflow-auto rounded-md border">
          <table className="w-max min-w-full text-sm tabular">
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
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 hover:text-foreground"
                            onClick={header.column.getToggleSortingHandler()}
                          >
                            {flexRender(header.column.columnDef.header, header.getContext())}
                            <SortIcon direction={sorted} />
                          </button>
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
                    const heat = CONTROL_COLUMNS.includes(columnId)
                      ? undefined
                      : getHeatCellStyle(columnId, (cell.getValue() as RowValue) ?? null, columnStats, theme, palette)
                    return (
                      <td
                        key={cell.id}
                        style={{ ...cellStyle(columnId, cell.column.getSize()), ...heat }}
                        className={cn('px-2 py-1.5 whitespace-nowrap', sticky && 'sticky z-10 bg-card')}
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
