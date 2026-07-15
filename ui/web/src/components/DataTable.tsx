import {
  type CellContext,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useMemo, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { formatValue } from '../format';
import { getMetricMetadata, getMetricTooltip } from '../metricMetadata';
import { TooltipLabel } from './TooltipLabel';
import { buildColumnStats, buildColumnWidths, getHeatCellStyle, sanitizeSorting } from '../tableState';
import type {
  EntityKind,
  PaletteMode,
  PrimaryView,
  RowValue,
  TablePayload,
  ThemeMode,
} from '../types';
import type { ResolvedEntityViewState } from '../viewModel';
import { ViewControls } from './ViewControls';

interface DataTableProps {
  basePath: string;
  compareIds: string[];
  defaultSortColumn: string;
  detailColumn: string;
  entityKind: EntityKind;
  identityColumns: string[];
  identityKey: string;
  onQueryChange: (query: string) => void;
  onResetView: () => void;
  onSelectTeamCategory: (category: string) => void;
  onSelectView: (view: PrimaryView) => void;
  onSortingChange: (sorting: SortingState) => void;
  onToggleSubcategory: (subcategory: string) => void;
  onToggleCompare: (entityId: string) => void;
  palette: PaletteMode;
  query: string;
  canReset: boolean;
  season: number;
  selectedColumns: string[];
  sorting: SortingState;
  theme: ThemeMode;
  title: string;
  table: TablePayload;
  viewState: ResolvedEntityViewState;
}

export function DataTable({
  basePath,
  compareIds,
  defaultSortColumn,
  detailColumn,
  entityKind,
  identityColumns,
  identityKey,
  onQueryChange,
  onResetView,
  onSelectTeamCategory,
  onSelectView,
  onSortingChange,
  onToggleSubcategory,
  onToggleCompare,
  palette,
  query,
  canReset,
  season,
  selectedColumns,
  sorting,
  theme,
  title,
  table,
  viewState,
}: DataTableProps): ReactElement {
  const availableColumnIds = useMemo(() => ['compare', 'rank', ...selectedColumns], [selectedColumns]);
  const defaultSortDescending = useMemo(
    () => getMetricMetadata(defaultSortColumn).polarity !== 'lower',
    [defaultSortColumn],
  );
  const fallbackSorting = useMemo<SortingState>(
    () =>
      availableColumnIds.includes(defaultSortColumn)
        ? [{ id: defaultSortColumn, desc: defaultSortDescending }]
        : [],
    [availableColumnIds, defaultSortColumn, defaultSortDescending],
  );

  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return table.rows;
    }
    return table.rows.filter((row) =>
      selectedColumns.some((column) => String(row[column] ?? '').toLowerCase().includes(normalized)),
    );
  }, [query, selectedColumns, table.rows]);

  const columnStats = useMemo(
    () => buildColumnStats(filteredRows, selectedColumns),
    [filteredRows, selectedColumns],
  );
  const columnWidths = useMemo(
    () => buildColumnWidths(table.rows, selectedColumns, identityColumns),
    [identityColumns, selectedColumns, table.rows],
  );

  const safeSorting = useMemo(
    () => sanitizeSorting(sorting, availableColumnIds, fallbackSorting),
    [availableColumnIds, fallbackSorting, sorting],
  );

  const stickyOffsets = useMemo(() => {
    const offsets: Record<string, number> = {};
    let left = 0;
    for (const id of ['compare', 'rank', ...identityColumns]) {
      offsets[id] = left;
      left += columnWidths[id] ?? 120;
    }
    return offsets;
  }, [columnWidths, identityColumns]);

  const columns = useMemo<ColumnDef<Record<string, string | number | boolean | null>>[]>(
    () =>
      [
        {
          id: 'compare',
          header: () => 'Compare',
          size: columnWidths.compare ?? 108,
          cell: (context: CellContext<Record<string, RowValue>, unknown>) => {
            const entityId = String(context.row.original[identityKey] ?? '');
            return (
              <label className="compare-checkbox">
                <input
                  type="checkbox"
                  checked={compareIds.includes(entityId)}
                  onChange={() => onToggleCompare(entityId)}
                />
                <span>Select</span>
              </label>
            );
          },
          enableSorting: false,
        },
        {
          id: 'rank',
          header: () => 'Rank',
          size: columnWidths.rank ?? 76,
          cell: () => null,
          enableSorting: false,
        },
        ...selectedColumns.map((column) => ({
          accessorKey: column,
          id: column,
          header: () => {
            const metadata = getMetricMetadata(column);
            return <TooltipLabel label={metadata.label} tooltip={getMetricTooltip(column)} />;
          },
          size: columnWidths[column] ?? 128,
          sortDescFirst: (() => {
            const sampleValue = filteredRows.find((row) => row[column] !== null)?.[column];
            if (typeof sampleValue === 'string') {
              return false;
            }
            return getMetricMetadata(column).polarity !== 'lower';
          })(),
          cell: (context: CellContext<Record<string, RowValue>, unknown>) => {
            const value = context.getValue() as RowValue;
            if (column === detailColumn) {
              const entityId = String(context.row.original[identityKey] ?? '');
              return (
                <Link className="detail-link" to={`${basePath}/${entityId}?season=${season}`}>
                  {formatValue(value)}
                </Link>
              );
            }
            return formatValue(value);
          },
        })),
      ],
    [
      basePath,
      columnStats,
      compareIds,
      columnWidths,
      detailColumn,
      identityKey,
      onToggleCompare,
      season,
      selectedColumns,
      filteredRows,
    ],
  );

  const reactTable = useReactTable({
    data: filteredRows,
    columns,
    state: { sorting: safeSorting },
    onSortingChange: (updater) => {
      const nextSorting = typeof updater === 'function' ? updater(safeSorting) : updater;
      onSortingChange(nextSorting);
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <section className="panel data-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Index View</p>
          <h2>{title}</h2>
        </div>
        <div className="panel-stats">
          <span>{filteredRows.length} rows</span>
          <span>{selectedColumns.length} columns</span>
          <span>{compareIds.length} compared</span>
        </div>
      </div>

      <div className="table-controls">
        <label className="search-control">
          <span>Search visible columns</span>
          <input
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search visible columns"
          />
        </label>

        <ViewControls
          canReset={canReset}
          kind={entityKind}
          onReset={onResetView}
          onSelectTeamCategory={onSelectTeamCategory}
          onSelectView={onSelectView}
          onToggleSubcategory={onToggleSubcategory}
          state={viewState}
        />
      </div>

      <div className="table-shell">
        <table>
          <thead>
            {reactTable.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sorted = header.column.getIsSorted();
                  const stickyLeft = stickyOffsets[header.column.id];
                  const columnWidth = header.getSize();
                  return (
                    <th
                      key={header.id}
                      className={stickyLeft !== undefined ? 'sticky-column sticky-header' : undefined}
                      style={
                        stickyLeft !== undefined
                          ? {
                              left: `${stickyLeft}px`,
                              minWidth: `${columnWidth}px`,
                              width: `${columnWidth}px`,
                              maxWidth: `${columnWidth}px`,
                            }
                          : {
                              minWidth: `${columnWidth}px`,
                              width: `${columnWidth}px`,
                            }
                      }
                    >
                      <button
                        type="button"
                        className="sort-button"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        <span className="sort-indicator">
                          {sorted === 'asc' ? '↑' : sorted === 'desc' ? '↓' : '·'}
                        </span>
                      </button>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {reactTable.getRowModel().rows.map((row, rowIndex) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => {
                  const stickyLeft = stickyOffsets[cell.column.id];
                  const columnWidth = cell.column.getSize();
                  const value =
                    cell.column.id === 'rank'
                      ? rowIndex + 1
                      : (cell.getValue() as RowValue | undefined);
                  const heatStyle =
                    cell.column.id === 'compare' || cell.column.id === 'rank'
                      ? undefined
                      : getHeatCellStyle(cell.column.id, value ?? null, columnStats, theme, palette);
                  return (
                    <td
                      key={cell.id}
                      className={stickyLeft !== undefined ? 'sticky-column sticky-cell' : undefined}
                      style={
                        stickyLeft !== undefined
                          ? {
                              left: `${stickyLeft}px`,
                              minWidth: `${columnWidth}px`,
                              width: `${columnWidth}px`,
                              maxWidth: `${columnWidth}px`,
                              ...heatStyle,
                            }
                          : {
                              minWidth: `${columnWidth}px`,
                              width: `${columnWidth}px`,
                              ...heatStyle,
                            }
                      }
                    >
                      {cell.column.id === 'rank'
                        ? rowIndex + 1
                        : flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}