import type { CSSProperties } from 'react';

import { formatValue } from './format';
import { getMetricMetadata } from './metricMetadata';
import type { PaletteMode, RowValue, ThemeMode } from './types';

export interface ColumnStats {
  min: number;
  max: number;
}

export function buildEnabledGroups(
  columnGroups: Record<string, string[]>,
  defaultGroups: string[],
): Record<string, boolean> {
  return Object.fromEntries(
    Object.keys(columnGroups).map((group) => [group, defaultGroups.includes(group)]),
  );
}

export function getSelectedColumns(
  columnGroups: Record<string, string[]>,
  enabledGroups: Record<string, boolean>,
  visibleColumns: string[],
): string[] {
  const identityColumns = columnGroups.identity ?? [];
  const ordered = Object.entries(columnGroups)
    .filter(([group]) => group !== 'identity')
    .filter(([group]) => enabledGroups[group])
    .flatMap(([, columns]) => columns);
  const fallback = visibleColumns.filter((column) => !identityColumns.includes(column));
  return [...identityColumns, ...(ordered.length > 0 ? ordered : fallback)];
}

export function sanitizeSorting(
  current: Array<{ id: string; desc: boolean }>,
  availableColumnIds: string[],
  fallback: Array<{ id: string; desc: boolean }>,
): Array<{ id: string; desc: boolean }> {
  const allowed = new Set(availableColumnIds);
  const valid = current.filter((entry) => allowed.has(entry.id));
  if (valid.length === current.length) {
    return current;
  }
  return fallback.filter((entry) => allowed.has(entry.id));
}

export function buildColumnStats(
  rows: Array<Record<string, RowValue>>,
  columns: string[],
): Record<string, ColumnStats> {
  const stats: Record<string, ColumnStats> = {};
  for (const column of columns) {
    const values = rows
      .map((row) => row[column])
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (values.length < 2) {
      continue;
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) {
      continue;
    }
    stats[column] = { min, max };
  }
  return stats;
}

export function buildColumnWidths(
  rows: Array<Record<string, RowValue>>,
  columns: string[],
  identityColumns: string[],
): Record<string, number> {
  const widths: Record<string, number> = {
    compare: 108,
    rank: Math.max(76, Math.min(108, 50 + String(rows.length).length * 10)),
  };

  const stickyColumns = new Set(identityColumns);

  for (const column of columns) {
    const isSticky = stickyColumns.has(column);
    const headerLength = getMetricMetadata(column).label.length;
    const valueLength = rows.reduce((currentMax, row) => {
      const formatted = formatValue(row[column] ?? null);
      return Math.max(currentMax, formatted.length);
    }, 0);
    const longest = Math.max(headerLength, valueLength);
    const width = Math.ceil(longest * 8.4 + (isSticky ? 42 : 36));

    widths[column] = Math.max(
      isSticky ? 96 : 112,
      Math.min(isSticky ? 280 : 360, width),
    );
  }

  return widths;
}

const HEAT_PALETTES = {
  classic: {
    light: { good: [225, 247, 237], bad: [252, 226, 222], mid: [255, 250, 240] },
    dark: { good: [8, 88, 64], bad: [103, 31, 38], mid: [22, 27, 34] },
  },
  broncos: {
    light: { good: [255, 231, 220], bad: [223, 233, 244], mid: [244, 247, 250] },
    dark: { good: [124, 51, 24], bad: [15, 48, 84], mid: [22, 27, 34] },
  },
} as const;

function interpolateColor(start: readonly number[], end: readonly number[], ratio: number): number[] {
  return start.map((value, index) => Math.round(value + (end[index] - value) * ratio));
}

function colorToCss(rgb: number[]): string {
  return `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})`;
}

export function getHeatCellStyle(
  column: string,
  value: RowValue,
  stats: Record<string, ColumnStats>,
  theme: ThemeMode,
  palette: PaletteMode,
): CSSProperties | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || !(column in stats)) {
    return undefined;
  }

  const metadata = getMetricMetadata(column);
  const polarity = metadata.polarity;
  if (polarity === 'neutral' || metadata.heatmap === false) {
    return undefined;
  }

  const { min, max } = stats[column];
  const span = max - min;
  if (span <= 0) {
    return undefined;
  }

  let normalized = (value - min) / span;
  if (polarity === 'lower') {
    normalized = 1 - normalized;
  }

  const paletteSet = HEAT_PALETTES[palette][theme];
  const color =
    normalized >= 0.5
      ? interpolateColor(paletteSet.mid, paletteSet.good, (normalized - 0.5) / 0.5)
      : interpolateColor(paletteSet.bad, paletteSet.mid, normalized / 0.5);

  return {
    backgroundColor: colorToCss(color),
  };
}