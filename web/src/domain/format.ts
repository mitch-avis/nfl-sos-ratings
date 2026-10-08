import type { MetricShape, RowValue } from '@/api/types';

const ACRONYMS = new Map<string, string>([
  ['qb', 'QB'],
  ['epa', 'EPA'],
  ['cpoe', 'CPOE'],
  ['td', 'TD'],
  ['tds', 'TDs'],
  ['int', 'INT'],
  ['ints', 'INTs'],
  ['id', 'ID'],
  ['sos', 'SoS'],
  ['srs', 'SRS'],
  ['pct', '%'],
  ['qopp', 'Opponent QB'],
  ['opp', 'Opponent'],
]);

export function humanizeColumn(column: string): string {
  if (/^[A-Z][A-Za-z]+/.test(column)) {
    return column;
  }

  return column
    .split('_')
    .map((part) => ACRONYMS.get(part.toLowerCase()) ?? `${part[0]?.toUpperCase() ?? ''}${part.slice(1)}`)
    .join(' ');
}

export function humanizeGroup(group: string): string {
  const overrides: Record<string, string> = {
    identity: 'Identity',
    ratings: 'Ratings',
    raw_total_stats: 'Raw Total Stats',
    raw_totals: 'Raw Totals',
    per_play_rates: 'Per-Play Rates',
    per_snap_rates: 'Per-Snap Rates',
    per_game_rates: 'Per-Game Rates',
    per_dropback_rates: 'Per-Dropback Rates',
    opponent_per_game_rates: 'Opponent Per-Game Rates',
    opponent_per_play_rates: 'Opponent Per-Play Rates',
    opponent_context: 'Opponent Context',
  };
  return overrides[group] ?? humanizeColumn(group);
}

export function formatValue(value: string | number | boolean | null): string {
  if (value === null) {
    return '—';
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toLocaleString(undefined, {
      maximumFractionDigits: 3,
      minimumFractionDigits: Math.abs(value) < 10 ? 2 : 1,
    });
  }
  return value;
}

/**
 * Decimal places for one column, so every value in it shows the same number and the decimal points
 * line up: whole numbers stay whole, points-per-game scores (`shape` `score`) get 2, and anything
 * else follows its scale (3 below 1, 2 below 100, 1 above). Null when the column has no numbers.
 */
export function columnDecimals(values: ReadonlyArray<RowValue>, shape: MetricShape | undefined): number | null {
  const numbers = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (numbers.length === 0) return null
  if (numbers.every((value) => Number.isInteger(value))) return 0
  if (shape === 'score') return 2
  const scale = Math.max(...numbers.map((value) => Math.abs(value)))
  if (scale < 1) return 3
  if (scale < 100) return 2
  return 1
}

/** Format `value` with a column's fixed `decimals`, or as `formatValue` would when there is none. */
export function formatFixed(value: RowValue, decimals: number | null): string {
  if (typeof value !== 'number' || decimals === null || !Number.isFinite(value)) return formatValue(value)
  // A value that rounds to zero shows as 0, not -0.
  const shown = Number(value.toFixed(decimals)) === 0 ? 0 : value
  return shown.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

/** `count` with its noun, singular for one: `1 game`, `4 games`, `2 matches` (given `plural`). */
export function countLabel(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}
