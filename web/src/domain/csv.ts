import type { EntityKind, RowValue } from '@/api/types'

/**
 * CSV export of a table as the analyst sees it: the caller passes the columns in display order and
 * the rows in the current sort, after any search filter. Values are written raw (full precision,
 * not the table's rounded display) so the numbers can be reused elsewhere.
 */

function escapeCell(text: string): string {
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

function cellText(value: RowValue | undefined): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return escapeCell(value)
}

/** A CSV document (RFC 4180 quoting, CRLF line ends) with a header row of column keys. */
export function toCsv(columns: readonly string[], rows: ReadonlyArray<Record<string, RowValue>>): string {
  const lines = [
    columns.map(escapeCell).join(','),
    ...rows.map((row) => columns.map((column) => cellText(row[column])).join(',')),
  ]
  return `${lines.join('\r\n')}\r\n`
}

/** The download name for one season's team or QB table. */
export function csvFileName(kind: EntityKind, season: number): string {
  return `nfl-sos-ratings-${kind}-${season}.csv`
}
