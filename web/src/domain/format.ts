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
  ['saor', 'SaOR'],
  ['sadr', 'SaDR'],
  ['saovr', 'SaOvR'],
  ['sacr', 'SaCR'],
  ['qraw', 'QRaw'],
  ['qsaor', 'QSaOR'],
  ['qsacr', 'QSaCR'],
  ['qoutcome', 'QOutcome'],
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