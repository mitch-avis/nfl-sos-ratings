import { humanizeColumn } from './format.js';
import type { ColumnMetadataPayload, EntityKind, MetricRegistryPayload } from './types.js';

export type MetricPolarity = 'higher' | 'lower' | 'neutral';

export interface MetricMetadata {
  label: string;
  fullName: string;
  shortDescription: string;
  detail: string;
  polarity: MetricPolarity;
  contextual?: boolean;
  heatmap?: boolean;
  category?: string;
  subcategory?: string;
}

// The backend metric registry is the single source of truth for labels,
// tooltips, polarity, and category placement. This map is hydrated from the
// column_metadata carried by every table payload plus the /api/metadata
// registry snapshot; the local inference below is only a fallback for
// columns that have not been hydrated yet.
const REGISTRY_METADATA = new Map<string, MetricMetadata>();

interface CategoryOrderEntry {
  rank: number;
  subcategories: string[];
}

const CATEGORY_ORDER: Record<'team' | 'qb', Map<string, CategoryOrderEntry>> = {
  team: new Map(),
  qb: new Map(),
};

export interface ColumnSection {
  title: string;
  rank: number;
}

export function getColumnSection(entity: 'team' | 'qb', column: string): ColumnSection | null {
  const metadata = REGISTRY_METADATA.get(column);
  if (!metadata?.category) {
    return null;
  }
  const orderEntry = CATEGORY_ORDER[entity].get(metadata.category);
  const categoryRank = orderEntry?.rank ?? CATEGORY_ORDER[entity].size;
  const subcategoryRank = metadata.subcategory
    ? Math.max(0, orderEntry?.subcategories.indexOf(metadata.subcategory) ?? 0)
    : -1;
  return {
    title: metadata.subcategory
      ? `${metadata.category} — ${metadata.subcategory}`
      : metadata.category,
    rank: categoryRank * 1000 + subcategoryRank + 1,
  };
}

export function hydrateColumnMetadata(
  columnMetadata: Record<string, ColumnMetadataPayload> | undefined,
): void {
  if (!columnMetadata) {
    return;
  }
  for (const [column, payload] of Object.entries(columnMetadata)) {
    REGISTRY_METADATA.set(column, toMetricMetadata(payload));
  }
}

export function hydrateMetricRegistry(registry: MetricRegistryPayload): void {
  for (const [name, payload] of Object.entries(registry.metrics)) {
    if (!REGISTRY_METADATA.has(name)) {
      REGISTRY_METADATA.set(name, toMetricMetadata(payload));
    }
  }
  for (const entity of ['team', 'qb'] as const) {
    registry.entities[entity].categories.forEach((category, rank) => {
      CATEGORY_ORDER[entity].set(category.name, {
        rank,
        subcategories: category.subcategories,
      });
    });
  }
}

function toMetricMetadata(payload: ColumnMetadataPayload): MetricMetadata {
  return {
    label: payload.label,
    fullName: payload.full_name,
    shortDescription: payload.description,
    detail: payload.description,
    polarity: payload.polarity,
    contextual: payload.contextual,
    heatmap: true,
    category: payload.category,
    subcategory: payload.subcategory ?? undefined,
  };
}

interface MetricTemplate {
  label: string;
  fullName: string;
}

interface PrefixContext {
  prefix: string;
  contextual: boolean;
  detailPrefix: string;
  apply: (template: MetricTemplate) => MetricTemplate;
}

interface SuffixContext {
  suffix: string;
  detailSuffix: string;
  apply: (template: MetricTemplate) => MetricTemplate;
}

const SUFFIX_CONTEXTS: SuffixContext[] = [
  {
    suffix: '_per_game',
    detailSuffix: 'This is normalized on a per-game basis.',
    apply: (template) => ({
      label: `${template.label} Per Game`,
      fullName: `${template.fullName} Per Game`,
    }),
  },
  {
    suffix: '_per_offensive_snap',
    detailSuffix: 'This is normalized by offensive snaps.',
    apply: (template) => ({
      label: `${template.label}/Off Snap`,
      fullName: `${template.fullName} Per Offensive Snap`,
    }),
  },
  {
    suffix: '_per_defensive_snap',
    detailSuffix: 'This is normalized by defensive snaps.',
    apply: (template) => ({
      label: `${template.label}/Def Snap`,
      fullName: `${template.fullName} Per Defensive Snap`,
    }),
  },
  {
    suffix: '_total',
    detailSuffix: 'This is the full season total.',
    apply: (template) => template,
  },
  {
    suffix: '_pct',
    detailSuffix: 'This is the percentile rank within the current season data.',
    apply: (template) => ({
      label: `${template.label} Pct`,
      fullName: `${template.fullName} Percentile`,
    }),
  },
];

const PREFIX_CONTEXTS: PrefixContext[] = [
  {
    prefix: 'qopp_',
    contextual: true,
    detailPrefix:
      'This is season-long opponent context for the selected quarterback. QB stat names here '
      + 'describe what those defenses allowed to opposing passers across the season.',
    apply: (template) => ({
      label: `Opp ${template.label}`,
      fullName: `Opponent ${template.fullName}`,
    }),
  },
  {
    prefix: 'opp_',
    contextual: true,
    detailPrefix:
      'This is season-long opponent context for the selected team or quarterback, not a '
      + 'single-game grade.',
    apply: (template) => ({
      label: `Opp ${template.label}`,
      fullName: `Opponent ${template.fullName}`,
    }),
  },
  {
    prefix: 'diff_',
    contextual: false,
    detailPrefix:
      'This compares the selected team or quarterback with the opponent context on the same metric.',
    apply: (template) => ({
      label: `${template.label} Diff`,
      fullName: `${template.fullName} Differential`,
    }),
  },
  {
    prefix: 'adj_',
    contextual: false,
    detailPrefix: 'Simultaneous-adjustment output from the ridge model.',
    apply: (template) => ({
      label: `Adj ${template.label}`,
      fullName: `Simultaneously Adjusted ${template.fullName}`,
    }),
  },
  {
    prefix: 'season_delta_',
    contextual: false,
    detailPrefix:
      'This is the average in these matchups compared with the selected team or quarterback\'s '
      + 'full-season average on the same metric.',
    apply: (template) => ({
      label: `${template.label} vs Season`,
      fullName: `${template.fullName} vs Season Baseline`,
    }),
  },
];

export const GLOSSARY_SECTIONS: Array<{ title: string; description: string; metrics: string[] }> = [
  {
    title: 'Team Rankings',
    description: 'Use these first when comparing full-team quality against actual schedules.',
    metrics: ['SaCR', 'sos', 'SRS', 'SaOR', 'SaDR', 'SaOvR'],
  },
  {
    title: 'QB Rankings',
    description: 'Use these first when comparing QB quality against the defenses each QB faced.',
    metrics: ['QSaCR', 'QSaOR', 'QSoS', 'faced_opp_SaCR', 'QRaw', 'QOutcome'],
  },
  {
    title: 'Key Supporting Stats',
    description: 'Representative efficiency metrics used throughout the current shell.',
    metrics: ['points_per_offensive_snap', 'qb_epa_per_dropback', 'qb_any_a', 'qb_sack_rate'],
  },
];

export function getMetricMetadata(column: string): MetricMetadata {
  return REGISTRY_METADATA.get(column) ?? buildFallbackMetricMetadata(column);
}

export function getMetricTooltip(column: string): string {
  const metadata = getMetricMetadata(column);
  const description = stripRepeatedMetricName(metadata.detail, metadata.fullName);
  return description ? `${metadata.fullName}. ${description}` : metadata.fullName;
}

export function getMetricDescription(column: string): string {
  const metadata = getMetricMetadata(column);
  return stripRepeatedMetricName(metadata.detail, metadata.fullName);
}

export function getGroupDescription(kind: EntityKind, group: string): string {
  const groupDescriptions: Record<string, string> = {
    identity: 'Names and identifiers used for deep linking and comparison.',
    ratings: 'Primary schedule-adjusted rating outputs. Start here for ranking.',
    raw_total_stats:
      'Count stats are shown as raw totals here, while rate and average stats keep their intrinsic values.',
    raw_totals:
      'For QBs, this group contains explicit season totals used to build the season-long passing surface.',
    per_game_rates:
      kind === 'teams'
        ? 'Season-average per-game team stats, kept alongside per-snap rates for a fuller surface view.'
        : 'Per-game QB rates derived from the season totals and game counts in the current data.',
    per_play_rates:
      'Play-normalized rates using each subcategory\'s natural denominator, such as snaps, dropbacks, attempts, carries, drives, or series.',
    per_snap_rates:
      'Snap-normalized rates that make teams comparable even when game environments differ.',
    per_dropback_rates: 'Per-dropback passing rates, generally the cleanest QB efficiency slice.',
    opponent_per_game_rates:
      'Season-long opponent context on a per-game basis. These columns describe the schedule, not the selected subject.',
    opponent_per_play_rates:
      'Season-long opponent context on the matching play denominator for the active surface.',
    opponent_context:
      'Context about the opponents faced. These are schedule descriptors, not direct better/worse scores.',
  };
  return groupDescriptions[group] ?? `${humanizeColumn(group)} fields in this view.`;
}

function buildFallbackMetricMetadata(column: string): MetricMetadata {
  const { prefixContext, coreColumn, suffixContext } = decomposeColumn(column);
  const coreTemplate = buildGenericTemplate(coreColumn);
  const withSuffix = suffixContext ? suffixContext.apply(coreTemplate) : coreTemplate;
  const template = prefixContext ? prefixContext.apply(withSuffix) : withSuffix;
  const detailParts: string[] = [];

  if (prefixContext) {
    detailParts.push(prefixContext.detailPrefix);
  }
  if (suffixContext) {
    detailParts.push(suffixContext.detailSuffix);
  }

  return {
    label: template.label,
    fullName: template.fullName,
    shortDescription:
      detailParts.find((detailPart) => detailPart.length > 0) ?? template.fullName,
    detail: detailParts.join(' '),
    polarity: inferMetricPolarity(column),
    contextual: prefixContext?.contextual ?? isContextColumn(column),
    heatmap: true,
  };
}

function decomposeColumn(column: string): {
  prefixContext: PrefixContext | undefined;
  coreColumn: string;
  suffixContext: SuffixContext | undefined;
} {
  const prefixContext = PREFIX_CONTEXTS.find((candidate) => column.startsWith(candidate.prefix));
  const withoutPrefix = prefixContext ? column.slice(prefixContext.prefix.length) : column;
  const suffixContext = SUFFIX_CONTEXTS.find((candidate) => withoutPrefix.endsWith(candidate.suffix));
  const coreColumn = suffixContext
    ? withoutPrefix.slice(0, -suffixContext.suffix.length)
    : withoutPrefix;
  return { prefixContext, coreColumn, suffixContext };
}

function buildGenericTemplate(column: string): MetricTemplate {
  const fullName = humanizeColumn(column);
  return {
    label: compactLabel(fullName),
    fullName,
  };
}

function compactLabel(label: string): string {
  return [
    ['Completion Percentage Above Expectation', 'CPOE'],
    ['Fourth Quarter Comebacks', '4QC'],
    ['Game Winning Drives', 'GWD'],
    ['Touchdown-Interception', 'TD-INT'],
    ['Touchdowns', 'TDs'],
    ['Interceptions', 'INTs'],
    ['Yards', 'Yds'],
    ['Attempts', 'Att'],
    ['Completions', 'Comp'],
    ['Percentage', '%'],
    ['Offensive', 'Off'],
    ['Defensive', 'Def'],
    ['Passing', 'Pass'],
    ['Rushing', 'Rush'],
    ['First Downs', '1Ds'],
  ].reduce((currentLabel, [from, to]) => currentLabel.replaceAll(from, to), label);
}

function stripRepeatedMetricName(detail: string, fullName: string): string {
  const trimmedDetail = detail.trim();
  if (!trimmedDetail) {
    return '';
  }

  const escapedFullName = fullName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const repeatedNamePattern = new RegExp(`^${escapedFullName}(?:[:.-]|\\s)+`, 'i');

  if (trimmedDetail.localeCompare(fullName, undefined, { sensitivity: 'accent' }) === 0) {
    return '';
  }

  return trimmedDetail.replace(repeatedNamePattern, '').trim();
}

function isContextColumn(column: string): boolean {
  return column === 'QSoS' || column.startsWith('opp_') || column.startsWith('qopp_');
}

function inferMetricPolarity(column: string): MetricPolarity {
  if (column.startsWith('opp_')) {
    return inferSubjectMetricPolarity(column.slice(4));
  }
  if (column.startsWith('qopp_')) {
    const baseColumn = column.slice(5);
    return baseColumn.startsWith('qb_')
      ? invertMetricPolarity(inferSubjectMetricPolarity(baseColumn))
      : inferSubjectMetricPolarity(baseColumn);
  }

  return inferSubjectMetricPolarity(column);
}

function invertMetricPolarity(polarity: MetricPolarity): MetricPolarity {
  if (polarity === 'higher') {
    return 'lower';
  }
  if (polarity === 'lower') {
    return 'higher';
  }
  return 'neutral';
}

function inferSubjectMetricPolarity(column: string): MetricPolarity {
  if (column.startsWith('def_')) {
    return 'higher';
  }
  if (column.includes('sack_yards_lost')) {
    return 'higher';
  }
  const lowerIsBetterPatterns = [
    'allowed',
    'fumbles_lost',
    'losses',
    'qb_sacks',
    'qb_sack_rate',
    'qb_sack_yards_lost',
    'qb_interceptions',
    'passing_interceptions',
    'sacks_suffered',
    'turnovers_committed',
  ];
  if (lowerIsBetterPatterns.some((token) => column.includes(token))) {
    return 'lower';
  }
  return 'higher';
}
