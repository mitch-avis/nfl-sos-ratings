import { humanizeColumn } from './format';
import type { EntityKind } from './types';

export type MetricPolarity = 'higher' | 'lower' | 'neutral';

export interface MetricMetadata {
  label: string;
  fullName: string;
  shortDescription: string;
  detail: string;
  polarity: MetricPolarity;
  contextual?: boolean;
  heatmap?: boolean;
}

interface MetricTemplate {
  label: string;
  fullName: string;
  shortDescription?: string;
  detail?: string;
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

const CORE_METRIC_TEMPLATES: Record<string, MetricTemplate> = {
  team: { label: 'Team', fullName: 'Team' },
  qb_id: { label: 'QB ID', fullName: 'Quarterback ID' },
  qb_name: { label: 'QB', fullName: 'Quarterback' },
  player_id: { label: 'Player ID', fullName: 'Player ID' },
  player_display_name: { label: 'QB', fullName: 'Quarterback' },
  games_played: { label: 'Games', fullName: 'Games Played' },
  qb_games_played: { label: 'QB Games', fullName: 'QB Games Played' },
  wins: { label: 'Wins', fullName: 'Wins' },
  losses: { label: 'Losses', fullName: 'Losses' },
  ties: { label: 'Ties', fullName: 'Ties' },
  win_pct: { label: 'Win %', fullName: 'Win Percentage' },
  qb_wins: { label: 'QB Wins', fullName: 'QB Wins' },
  qb_losses: { label: 'QB Losses', fullName: 'QB Losses' },
  qb_ties: { label: 'QB Ties', fullName: 'QB Ties' },
  qb_win_pct: { label: 'QB Win %', fullName: 'QB Win Percentage' },
  points_for: { label: 'Points For', fullName: 'Points For' },
  points_allowed: { label: 'Points Allowed', fullName: 'Points Allowed' },
  point_margin: { label: 'Point Margin', fullName: 'Point Margin' },
  win_value: { label: 'Win Value', fullName: 'Win Value' },
  turnover_margin: { label: 'TO Margin', fullName: 'Turnover Margin' },
  passing_yards: { label: 'Pass Yds', fullName: 'Passing Yards' },
  rushing_yards: { label: 'Rush Yds', fullName: 'Rushing Yards' },
  total_yards: { label: 'Total Yds', fullName: 'Total Yards' },
  passing_epa: { label: 'Pass EPA', fullName: 'Passing EPA' },
  rushing_epa: { label: 'Rush EPA', fullName: 'Rushing EPA' },
  passing_tds: { label: 'Pass TDs', fullName: 'Passing Touchdowns' },
  rushing_tds: { label: 'Rush TDs', fullName: 'Rushing Touchdowns' },
  passing_first_downs: { label: 'Pass 1Ds', fullName: 'Passing First Downs' },
  rushing_first_downs: { label: 'Rush 1Ds', fullName: 'Rushing First Downs' },
  passing_cpoe: {
    label: 'Pass CPOE',
    fullName: 'Passing Completion Percentage Above Expectation',
  },
  sacks_suffered: { label: 'Sacks', fullName: 'Sacks Suffered' },
  passing_interceptions: { label: 'Pass INTs', fullName: 'Passing Interceptions' },
  sack_fumbles_lost: { label: 'Sack Fum Lost', fullName: 'Sack Fumbles Lost' },
  rushing_fumbles_lost: { label: 'Rush Fum Lost', fullName: 'Rushing Fumbles Lost' },
  passing_yards_allowed: { label: 'Pass Yds Allowed', fullName: 'Passing Yards Allowed' },
  rushing_yards_allowed: { label: 'Rush Yds Allowed', fullName: 'Rushing Yards Allowed' },
  total_yards_allowed: { label: 'Total Yds Allowed', fullName: 'Total Yards Allowed' },
  passing_epa_allowed: { label: 'Pass EPA Allowed', fullName: 'Passing EPA Allowed' },
  rushing_epa_allowed: { label: 'Rush EPA Allowed', fullName: 'Rushing EPA Allowed' },
  passing_tds_allowed: { label: 'Pass TDs Allowed', fullName: 'Passing Touchdowns Allowed' },
  rushing_tds_allowed: { label: 'Rush TDs Allowed', fullName: 'Rushing Touchdowns Allowed' },
  passing_first_downs_allowed: {
    label: 'Pass 1Ds Allowed',
    fullName: 'Passing First Downs Allowed',
  },
  rushing_first_downs_allowed: {
    label: 'Rush 1Ds Allowed',
    fullName: 'Rushing First Downs Allowed',
  },
  passing_cpoe_allowed: {
    label: 'Pass CPOE Allowed',
    fullName: 'Passing Completion Percentage Above Expectation Allowed',
  },
  offensive_snaps: { label: 'Off Snaps', fullName: 'Offensive Snaps' },
  defensive_snaps: { label: 'Def Snaps', fullName: 'Defensive Snaps' },
  def_tackles_for_loss: { label: 'Def TFL', fullName: 'Defensive Tackles for Loss' },
  def_fumbles_forced: { label: 'Def Fum Forced', fullName: 'Defensive Fumbles Forced' },
  def_sacks: { label: 'Def Sacks', fullName: 'Defensive Sacks' },
  def_qb_hits: { label: 'Def QB Hits', fullName: 'Defensive QB Hits' },
  def_interceptions: { label: 'Def INTs', fullName: 'Defensive Interceptions' },
  def_pass_defended: { label: 'Def Pass Def.', fullName: 'Defensive Passes Defended' },
  def_safeties: { label: 'Def Safeties', fullName: 'Defensive Safeties' },
  qb_dropbacks: { label: 'QB Dropbacks', fullName: 'QB Dropbacks' },
  qb_offense_snaps: { label: 'QB Off Snaps', fullName: 'QB Offensive Snaps' },
  qb_attempts: { label: 'QB Att', fullName: 'QB Attempts' },
  qb_completions: { label: 'QB Comp', fullName: 'QB Completions' },
  qb_pass_yards: { label: 'QB Pass Yds', fullName: 'QB Passing Yards' },
  qb_pass_touchdowns: { label: 'QB Pass TDs', fullName: 'QB Passing Touchdowns' },
  qb_interceptions: { label: 'QB INTs', fullName: 'QB Interceptions' },
  qb_sacks: { label: 'QB Sacks', fullName: 'QB Sacks Taken' },
  qb_sack_yards_lost: { label: 'QB Sack Yds Lost', fullName: 'QB Sack Yards Lost' },
  qb_sack_fumbles_lost: { label: 'QB Sack Fum Lost', fullName: 'QB Sack Fumbles Lost' },
  qb_passing_epa: { label: 'QB Pass EPA', fullName: 'QB Passing EPA' },
  qb_epa_per_dropback: {
    label: 'QB EPA/DB',
    fullName: 'QB EPA Per Dropback',
    shortDescription: 'Per-dropback passing efficiency.',
    detail: 'QB passing EPA divided by dropbacks.',
  },
  qb_pass_yards_per_dropback: {
    label: 'QB Pass Yds/DB',
    fullName: 'QB Passing Yards Per Dropback',
    shortDescription: 'Passing yards per dropback.',
    detail: 'Passing yards divided by dropbacks.',
  },
  qb_td_int_margin_rate: {
    label: 'QB TD-INT Margin/DB',
    fullName: 'QB TD-INT Margin Rate',
    shortDescription: 'Touchdown minus interception margin per dropback.',
    detail: '(Passing touchdowns minus interceptions) divided by dropbacks.',
  },
  qb_sack_rate: {
    label: 'QB Sack Rate',
    fullName: 'QB Sack Rate',
    shortDescription: 'Sacks taken per dropback.',
    detail: 'Lower is better because it means fewer sacks absorbed per dropback.',
  },
  qb_any_a: {
    label: 'QB ANY/A',
    fullName: 'QB Adjusted Net Yards per Attempt',
    shortDescription: 'Adjusted net yards per attempt.',
    detail:
      'Adjusted Net Yards per Attempt using pass yards, touchdowns, interceptions, and '
      + 'sack-yard losses.',
  },
  qb_completion_percentage_above_expectation: {
    label: 'QB CPOE',
    fullName: 'QB Completion Percentage Above Expectation',
    shortDescription: 'Quarterback accuracy relative to expectation.',
    detail:
      'Completion Percentage Above Expectation compares actual completions to model-expected '
      + 'completions.',
  },
  qb_passer_rating: { label: 'QB Passer Rating', fullName: 'QB Passer Rating' },
  qb_fourth_quarter_comeback: { label: 'QB 4QC', fullName: 'QB Fourth-Quarter Comebacks' },
  qb_fourth_quarter_comebacks: { label: 'QB 4QC', fullName: 'QB Fourth-Quarter Comebacks' },
  qb_game_winning_drive: { label: 'QB GWD', fullName: 'QB Game-Winning Drives' },
  qb_game_winning_drives: { label: 'QB GWD', fullName: 'QB Game-Winning Drives' },
  qb_yards_per_attempt: { label: 'QB YPA', fullName: 'QB Yards Per Attempt' },
  qb_touchdown_rate: { label: 'QB TD Rate', fullName: 'QB Touchdown Rate' },
  qb_interception_rate: { label: 'QB INT Rate', fullName: 'QB Interception Rate' },
  qb_td_int_differential: {
    label: 'QB TD-INT Diff',
    fullName: 'QB Touchdown-Interception Differential',
  },
  qb_is_eligible: { label: 'Eligible', fullName: 'QB Eligibility Flag' },
};

const SUFFIX_CONTEXTS: SuffixContext[] = [
  {
    suffix: '_per_game',
    detailSuffix: 'This is normalized on a per-game basis.',
    apply: (template) => ({
      label: `${template.label} Per Game`,
      fullName: `${template.fullName} Per Game`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    suffix: '_per_offensive_snap',
    detailSuffix: 'This is normalized by offensive snaps.',
    apply: (template) => ({
      label: `${template.label}/Off Snap`,
      fullName: `${template.fullName} Per Offensive Snap`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    suffix: '_per_defensive_snap',
    detailSuffix: 'This is normalized by defensive snaps.',
    apply: (template) => ({
      label: `${template.label}/Def Snap`,
      fullName: `${template.fullName} Per Defensive Snap`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    suffix: '_total',
    detailSuffix: 'This is the full season total.',
    apply: (template) => template,
  },
  {
    suffix: '_pct',
    detailSuffix: 'This is the percentile rank within the current season output.',
    apply: (template) => ({
      label: `${template.label} Pct`,
      fullName: `${template.fullName} Percentile`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
];

const PREFIX_CONTEXTS: PrefixContext[] = [
  {
    prefix: 'qopp_',
    contextual: true,
    detailPrefix:
      'Opponent-context metric based on the defenses this QB faced; QB stat stems describe what '
      + 'those defenses allowed to opposing passers.',
    apply: (template) => ({
      label: `Opp ${template.label}`,
      fullName: `Opponent ${template.fullName}`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    prefix: 'opp_',
    contextual: true,
    detailPrefix: 'Opponent-context metric built from the teams this subject faced.',
    apply: (template) => ({
      label: `Opp ${template.label}`,
      fullName: `Opponent ${template.fullName}`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    prefix: 'diff_',
    contextual: false,
    detailPrefix: 'Subject minus opponent-context differential.',
    apply: (template) => ({
      label: `${template.label} Diff`,
      fullName: `${template.fullName} Differential`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
  {
    prefix: 'adj_',
    contextual: false,
    detailPrefix: 'Simultaneous-adjustment output from the ridge model.',
    apply: (template) => ({
      label: `Adj ${template.label}`,
      fullName: `Simultaneous-adjusted ${template.fullName}`,
      shortDescription: template.shortDescription,
      detail: template.detail,
    }),
  },
];

const METRIC_METADATA: Record<string, MetricMetadata> = {
  SaCR: {
    label: 'SaCR',
    fullName: 'Schedule-Adjusted Composite Rating',
    shortDescription: 'Final team composite rating.',
    detail:
      'Schedule-adjusted team composite. It blends offense, defense, and a smaller overall '
      + 'outcome-oriented layer into one summary score.',
    polarity: 'higher',
  },
  SaOR: {
    label: 'SaOR',
    fullName: 'Schedule-Adjusted Offense Rating',
    shortDescription: 'Schedule-adjusted team offense rating.',
    detail:
      'Team offense score after adjusting the offensive stat surface for the defenses each team '
      + 'actually faced.',
    polarity: 'higher',
  },
  SaDR: {
    label: 'SaDR',
    fullName: 'Schedule-Adjusted Defense Rating',
    shortDescription: 'Schedule-adjusted team defense rating.',
    detail:
      'Team defense score after adjusting the defensive stat surface for the offenses each team '
      + 'actually faced.',
    polarity: 'higher',
  },
  SaOvR: {
    label: 'SaOvR',
    fullName: 'Schedule-Adjusted Overall Rating',
    shortDescription: 'Schedule-adjusted overall signal.',
    detail:
      'Overall team component built from outcome-oriented signals such as win value and turnover '
      + 'margin before being schedule-adjusted.',
    polarity: 'higher',
  },
  SRS: {
    label: 'SRS',
    fullName: 'Simple Rating System',
    shortDescription: 'Point-margin simultaneous-adjustment reference.',
    detail:
      'A schedule-aware point-margin rating solved across the full league graph. Positive values '
      + 'mean a team outscored opponents by more than a league-average team would against the same '
      + 'schedule, which makes it a strong whole-team ordering baseline.',
    polarity: 'higher',
  },
  QSaCR: {
    label: 'QSaCR',
    fullName: 'QB Schedule-Adjusted Composite Rating',
    shortDescription: 'Final QB schedule-adjusted composite.',
    detail:
      'The best single overall QB ranking in this UI. It starts from the opponent-adjusted '
      + 'passing-performance base, then adds a smaller outcome layer.',
    polarity: 'higher',
  },
  QSaOR: {
    label: 'QSaOR',
    fullName: 'QB Schedule-Adjusted Offense Rating',
    shortDescription: 'Schedule-adjusted QB offense rating.',
    detail:
      'QB passing-performance rating after adjusting for the defenses faced. Use this when you '
      + 'want the cleanest opponent-adjusted QB performance signal without the extra outcome blend.',
    polarity: 'higher',
  },
  QRaw: {
    label: 'QRaw',
    fullName: 'QB Raw Performance Composite',
    shortDescription: 'Unadjusted QB performance composite.',
    detail:
      'Unadjusted QB composite from the core passing stat pool before schedule context is added.',
    polarity: 'higher',
  },
  QSoS: {
    label: 'QSoS',
    fullName: 'QB Strength of Schedule',
    shortDescription: 'QB schedule strength.',
    detail:
      'Difficulty of the defenses a QB faced. Higher means a tougher opponent slate, not a better '
      + 'QB performance. Treat it as context, not as the main ranking output.',
    polarity: 'higher',
    contextual: true,
  },
  QOutcome: {
    label: 'QOutcome',
    fullName: 'QB Outcome Layer',
    shortDescription: 'Secondary QB outcome layer.',
    detail:
      'Secondary QB outcome signal built from wins and late-game events such as fourth-quarter '
      + 'comebacks and game-winning drives.',
    polarity: 'higher',
  },
  QRaw_pct: {
    label: 'QRaw Pct',
    fullName: 'QB Raw Performance Composite Percentile',
    shortDescription: 'Percentile rank for QRaw.',
    detail: 'Percentile rank for the raw QB composite within the current season output.',
    polarity: 'higher',
  },
  QSaOR_pct: {
    label: 'QSaOR Pct',
    fullName: 'QB Schedule-Adjusted Offense Rating Percentile',
    shortDescription: 'Percentile rank for QSaOR.',
    detail:
      'Percentile rank for the schedule-adjusted QB offense rating within the current season '
      + 'output.',
    polarity: 'higher',
  },
  QSoS_pct: {
    label: 'QSoS Pct',
    fullName: 'QB Strength of Schedule Percentile',
    shortDescription: 'Percentile rank for QSoS.',
    detail: 'Percentile rank for schedule difficulty within the current season output.',
    polarity: 'higher',
    contextual: true,
  },
  QSaCR_pct: {
    label: 'QSaCR Pct',
    fullName: 'QB Schedule-Adjusted Composite Rating Percentile',
    shortDescription: 'Percentile rank for QSaCR.',
    detail:
      'Percentile rank for the final schedule-adjusted QB composite within the current season '
      + 'output.',
    polarity: 'higher',
  },
  QOutcome_pct: {
    label: 'QOutcome Pct',
    fullName: 'QB Outcome Layer Percentile',
    shortDescription: 'Percentile rank for QOutcome.',
    detail: 'Percentile rank for the QB outcome layer within the current season output.',
    polarity: 'higher',
  },
  points_per_offensive_snap: {
    label: 'Points/Off Snap',
    fullName: 'Points Per Offensive Snap',
    shortDescription: 'Team scoring efficiency by offensive snap.',
    detail: 'Points scored divided by offensive snaps.',
    polarity: 'higher',
  },
};

export const GLOSSARY_SECTIONS: Array<{ title: string; description: string; metrics: string[] }> = [
  {
    title: 'Team Rankings',
    description: 'Use these first when comparing full-team quality against actual schedules.',
    metrics: ['SaCR', 'SaOR', 'SaDR', 'SaOvR', 'SRS'],
  },
  {
    title: 'QB Rankings',
    description: 'Use these first when comparing QB quality against the defenses each QB faced.',
    metrics: ['QSaCR', 'QSaOR', 'QRaw', 'QSoS', 'QOutcome'],
  },
  {
    title: 'Key Supporting Stats',
    description: 'Representative efficiency metrics used throughout the current shell.',
    metrics: ['points_per_offensive_snap', 'qb_epa_per_dropback', 'qb_any_a', 'qb_sack_rate'],
  },
];

export function getMetricMetadata(column: string): MetricMetadata {
  return METRIC_METADATA[column] ?? buildFallbackMetricMetadata(column);
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
    raw_totals:
      'For QBs, this group contains explicit season totals used to build the season-long passing surface.',
    per_game_rates:
      kind === 'teams'
        ? 'Season-average per-game team stats, kept alongside per-snap rates for a fuller surface view.'
        : 'Per-game QB rates derived from the season totals and game counts in the current output.',
    per_snap_rates:
      'Snap-normalized rates that make teams comparable even when game environments differ.',
    per_dropback_rates: 'Per-dropback passing rates, generally the cleanest QB efficiency slice.',
    opponent_context:
      'Context about the opponents faced. These are schedule descriptors, not direct better/worse scores.',
  };
  return groupDescriptions[group] ?? `${humanizeColumn(group)} fields in this view.`;
}

export function inferMetricPolarity(column: string): MetricPolarity {
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

function buildFallbackMetricMetadata(column: string): MetricMetadata {
  const { prefixContext, coreColumn, suffixContext } = decomposeColumn(column);
  const coreTemplate = CORE_METRIC_TEMPLATES[coreColumn] ?? buildGenericTemplate(coreColumn);
  const withSuffix = suffixContext ? suffixContext.apply(coreTemplate) : coreTemplate;
  const template = prefixContext ? prefixContext.apply(withSuffix) : withSuffix;
  const detailParts = [stripRepeatedMetricName(template.detail ?? '', template.fullName)];

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
      template.shortDescription
      ?? detailParts.find((detailPart) => detailPart.length > 0)
      ?? template.fullName,
    detail: detailParts.filter((detailPart) => detailPart.length > 0).join(' '),
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