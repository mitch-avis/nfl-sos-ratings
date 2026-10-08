import { assert, test } from 'vitest';

import type { TablePayload } from '@/api/types';

import {
  buildGameLogGroups,
  buildOpponentBreakdown,
  enrichGameLogsWithOpponentRatings,
} from './detailAnalytics';
import {
  buildGameLogColumnSelection,
  buildSeasonViewTable,
  resolveEntityViewState,
} from './viewModel';

test('resolveEntityViewState defaults teams to Ratings, Offense, and all subcategories enabled', () => {
  // Act
  const teamState = resolveEntityViewState('teams');

  // Assert
  assert.strictEqual(teamState.primaryView, 'ratings');
  assert.strictEqual(teamState.teamCategory, 'Offense');
  assert.ok(Object.values(teamState.teamSubcategories.Offense).every(Boolean));
});

test('resolveEntityViewState defaults QBs to Ratings with all subcategories enabled', () => {
  // Act
  const qbState = resolveEntityViewState('qbs');

  // Assert
  assert.strictEqual(qbState.primaryView, 'ratings');
  assert.ok(Object.values(qbState.qbSubcategories).every(Boolean));
});

test('buildSeasonViewTable expands team per-game counts into raw totals', () => {
  // Arrange
  const viewState = resolveEntityViewState('teams', {
    primaryView: 'raw_total_stats',
    teamCategory: 'Overall',
  });
  const table: TablePayload = {
    column_groups: {
      identity: ['team'],
      ratings: ['team_rating'],
    },
    column_metadata: {
      points_for: {
        base_name: 'points_for',
        category: 'Overall',
        contextual: false,
        denominator: null,
        description: 'Points scored.',
        full_name: 'Points For',
        label: 'Pts',
        polarity: 'higher',
        shape: 'count',
        source: 'SCH',
        subcategory: null,
        percent: false,
      },
      games_played: {
        base_name: 'games_played',
        category: 'Overall',
        contextual: false,
        denominator: null,
        description: 'Games played.',
        full_name: 'Games Played',
        label: 'G',
        polarity: 'neutral',
        shape: 'count',
        source: 'SCH',
        subcategory: null,
        percent: false,
      },
      team_rating: {
        base_name: 'team_rating',
        category: 'Schedule-Adjusted Ratings',
        contextual: false,
        denominator: null,
        description: 'Rating.',
        full_name: 'Team Rating',
        label: 'Team Rating',
        polarity: 'higher',
        shape: 'score',
        source: 'D',
        subcategory: null,
        percent: false,
      },
    },
    rows: [{ team: 'DET', points_for: 28, games_played: 17, team_rating: 6.2 }],
    visible_columns: ['team', 'team_rating', 'points_for', 'games_played'],
  };

  // Act
  const derived = buildSeasonViewTable('teams', table, viewState);

  // Assert
  assert.deepEqual(derived.selectedColumns, ['team', 'points_for', 'games_played']);
  assert.strictEqual(derived.table.rows[0].points_for, 476);
  assert.strictEqual(derived.table.rows[0].games_played, 17);
});

test('buildSeasonViewTable keeps a season maximum as it is in raw totals', () => {
  // Arrange
  const viewState = resolveEntityViewState('teams', {
    primaryView: 'raw_total_stats',
    teamCategory: 'Offense',
  });
  const passing = {
    category: 'Offense',
    contextual: false,
    percent: false,
    polarity: 'higher',
    subcategory: 'Passing',
  } as const;
  const table: TablePayload = {
    column_groups: { identity: ['team'], ratings: [] },
    column_metadata: {
      passing_yards: {
        ...passing,
        base_name: 'passing_yards',
        denominator: null,
        description: 'Passing yards.',
        full_name: 'Passing Yards',
        label: 'Pass Yds',
        shape: 'count',
        source: 'PBP',
      },
      longest_pass: {
        ...passing,
        base_name: 'longest_pass',
        denominator: null,
        description: 'The longest completed pass.',
        full_name: 'Longest Completed Pass',
        label: 'Long Pass',
        shape: 'max',
        source: 'PBP',
      },
    },
    rows: [{ team: 'NE', games_played: 17, passing_yards: 230.5, longest_pass: 72 }],
    visible_columns: ['team', 'passing_yards', 'longest_pass'],
  };

  // Act
  const derived = buildSeasonViewTable('teams', table, viewState);

  // Assert
  assert.strictEqual(derived.table.rows[0].passing_yards, 3918.5);
  assert.strictEqual(derived.table.rows[0].longest_pass, 72);
});

test('buildGameLogColumnSelection folds results into the weekly base columns', () => {
  // Arrange
  const viewState = resolveEntityViewState('qbs', {
    primaryView: 'per_game_rates',
    qbSubcategories: {
      'Identity & Availability': false,
      'Passing Volume': true,
      'Passing Efficiency': false,
      'Advanced & Expected': false,
      'Pressure, Sacks & Pocket': false,
      Rushing: false,
      'Scoring, Clutch & Outcomes': false,
      'Turnovers & Ball Security': false,
    },
  });
  const gameLogs: TablePayload = {
    column_groups: {},
    column_metadata: {
      qb_pass_yards: {
        base_name: 'qb_pass_yards',
        category: 'Passing Volume',
        contextual: false,
        denominator: null,
        description: 'Passing yards.',
        full_name: 'QB Passing Yards',
        label: 'Pass Yds',
        polarity: 'higher',
        shape: 'count',
        source: 'PLS',
        subcategory: null,
        percent: false,
      },
      qb_game_winning_drive: {
        base_name: 'qb_game_winning_drive',
        category: 'Scoring, Clutch & Outcomes',
        contextual: false,
        denominator: null,
        description: 'Game-winning drives.',
        full_name: 'GWD',
        label: 'GWD',
        polarity: 'higher',
        shape: 'count',
        source: 'D',
        subcategory: null,
        percent: false,
      },
    },
    rows: [],
    visible_columns: [
      'week',
      'opponent_team',
      'game_id',
      'points_for',
      'points_allowed',
      'point_margin',
      'win_value',
      'turnover_margin',
      'opp_defense_rating',
      'qb_pass_yards',
      'qb_game_winning_drive',
    ],
  };

  // Act
  const selection = buildGameLogColumnSelection('qbs', gameLogs, viewState);

  // Assert
  assert.deepEqual(selection.columns.slice(0, 8), [
    'week',
    'opponent_team',
    'game_id',
    'points_for',
    'points_allowed',
    'point_margin',
    'win_value',
    'turnover_margin',
  ]);
  assert.ok(selection.columns.includes('opp_defense_rating'));
  assert.ok(selection.columns.includes('qb_pass_yards'));
  assert.ok(!selection.columns.includes('qb_game_winning_drive'));
});

test('buildOpponentBreakdown curates a team offense ledger with season-delta context', () => {
  // Arrange
  const seasonRow = {
    passing_epa: 4,
  };
  const gameLogs: TablePayload = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_SEA_LAR',
        opponent_team: 'SEA',
        opp_defense_rating_rank: 2,
        opp_league_size: 32,
        opp_team_rating: 1.1,
        opp_defense_rating: 1.4,
        opp_offense_rating: 0.9,
        passing_epa: 8,
        passing_yards: 280,
        point_margin: 7,
        week: 1,
      },
      {
        game_id: '2025_05_LAR_SEA',
        opponent_team: 'SEA',
        opp_defense_rating_rank: 2,
        opp_league_size: 32,
        opp_team_rating: 1.1,
        opp_defense_rating: 1.4,
        opp_offense_rating: 0.9,
        passing_epa: 10,
        passing_yards: 305,
        point_margin: 10,
        week: 5,
      },
      {
        game_id: '2025_02_LAR_ARI',
        opponent_team: 'ARI',
        opp_defense_rating_rank: 20,
        opp_league_size: 32,
        opp_team_rating: -0.4,
        opp_defense_rating: -0.2,
        opp_offense_rating: -0.5,
        passing_epa: 2,
        passing_yards: 210,
        point_margin: -3,
        week: 2,
      },
      {
        game_id: '2025_03_LAR_SF',
        opponent_team: 'SF',
        opp_defense_rating_rank: 9,
        opp_league_size: 32,
        opp_team_rating: 0.2,
        opp_defense_rating: 0.6,
        opp_offense_rating: 0.3,
        passing_epa: 4,
        passing_yards: 245,
        point_margin: 2,
        week: 3,
      },
    ],
    visible_columns: [
      'game_id',
      'week',
      'opponent_team',
      'opp_team_rating',
      'opp_defense_rating',
      'opp_offense_rating',
      'point_margin',
      'passing_epa',
      'passing_yards',
    ],
  };

  // Act
  const breakdown = buildOpponentBreakdown('teams', seasonRow, gameLogs, 'offense');

  // Assert
  const sea = breakdown.rows.find((row) => row.opponent_team === 'SEA')!;

  assert.deepEqual(
    breakdown.columns.map((column) => column.id),
    [
      'opponent_team',
      'games',
      'weeks',
      'opp_defense_rating',
      'opp_schedule_bucket',
      'passing_epa',
      'passing_yards',
      'season_delta_passing_epa',
    ],
  );
  assert.ok(sea);
  assert.strictEqual(sea.games, 2);
  assert.strictEqual(sea.weeks, '1, 5');
  assert.strictEqual(sea.passing_epa, 9);
  assert.strictEqual(sea.season_delta_passing_epa, 5);
  assert.strictEqual(sea.opp_schedule_bucket, 'Tougher');
  // Tiers come from the opponent's league rank on the tier metric: top 10 Tougher, bottom 10 Softer.
  assert.strictEqual(
    breakdown.rows.find((row) => row.opponent_team === 'SF')!.opp_schedule_bucket,
    'Tougher',
  );
  assert.strictEqual(
    breakdown.rows.find((row) => row.opponent_team === 'ARI')!.opp_schedule_bucket,
    'Middle',
  );
  assert.match(breakdown.description, /one row per opponent/i);
  assert.match(breakdown.description, /a division rival, gets one row that averages both games/i);
});

test.each([
  ['offense', 'opp_defense_rating'],
  ['defense', 'opp_offense_rating'],
  ['results', 'opp_team_rating'],
])('buildOpponentBreakdown uses the %s schedule tier metric %s', (groupId, expectedMetric) => {
  // Arrange
  const seasonRow = {
    passing_epa: 4,
    points_allowed_per_defensive_snap: 0.28,
    point_margin: 4,
  };
  const gameLogs: TablePayload = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_LAR_SEA',
        opponent_team: 'SEA',
        opp_team_rating: 1.4,
        opp_defense_rating: 1.1,
        opp_offense_rating: 0.7,
        point_margin: 6,
        win_value: 1,
        turnover_margin: 1,
        passing_epa: 8,
        points_allowed_per_defensive_snap: 0.22,
        week: 1,
      },
      {
        game_id: '2025_02_LAR_ARI',
        opponent_team: 'ARI',
        opp_team_rating: -1.2,
        opp_defense_rating: -0.8,
        opp_offense_rating: -0.4,
        point_margin: -2,
        win_value: 0,
        turnover_margin: -1,
        passing_epa: 2,
        points_allowed_per_defensive_snap: 0.31,
        week: 2,
      },
    ],
    visible_columns: [
      'game_id',
      'week',
      'opponent_team',
      'opp_team_rating',
      'opp_defense_rating',
      'opp_offense_rating',
      'point_margin',
      'win_value',
      'turnover_margin',
      'passing_epa',
      'points_allowed_per_defensive_snap',
    ],
  };

  // Act
  const breakdown = buildOpponentBreakdown('teams', seasonRow, gameLogs, groupId);

  // Assert
  assert.deepEqual(
    breakdown.columns.slice(3, 5).map((column) => column.id),
    [expectedMetric, 'opp_schedule_bucket'],
  );
});

test('buildGameLogGroups keeps weekly category columns aligned with the selected surface', () => {
  // Arrange
  const gameLogs: TablePayload = {
    column_groups: {},
    rows: [],
    visible_columns: [
      'game_id',
      'week',
      'opponent_team',
      'opp_team_rating',
      'opp_defense_rating',
      'point_margin',
      'passing_epa',
      'passing_yards',
    ],
  };

  // Act
  const groups = buildGameLogGroups('teams', gameLogs);

  // Assert
  assert.deepEqual(
    groups.find((group) => group.id === 'offense')!.columns,
    ['passing_epa', 'passing_yards'],
  );
});

test('buildOpponentBreakdown curates a QB ledger around passing performance and context', () => {
  // Arrange
  const seasonRow = {
    qb_any_a: 6.8,
    qb_epa_per_dropback: 0.18,
  };
  const gameLogs: TablePayload = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_BUF_KC',
        opponent_team: 'KC',
        opp_defense_rating_rank: 3,
        opp_league_size: 32,
        opp_team_rating: 1.3,
        opp_defense_rating: 1.6,
        point_margin: 6,
        qb_any_a: 7.4,
        qb_epa_per_dropback: 0.24,
        week: 1,
      },
      {
        game_id: '2025_08_BUF_KC',
        opponent_team: 'KC',
        opp_defense_rating_rank: 3,
        opp_league_size: 32,
        opp_team_rating: 1.3,
        opp_defense_rating: 1.6,
        point_margin: -2,
        qb_any_a: 6.6,
        qb_epa_per_dropback: 0.12,
        week: 8,
      },
      {
        game_id: '2025_03_BUF_NE',
        opponent_team: 'NE',
        opp_defense_rating_rank: 30,
        opp_league_size: 32,
        opp_team_rating: -0.7,
        opp_defense_rating: -0.5,
        point_margin: 10,
        qb_any_a: 7.1,
        qb_epa_per_dropback: 0.2,
        week: 3,
      },
    ],
    visible_columns: [
      'game_id',
      'week',
      'opponent_team',
      'opp_defense_rating',
      'opp_team_rating',
      'point_margin',
      'qb_epa_per_dropback',
      'qb_any_a',
    ],
  };

  // Act
  const breakdown = buildOpponentBreakdown('qbs', seasonRow, gameLogs, 'efficiency');

  // Assert
  const chiefs = breakdown.rows.find((row) => row.opponent_team === 'KC')!;

  assert.deepEqual(
    breakdown.columns.map((column) => column.id),
    [
      'opponent_team',
      'games',
      'weeks',
      'opp_defense_rating',
      'opp_schedule_bucket',
      'qb_any_a',
      'season_delta_qb_any_a',
    ],
  );
  assert.ok(chiefs);
  assert.strictEqual(chiefs.games, 2);
  assert.strictEqual(chiefs.qb_any_a, 7);
  assert.strictEqual(chiefs.season_delta_qb_any_a, 0.2);
  assert.strictEqual(chiefs.opp_schedule_bucket, 'Tougher');
  assert.strictEqual(
    breakdown.rows.find((row) => row.opponent_team === 'NE')!.opp_schedule_bucket,
    'Softer',
  );
});

test('enrichGameLogsWithOpponentRatings attaches opponent ratings and their league ranks', () => {
  // Arrange
  const gameLogs: TablePayload = {
    column_groups: {},
    rows: [
      { opponent_team: 'KC', week: 1 },
      { opponent_team: 'LV', week: 2 },
    ],
    visible_columns: ['week', 'opponent_team'],
  };
  const ratings: TablePayload = {
    column_groups: {},
    rows: [
      { team: 'DEN', team_rating: 8.4, defense_rating: 3.9 },
      { team: 'KC', team_rating: 3.5, defense_rating: 0.7 },
      { team: 'LV', team_rating: -9.0, defense_rating: -3.1 },
    ],
    visible_columns: ['team', 'team_rating', 'defense_rating'],
  };

  // Act
  const enriched = enrichGameLogsWithOpponentRatings(gameLogs, ratings);

  // Assert
  const kc = enriched.rows[0];
  assert.strictEqual(kc.opp_team_rating, 3.5);
  assert.strictEqual(kc.opp_team_rating_rank, 2);
  assert.strictEqual(enriched.rows[1].opp_defense_rating_rank, 3);
  assert.strictEqual(kc.opp_league_size, 3);
  assert.deepEqual(enriched.visible_columns, [
    'week',
    'opponent_team',
    'opp_team_rating',
    'opp_defense_rating',
  ]);
});
