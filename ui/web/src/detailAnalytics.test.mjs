import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildGameLogGroups,
  buildOpponentBreakdown,
  buildWeeklyHighlights,
} from '../.detail-test-dist/src/detailAnalytics.js';
import {
  buildGameLogColumnSelection,
  buildSeasonViewTable,
  resolveEntityViewState,
} from '../.detail-test-dist/src/viewModel.js';

test('resolveEntityViewState defaults to Ratings, Offense, and all subcategories enabled', () => {
  const teamState = resolveEntityViewState('teams');
  const qbState = resolveEntityViewState('qbs');

  assert.equal(teamState.primaryView, 'ratings');
  assert.equal(teamState.teamCategory, 'Offense');
  assert.ok(Object.values(teamState.teamSubcategories.Offense).every(Boolean));
  assert.equal(qbState.primaryView, 'ratings');
  assert.ok(Object.values(qbState.qbSubcategories).every(Boolean));
});

test('buildSeasonViewTable expands team per-game counts into raw totals', () => {
  const viewState = resolveEntityViewState('teams', {
    primaryView: 'raw_total_stats',
    teamCategory: 'Overall',
  });
  const table = {
    column_groups: {
      identity: ['team'],
      ratings: ['SaCR'],
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
      },
      SaCR: {
        base_name: 'SaCR',
        category: 'Schedule-Adjusted Ratings',
        contextual: false,
        denominator: null,
        description: 'Rating.',
        full_name: 'SaCR',
        label: 'SaCR',
        polarity: 'higher',
        shape: 'score',
        source: 'D',
        subcategory: null,
      },
    },
    rows: [{ team: 'DET', points_for: 28, games_played: 17, SaCR: 1.2 }],
    visible_columns: ['team', 'SaCR', 'points_for', 'games_played'],
  };

  const derived = buildSeasonViewTable('teams', table, viewState);

  assert.deepEqual(derived.selectedColumns, ['team', 'points_for', 'games_played']);
  assert.equal(derived.table.rows[0].points_for, 476);
  assert.equal(derived.table.rows[0].games_played, 17);
});

test('buildGameLogColumnSelection folds results into the weekly base columns', () => {
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
  const gameLogs = {
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
      'opp_SaDR',
      'qb_pass_yards',
      'qb_game_winning_drive',
    ],
  };

  const selection = buildGameLogColumnSelection('qbs', gameLogs, viewState);

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
  assert.ok(selection.columns.includes('opp_SaDR'));
  assert.ok(selection.columns.includes('qb_pass_yards'));
  assert.ok(!selection.columns.includes('qb_game_winning_drive'));
});

test('buildWeeklyHighlights adds a recent three-game card with season-baseline context', () => {
  const seasonRow = {
    points_per_offensive_snap: 0.3,
  };
  const gameLogs = {
    column_groups: {},
    rows: [
      {
        opponent_team: 'ATL',
        opp_SaCR: 0.2,
        points_per_offensive_snap: 0.2,
        week: 1,
      },
      {
        opponent_team: 'SEA',
        opp_SaCR: 0.5,
        points_per_offensive_snap: 0.25,
        week: 2,
      },
      {
        opponent_team: 'SF',
        opp_SaCR: 1.4,
        points_per_offensive_snap: 0.35,
        week: 3,
      },
      {
        opponent_team: 'LAR',
        opp_SaCR: 0.9,
        points_per_offensive_snap: 0.45,
        week: 4,
      },
    ],
    visible_columns: ['week', 'opponent_team', 'points_per_offensive_snap', 'opp_SaCR'],
  };

  const highlights = buildWeeklyHighlights('teams', seasonRow, gameLogs);
  const rollingHighlight = highlights.find((highlight) => highlight.eyebrow === 'Recent 3-Game');

  assert.ok(rollingHighlight);
  assert.equal(rollingHighlight.value, '0.35');
  assert.match(rollingHighlight.context, /Weeks 2-4 avg 0.35/);
  assert.match(rollingHighlight.context, /vs season 0.30/);
});

test('buildOpponentBreakdown curates a team offense ledger with season-delta context', () => {
  const seasonRow = {
    passing_epa: 4,
  };
  const gameLogs = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_SEA_LAR',
        opponent_team: 'SEA',
        opp_SaCR: 1.1,
        opp_SaDR: 1.4,
        opp_SaOR: 0.9,
        passing_epa: 8,
        passing_yards: 280,
        point_margin: 7,
        week: 1,
      },
      {
        game_id: '2025_05_LAR_SEA',
        opponent_team: 'SEA',
        opp_SaCR: 1.1,
        opp_SaDR: 1.4,
        opp_SaOR: 0.9,
        passing_epa: 10,
        passing_yards: 305,
        point_margin: 10,
        week: 5,
      },
      {
        game_id: '2025_02_LAR_ARI',
        opponent_team: 'ARI',
        opp_SaCR: -0.4,
        opp_SaDR: -0.2,
        opp_SaOR: -0.5,
        passing_epa: 2,
        passing_yards: 210,
        point_margin: -3,
        week: 2,
      },
      {
        game_id: '2025_03_LAR_SF',
        opponent_team: 'SF',
        opp_SaCR: 0.2,
        opp_SaDR: 0.6,
        opp_SaOR: 0.3,
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
      'opp_SaCR',
      'opp_SaDR',
      'opp_SaOR',
      'point_margin',
      'passing_epa',
      'passing_yards',
    ],
  };

  const breakdown = buildOpponentBreakdown('teams', seasonRow, gameLogs, 'offense');
  const sea = breakdown.rows.find((row) => row.opponent_team === 'SEA');

  assert.deepEqual(
    breakdown.columns.map((column) => column.id),
    [
      'opponent_team',
      'games',
      'weeks',
      'opp_SaDR',
      'opp_schedule_bucket',
      'passing_epa',
      'passing_yards',
      'season_delta_passing_epa',
    ],
  );
  assert.ok(sea);
  assert.equal(sea.games, 2);
  assert.equal(sea.weeks, '1, 5');
  assert.equal(sea.passing_epa, 9);
  assert.equal(sea.season_delta_passing_epa, 5);
  assert.equal(sea.opp_schedule_bucket, 'Tougher');
  // Buckets apply the fixed thresholds to the raw league z-score:
  // +0.5 or higher is Tougher, -0.5 or lower is Softer.
  assert.equal(
    breakdown.rows.find((row) => row.opponent_team === 'SF').opp_schedule_bucket,
    'Tougher',
  );
  assert.equal(
    breakdown.rows.find((row) => row.opponent_team === 'ARI').opp_schedule_bucket,
    'Middle',
  );
  assert.match(breakdown.description, /unique opponents/i);
  assert.match(breakdown.description, /division opponents are averaged together/i);
});

test('buildOpponentBreakdown switches the team schedule tier metric with the active category', () => {
  const seasonRow = {
    passing_epa: 4,
    points_allowed_per_defensive_snap: 0.28,
    point_margin: 4,
  };
  const gameLogs = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_LAR_SEA',
        opponent_team: 'SEA',
        opp_SaCR: 1.4,
        opp_SaDR: 1.1,
        opp_SaOR: 0.7,
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
        opp_SaCR: -1.2,
        opp_SaDR: -0.8,
        opp_SaOR: -0.4,
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
      'opp_SaCR',
      'opp_SaDR',
      'opp_SaOR',
      'point_margin',
      'win_value',
      'turnover_margin',
      'passing_epa',
      'points_allowed_per_defensive_snap',
    ],
  };

  const offense = buildOpponentBreakdown('teams', seasonRow, gameLogs, 'offense');
  const defense = buildOpponentBreakdown('teams', seasonRow, gameLogs, 'defense');
  const results = buildOpponentBreakdown('teams', seasonRow, gameLogs, 'results');

  assert.deepEqual(
    offense.columns.slice(3, 5).map((column) => column.id),
    ['opp_SaDR', 'opp_schedule_bucket'],
  );
  assert.deepEqual(
    defense.columns.slice(3, 5).map((column) => column.id),
    ['opp_SaOR', 'opp_schedule_bucket'],
  );
  assert.deepEqual(
    results.columns.slice(3, 5).map((column) => column.id),
    ['opp_SaCR', 'opp_schedule_bucket'],
  );
});

test('buildGameLogGroups keeps weekly category columns aligned with the selected surface', () => {
  const gameLogs = {
    column_groups: {},
    rows: [],
    visible_columns: [
      'game_id',
      'week',
      'opponent_team',
      'opp_SaCR',
      'opp_SaDR',
      'point_margin',
      'passing_epa',
      'passing_yards',
    ],
  };

  const groups = buildGameLogGroups('teams', gameLogs);
  assert.deepEqual(
    groups.find((group) => group.id === 'offense').columns,
    ['passing_epa', 'passing_yards'],
  );
});

test('buildOpponentBreakdown curates a QB ledger around passing performance and context', () => {
  const seasonRow = {
    qb_any_a: 6.8,
    qb_epa_per_dropback: 0.18,
  };
  const gameLogs = {
    column_groups: {},
    rows: [
      {
        game_id: '2025_01_BUF_KC',
        opponent_team: 'KC',
        opp_SaCR: 1.3,
        opp_SaDR: 1.6,
        point_margin: 6,
        qb_any_a: 7.4,
        qb_epa_per_dropback: 0.24,
        week: 1,
      },
      {
        game_id: '2025_08_BUF_KC',
        opponent_team: 'KC',
        opp_SaCR: 1.3,
        opp_SaDR: 1.6,
        point_margin: -2,
        qb_any_a: 6.6,
        qb_epa_per_dropback: 0.12,
        week: 8,
      },
      {
        game_id: '2025_03_BUF_NE',
        opponent_team: 'NE',
        opp_SaCR: -0.7,
        opp_SaDR: -0.5,
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
      'opp_SaDR',
      'opp_SaCR',
      'point_margin',
      'qb_epa_per_dropback',
      'qb_any_a',
    ],
  };

  const breakdown = buildOpponentBreakdown('qbs', seasonRow, gameLogs, 'efficiency');
  const chiefs = breakdown.rows.find((row) => row.opponent_team === 'KC');

  assert.deepEqual(
    breakdown.columns.map((column) => column.id),
    [
      'opponent_team',
      'games',
      'weeks',
      'opp_SaDR',
      'opp_schedule_bucket',
      'qb_any_a',
      'season_delta_qb_any_a',
    ],
  );
  assert.ok(chiefs);
  assert.equal(chiefs.games, 2);
  assert.equal(chiefs.qb_any_a, 7);
  assert.equal(chiefs.season_delta_qb_any_a, 0.2);
  assert.equal(chiefs.opp_schedule_bucket, 'Tougher');
});