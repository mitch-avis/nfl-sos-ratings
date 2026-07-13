import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildGameLogGroups,
  buildOpponentBreakdown,
  buildWeeklyHighlights,
} from '../.detail-test-dist/src/detailAnalytics.js';

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
  assert.equal(
    breakdown.rows.find((row) => row.opponent_team === 'SF').opp_schedule_bucket,
    'Middle',
  );
  assert.equal(
    breakdown.rows.find((row) => row.opponent_team === 'ARI').opp_schedule_bucket,
    'Softer',
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