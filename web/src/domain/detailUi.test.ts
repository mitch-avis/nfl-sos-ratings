import { assert, test } from 'vitest';

import {
  buildGameOverviewUrl,
  compareDetailCellValues,
  formatDetailCellValue,
  getPageJumpSlots,
} from './detailUi';

test('formatDetailCellValue maps win values to readable result markers', () => {
  assert.strictEqual(formatDetailCellValue('win_value', 1), 'W');
  assert.strictEqual(formatDetailCellValue('win_value', 0), 'L');
  assert.strictEqual(formatDetailCellValue('win_value', 0.5), 'T');
  assert.strictEqual(formatDetailCellValue('win_value', 0.75), '0.75');
  assert.strictEqual(formatDetailCellValue('point_margin', 7), '7');
});

test('buildGameOverviewUrl returns the nflsavant game-overview link', () => {
  assert.strictEqual(
    buildGameOverviewUrl('2025_07_HOU_SEA'),
    'https://www.nflsavant.com/game/2025_07_HOU_SEA',
  );
});

test('compareDetailCellValues sorts schedule tiers by difficulty instead of alphabetically', () => {
  const buckets = ['Middle', 'Softer', 'Tougher'];
  buckets.sort((left, right) => compareDetailCellValues('opp_schedule_bucket', left, right));
  assert.deepEqual(buckets, ['Softer', 'Middle', 'Tougher']);
});

test('getPageJumpSlots keeps the up slot above the down slot while toggling visibility', () => {
  assert.deepEqual(getPageJumpSlots({ atBottom: false, atTop: false, canScroll: true }), [
    { direction: 'up', visible: true },
    { direction: 'down', visible: true },
  ]);
  assert.deepEqual(getPageJumpSlots({ atBottom: false, atTop: true, canScroll: true }), [
    { direction: 'up', visible: false },
    { direction: 'down', visible: true },
  ]);
  assert.deepEqual(getPageJumpSlots({ atBottom: true, atTop: false, canScroll: true }), [
    { direction: 'up', visible: true },
    { direction: 'down', visible: false },
  ]);
});
