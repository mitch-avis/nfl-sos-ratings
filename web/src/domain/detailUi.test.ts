import { assert, test } from 'vitest';

import {
  buildGameOverviewUrl,
  compareDetailCellValues,
  formatDetailCellValue,
  getPageJumpSlots,
} from './detailUi';

test.each([
  ['win_value', 1, 'W'],
  ['win_value', 0, 'L'],
  ['win_value', 0.5, 'T'],
  ['win_value', 0.75, '0.75'],
  ['point_margin', 7, '7'],
])('formatDetailCellValue shows %s %s as %s', (column, value, expected) => {
  // Act
  const formatted = formatDetailCellValue(column, value);

  // Assert
  assert.strictEqual(formatted, expected);
});

test('buildGameOverviewUrl returns the nflsavant game-overview link', () => {
  // Act
  const url = buildGameOverviewUrl('2025_07_HOU_SEA');

  // Assert
  assert.strictEqual(url, 'https://www.nflsavant.com/game/2025_07_HOU_SEA');
});

test('compareDetailCellValues sorts schedule tiers by difficulty instead of alphabetically', () => {
  // Arrange
  const buckets = ['Middle', 'Softer', 'Tougher'];

  // Act
  buckets.sort((left, right) => compareDetailCellValues('opp_schedule_bucket', left, right));

  // Assert
  assert.deepEqual(buckets, ['Softer', 'Middle', 'Tougher']);
});

test.each([
  [{ atBottom: false, atTop: false, canScroll: true }, true, true],
  [{ atBottom: false, atTop: true, canScroll: true }, false, true],
  [{ atBottom: true, atTop: false, canScroll: true }, true, false],
])('getPageJumpSlots keeps up above down for %o', (position, upVisible, downVisible) => {
  // Act
  const slots = getPageJumpSlots(position);

  // Assert
  assert.deepEqual(slots, [
    { direction: 'up', visible: upVisible },
    { direction: 'down', visible: downVisible },
  ]);
});
