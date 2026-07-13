import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildGameOverviewUrl,
  compareDetailCellValues,
  computeTooltipBubbleLayout,
  formatDetailCellValue,
  getPageJumpSlots,
} from '../.detail-test-dist/src/detailUi.js';

test('formatDetailCellValue maps win values to readable result markers', () => {
  assert.equal(formatDetailCellValue('win_value', 1), 'W');
  assert.equal(formatDetailCellValue('win_value', 0), 'L');
  assert.equal(formatDetailCellValue('win_value', 0.5), 'T');
  assert.equal(formatDetailCellValue('win_value', 0.75), '0.75');
  assert.equal(formatDetailCellValue('point_margin', 7), '7');
});

test('buildGameOverviewUrl returns the nflsavant game-overview link', () => {
  assert.equal(
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

test('computeTooltipBubbleLayout shifts wide tooltips left when the pointer is near the right edge', () => {
  const layout = computeTooltipBubbleLayout({
    anchorRect: { bottom: 80, left: 620, right: 700, top: 40 },
    bubbleRect: { height: 110, width: 180 },
    pointer: { clientX: 690, clientY: 60 },
    viewport: { height: 900, width: 720 },
  });

  assert.equal(layout.minWidth, 280);
  assert.equal(layout.maxWidth, 360);
  assert.ok(layout.left < 690);
  assert.ok(layout.top > 60);
});