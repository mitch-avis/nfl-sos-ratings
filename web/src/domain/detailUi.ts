import { formatValue } from './format';
import type { RowValue } from '@/api/types';

export interface ScrollPositionState {
  atBottom: boolean;
  atTop: boolean;
  canScroll: boolean;
}

export interface PageJumpSlot {
  direction: 'up' | 'down';
  visible: boolean;
}

const SCHEDULE_BUCKET_ORDER: Record<string, number> = {
  Softer: 0,
  Middle: 1,
  'Only Opponent': 1,
  Tougher: 2,
};

export function getPageJumpSlots(scrollPosition: ScrollPositionState): PageJumpSlot[] {
  if (!scrollPosition.canScroll) {
    return [];
  }

  return [
    { direction: 'up', visible: !scrollPosition.atTop },
    { direction: 'down', visible: !scrollPosition.atBottom },
  ];
}

export function formatDetailCellValue(column: string, value: RowValue): string {
  if (column === 'win_value' && typeof value === 'number' && Number.isFinite(value)) {
    if (value === 1) {
      return 'W';
    }
    if (value === 0) {
      return 'L';
    }
    if (value === 0.5) {
      return 'T';
    }
  }

  return formatValue(value);
}

export function buildGameOverviewUrl(gameId: string): string {
  return `https://www.nflsavant.com/game/${encodeURIComponent(gameId)}`;
}

export function compareDetailCellValues(
  column: string,
  left: RowValue,
  right: RowValue,
): number {
  if (column === 'opp_schedule_bucket') {
    return (SCHEDULE_BUCKET_ORDER[String(left ?? '')] ?? 1) - (SCHEDULE_BUCKET_ORDER[String(right ?? '')] ?? 1);
  }
  if (left === right) {
    return 0;
  }
  if (left === null) {
    return 1;
  }
  if (right === null) {
    return -1;
  }
  if (typeof left === 'number' && typeof right === 'number') {
    return left - right;
  }
  if (typeof left === 'boolean' && typeof right === 'boolean') {
    return Number(left) - Number(right);
  }
  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: 'base',
  });
}
