import { formatValue } from './format.js';
import type { RowValue } from './types.js';

export interface ScrollPositionState {
  atBottom: boolean;
  atTop: boolean;
  canScroll: boolean;
}

export interface PageJumpSlot {
  direction: 'up' | 'down';
  visible: boolean;
}

interface TooltipAnchorRect {
  bottom: number;
  left: number;
  right: number;
  top: number;
}

interface TooltipBubbleRect {
  height: number;
  width: number;
}

interface TooltipPointer {
  clientX?: number;
  clientY?: number;
}

interface TooltipViewport {
  height: number;
  width: number;
}

export interface TooltipBubbleLayout {
  left: number;
  maxWidth: number;
  minWidth: number;
  top: number;
}

const TOOLTIP_PADDING = 12;
const TOOLTIP_X_OFFSET = 14;
const TOOLTIP_Y_OFFSET = 18;
const TOOLTIP_MAX_WIDTH = 360;
const TOOLTIP_MIN_WIDTH = 280;

const SCHEDULE_BUCKET_ORDER: Record<string, number> = {
  Softer: 0,
  Middle: 1,
  'Only Opponent': 1,
  Tougher: 2,
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

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

export function computeTooltipBubbleLayout({
  anchorRect,
  bubbleRect,
  pointer,
  viewport,
}: {
  anchorRect: TooltipAnchorRect;
  bubbleRect: TooltipBubbleRect;
  pointer: TooltipPointer;
  viewport: TooltipViewport;
}): TooltipBubbleLayout {
  const maxWidth = Math.min(TOOLTIP_MAX_WIDTH, viewport.width - TOOLTIP_PADDING * 2);
  const minWidth = Math.min(TOOLTIP_MIN_WIDTH, maxWidth);
  const desiredWidth = Math.max(bubbleRect.width, minWidth);
  const pointerX = pointer.clientX ?? anchorRect.right;
  const pointerY = pointer.clientY ?? anchorRect.bottom;
  const openLeft = viewport.width - pointerX - TOOLTIP_PADDING < minWidth;
  const preferredLeft = openLeft
    ? pointerX - desiredWidth - TOOLTIP_X_OFFSET
    : pointerX + TOOLTIP_X_OFFSET;
  const left = clamp(
    preferredLeft,
    TOOLTIP_PADDING,
    Math.max(TOOLTIP_PADDING, viewport.width - desiredWidth - TOOLTIP_PADDING),
  );

  const preferredTop = pointerY + TOOLTIP_Y_OFFSET;
  const aboveTop = pointerY - bubbleRect.height - TOOLTIP_PADDING;
  const showAbove =
    preferredTop + bubbleRect.height > viewport.height - TOOLTIP_PADDING
    && aboveTop >= TOOLTIP_PADDING;
  const top = showAbove
    ? aboveTop
    : clamp(
        preferredTop,
        TOOLTIP_PADDING,
        Math.max(TOOLTIP_PADDING, viewport.height - bubbleRect.height - TOOLTIP_PADDING),
      );

  return { left, maxWidth, minWidth, top };
}