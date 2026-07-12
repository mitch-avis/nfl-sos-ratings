import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from 'react';
import { createPortal } from 'react-dom';

interface TooltipLabelProps {
  label: string;
  tooltip: string;
}

export function TooltipLabel({ label, tooltip }: TooltipLabelProps): ReactElement {
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const bubbleRef = useRef<HTMLSpanElement | null>(null);
  const pointerRef = useRef<{ clientX?: number; clientY?: number }>({});
  const [isOpen, setIsOpen] = useState(false);
  const [style, setStyle] = useState<CSSProperties>({});
  const [positionTick, setPositionTick] = useState(0);

  const tooltipId = useMemo(
    () => `tooltip-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`,
    [label],
  );

  const requestPositionUpdate = (clientX?: number, clientY?: number): void => {
    pointerRef.current = { clientX, clientY };
    setPositionTick((current) => current + 1);
  };

  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const bubble = bubbleRef.current;
    if (!isOpen || !anchor || !bubble) {
      return;
    }

    const rect = anchor.getBoundingClientRect();
    const { clientX, clientY } = pointerRef.current;
    const maxWidth = Math.min(360, window.innerWidth - 24);
    const bubbleRect = bubble.getBoundingClientRect();
    const preferredLeft = clientX !== undefined ? clientX + 14 : rect.left + 12;
    const left = Math.max(12, Math.min(preferredLeft, window.innerWidth - bubbleRect.width - 12));
    const preferredTop = clientY !== undefined ? clientY + 18 : rect.bottom + 10;
    const aboveTop = (clientY !== undefined ? clientY : rect.top) - bubbleRect.height - 12;
    const showAbove = preferredTop + bubbleRect.height > window.innerHeight - 12 && aboveTop >= 12;
    const top = showAbove
      ? aboveTop
      : Math.max(12, Math.min(preferredTop, window.innerHeight - bubbleRect.height - 12));

    setStyle({
      left: `${left}px`,
      top: `${top}px`,
      maxWidth: `${maxWidth}px`,
    });
  }, [isOpen, positionTick]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleViewportChange = (): void => {
      setPositionTick((current) => current + 1);
    };

    window.addEventListener('resize', handleViewportChange);
    window.addEventListener('scroll', handleViewportChange, true);
    return () => {
      window.removeEventListener('resize', handleViewportChange);
      window.removeEventListener('scroll', handleViewportChange, true);
    };
  }, [isOpen]);

  const tooltipBubble = (
    <span
      ref={bubbleRef}
      aria-hidden={!isOpen}
      className="tooltip-bubble"
      data-visible={isOpen}
      id={tooltipId}
      role="tooltip"
      style={style}
    >
      {tooltip}
    </span>
  );

  return (
    <>
      <span
        ref={anchorRef}
        aria-describedby={tooltipId}
        className="tooltip-anchor"
        tabIndex={0}
        onBlur={() => setIsOpen(false)}
        onFocus={() => {
          requestPositionUpdate();
          setIsOpen(true);
        }}
        onMouseEnter={(event) => {
          requestPositionUpdate(event.clientX, event.clientY);
          setIsOpen(true);
        }}
        onMouseMove={(event) => {
          requestPositionUpdate(event.clientX, event.clientY);
          setIsOpen(true);
        }}
        onMouseLeave={() => setIsOpen(false)}
      >
        <span className="tooltip-label">{label}</span>
      </span>
      {typeof document !== 'undefined' ? createPortal(tooltipBubble, document.body) : null}
    </>
  );
}