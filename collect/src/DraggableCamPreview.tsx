import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type Ref } from 'react';
import { createPortal } from 'react-dom';

type Props = {
  videoRef: Ref<HTMLVideoElement>;
  onInteractionLock?: (locked: boolean) => void;
  className?: string;
};

export function DraggableCamPreview ({
  videoRef,
  onInteractionLock,
  className
}: Props): JSX.Element {
  const rootRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    origLeft: number;
    origTop: number;
  } | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  const unlockSoon = useCallback(() => {
    window.setTimeout(() => onInteractionLock?.(false), 0);
  }, [onInteractionLock]);

  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    if (!(event.target as HTMLElement | null)?.closest?.('[data-cam-drag]')) return;
    const el = rootRef.current;
    if (!el) return;

    onInteractionLock?.(true);
    event.preventDefault();
    event.stopPropagation();

    const rect = el.getBoundingClientRect();
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origLeft: rect.left,
      origTop: rect.top
    };
    el.setPointerCapture(event.pointerId);
  }, [onInteractionLock]);

  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;

    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    const el = rootRef.current;
    const w = el?.offsetWidth ?? 300;
    const h = el?.offsetHeight ?? 280;
    const left = Math.min(window.innerWidth - w - 8, Math.max(8, drag.origLeft + dx));
    const top = Math.min(window.innerHeight - h - 8, Math.max(8, drag.origTop + dy));
    setPos({ left, top });
  }, []);

  const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    try {
      rootRef.current?.releasePointerCapture(event.pointerId);
    } catch {
      // already released
    }
    event.preventDefault();
    event.stopPropagation();
    unlockSoon();
  }, [unlockSoon]);

  useEffect(() => {
    const el = rootRef.current?.querySelector('[data-cam-drag]');
    if (!el) return;
    const block = (e: Event): void => {
      e.preventDefault();
      e.stopPropagation();
    };
    el.addEventListener('click', block, true);
    el.addEventListener('mousedown', block, true);
    return () => {
      el.removeEventListener('click', block, true);
      el.removeEventListener('mousedown', block, true);
    };
  }, []);

  const style = pos
    ? { left: pos.left, top: pos.top, right: 'auto', bottom: 'auto' }
    : undefined;

  return createPortal(
    (
      <div
        ref={rootRef}
        className={`cam-preview ${className ?? ''}`}
        data-cam-preview
        style={style}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <div className="cam-preview-shell">
          <div className="cam-preview-body" data-cam-drag>
            <div className="cam-preview-stage">
              <video ref={videoRef} className="cam-preview-video" autoPlay muted playsInline />
            </div>
          </div>
        </div>
      </div>
    ),
    document.body
  );
}
