import { useCallback, useState } from 'react';

/** Center of the 22px dot. Margin in CSS pulls it back by the radius, so 11px sits on the bezel. */
const EDGE = '11px';

const CALIBRATION_POSITIONS: { id: string; top: string; left: string; index: number }[] = [
  { id: 'Pt1', top: EDGE, left: EDGE, index: 0 },
  { id: 'Pt2', top: EDGE, left: '50%', index: 1 },
  { id: 'Pt3', top: EDGE, left: `calc(100% - ${EDGE})`, index: 2 },
  { id: 'Pt4', top: '50%', left: EDGE, index: 3 },
  { id: 'Pt5', top: '50%', left: '50%', index: 4 },
  { id: 'Pt6', top: '50%', left: `calc(100% - ${EDGE})`, index: 5 },
  { id: 'Pt7', top: `calc(100% - ${EDGE})`, left: EDGE, index: 6 },
  { id: 'Pt8', top: `calc(100% - ${EDGE})`, left: '50%', index: 7 },
  { id: 'Pt9', top: `calc(100% - ${EDGE})`, left: `calc(100% - ${EDGE})`, index: 8 }
];

/** Same 3×3 order as the live nine-dot probe. */
export const DIAG_LOOK_LABELS = [
  'top-left',
  'top',
  'top-right',
  'left',
  'center',
  'right',
  'bot-left',
  'bot',
  'bot-right'
] as const;

export type DiagLookLabel = (typeof DIAG_LOOK_LABELS)[number];

type Props = {
  visible: boolean;
  /**
   * Look at the point, then click it. Return true if the sample was stored
   * (so the overlay can mark the point done).
   */
  onPointClick: (look: DiagLookLabel, pointIndex: number) => boolean;
  onComplete?: () => void;
};

/**
 * Full-screen 9-point probe. Dots sit on the bezel (same layout as calib),
 * one click per point for diagnostics.
 */
export function DiagProbeOverlay ({
  visible,
  onPointClick,
  onComplete
}: Props): JSX.Element | null {
  const [completed, setCompleted] = useState<Record<string, boolean>>({});

  const handlePointClick = useCallback(
    (id: string, index: number) => {
      if (completed[id]) return;
      const look = DIAG_LOOK_LABELS[index];
      if (!look) return;
      if (!onPointClick(look, index)) return;

      setCompleted((done) => {
        const updated = { ...done, [id]: true };
        if (Object.keys(updated).length === CALIBRATION_POSITIONS.length) {
          onComplete?.();
        }
        return updated;
      });
    },
    [completed, onComplete, onPointClick]
  );

  if (!visible) return null;

  const doneCount = Object.keys(completed).length;

  return (
    <div className="calibration-layer">
      <p className="calibration-hint">
        Look at each point, then click it once ({doneCount}/{CALIBRATION_POSITIONS.length}{' '}
        recorded).
      </p>
      {CALIBRATION_POSITIONS.map(({ id, top, left, index }) => {
        const isDone = completed[id];
        return (
          <button
            key={id}
            type="button"
            className={`calibration-point${isDone ? ' calibration-point--done' : ''}`}
            style={{ top, left }}
            disabled={isDone}
            onClick={() => handlePointClick(id, index)}
            aria-label={`Probe point ${DIAG_LOOK_LABELS[index] ?? id}`}
          />
        );
      })}
    </div>
  );
}
