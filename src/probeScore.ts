/** Dot radius. The disc sits against the bezel; this is also its center inset. */
export const PROBE_DOT_RADIUS_PX = 11;

/** Same dots as the live nine-dot probe, in normalized viewport. */
export const probeTargetsForViewport = (
  viewportW: number,
  viewportH: number
): Record<string, { x: number; y: number }> => {
  const x0 = PROBE_DOT_RADIUS_PX / Math.max(viewportW, 1);
  const y0 = PROBE_DOT_RADIUS_PX / Math.max(viewportH, 1);
  const x1 = 1 - x0;
  const y1 = 1 - y0;
  return {
    'top-left': { x: x0, y: y0 },
    top: { x: 0.5, y: y0 },
    'top-right': { x: x1, y: y0 },
    left: { x: x0, y: 0.5 },
    center: { x: 0.5, y: 0.5 },
    right: { x: x1, y: 0.5 },
    'bot-left': { x: x0, y: y1 },
    bot: { x: 0.5, y: y1 },
    'bot-right': { x: x1, y: y1 }
  };
};

/** Fallback for a 1294×620 page. Live scoring uses probeTargetsForViewport. */
export const PROBE_TARGETS = probeTargetsForViewport(1294, 620);

export interface ProbeScorePoint {
  look: string;
  predX: number;
  predY: number;
  targetX: number;
  targetY: number;
  err: number;
}

export interface ProbeScore {
  /** Mean Euclidean miss in the unit square. Shown as avgNorm. */
  maeNorm: number;
  maeX: number;
  maeY: number;
  /** sum(d²) / sum(d). */
  wNorm: number;
  /** sum(d³) / sum(d²). */
  sqrNorm: number;
  /** Mean miss with horizontal distance stretched by page width / height. */
  scrNorm: number;
  points: ProbeScorePoint[];
}

export interface ProbeFeatureSample {
  look: string;
  features: Record<string, number>;
  npredX?: number | null;
  npredY?: number | null;
  /** Iris in the eye-corner frame. Mesh x/y, plus dark-pixel vertical. */
  corner?: {
    leftX: number;
    leftY: number;
    rightX: number;
    rightY: number;
    ply: number;
    pry: number;
  };
}

const summarize = (
  points: { predX: number; predY: number; targetX: number; targetY: number }[],
  aspect: number
): Pick<ProbeScore, 'maeNorm' | 'maeX' | 'maeY' | 'wNorm' | 'sqrNorm' | 'scrNorm'> => {
  const n = points.length;
  let absX = 0;
  let absY = 0;
  let norm = 0;
  let screen = 0;
  let sumD2 = 0;
  let sumD3 = 0;
  for (const point of points) {
    const dx = point.predX - point.targetX;
    const dy = point.predY - point.targetY;
    const dist = Math.hypot(dx, dy);
    absX += Math.abs(dx);
    absY += Math.abs(dy);
    norm += dist;
    screen += Math.hypot(dx * aspect, dy);
    sumD2 += dist * dist;
    sumD3 += dist * dist * dist;
  }
  return {
    maeNorm: norm / n,
    maeX: absX / n,
    maeY: absY / n,
    wNorm: norm > 0 ? sumD2 / norm : 0,
    sqrNorm: sumD2 > 0 ? sumD3 / sumD2 : 0,
    scrNorm: screen / n
  };
};

/** Euclidean error of the displayed red dot (npred) against the nine dots. */
export const scoreDisplayedProbe = (
  samples: ProbeFeatureSample[],
  targets: Record<string, { x: number; y: number }> = PROBE_TARGETS,
  aspect = 1294 / 620
): ProbeScore | null => {
  const points: ProbeScorePoint[] = [];
  for (const sample of samples) {
    const target = targets[sample.look];
    if (!target || sample.npredX == null || sample.npredY == null) continue;
    const dx = sample.npredX - target.x;
    const dy = sample.npredY - target.y;
    points.push({
      look: sample.look,
      predX: sample.npredX,
      predY: sample.npredY,
      targetX: target.x,
      targetY: target.y,
      err: Math.hypot(dx, dy)
    });
  }
  if (points.length < 2) return null;
  return { ...summarize(points, aspect), points };
};

export const formatProbeNorm = (label: string, score: ProbeScore): string =>
  `${label} norm ${score.maeNorm.toFixed(3)} (x ${score.maeX.toFixed(3)}, y ${score.maeY.toFixed(3)})`;
