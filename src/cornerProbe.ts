import rawBundle from './models/gaze_corner_frame.json' with { type: 'json' };
import { cornerIrisFromEyes } from './canthusIris';
import type { EyesData } from './facemesh';
import { buildMlpFeatureVector, MLP_FEATURE_NAMES } from './mlpFeatures';
import { type ProbeFeatureSample } from './probeScore';

interface CornerTree {
  f: number[];
  th: number[];
  L: number[];
  R: number[];
  v: number[][];
}

interface EdgeStretch {
  dead_x: number;
  dead_y: number;
  gain_x: number;
  gain_y: number;
}

interface FrameBundle {
  method: string;
  feature_order: string[];
  aspect: number;
  edge_stretch?: EdgeStretch;
  trees: CornerTree[];
}

const BUNDLE = rawBundle as FrameBundle;

const frameValue = (sample: ProbeFeatureSample, name: string): number | null => {
  const corner = sample.corner;
  if (!corner) return null;
  if (name === 'm_lx') return corner.leftX;
  if (name === 'm_ly') return corner.leftY;
  if (name === 'm_rx') return corner.rightX;
  if (name === 'm_ry') return corner.rightY;
  if (name === 'p_ly') return corner.ply;
  if (name === 'p_ry') return corner.pry;
  const value = sample.features[name];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const frameRow = (sample: ProbeFeatureSample): number[] | null => {
  const row: number[] = [];
  for (const name of BUNDLE.feature_order) {
    const value = frameValue(sample, name);
    if (value == null) return null;
    row.push(value);
  }
  return row;
};

/** Push a prediction that is already well away from the center farther out. */
const stretchAxis = (value: number, dead: number, gain: number): number => {
  const offset = value - 0.5;
  const magnitude = Math.abs(offset);
  const extra = Math.max(magnitude - dead, 0) * (gain - 1);
  const stretched = 0.5 + Math.sign(offset) * (magnitude + extra);
  return Math.min(1, Math.max(0, stretched));
};

const forestPredict = (rows: number[][]): number[][] => {
  const total = rows.map(() => [0, 0]);
  for (const tree of BUNDLE.trees) {
    rows.forEach((row, index) => {
      let node = 0;
      while (tree.L[node] !== -1) {
        node = row[tree.f[node]] <= tree.th[node] ? tree.L[node] : tree.R[node];
      }
      total[index][0] += tree.v[node][0];
      total[index][1] += tree.v[node][1];
    });
  }
  const count = BUNDLE.trees.length;
  return total.map(([x, y]) => [x / count, y / count]);
};

/** One live frame in the same feature layout the probe uses. */
export const liveSampleFromEyes = (eyes: EyesData): ProbeFeatureSample | null => {
  const values = buildMlpFeatureVector(eyes);
  if (!values) return null;
  const features = Object.fromEntries(
    MLP_FEATURE_NAMES.map((name, index) => [name, values[index]])
  );
  const corner = cornerIrisFromEyes(eyes);
  if (!corner) return null;
  return { look: 'live', features, corner };
};

/**
 * Screen position for each frame on its own, normalized.
 * Eye-corner iris, head pose, and camera distance (eye separation and head depth).
 */
export const predictCorner = (
  samples: ProbeFeatureSample[]
): Array<{ x: number; y: number }> | null => {
  const rows: number[][] = [];
  for (const sample of samples) {
    const row = frameRow(sample);
    if (!row) return null;
    rows.push(row);
  }
  if (!rows.length) return null;
  const stretch = BUNDLE.edge_stretch;
  return forestPredict(rows).map(([x, y]) => {
    if (!stretch) return { x, y };
    return {
      x: stretchAxis(x, stretch.dead_x, stretch.gain_x),
      y: stretchAxis(y, stretch.dead_y, stretch.gain_y)
    };
  });
};

