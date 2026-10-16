import { loadGazeModel } from 'gaze-model-source';
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

interface RidgeHead {
  mean: number[];
  scale: number[];
  coef: number[][];
  intercept: number[];
  anchors?: Record<string, number[]>;
}

interface PoseHandoff {
  confidence_hi: number;
  confidence_lo: number;
  outside: number;
  outside_dead?: number;
  bound_index: number[];
  bounds: Record<string, { lo: number[], hi: number[] }>;
}

interface BoostTree {
  f: number[];
  th: number[];
  L: number[];
  R: number[];
  v: number[];
}

interface BoostHead {
  baseline: number;
  trees: BoostTree[];
}

interface MixModel {
  pose_index: number[];
  group_ridge: Record<string, RidgeHead>;
  chairs: RidgeHead[];
  router: { classes: string[], trees: CornerTree[] };
  weights: {
    tree_x: number
    ridge_x: number
    tree_y: number
    ridge_y: number
    chair_x: number
    chair_y: number
    expert?: number
  };
  group_forest?: Record<string, CornerTree[]>;
  global_ridge?: RidgeHead;
  handoff?: PoseHandoff;
  edge_forest?: CornerTree[];
  edge_mix?: number;
  boost?: { x: BoostHead, y: BoostHead, share: Record<string, number> };
  wide?: { x: BoostHead, y: BoostHead, share: Record<string, number> };
  stack?: { x: BoostHead, y: BoostHead, share: Record<string, number> };
  head_mix?: Record<string, { x: Record<string, number>, y: Record<string, number> }>;
  session_fix?: Record<string, SessionFix>;
  session_close?: Record<string, SessionFix>;
  ridge_pull?: Record<string, RidgePull>;
  scale_blend?: Record<string, ScaleBlend>;
  pool_blend?: Record<string, ScaleBlend>;
  gaze_offset?: Record<string, GazeOffset>;
  poly_blend?: Record<string, ScaleBlend>;
  cross_blend?: Record<string, ScaleBlend>;
  axis_gain?: Record<string, AxisGain>;
  tail_blend?: Record<string, ScaleBlend>;
}

interface AxisGain {
  x_gain?: number;
  x_dead?: number;
  y_gain?: number;
  y_dead?: number;
  far_x_gain?: number;
  far_x_dead?: number;
  far_y_gain?: number;
  far_y_dead?: number;
}

interface GazeOffset {
  frac?: number;
  close?: number[];
  far_frac?: number;
  far?: number[];
  axes?: number[];
}

interface RidgeRescue {
  group: string;
  head_ty_above?: number;
  head_ty_below?: number;
  head_roll_below?: number;
}

interface RidgePull {
  frac: number;
  close_only?: boolean;
  nudge?: { dx: number; x_above: number; dy: number };
  rescue?: RidgeRescue[];
}

interface ScaleBlend {
  coef?: number[][];
  intercept?: number[];
  frac: number;
  y_frac?: number;
  far_coef?: number[][];
  far_intercept?: number[];
  far_frac?: number;
  far_y_frac?: number;
  onto?: "close" | "far";
  columns?: number[];
}

interface SessionFix {
  mean: number[];
  scale: number[];
  coef: number[][];
  intercept: number[];
  mode: string;
  margin: number;
  axes: number[];
  dead: number;
  frac?: number;
}

interface FrameBundle {
  method: string;
  feature_order: string[];
  aspect: number;
  edge_stretch?: EdgeStretch;
  trees: CornerTree[];
  mix?: MixModel;
}

let BUNDLE: FrameBundle | null = null;
let loading: Promise<FrameBundle> | null = null;
const SESSION_KEEP = 80;
const sessionRows: Record<string, number[][]> = {};

const loadBundle = (): Promise<FrameBundle> => {
  if (BUNDLE) return Promise.resolve(BUNDLE);
  if (!loading) {
    loading = loadGazeModel().then(loaded => {
      BUNDLE = loaded as FrameBundle;
      return BUNDLE;
    });
  }
  return loading;
};

/** Resolves once the red-dot weights are in memory. */
export const cornerModelReady = (): Promise<void> => loadBundle().then(() => undefined);

const bundle = (): FrameBundle => {
  if (!BUNDLE) throw new Error('gaze model is not loaded');
  return BUNDLE;
};

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
  for (const name of bundle().feature_order) {
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

const expand = (row: number[]): number[] => {
  const iris = row.slice(0, 6);
  return row.concat(...[row[7], row[8], row[10]].map(factor => iris.map(value => value * factor)));
};

const columnMedian = (samples: number[][], index: number): number => {
  const values = samples.map(sample => sample[index]).sort((left, right) => left - right);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 === 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
};

const sessionQuad = (row: number[], history: number[][]): number[] => {
  const centered = row.map((value, index) => (
    index === 14 || index === 15 ? value : value - columnMedian(history, index)
  ));
  const pupil = centered.slice(0, 6);
  return centered.concat(
    pupil.map(value => value * value),
    pupil.map(value => value * centered[14])
  );
};

const walkSession = (fix: SessionFix, quad: number[]): [number, number] => {
  const scaled = quad.map((value, index) => {
    const scale = fix.scale[index];
    return scale === 0 ? 0 : (value - fix.mean[index]) / scale;
  });
  let x = fix.intercept[0];
  let y = fix.intercept[1];
  for (let index = 0; index < scaled.length; index++) {
    x += fix.coef[0][index] * scaled[index];
    y += fix.coef[1][index] * scaled[index];
  }
  return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
};

const gateSession = (fix: SessionFix, x: number, y: number, otherX: number, otherY: number): [number, number] => {
  const current = [x, y];
  const other = [otherX, otherY];
  const next = [x, y];
  for (const axis of fix.axes) {
    const further = Math.abs(other[axis] - 0.5) > Math.abs(current[axis] - 0.5) + fix.margin
      && Math.abs(other[axis] - 0.5) >= fix.dead
      && (fix.mode !== 'same' || Math.sign(current[axis] - 0.5) * Math.sign(other[axis] - 0.5) > 0);
    if (further) next[axis] = other[axis];
  }
  return [next[0], next[1]];
};

const nearerCloseAnchor = (group: string, row: number[]): boolean => {
  const chairs = bundle().mix?.chairs ?? [];
  const near = chairs[0]?.anchors?.[group];
  const far = chairs[1]?.anchors?.[group];
  if (!near || !far) return false;
  const closeDistance = (row[14] - near[0]) ** 2 + (row[15] - near[1]) ** 2;
  const farDistance = (row[14] - far[0]) ** 2 + (row[15] - far[1]) ** 2;
  return closeDistance <= farDistance;
};

/** Recenters this frame on the pose seen so far, then steps outward only when that fit reaches farther. */
const applySession = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const fix = bundle().mix?.session_fix?.[group];
  const closeFix = bundle().mix?.session_close?.[group];
  if (!fix && !closeFix) return { x, y };
  const history = sessionRows[group] ?? [];
  history.push(row.slice());
  if (history.length > SESSION_KEEP) history.shift();
  sessionRows[group] = history;
  if (history.length < 12) return { x, y };
  const quad = sessionQuad(row, history);
  let nextX = x;
  let nextY = y;
  if (fix) {
    const [otherX, otherY] = walkSession(fix, quad);
    [nextX, nextY] = gateSession(fix, nextX, nextY, otherX, otherY);
  }
  if (closeFix && nearerCloseAnchor(group, row)) {
    const [otherX, otherY] = walkSession(closeFix, quad);
    const [gatedX, gatedY] = gateSession(closeFix, nextX, nextY, otherX, otherY);
    const frac = closeFix.frac ?? 1;
    nextX = (1 - frac) * nextX + frac * gatedX;
    nextY = (1 - frac) * nextY + frac * gatedY;
  }
  return { x: nextX, y: nextY };
};

const HEAD_TY = 10;
const HEAD_ROLL = 8;

/** Pull the finished point toward a pose ridge. Rescue rules fire when the router picked a neighbor. */
const pullRidge = (
  group: string,
  row: number[],
  x: number,
  y: number,
  routedRidge: [number, number]
): { x: number, y: number } => {
  const pulls = bundle().mix?.ridge_pull;
  if (!pulls) return { x, y };
  const own = pulls[group];
  if (own && (!own.close_only || nearerCloseAnchor(group, row))) {
    const [ridgeX, ridgeY] = stretchPair(routedRidge[0], routedRidge[1]);
    let nextX = (1 - own.frac) * x + own.frac * ridgeX;
    let nextY = (1 - own.frac) * y + own.frac * ridgeY;
    if (own.nudge) {
      if (nextX > own.nudge.x_above) nextX += own.nudge.dx;
      nextY += own.nudge.dy;
    }
    return {
      x: Math.min(1, Math.max(0, nextX)),
      y: Math.min(1, Math.max(0, nextY))
    };
  }
  for (const [name, cfg] of Object.entries(pulls)) {
    const head = bundle().mix?.group_ridge[name];
    if (!head) continue;
    for (const rescue of cfg.rescue ?? []) {
      if (rescue.group !== group) continue;
      if (rescue.head_ty_above !== undefined && row[HEAD_TY] <= rescue.head_ty_above) continue;
      if (rescue.head_ty_below !== undefined && row[HEAD_TY] >= rescue.head_ty_below) continue;
      if (rescue.head_roll_below !== undefined && row[HEAD_ROLL] >= rescue.head_roll_below) continue;
      const [ridgeX, ridgeY] = stretchPair(...ridgePair(head, row));
      return {
        x: Math.min(1, Math.max(0, (1 - cfg.frac) * x + cfg.frac * ridgeX)),
        y: Math.min(1, Math.max(0, (1 - cfg.frac) * y + cfg.frac * ridgeY))
      };
    }
  }
  return { x, y };
};

const scaledBlend = (
  blend: ScaleBlend | undefined,
  group: string,
  row: number[],
  x: number,
  y: number
): { x: number, y: number } => {
  if (!blend) return { x, y };
  const near = nearerCloseAnchor(group, row);
  const coef = near ? blend.coef : blend.far_coef;
  const intercept = near ? blend.intercept : blend.far_intercept;
  const frac = near ? blend.frac : (blend.far_frac ?? 0);
  const yFrac = near ? (blend.y_frac ?? frac) : (blend.far_y_frac ?? frac);
  if (!coef || !intercept || (frac <= 0 && yFrac <= 0)) return { x, y };
  const history = sessionRows[group] ?? [];
  if (history.length < 12) return { x, y };
  const width = row.length;
  const mean = new Array<number>(width).fill(0);
  for (const sample of history) {
    for (let index = 0; index < width; index += 1) mean[index] += sample[index];
  }
  for (let index = 0; index < width; index += 1) mean[index] /= history.length;
  const scale = new Array<number>(width).fill(0);
  for (const sample of history) {
    for (let index = 0; index < width; index += 1) {
      const delta = sample[index] - mean[index];
      scale[index] += delta * delta;
    }
  }
  for (let index = 0; index < width; index += 1) scale[index] = Math.sqrt(scale[index] / history.length) || 1;
  const hat = intercept.slice();
  for (let axis = 0; axis < hat.length; axis += 1) {
    for (let index = 0; index < width; index += 1) {
      hat[axis] += coef[axis][index] * ((row[index] - mean[index]) / scale[index]);
    }
  }
  return {
    x: Math.min(1, Math.max(0, (1 - frac) * x + frac * hat[0])),
    y: Math.min(1, Math.max(0, (1 - yFrac) * y + yFrac * hat[1]))
  };
};

const scaleBlend = (group: string, row: number[], x: number, y: number): { x: number, y: number } =>
  scaledBlend(bundle().mix?.scale_blend?.[group], group, row, x, y);

const poolBlend = (group: string, row: number[], x: number, y: number): { x: number, y: number } =>
  scaledBlend(bundle().mix?.pool_blend?.[group], group, row, x, y);

const quadratic = (values: number[]): number[] => {
  const expanded = values.slice();
  for (let left = 0; left < values.length; left += 1) {
    for (let right = left; right < values.length; right += 1) expanded.push(values[left] * values[right]);
  }
  return expanded;
};

const crossBlend = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const blend = bundle().mix?.cross_blend?.[group];
  if (!blend?.coef || !blend.intercept) return { x, y };
  const ontoClose = (blend.onto ?? "close") === "close";
  if (ontoClose !== nearerCloseAnchor(group, row)) return { x, y };
  return scaledBlend(
    {
      coef: blend.coef,
      intercept: blend.intercept,
      frac: blend.frac,
      far_coef: blend.coef,
      far_intercept: blend.intercept,
      far_frac: blend.frac
    },
    group,
    row,
    x,
    y
  );
};

const polyBlend = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const blend = bundle().mix?.poly_blend?.[group];
  if (!blend) return { x, y };
  const near = nearerCloseAnchor(group, row);
  const coef = near ? blend.coef : blend.far_coef;
  const intercept = near ? blend.intercept : blend.far_intercept;
  const frac = near ? blend.frac : (blend.far_frac ?? 0);
  if (!coef || !intercept || frac <= 0) return { x, y };
  const history = sessionRows[group] ?? [];
  if (history.length < 12) return { x, y };
  const width = row.length;
  const mean = new Array<number>(width).fill(0);
  for (const sample of history) {
    for (let index = 0; index < width; index += 1) mean[index] += sample[index];
  }
  for (let index = 0; index < width; index += 1) mean[index] /= history.length;
  const scale = new Array<number>(width).fill(0);
  for (const sample of history) {
    for (let index = 0; index < width; index += 1) {
      const delta = sample[index] - mean[index];
      scale[index] += delta * delta;
    }
  }
  for (let index = 0; index < width; index += 1) scale[index] = Math.sqrt(scale[index] / history.length) || 1;
  const z = quadratic(row.map((value, index) => (value - mean[index]) / scale[index]));
  const hat = intercept.slice();
  for (let axis = 0; axis < hat.length; axis += 1) {
    for (let index = 0; index < z.length; index += 1) hat[axis] += coef[axis][index] * z[index];
  }
  return {
    x: Math.min(1, Math.max(0, (1 - frac) * x + frac * hat[0])),
    y: Math.min(1, Math.max(0, (1 - frac) * y + frac * hat[1]))
  };
};

const gainAxis = (value: number, gain: number, dead: number): number => {
  if (gain === 1) return value;
  const offset = value - 0.5;
  const magnitude = Math.abs(offset);
  const extra = Math.max(magnitude - dead, 0) * (gain - 1);
  const sign = offset === 0 ? 0 : Math.sign(offset);
  return Math.min(1, Math.max(0, 0.5 + sign * (magnitude + extra)));
};

/** Push one axis farther from center once the other blends have finished. */
const axisGain = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const cfg = bundle().mix?.axis_gain?.[group];
  if (!cfg) return { x, y };
  const near = nearerCloseAnchor(group, row);
  return {
    x: gainAxis(x, near ? (cfg.x_gain ?? 1) : (cfg.far_x_gain ?? 1), near ? (cfg.x_dead ?? 0) : (cfg.far_x_dead ?? 0)),
    y: gainAxis(y, near ? (cfg.y_gain ?? 1) : (cfg.far_y_gain ?? 1), near ? (cfg.y_dead ?? 0) : (cfg.far_y_dead ?? 0))
  };
};

const tailBlend = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const blend = bundle().mix?.tail_blend?.[group];
  if (!blend) return { x, y };
  const columns = blend.columns ?? row.map((_, index) => index);
  const sliced = columns.map(index => row[index]);
  const history = (sessionRows[group] ?? []).map(sample => columns.map(index => sample[index]));
  const saved = sessionRows[group];
  sessionRows[group] = history;
  const mixed = scaledBlend(blend, group, sliced, x, y);
  if (saved) sessionRows[group] = saved;
  else delete sessionRows[group];
  return mixed;
};

/**
 * On these tilts the finished point is high on one side of the screen and low
 * on the other. Subtract that slope. Leave-one-sitting medians from the
 * 2026-10-13 predictions; poses whose corners got worse are omitted.
 */
const TILT_SHEAR: Record<string, number> = {
  leftTilt: 0.086,
  rightTurnLeftTilt: 0.078,
  upLeftTilt: 0.051,
  downLeftTilt: 0.062,
  downRightTilt: -0.069
};

const tiltShear = (group: string, x: number, y: number): { x: number, y: number } => {
  const beta = TILT_SHEAR[group];
  if (!beta) return { x, y };
  return { x, y: Math.min(1, Math.max(0, y - beta * (x - 0.5))) };
};

const gazeOffset = (group: string, row: number[], x: number, y: number): { x: number, y: number } => {
  const cfg = bundle().mix?.gaze_offset?.[group];
  if (!cfg) return { x, y };
  const near = nearerCloseAnchor(group, row);
  const frac = near ? (cfg.frac ?? 0) : (cfg.far_frac ?? 0);
  const offset = near ? cfg.close : cfg.far;
  if (!offset || frac <= 0) return { x, y };
  const axes = cfg.axes ?? [0, 1];
  return {
    x: axes.includes(0) ? Math.min(1, Math.max(0, x + frac * offset[0])) : x,
    y: axes.includes(1) ? Math.min(1, Math.max(0, y + frac * offset[1])) : y
  };
};

const ridgePair = (head: RidgeHead, row: number[]): [number, number] => {
  const scaled = expand(row).map((value, index) => {
    const scale = head.scale[index];
    return scale === 0 ? 0 : (value - head.mean[index]) / scale;
  });
  let x = head.intercept[0];
  let y = head.intercept[1];
  for (let index = 0; index < scaled.length; index++) {
    x += head.coef[0][index] * scaled[index];
    y += head.coef[1][index] * scaled[index];
  }
  return [x, y];
};

const stretchPair = (x: number, y: number): [number, number] => {
  const stretch = bundle().edge_stretch;
  if (!stretch) return [x, y];
  return [
    stretchAxis(x, stretch.dead_x, stretch.gain_x),
    stretchAxis(y, stretch.dead_y, stretch.gain_y)
  ];
};

const leafScores = (tree: CornerTree, row: number[]): number[] => {
  let node = 0;
  while (tree.L[node] !== -1) {
    node = row[tree.f[node]] <= tree.th[node] ? tree.L[node] : tree.R[node];
  }
  return tree.v[node];
};

const routePose = (row: number[]): { group: string, confidence: number } => {
  const mix = bundle().mix;
  if (!mix) return { group: 'center', confidence: 1 };
  const pose = mix.pose_index.map(index => row[index]);
  const totals = mix.router.classes.map(() => 0);
  for (const tree of mix.router.trees) {
    const scores = leafScores(tree, pose);
    const sum = scores.reduce((total, score) => total + score, 0) || 1;
    scores.forEach((score, index) => {
      totals[index] += score / sum;
    });
  }
  const total = totals.reduce((sum, score) => sum + score, 0) || 1;
  let best = 0;
  for (let index = 1; index < totals.length; index++) {
    if (totals[index] > totals[best]) best = index;
  }
  return { group: mix.router.classes[best], confidence: totals[best] / total };
};

/** How far this head sits outside the faces recorded for that direction, in widths of that recording. */
const poseExcess = (row: number[], group: string): number => {
  const handoff = bundle().mix?.handoff;
  const box = handoff?.bounds[group];
  if (!handoff || !box) return 0;
  let excess = 0;
  handoff.bound_index.forEach((feature, index) => {
    const span = Math.max(box.hi[index] - box.lo[index], 1e-6);
    const value = row[feature];
    const over = Math.max(box.lo[index] - value, 0) + Math.max(value - box.hi[index], 0);
    excess = Math.max(excess, over / span);
  });
  return excess;
};

const pickChair = (row: number[], group: string): RidgeHead => {
  const chairs = bundle().mix?.chairs ?? [];
  let best = chairs[0];
  let bestDistance = Infinity;
  for (const chair of chairs) {
    const anchor = chair.anchors?.[group];
    if (!anchor) continue;
    const distance = (row[14] - anchor[0]) ** 2 + (row[15] - anchor[1]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = chair;
    }
  }
  return best;
};

const walkTree = (tree: CornerTree, row: number[]): number[] => {
  let node = 0;
  while (tree.L[node] !== -1) {
    node = row[tree.f[node]] <= tree.th[node] ? tree.L[node] : tree.R[node];
  }
  return tree.v[node];
};

const walkBoost = (head: BoostHead, row: number[]): number => {
  let total = head.baseline;
  for (const tree of head.trees) {
    let node = 0;
    while (tree.L[node] !== -1) {
      node = row[tree.f[node]] <= tree.th[node] ? tree.L[node] : tree.R[node];
    }
    total += tree.v[node];
  }
  return total;
};

const walkForest = (trees: CornerTree[], row: number[]): [number, number] => {
  let x = 0;
  let y = 0;
  for (const tree of trees) {
    const value = walkTree(tree, row);
    x += value[0];
    y += value[1];
  }
  const count = trees.length || 1;
  return [x / count, y / count];
};

const forestPredict = (rows: number[][]): number[][] => {
  const total = rows.map(() => [0, 0]);
  for (const tree of bundle().trees) {
    rows.forEach((row, index) => {
      let node = 0;
      while (tree.L[node] !== -1) {
        node = row[tree.f[node]] <= tree.th[node] ? tree.L[node] : tree.R[node];
      }
      total[index][0] += tree.v[node][0];
      total[index][1] += tree.v[node][1];
    });
  }
  const count = bundle().trees.length;
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
  if (!BUNDLE) {
    void loadBundle();
    return null;
  }
  const rows: number[][] = [];
  for (const sample of samples) {
    const row = frameRow(sample);
    if (!row) return null;
    rows.push(row);
  }
  if (!rows.length) return null;
  const forest = forestPredict(rows);
  const mix = bundle().mix;
  if (!mix) {
    return forest.map(([x, y]) => {
      const [sx, sy] = stretchPair(x, y);
      return { x: sx, y: sy };
    });
  }
  return rows.map((row, index) => {
    const { group, confidence } = routePose(row);
    const groupHead = mix.group_ridge[group] ?? mix.group_ridge.center;
    const groupPred = ridgePair(groupHead, row);
    const expertTrees = mix.group_forest?.[group];
    const expertShare = expertTrees ? (mix.weights.expert ?? 0) : 0;
    const expert = expertTrees ? walkForest(expertTrees, row) : forest[index];
    const treeX = (1 - expertShare) * forest[index][0] + expertShare * expert[0];
    const treeY = (1 - expertShare) * forest[index][1] + expertShare * expert[1];
    const blendedX = mix.weights.tree_x * treeX + mix.weights.ridge_x * groupPred[0];
    const blendedY = mix.weights.tree_y * treeY + mix.weights.ridge_y * groupPred[1];
    const step = stretchPair(blendedX, blendedY);
    const chairPred = stretchPair(...ridgePair(pickChair(row, group), row));
    const local = {
      x: Math.min(1, Math.max(0, (1 - mix.weights.chair_x) * step[0] + mix.weights.chair_x * chairPred[0])),
      y: Math.min(1, Math.max(0, (1 - mix.weights.chair_y) * step[1] + mix.weights.chair_y * chairPred[1]))
    };
    const handoff = mix.handoff;
    const globalHead = mix.global_ridge;
    let x = local.x;
    let y = local.y;
    if (handoff && globalHead) {
      const span = handoff.confidence_hi - handoff.confidence_lo;
      const fromConfidence = span <= 0 ? 0 : Math.min(1, Math.max(0, (handoff.confidence_hi - confidence) / span));
      const fromRange = Math.min(1, Math.max(0, (poseExcess(row, group) - (handoff.outside_dead ?? 0)) / handoff.outside));
      const alpha = Math.max(fromConfidence, fromRange);
      const [globalX, globalY] = stretchPair(...ridgePair(globalHead, row));
      x = (1 - alpha) * local.x + alpha * globalX;
      y = (1 - alpha) * local.y + alpha * globalY;
    }
    const edgeMix = mix.edge_mix ?? 0;
    let edgePair: [number, number] | null = null;
    if (mix.edge_forest) {
      const edge = walkForest(mix.edge_forest, row);
      edgePair = stretchPair(edge[0], edge[1]);
      if (edgeMix) {
        x = Math.min(1, Math.max(0, (1 - edgeMix) * x + edgeMix * edgePair[0]));
        y = Math.min(1, Math.max(0, (1 - edgeMix) * y + edgeMix * edgePair[1]));
      }
    }
    const headMix = mix.head_mix?.[group];
    if (headMix) {
      const bank: Record<string, [number, number]> = {
        mixed: [x, y],
        forest: stretchPair(forest[index][0], forest[index][1]),
        expert: stretchPair(expert[0], expert[1]),
        ridge: stretchPair(groupPred[0], groupPred[1])
      };
      if (edgePair) bank.edge = edgePair;
      if (mix.boost) {
        bank.edgew = stretchPair(walkBoost(mix.boost.x, row), walkBoost(mix.boost.y, row));
      }
      if (mix.wide) {
        const expanded = expand(row);
        bank.wide = stretchPair(walkBoost(mix.wide.x, expanded), walkBoost(mix.wide.y, expanded));
      }
      let mixedX = 0;
      let mixedY = 0;
      for (const [name, pair] of Object.entries(bank)) {
        mixedX += (headMix.x[name] ?? 0) * pair[0];
        mixedY += (headMix.y[name] ?? 0) * pair[1];
      }
      const session = applySession(
        group,
        row,
        Math.min(1, Math.max(0, mixedX)),
        Math.min(1, Math.max(0, mixedY))
      );
      const pulled = pullRidge(group, row, session.x, session.y, groupPred);
      const scaled = scaleBlend(group, row, pulled.x, pulled.y);
      const pooled = poolBlend(group, row, scaled.x, scaled.y);
      const shifted = gazeOffset(group, row, pooled.x, pooled.y);
      const curved = polyBlend(group, row, shifted.x, shifted.y);
      const crossed = crossBlend(group, row, curved.x, curved.y);
      const gained = axisGain(group, row, crossed.x, crossed.y);
      const tailed = tailBlend(group, row, gained.x, gained.y);
      return tiltShear(group, tailed.x, tailed.y);
    }
    const boost = mix.boost;
    const share = boost?.share[group] ?? 0;
    if (boost && share) {
      const stretched = stretchPair(walkBoost(boost.x, row), walkBoost(boost.y, row));
      x = Math.min(1, Math.max(0, (1 - share) * x + share * stretched[0]));
      y = Math.min(1, Math.max(0, (1 - share) * y + share * stretched[1]));
    }
    const beforeWideX = x;
    const beforeWideY = y;
    const wide = mix.wide;
    const wideShare = wide?.share[group] ?? 0;
    if (wide && wideShare) {
      const expanded = expand(row);
      const stretched = stretchPair(walkBoost(wide.x, expanded), walkBoost(wide.y, expanded));
      x = Math.min(1, Math.max(0, (1 - wideShare) * x + wideShare * stretched[0]));
      y = Math.min(1, Math.max(0, (1 - wideShare) * y + wideShare * stretched[1]));
    }
    const stack = mix.stack;
    const stackShare = stack?.share[group] ?? 0;
    if (stack && stackShare) {
      const stacked = row.concat(beforeWideX, beforeWideY);
      const stretched = stretchPair(walkBoost(stack.x, stacked), walkBoost(stack.y, stacked));
      x = Math.min(1, Math.max(0, (1 - stackShare) * x + stackShare * stretched[0]));
      y = Math.min(1, Math.max(0, (1 - stackShare) * y + stackShare * stretched[1]));
    }
    const session = applySession(group, row, x, y);
    const pulled = pullRidge(group, row, session.x, session.y, groupPred);
    const scaled = scaleBlend(group, row, pulled.x, pulled.y);
    const pooled = poolBlend(group, row, scaled.x, scaled.y);
    const shifted = gazeOffset(group, row, pooled.x, pooled.y);
    const curved = polyBlend(group, row, shifted.x, shifted.y);
    const crossed = crossBlend(group, row, curved.x, curved.y);
    const gained = axisGain(group, row, crossed.x, crossed.y);
    const tailed = tailBlend(group, row, gained.x, gained.y);
    return tiltShear(group, tailed.x, tailed.y);
  });
};

