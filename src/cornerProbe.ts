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
    if (mix.edge_forest && edgeMix) {
      const edge = walkForest(mix.edge_forest, row);
      const stretched = stretchPair(edge[0], edge[1]);
      x = Math.min(1, Math.max(0, (1 - edgeMix) * x + edgeMix * stretched[0]));
      y = Math.min(1, Math.max(0, (1 - edgeMix) * y + edgeMix * stretched[1]));
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
    return { x, y };
  });
};

