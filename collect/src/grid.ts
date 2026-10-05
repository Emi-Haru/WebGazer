import type { TargetPoint } from './types';

export type GridInsetPx = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

export type GridPlan = {
  targets: TargetPoint[];
  cols: number;
  rows: number;
  /** Actual square cell spacing in CSS pixels (guided) or min(cellW, cellH). */
  cellPx: number;
  /** Stratified cell size in CSS px (may be slightly non-square to fill the viewport). */
  cellW?: number;
  cellH?: number;
  viewportW: number;
  viewportH: number;
  /** Top-left of the cell lattice in CSS px (stratified mode). */
  originX?: number;
  originY?: number;
  kind?: 'lattice' | 'stratified';
};

/** Minimum cells for stratified-random capture (≥50). */
export const STRATIFIED_MIN_CELLS = 50;

/** Keep the random target slightly inside each cell (fraction of cell size). */
const STRATIFIED_INSET = 0.08;

/**
 * No bezel margin — stratified cells must cover the real viewport edges.
 */
export function stratifiedEdgeInsetPx (
  _viewportW?: number,
  _viewportH?: number
): GridInsetPx {
  return { top: 0, right: 0, bottom: 0, left: 0 };
}

function mulberry32 (seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function usableArea (
  viewportW: number,
  viewportH: number,
  inset: GridInsetPx
): { vw: number; vh: number; left: number; top: number; usableW: number; usableH: number } {
  const vw = Math.max(1, viewportW);
  const vh = Math.max(1, viewportH);
  const left = Math.min(inset.left, vw * 0.35);
  const right = Math.min(inset.right, vw * 0.35);
  const top = Math.min(inset.top, vh * 0.35);
  const bottom = Math.min(inset.bottom, vh * 0.35);
  return {
    vw,
    vh,
    left,
    top,
    usableW: Math.max(1, vw - left - right),
    usableH: Math.max(1, vh - top - bottom)
  };
}

/**
 * Tile the full usable viewport (≥ minCells), then pick one uniform-random
 * point inside each cell. Cells fill edge-to-edge (near-square; exact squares
 * cannot fill a non-square screen without leaving a margin).
 * Seeded so resume/rebuild stays stable for the same viewport + seed.
 */
export function buildStratifiedRandomGrid (
  viewportW: number,
  viewportH: number,
  seed: number,
  minCells: number = STRATIFIED_MIN_CELLS,
  inset: GridInsetPx = stratifiedEdgeInsetPx(viewportW, viewportH)
): GridPlan {
  const { vw, vh, left, top, usableW, usableH } = usableArea(viewportW, viewportH, inset);
  const need = Math.max(1, Math.floor(minCells));

  // Near-square tiling that covers the whole usable rect.
  const ideal = Math.sqrt((usableW * usableH) / need);
  let cols = Math.max(1, Math.round(usableW / ideal));
  let rows = Math.max(1, Math.round(usableH / ideal));
  while (cols * rows < need) {
    const cellWIfCol = usableW / (cols + 1);
    const cellHIfRow = usableH / (rows + 1);
    const sqIfCol = Math.abs(cellWIfCol - usableH / rows);
    const sqIfRow = Math.abs(usableW / cols - cellHIfRow);
    if (sqIfCol <= sqIfRow) cols += 1;
    else rows += 1;
  }

  const cellW = usableW / cols;
  const cellH = usableH / rows;
  const originX = left;
  const originY = top;
  const rand = mulberry32(seed || 1);
  // Inner sides keep a small pad so clicks stay inside the cell; outer
  // sides of the grid go to 0 so targets can sit on the real screen edge.
  const pad = STRATIFIED_INSET;
  const minEdgePx = 2;

  const targets: TargetPoint[] = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x0 = originX + col * cellW;
      const y0 = originY + row * cellH;
      const padL = col === 0 ? 0 : pad;
      const padR = col === cols - 1 ? 0 : pad;
      const padT = row === 0 ? 0 : pad;
      const padB = row === rows - 1 ? 0 : pad;
      const xLo = x0 + padL * cellW;
      const xHi = x0 + cellW - padR * cellW;
      const yLo = y0 + padT * cellH;
      const yHi = y0 + cellH - padB * cellH;
      // Keep a couple CSS pixels inside the viewport so the target stays clickable.
      const x = Math.min(
        vw - minEdgePx,
        Math.max(minEdgePx, xLo + rand() * Math.max(1, xHi - xLo))
      );
      const y = Math.min(
        vh - minEdgePx,
        Math.max(minEdgePx, yLo + rand() * Math.max(1, yHi - yLo))
      );
      targets.push({
        id: `r${row}c${col}`,
        nx: x / vw,
        ny: y / vh
      });
    }
  }

  return {
    targets,
    cols,
    rows,
    cellPx: Math.min(cellW, cellH),
    cellW,
    cellH,
    viewportW: vw,
    viewportH: vh,
    originX,
    originY,
    kind: 'stratified'
  };
}

export function formatSessionId (date = new Date()): string {
  const pad = (v: number, w = 2) => String(v).padStart(w, '0');
  return (
    `session_${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_` +
    `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
  );
}
