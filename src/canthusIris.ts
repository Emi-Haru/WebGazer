import type { EyesData } from './facemesh';

/** Subject's left eye in this codebase: outer 263, inner 362, iris center 473. */
const LEFT = { outer: 263, inner: 362, iris: 473 };
/** Subject's right eye: outer 33, inner 133, iris center 468. */
const RIGHT = { outer: 33, inner: 133, iris: 468 };

export interface CornerIris {
  leftX: number;
  leftY: number;
  rightX: number;
  rightY: number;
  ply: number;
  pry: number;
}

const point2 = (mesh: [number, number, number][], index: number): [number, number] => {
  const point = mesh[index];
  return [point[0], point[1]];
};

/** Position inside the eye-corner frame. Matches the training measurement. */
export const canthusOffset = (
  point: [number, number],
  outer: [number, number],
  inner: [number, number]
): [number, number] => {
  const eyeX = inner[0] - outer[0];
  const eyeY = inner[1] - outer[1];
  const width2 = eyeX * eyeX + eyeY * eyeY;
  if (width2 < 1e-12) return [0, 0];
  const relX = point[0] - (inner[0] + outer[0]) / 2;
  const relY = point[1] - (inner[1] + outer[1]) / 2;
  const half = width2 / 2;
  const along = (relX * eyeX + relY * eyeY) / half;
  const vert = (relX * -eyeY + relY * eyeX) / half;
  return [along, vert];
};

/**
 * Mesh iris and dark-pixel pupil, both in the eye-corner frame.
 * Returns null when the frame has no iris landmarks or no pupil.
 */
export const cornerIrisFromEyes = (eyes: EyesData): CornerIris | null => {
  const mesh = eyes.facemesh;
  if (mesh.length <= LEFT.iris || !eyes.left.pupil || !eyes.right.pupil) return null;

  const leftOuter = point2(mesh, LEFT.outer);
  const leftInner = point2(mesh, LEFT.inner);
  const rightOuter = point2(mesh, RIGHT.outer);
  const rightInner = point2(mesh, RIGHT.inner);
  const [leftX, leftY] = canthusOffset(point2(mesh, LEFT.iris), leftOuter, leftInner);
  const [rightX, rightY] = canthusOffset(point2(mesh, RIGHT.iris), rightOuter, rightInner);
  const [, ply] = canthusOffset(
    [eyes.left.imagex + eyes.left.pupil[0], eyes.left.imagey + eyes.left.pupil[1]],
    leftOuter,
    leftInner
  );
  const [, pry] = canthusOffset(
    [eyes.right.imagex + eyes.right.pupil[0], eyes.right.imagey + eyes.right.pupil[1]],
    rightOuter,
    rightInner
  );
  if (![leftX, leftY, rightX, rightY, ply, pry].every(Number.isFinite)) return null;
  return { leftX, leftY, rightX, rightY, ply, pry };
};
