/**
 * Built package: host app must serve the shipped file at this URL.
 * (Copy dist/models/gaze_corner_frame.bin → public/models/gaze_corner_frame.bin.)
 * Same weights and predictCorner as collect; only the fetch URL differs from Vite-dev.
 */
export const loadGazeModel = async (): Promise<unknown> => {
  const response = await fetch('/models/gaze_corner_frame.bin');
  if (!response.ok || !response.body) throw new Error(`gaze model HTTP ${response.status}`);
  const inflated = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(inflated).json();
};
