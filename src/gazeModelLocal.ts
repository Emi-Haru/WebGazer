/**
 * Collect / Vite-dev: load the compressed model from src/models/ next to this file.
 * Vite must resolve this URL (no @vite-ignore) so the asset is served correctly.
 */
export const loadGazeModel = async (): Promise<unknown> => {
  const response = await fetch(new URL('./models/gaze_corner_frame.bin', import.meta.url));
  if (!response.ok || !response.body) throw new Error(`gaze model HTTP ${response.status}`);
  const inflated = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(inflated).json();
};
