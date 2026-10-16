/**
 * Dev and library builds read the model next to this module.
 * The bytes are gzip of gaze_corner_frame.json, named .bin so a server
 * does not decompress them before this code does.
 */
export const loadGazeModel = async (): Promise<unknown> => {
  const response = await fetch(new URL('./models/gaze_corner_frame.bin', import.meta.url));
  if (!response.ok || !response.body) throw new Error(`gaze model HTTP ${response.status}`);
  const inflated = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(inflated).json();
};
