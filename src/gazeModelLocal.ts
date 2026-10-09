/** Dev and library builds read the model file next to this module. */
export const loadGazeModel = (): Promise<unknown> =>
  fetch(new URL('./models/gaze_corner_frame.json', import.meta.url)).then(response => {
    if (!response.ok) throw new Error(`gaze model HTTP ${response.status}`);
    return response.json();
  });
