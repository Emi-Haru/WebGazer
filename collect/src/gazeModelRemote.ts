/** Pages build. The workflow splits the release file into these two static pieces. */
export const loadGazeModel = async (): Promise<unknown> => {
  const base = import.meta.env.BASE_URL;
  const [head, tail] = await Promise.all(['gaze-model-a', 'gaze-model-b'].map(async name => {
    const response = await fetch(`${base}${name}`);
    if (!response.ok) throw new Error(`gaze model HTTP ${response.status}`);
    return response.text();
  }));
  return JSON.parse(head + tail);
};
