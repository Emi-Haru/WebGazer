/** One saved training sample (face image + click label). */
export type Sample = {
  id: string;
  /** Zero-padded filename without folder, e.g. 000001.jpg */
  filename: string;
  imageBlob: Blob;
  /** Preview URL (object URL) for the UI */
  previewUrl: string;
  /** Normalized click in viewport [0, 1] */
  label_x: number;
  label_y: number;
  client_x: number;
  client_y: number;
  screen_w: number;
  screen_h: number;
  timestamp: number;
  /** Which cell this click came from (labels only; files stay sequential). */
  grid_cols?: number;
  grid_rows?: number;
  grid_index?: number;
  grid_row?: number;
  grid_col?: number;
  /** Legacy field from older sessions; ignored by training features. */
  head_pose?: string;
};

/** Manifest line written next to images/ — matches training personal loader. */
export type ManifestRow = {
  image: string;
  label_x: number;
  label_y: number;
  label_space: 'normalized_viewport';
  screen_w: number;
  screen_h: number;
  client_x: number;
  client_y: number;
  subject_id: string;
  timestamp: number;
  grid_cols?: number;
  grid_rows?: number;
  grid_index?: number;
  grid_row?: number;
  grid_col?: number;
};

export type TargetPoint = {
  id: string;
  /** Normalized [0, 1] position on the click surface */
  nx: number;
  ny: number;
};
