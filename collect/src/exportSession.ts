import type { ManifestRow, Sample } from './types';
import { blobToUint8Array, buildZip, downloadBlob } from './zip';

export function sampleToManifestRow (sample: Sample, subjectId: string): ManifestRow {
  return {
    image: `images/${sample.filename}`,
    label_x: sample.label_x,
    label_y: sample.label_y,
    label_space: 'normalized_viewport',
    screen_w: sample.screen_w,
    screen_h: sample.screen_h,
    client_x: sample.client_x,
    client_y: sample.client_y,
    subject_id: subjectId,
    timestamp: sample.timestamp,
    grid_cols: sample.grid_cols,
    grid_rows: sample.grid_rows,
    grid_index: sample.grid_index,
    grid_row: sample.grid_row,
    grid_col: sample.grid_col
  };
}

export function buildManifestText (samples: Sample[], subjectId: string): string {
  return samples
    .map((s) => JSON.stringify(sampleToManifestRow(s, subjectId)))
    .join('\n') + (samples.length ? '\n' : '');
}

/** Write session into a user-picked folder (Chrome / Edge). */
export async function saveSessionToDirectory (
  samples: Sample[],
  sessionId: string,
  subjectId: string
): Promise<void> {
  if (!window.showDirectoryPicker) {
    throw new Error('Folder save is not supported in this browser. Use Download ZIP instead.');
  }
  const root = await window.showDirectoryPicker({ mode: 'readwrite' });
  const sessionDir = await root.getDirectoryHandle(sessionId, { create: true });
  const imagesDir = await sessionDir.getDirectoryHandle('images', { create: true });

  for (const sample of samples) {
    const handle = await imagesDir.getFileHandle(sample.filename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(sample.imageBlob);
    await writable.close();
  }

  const manifestHandle = await sessionDir.getFileHandle('manifest.jsonl', { create: true });
  const manifestWritable = await manifestHandle.createWritable();
  await manifestWritable.write(buildManifestText(samples, subjectId));
  await manifestWritable.close();

  const metaHandle = await sessionDir.getFileHandle('session.json', { create: true });
  const metaWritable = await metaHandle.createWritable();
  await metaWritable.write(
    JSON.stringify(
      {
        session_id: sessionId,
        subject_id: subjectId,
        n_samples: samples.length,
        created_at: new Date().toISOString(),
        format: 'webgazer-personal-v1'
      },
      null,
      2
    )
  );
  await metaWritable.close();
}

/** Download a ZIP: sessionId/images/000001.jpg + manifest.jsonl + session.json */
export async function downloadSessionZip (
  samples: Sample[],
  sessionId: string,
  subjectId: string
): Promise<void> {
  const entries = [];
  for (const sample of samples) {
    entries.push({
      path: `${sessionId}/images/${sample.filename}`,
      data: await blobToUint8Array(sample.imageBlob)
    });
  }
  const manifest = new TextEncoder().encode(buildManifestText(samples, subjectId));
  entries.push({ path: `${sessionId}/manifest.jsonl`, data: manifest });
  const meta = new TextEncoder().encode(
    JSON.stringify(
      {
        session_id: sessionId,
        subject_id: subjectId,
        n_samples: samples.length,
        created_at: new Date().toISOString(),
        format: 'webgazer-personal-v1'
      },
      null,
      2
    )
  );
  entries.push({ path: `${sessionId}/session.json`, data: meta });

  const zip = buildZip(entries);
  downloadBlob(zip, `${sessionId}.zip`);
}
