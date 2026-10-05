import type { Sample } from './types';

const DB_NAME = 'webgazer-collect';
const DB_VERSION = 1;
const STORE = 'sessions';

type StoredSample = {
  id: string;
  filename: string;
  imageBlob: Blob;
  label_x: number;
  label_y: number;
  client_x: number;
  client_y: number;
  screen_w: number;
  screen_h: number;
  timestamp: number;
  grid_cols?: number;
  grid_rows?: number;
  grid_index?: number;
  grid_row?: number;
  grid_col?: number;
  /** Legacy — kept so older IndexedDB sessions still load. */
  head_pose?: string;
};

type StoredSession = {
  sessionId: string;
  subjectId: string;
  samples: StoredSample[];
  updatedAt: number;
};

function openDb (): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB open failed'));
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'sessionId' });
      }
    };
  });
}

function toStored (sample: Sample): StoredSample {
  return {
    id: sample.id,
    filename: sample.filename,
    imageBlob: sample.imageBlob,
    label_x: sample.label_x,
    label_y: sample.label_y,
    client_x: sample.client_x,
    client_y: sample.client_y,
    screen_w: sample.screen_w,
    screen_h: sample.screen_h,
    timestamp: sample.timestamp,
    grid_cols: sample.grid_cols,
    grid_rows: sample.grid_rows,
    grid_index: sample.grid_index,
    grid_row: sample.grid_row,
    grid_col: sample.grid_col
  };
}

function fromStored (sample: StoredSample): Sample {
  return {
    ...sample,
    previewUrl: URL.createObjectURL(sample.imageBlob)
  };
}

export async function saveSessionToIdb (
  sessionId: string,
  subjectId: string,
  samples: Sample[]
): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB write failed'));
    tx.objectStore(STORE).put({
      sessionId,
      subjectId,
      samples: samples.map(toStored),
      updatedAt: Date.now()
    } satisfies StoredSession);
  });
  db.close();
}

export async function loadSessionFromIdb (
  sessionId: string
): Promise<{ subjectId: string; samples: Sample[] } | null> {
  const db = await openDb();
  const row = await new Promise<StoredSession | undefined>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(sessionId);
    req.onsuccess = () => resolve(req.result as StoredSession | undefined);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB read failed'));
  });
  db.close();
  if (!row) return null;
  return {
    subjectId: row.subjectId,
    samples: row.samples.map(fromStored)
  };
}

/** Most recently updated session (for recovery after refresh / mode switch). */
export async function loadLatestSessionFromIdb (): Promise<{
  sessionId: string;
  subjectId: string;
  samples: Sample[];
} | null> {
  const sessions = await listSessionsFromIdb();
  if (!sessions.length) return null;
  return sessions[0];
}

/** All stored collect sessions, newest first. */
export async function listSessionsFromIdb (): Promise<Array<{
  sessionId: string;
  subjectId: string;
  samples: Sample[];
}>> {
  const db = await openDb();
  const rows = await new Promise<StoredSession[]>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve((req.result as StoredSession[]) ?? []);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB list failed'));
  });
  db.close();
  rows.sort((a, b) => b.updatedAt - a.updatedAt);
  return rows.map((row) => ({
    sessionId: row.sessionId,
    subjectId: row.subjectId,
    samples: row.samples.map(fromStored)
  }));
}

export async function clearSessionFromIdb (sessionId: string): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB delete failed'));
    tx.objectStore(STORE).delete(sessionId);
  });
  db.close();
}
