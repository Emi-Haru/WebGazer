import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from 'react';
import { captureVideoFrame, startCamera, stopCamera } from './capture';
import { downloadSessionZip, saveSessionToDirectory } from './exportSession';
import {
  buildStratifiedRandomGrid,
  formatSessionId,
  STRATIFIED_MIN_CELLS,
  type GridPlan
} from './grid';
import {
  clearSessionFromIdb,
  loadLatestSessionFromIdb,
  saveSessionToIdb
} from './sessionStore';
import { TestMode } from './TestMode';
import { type Sample } from './types';

type CamStatus = 'idle' | 'starting' | 'ready' | 'error';
type AppMode = 'collect' | 'test';

const CLICK_COOLDOWN_MS = 350;
const CURRENT_TARGET_COLOR = '#ff4d4d';
const TARGET_HIT_RADIUS_PX = 72;

function nextFilename (index: number): string {
  return `${String(index).padStart(6, '0')}.jpg`;
}

function clamp01 (v: number): number {
  return Math.min(1, Math.max(0, v));
}

function newStratifiedSeed (): number {
  return (Math.floor(Math.random() * 0xffffffff) ^ Date.now()) >>> 0 || 1;
}

function makeGrid (stratifiedSeed: number): GridPlan {
  return buildStratifiedRandomGrid(
    window.innerWidth,
    window.innerHeight,
    stratifiedSeed,
    STRATIFIED_MIN_CELLS
  );
}

export function App (): JSX.Element {
  const [appMode, setAppMode] = useState<AppMode>('collect');

  return (
    <>
      <div className={appMode === 'collect' ? undefined : 'mode-hidden'} aria-hidden={appMode !== 'collect'}>
        <CollectorApp onOpenTest={() => setAppMode('test')} />
      </div>
      {appMode === 'test' && <TestMode onBack={() => setAppMode('collect')} />}
    </>
  );
}

function CollectorApp ({
  onOpenTest
}: {
  onOpenTest: () => void;
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastCaptureAt = useRef(0);
  const samplesRef = useRef<Sample[]>([]);
  const hydratedRef = useRef(false);

  const [camStatus, setCamStatus] = useState<CamStatus>('idle');
  const [camError, setCamError] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [stratifiedSeed, setStratifiedSeed] = useState(newStratifiedSeed);
  const [subjectId, setSubjectId] = useState('me');
  const [sessionId, setSessionId] = useState(() => formatSessionId());
  const [samples, setSamples] = useState<Sample[]>([]);
  const [grid, setGrid] = useState<GridPlan>(() => makeGrid(stratifiedSeed));
  const [targetIndex, setTargetIndex] = useState(0);
  const [flash, setFlash] = useState(false);
  const [lastClick, setLastClick] = useState<{ x: number; y: number } | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  samplesRef.current = samples;

  const displayTargets = grid.targets;
  const currentTarget = displayTargets[targetIndex] ?? null;
  const latticeDone = targetIndex >= displayTargets.length;

  const gridSummary = useMemo(() => {
    const n = grid.cols * grid.rows;
    return (
      `${grid.cols}×${grid.rows} cells (${n} pts, ≥${STRATIFIED_MIN_CELLS}) · ` +
      `${Math.round(grid.cellW ?? grid.cellPx)}×${Math.round(grid.cellH ?? grid.cellPx)}px · ` +
      `edge-to-edge · random in each · viewport ${grid.viewportW}×${grid.viewportH}`
    );
  }, [grid]);

  // Restore last session from IndexedDB (survives refresh; Test no longer unmounts Collect).
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const latest = await loadLatestSessionFromIdb();
        if (cancelled || !latest || latest.samples.length === 0) {
          hydratedRef.current = true;
          return;
        }
        setSessionId(latest.sessionId);
        setSubjectId(latest.subjectId);
        setSamples(latest.samples);
        setStatusMsg(
          `Restored ${latest.samples.length} samples from browser storage (${latest.sessionId}).`
        );
      } catch {
        // ignore — fresh session
      } finally {
        if (!cancelled) hydratedRef.current = true;
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Persist after every change once hydrated (so Test / refresh cannot wipe the session).
  useEffect(() => {
    if (!hydratedRef.current) return;
    const handle = window.setTimeout(() => {
      void saveSessionToIdb(sessionId, subjectId, samples).catch(() => {
        setStatusMsg('Warning: could not auto-save samples to browser storage.');
      });
    }, 200);
    return () => window.clearTimeout(handle);
  }, [sessionId, subjectId, samples]);

  useEffect(() => {
    return () => {
      stopCamera(streamRef.current);
      streamRef.current = null;
      for (const s of samplesRef.current) {
        URL.revokeObjectURL(s.previewUrl);
      }
    };
  }, []);

  useEffect(() => {
    const syncViewport = (): void => {
      if (!recording) {
        setGrid(makeGrid(stratifiedSeed));
        setTargetIndex(0);
      }
    };
    window.addEventListener('resize', syncViewport);
    return () => window.removeEventListener('resize', syncViewport);
  }, [recording, stratifiedSeed]);

  // After the instruction panel unmounts, rebuild once layout has settled
  // (scrollbar / innerWidth can shift slightly).
  useEffect(() => {
    if (!recording) return;
    let cancelled = false;
    const outer = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (cancelled) return;
        const plan = makeGrid(stratifiedSeed);
        setGrid(plan);
        setTargetIndex((i) => Math.min(i, plan.targets.length));
      });
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(outer);
    };
  }, [recording, stratifiedSeed]);

  const enableCamera = useCallback(async () => {
    setCamStatus('starting');
    setCamError(null);
    try {
      if (!videoRef.current) throw new Error('Video element missing.');
      streamRef.current = await startCamera(videoRef.current);
      setCamStatus('ready');
    } catch (err) {
      setCamStatus('error');
      setCamError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const resetLatticeTargets = useCallback((nextSeed: number = stratifiedSeed) => {
    const plan = makeGrid(nextSeed);
    setGrid(plan);
    setTargetIndex(0);
    return plan;
  }, [stratifiedSeed]);

  const beginRecording = useCallback(() => {
    // Panel unmounts this frame; rebuild the grid from the live viewport.
    // A finished set, or a start from the first cell, draws new points.
    // Undo of an unfinished set keeps the same points.
    let seed = stratifiedSeed;
    if (targetIndex === 0 || latticeDone) {
      seed = newStratifiedSeed();
      setStratifiedSeed(seed);
    }
    const plan = makeGrid(seed);
    setGrid(plan);
    if (latticeDone) {
      setTargetIndex(0);
    } else {
      setTargetIndex((i) => Math.min(i, plan.targets.length));
    }
    setRecording(true);
    setStatusMsg('Look at the random target in each cell, then click it.');
  }, [stratifiedSeed, targetIndex, latticeDone]);

  const pauseRecording = useCallback(() => {
    setRecording(false);
    setStatusMsg('Back to the panel. Start recording draws a new set.');
  }, []);

  const undoLast = useCallback(() => {
    setSamples((prev) => {
      if (!prev.length) return prev;
      const next = prev.slice(0, -1);
      const removed = prev[prev.length - 1];
      URL.revokeObjectURL(removed.previewUrl);
      if (targetIndex > 0) {
        setTargetIndex((i) => Math.max(0, i - 1));
      }
      return next;
    });
    setStatusMsg('Removed last sample.');
  }, [targetIndex]);

  const clearAll = useCallback(() => {
    if (!samples.length) return;
    if (!window.confirm(`Delete all ${samples.length} samples in this session?`)) return;
    for (const s of samples) URL.revokeObjectURL(s.previewUrl);
    setSamples([]);
    void clearSessionFromIdb(sessionId);
    resetLatticeTargets();
    setStatusMsg('Session cleared.');
  }, [samples, resetLatticeTargets, sessionId]);

  const recordAt = useCallback(
    async (clientX: number, clientY: number) => {
      if (!recording) return;
      if (camStatus !== 'ready') {
        setStatusMsg('Camera not ready.');
        return;
      }
      if (!videoRef.current) {
        setStatusMsg('Camera preview missing.');
        return;
      }
      if (latticeDone) {
        setStatusMsg('Set complete. Done returns to the panel.');
        return;
      }

      const now = Date.now();
      if (now - lastCaptureAt.current < CLICK_COOLDOWN_MS) return;
      lastCaptureAt.current = now;

      const screen_w = window.innerWidth;
      const screen_h = window.innerHeight;
      if (screen_w <= 0 || screen_h <= 0) return;

      try {
        const imageBlob = await captureVideoFrame(videoRef.current);
        const index = samplesRef.current.length + 1;
        const filename = nextFilename(index);
        const previewUrl = URL.createObjectURL(imageBlob);

        const latticeMeta = currentTarget
          ? {
              grid_cols: grid.cols,
              grid_rows: grid.rows,
              grid_index: targetIndex,
              grid_row: Math.floor(targetIndex / grid.cols),
              grid_col: targetIndex % grid.cols
            }
          : {};

        const sample: Sample = {
          id: `${sessionId}-${filename}`,
          filename,
          imageBlob,
          previewUrl,
          label_x: clamp01(clientX / screen_w),
          label_y: clamp01(clientY / screen_h),
          client_x: clientX,
          client_y: clientY,
          screen_w,
          screen_h,
          timestamp: now,
          ...latticeMeta
        };

        setSamples((prev) => [...prev, sample]);
        setLastClick({ x: clientX, y: clientY });
        setFlash(true);
        window.setTimeout(() => setFlash(false), 120);

        setTargetIndex((i) => i + 1);
      } catch (err) {
        setStatusMsg(err instanceof Error ? err.message : String(err));
      }
    },
    [
      recording,
      camStatus,
      latticeDone,
      sessionId,
      targetIndex,
      currentTarget,
      grid.cols,
      grid.rows
    ]
  );

  const onSurfaceClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (!recording) return;
      if ((event.target as HTMLElement).closest('[data-hud]')) return;

      const x = event.clientX;
      const y = event.clientY;

      if (currentTarget) {
        const tx = currentTarget.nx * window.innerWidth;
        const ty = currentTarget.ny * window.innerHeight;
        const dist = Math.hypot(x - tx, y - ty);
        if (dist > TARGET_HIT_RADIUS_PX) {
          setStatusMsg('Click the highlighted target (look at it first).');
          return;
        }
      }

      void recordAt(x, y);
    },
    [recording, currentTarget, recordAt]
  );

  const exportZip = useCallback(async () => {
    if (!samples.length) {
      setStatusMsg('No samples to export yet.');
      return;
    }
    setBusy(true);
    try {
      await downloadSessionZip(samples, sessionId, subjectId.trim() || 'me');
      setStatusMsg(
        `Downloaded ${sessionId}.zip — unzip into training/data/personal/raw/`
      );
    } catch (err) {
      setStatusMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [samples, sessionId, subjectId]);

  const exportFolder = useCallback(async () => {
    if (!samples.length) {
      setStatusMsg('No samples to export yet.');
      return;
    }
    setBusy(true);
    try {
      await saveSessionToDirectory(samples, sessionId, subjectId.trim() || 'me');
      setStatusMsg(
        `Saved folder ${sessionId}. Place it under training/data/personal/raw/`
      );
    } catch (err) {
      setStatusMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [samples, sessionId, subjectId]);

  return (
    <div
      className={`app ${recording ? 'recording' : ''} ${flash ? 'flash' : ''}`}
      onClick={onSurfaceClick}
    >
      <div className="surface show-grid">
        {grid.originX != null &&
          grid.originY != null &&
          grid.cols > 0 &&
          grid.rows > 0 && (
          <svg className="grid-lines" aria-hidden="true">
            {Array.from({ length: grid.rows * grid.cols }, (_, i) => {
              const row = Math.floor(i / grid.cols);
              const col = i % grid.cols;
              const cellW = grid.cellW ?? grid.cellPx;
              const cellH = grid.cellH ?? grid.cellPx;
              const x = grid.originX! + col * cellW;
              const y = grid.originY! + row * cellH;
              return (
                <rect
                  key={`c${i}`}
                  className="grid-line"
                  x={x}
                  y={y}
                  width={cellW}
                  height={cellH}
                  fill="none"
                />
              );
            })}
          </svg>
        )}
        {displayTargets.map((t, i) => {
            const isCurrent = recording && i === targetIndex;
            const isDone = recording && i < targetIndex;
            return (
              <div
                key={t.id}
                className={
                  isCurrent
                    ? 'target current'
                    : isDone
                      ? 'target done'
                      : recording
                        ? 'target upcoming'
                        : 'target preview'
                }
                style={{
                  left: `${t.nx * 100}%`,
                  top: `${t.ny * 100}%`,
                  ...(isCurrent
                    ? ({
                        '--target': CURRENT_TARGET_COLOR,
                        '--target-glow': `${CURRENT_TARGET_COLOR}40`
                      } as CSSProperties)
                    : {})
                }}
              >
                {isCurrent ? (
                  <>
                    <span className="target-ring" />
                    <span className="target-core" />
                  </>
                ) : (
                  <span className="target-dot" />
                )}
              </div>
            );
          })}
        {lastClick && recording && (
          <div
            className="click-mark"
            style={{ left: lastClick.x, top: lastClick.y }}
          />
        )}
        {!recording && (
          <p className="surface-hint">
            {`Stratified preview — ${grid.cols}×${grid.rows} cells to the edges (≥${STRATIFIED_MIN_CELLS}), one random target each (${displayTargets.length} photos).`}
          </p>
        )}
        {recording && latticeDone && (
          <p className="surface-hint done">
            Set complete. Done returns to the panel so you can export or start a new set.
          </p>
        )}
      </div>

      {recording && latticeDone && (
        <div className="recording-hud" data-hud onClick={(e) => e.stopPropagation()}>
          <button type="button" className="hud-pause" onClick={pauseRecording}>
            Done
          </button>
        </div>
      )}

      <aside
        className={`panel ${recording ? 'is-recording-hidden' : ''}`}
        data-panel
        aria-hidden={recording}
        onClick={(e) => e.stopPropagation()}
      >
          <div className="panel-top">
            <div>
              <h1>Training collector</h1>
              <p className="sub">
                Labels use <code>innerWidth</code>×<code>innerHeight</code>. This
                panel hides while you record so the grid uses the real viewport.
              </p>
            </div>
          </div>

          <div className="mode-tabs">
            <button type="button" className="mode-tab active" disabled>
              Collect
            </button>
            <button
              type="button"
              className="mode-tab"
              disabled={recording}
              onClick={() => {
                stopCamera(streamRef.current);
                streamRef.current = null;
                setCamStatus('idle');
                setRecording(false);
                onOpenTest();
              }}
            >
              Test gaze
            </button>
          </div>

          <section>
            <h2>1. Camera</h2>
            <div className="video-wrap">
              <video ref={videoRef} playsInline muted autoPlay />
            </div>
            {camStatus !== 'ready' ? (
              <button type="button" className="primary" onClick={() => void enableCamera()} disabled={camStatus === 'starting'}>
                {camStatus === 'starting' ? 'Starting…' : 'Enable webcam'}
              </button>
            ) : (
              <p className="ok">Camera ready — keep your face visible in the preview.</p>
            )}
            {camError && <p className="error">{camError}</p>}
          </section>

          <section>
            <h2>2. Session</h2>
            <label className="field">
              <span>Subject id</span>
              <input
                value={subjectId}
                onChange={(e) => setSubjectId(e.target.value)}
              />
            </label>
            <p className="meta">Session: <code>{sessionId}</code></p>
            <p className="meta">Samples: <strong>{samples.length}</strong>
              {samples.length > 0 ? ' (auto-saved in this browser)' : ''}
            </p>
          </section>

          <section>
            <h2>3. Capture</h2>
            <p className="meta">{gridSummary}</p>
            <p className="hint">
              Equal-sized cells covering the viewport edge-to-edge (at least{' '}
              {STRATIFIED_MIN_CELLS}). One random target per cell. A new set is
              drawn when recording starts.
            </p>
          </section>

          <section className="actions">
            <button
              type="button"
              className="primary"
              disabled={camStatus !== 'ready'}
              onClick={beginRecording}
            >
              Start recording
            </button>
            <div className="row">
              <button type="button" className="secondary" onClick={undoLast} disabled={!samples.length}>
                Undo last
              </button>
              <button type="button" className="secondary" onClick={clearAll} disabled={!samples.length}>
                Clear all
              </button>
            </div>
          </section>

          <section>
            <h2>4. Export</h2>
              <p className="hint">
                Unzip / place the session folder under{' '}
                <code>training/data/personal/raw/</code>, then run extract + train.
              </p>
            <button
              type="button"
              className="primary"
              onClick={() => void exportZip()}
              disabled={!samples.length || busy}
            >
              Download ZIP
            </button>
            {typeof window !== 'undefined' && window.showDirectoryPicker && (
              <button
                type="button"
                className="secondary"
                onClick={() => void exportFolder()}
                disabled={!samples.length || busy}
              >
                Save to folder…
              </button>
            )}
          </section>

          {samples.length > 0 && (
            <section>
              <h2>Recent</h2>
              <div className="thumbs">
                {samples.slice(-8).reverse().map((s) => (
                  <figure
                    key={s.id}
                    title={`(${s.label_x.toFixed(2)}, ${s.label_y.toFixed(2)})`}
                  >
                    <img src={s.previewUrl} alt="" />
                    <figcaption>
                      {s.label_x.toFixed(2)}, {s.label_y.toFixed(2)}
                    </figcaption>
                  </figure>
                ))}
              </div>
            </section>
          )}

          {statusMsg && !recording && <p className="status" role="status">{statusMsg}</p>}
        </aside>
    </div>
  );
}
