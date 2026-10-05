import { useCallback, useEffect, useRef, useState } from 'react';
import {
  WebGazer,
  type PredictionResult
} from 'webgazer';
import { COLLECT_VIDEO_CONSTRAINTS } from './capture';
import { DiagProbeOverlay, type DiagLookLabel } from './DiagProbeOverlay';
import { cornerIrisFromEyes } from '../../src/canthusIris';
import {
  probeTargetsForViewport,
  scoreDisplayedProbe,
  type ProbeFeatureSample,
  type ProbeScore
} from '../../src/probeScore';
import { DraggableCamPreview } from './DraggableCamPreview';
import { buildMlpFeatureVector, MLP_FEATURE_NAMES } from '../../src/mlpFeatures';

type Status = 'idle' | 'starting' | 'calibrating' | 'running' | 'error';

const READING_SAMPLE = `
The giraffe (Giraffa) is an African artiodactyl mammal, the tallest living terrestrial animal
and the largest ruminant. It is traditionally considered to be one species, Giraffa camelopardalis,
the tallest living terrestrial animals. Males can be 4.8 to 5.5 metres tall and weigh up to
1,900 kilograms; females are generally slightly shorter and lighter.

The giraffe's chief distinguishing characteristics are its extremely long neck and legs, its
horn-like ossicones, and its distinctive coat patterns. It is classified as one of seven species
in the genus Giraffa, but the number of recognised species varies.

Giraffes usually inhabit savannahs and woodlands. Their food source is leaves, fruits and flowers
of woody plants, primarily acacia species, which they browse at heights most other herbivores
cannot reach. They may be preyed on by lions, leopards, spotted hyenas and African wild dogs.
Giraffes live in herds of related females and their offspring, or bachelor herds of unrelated
adult males, but are gregarious and may gather in large aggregations.

When feeding, a giraffe can go for days without drinking water. It spreads its front legs or
bends its knees to reach the ground when drinking. The neck is used for a combat behaviour
called "necking", where bulls swing their necks at each other to establish dominance.

Try reading this paragraph slowly, then jump to a word in a far corner of the page, then back
to the middle. Watch whether the red gaze dot follows.
`.trim().replace(/\s+/g, ' ');

type Props = {
  onBack: () => void;
};

const probeScores = (samples: ProbeFeatureSample[]): ProbeScore | null => {
  const targets = probeTargetsForViewport(window.innerWidth, window.innerHeight);
  const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
  return scoreDisplayedProbe(samples, targets, aspect);
};

const dotScoreText = (score: ProbeScore | null): string =>
  score
    ? `red dot scrNorm ${score.scrNorm.toFixed(3)} (avg ${score.maeNorm.toFixed(3)})`
    : 'red dot position missing on a click';

/**
 * Leave-one-sitting scores for the red-dot forest, after the edge stretch.
 * Basic: every photo, against that photo's aim.
 * Simulated: the photo closest to each dot, scored against the dot,
 * the same way the live nine-dot test is scored.
 */
const TRAINING_ALL = {
  avgNorm: 0.124,
  wNorm: 0.171,
  sqrNorm: 0.221,
  scrNorm: 0.170
} as const;

const TRAINING_NINE_DOT = {
  avgNorm: 0.111,
  wNorm: 0.202,
  sqrNorm: 0.259,
  scrNorm: 0.148
} as const;

const SCORE_ROWS: { label: keyof typeof TRAINING_ALL; field: keyof ProbeScore }[] = [
  { label: 'avgNorm', field: 'maeNorm' },
  { label: 'wNorm', field: 'wNorm' },
  { label: 'sqrNorm', field: 'sqrNorm' },
  { label: 'scrNorm', field: 'scrNorm' }
];

const NineDotScoreTable = ({ score }: { score: ProbeScore }): JSX.Element => (
    <table className="score-table">
      <thead>
        <tr>
          <th>Norm</th>
          <th>Basic T.</th>
          <th>Simulated T.</th>
          <th>Real</th>
        </tr>
      </thead>
      <tbody>
        {SCORE_ROWS.map((row) => (
          <tr key={row.label}>
            <td>{row.label}</td>
            <td>{TRAINING_ALL[row.label].toFixed(3)}</td>
            <td>{TRAINING_NINE_DOT[row.label].toFixed(3)}</td>
            <td>{Number(score[row.field]).toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
);

export function TestMode ({ onBack }: Props): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const webgazerRef = useRef<WebGazer | null>(null);
  const lastPredAt = useRef(0);
  const lastPredRef = useRef<PredictionResult | null>(null);
  const calibAcceptedRef = useRef(0);
  const calibIgnoredRef = useRef(0);
  const calibInteractLockRef = useRef(false);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [panelOpen, setPanelOpen] = useState(true);
  const [faceLive, setFaceLive] = useState(false);
  type DiagSample = ProbeFeatureSample & {
    gazeX: number;
    gazeY: number;
  };
  const [diagSamples, setDiagSamples] = useState<DiagSample[]>([]);
  const diagSamplesRef = useRef<DiagSample[]>([]);
  const [diagMsg, setDiagMsg] = useState<string | null>(null);
  const [diagScore, setDiagScore] = useState<ProbeScore | null>(null);
  const [diagProbeOpen, setDiagProbeOpen] = useState(false);
  const [diagProbeKey, setDiagProbeKey] = useState(0);
  useEffect(() => {
    return () => {
      webgazerRef.current?.destroy();
      webgazerRef.current = null;
      document.documentElement.classList.remove('calibrating-lock');
    };
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (status === 'calibrating') root.classList.add('calibrating-lock');
    else root.classList.remove('calibrating-lock');
    return () => root.classList.remove('calibrating-lock');
  }, [status]);

  useEffect(() => {
    if (status !== 'running') {
      setFaceLive(false);
      return;
    }
    const id = window.setInterval(() => {
      setFaceLive(performance.now() - lastPredAt.current < 500);
    }, 100);
    return () => window.clearInterval(id);
  }, [status]);

  const stop = useCallback(() => {
    webgazerRef.current?.destroy();
    webgazerRef.current = null;
    setStatus('idle');
    setPrediction(null);
    lastPredRef.current = null;
    setError(null);
    setFaceLive(false);
    calibAcceptedRef.current = 0;
    calibIgnoredRef.current = 0;
    diagSamplesRef.current = [];
    setDiagSamples([]);
    setDiagMsg(null);
    setDiagProbeOpen(false);
  }, []);

  const recordDiagSample = useCallback((look: string): boolean => {
    const wg = webgazerRef.current;
    const eyes = wg?.lastEyePatches;
    const pred = prediction;
    if (!eyes || !pred) {
      setDiagMsg('No face / prediction yet — wait until gaze updates, then retry.');
      return false;
    }
    const feats = buildMlpFeatureVector(eyes);
    if (!feats) {
      setDiagMsg('Could not build MLP features from this frame.');
      return false;
    }
    const features = Object.fromEntries(
      MLP_FEATURE_NAMES.map((name, i) => [name, Number(feats[i].toFixed(4))])
    );
    const corner = cornerIrisFromEyes(eyes);
    const vw = Math.max(window.innerWidth, 1);
    const vh = Math.max(window.innerHeight, 1);
    const sample = {
      look,
      corner: corner ?? undefined,
      gazeX: Math.round(pred.x),
      gazeY: Math.round(pred.y),
      npredX: pred.x / vw,
      npredY: pred.y / vh,
      features
    };
    const next = [...diagSamplesRef.current, sample];
    diagSamplesRef.current = next;
    setDiagSamples(next);
    setDiagMsg(
      corner
        ? `Recorded “${look}”.`
        : `Recorded “${look}”. Corner iris missing on this frame.`
    );
    return true;
  }, [prediction]);

  const onDiagProbeClick = useCallback(
    (look: DiagLookLabel, _pointIndex: number): boolean => recordDiagSample(look),
    [recordDiagSample]
  );

  const startDiagProbe = useCallback(() => {
    diagSamplesRef.current = [];
    setDiagSamples([]);
    setDiagScore(null);
    setDiagMsg('9-point probe: look at each red dot, then click it once.');
    setDiagProbeKey((k) => k + 1);
    setDiagProbeOpen(true);
    setPanelOpen(false);
  }, []);

  // Gaze cursor pulls focus off the probe targets — hide while collecting samples.
  useEffect(() => {
    const wg = webgazerRef.current;
    if (!wg || status !== 'running') return;
    if (diagProbeOpen) {
      wg.hideGazeDot();
      wg.hideGazeTrail();
    } else {
      wg.showGazeDot();
      wg.showGazeTrail();
    }
  }, [diagProbeOpen, status]);
  const copyDiag = useCallback(async () => {
    if (!diagSamples.length) {
      setDiagMsg('No samples yet — start the 9-point probe.');
      return;
    }
    const scores = probeScores(diagSamples);
    const payload = {
      mode: 'corner',
      viewport: { w: window.innerWidth, h: window.innerHeight },
      samples: diagSamples,
      redDot: scores
    };
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    const cornerText = dotScoreText(scores);
    setDiagMsg(`Copied ${diagSamples.length} probe clicks. ${cornerText}.`);
  }, [diagSamples]);

  const start = useCallback(async () => {
    if (status === 'starting' || status === 'calibrating' || status === 'running') {
      return;
    }
    if (!videoRef.current) {
      setError('Camera preview not mounted — hard-refresh and try again.');
      setStatus('error');
      return;
    }

    setStatus('starting');
    setError(null);
    setPrediction(null);
    lastPredRef.current = null;
    calibAcceptedRef.current = 0;
    calibIgnoredRef.current = 0;

    try {
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview not mounted.');

      const wg = new WebGazer({
        saveDataAcrossSessions: false,
        useCalibration: false,
        useKalmanFilter: false,
        regression: 'mlpReg',
        cornerGaze: true,
        // Match collect JPEG resolution so ipd/face_w/z match personal training.
        videoConstraints: COLLECT_VIDEO_CONSTRAINTS
      });
      webgazerRef.current = wg;

      wg.addGazeListener((data: PredictionResult | undefined) => {
        if (data) {
          lastPredAt.current = performance.now();
          lastPredRef.current = data;
          setPrediction(data);
        }
      });

      await wg.start({
        gazeDot: true,
        gazeTrail: true,
        trackEye: 'both'
      });

      // Patch starts the tracking loop here (not in start()) — same as demo.
      await wg.showVideoFeedback(video, {
        mirrorVideo: true,
        faceFeedback: false
      });

      wg.setEyeRegionsVisible(false);
      wg.hideFaceMeshOverlay();
      wg.setIrisOverlayVisible(false);

      if (!video.videoWidth) {
        await new Promise<void>((resolve) => {
          video.addEventListener('loadeddata', () => resolve(), { once: true });
          window.setTimeout(() => resolve(), 2000);
        });
      }
      if (!video.videoWidth) {
        throw new Error('Webcam started but produced no frames. Check camera permissions / close Discord.');
      }

      const faceWaitStart = performance.now();
      while (performance.now() - faceWaitStart < 8000) {
        if (wg.lastEyePatches) break;
        await new Promise((r) => setTimeout(r, 200));
      }

      wg.stopCalibration();
      wg.showGazeDot();
      wg.showGazeTrail();
      setStatus('running');
      setPanelOpen(true);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : String(err));
      webgazerRef.current?.destroy();
      webgazerRef.current = null;
    }
  }, [status]);

  useEffect(() => {
    if (status !== 'calibrating') return;
    const onClick = (event: MouseEvent): void => {
      if (calibInteractLockRef.current) return;
      if ((event.target as HTMLElement | null)?.closest?.('[data-cam-preview]')) return;
      const wg = webgazerRef.current;
      if (!wg) return;
      if (wg.lastEyePatches) {
        calibAcceptedRef.current += 1;
      } else {
        calibIgnoredRef.current += 1;
      }
    };
    document.addEventListener('click', onClick, true);
    return () => document.removeEventListener('click', onClick, true);
  }, [status]);

  const onCamInteractionLock = useCallback((locked: boolean) => {
    calibInteractLockRef.current = locked;
  }, []);

  const handleBack = useCallback(() => {
    stop();
    onBack();
  }, [stop, onBack]);

  return (
    <div className={`test-mode ${status === 'calibrating' ? 'is-calibrating' : ''}`}>
      <div className="test-reading" aria-label="Reading test article">
        <article>
          <h1>Reading test</h1>
            <p className="lede">
            The nine-point score is the error of the red dot.
            No click calibration; prediction starts immediately.
          </p>
          <p>{READING_SAMPLE}</p>
          <p>{READING_SAMPLE}</p>
          <h2>Corner check</h2>
          <p>
            Look at the top-left of this page, then the bottom-right, without moving your
            head much. Expect “roughly the right region,” not pixel-perfect.
          </p>
        </article>
      </div>

      <DraggableCamPreview
        videoRef={videoRef}
        onInteractionLock={onCamInteractionLock}
      />

      {status === 'running' && diagProbeOpen && (
        <DiagProbeOverlay
          key={diagProbeKey}
          visible
          onPointClick={onDiagProbeClick}
          onComplete={() => {
            const scores = probeScores(diagSamplesRef.current);
            setDiagScore(scores);
            setDiagMsg(scores ? null : dotScoreText(scores));
            setDiagProbeOpen(false);
            setPanelOpen(true);
          }}
        />
      )}

      {panelOpen && (
        <aside className="test-panel" data-hud onClick={(e) => e.stopPropagation()}>
          <div className="panel-top">
            <div>
              <h1>Gaze test</h1>
              <p className="sub">
                Prediction starts immediately. No click calibration.
              </p>
            </div>
            <button type="button" className="icon-btn" onClick={() => setPanelOpen(false)} aria-label="Hide panel">
              ▾
            </button>
          </div>

          <section>
            <p className="meta">
              Status: <strong>{status}</strong>
              {status === 'running' && (
                <>
                  {' · '}
                  face:{' '}
                  <strong className={faceLive ? 'ok' : 'error'}>
                    {faceLive ? 'live' : 'lost / stalled'}
                  </strong>
                </>
              )}
              {prediction && (
                <>
                  {' · '}
                  gaze ({Math.round(prediction.x)}, {Math.round(prediction.y)})
                </>
              )}
            </p>
            {status === 'running' && !faceLive && (
              <p className="error">
                Gaze updates stopped — usually face not detected. Move into frame or restart.
              </p>
            )}
            {error && <p className="error">{error}</p>}

            {(status === 'idle' || status === 'error') && (
              <>
                <button type="button" className="primary" onClick={() => void start()}>
                  Start test
                </button>
              </>
            )}
            {status === 'starting' && (
              <>
                <p className="ok">
                  Starting camera + model… allow camera if prompted.
                </p>
                <button type="button" className="secondary" onClick={stop}>
                  Cancel
                </button>
              </>
            )}
            {status === 'running' && (
              <div className="row">
                <button type="button" className="secondary" onClick={stop}>
                  Stop
                </button>
              </div>
            )}
            {status === 'running' && (
              <div style={{ marginTop: '0.75rem' }}>
                <p className="hint">
                  <strong>9-point probe</strong>: look at each on-screen dot and click once. The
                  score is the error of the red dot.
                </p>
                <div className="row" style={{ flexWrap: 'wrap', gap: '0.35rem' }}>
                  <button type="button" className="primary" onClick={startDiagProbe}>
                    {diagProbeOpen ? 'Restart 9-point probe' : 'Start 9-point probe'}
                  </button>
                  {diagProbeOpen && (
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => setDiagProbeOpen(false)}
                    >
                      Hide probe
                    </button>
                  )}
                </div>
                <div className="row" style={{ marginTop: '0.35rem' }}>
                  <button type="button" className="secondary" onClick={() => void copyDiag()}>
                    Copy diagnostics
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => {
                      diagSamplesRef.current = [];
                      setDiagSamples([]);
                      setDiagScore(null);
                      setDiagMsg(null);
                      setDiagProbeKey((k) => k + 1);
                    }}
                  >
                    Clear
                  </button>
                </div>
                {diagScore && <NineDotScoreTable score={diagScore} />}
                {diagMsg && <p className="hint">{diagMsg}</p>}
              </div>
            )}
          </section>

          <section className="actions">
            <button type="button" className="secondary" onClick={handleBack}>
              ← Back to Collect
            </button>
          </section>
        </aside>
      )}

      {!panelOpen && status !== 'calibrating' && (
        <button
          type="button"
          className="test-show-panel"
          onClick={() => setPanelOpen(true)}
        >
          Show controls
        </button>
      )}

    </div>
  );
}
