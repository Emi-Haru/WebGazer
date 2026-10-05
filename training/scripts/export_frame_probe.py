#!/usr/bin/env python3
"""Train the red-dot forest on the same face reading the page uses.

The collect page runs MediaPipe Face Landmarker in VIDEO mode. That tracker
settles, over a few frames, to a different iris position than a still-image
mesh. This script feeds each training photo through that same tracker until
it settles, then fits the per-frame forest the red dot walks.

Expects the twelve center sittings already extracted to
training/data/processed/extracted.parquet, with the JPEG paths in that
table still pointing at the session folders under training/data/personal/raw/.
training/models/face_landmarker.task has to be present too. The .npz beside
this script is only a cache of those settled features; a missing cache is
rebuilt from the photos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    EYE_ARCS,
    eye_aspect_ratio,
    eye_patch_bounds_from_landmarks,
    scale_proxies,
)
from head_pose import estimate_head_pose  # noqa: E402
from pupil import detect_pupil_center, eyelid_polygon_from_arcs, fill_polygon_mask  # noqa: E402

def load_center_frames(parquet_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path)
    frame = frame[frame.dataset == "personal"].copy()
    frame["folder"] = frame.image_path.map(lambda p: Path(str(p)).parent.parent.name)
    frame = frame[frame.folder.str.contains("20260929_center")].reset_index(drop=True)
    frame["n"] = frame.folder.str.extract(r"center(\d+)$")[0].astype(int)
    return frame

MODEL = ROOT / "models" / "face_landmarker.task"
CACHE = ROOT / "logs" / "video_settled_features.npz"
OUT = ROOT.parent / "src" / "models" / "gaze_corner_frame.json"
ASPECT = 1294 / 620
N_TREES = 200
LEAF = 2
SETTLE_STEPS = 16
STEP_MS = 33
FEATURE_ORDER = [
    "m_lx",
    "m_ly",
    "m_rx",
    "m_ry",
    "p_ly",
    "p_ry",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_tx",
    "head_ty",
    "head_tz",
    "ear_left",
    "ear_right",
    "ipd_px",
    "face_width_px",
]
# Subject's left, then subject's right. Matches src/canthusIris.ts.
EYES = {
    "left": (263, 362, 473),
    "right": (33, 133, 468),
}
TARGETS = [(x, y) for y in (0.0177, 0.5, 0.9823) for x in (0.0085, 0.5, 0.9915)]


def canthus_offset(point: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> tuple[float, float]:
    eye = inner - outer
    width2 = float(np.dot(eye, eye))
    if width2 < 1e-12:
        return 0.0, 0.0
    rel = point - (inner + outer) / 2
    half = width2 / 2
    along = float(np.dot(rel, eye) / half)
    vert = float(np.dot(rel, np.array([-eye[1], eye[0]])) / half)
    return along, vert


def pupil_abs(image_bgr: np.ndarray, landmarks: np.ndarray, side: str) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    eye = eye_patch_bounds_from_landmarks(landmarks, side, width, height)
    if eye is None:
        return None
    arcs = EYE_ARCS[side]
    upper = np.array([landmarks[i, :2] for i in arcs["upper"]], dtype=np.float64)
    lower = np.array([landmarks[i, :2] for i in arcs["lower"]], dtype=np.float64)
    poly = eyelid_polygon_from_arcs(upper, lower)
    poly[:, 0] -= eye.x
    poly[:, 1] -= eye.y
    mask = fill_polygon_mask(eye.width, eye.height, poly)
    patch = image_bgr[eye.y : eye.y + eye.height, eye.x : eye.x + eye.width]
    local = detect_pupil_center(patch, mask)
    return np.array([eye.x + local[0], eye.y + local[1]], dtype=np.float64)


def row_from_landmarks(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    pose = estimate_head_pose(landmarks, width, height)
    if pose is None:
        return None
    values: dict[str, float] = {}
    for side, (outer, inner, iris) in EYES.items():
        mesh_x, mesh_y = canthus_offset(landmarks[iris, :2], landmarks[outer, :2], landmarks[inner, :2])
        pupil = pupil_abs(image_bgr, landmarks, side)
        if pupil is None:
            return None
        _, pupil_y = canthus_offset(pupil, landmarks[outer, :2], landmarks[inner, :2])
        if side == "left":
            values["m_lx"] = mesh_x
            values["m_ly"] = mesh_y
            values["p_ly"] = pupil_y
        else:
            values["m_rx"] = mesh_x
            values["m_ry"] = mesh_y
            values["p_ry"] = pupil_y
    yaw, pitch, roll, tx, ty, tz = pose
    ipd, face_w = scale_proxies(landmarks)
    values.update(
        {
            "head_yaw": yaw,
            "head_pitch": pitch,
            "head_roll": roll,
            "head_tx": tx,
            "head_ty": ty,
            "head_tz": tz,
            "ear_left": eye_aspect_ratio(landmarks, "left"),
            "ear_right": eye_aspect_ratio(landmarks, "right"),
            "ipd_px": ipd,
            "face_width_px": face_w,
        }
    )
    row = np.array([values[name] for name in FEATURE_ORDER], dtype=np.float64)
    if not np.isfinite(row).all():
        return None
    return row


def landmarks_of(result, width: int, height: int) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    points = result.face_landmarks[0]
    if len(points) <= 473:
        return None
    out = np.zeros((len(points), 3), dtype=np.float64)
    for index, point in enumerate(points):
        out[index, 0] = point.x * width
        out[index, 1] = point.y * height
        out[index, 2] = point.z * width
    return out


def extract(frame) -> np.ndarray:
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    rows = np.full((len(frame), len(FEATURE_ORDER)), np.nan)
    clock = 0
    for index, path in enumerate(frame.image_path):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        settled = None
        for _ in range(SETTLE_STEPS):
            clock += STEP_MS
            detected = landmarker.detect_for_video(mp_image, clock)
            found = landmarks_of(detected, width, height)
            if found is not None:
                settled = found
        if settled is None:
            continue
        row = row_from_landmarks(image, settled)
        if row is not None:
            rows[index] = row
            if str(path).endswith("center1/images/000025.jpg"):
                print(f"center1/000025 m_ly {row[FEATURE_ORDER.index('m_ly')]:.4f}", flush=True)
        if (index + 1) % 50 == 0:
            print(f"extracted {index + 1}/{len(frame)}", flush=True)
    landmarker.close()
    return rows


def metrics(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    err = pred - labels
    dist = np.hypot(err[:, 0], err[:, 1])
    screen = np.hypot(err[:, 0] * ASPECT, err[:, 1])
    return (
        float(dist.mean()),
        float(np.sum(dist * dist) / dist.sum()),
        float(np.sum(dist**3) / np.sum(dist * dist)),
        float(screen.mean()),
    )


def edge_index(labels: np.ndarray) -> np.ndarray:
    taken: list[int] = []
    used: set[int] = set()
    for tx, ty in TARGETS:
        dist = np.hypot(labels[:, 0] - tx, labels[:, 1] - ty)
        for index in np.argsort(dist):
            index = int(index)
            if index not in used:
                used.add(index)
                taken.append(index)
                break
    return np.asarray(taken, dtype=int)


def fit_tree(features: np.ndarray, labels: np.ndarray) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=LEAF,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def dump_forest(model: ExtraTreesRegressor) -> list[dict]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        value = tree.value.reshape(tree.value.shape[0], -1)
        trees.append(
            {
                "f": tree.feature.astype(int).tolist(),
                "th": np.round(tree.threshold, 5).tolist(),
                "L": tree.children_left.astype(int).tolist(),
                "R": tree.children_right.astype(int).tolist(),
                "v": np.round(value, 5).tolist(),
            }
        )
    return trees


def forest_predict(trees: list[dict], rows: np.ndarray) -> np.ndarray:
    total = np.zeros((len(rows), 2))
    for tree in trees:
        feature = tree["f"]
        threshold = tree["th"]
        left = tree["L"]
        right = tree["R"]
        value = np.asarray(tree["v"])
        for index, row in enumerate(rows):
            node = 0
            while left[node] != -1:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
            total[index] += value[node]
    return total / len(trees)


def main() -> None:
    frame = load_center_frames(ROOT / "data" / "processed" / "extracted.parquet").reset_index(drop=True)
    if CACHE.exists():
        stored = np.load(CACHE)
        rows = stored["rows"]
        print(f"loaded {CACHE}", flush=True)
    else:
        rows = extract(frame)
        np.savez(CACHE, rows=rows, cols=np.array(FEATURE_ORDER))
        print(f"wrote {CACHE}", flush=True)

    ok = np.isfinite(rows).all(axis=1)
    print(f"finite {int(ok.sum())}/{len(rows)}", flush=True)
    frame = frame.loc[ok].reset_index(drop=True)
    rows = rows[ok]
    labels = frame[["label_x", "label_y"]].to_numpy(float)
    sittings = frame.n.to_numpy(int)

    center = (labels[:, 0] > 0.4) & (labels[:, 0] < 0.6) & (labels[:, 1] > 0.4) & (labels[:, 1] < 0.6)
    top = labels[:, 1] < 0.15
    bot = labels[:, 1] > 0.85
    ly = FEATURE_ORDER.index("m_ly")
    lx = FEATURE_ORDER.index("m_lx")
    print(
        f"center m_ly {np.median(rows[center, ly]):.3f}  m_lx {np.median(rows[center, lx]):.3f}  "
        f"top m_ly {np.median(rows[top, ly]):.3f}  bot m_ly {np.median(rows[bot, ly]):.3f}",
        flush=True,
    )

    ids = [int(value) for value in np.unique(sittings)]
    all_scores = []
    edge_scores = []
    for held in ids:
        train = sittings != held
        model = fit_tree(rows[train], labels[train])
        test = sittings == held
        pred = model.predict(rows[test])
        all_scores.append(metrics(pred, labels[test]))
        chosen = edge_index(labels[test])
        edge_scores.append(metrics(pred[chosen], labels[test][chosen]))
        print(f"loo {held} scr {all_scores[-1][3]:.3f} edge {edge_scores[-1][3]:.3f}", flush=True)

    all_mean = np.mean(all_scores, axis=0)
    edge_mean = np.mean(edge_scores, axis=0)
    print(
        f"all avg {all_mean[0]:.3f} w {all_mean[1]:.3f} sqr {all_mean[2]:.3f} scr {all_mean[3]:.3f}",
        flush=True,
    )
    print(
        f"edge avg {edge_mean[0]:.3f} w {edge_mean[1]:.3f} sqr {edge_mean[2]:.3f} scr {edge_mean[3]:.3f}",
        flush=True,
    )

    model = fit_tree(rows, labels)
    exported = dump_forest(model)
    gap = float(np.max(np.abs(forest_predict(exported, rows[:8]) - model.predict(rows[:8]))))
    print(f"tree walk gap {gap:.6f}", flush=True)
    if gap > 1e-3:
        raise SystemExit("exported trees do not match sklearn")

    payload = {
        "method": "corner-iris-per-frame",
        "feature_order": FEATURE_ORDER,
        "aspect": ASPECT,
        # Predictions already this far from 0.5 are pushed farther, by the
        # excess times (gain - 1). Fit on the leave-one-sitting nine dots.
        "edge_stretch": {"dead_x": 0.22, "dead_y": 0.17, "gain_x": 1.6, "gain_y": 1.8},
        "trees": exported,
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
