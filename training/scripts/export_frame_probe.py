#!/usr/bin/env python3
"""Shared face reading and tree helpers for the red-dot model.

The collect page runs MediaPipe Face Landmarker in VIDEO mode. That tracker
settles, over a few frames, to a different iris position than a still-image
mesh. Photos are replayed through that tracker until the reading settles.
The model file itself is written by export_corner_mix.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
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

MODEL = ROOT / "models" / "face_landmarker.task"
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

