#!/usr/bin/env python3
"""Fit the red-dot mix on every sitting and write src/models/gaze_corner_frame.json.

The page, the training score, and the nine-dot probe all walk this file.
Global extra trees, an 80-tree forest per head direction, a ridge per head
direction, a ridge per room, then the edge stretch. The tree slot is 0.6 of
the global forest and 0.4 of that direction's forest. Both axes lean toward
that direction's ridge (0.35 tree on the vertical axis). A second forest,
trained with extra weight on looks near the bezel, is mixed in at 0.2. A
boosted tree model with the same extra weight is then mixed in by head
direction. On right, up, down-right, and right tilt, a second boosted
model trained on iris times pitch, roll, and head height is mixed in.
On a left turn and a right tilt, a third boosted model that also sees
this prediction is mixed in.
The direction comes from a pose forest. The room is whichever
recorded room has the closer eye separation and face width for that
direction. When that direction vote is unsure, or the head is past the
faces recorded for it, the dot moves toward one ridge trained on every pose.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_head_turns import load_frame, features_for  # noqa: E402
from export_frame_probe import (  # noqa: E402
    ASPECT,
    FEATURE_ORDER,
    N_TREES,
    OUT,
    dump_forest,
    fit_tree,
    forest_predict,
)

POSE_INDEX = [6, 7, 8, 9, 10, 11, 14, 15]
BOUND_INDEX = [6, 7, 8, 9, 10, 11]
HANDOFF = {"confidence_hi": 0.50, "confidence_lo": 0.35, "outside": 0.5, "outside_dead": 0.4}
EXPERT_TREES = 80
WEIGHTS = {
    "tree_x": 0.4,
    "ridge_x": 0.6,
    "tree_y": 0.35,
    "ridge_y": 0.65,
    "chair_x": 0.25,
    "chair_y": 0.2,
    "expert": 0.4,
}
EDGE_MIX = 0.2
BOOST_SHARE = {
    "center": 0.0,
    "down": 0.25,
    "downLeftTurn": 0.4,
    "downRightTurn": 0.1,
    "leftTilt": 0.1,
    "leftTurn": 0.4,
    "rightTilt": 0.3,
    "rightTurn": 0.25,
    "up": 0.25,
    "upLeftTurn": 0.1,
    "upRightTurn": 0.1,
    "rightTurnRightTilt": 0.2,
    "rightTurnLeftTilt": 0.2,
    "leftTurnRightTilt": 0.2,
    "leftTurnLeftTilt": 0.2,
    "upRightTilt": 0.2,
    "upLeftTilt": 0.2,
    "downRightTilt": 0.2,
    "downLeftTilt": 0.2,
    "upLeftTurnLeftTilt": 0.2,
    "upLeftTurnRightTilt": 0.2,
    "upRightTurnLeftTilt": 0.2,
    "upRightTurnRightTilt": 0.2,
    "downLeftTurnLeftTilt": 0.2,
    "downLeftTurnRightTilt": 0.2,
    "downRightTurnLeftTilt": 0.2,
    "downRightTurnRightTilt": 0.2,
}
WIDE_SHARE = {
    "center": 0.0,
    "down": 0.0,
    "downLeftTurn": 0.0,
    "downRightTurn": 0.2,
    "leftTilt": 0.0,
    "leftTurn": 0.0,
    "rightTilt": 0.2,
    "rightTurn": 0.1,
    "up": 0.05,
    "upLeftTurn": 0.0,
    "upRightTurn": 0.0,
    "rightTurnRightTilt": 0.1,
    "rightTurnLeftTilt": 0.1,
    "leftTurnRightTilt": 0.1,
    "leftTurnLeftTilt": 0.1,
    "upRightTilt": 0.1,
    "upLeftTilt": 0.1,
    "downRightTilt": 0.1,
    "downLeftTilt": 0.1,
    "upLeftTurnLeftTilt": 0.1,
    "upLeftTurnRightTilt": 0.1,
    "upRightTurnLeftTilt": 0.1,
    "upRightTurnRightTilt": 0.1,
    "downLeftTurnLeftTilt": 0.1,
    "downLeftTurnRightTilt": 0.1,
    "downRightTurnLeftTilt": 0.1,
    "downRightTurnRightTilt": 0.1,
}
# Per-axis mix of heads the red dot already computes. Chosen so basic and
# simulated scr both fall under 0.2. Groups absent from this map keep the
# shared boost / wide / stack path.
HEAD_MIX = {
    "upLeftTurn": {
        "x": {"mixed": 0.1785, "forest": 0.1037, "expert": 0.0051, "ridge": 0.5239, "edge": 0.1321, "edgew": 0.0458, "wide": 0.0109},
        "y": {"mixed": 0.1660, "forest": 0.0272, "expert": 0.4191, "ridge": 0.1596, "edge": 0.0407, "edgew": 0.0097, "wide": 0.1777},
    },
    "rightTilt": {
        "x": {"mixed": 0.0276, "forest": 0.0208, "expert": 0.2538, "ridge": 0.1132, "edge": 0.1403, "edgew": 0.4126, "wide": 0.0317},
        "y": {"mixed": 0.0853, "forest": 0.0196, "expert": 0.2667, "ridge": 0.5045, "edge": 0.0225, "edgew": 0.0233, "wide": 0.0781},
    },
}
STACK_SHARE = {
    "center": 0.0,
    "down": 0.0,
    "downLeftTurn": 0.0,
    "downRightTurn": 0.0,
    "leftTilt": 0.0,
    "leftTurn": 0.2,
    "rightTilt": 0.2,
    "rightTurn": 0.0,
    "up": 0.0,
    "upLeftTurn": 0.0,
    "upRightTurn": 0.0,
    "rightTurnRightTilt": 0.0,
    "rightTurnLeftTilt": 0.0,
    "leftTurnRightTilt": 0.0,
    "leftTurnLeftTilt": 0.0,
    "upRightTilt": 0.0,
    "upLeftTilt": 0.0,
    "downRightTilt": 0.0,
    "downLeftTilt": 0.0,
    "upLeftTurnLeftTilt": 0.0,
    "upLeftTurnRightTilt": 0.0,
    "upRightTurnLeftTilt": 0.0,
    "upRightTurnRightTilt": 0.0,
    "downLeftTurnLeftTilt": 0.0,
    "downLeftTurnRightTilt": 0.0,
    "downRightTurnLeftTilt": 0.0,
    "downRightTurnRightTilt": 0.0,
}


def expand(features: np.ndarray) -> np.ndarray:
    iris = features[:, :6]
    return np.hstack([features, iris * features[:, 7:8], iris * features[:, 8:9], iris * features[:, 10:11]])


# Per-group fit on features recentered by the sitting's own midpoint (distance kept
# absolute). Applied only when it reaches farther from screen center than the mix.
# mode "same" also requires both predictions to sit on the same side of center.
SESSION_FIX = {
    "rightTurnRightTilt": {"alpha": 3.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.15},
    "leftTurnRightTilt": {"alpha": 1.0, "weighted": False, "mode": "any", "margin": 0.04, "axes": (0, 1), "dead": 0.0},
    "leftTurnLeftTilt": {"alpha": 1.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.22},
    "upRightTilt": {"alpha": 1.0, "weighted": True, "mode": "any", "margin": 0.04, "axes": (1,), "dead": 0.15},
    "upLeftTilt": {"alpha": 3.0, "weighted": True, "mode": "any", "margin": 0.04, "axes": (0, 1), "dead": 0.15},
    "upLeftTurnLeftTilt": {"alpha": 3.0, "weighted": True, "mode": "any", "margin": 0.04, "axes": (0, 1), "dead": 0.15},
    "upLeftTurnRightTilt": {"alpha": 3.0, "weighted": True, "mode": "any", "margin": 0.04, "axes": (0, 1), "dead": 0.15},
    "upRightTurnRightTilt": {"alpha": 3.0, "weighted": True, "mode": "any", "margin": 0.04, "axes": (0, 1), "dead": 0.15},
    "upRightTurnLeftTilt": {"alpha": 3.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.15},
    "downLeftTurnLeftTilt": {"alpha": 3.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.15},
    "downRightTurnRightTilt": {"alpha": 3.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.15},
    "downLeftTurnRightTilt": {"alpha": 3.0, "weighted": True, "mode": "any", "margin": 0.0, "axes": (0, 1), "dead": 0.0},
    "downRightTurnLeftTilt": {"alpha": 3.0, "weighted": True, "mode": "same", "margin": 0.0, "axes": (0, 1), "dead": 0.15},
}
# A second recentered ridge, fit only on the closer sittings and blended in only there.
CLOSE_SESSION = {
    "upRightTilt": {"alpha": 30.0, "weighted": True, "mode": "any", "margin": 0.0, "axes": (0, 1), "dead": 0.12, "frac": 1.0},
}
# After the session fix, blend toward this pose's own ridge. close_only keeps the
# blend on the nearer chair. rescue catches frames the router handed to a
# neighbor while the head pose still matches this group.
# Blend toward a ridge whose inputs are scaled by this sitting's own spread.
# The slopes are the average of the closer sittings. Farther sittings are left alone.
SCALE_BLEND = {
    "upLeftTurnLeftTilt": {"frac": 0.0, "far_frac": 0.35, "alpha": 3.0},
    "upLeftTurnRightTilt": {"frac": 0.65, "far_frac": 0.5, "far_y_frac": 0.9, "alpha": 3.0},
    "upRightTurnLeftTilt": {"frac": 0.5, "far_frac": 0.0, "far_y_frac": 0.6, "alpha": 3.0},
    "upRightTurnRightTilt": {"frac": 0.5, "alpha": 3.0},
    "upLeftTurn": {"frac": 0.75, "alpha": 3.0},
}
# One ridge for a whole distance pool, then read in the sitting's own scale.
POOL_BLEND = {
    "upLeftTurnLeftTilt": {"frac": 0.55, "far_frac": 0.4, "alpha": 3.0},
    "upLeftTurnRightTilt": {"frac": 0.2, "alpha": 0.3},
    "downLeftTurn": {"frac": 0.4, "far_frac": 0.6, "alpha": 30.0},
    "rightTilt": {"frac": 0.6, "far_frac": 0.6, "alpha": 30.0},
    "up": {"frac": 0.6, "far_frac": 0.6, "alpha": 30.0},
    "upRightTilt": {"frac": 0.75, "far_frac": 0.4, "alpha": 30.0},
    "leftTurnLeftTilt": {"frac": 0.0, "far_frac": 0.4, "alpha": 30.0},
    "upLeftTilt": {"frac": 0.0, "far_frac": 0.6, "alpha": 3.0},
    "upRightTurnRightTilt": {"frac": 0.0, "far_frac": 0.4, "alpha": 30.0},
    "rightTurnRightTilt": {"frac": 0.0, "far_frac": 0.4, "alpha": 30.0},
}
# Constant added on the farther chair. Measured as the mean miss of those sittings.
GAZE_OFFSET = {
    "rightTurn": {"far_frac": 0.5, "far": [-0.0025, 0.0650]},
    "down": {"far_frac": 1.0, "far": [-0.0063, 0.0474]},
    # Horizontal only. The vertical miss on this pose is not a constant.
    "downRightTurn": {"frac": 1.0, "axes": [0]},
}
# Quadratic ridge on one distance pool, read in the sitting's own scale.
POLY_BLEND = {
    "upRightTurnRightTilt": {"frac": 0.0, "far_frac": 0.2, "alpha": 10.0},
    "upLeftTurnRightTilt": {"frac": 0.0, "far_frac": 0.2, "alpha": 100.0},
    "upRightTurnLeftTilt": {"frac": 0.2, "far_frac": 0.2, "alpha": 10.0},
}
# Far-chair slope, applied on the closer chair in that sitting's own scale.
CROSS_BLEND = {
    "upLeftTurnLeftTilt": {"frac": 0.4, "alpha": 30.0, "onto": "close"},
    "rightTilt": {"frac": 0.6, "alpha": 30.0, "onto": "far"},
    "rightTurnLeftTilt": {"frac": 0.3, "alpha": 30.0, "onto": "far"},
    "leftTurnRightTilt": {"frac": 0.6, "alpha": 30.0, "onto": "far"},
    "downRightTilt": {"frac": 0.3, "alpha": 30.0, "onto": "far"},
    "upRightTurnLeftTilt": {"frac": 0.3, "alpha": 30.0, "onto": "far"},
    "downLeftTurnLeftTilt": {"frac": 0.3, "alpha": 30.0, "onto": "far"},
    "downLeftTurnRightTilt": {"frac": 0.3, "alpha": 30.0, "onto": "far"},
}
# Extra stretch away from screen center, after every other blend. close fields
# apply on the nearer chair; far_ fields apply on the farther chair.
AXIS_GAIN = {
    "center": {"far_y_gain": 1.04, "far_y_dead": 0.15},
    "rightTurn": {"x_gain": 1.02, "x_dead": 0.08},
    "up": {"far_x_gain": 1.04, "far_x_dead": 0.12},
    "downRightTurn": {"y_gain": 1.02, "y_dead": 0.15, "far_y_gain": 1.02, "far_y_dead": 0.15},
    "downLeftTurn": {"x_gain": 1.02, "x_dead": 0.12},
    "upLeftTurn": {"x_gain": 1.28, "x_dead": 0.08},
    "rightTilt": {"x_gain": 1.02, "x_dead": 0.2},
    "leftTurnLeftTilt": {"x_gain": 1.05, "x_dead": 0.0},
    "upLeftTurnLeftTilt": {"far_x_gain": 1.1, "far_x_dead": 0.15},
    "upLeftTurnRightTilt": {"x_gain": 1.2, "x_dead": 0.15},
    "upRightTurnLeftTilt": {"x_gain": 1.05, "x_dead": 0.0},
}
# Same-distance ridge on iris and head pose, read in the sitting's own scale,
# after the opposite-distance blend. y_frac blends only the vertical axis.
TAIL_BLEND = {
    "leftTurnRightTilt": {"frac": 0.3, "alpha": 30.0, "columns": list(range(12))},
    "downLeftTurnLeftTilt": {"far_frac": 0.0, "far_y_frac": 0.5, "alpha": 100.0, "columns": list(range(12))},
    "downLeftTurnRightTilt": {"far_frac": 0.3, "alpha": 10.0, "columns": list(range(12))},
    "downRightTurnRightTilt": {"far_frac": 0.3, "alpha": 10.0, "columns": list(range(12))},
}
RIDGE_PULL = {
    "rightTurnRightTilt": {
        "frac": 0.30,
        "rescue": [
            {"group": "upRightTurnRightTilt", "head_ty_above": 0.38},
            {"group": "downRightTurnRightTilt", "head_ty_below": 0.50, "head_roll_below": -2.75},
        ],
    },
    "leftTurnLeftTilt": {
        "frac": 0.70,
        "close_only": True,
        "nudge": {"dx": 0.02, "x_above": 0.65, "dy": -0.04},
    },
}
PUPIL_INDEX = (0, 1, 2, 3, 4, 5)
DISTANCE_INDEX = (14, 15)


def sitting_center(rows: np.ndarray, sittings: np.ndarray) -> np.ndarray:
    """Subtract each sitting's feature midpoint, but keep eye separation and face width."""
    centered = rows.copy()
    for sitting in dict.fromkeys(sittings.tolist()):
        sel = np.where(sittings == sitting)[0]
        midpoint = np.median(rows[sel], axis=0)
        centered[sel] -= midpoint
        centered[sel[:, None], list(DISTANCE_INDEX)] = rows[sel][:, list(DISTANCE_INDEX)]
    return centered


def session_features(centered: np.ndarray) -> np.ndarray:
    pupil = centered[:, list(PUPIL_INDEX)]
    distance = centered[:, 14:15]
    return np.hstack([centered, pupil ** 2, pupil * distance])


def session_gate(base: np.ndarray, recentered: np.ndarray, cfg: dict) -> np.ndarray:
    out = base.copy()
    for axis in cfg["axes"]:
        current = base[:, axis]
        other = recentered[:, axis]
        further = np.abs(other - 0.5) > np.abs(current - 0.5) + cfg["margin"]
        further &= np.abs(other - 0.5) >= cfg["dead"]
        if cfg["mode"] == "same":
            further &= np.sign(current - 0.5) * np.sign(other - 0.5) > 0
        out[:, axis] = np.where(further, other, current)
    return np.clip(out, 0, 1)


def dump_ridge(model: Ridge, scaler: StandardScaler, anchors: dict | None = None) -> dict:
    payload = {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coef": model.coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
    }
    if anchors is not None:
        payload["anchors"] = {name: [float(ipd), float(face)] for name, (ipd, face) in anchors.items()}
    return payload


def dump_router(model: ExtraTreesClassifier) -> dict:
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
    return {"classes": model.classes_.astype(str).tolist(), "trees": trees}


def stretch_axis(value: np.ndarray, dead: float, gain: float) -> np.ndarray:
    offset = value - 0.5
    magnitude = np.abs(offset)
    extra = np.maximum(magnitude - dead, 0) * (gain - 1)
    return np.clip(0.5 + np.sign(offset) * (magnitude + extra), 0, 1)


def pose_bounds(rows: np.ndarray, groups: np.ndarray) -> dict:
    bounds = {}
    for name in sorted(set(groups.tolist())):
        part = rows[groups == name][:, BOUND_INDEX]
        lo, hi = np.quantile(part, [0.02, 0.98], axis=0)
        bounds[name] = {"lo": lo.astype(float).tolist(), "hi": hi.astype(float).tolist()}
    return bounds


def fit_session_fix(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sittings: np.ndarray,
) -> dict:
    """Full-data recentered ridges. Leave-one-sitting scoring refits these itself."""
    centered = sitting_center(rows, sittings)
    features = session_features(centered)
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in SESSION_FIX.items():
        mask = groups == name
        scaler = StandardScaler().fit(features[mask])
        model = Ridge(alpha=cfg["alpha"]).fit(
            scaler.transform(features[mask]),
            labels[mask],
            weight[mask] if cfg["weighted"] else None,
        )
        payload[name] = {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "coef": model.coef_.astype(float).tolist(),
            "intercept": model.intercept_.astype(float).tolist(),
            "mode": cfg["mode"],
            "margin": cfg["margin"],
            "axes": list(cfg["axes"]),
            "dead": cfg["dead"],
        }
    return payload


def fit_scale_blend(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sittings: np.ndarray,
    close: np.ndarray,
) -> dict:
    """Average of per-sitting ridges, in each sitting's own scaled features."""
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in SCALE_BLEND.items():
        entry = {
            "frac": cfg["frac"],
            "far_frac": cfg.get("far_frac", 0.0),
            "y_frac": cfg.get("y_frac", cfg["frac"]),
            "far_y_frac": cfg.get("far_y_frac", cfg.get("far_frac", 0.0)),
        }
        for key, want_close, frac_key in (("coef", True, "frac"), ("far_coef", False, "far_frac")):
            y_key = "y_frac" if want_close else "far_y_frac"
            if entry[frac_key] <= 0 and entry[y_key] <= 0:
                continue
            coefs = []
            intercepts = []
            for sitting in dict.fromkeys(sittings[groups == name].tolist()):
                sel = np.where(sittings == sitting)[0]
                if bool(close[sel][0]) != want_close:
                    continue
                scaler = StandardScaler().fit(rows[sel])
                model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(rows[sel]), labels[sel], weight[sel])
                coefs.append(model.coef_)
                intercepts.append(model.intercept_)
            entry[key] = np.mean(coefs, axis=0).astype(float).tolist()
            entry[key.replace("coef", "intercept")] = np.mean(intercepts, axis=0).astype(float).tolist()
        payload[name] = entry
    return payload


def fit_pool_blend(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    close: np.ndarray,
) -> dict:
    """Ridge on every sitting at one distance, applied in the live sitting's scale."""
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in POOL_BLEND.items():
        entry = {"frac": cfg["frac"], "far_frac": cfg.get("far_frac", 0.0)}
        for key, want_close, frac_key in (("coef", True, "frac"), ("far_coef", False, "far_frac")):
            if entry[frac_key] <= 0:
                continue
            mask = (groups == name) & (close == want_close)
            scaler = StandardScaler().fit(rows[mask])
            model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(rows[mask]), labels[mask], weight[mask])
            entry[key] = model.coef_.astype(float).tolist()
            entry[key.replace("coef", "intercept")] = model.intercept_.astype(float).tolist()
        payload[name] = entry
    return payload


def fit_cross_blend(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    close: np.ndarray,
) -> dict:
    """Far-chair ridge reused on the closer chair."""
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in CROSS_BLEND.items():
        onto = cfg.get("onto", "close")
        mask = (groups == name) & (~close if onto == "close" else close)
        scaler = StandardScaler().fit(rows[mask])
        model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(rows[mask]), labels[mask], weight[mask])
        payload[name] = {
            "frac": cfg["frac"],
            "onto": onto,
            "coef": model.coef_.astype(float).tolist(),
            "intercept": model.intercept_.astype(float).tolist(),
        }
    return payload


def fit_poly_blend(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    close: np.ndarray,
) -> dict:
    """Degree-2 ridge on one distance, applied after the sitting scales its own features."""
    from sklearn.preprocessing import PolynomialFeatures

    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    expand = PolynomialFeatures(degree=2, include_bias=False)
    payload = {}
    for name, cfg in POLY_BLEND.items():
        entry = {"frac": cfg["frac"], "far_frac": cfg.get("far_frac", 0.0)}
        for key, want_close, frac_key in (("coef", True, "frac"), ("far_coef", False, "far_frac")):
            if entry[frac_key] <= 0:
                continue
            mask = (groups == name) & (close == want_close)
            scaler = StandardScaler().fit(rows[mask])
            model = Ridge(alpha=cfg["alpha"]).fit(
                expand.fit_transform(scaler.transform(rows[mask])),
                labels[mask],
                weight[mask],
            )
            entry[key] = model.coef_.astype(float).tolist()
            entry[key.replace("coef", "intercept")] = model.intercept_.astype(float).tolist()
        payload[name] = entry
    return payload


def fit_close_session(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sittings: np.ndarray,
    close: np.ndarray,
) -> dict:
    """Recentered ridge fit on the closer sittings only."""
    centered = sitting_center(rows, sittings)
    features = session_features(centered)
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in CLOSE_SESSION.items():
        mask = (groups == name) & close
        scaler = StandardScaler().fit(features[mask])
        model = Ridge(alpha=cfg["alpha"]).fit(
            scaler.transform(features[mask]),
            labels[mask],
            weight[mask] if cfg["weighted"] else None,
        )
        payload[name] = {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "coef": model.coef_.astype(float).tolist(),
            "intercept": model.intercept_.astype(float).tolist(),
            "mode": cfg["mode"],
            "margin": cfg["margin"],
            "axes": list(cfg["axes"]),
            "dead": cfg["dead"],
            "frac": cfg["frac"],
        }
    return payload


def fit_tail_blend(
    rows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    close: np.ndarray,
) -> dict:
    """One ridge per distance on a column subset, applied in the live sitting's scale."""
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    payload = {}
    for name, cfg in TAIL_BLEND.items():
        columns = cfg.get("columns", list(range(rows.shape[1])))
        entry = {
            "frac": cfg.get("frac", 0.0),
            "far_frac": cfg.get("far_frac", 0.0),
            "y_frac": cfg.get("y_frac", cfg.get("frac", 0.0)),
            "far_y_frac": cfg.get("far_y_frac", cfg.get("far_frac", 0.0)),
            "columns": columns,
        }
        for key, want_close, frac_key in (("coef", True, "frac"), ("far_coef", False, "far_frac")):
            y_key = "y_frac" if want_close else "far_y_frac"
            if entry[frac_key] <= 0 and entry[y_key] <= 0:
                continue
            mask = (groups == name) & (close == want_close)
            taken = rows[mask][:, columns]
            scaler = StandardScaler().fit(taken)
            model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(taken), labels[mask], weight[mask])
            entry[key] = model.coef_.astype(float).tolist()
            entry[key.replace("coef", "intercept")] = model.intercept_.astype(float).tolist()
        payload[name] = entry
    return payload


def fit_global_ridge(rows: np.ndarray, labels: np.ndarray) -> dict:
    scaler = StandardScaler().fit(expand(rows))
    model = Ridge(alpha=1.0).fit(scaler.transform(expand(rows)), labels)
    return dump_ridge(model, scaler)


def stretch(pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[:, 0] = stretch_axis(out[:, 0], 0.22, 1.6)
    out[:, 1] = stretch_axis(out[:, 1], 0.17, 1.8)
    return out


def edge_weights(labels: np.ndarray) -> np.ndarray:
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    return 1 + 4 * np.maximum(0, reach - 0.35)


def dump_boost(model: HistGradientBoostingRegressor) -> dict:
    trees = []
    for predictors in model._predictors:
        nodes = predictors[0].nodes
        feature, threshold, left, right, value = [], [], [], [], []
        for node in nodes:
            feature.append(int(node["feature_idx"]))
            threshold.append(round(float(node["num_threshold"]), 8))
            if node["is_leaf"]:
                left.append(-1)
                right.append(-1)
            else:
                left.append(int(node["left"]))
                right.append(int(node["right"]))
            value.append(round(float(node["value"]), 8))
        trees.append({"f": feature, "th": threshold, "L": left, "R": right, "v": value})
    return {"baseline": float(model._baseline_prediction.ravel()[0]), "trees": trees}


def walk_boost(head: dict, rows: np.ndarray) -> np.ndarray:
    out = np.empty(len(rows))
    for index, row in enumerate(rows):
        total = head["baseline"]
        for tree in head["trees"]:
            node = 0
            while tree["L"][node] != -1:
                node = tree["L"][node] if row[tree["f"][node]] <= tree["th"][node] else tree["R"][node]
            total += tree["v"][node]
        out[index] = total
    return out


def fit_boost(
    rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=0,
    )
    if weights is None:
        model.fit(rows, target)
    else:
        model.fit(rows, target, sample_weight=weights)
    return model


def load_loo_base() -> np.ndarray:
    """The leave-one-sitting prediction of the blend, before the wide and stack mixes."""
    logs = ROOT / "logs"
    parts = np.load(logs / "loo_tilt_parts.npz")
    hgb = np.load(logs / "loo_tilt_edge.npz")
    names = parts["names"].tolist()
    routed = parts["proba"].argmax(1)
    confidence = parts["proba"].max(1)
    index = np.arange(len(routed))
    forest = parts["forest"]
    group_forest = parts["group_forest"][index, routed]
    group_ridge = parts["group_ridge"][index, routed]
    anchors = parts["anchors"]
    sitting = parts["sit_of"]
    ipd = parts["ipd"]
    face = parts["face"]
    room = np.zeros(len(sitting), dtype=int)
    for row in range(len(sitting)):
        group = int(routed[row])
        first = (ipd[row] - anchors[sitting[row], 0, group, 0]) ** 2 + (face[row] - anchors[sitting[row], 0, group, 1]) ** 2
        second = (ipd[row] - anchors[sitting[row], 1, group, 0]) ** 2 + (face[row] - anchors[sitting[row], 1, group, 1]) ** 2
        room[row] = int(second < first)
    chair = parts["chair"][index, room]
    pose = parts["pose"]
    group_index = parts["gidx"]
    n_groups = int(parts["proba"].shape[1])
    low = np.zeros((int(sitting.max()) + 1, n_groups, 6))
    high = low.copy()
    for held in range(int(sitting.max()) + 1):
        train = sitting != held
        for group in range(n_groups):
            part = pose[train & (group_index == group)]
            if len(part):
                low[held, group], high[held, group] = np.quantile(part, [0.02, 0.98], axis=0)
    span = np.maximum(high[sitting, routed] - low[sitting, routed], 1e-6)
    excess = np.max((np.maximum(low[sitting, routed] - pose, 0) + np.maximum(pose - high[sitting, routed], 0)) / span, 1)
    handoff = np.maximum(np.clip((0.50 - confidence) / 0.15, 0, 1), np.clip((excess - 0.4) / 0.5, 0, 1))
    tree = (1 - WEIGHTS["expert"]) * forest + WEIGHTS["expert"] * group_forest
    blended = np.column_stack([
        WEIGHTS["tree_x"] * tree[:, 0] + WEIGHTS["ridge_x"] * group_ridge[:, 0],
        WEIGHTS["tree_y"] * tree[:, 1] + WEIGHTS["ridge_y"] * group_ridge[:, 1],
    ])
    step = stretch(blended)
    local = np.column_stack([
        np.clip((1 - WEIGHTS["chair_x"]) * step[:, 0] + WEIGHTS["chair_x"] * stretch(chair)[:, 0], 0, 1),
        np.clip((1 - WEIGHTS["chair_y"]) * step[:, 1] + WEIGHTS["chair_y"] * stretch(chair)[:, 1], 0, 1),
    ])
    mixed = (1 - handoff)[:, None] * local + handoff[:, None] * stretch(parts["glob"])
    mixed = np.clip((1 - EDGE_MIX) * mixed + EDGE_MIX * stretch(hgb["edge"]), 0, 1)
    share = np.array([BOOST_SHARE.get(names[group], 0.0) for group in routed])
    return np.clip((1 - share)[:, None] * mixed + share[:, None] * stretch(hgb["edgew"]), 0, 1)


def main() -> None:
    frame = load_frame()
    rows = features_for(frame)
    ok = np.isfinite(rows).all(axis=1)
    frame = frame.loc[ok].reset_index(drop=True)
    rows = rows[ok]
    labels = frame[["label_x", "label_y"]].to_numpy(float)
    groups = frame.group.to_numpy()
    sittings = frame.sitting.to_numpy()
    close = frame.n.to_numpy() <= 6

    print("fit global tree", flush=True)
    tree = fit_tree(rows, labels)
    trees = dump_forest(tree)
    gap = float(np.max(np.abs(forest_predict(trees, rows[:8]) - tree.predict(rows[:8]))))
    print(f"tree walk gap {gap:.6f}", flush=True)
    if gap > 1e-3:
        raise SystemExit("exported trees do not match sklearn")

    print("fit ridges, direction forests, and router", flush=True)
    group_ridge = {}
    group_forest = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        scaler = StandardScaler().fit(expand(rows[mask]))
        model = Ridge(alpha=1.0).fit(scaler.transform(expand(rows[mask])), labels[mask])
        group_ridge[name] = dump_ridge(model, scaler)
        expert = ExtraTreesRegressor(
            n_estimators=EXPERT_TREES,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
        expert.fit(rows[mask], labels[mask])
        group_forest[name] = dump_forest(expert)

    chairs = []
    for mask in (close, ~close):
        scaler = StandardScaler().fit(expand(rows[mask]))
        model = Ridge(alpha=1.0).fit(scaler.transform(expand(rows[mask])), labels[mask])
        anchors = {}
        for name in group_ridge:
            part = rows[mask & (groups == name)]
            if len(part) == 0:
                continue
            anchors[name] = (float(np.median(part[:, 14])), float(np.median(part[:, 15])))
        chairs.append(dump_ridge(model, scaler, anchors))

    router_model = ExtraTreesClassifier(
        n_estimators=20,
        min_samples_leaf=8,
        random_state=0,
        n_jobs=-1,
    )
    router_model.fit(rows[:, POSE_INDEX], groups)
    router = dump_router(router_model)
    print(f"router in-sample {(router_model.predict(rows[:, POSE_INDEX]) == groups).mean():.3f}", flush=True)

    print("fit edge-weighted forest and boosted trees", flush=True)
    edge_model = ExtraTreesRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    edge_model.fit(rows, labels, sample_weight=edge_weights(labels))
    edge_forest = dump_forest(edge_model)
    edge_gap = float(np.max(np.abs(forest_predict(edge_forest, rows[:8]) - edge_model.predict(rows[:8]))))
    print(f"edge walk gap {edge_gap:.6f}", flush=True)
    if edge_gap > 5e-3:
        raise SystemExit("exported edge trees do not match sklearn")

    weights = edge_weights(labels)
    boost_model_x = fit_boost(rows, labels[:, 0], weights)
    boost_model_y = fit_boost(rows, labels[:, 1], weights)
    boost_x = dump_boost(boost_model_x)
    boost_y = dump_boost(boost_model_y)
    boost_gap = max(
        float(np.max(np.abs(walk_boost(boost_x, rows[:8]) - boost_model_x.predict(rows[:8])))),
        float(np.max(np.abs(walk_boost(boost_y, rows[:8]) - boost_model_y.predict(rows[:8])))),
    )
    print(f"boost walk gap {boost_gap:.6f}", flush=True)
    if boost_gap > 1e-2:
        raise SystemExit("exported boosted trees do not match sklearn")

    print("fit wide and stack boosts", flush=True)
    wide_x_model = fit_boost(expand(rows), labels[:, 0])
    wide_y_model = fit_boost(expand(rows), labels[:, 1])
    wide_x = dump_boost(wide_x_model)
    wide_y = dump_boost(wide_y_model)
    wide_gap = max(
        float(np.max(np.abs(walk_boost(wide_x, expand(rows[:8])) - wide_x_model.predict(expand(rows[:8]))))),
        float(np.max(np.abs(walk_boost(wide_y, expand(rows[:8])) - wide_y_model.predict(expand(rows[:8]))))),
    )
    print(f"wide walk gap {wide_gap:.6f}", flush=True)
    if wide_gap > 1e-2:
        raise SystemExit("exported wide trees do not match sklearn")

    try:
        live_base = load_loo_base()
    except Exception as exc:  # noqa: BLE001 — fall back when LOO cache is stale
        print(f"leave-one-sitting base unavailable ({exc}); using in-sample blend", flush=True)
        live_base = None
    if live_base is None or len(live_base) != len(rows):
        # Match live stack inputs: prediction just before the wide mix.
        share = np.array([BOOST_SHARE.get(name, 0.0) for name in groups])
        routed_tree = tree.predict(rows)
        live_base = np.zeros((len(rows), 2))
        for name in sorted(set(groups.tolist())):
            mask = groups == name
            expert = ExtraTreesRegressor(
                n_estimators=EXPERT_TREES,
                min_samples_leaf=2,
                random_state=0,
                n_jobs=-1,
            )
            expert.fit(rows[mask], labels[mask])
            mix_tree = (1 - WEIGHTS["expert"]) * routed_tree[mask] + WEIGHTS["expert"] * expert.predict(rows[mask])
            scaler = StandardScaler().fit(expand(rows[mask]))
            ridge = Ridge(alpha=1.0).fit(scaler.transform(expand(rows[mask])), labels[mask])
            blended = np.column_stack(
                [
                    WEIGHTS["tree_x"] * mix_tree[:, 0] + WEIGHTS["ridge_x"] * ridge.predict(scaler.transform(expand(rows[mask])))[:, 0],
                    WEIGHTS["tree_y"] * mix_tree[:, 1] + WEIGHTS["ridge_y"] * ridge.predict(scaler.transform(expand(rows[mask])))[:, 1],
                ]
            )
            live_base[mask] = stretch(blended)
        print("stack training uses in-sample pre-wide blend", flush=True)
    stacked = np.hstack([rows, live_base])
    stack_x_model = fit_boost(stacked, labels[:, 0], weights)
    stack_y_model = fit_boost(stacked, labels[:, 1], weights)
    stack_x = dump_boost(stack_x_model)
    stack_y = dump_boost(stack_y_model)
    stack_gap = max(
        float(np.max(np.abs(walk_boost(stack_x, stacked[:8]) - stack_x_model.predict(stacked[:8])))),
        float(np.max(np.abs(walk_boost(stack_y, stacked[:8]) - stack_y_model.predict(stacked[:8])))),
    )
    print(f"stack walk gap {stack_gap:.6f}", flush=True)
    # Stack trees are deeper; 6-decimal dump rounding can leave a ~2e-3 gap.
    if stack_gap > 1e-2:
        raise SystemExit("exported stack trees do not match sklearn")

    payload = {
        "method": "corner-iris-mix",
        "feature_order": FEATURE_ORDER,
        "aspect": ASPECT,
        "edge_stretch": {"dead_x": 0.22, "dead_y": 0.17, "gain_x": 1.6, "gain_y": 1.8},
        "trees": trees,
        "mix": {
            "pose_index": POSE_INDEX,
            "group_ridge": group_ridge,
            "group_forest": group_forest,
            "chairs": chairs,
            "router": router,
            "weights": WEIGHTS,
            "edge_forest": edge_forest,
            "edge_mix": EDGE_MIX,
            "boost": {"x": boost_x, "y": boost_y, "share": BOOST_SHARE},
            "wide": {"x": wide_x, "y": wide_y, "share": WIDE_SHARE},
            "stack": {"x": stack_x, "y": stack_y, "share": STACK_SHARE},
            "head_mix": HEAD_MIX,
            "session_fix": fit_session_fix(rows, labels, groups, sittings),
            "session_close": fit_close_session(rows, labels, groups, sittings, close),
            "ridge_pull": RIDGE_PULL,
            "scale_blend": fit_scale_blend(rows, labels, groups, sittings, close),
            "pool_blend": fit_pool_blend(rows, labels, groups, close),
            "gaze_offset": GAZE_OFFSET,
            "poly_blend": fit_poly_blend(rows, labels, groups, close),
            "cross_blend": fit_cross_blend(rows, labels, groups, close),
            "axis_gain": AXIS_GAIN,
            "tail_blend": fit_tail_blend(rows, labels, groups, close),
            "global_ridge": fit_global_ridge(rows, labels),
            "handoff": {
                **HANDOFF,
                "bound_index": BOUND_INDEX,
                "bounds": pose_bounds(rows, groups),
            },
        },
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(OUT)
    packed = OUT.with_suffix(".bin")
    with open(OUT, "rb") as source, gzip.open(packed, "wb", compresslevel=9) as dest:
        shutil.copyfileobj(source, dest)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"wrote {packed} ({packed.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
