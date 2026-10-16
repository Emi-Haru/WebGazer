#!/usr/bin/env python3
"""Print the Basic / Simulated table for the current leave-one-sitting mix."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_head_turns import features_for, load_frame  # noqa: E402
from export_corner_mix import (  # noqa: E402
    AXIS_GAIN,
    BOOST_SHARE,
    EDGE_MIX,
    HEAD_MIX,
    CLOSE_SESSION,
    RIDGE_PULL,
    GAZE_OFFSET,
    CROSS_BLEND,
    POLY_BLEND,
    POOL_BLEND,
    SCALE_BLEND,
    SESSION_FIX,
    TAIL_BLEND,
    WEIGHTS,
    WIDE_SHARE,
    session_features,
    session_gate,
    sitting_center,
    stretch,
)
from export_frame_probe import ASPECT, FEATURE_ORDER, TARGETS, edge_index, metrics  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

PARTS = ROOT / "logs" / "loo_tilt_parts.npz"
EDGE = ROOT / "logs" / "loo_tilt_edge.npz"

# Last shipped table (2026-10-12), for the improvement row. Only the scr slots are read.
LIVE = {
    "center": (0.119, 0.160, 0.209, 0.165, 0.101, 0.183, 0.233, 0.138),
    "rightTurn": (0.123, 0.156, 0.185, 0.167, 0.118, 0.186, 0.223, 0.170),
    "leftTurn": (0.120, 0.157, 0.187, 0.165, 0.106, 0.155, 0.188, 0.154),
    "up": (0.140, 0.182, 0.217, 0.154, 0.135, 0.197, 0.244, 0.173),
    "down": (0.103, 0.135, 0.161, 0.145, 0.089, 0.139, 0.173, 0.127),
    "upRightTurn": (0.106, 0.147, 0.202, 0.151, 0.083, 0.134, 0.169, 0.124),
    "downRightTurn": (0.104, 0.138, 0.168, 0.147, 0.105, 0.171, 0.214, 0.140),
    "downLeftTurn": (0.137, 0.178, 0.211, 0.160, 0.134, 0.197, 0.240, 0.185),
    "upLeftTurn": (0.132, 0.170, 0.200, 0.169, 0.120, 0.180, 0.216, 0.158),
    "leftTilt": (0.114, 0.151, 0.180, 0.160, 0.121, 0.174, 0.208, 0.160),
    "rightTilt": (0.132, 0.168, 0.200, 0.142, 0.131, 0.187, 0.231, 0.176),
}


def build_pred():
    frame = load_frame()
    sittings = frame.sitting.to_numpy()
    groups = frame.group.to_numpy()
    parts = np.load(PARTS, allow_pickle=True)
    hgb = np.load(EDGE, allow_pickle=True)
    if len(parts["forest"]) != len(frame):
        raise SystemExit("LOO cache length does not match load_frame()")

    names = parts["names"].tolist()
    routed = parts["proba"].argmax(1)
    conf = parts["proba"].max(1)
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
        first = (ipd[row] - anchors[sitting[row], 0, group, 0]) ** 2 + (
            face[row] - anchors[sitting[row], 0, group, 1]
        ) ** 2
        second = (ipd[row] - anchors[sitting[row], 1, group, 0]) ** 2 + (
            face[row] - anchors[sitting[row], 1, group, 1]
        ) ** 2
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
    excess = np.max(
        (np.maximum(low[sitting, routed] - pose, 0) + np.maximum(pose - high[sitting, routed], 0))
        / span,
        1,
    )
    handoff = np.maximum(
        np.clip((0.50 - conf) / 0.15, 0, 1),
        np.clip((excess - 0.4) / 0.5, 0, 1),
    )
    tree = (1 - WEIGHTS["expert"]) * forest + WEIGHTS["expert"] * group_forest
    blended = np.column_stack(
        [
            WEIGHTS["tree_x"] * tree[:, 0] + WEIGHTS["ridge_x"] * group_ridge[:, 0],
            WEIGHTS["tree_y"] * tree[:, 1] + WEIGHTS["ridge_y"] * group_ridge[:, 1],
        ]
    )
    step = stretch(blended)
    local = np.column_stack(
        [
            np.clip((1 - WEIGHTS["chair_x"]) * step[:, 0] + WEIGHTS["chair_x"] * stretch(chair)[:, 0], 0, 1),
            np.clip((1 - WEIGHTS["chair_y"]) * step[:, 1] + WEIGHTS["chair_y"] * stretch(chair)[:, 1], 0, 1),
        ]
    )
    mixed = (1 - handoff)[:, None] * local + handoff[:, None] * stretch(parts["glob"])
    mixed = np.clip((1 - EDGE_MIX) * mixed + EDGE_MIX * stretch(hgb["edge"]), 0, 1)
    share = np.array([BOOST_SHARE.get(names[group], 0.0) for group in routed])
    pred = np.clip((1 - share)[:, None] * mixed + share[:, None] * stretch(hgb["edgew"]), 0, 1)
    bank = {
        "mixed": mixed,
        "forest": stretch(forest),
        "expert": stretch(parts["group_forest"][index, parts["gidx"]]),
        "ridge": stretch(parts["group_ridge"][index, parts["gidx"]]),
        "edge": stretch(hgb["edge"]),
        "edgew": stretch(hgb["edgew"]),
        "wide": stretch(hgb["wide"]),
    }
    for name, weights in HEAD_MIX.items():
        rows = np.array(names)[routed] == name
        if not rows.any():
            continue
        acc = np.zeros((int(rows.sum()), 2))
        for axis, key in ((0, "x"), (1, "y")):
            total = np.zeros(int(rows.sum()))
            for head, coef in weights[key].items():
                total += coef * bank[head][rows, axis]
            acc[:, axis] = total
        pred[rows] = np.clip(acc, 0, 1)
    if "wide" in hgb:
        wide_share = np.array([
            0.0 if names[group] in HEAD_MIX else WIDE_SHARE.get(names[group], 0.0)
            for group in routed
        ])
        pred = np.clip((1 - wide_share)[:, None] * pred + wide_share[:, None] * stretch(hgb["wide"]), 0, 1)
    labels = np.column_stack([parts["label_x"], parts["label_y"]])
    pred = apply_session(pred, labels, sittings, groups, frame)
    pred = apply_ridge_pull(pred, parts, frame)
    pred = apply_scale_blend(pred, labels, sittings, groups, frame)
    pred = apply_pool_blend(pred, labels, sittings, groups, frame)
    pred = apply_gaze_offset(pred, labels, sittings, groups, frame)
    pred = apply_poly_blend(pred, labels, sittings, groups, frame)
    pred = apply_cross_blend(pred, labels, sittings, groups, frame)
    pred = apply_axis_gain(pred, groups, frame)
    pred = apply_tail_blend(pred, labels, sittings, groups, frame)
    pred = apply_tilt_shear(pred, groups)
    return pred, labels, sittings, groups, names


# Same slopes as TILT_SHEAR in src/cornerProbe.ts. Vertical only.
TILT_SHEAR = {
    "leftTilt": 0.086,
    "rightTurnLeftTilt": 0.078,
    "upLeftTilt": 0.051,
    "downLeftTilt": 0.062,
    "downRightTilt": -0.069,
}


def apply_tilt_shear(pred, groups):
    out = pred.copy()
    for name, beta in TILT_SHEAR.items():
        sel = groups == name
        out[sel, 1] = np.clip(pred[sel, 1] - beta * (pred[sel, 0] - 0.5), 0, 1)
    return out


def apply_session(pred, labels, sittings, groups, frame):
    """Leave-one-sitting recentered ridges. Uses the sitting's feature midpoint, not its labels."""
    rows = features_for(frame)
    if len(rows) != len(pred):
        raise SystemExit("feature cache length does not match the mix")
    centered = sitting_center(rows, sittings)
    features = session_features(centered)
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    out = pred.copy()
    for name, cfg in SESSION_FIX.items():
        recentered = np.zeros_like(pred)
        for sitting in dict.fromkeys(sittings[groups == name].tolist()):
            test = sittings == sitting
            train = (groups == name) & ~test
            scaler = StandardScaler().fit(features[train])
            model = Ridge(alpha=cfg["alpha"]).fit(
                scaler.transform(features[train]),
                labels[train],
                weight[train] if cfg["weighted"] else None,
            )
            recentered[test] = np.clip(model.predict(scaler.transform(features[test])), 0, 1)
        rows_of = groups == name
        out[rows_of] = session_gate(pred[rows_of], recentered[rows_of], cfg)
    close = frame.n.to_numpy() <= 6
    for name, cfg in CLOSE_SESSION.items():
        recentered = np.zeros_like(pred)
        close_sittings = [sitting for sitting in dict.fromkeys(sittings[groups == name].tolist()) if close[sittings == sitting][0]]
        for sitting in close_sittings:
            test = sittings == sitting
            train = (groups == name) & close & ~test
            scaler = StandardScaler().fit(features[train])
            model = Ridge(alpha=cfg["alpha"]).fit(
                scaler.transform(features[train]),
                labels[train],
                weight[train] if cfg["weighted"] else None,
            )
            recentered[test] = np.clip(model.predict(scaler.transform(features[test])), 0, 1)
        rows_of = (groups == name) & close
        gated = session_gate(out[rows_of], recentered[rows_of], cfg)
        frac = cfg["frac"]
        out[rows_of] = np.clip((1 - frac) * out[rows_of] + frac * gated, 0, 1)
    return out


def apply_ridge_pull(pred, parts, frame):
    """Blend the finished mix toward a pose ridge. Rescue rules use head pose, not the label."""
    names = parts["names"].tolist()
    routed_index = parts["proba"].argmax(1)
    routed = np.asarray(names)[routed_index]
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    head_ty = rows[:, FEATURE_ORDER.index("head_ty")]
    head_roll = rows[:, FEATURE_ORDER.index("head_roll")]
    out = pred.copy()
    for name, cfg in RIDGE_PULL.items():
        head = stretch(parts["group_ridge"][:, names.index(name)])
        own = routed == name
        if cfg.get("close_only"):
            own = own & close
        use = own.copy()
        for rescue in cfg.get("rescue", []):
            mask = routed == rescue["group"]
            if "head_ty_above" in rescue:
                mask = mask & (head_ty > rescue["head_ty_above"])
            if "head_ty_below" in rescue:
                mask = mask & (head_ty < rescue["head_ty_below"])
            if "head_roll_below" in rescue:
                mask = mask & (head_roll < rescue["head_roll_below"])
            use = use | mask
        frac = cfg["frac"]
        out[use] = np.clip((1 - frac) * pred[use] + frac * head[use], 0, 1)
        nudge = cfg.get("nudge")
        if nudge:
            moved = out[own, 0]
            out[own, 0] = np.where(moved > nudge["x_above"], np.clip(moved + nudge["dx"], 0, 1), moved)
            out[own, 1] = np.clip(out[own, 1] + nudge["dy"], 0, 1)
    return out


def apply_scale_blend(pred, labels, sittings, groups, frame):
    """Leave-one-sitting average of same-distance ridges, scaled by the held-out sitting."""
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    out = pred.copy()
    for name, cfg in SCALE_BLEND.items():
        for want_close, frac, y_frac in (
            (True, cfg["frac"], cfg.get("y_frac", cfg["frac"])),
            (False, cfg.get("far_frac", 0.0), cfg.get("far_y_frac", cfg.get("far_frac", 0.0))),
        ):
            if frac <= 0 and y_frac <= 0:
                continue
            packs = {}
            for sitting in dict.fromkeys(sittings[groups == name].tolist()):
                sel = np.where(sittings == sitting)[0]
                if bool(close[sel][0]) != want_close:
                    continue
                scaler = StandardScaler().fit(rows[sel])
                model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(rows[sel]), labels[sel], weight[sel])
                packs[sitting] = (model.coef_, model.intercept_)
            for sitting in packs:
                donors = [packs[other] for other in packs if other != sitting]
                sel = np.where(sittings == sitting)[0]
                scaler = StandardScaler().fit(rows[sel])
                hat = np.clip(
                    scaler.transform(rows[sel]) @ np.mean([c for c, _ in donors], 0).T
                    + np.mean([b for _, b in donors], 0),
                    0,
                    1,
                )
                mixed = out[sel].copy()
                mixed[:, 0] = (1 - frac) * out[sel, 0] + frac * hat[:, 0]
                mixed[:, 1] = (1 - y_frac) * out[sel, 1] + y_frac * hat[:, 1]
                out[sel] = np.clip(mixed, 0, 1)
    return out


def apply_pool_blend(pred, labels, sittings, groups, frame):
    """Leave-one-sitting pooled ridge, read back in the held-out sitting's scale."""
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    out = pred.copy()
    for name, cfg in POOL_BLEND.items():
        for want_close, frac in ((True, cfg["frac"]), (False, cfg.get("far_frac", 0.0))):
            if frac <= 0:
                continue
            pool = [
                sitting
                for sitting in dict.fromkeys(sittings[groups == name].tolist())
                if bool(close[sittings == sitting][0]) == want_close
            ]
            for sitting in pool:
                donor = np.zeros(len(sittings), dtype=bool)
                for other in pool:
                    if other != sitting:
                        donor |= sittings == other
                scaler = StandardScaler().fit(rows[donor])
                model = Ridge(alpha=cfg["alpha"]).fit(
                    scaler.transform(rows[donor]), labels[donor], weight[donor]
                )
                sel = np.where(sittings == sitting)[0]
                held = StandardScaler().fit(rows[sel])
                hat = np.clip(held.transform(rows[sel]) @ model.coef_.T + model.intercept_, 0, 1)
                out[sel] = np.clip((1 - frac) * out[sel] + frac * hat, 0, 1)
    return out


def apply_cross_blend(pred, labels, sittings, groups, frame):
    """Far-chair ridge, read in each closer sitting's own scale."""
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    out = pred.copy()
    for name, cfg in CROSS_BLEND.items():
        onto_close = cfg.get("onto", "close") == "close"
        source = (groups == name) & (~close if onto_close else close)
        scaler = StandardScaler().fit(rows[source])
        model = Ridge(alpha=cfg["alpha"]).fit(scaler.transform(rows[source]), labels[source], weight[source])
        target = (groups == name) & (close if onto_close else ~close)
        for sitting in dict.fromkeys(sittings[target].tolist()):
            sel = np.where(sittings == sitting)[0]
            held = StandardScaler().fit(rows[sel])
            hat = np.clip(held.transform(rows[sel]) @ model.coef_.T + model.intercept_, 0, 1)
            out[sel] = np.clip((1 - cfg["frac"]) * out[sel] + cfg["frac"] * hat, 0, 1)
    return out


def apply_poly_blend(pred, labels, sittings, groups, frame):
    """Leave-one-sitting quadratic ridge, read in the held-out sitting's scale."""
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    expand = PolynomialFeatures(degree=2, include_bias=False)
    out = pred.copy()
    for name, cfg in POLY_BLEND.items():
        for want_close, frac in ((True, cfg["frac"]), (False, cfg.get("far_frac", 0.0))):
            if frac <= 0:
                continue
            pool = [
                sitting
                for sitting in dict.fromkeys(sittings[groups == name].tolist())
                if bool(close[sittings == sitting][0]) == want_close
            ]
            for sitting in pool:
                donor = np.zeros(len(sittings), dtype=bool)
                for other in pool:
                    if other != sitting:
                        donor |= sittings == other
                scaler = StandardScaler().fit(rows[donor])
                model = Ridge(alpha=cfg["alpha"]).fit(
                    expand.fit_transform(scaler.transform(rows[donor])),
                    labels[donor],
                    weight[donor],
                )
                sel = np.where(sittings == sitting)[0]
                held = StandardScaler().fit(rows[sel])
                hat = np.clip(model.predict(expand.transform(held.transform(rows[sel]))), 0, 1)
                out[sel] = np.clip((1 - frac) * out[sel] + frac * hat, 0, 1)
    return out


def _axis_stretch(values, gain, dead):
    offset = values - 0.5
    mag = np.abs(offset)
    extra = np.maximum(mag - dead, 0) * (gain - 1)
    return np.clip(0.5 + np.sign(offset) * (mag + extra), 0, 1)


def apply_axis_gain(pred, groups, frame):
    """Push a finished axis farther from center. Dead zone stays put."""
    close = frame.n.to_numpy() <= 6
    out = pred.copy()
    for name, cfg in AXIS_GAIN.items():
        for want_close, gain_key, dead_key, axis in (
            (True, "x_gain", "x_dead", 0),
            (True, "y_gain", "y_dead", 1),
            (False, "far_x_gain", "far_x_dead", 0),
            (False, "far_y_gain", "far_y_dead", 1),
        ):
            gain = cfg.get(gain_key, 1.0)
            if gain == 1:
                continue
            mask = (groups == name) & (close if want_close else ~close)
            out[mask, axis] = _axis_stretch(out[mask, axis], gain, cfg.get(dead_key, 0.0))
    return out


def apply_tail_blend(pred, labels, sittings, groups, frame):
    """Leave-one-sitting same-distance ridge on a column subset."""
    rows = features_for(frame)
    close = frame.n.to_numpy() <= 6
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 8 * np.maximum(0, reach - 0.35)
    out = pred.copy()
    for name, cfg in TAIL_BLEND.items():
        columns = cfg.get("columns", list(range(rows.shape[1])))
        for want_close, frac, y_frac in (
            (True, cfg.get("frac", 0.0), cfg.get("y_frac", cfg.get("frac", 0.0))),
            (False, cfg.get("far_frac", 0.0), cfg.get("far_y_frac", cfg.get("far_frac", 0.0))),
        ):
            if frac <= 0 and y_frac <= 0:
                continue
            pool = [
                sitting
                for sitting in dict.fromkeys(sittings[groups == name].tolist())
                if bool(close[sittings == sitting][0]) == want_close
            ]
            taken = rows[:, columns]
            for sitting in pool:
                donor = np.zeros(len(sittings), dtype=bool)
                for other in pool:
                    if other != sitting:
                        donor |= sittings == other
                scaler = StandardScaler().fit(taken[donor])
                model = Ridge(alpha=cfg["alpha"]).fit(
                    scaler.transform(taken[donor]), labels[donor], weight[donor]
                )
                sel = np.where(sittings == sitting)[0]
                held = StandardScaler().fit(taken[sel])
                hat = np.clip(held.transform(taken[sel]) @ model.coef_.T + model.intercept_, 0, 1)
                mixed = out[sel].copy()
                mixed[:, 0] = (1 - frac) * out[sel, 0] + frac * hat[:, 0]
                mixed[:, 1] = (1 - y_frac) * out[sel, 1] + y_frac * hat[:, 1]
                out[sel] = np.clip(mixed, 0, 1)
    return out


def apply_gaze_offset(pred, labels, sittings, groups, frame):
    """Leave-one-sitting mean miss, added on the same-distance sittings."""
    close = frame.n.to_numpy() <= 6
    out = pred.copy()
    for name, cfg in GAZE_OFFSET.items():
        for want_close, frac, _key in (
            (True, cfg.get("frac", 0.0), "close"),
            (False, cfg.get("far_frac", 0.0), "far"),
        ):
            if frac <= 0:
                continue
            pool = [
                sitting
                for sitting in dict.fromkeys(sittings[groups == name].tolist())
                if bool(close[sittings == sitting][0]) == want_close
            ]
            for sitting in pool:
                donor = np.zeros(len(sittings), dtype=bool)
                for other in pool:
                    if other != sitting:
                        donor |= sittings == other
                bias = np.mean(labels[donor] - pred[donor], axis=0)
                sel = np.where(sittings == sitting)[0]
                out[sel] = np.clip(out[sel] + frac * bias, 0, 1)
    return out


def screen_weighted(pred: np.ndarray, labels: np.ndarray) -> float:
    """sum(s²) / sum(s), with s the screen-stretched miss used by scr."""
    err = pred - labels
    screen = np.hypot(err[:, 0] * ASPECT, err[:, 1])
    total = float(screen.sum())
    if total <= 0:
        return 0.0
    return float(np.sum(screen * screen) / total)


def main() -> None:
    pred, labels, sittings, groups, _names = build_pred()
    dots = np.asarray(TARGETS, dtype=float)
    order = list(dict.fromkeys(groups.tolist()))
    # Keep old poses first, then new ones.
    preferred = list(LIVE) + [name for name in order if name not in LIVE]
    order = [name for name in preferred if name in set(order)]

    print("| Group | Basic scr | Basic wScr | Sim scr | Sim wScr |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    rows = []
    basic_w = []
    sim_w = []
    for group in order:
        basic_scores = []
        sim_scores = []
        group_basic_w = []
        group_sim_w = []
        for sitting in dict.fromkeys(sittings[groups == group].tolist()):
            sel = np.where(sittings == sitting)[0]
            basic_scores.append(metrics(pred[sel], labels[sel]))
            sim_scores.append(metrics(pred[sel][edge_index(labels[sel])], dots))
            group_basic_w.append(screen_weighted(pred[sel], labels[sel]))
            group_sim_w.append(screen_weighted(pred[sel][edge_index(labels[sel])], dots))
        row = np.round(np.concatenate([np.mean(basic_scores, 0), np.mean(sim_scores, 0)]), 3)
        basic_mean = float(np.mean(group_basic_w))
        sim_mean = float(np.mean(group_sim_w))
        rows.append(row)
        basic_w.append(basic_mean)
        sim_w.append(sim_mean)
        print(
            f"| {group} | {row[3]:.3f} | {basic_mean:.3f} | {row[7]:.3f} | {sim_mean:.3f} |",
            flush=True,
        )
    print(
        f"| mean | {np.mean([row[3] for row in rows]):.3f} | {np.mean(basic_w):.3f} | {np.mean([row[7] for row in rows]):.3f} | {np.mean(sim_w):.3f} |",
        flush=True,
    )

    old = [LIVE[name] for name in order if name in LIVE]
    new = [rows[index] for index, name in enumerate(order) if name in LIVE]
    if old and new:
        imp = np.mean(np.asarray(old) - np.asarray(new), axis=0)
        print(f"| improvement | {imp[3]:+.3f} | {imp[7]:+.3f} |", flush=True)


if __name__ == "__main__":
    main()
