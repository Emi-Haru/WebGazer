#!/usr/bin/env python3
"""Leave-one-sitting parts for the red-dot mix (forests, ridges, router, chairs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_head_turns import load_frame, features_for  # noqa: E402
from export_corner_mix import expand  # noqa: E402
from export_frame_probe import LEAF, N_TREES  # noqa: E402

POSE = [6, 7, 8, 9, 10, 11, 14, 15]
WEAK = {"rightTilt", "leftTilt", "up", "upLeftTurn", "downLeftTurn"}
OUT = ROOT / "logs" / "loo_tilt_parts.npz"


def main() -> None:
    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    jobs = max(1, (os.cpu_count() or 4) // nshard)
    frame = load_frame()
    rows = features_for(frame)
    ok = np.isfinite(rows).all(1)
    frame = frame.loc[ok].reset_index(drop=True)
    rows = rows[ok]
    labels = frame[["label_x", "label_y"]].to_numpy(float)
    groups = frame.group.to_numpy()
    sittings = frame.sitting.to_numpy()
    close = frame.n.to_numpy() <= 6
    names = sorted(set(groups.tolist()))
    name_i = {name: index for index, name in enumerate(names)}
    gidx = np.array([name_i[group] for group in groups])
    features = expand(rows)
    n_rows = len(rows)
    n_groups = len(names)

    forest = np.zeros((n_rows, 2))
    forest_w = np.zeros((n_rows, 2))
    forest_x = np.zeros((n_rows, 2))
    group_forest = np.zeros((n_rows, n_groups, 2))
    group_ridge = np.zeros((n_rows, n_groups, 2))
    chair = np.zeros((n_rows, 2, 2))
    glob = np.zeros((n_rows, 2))
    proba = np.zeros((n_rows, n_groups))
    order = list(dict.fromkeys(sittings.tolist()))
    anchors = np.zeros((len(order), 2, n_groups, 2))
    sit_index = {}

    owned = 0
    for number, sitting in enumerate(order):
        sit_index[sitting] = number
        if number % nshard != shard:
            continue
        owned += 1
        test = sittings == sitting
        train = ~test

        def fit_forest(feat: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
            model = ExtraTreesRegressor(
                n_estimators=N_TREES,
                min_samples_leaf=LEAF,
                random_state=0,
                n_jobs=jobs,
            )
            model.fit(feat[train], labels[train], sample_weight=sample_weight)
            return model.predict(feat[test])

        forest[test] = fit_forest(rows)
        weight = np.ones(train.sum())
        weight[np.isin(groups[train], list(WEAK))] = 3
        forest_w[test] = fit_forest(rows, weight)
        forest_x[test] = fit_forest(features)

        for name, index in name_i.items():
            mask = train & (groups == name)
            if mask.sum() < 8:
                continue
            model = ExtraTreesRegressor(
                n_estimators=80,
                min_samples_leaf=LEAF,
                random_state=0,
                n_jobs=jobs,
            )
            model.fit(rows[mask], labels[mask])
            group_forest[test, index] = model.predict(rows[test])
            scaler = StandardScaler().fit(features[mask])
            ridge = Ridge(alpha=1).fit(scaler.transform(features[mask]), labels[mask])
            group_ridge[test, index] = ridge.predict(scaler.transform(features[test]))

        for room_index, room in enumerate((close, ~close)):
            mask = train & room
            if mask.sum() < 8:
                continue
            scaler = StandardScaler().fit(features[mask])
            model = Ridge(alpha=1).fit(scaler.transform(features[mask]), labels[mask])
            chair[test, room_index] = model.predict(scaler.transform(features[test]))
            part_g = gidx[mask]
            for index in range(n_groups):
                part = rows[mask][part_g == index]
                if len(part):
                    anchors[number, room_index, index] = [
                        float(np.median(part[:, 14])),
                        float(np.median(part[:, 15])),
                    ]

        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=1).fit(scaler.transform(features[train]), labels[train])
        glob[test] = model.predict(scaler.transform(features[test]))

        router = ExtraTreesClassifier(
            n_estimators=20,
            min_samples_leaf=8,
            random_state=0,
            n_jobs=jobs,
        )
        router.fit(rows[train][:, POSE], gidx[train])
        raw = router.predict_proba(rows[test][:, POSE])
        for col, cls in enumerate(router.classes_):
            proba[test, int(cls)] = raw[:, col]

        if owned % 4 == 0 or number + 1 == len(order):
            print(f"shard {shard} {owned} sittings (through {number + 1}/{len(order)})", flush=True)

    sit_of = np.array([sit_index[sitting] for sitting in sittings])
    out = OUT if nshard == 1 else OUT.with_name(f"loo_tilt_parts_{shard}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        forest=forest,
        forest_w=forest_w,
        forest_x=forest_x,
        group_forest=group_forest,
        group_ridge=group_ridge,
        chair=chair,
        glob=glob,
        proba=proba,
        anchors=anchors,
        sit_of=sit_of,
        label_x=labels[:, 0],
        label_y=labels[:, 1],
        gidx=gidx,
        names=np.array(names),
        ipd=rows[:, 14],
        face=rows[:, 15],
        pose=rows[:, [6, 7, 8, 9, 10, 11]],
    )
    print(f"saved {out} groups={names}", flush=True)


if __name__ == "__main__":
    main()
