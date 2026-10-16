#!/usr/bin/env python3
"""Leave-one-sitting edge forest and boosted heads used by the red-dot mix.

Usage: rebuild_loo_edge.py SHARD NSHARD
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_head_turns import CACHE, load_frame  # noqa: E402
from export_corner_mix import expand  # noqa: E402
from export_frame_probe import LEAF, N_TREES  # noqa: E402


def main() -> None:
    shard = int(sys.argv[1])
    nshard = int(sys.argv[2])
    jobs = max(1, (os.cpu_count() or 4) // nshard)
    frame = load_frame()
    stored = np.load(CACHE, allow_pickle=True)
    if stored["paths"].tolist() != frame.image_path.tolist():
        raise SystemExit("feature cache is not aligned with the frame")
    rows = stored["rows"]
    if not np.isfinite(rows).all():
        raise SystemExit("non-finite features")
    labels = frame[["label_x", "label_y"]].to_numpy(float)
    sittings = frame.sitting.to_numpy()
    reach = np.maximum(np.abs(labels[:, 0] - 0.5), np.abs(labels[:, 1] - 0.5))
    weight = 1 + 4 * np.maximum(0, reach - 0.35)
    wide_x = expand(rows)
    order = list(dict.fromkeys(sittings.tolist()))
    edge = np.zeros((len(rows), 2))
    edgew = np.zeros((len(rows), 2))
    wide = np.zeros((len(rows), 2))
    owned = 0
    for number, sitting in enumerate(order):
        if number % nshard != shard:
            continue
        owned += 1
        test = sittings == sitting
        train = ~test
        model = ExtraTreesRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=LEAF,
            random_state=0,
            n_jobs=jobs,
        )
        model.fit(rows[train], labels[train], sample_weight=weight[train])
        edge[test] = model.predict(rows[test])
        for axis in (0, 1):
            boosted = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=0,
            )
            boosted.fit(rows[train], labels[train, axis], sample_weight=weight[train])
            edgew[test, axis] = boosted.predict(rows[test])
            wide_model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=0,
            )
            wide_model.fit(wide_x[train], labels[train, axis])
            wide[test, axis] = wide_model.predict(wide_x[test])
        if owned % 4 == 0 or number + 1 == len(order):
            print(f"shard {shard} {owned} sittings (through {number + 1}/{len(order)})", flush=True)
    out = ROOT / "logs" / f"loo_tilt_edge_{shard}.npz"
    np.savez(out, edge=edge, edgew=edgew, wide=wide)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
