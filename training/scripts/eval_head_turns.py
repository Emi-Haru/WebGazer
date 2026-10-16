#!/usr/bin/env python3
"""Leave-one-sitting scores for upright sittings, head turns, and head tilts.

Center-only is the upright model from before the turned sittings. All-data
trains on the twelve upright sittings plus every turn and tilt, sets 1–12.
Predictions pass through the same edge stretch as the live red dot.
Simulated T. scores the photo closest to each bezel dot against that dot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_frame_probe import (  # noqa: E402
    FEATURE_ORDER,
    TARGETS,
    edge_index,
    extract,
    fit_tree,
    metrics,
)

# Known head directions. A missing set number is skipped so partial
# recordings (for example only sets 7–12 of a turn+tilt) can train.
POSE_GROUPS = [
    "rightTurn",
    "leftTurn",
    "up",
    "down",
    "downLeftTurn",
    "downRightTurn",
    "upLeftTurn",
    "upRightTurn",
    "leftTilt",
    "rightTilt",
    "rightTurnRightTilt",
    "rightTurnLeftTilt",
    "leftTurnRightTilt",
    "leftTurnLeftTilt",
    "upRightTilt",
    "upLeftTilt",
    "downRightTilt",
    "downLeftTilt",
    "upLeftTurnLeftTilt",
    "upLeftTurnRightTilt",
    "upRightTurnLeftTilt",
    "upRightTurnRightTilt",
    "downLeftTurnLeftTilt",
    "downLeftTurnRightTilt",
    "downRightTurnLeftTilt",
    "downRightTurnRightTilt",
]
DOWNLOADS = Path("/home/anni/Downloads/webgazer/random")
CENTER_RAW = ROOT / "data" / "personal" / "raw"
CACHE = ROOT / "logs" / "video_settled_turns.npz"
STRETCH = {"dead_x": 0.22, "dead_y": 0.17, "gain_x": 1.6, "gain_y": 1.8}


def stretch_axis(value: np.ndarray, dead: float, gain: float) -> np.ndarray:
    offset = value - 0.5
    magnitude = np.abs(offset)
    extra = np.maximum(magnitude - dead, 0) * (gain - 1)
    stretched = 0.5 + np.sign(offset) * (magnitude + extra)
    return np.clip(stretched, 0, 1)


def apply_stretch(pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[:, 0] = stretch_axis(out[:, 0], STRETCH["dead_x"], STRETCH["gain_x"])
    out[:, 1] = stretch_axis(out[:, 1], STRETCH["dead_y"], STRETCH["gain_y"])
    return out


def manifest_rows(manifest: Path, group: str, number: int) -> list[dict]:
    base = manifest.parent
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            {
                "image_path": str((base / record["image"]).resolve()),
                "label_x": float(record["label_x"]),
                "label_y": float(record["label_y"]),
                "group": group,
                "n": number,
                "sitting": f"{group}{number}",
            }
        )
    return rows


def load_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for number in range(1, 13):
        manifest = CENTER_RAW / f"session_20260929_center{number}" / "manifest.jsonl"
        rows.extend(manifest_rows(manifest, "center", number))
    found: dict[str, list[int]] = {}
    for group in POSE_GROUPS:
        for number in range(1, 13):
            matches = list((DOWNLOADS / f"random{number}" / f"{group}{number}").glob("session_*/manifest.jsonl"))
            if not matches:
                continue
            if len(matches) != 1:
                raise SystemExit(f"expected one manifest for {group}{number}, found {matches}")
            rows.extend(manifest_rows(matches[0], group, number))
            found.setdefault(group, []).append(number)
    for group, numbers in found.items():
        print(f"loaded {group}: {len(numbers)} sittings ({numbers[0]}–{numbers[-1]})", flush=True)
    return pd.DataFrame(rows)


def features_for(frame: pd.DataFrame) -> np.ndarray:
    wanted = frame.image_path.tolist()
    cached: dict[str, np.ndarray] = {}
    if CACHE.exists():
        stored = np.load(CACHE, allow_pickle=True)
        cached = {path: row for path, row in zip(stored["paths"].tolist(), stored["rows"])}
    missing_idx = [i for i, path in enumerate(wanted) if path not in cached]
    if missing_idx:
        print(f"extracting {len(missing_idx)} new photos", flush=True)
        batch = 200
        for start in range(0, len(missing_idx), batch):
            chunk = missing_idx[start : start + batch]
            fresh = extract(frame.iloc[chunk].reset_index(drop=True))
            for offset, index in enumerate(chunk):
                cached[wanted[index]] = fresh[offset]
            partial = np.vstack([cached[path] for path in wanted if path in cached])
            np.savez(CACHE, rows=partial, paths=np.array([path for path in wanted if path in cached]))
            print(f"cached {start + len(chunk)}/{len(missing_idx)}", flush=True)
    rows = np.vstack([cached[path] for path in wanted])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, rows=rows, paths=np.array(wanted))
    print(f"features {len(wanted)}", flush=True)
    return rows


def score_split(pred: np.ndarray, labels: np.ndarray) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    pred = apply_stretch(pred)
    basic = metrics(pred, labels)
    chosen = edge_index(labels)
    dots = np.asarray(TARGETS, dtype=float)
    simulated = metrics(pred[chosen], dots)
    return basic, simulated


def loo(frame: pd.DataFrame, rows: np.ndarray, train_groups: set[str] | None) -> pd.DataFrame:
    labels = frame[["label_x", "label_y"]].to_numpy(float)
    records = []
    sittings = frame.sitting.to_numpy()
    groups = frame.group.to_numpy()
    for sitting in pd.unique(frame.sitting):
        test = sittings == sitting
        if train_groups is None:
            train = ~test
        else:
            train = np.isin(groups, list(train_groups)) & ~test
        model = fit_tree(rows[train], labels[train])
        basic, simulated = score_split(model.predict(rows[test]), labels[test])
        group = frame.loc[test, "group"].iloc[0]
        records.append(
            {
                "sitting": sitting,
                "group": group,
                "basic": basic,
                "simulated": simulated,
            }
        )
        print(
            f"{sitting} basic scr {basic[3]:.3f} sim scr {simulated[3]:.3f}",
            flush=True,
        )
    return pd.DataFrame(records)


def mean_block(scores: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in np.mean(np.asarray(scores), axis=0))


def report(title: str, table: pd.DataFrame, groups: list[str]) -> None:
    print(f"\n{title}", flush=True)
    print(f"{'group':8} {'kind':10} {'avg':7} {'w':7} {'sqr':7} {'scr':7}", flush=True)
    for group in groups:
        part = table[table.group == group]
        for kind in ("basic", "simulated"):
            avg, w, sqr, scr = mean_block(part[kind].tolist())
            print(f"{group:8} {kind:10} {avg:7.3f} {w:7.3f} {sqr:7.3f} {scr:7.3f}", flush=True)


def main() -> None:
    frame = load_frame()
    rows = features_for(frame)
    ok = np.isfinite(rows).all(axis=1)
    print(f"finite {int(ok.sum())}/{len(rows)}", flush=True)
    frame = frame.loc[ok].reset_index(drop=True)
    rows = rows[ok]

    center = loo(frame[frame.group == "center"].reset_index(drop=True), rows[frame.group == "center"], None)
    report("center only (before)", center, ["center"])

    after = loo(frame, rows, None)
    groups = ["center", *sorted(frame.group.unique().tolist())]
    groups = list(dict.fromkeys(groups))
    report("all sittings (after)", after, groups)


if __name__ == "__main__":
    main()
