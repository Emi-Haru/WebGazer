#!/bin/bash
set -euo pipefail
cd /home/anni/Projects/WebGazer
PY=training/.venv/bin/python
LOG=training/logs/rebuild_wscr.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -Is)"

$PY - << 'PY'
import sys
sys.path.insert(0, "training/scripts")
from eval_head_turns import features_for, load_frame
frame = load_frame()
rows = features_for(frame)
print("feature rows", len(rows), flush=True)
PY

for shard in 0 1 2 3; do
  $PY -u training/scripts/rebuild_loo_parts.py "$shard" 4 > "training/logs/loo_parts_${shard}.log" 2>&1 &
done
wait

$PY - << 'PY'
import numpy as np
from pathlib import Path
logs = Path("training/logs")
shards = [np.load(logs / f"loo_tilt_parts_{i}.npz", allow_pickle=True) for i in range(4)]
keys = ["forest", "forest_w", "forest_x", "group_forest", "group_ridge", "chair", "glob", "proba", "anchors"]
out = {k: sum(s[k] for s in shards) for k in keys}
base = shards[0]
for k in ["sit_of", "label_x", "label_y", "gidx", "names", "ipd", "face", "pose"]:
    out[k] = base[k]
empty = np.abs(out["forest"]).sum(1) == 0
print("parts empty", int(empty.sum()), "shape", out["forest"].shape, flush=True)
if empty.any():
    raise SystemExit("some rows have no forest prediction")
np.savez(logs / "loo_tilt_parts.npz", **out)
print("merged parts", flush=True)
PY

for shard in 0 1 2 3; do
  $PY -u training/scripts/rebuild_loo_edge.py "$shard" 4 > "training/logs/loo_edge_${shard}.log" 2>&1 &
done
wait

$PY - << 'PY'
import numpy as np
from pathlib import Path
logs = Path("training/logs")
shards = [np.load(logs / f"loo_tilt_edge_{i}.npz") for i in range(4)]
keys = ["edge", "edgew", "wide"]
out = {k: sum(s[k] for s in shards) for k in keys}
empty = np.abs(out["edge"]).sum(1) == 0
print("edge empty", int(empty.sum()), "shape", out["edge"].shape, flush=True)
if empty.any():
    raise SystemExit("unfilled edge rows")
np.savez(logs / "loo_tilt_edge.npz", **out)
print("merged edge", flush=True)
PY

$PY -u training/scripts/score_mix_table.py
echo "done $(date -Is)"
