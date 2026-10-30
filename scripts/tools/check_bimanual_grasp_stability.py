"""Measure reader motion relative to TCP in saved pick audits (no physics tuning)."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from blind_insertion.scenes import pc_bench as S

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("states", type=Path)
args = parser.parse_args()
profile = json.loads(Path("data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json").read_text())
model, layout = S.build_scene(detailed=False, bimanual=True, support_mode="free",
                             profile=profile, measured_support=True,
                             controller_profile="native_free_settle_20261028")
data = mujoco.MjData(model)
records = {}
with np.load(args.states) as saved:
    for phase in saved.files:
        data.qpos[:] = saved[phase]
        mujoco.mj_forward(model, data)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[layout.peg_qadr+3:layout.peg_qadr+7])
        reader_R = rotation.reshape(3, 3)
        tcp_R = data.site_xmat[layout.tcp_site].reshape(3, 3)
        center = data.qpos[layout.peg_qadr:layout.peg_qadr+3] + reader_R @ np.array([0, 0, layout.grasp_z[0]])
        records[phase] = (tcp_R.T @ (center-data.site_xpos[layout.tcp_site]), tcp_R.T @ reader_R)
if "close" not in records:
    raise ValueError("Audit must include a close reference state")
reference_p, reference_R = records["close"]
metrics = {}
for phase, (p, R) in records.items():
    metrics[phase] = {
        "center_shift_from_close_mm": float(1000*np.linalg.norm(p-reference_p)),
        "rotation_from_close_deg": float(np.rad2deg(Rotation.from_matrix(R @ reference_R.T).magnitude())),
    }
report = {
    "source": str(args.states),
    "scope": "Phase endpoint relative-pose audit; not continuous slip detection or hardware validation",
    "threshold_provenance": "Engineering diagnostic thresholds, not measured physical tolerances",
    "stable_at_saved_carry_endpoints": all(
        metrics[p]["center_shift_from_close_mm"] < 1 and metrics[p]["rotation_from_close_deg"] < 1
        for p in ("lift", "preinsert", "hold")),
    "phases": metrics,
}
output = args.states.with_name(args.states.stem.replace("_states", "")+"_grasp_stability.json")
output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
