"""Smoke test of the end-to-end chain (scripts/eval/eval_chain.py): 16 worlds, one batch.

Checks: the chain builds on one runtime (insertion env + pick task + harness share it), every stage
runs, states stay finite, the pick lifts >= 90 % of readers, the hold latches, the transport keeps the
reader in the hand (in-hand change < 10 mm), and at least half of the episodes seat. Needs the
perception weights (runs/perception/zed.pt, zed_presence.pt, wrist_v10w.pt) and the insertion policy
(runs/chain/P4-insert-proprio-v5-s0_last.pt), all outside git.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/checks/check_chain.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'eval'))
import pandas as pd  # noqa: E402

import eval_chain as E  # noqa: E402

a = E.parse(['--nworld', '16', '--seed', '123', '--name', 'check_chain'])
ch = E.Chain(a)
df = pd.DataFrame(ch.run_batch(a.seed))
df['e2e_class'] = E.classify(df)
print(df.e2e_class.value_counts().to_dict())
print(df[['zed_start_xy_mm', 'pick_ok', 'hold_latched', 'transport_inhand_dp_mm', 'pre_lateral_mm', 'pre_tilt_deg', 'insert_outcome']].round(2).to_string())
assert df.finite_end.all(), 'non-finite state'
assert df.pick_ok.mean() >= .9, 'pick'
assert df.hold_latched.mean() >= .9, 'hold'
assert (df.transport_inhand_dp_mm < 10).mean() >= .9, 'transport'
assert (df.e2e_class == 'success').mean() >= .5, 'end to end'
print('check_chain: PASS')
