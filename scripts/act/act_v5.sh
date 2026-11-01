#!/usr/bin/env bash
# ACT v5: v4 inputs (wrist + joints, close labelled from phase 2) + DAgger recovery episodes from v4 80k
# (scripts/policy/gen_pick_dagger.py -> data/act/pick_raw_dagger1). 120k steps; evals at 80k / 120k, n = 1024.
cd /home/yashchitambar/wt/policy-pick
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD MUJOCO_GL=egl
while ! grep -q "^batch 9:" runs/gen_dagger1.log || [ ! -f data/act/pick_raw_dagger1/summary.json ]; do sleep 60; done
rm -rf data/act/pick_v5 data/act/pick_dg1_s*
for i in 0 1 2 3 4 5 6 7; do
  .venv-act/bin/python scripts/policy/pick_to_lerobot.py --raw data/act/pick_raw_dagger1 --root data/act/pick_dg1_s$i --repo_id local/pick_dg1_s$i --shard $i --nshards 8 --encoder_threads 2 --cams wrist --grip_from_phase 2 > runs/pick_to_lerobot_dg1_s$i.log 2>&1 &
done
wait
.venv-act/bin/python scripts/policy/pick_merge_shards.py --shards data/act/pick_v4_s{0,1,2,3,4,5,6,7} data/act/pick_dg1_s{0,1,2,3,4,5,6,7} --root data/act/pick_v5 > runs/pick_merge_v5.log 2>&1 || { echo MERGE_FAIL >> runs/act_v5.log; exit 1; }
echo CONVERTED >> runs/act_v5.log
scripts/act/train_act.sh data/act/pick_v5 runs/P1-act-pick-v5 120000 > runs/P1-act-pick-v5.log 2>&1
echo TRAIN_DONE >> runs/act_v5.log
