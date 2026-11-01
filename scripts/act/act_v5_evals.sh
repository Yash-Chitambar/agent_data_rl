#!/usr/bin/env bash
cd /home/yashchitambar/wt/policy-pick
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1
ev(){ s=$1; tag=$2; n=$3; shift 3
  ck=runs/P1-act-pick-v5/checkpoints/$s/pretrained_model
  while [ ! -f $ck/model.safetensors ]; do sleep 60; done; sleep 60
  [ -f runs/P1-act-pick-v5/act_${s}_traced.pt ] || .venv-act/bin/python scripts/act/export_act.py --ckpt $ck --dataset data/act/pick_v5 --out runs/P1-act-pick-v5/act_${s}_traced.pt > runs/export_v5_$s.log 2>&1
  .venv/bin/python scripts/policy/pick_act_eval.py --policy runs/P1-act-pick-v5/act_${s}_traced.pt --episodes $n --nworld 256 --seed 30000 --episode_s 20 --out results/usbc/policy/pick/act_v5_$s "$@" > runs/act_eval_v5_${s}_$tag.log 2>&1
  echo "EVAL v5 $s $tag n=$n $(grep -E '"rate"' runs/act_eval_v5_${s}_$tag.log)" >> runs/act_v5_evals.log; }
ev 040000 ens 256
ev 080000 ens 1024
ev 080000 ac 1024 --retry 2 --auto_close
ev 120000 ens 1024
ev 120000 ac 1024 --retry 2 --auto_close
echo ALLDONE >> runs/act_v5_evals.log
