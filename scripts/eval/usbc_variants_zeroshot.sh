#!/usr/bin/env bash
# E1 sphere-pad policy (athena5) evaluated zero-shot on the E4 chamfer, E5 latch and E6 side-port
# assemblies, final-stage distribution (20-100 um, median start 3 mm), 2,048 matched starts each.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
CK=runs/E-usbc-proprio-currsph-s0/last.pt; H=$(python3 -c "print(round(3e-3/0.765, 7))")
E=(.venv/bin/python -m blind_insertion.eval --policy $CK --family usbc --n_variants 320 --episodes 2048 --peg_offset $H --log_worlds 0,1,2,3)
"${E[@]}" --env_file blind_insertion/configs/phase_e_usbc_spheres.json --out results/E/variants/curr_on_base.parquet
"${E[@]}" --env_file blind_insertion/configs/phase_e_usbc_spheres.json --variant_kw '{"chamfer": [2e-4, 5e-4]}' --out results/E/variants/curr_on_chamfer.parquet
"${E[@]}" --env_file blind_insertion/configs/phase_e_usbc_latch.json --out results/E/variants/curr_on_latch.parquet
"${E[@]}" --env_file blind_insertion/configs/phase_e_usbc_side.json --out results/E/variants/curr_on_side.parquet
