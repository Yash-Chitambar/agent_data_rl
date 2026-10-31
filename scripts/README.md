# scripts/

Entry points that sit on top of the `blind_insertion` package. Run everything from the repo root
with `PYTHONPATH=$PWD` and `.venv/bin/python` (the shell wrappers set this up themselves).
Training, single evals and checks are package modules (`python -m blind_insertion.<module>`);
these scripts batch them and turn results into figures.

| Folder | Purpose | Main files |
|---|---|---|
| `eval/` | Batched evaluations of one checkpoint. Each writes Parquet files to `results/<phase>/`. | `phase_b.sh` (in-dist + 6 OOD bins), `phase_c.sh` (YAM robot conditions + gain sweep), `phase_e.sh` (USB-C final stage [+ capture radius + clearance sweep]), `phase_f.sh`, `phase_g.sh`, `ood.sh`, `usbc_*.sh` (USB-C latency, friction, stages, deployment grid), `usbc_capture.py` |
| `analysis/` | Analyses that need no training: probes, signal checks, replays, failure studies. | `run_probes.sh` (probe + latent swap for C/D/E policies), `run_signal_vs_size.sh` (A-S5/A-S6 per peg size), `run_replay_check.sh` (USB-C §11.5 replays), `c_failures.py` (why latency breaks the base policy) |
| `report/` | Results -> figures, report data and videos. | `phase_{b,c,d,g}.py` (per-phase tables and PNGs), `build_report.py` (report/static data), `make_demos.sh` + `build_demos_data.py` (example videos), `yam_example_*.{sh,py}`, `*_views.py` (renders) |
| `real/` | Real-arm tools (duchamp3): safety, rig console, cameras, calibration, measurement, step-test fits. Rig development happens on `real/duchamp3`. | See [real/README.md](real/README.md) |
| `server/` | Multi-GPU and SLURM runs: setup, job manifests, the job runner, sync. See `docs/operations/SERVER.md`. | `run_jobs.py`, `jobs/make_manifests.py`, `jobs/*.txt`, `bootstrap.sh`, `sync_{up,down}.sh` |
| `real2sim/` | Matching the sim to the rig: dynamics identification and replays, fingertip/camera fits, backdrop fit, real-vs-sim renders. | `identify_arm_dynamics.py`, `check_bimanual_dynamics_replay.py`, `fit_gripper_tips.py`, `fit_backdrop.py`, `render_sim_real_alignment.py`, `render_layout_views.py` |
| `task/` | Bimanual task design: placement search, scripted pick/insert audits and their renders. | `explore_pc_placement.py`, `audit_bimanual_pick.py`, `audit_bimanual_reach.py` |
| `checks/` | Regression checks of scenes, runtime, task, cameras and metrics (CPU and GPU). | `check_bimanual_runtime.py`, `check_bimanual_task.py`, `check_bimanual_warp.py`, `check_bimanual_scene.py` |
| `tools/` | Environment checks and the local single-GPU queue. | `check_setup.py`, `test_mjfit.py`, `local_queue.sh` |

## Typical workflow

```bash
export PYTHONPATH=$PWD
# 1. train (one GPU)
.venv/bin/python -m blind_insertion.train --name E-usbc-proprio-h10-s0 --family usbc --curriculum usbc \
    --n_variants 320 --env_file blind_insertion/configs/phase_e_usbc_spheres.json --env_json '{"history":10}'
# 2. evaluate (writes results/E/eval_h10_s0_final.parquet)
scripts/eval/phase_e.sh E-usbc-proprio-h10-s0 h10_s0 indist --env_json '{"history":10}'
# 3. analyse / plot
scripts/analysis/run_probes.sh
.venv/bin/python scripts/report/phase_d.py
```

Many GPU jobs at once: add lines to a manifest and run `scripts/server/run_jobs.py` (one job per free GPU).
