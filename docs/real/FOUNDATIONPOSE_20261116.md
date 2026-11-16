# ZED PC pose with SAM 3 + FoundationPose (2026-11-16)

Task: a second, independent PC pose estimator from the ZED top camera: SAM 3 (facebook/sam3) segments the PC in the first frame, and FoundationPose (NVlabs, model-based) fits the measured case mesh to the RGB image and the stereo depth. It is for the rig test on 2026-11-17. Worker on mason, GPU 1, no rig access. **Nothing here has run on the arm.** No weights changed. No existing config changed. Sim numbers are sim only. Every number has its n.

## Summary

- A separate env (`fp`, conda, outside every repo venv) runs FoundationPose (refiner + scorer weights, nvdiffrast, pytorch3d) and SAM 3 (image model `sam3.pt`, gated, downloaded with the operator's token). Install script: `scripts/perception/zed_fp/install_fp_env.sh`. Weights and their sha256 are in "Setup".
- Pipeline (`scripts/perception/zed_fp/`): ZED rectified HD1080 **stereo pair** → SAM 3 text-prompt mask ("small silver box"; the prior box picks the detection) → SGBM depth (CPU) → FoundationPose register (12 upright hypotheses, 5 refiner passes, scorer) → 90-deg near-symmetry resolved against the prior yaw → **plane correction** (the PC stands on the table: the case centre slides along its camera ray to the known top height, and the case is levelled) → port 2 (x, y), yaw.
- **Sim truth** (CPU render of the calibrated ZED, stereo pair, n = 40 PC poses in dx -5..+25, dy -12..+25 mm, dyaw -5..+15 deg): full pipeline |dxy| **p50 0.76 mm, p90 1.17, max 1.80 mm**; yaw error -0.38 deg median, |dyaw| max 0.75 deg. SAM 3 mask IoU vs the truth mask p50 0.952 (min 0.477, n = 40).
- **Real frames**: only `data/zed_gt` has the right image (n = 18 frames, 2 spots). FP − edge fit: dx **-0.41 mm** median [-2.63, +1.23], dy **-1.57 mm** [-2.50, -0.68], yaw +0.60 deg [+0.03, +1.06]; |dxy| p50 1.74, max 3.36 mm. SAM 3 mask IoU vs the edge-fit pose render 0.928 [0.919, 0.933]. On repeated frames of one placement (n = 10) FP scatters more than the edge fit: sd x 0.67 / y 0.38 mm (edge fit 0.09 / 0.09 mm).
- **No real frame with both a stereo pair and the arm-derived port exists** (run folders keep the left image only). So FP vs the arm (the ZED_MAP truth method) is still open. Tomorrow's captures give this data (runbook).
- The real stereo depth on the silver lid reads **+6.5..+17.6 mm high** (n = 18; the panel face depth agrees with the edge fit within 0.8 mm). Raw FP follows that depth (port 4-11 mm off along the camera ray). The plane correction removes it. FP alone (no prior) is 180-deg ambiguous on real frames (10 of 18): the prior yaw (any net) is required.
- Runtime on a V100 (idle GPU, n = 18 real frames): **1.06 s per frame** p50 (SAM 3 0.40 s, SGBM 0.15 s CPU, FP 0.45 s), peak GPU memory 5.5 GB (torch), model load 24 s.
- Rig integration, off by default: config block **`zed_fp`** (`deploy/bimanual/zed_fp.py`, three small hooks in `chain.py`). Mode `shadow` saves 5 stereo pairs at the PC capture (and at the reader frame) for FP on mason after the session; mode `use` takes gated FP x, y from a running FP server. New configs: `bimanual_align12_wide_k7_fp_shadow.json` (chain = `bimanual_align11_wide_k7.json`), `bimanual_align12_wide_k7_fp_use.json` (needs approval). Conventions check `FAILED: none` on both. Mock-hardware check `deploy/checks/zed_fp_realio.py`: `FAILED: none`.
- The rig GPU model is not in the docs. Tomorrow: shadow only, FP runs on mason after the session.
- The SD reader: not done (no measured reader mesh; the reader is hidden by the left arm at the capture pose). The shadow config saves stereo pairs of the reader frame, so the server can fit it later.

## Setup (mason)

| Item | Value |
|---|---|
| Env | conda env `fp` (`/home/yashchitambar/miniforge3/envs/fp`), Python 3.12; runner `/home/yashchitambar/fp_work/fp_python` (sets `LD_LIBRARY_PATH`, `FP_ROOT`, `SAM3_CKPT`) |
| torch / torchvision | 2.7.1+cu126 / 0.22.1 (the cu126 build keeps sm_70 for the V100) |
| nvdiffrast | 0.4.0 (NVlabs/nvdiffrast `253ac4f`), built with CUDA 12.5 (gcc 13), arch 7.0 |
| pytorch3d | 0.7.9, built from source (about 30 min on mason). FoundationPose uses only `so3_exp_map` and `rotation_6d_to_matrix`; `scripts/perception/zed_fp/p3d_stub` has both (same formulas) and is used when pytorch3d is missing |
| warp-lang | 1.18.0+cu12 (the PyPI default 1.19 is a CUDA 13 build and refuses sm_70) |
| other | numpy 1.26.4 (SAM 3 needs < 2), setuptools < 80 (SAM 3 imports pkg_resources), open3d 0.20 (+ conda libusb), opencv 4.11, trimesh 5.2.1, kornia 0.9.1, timm 1.0.31 |
| FoundationPose code | NVlabs/FoundationPose `a1b694b` (`/home/yashchitambar/fp_work/FoundationPose`). mycpp is not built: the rotation grid is made without `mycpp.cluster_poses` (`estimate._rotation_grid_nocluster`) |
| SAM 3 code | facebookresearch/sam3 `eef6839`, installed editable |
| SAM 3 inference | fp16 autocast (the V100 has no fast bf16); crop of the table area before the 1008 px resize |

Weight files:

| File | Size | sha256 |
|---|---|---|
| FoundationPose refiner `weights/2023-10-28-18-33-37/model_best.pth` | 68.2 MB | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| refiner `config.yml` | 708 B | `28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686` |
| FoundationPose scorer `weights/2024-01-11-20-02-45/model_best.pth` | 190.2 MB | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
| scorer `config.yml` | 778 B | `a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f` |
| SAM 3 `sam3.pt` (facebook/sam3, revision `3c879f3`) | 3.45 GB | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| SAM 3 `config.json` | 25.8 KB | `4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f` |
| sim PC mesh `results/perception/fp/mesh_pc_sim.obj` (+ `.json`: mesh frame, height) | | `0348142f93a897b62e6a2a39a472c373b5f71657736ade1aad3320df39830111` |

The FoundationPose weights come from the Google Drive folder of the FoundationPose readme (`no_diffusion/`). Not used: SAM 3.1 (multi-object video tracking), sam-3d-objects (we have meshes).

Install on another machine (rig or server; about 20 min without pytorch3d, plus the Hugging Face login for SAM 3):

```bash
SKIP_P3D=1 bash scripts/perception/zed_fp/install_fp_env.sh $HOME/fp_work $HOME/miniforge3 /usr/local/cuda-12.5 "<arch: 7.0 V100, 8.6 RTX 30xx, 8.9 RTX 40xx>"
```

## Pipeline

Code: `scripts/perception/zed_fp/` (`geometry.py`, `stereo.py`, `segment.py`, `estimate.py`, `pipeline.py`; tools `server.py`, `eval_sim.py`, `eval_real.py`, `summary.py`). It runs only in the `fp` env.

1. **Input**: the rectified HD1080 left and right images of one ZED grab; calibration `real_cameras_duchamp3.json` (`top`: fx 1055.02, baseline 120.17 mm, `extrinsics_cam_in_right_base`).
2. **Mask (SAM 3)**: text prompts in this order, on one image encoding: "small silver box", "silver box", "white box", "computer", "small white object", "box". The prior box (the case mesh at the prior pose, projected, +12 px) must overlap a detection with box IoU >= 0.3; the detection with the highest score x IoU wins. "mini PC" gives no detection on real frames; the sim render needs the later prompts (flat white lid). The box-only prompt is not used: it segments the background inside the box (checked on 1 real and 3 sim frames).
3. **Depth**: OpenCV SGBM (block 5, HH mode, disparity 48-272 px) in a box around the mask. ZED SDK depth is off on duchamp3 (`DEPTH_MODE.NONE`: no neural depth model installed).
4. **FoundationPose register**: mesh = the real case (`geometry.pc_real_mesh`: 103.5 x 103 x 44.5 mm with feet 3.3 mm, rounded corners 6 mm, port 2 31.95 mm from the left edge, top at z 11.5 mm; vertex colours sampled from real ZED frames: lid (205, 205, 195), panel (112, 110, 110)). Hypotheses: upright, yaw every 30 deg (12). 5 refiner passes, then the scorer ranks.
5. **Near-symmetry**: the case footprint is almost square, and the scorer often prefers the 180-deg pose (real: 13 of 18 frames needed a turn). The pose is turned by the 90-deg multiple nearest to the prior yaw about the case centre. The prior is any net (the chain uses the gate net).
6. **Plane correction**: the case centre slides along its camera ray to the known height, and the case is levelled with FP's yaw. Reason: section "Real stereo depth". The raw FP height error is logged (`dz_mm`) and gated.
7. **Checks per frame** (logged and used by the `zed_fp` gate): FP score, IoU of the FP silhouette with the SAM 3 mask, SAM 3 score, raw height error, distance to the gate net and to the edge fit.

## Results

### Sim truth (n = 40)

Frames: `scripts/perception/fp_sim_render.py` (CPU MuJoCo of the deploy scene with the real PC look, calibrated ZED pose and intrinsics, principal point checked by the edge fit to 0.6 mm / 0.14 px rms; right camera = left + 120.17 mm along x; arms at the 2026-11-01 capture joints; reader moved per case). Seed 0: case 0 at (231, 398) mm, 4 deg; cases 1-39 at dx -5..+25, dy -12..+25 mm, dyaw -5..+15 deg from it. Mesh = the sim PC geoms (`mesh_pc_sim.obj`). Prior = truth + N(0, 5 mm) in x, y and N(0, 3 deg) in yaw (zed_v5 level). `results/perception/fp/sim_eval.json` (plane correction on), `sim_eval_noplanefix.json` (off).

| Variant (n = 40 each) | \|dxy\| p50 / p90 / max (mm) | dx median [min, max] | dy median [min, max] | yaw error median [min, max] (deg) | raw z error (mm, no plane fix) |
|---|---|---|---|---|---|
| oracle: render depth + truth mask | 0.94 / 1.19 / 1.43 | +0.62 [+0.34, +1.08] | +0.60 [+0.13, +1.10] | -0.35 [-0.81, +0.02] | -0.3 [-2.0, +1.3] |
| stereo depth + truth mask | 0.74 / 1.09 / 2.29 | +0.06 [-0.43, +2.02] | +0.66 [-0.08, +1.37] | -0.30 [-1.02, +0.10] | +0.3 [-1.4, +2.3] |
| **full: stereo + SAM 3 (rig pipeline)** | **0.76 / 1.17 / 1.80** | +0.00 [-0.43, +1.52] | +0.70 [-0.19, +1.28] | -0.38 [-0.75, +0.20] | +0.0 [-1.4, +2.3] |
| same, no plane correction | 0.79 / 1.18 / 1.37 | -0.03 [-0.63, +0.79] | +0.73 [-0.20, +1.29] | -0.38 [-0.75, +0.20] | |
| edge fit (geometry sim) from the same prior | 0.91 / - / 1.88 | -0.14 [-1.81, +0.82] | +0.75 [-0.02, +1.27] | -0.26 [-0.80, +0.24] | |

- 40 of 40 ok in every variant. FP needed a 180-deg turn in 19 of 40 (full).
- SAM 3 mask IoU vs the truth mask (render segmentation, largest component): p50 **0.952**, min 0.477 (n = 40; the minimum is a mask that missed part of the case; FP was still within 1.8 mm). vs the edge-fit pose render (full silhouette, no occlusion): p50 0.866 [0.439, 0.949]. The truth mask itself has IoU 0.928 [0.844, 0.966] with the truth render (the grippers hide part of the case), so 0.87-0.93 is the ceiling for that comparison.
- The edge fit on these CPU renders is as good as FP, but its quality gate passes 0 of 40 (the panel edge of the sim render, `ZED_EDGEFIT_20261115.md`).
- In the sim the stereo depth is unbiased (raw z error within +/-2.3 mm). So the real height bias below is a property of the real lid, not of FP or SGBM in general.

### Real frames (data/zed_gt, n = 18)

The 18 stereo pairs of 2026-11-01 (spot 1: 8 frames, spot 2 and its two repeats: 10 frames, PC not moved between them; arms in 4 poses; reader on / off). Prior = the v1 net reading in `score_spot*.json` (else `spots.json`). Edge fit (geometry real) from the same prior. `results/perception/fp/real_zed_gt.json`, overlays `results/perception/fp/real_vis/`.

| | n | median [min, max] |
|---|---|---|
| FP − edge fit, dx (mm) | 18 | **-0.41** [-2.63, +1.23] |
| FP − edge fit, dy (mm) | 18 | **-1.57** [-2.50, -0.68] |
| FP − edge fit, yaw (deg) | 18 | +0.60 [+0.03, +1.06] |
| \|FP − edge fit\| (mm) | 18 | 1.74 [0.78, 3.36] |
| FP score | 18 | 117.5 [109.9, 124.0] |
| IoU FP silhouette / SAM 3 mask | 18 | 0.937 [0.915, 0.946] |
| IoU SAM 3 mask / edge-fit pose render | 18 | **0.928** [0.919, 0.933] |
| IoU FP render / edge-fit render | 18 | 0.976 [0.953, 0.981] |
| SAM 3 score ("small silver box" in 18 of 18) | 18 | 0.86 [0.61, 0.95] |
| raw FP height − 11.5 mm (before the plane correction) | 18 | **+9.0** [+6.5, +17.6] |
| FP tilt (deg, before levelling) | 18 | 2.5 [0.7, 4.3] |
| stereo depth valid on the mask | 18 | 0.89 [0.79, 0.94] |

Repeatability and moves:

| | n | FP sd x / y (mm), yaw (deg) | edge fit sd x / y (mm), yaw (deg) |
|---|---|---|---|
| spot 2 + repeats (PC not moved) | 10 | 0.67 / 0.38, 0.23 | 0.09 / 0.09, 0.05 |
| spot 1 (PC not moved; arms in 4 poses) | 8 | 1.16 / 0.57, 0.20 | 0.44 / 0.65, 0.09 |
| move spot 1 → spot 2 (operator: slid about 20 mm toward the arms) | 8 + 10 | x -19.2, y +0.7 mm, yaw +0.4 deg | x -20.5, y +0.1 mm, yaw +0.6 deg |

Against the arm-derived port: not possible on these frames (no align measurement). The edge fit reads +0.69 mm in y and -0.02 mm in x against the arm (n = 41, `ZED_EDGEFIT_20261115.md`). If that holds here, FP reads about -0.9 mm in y and -0.4 mm in x against the arm. This is an inference, not a measurement.

Without a prior (SAM 3 text only, raw FP yaw), n = 18: raw yaw within 10 deg of the edge fit on 6, near 180 deg on 10, near 90 deg on 2. So FP needs the prior yaw on this near-square case.

### Real stereo depth on the lid

On 2 real frames the SGBM depth of the lid reads the top 10.7-14.0 mm above 11.5 mm with every block size (3-15) and mode (HH, 3WAY), while the panel face lies at the edge-fit x within -0.7..+0.8 mm. SIFT matches on the lid stickers read only about 0.5 px (2.5 mm) high. The lid is brushed silver with low texture; SGBM locks on reflections there. The table height under the case is consistent with the edge fit (arm-derived port height -7.2..-8.6 mm, `ZED_MAP_20261114.md`). The plane correction uses the known top height (11.5 mm, as the edge fit). It moves the result by 2-8 mm (raw FP − edge fit: dx -4..-11 mm before, -2.6..+1.2 mm after). A better stereo method (a learned stereo net) could replace it; not done.

### FoundationPose settings (real frames, n = 18 each)

`eval_real.py` variants, `results/perception/fp/variants/` (the default row is a re-run of the main result: FoundationPose on the GPU is not bitwise repeatable, the medians move by up to 0.2 mm). FP time in these rows ran next to dry runs on the same GPU (not a clean timing).

| Setting | n | FP − edge dx (mm) | dy (mm) | yaw (deg) | \|FP − edge\| p50 / max (mm) | sd x / y (mm) / yaw (deg), spot 2 (n = 10) | same, spot 1 (n = 8) | quarter turns 0/1/2/3 | FP s p50 |
|---|---|---|---|---|---|---|---|---|---|
| default: upright, 30 deg grid (12), 5 passes, plane correction | 18 | -0.24 [-2.74, +1.23] | -1.55 [-2.44, -0.68] | +0.56 [+0.03, +1.02] | 1.76 / 3.50 | 0.67 / 0.37 / 0.24 | 1.20 / 0.51 / 0.21 | [5, 0, 12, 1] | 0.45 |
| upright, 15 deg grid (24) | 18 | -0.41 [-3.21, +2.72] | -1.28 [-2.10, -0.45] | +0.42 [-0.53, +0.84] | 1.78 / 3.83 | 1.09 / 0.25 / 0.35 | 1.13 / 0.72 / 0.15 | [8, 7, 1, 2] | 3.34 |
| prior yaw +/-30 deg in 15 deg steps (5) | 18 | -0.45 [-2.53, +0.63] | -1.45 [-2.48, -0.61] | +0.48 [+0.07, +1.09] | 1.86 / 3.43 | 0.67 / 0.49 / 0.32 | 1.02 / 0.63 / 0.18 | [18, 0, 0, 0] | 0.92 |
| 12 hypotheses, 10 refiner passes | 18 | -0.37 [-2.45, +1.18] | -1.53 [-2.22, -0.57] | +0.58 [-0.09, +1.22] | 1.65 / 3.02 | 0.63 / 0.35 / 0.28 | 1.12 / 0.47 / 0.23 | [7, 0, 11, 0] | 1.14 |
| default without the plane correction | 18 | -6.40 [-11.40, -4.32] | -1.86 [-2.73, -0.99] | +0.56 [+0.03, +1.02] | 6.80 / 11.48 | 0.89 / 0.38 / 0.24 | 2.09 / 0.50 / 0.21 | [5, 0, 12, 1] | 0.91 |

No setting is clearly better: 10 refiner passes lower the max |FP − edge| from 3.50 to 3.02 mm at 2.5x the time. Prior-centred hypotheses remove the turns (all k = 0) but not the scatter. The default stays (upright 30 deg, 5 passes).

### Runtime and memory

Idle V100 (GPU 1), n = 18 real frames, settings as above: total **1.06 s** per frame p50 (max 1.75 s on the first frame): SAM 3 0.40 s (3 prompts at most), SGBM 0.15 s (CPU, box around the mask), FoundationPose 0.45 s (12 hypotheses x 5 refiner passes + scorer). Peak torch GPU memory 5.5 GB (SAM 3 + FoundationPose in one process). Model load 24 s (mostly SAM 3). With 5 pairs per capture: about 5 s per capture on mason.

## Rig integration (off by default)

### The block `zed_fp`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | absent or false = off: nothing is patched or saved, the chain is unchanged |
| `mode` | `shadow` | `shadow`: save (and log FP if a server answers), the chain pose is unchanged; `use`: the chain takes FP x, y when the FP gate passes |
| `server` | `none` | `none`: save only, FP runs after the session; `wait`: wait for `result.json` from a running `server.py` (required by `use`) |
| `timeout_s` | 30 | wait for the result (the arms hold at the capture pose) |
| `n_pairs` | 1 | stereo pairs saved per capture (the server can take the median) |
| `save_reader` | false | also save the pairs of the reader frame (left arm parked), save only |
| `yaw_source` | `gate` | `gate` = the chain's yaw (the gate net, as `wide_k7`), `fp` = FP's yaw |
| `prior` | `gate` | FP prior = the gate net pose |
| gate (use) | | FP ok; score >= 80 (real 109.9-124.0); IoU FP/SAM >= 0.85 (real >= 0.915); SAM 3 score >= 0.3; \|raw height error\| <= 25 mm (real <= 17.6); \|FP − edge fit\| <= 4 mm and 2 deg (real <= 3.4 mm, 1.1 deg) when the edge fit ran and passed; \|FP − gate net\| <= 8 mm (`max_shift_mm`, as the edge fit) and 4 deg. A failed gate or a timeout fails the capture (no fall-back) |

Rig path: `deploy/cameras.ZedTop` retrieves only the left view. When the block is on, `zed_fp.py` wraps the `_grab` of the **one ZedTop instance** (not the class or the file): while a capture asks for it, the camera thread also retrieves the right view of the same grab (about 2 ms per frame). Each pair is saved as lossless PNG with `request.json` (prior = gate net, edge fit, arms, frame tag) in `<run>/zed_fp/<tag>/` (`left.png`, `right.png`, `left_<i>.png`, ...). Cost at the capture pose: about 1.7 s for 5 pairs (PNG write 0.13 s per image, measured on mason). In the dry run SimIO has no HD1080 frames, so `zed_fp.py` renders the pair on the CPU from the sim state (same camera code as `fp_sim_render.py`).

### What changed

| File | Change |
|---|---|
| `deploy/bimanual/zed_fp.py` (new) | block parsing, stereo grab (rig hook, dry-run render), request / result files, FP gate, chain hooks |
| `deploy/bimanual/chain.py` | `RealChain.__init__`: parse the block, log a `zed_fp` event. `capture_pc`: calls `zed_fp.apply` after the edge fit when the block is on. `reader_frame`: calls `zed_fp.save_reader` when `save_reader` is on. 13 lines; nothing changes when the block is absent |
| `deploy/checks/zed_fp_realio.py` (new) | mock-hardware check (below) |
| `deploy/config/bimanual_align12_wide_k7_fp_shadow.json`, `..._fp_use.json` (new) | the two configs |
| `scripts/perception/zed_fp/` (new) | pipeline, server / batch, evals, summary, install script, pytorch3d stub |
| `scripts/perception/fp_sim_render.py` (new) | sim truth set (stereo pair, depth, masks, truth pose, mesh export) |

Not touched: `deploy/cameras.py`, `zed_edgefit.py`, every existing config, all weights.

### Configs

| Config | Content | Conventions |
|---|---|---|
| `deploy/config/bimanual_align12_wide_k7_fp_shadow.json` | `bimanual_align11_wide_k7` + `zed_fp` {mode shadow, server none, n_pairs 5, save_reader true}. The chain is exactly `wide_k7` | FAILED: none |
| `deploy/config/bimanual_align12_wide_k7_fp_use.json` | `bimanual_align11_wide_k7` + `zed_fp` {mode use, server wait, timeout 30 s, yaw_source gate}. **Needs approval**, a CUDA GPU and the fp env next to the rig process | FAILED: none |

Note: another worker added `bimanual_align12_wide_k7.json` / `_wide_k4.json` today. My configs are built on `bimanual_align11_wide_k7.json`, as the task asked. If `align12_wide_k7` becomes the base, the `zed_fp` block copies over unchanged.

No safety limit changes. In shadow nothing the chain sends changes. In use, the approach start moves by at most 4 mm from the edge fit (and at most 8 mm from the gate net, as with the edge fit today).

### Checks

- `deploy/checks/zed_fp_realio.py` (mock arms; a fake ZED SDK under the real `ZedTop._grab` code serving the zed_gt spot 1 `capture_reader` stereo pair; `results/perception/fp/zed_fp_realio.log`): block off → no folder, no hook; shadow → hook on, 5 pairs saved, the saved pair equals the served pair pixel for pixel, request complete, chain pose = edge fit = `wide_k7`, FP on the saved pair ok, |FP − edge fit| 1.54 mm, median of 5 pairs ok, reader frame: 5 pairs saved; use with a live server → port = FP x, y, yaw = gate net yaw, port frame from FP, ok = all gates (wait 6.9 s including the server's first-call warm-up); use without a server → capture rejected after the 3 s timeout. **`FAILED: none`.**
- Existing checks still pass with the new chain code: `deploy.checks.bimanual_realio_mock` and `deploy.checks.zed_edgefit_realio`: `FAILED: none`.

### Sim dry runs

`deploy.run_bimanual --dry-run --stage full --align-mode full`, GPU 1, `--out results/perception/fp/dry` (first configs) and `results/perception/fp/dry_final` (final shadow config). FP by `server.py --geoms sim` (sim mesh); errors against the sim truth. n = 1 per row. Sim only.

| Seed | Config | PC truth (mm, deg) | net err (mm) | edge fit err x, y (mm) | FP err x, y (mm), yaw (deg) | Used | FP gate, wait | Result |
|---|---|---|---|---|---|---|---|---|
| 7043 | fp_shadow | (215.3, 363.6), 1.7 | 1.82 | -1.31, -0.31 | -1.33, +0.17, +0.09 | edge fit (as wide_k7) | - (shadow), 0.5 s save | align stop "after the press: step limit" (the known sim-noise class, 7 of 67 K7 grid runs in `WIDE_PLACEMENT_READY_20261116.md`) |
| 7045 | fp_shadow | (207.7, 368.9), -14.3 | 2.13 | -0.86, -1.18 | -0.80, -0.48, -0.11 | edge fit | -, 0.5 s | seated, true seated after the release |
| 7043 | fp_use | same | 1.82 | -1.31, -0.31 | -1.33, +0.17, +0.09 | FP x, y + gate yaw | ok, 3.1 s | seated, true seated after the release |
| 7045 | fp_use | same | 2.13 | -0.86, -1.18 | -0.80, -0.48, -0.11 | FP x, y + gate yaw | ok, 2.0 s | seated, true seated after the release |
| 7043 | fp_shadow final (n_pairs 5, save_reader) | (215.3, 363.6), 1.7 | 1.82 | -1.31, -0.31 | -1.33, +0.19, +0.08 | edge fit | -, reader frame saved | seated, true seated after the release |
| 7045 | fp_shadow final | (207.7, 368.9), -14.3 | 2.13 | -0.86, -1.18 | -0.76, -0.51, -0.18 | edge fit | -, reader frame saved | seated, true seated after the release |
| 7047 | fp_shadow final | (236.9, 374.4), 9.5 | 2.80 | -0.51, -0.34 | -1.08, -0.36, +0.34 | edge fit | -, reader frame saved | align stop "after the press: step limit" (same sim-noise class) |

The dry runs cover 3 distinct PC poses (seeds 7043, 7045, 7047; n = 3): FP − truth |dxy| 0.92-1.34 mm, yaw within 0.34 deg; edge fit 0.61-1.46 mm. 5 of 7 full runs seated; the 2 align stops are the known sim-noise class (they also occur with `wide_k7`, and shadow does not change what the chain sends). No guard or safety stop. The CPU render is deterministic, so FP gives the same pose for the same seed in both configs (to 0.05 mm). Use mode was also dry-run at the capture only (seed 7045, wait 6.3 s on the first call).

## Limits

- **No real truth for FP yet.** On real frames FP is compared with the edge fit only (n = 18, one session, 2 spots, yaw 1-2 deg). FP and the edge fit share the camera calibration and the case geometry (103.5 x 103 mm, top 11.5 mm). An error of either one moves both.
- The plane correction uses the known top height. So FP's real height is not measured, and FP is not independent of the table model. The raw stereo height bias (+6.5..+17.6 mm) is not explained beyond "low-texture silver lid".
- FP scatters more than the edge fit on repeated real frames (sd 0.4-1.2 mm against 0.1-0.65 mm). Use the median of several pairs. A systematic -1.6 mm dy against the edge fit remains (cause unknown: mesh port offset, panel colour or the mask).
- FP needs a prior yaw within +/-45 deg (the 4-fold near-symmetry). Real raw FP was 180 deg off on 10 of 18 frames without it.
- The gate thresholds come from 18 real frames of one day and 40 sim frames.
- Sim: the CPU render has no camera DR, no image noise and ideal stereo; the sim mesh and the real mesh differ (102.5 x 97 mm vs 103.5 x 103 mm).
- The SAM 3 text prompts are tuned on these frames ("mini PC" finds nothing). A cable or a hand on the case can change the mask. The prior-box rule rejects detections that do not overlap the expected case.
- The rig GPU is unknown. `use` needs the fp env on a CUDA GPU with about 6 GB free next to the rig nets, and 24 s of model load before the arms connect. Not tested on the rig.
- The reader is not done.

## Rig runbook (2026-11-17)

Goal: at every taped PC pose of the wide runbook (`WIDE_PLACEMENT_READY_20261116.md`, part A P0-P6, and the part B full runs), save ZED stereo pairs so that mason can measure the true pose with FoundationPose and compare it with the edge fit, the nets and the arm. **Shadow only. No approval needed for the capture part: the chain is `bimanual_align11_wide_k7.json`.**

### Before

```bash
git pull     # brings deploy/bimanual/zed_fp.py, the chain hooks, the two configs
.venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/bimanual_align12_wide_k7_fp_shadow.json | tail -1   # FAILED: none
.venv/bin/python -m deploy.run_bimanual --dry-run --stage capture --seed 7043 --config deploy/config/bimanual_align12_wide_k7_fp_shadow.json --name dry_fp
#   expect the line "zed_fp: stereo pair saved (zed_fp/zed_pc_<k>), deferred ..." and a folder <run>/zed_fp/zed_pc_<k>/ with left.png, right.png, request.json
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv    # write the rig GPU into the session doc (for a later 'use' decision)
```

### Part A with stereo (replaces part A of the wide runbook; same poses, same tape procedure)

At each pose P0-P6 (the tape notes stay the truth for the move):

```bash
S='{"port_xy": [<x m>, <y m>], "yaw_deg": <d>, "method": "tape from B", "unc_mm": 1}'
.venv/bin/python -m deploy.run_bimanual --stage capture --config deploy/config/bimanual_align12_wide_k7_fp_shadow.json --name fpcap_P<n> --pc-spot "$S"
```

Read, in addition to the wide runbook lines:
- `zed_fp: stereo pair saved (zed_fp/zed_pc_<k>), deferred ...` once per capture, and `zed_fp: reader frame stereo pairs saved: 5 (...)`.
- Pass per pose (in addition to the wide runbook's pass rules): both lines appear; the capture takes at most about 4 s longer than with `wide_k7`; no CAN, motor or camera-age error during the capture.
- Stop using the fp config (go back to `bimanual_align11_wide_k7.json`, the same runbook): `zed_fp: ... no stereo pair`, a camera-age error, or a CAN watchdog warning at the capture. Note it.

Optional extra truth (1 minute per pose, no motion): with the chain stopped and `scripts/real/estop.py` running, `.venv/bin/python scripts/real/snapshot.py capture --out data/zed_fp_20261117 --tag P<n>` saves a rectified stereo pair (`P<n>_zed_left.png`, `P<n>_zed_right.png`) and the joints, as in `data/zed_gt`. On mason: `eval_real.py --root <that folder> --init <tape x m> <tape y m> <tape yaw deg> --out ...`.

### Part B with stereo

Use `bimanual_align12_wide_k7_fp_shadow.json` in place of `bimanual_align11_wide_k7.json` for the part B full runs (only after the approvals A1-A5 of the wide runbook). The chain is identical. These runs give the arm-derived port (align measurements of the right wrist camera) together with the stereo pairs: this is the real FP truth test.

### What the server needs back

The run folders as usual (`data/real_runs/2026-11-17/`, with the new `zed_fp/` subfolders: 10 PNG per capture and 10 per reader frame, about 60 MB per capture), the tape notes per pose, and the `nvidia-smi` line. For a later reader fit (optional, 2 minutes): calipers of the reader's **white housing** as seen from above (length, width, thickness) and the black neck (length to the plug, width). The profile has only the silver inset (34.6 x 29.6 mm); on the ZED frames the white part is about 44 x 36 mm.

### On mason after the session

```bash
CUDA_VISIBLE_DEVICES=1 /home/yashchitambar/fp_work/fp_python scripts/perception/zed_fp/server.py \
    --root /home/yashchitambar/peg_test/data/real_runs/2026-11-17 --once --geoms real --all-pairs \
    --summary results/perception/fp/rig_20261117.json
PYTHONPATH=. .venv/bin/python scripts/perception/zed_arm_port.py --dates 2026-11-17 --out results/perception/fp/arm_port_20261117.json
```

Pass criteria for FP as the PC truth (decide before a `use` request):
- FP ok on every capture with the PC fully visible; the median of 5 pairs has a per-capture sd <= 0.7 mm;
- FP pose change between taped poses = the tape move within 1.5 mm and 1 deg;
- FP − arm-derived port (part B runs) within 1.5 mm in x and y, n >= 3;
- if the edge fit also passes these, keep the edge fit in the chain (faster, on the rig CPU) and use FP as the offline truth.

## Approvals needed

1. None for tomorrow's shadow captures (`bimanual_align12_wide_k7_fp_shadow.json`: the chain sends the same as `bimanual_align11_wide_k7.json`; the capture pose holds about 2 s longer per capture and per reader frame).
2. `bimanual_align12_wide_k7_fp_use.json` (FP x, y in the chain): only after the pass criteria above, with the fp env installed on a rig-side GPU and the server started before the arms connect. It moves the approach start by up to 4 mm from the edge fit.

## Files

- `scripts/perception/zed_fp/` (`geometry.py`, `stereo.py`, `segment.py`, `estimate.py`, `pipeline.py`, `server.py`, `eval_sim.py`, `eval_real.py`, `summary.py`, `install_fp_env.sh`, `p3d_stub/`), `scripts/perception/fp_sim_render.py`
- `deploy/bimanual/zed_fp.py`, `deploy/bimanual/chain.py` (3 hooks), `deploy/checks/zed_fp_realio.py`
- `deploy/config/bimanual_align12_wide_k7_fp_shadow.json`, `deploy/config/bimanual_align12_wide_k7_fp_use.json`
- `results/perception/fp/`: `sim_eval.json`, `sim_eval_noplanefix.json`, `real_zed_gt.json`, `real_vis/`, `sim_vis/`, `variants/`, `summary.json`, `mesh_pc_sim.obj/.json`, `zed_fp_realio.log`, `conventions_*.json`, `dry/`, `dry_final/`
- Env and data outside the repo (mason): `/home/yashchitambar/fp_work/` (FoundationPose, weights, sam3, sam3_ckpt, `fp_python`, sim truth set `sim/`)
