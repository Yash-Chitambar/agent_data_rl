# Instructions for the agent on the training server

You are taking over the remaining RL training of this project on a shared GPU server (first
target: **jensen**, 8 × V100, `jensen.ist.berkeley.edu`, user `yashchitambar`). Another agent and
Yash work on the local machine (athena5, 1 × RTX 3090 Ti). Read this file completely before you do
anything. When this file and another document disagree about the server, this file wins.

> **ACTION (2026-10-21 22:40 PDT, Yash's decision): redo every run that used the old backlash model.**
> The backlash hinges used MuJoCo's soft default joint limit, so every "backlash" level (0.01 to
> 5 arcmin) acted as ~0.6 deg of extra joint compliance. `main` fixes it (3006b51, stiff limits by
> default). Details: `docs/FINDINGS.md`, entry "BUG: the backlash model was a soft spring".
> - **Affected (47 runs, ~67 GPU-h on the 3090 Ti estimate, seed 0):** C backlash1/3/5, allmid, allmax, randall; all of
>   Phase D tier 1 and tier 2 (the random conditions include backlash); E `noise` (not started yet:
>   start it only on the fixed model). Plus the backlash columns of the other C policies (evals only,
>   `jobs/C_eval_bl.txt`, 11 jobs, ~0.7 GPU-h). Latency, noise, quantization, gain, B, F and G
>   results are not affected.
> - **Procedure:**
>   1. Stop the runner (Ctrl-C in its tmux window). Affected jobs that are running use the old model in
>      memory, so they must stop. Unaffected training jobs resume later from `last.pt`.
>   2. Merge `origin/main` into `mason-runs`.
>   3. `python3 scripts/server/redo_backlash.py` (dry run), check the list, then `--apply`. It moves
>      `runs/X` to `runs/_softbl/X` and the old results to `results/<phase>/_softbl/` (kept, not
>      deleted), and removes the affected commands from `runs/queue_server/done.txt`.
>   4. Move any other old-model result by hand to the same `_softbl/` folders (for example extra
>      Phase D evals that are not in the manifests).
>   5. Restart the runner with the same manifests plus `scripts/server/jobs/C_eval_bl.txt`, on the
>      GPUs Yash already approved (section 1 rules still apply: check capacity first).
>   6. Commit the archive moves. In FINDINGS and the report, mark the old numbers "(old backlash
>      model)" until the new ones replace them. Do not mix old and new runs in one table.
> - `add_backlash(..., legacy_soft_limit=True)` reproduces the old model if you must check an old run.

## 0. Context in five lines

- The project reproduces Bronars et al., *Blind Precise Assembly* (arXiv:2610.23849, PDF in `docs/`):
  a robot inserts parts using only joint data, trained with PPO in MuJoCo Warp.
- What to do and in which order: `docs/RESEARCH_PLAN.md`. Results and discoveries so far:
  `docs/FINDINGS.md`. Repo conventions and gotchas: `CLAUDE.md`. Server mechanics in depth:
  `docs/SERVER.md`.
- Phase A (checks) and most of Phase B (paper reproduction on the Flexiv arm) are done and match the
  paper. The server work is the rest of B, and Phases C (YAM arm), D (scale grid), E (USB-C) and F
  (USB-C with a wrist camera).
- Every job is one command on one GPU. The job lists are in `scripts/server/jobs/*.txt` (160 jobs with
  seed 0 for C-F, about 252 GPU-hours on the 3090 Ti; `make_manifests.py` prints the table).
- Nothing in this project needs a dataset. All training data comes from the simulator.

## 1. Rules for a shared server (Yash's rules; follow them exactly)

Other people use these GPUs. Treat every GPU as someone else's until a live check shows it is free.

1. **Check capacity before every GPU job, and again just before you launch:**
   ```bash
   nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
   ```
   A GPU is FREE only if it uses less than 1 GB and about 0 % compute over a few samples. Memory that
   is held with no visible process still belongs to someone.
2. **Use only FREE GPUs.** Pin every job with `CUDA_VISIBLE_DEVICES` (the job runner does this; pass
   `--gpus` with the free indices only). Never let a job take all GPUs by default.
3. **Never kill, suspend, renice or signal another user's process.** Never restart a shared service.
4. **No sudo and no system packages.** Install everything at user level (the bootstrap does this).
5. **Ask Yash before** you hold more than 2 GPUs, or hold any GPU for more than one day. The full
   job lists need both, so get Yash's explicit OK on the GPU count before section 5.
6. **Run long work in `tmux`**, never in the SSH session itself. Release GPUs when a job ends.
7. **Keep files in your own directories.** Check `df -h ~` before large writes.
8. **Never commit secrets** (tokens, keys) and never copy private keys between machines.

## 2. Setup (no training)

```bash
git clone --recurse-submodules https://github.com/Yash-Chitambar/peg_test.git ~/peg_test
cd ~/peg_test
bash scripts/server/bootstrap.sh 2>&1 | tee runs/bootstrap.log
```

The repo is private: `git clone` needs Yash's GitHub access on this machine (`gh auth login`, a
deploy key or a token). If it fails, stop and ask Yash; do not work around it.

What the bootstrap does (each step is skipped when already done):
- installs uv, Python 3.12 and the locked environment (`.venv`);
- **selects the CUDA stack**: CUDA 13 when every GPU has compute capability 7.5 or more and the driver
  is 580 or newer, else CUDA 12. **V100s are 7.0, so jensen uses CUDA 12**: torch `2.14.1+cu126` and
  warp-lang `1.18.0+cu12` replace the CUDA 13 wheels. The code and library versions are the same;
  only the compiled GPU binaries differ;
- sparse-clones the three Menagerie robots into `third_party/`;
- runs `scripts/server/verify.py` (imports, GPU visible to torch and Warp, EGL, a MuJoCo Warp step);
- runs about 2 minutes of smoke training and evaluation on small world counts.

**On a CUDA 12 machine, never run `uv sync` or `uv run` by hand.** They restore the CUDA 13 wheels.
Call `.venv/bin/python` directly, or run the bootstrap again. All scripts already do this.

The first run on a new GPU type compiles all Warp kernels (several minutes). Later runs use the cache
in `~/.cache/warp`.

Pick a FREE GPU for the smoke runs: `CUDA_VISIBLE_DEVICES=<free index> bash scripts/server/bootstrap.sh`.

## 3. Benchmark (no training)

On one FREE GPU:

```bash
bash scripts/server/benchmark.sh --gpu <free index>                 # blind, 12,288 worlds
bash scripts/server/benchmark.sh --gpu <free index> --visual        # visual, 4,096 worlds
```

Each writes `runs/_bench/<host>_<gpu>_<kind>_<nworld>.json` with the steps/s and the peak GPU memory.
Reference on the 3090 Ti: blind about 37,500 steps/s at 12,288 worlds (10.9 GB), visual about
8,500 steps/s at 4,096 worlds.

- If 12,288 worlds do not fit (16 GB V100s; OOM in the log), try `--nworld 8192`. A smaller world
  count changes the PPO batch size, which is a deviation from the paper. Record it in the deviations
  table of `docs/FINDINGS.md`, and ask Yash before you change the job lists.
- Copy the benchmark result into the compute-budget section of `docs/FINDINGS.md` (date, host, GPU,
  steps/s, memory, hours per 300M-step run).

## 4. Validation before the full batch

Results must not depend on the machine. Before the full batch, train one known configuration and
compare it with the two finished local seeds:

```bash
# on one FREE GPU, in tmux
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -m blind_insertion.train --name B-peg-proprio-base-s3 \
    --seed 3 --nworld 12288 --steps 300e6 > runs/B-peg-proprio-base-s3.log 2>&1
```

Pass criteria (local seeds 0 and 1, `docs/FINDINGS.md`): training success passes 90 % at 35–50M
steps and is at least 99 % from 150M steps. MuJoCo Warp is not bitwise deterministic, so compare
curves, not numbers. If the run is clearly worse, stop and report to Yash with the log; do not start
the batch.

## 5. The training batch (only after Yash's OK on the GPU count)

> **Update 2026-10-21 (main): Phases E and F are unblocked. Pull `main` into `mason-runs`, then run
> `make_manifests.py --check`.**
> - **Resume exists.** `train.py` / `train_visual.py --resume`; `run_jobs.py` and the SLURM script add
>   it at launch when a partial run has `last.pt` (manifest lines are unchanged). See `docs/SERVER.md` §9.
> - **Phase E uses sphere pads** (`configs/phase_e_usbc_spheres.json`, your table mount kept): with
>   box pads 11–14% of USB-C resets drop the drive (`report_usbc/`). `E_train.txt` now has E1, the E4
>   ablations (no contact, history 10/60, noise fixed/randomized, chamfer), E5 latch and E6 side port;
>   `E_eval.txt` has their evals and E1 zero-shot on chamfer/latch/side. `eval_phase_e.sh` now takes
>   `--env_file/--env_json/--variant_kw` and defaults to the sphere-pad config (it used box pads before).
> - **Phase F is runnable** (`F_train.txt`, `F_eval.txt`): five visual USB-C policies at 2,048 worlds,
>   150M steps, ~14 GPU-h each on a V100 (estimate). Measure the first job's memory and steps/s, and
>   ask Yash before you start F (it is the most expensive phase).
> - **C has one new line**, `C-peg-proprio-randall-s0` (all noise factors randomized per world, plan
>   §9.3); it is appended, so the running C batch is not affected.
> - The noise gate for D and E is unchanged (placeholder `{"q_noise": 1e-4}` until Gate C).

```bash
tmux new -d -s jobs "python3 scripts/server/run_jobs.py --gpus <free indices> --require-idle-gpus \
    --skip-tag also_in_local_queue --skip-tag needs_local_ckpt \
    scripts/server/jobs/B_train.txt scripts/server/jobs/B_eval.txt \
    scripts/server/jobs/C_train.txt scripts/server/jobs/C_eval.txt"
python3 scripts/server/run_jobs.py scripts/server/jobs/B_train.txt --status
```

- **Order:** B, then C (seed 0 of every condition first), then D tier 1, D tier 2, then E. The job
  lists already put seed 0 first. Run the evals of a phase after its training (the eval lines carry
  `after=` dependencies).
- **Lines tagged `also_in_local_queue` are skipped** by `--skip-tag also_in_local_queue`: athena5 trains those itself (blind s2, no-contact
  s1, USB-C E1 s0, and their evals), and its finished runs come to you only if Yash copies them. Use
  `--mark-complete <run name>` for a run that athena5 finished, so its evals can start here.
- **Lines tagged `needs_local_ckpt`** evaluate runs that were trained on athena5 (blind s1, no-contact
  s0, C base s0). `--skip-tag needs_local_ckpt` skips them; drop that flag once Yash has copied
  those run directories here.
- **Phase D and E use a placeholder noise model** (`{"q_noise": 1e-4}`). The final value comes from
  Phase C (Gate C in the plan). Do not start D or E before Yash or the local agent confirms the noise
  level. Then set `NOISE` in `scripts/server/jobs/make_manifests.py`, regenerate, and run
  `make_manifests.py --check`.
- **Phase F** (`F_train.txt`, `F_eval.txt`) comes after E, with Yash's OK.
- **Resume:** a crashed or interrupted run continues from its last checkpoint (every 25 iterations)
  when the runner retries it. To continue a run by hand, rerun its manifest line with `--resume`.
- **Check the first minutes of every new job type:** memory as in the benchmark, no OOM, and other
  users' GPUs not affected (capacity check again).

## 6. Monitoring

```bash
python3 scripts/server/run_jobs.py <manifests> --status       # done / running / failed / blocked
tail -n 3 runs/<run name>.log                                  # it, steps, succ, sps
cat runs/queue_server/failed.txt
```

If a job fails twice, read its log in `runs/queue_server/jobs/`. Fix only clear, local problems (a
typo in a manifest, a full disk). For anything that touches the method or results, stop that phase
and report to Yash.

## 7. Results and GitHub

Work on a branch, never directly on `main`:

```bash
git switch -c mason-runs        # once
```

Commit and push **after each phase's evaluations finish, and at least once a day:**

- `results/<phase>/`: eval parquets, probe JSON, figures (small files).
- Training logs: copy `runs/<name>/config.json` and `runs/<name>/train_log.csv` into
  `results/<phase>/train_logs/<name>/`. Do not commit checkpoints (`*.pt`); `runs/` is gitignored and
  stays on the server. athena5 pulls the checkpoints it needs with `scripts/server/sync_down.sh`.
- `docs/FINDINGS.md`: dated entries, newest first in each section. Record the host, GPUs, the
  benchmark, every deviation, failures and their causes, and the result tables.
- Code changes: only on this branch, each with a reason in the commit message. Run
  `python3 scripts/server/jobs/make_manifests.py --check` after any change to `blind_insertion/`.

```bash
git add results docs/FINDINGS.md scripts/server/jobs
git -c user.name="Yash Chitambar" -c user.email="yash_chitambar@berkeley.edu" commit -m "<phase>: <what finished>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin mason-runs
```

Yash merges `mason-runs` into `main`. Before each push, `git pull --rebase origin main` into the
branch if `main` moved, and resolve conflicts in `docs/FINDINGS.md` by keeping both sides' entries.

## 8. When to stop and ask Yash

- before holding more than 2 GPUs or any GPU for more than a day (section 1);
- if the bootstrap, verification or benchmark fails and the cause is not a clear local problem;
- if the validation run (section 4) does not match the local seeds;
- before you change world counts, PPO settings, ranges or any other method choice;
- before Phase D or E (noise level), and before you start Phase F work;
- if another user's job appears on a GPU you planned to use: wait, do not compete.

## 9. Status board (update this table when you start or finish something)

| Date | Host | What | GPUs | tmux session | Log | State |
|---|---|---|---|---|---|---|
| 2026-10-20 | athena5 | Local queue: C-peg-proprio-base-s0, then E1 s0, B blind s2, no-contact s1 (and the gear task when its agent adds it) | 0 | (queue.sh) | runs/queue/queue.log | running |
| 2026-10-20 | mason (actual migration host; 5 × V100 32 GB) | CUDA 12 bootstrap + all smoke tests | 0 | peg-setup (finished) | runs/bootstrap.log | passed, exit 0 |
| 2026-10-20 | mason | Blind benchmark, 12,288 worlds | 0 | peg-bench-blind (finished) | runs/benchmark-blind.log | passed: 20,490 steps/s, 10,706 MiB |
| 2026-10-20 | mason | Visual benchmark, 4,096 worlds | 1 (released) | peg-bench-visual (finished) | runs/benchmark-visual.log | passed: 3,922 steps/s, 28,558 MiB |
| 2026-10-20 | mason | 300M-step validation, B-peg-proprio-base-s3 | 0 (released) | peg-validation (finished) | runs/B-peg-proprio-base-s3.log | passed: 90% at 37.4M, >=99.2% from 150M, 99.8% at 300M |
| 2026-10-20 | mason | YAM lab-table scene (builder.YAM_TABLE, EnvCfg.table_top) + 25M smoke | 1 (released) | - | runs/_smoke-C-yam-table | passed: 90% at 18.1M |
| 2026-10-20 | mason | Batch: B_train, B_eval, C_train, C_eval (+ C base s0 on the table setup) | 0–4 (Yash OK) | jobs | runs/queue_server/runner.log | running |
| 2026-10-21 | mason | Redo of the 47 old-backlash-model runs (C backlash/allmid/allmax/randall, D tier 1+2, E noise) + C_eval_bl.txt | (approved set) | jobs | runs/queue_server/runner.log | to do |
| 2026-10-22 | mason | Backlash-model redo (47 runs, priority.txt order) + C_eval_bl | 0–4 (Yash OK) | runnerD2 | runs/queue_server/runnerD.log | running |
