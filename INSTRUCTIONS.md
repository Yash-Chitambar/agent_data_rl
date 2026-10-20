# Instructions for the agent on the training server

You are taking over the remaining RL training of this project on a shared GPU server (first
target: **jensen**, 8 × V100, `jensen.ist.berkeley.edu`, user `yashchitambar`). Another agent and
Yash work on the local machine (athena5, 1 × RTX 3090 Ti). Read this file completely before you do
anything. When this file and another document disagree about the server, this file wins.

## 0. Context in five lines

- The project reproduces Bronars et al., *Blind Precise Assembly* (arXiv:2610.23849, PDF in `docs/`):
  a robot inserts parts using only joint data, trained with PPO in MuJoCo Warp.
- What to do and in which order: `docs/RESEARCH_PLAN.md`. Results and discoveries so far:
  `docs/FINDINGS.md`. Repo conventions and gotchas: `CLAUDE.md`. Server mechanics in depth:
  `docs/SERVER.md`.
- Phase A (checks) and most of Phase B (paper reproduction on the Flexiv arm) are done and match the
  paper. The server work is the rest of B, and Phases C (YAM arm), D (scale grid) and E (USB-C).
- Every job is one command on one GPU. The job lists are in `scripts/server/jobs/*.txt` (381 jobs,
  about 457 GPU-hours measured on the 3090 Ti).
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

```bash
tmux new -d -s jobs "python3 scripts/server/run_jobs.py --gpus <free indices> \
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
- **Phase F is blocked** (`F_blocked.txt`): it needs code changes (`docs/SERVER.md` §9.4).
- **There is no resume.** A crashed or interrupted run restarts from step 0 (the runner retries once).
  If interruptions become common, implement `--resume` from `docs/SERVER.md` §9.1 on a branch first.
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
git switch -c jensen-runs        # once
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
git push -u origin jensen-runs
```

Yash merges `jensen-runs` into `main`. Before each push, `git pull --rebase origin main` into the
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
| | jensen | Setup + benchmark | | | | not started |
