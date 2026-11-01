# agent_data_rl: contact-aware real-robot policies from an agent-built simulator

**Question:** How do we use the agent **GaP harness + RL in simulation** to train a
**real** bimanual policy that is aware of contact, on robots without force sensing?

The GaP self-learning pipeline already solves many manipulation tasks *in simulation*
(latched toolbox, tool and wrench hanging, peg insertion 500/500; coffee machine 493/500;
YAM scissor-hook 200/200). The key-box study ([slides](docs/research/keybox_sim2real.pptx))
showed what it takes to turn a sim-only success into a real one: a policy that scored
496/500 in sim reached the key turn in **0** real runs until three gaps were closed.

| Gap | What fixed it on the key box |
|---|---|
| **Motion** | reachability-driven layout; gravity sag 18.7 → 0.7 mm; elbow static friction fed forward; sim joints land like the real ones |
| **Perception** | sim cameras at least as bad as the real ones (depth holes on shiny/dark surfaces, calibration noise ≥ real); robot-fit camera calibration; per-episode appearance randomization |
| **Physics** | contact priors from real runs, demos and plain-language facts: key stands, grasp holds, pin friction, box stays put, recessed tab with a stiff point |

The recipe: **physics priors first → a privileged agent solves the task → a vision agent
solves it with sim cameras at least as bad as real → zero-shot to the real robot.**

This repo applies that recipe to a contact-rich bimanual task on two YAM arms (rig PC
duchamp3): the **right arm picks a SanDisk USB-C SD reader and inserts it into port 2 of a
GMKtec mini PC while the left arm holds the PC** (sub-0.1 mm clearance, no force sensor, PC
free on the table). It grew out of a reproduction of Bronars et al., *Blind Precise
Assembly* (arXiv:2610.23849): proprioceptive PPO policies that sense contact through joint
signals. That single-arm work is complete and is the policy/training foundation here.

## Start here

| Want to… | Read |
|---|---|
| Know what is done, what is next and what is open | [Current campaign](docs/campaign/CURRENT.md) |
| Understand the research question, hypothesis and gates | [Research direction](docs/research/RESEARCH_DIRECTION.md) |
| See the policy plan (pick → approach → hold → insert, DR, accuracy targets) | [Policy plan](docs/rl/POLICY_PLAN.md) |
| Browse all documentation by theme | [Documentation map](docs/README.md) |
| Find code and the data flow | [Code map](docs/operations/CODE_MAP.md) |
| Know which branch to use (sim, real rig, experiments) | [Branches](docs/operations/BRANCHES.md) |
| See how the sim was matched to the real rig, with numbers | [Real-to-sim](docs/real2sim/README.md) |
| Use the bimanual training runtime and policy interface | [RL](docs/rl/README.md) |
| Read every dated result and deviation | [Lab notebook](docs/campaign/FINDINGS.md) |

## Current state (2026-10-31)

The bimanual simulator (`blind_insertion/envs/bimanual_runtime.py`, MuJoCo Warp, ~28k
env-steps/s at 2048 worlds on a V100) is matched to the real rig where we could measure it:
identified per-arm gravity and friction (held-out tool error 8.5 → 4.4 mm), fingertip and
wrist-camera alignment held to calipers, a background fitted to 21 real frames, backlash
measured (≤ 2 arcmin), and a task layout from a reachability search with per-episode PC and
reader randomization. **No bimanual policy has been trained yet, and no real insertion has
been attempted with this simulator.** Next: action latency and encoder-offset noise in the
bimanual runtime, contact randomization, then a privileged PPO teacher. Details and gates:
[CURRENT](docs/campaign/CURRENT.md).

## Quick start

```bash
git clone --recurse-submodules git@github.com:Yash-Chitambar/agent_data_rl.git && cd agent_data_rl
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv sync --locked --group train
export PYTHONPATH=$PWD MUJOCO_GL=egl
.venv/bin/python scripts/checks/check_bimanual_runtime.py      # bimanual GPU runtime + randomization checks
.venv/bin/python scripts/real2sim/render_layout_views.py       # real vs sim views with the PC placement space
```

Setup details (submodules, optional path tracer, real-arm dependencies):
[Setup](docs/operations/SETUP.md). Training on the shared server: [Server](docs/operations/SERVER.md).
