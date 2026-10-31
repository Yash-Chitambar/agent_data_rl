# Documentation map

Start with the [README](../README.md) (question and motivation) and
[CURRENT](campaign/CURRENT.md) (live status). Docs on `main` are the source of truth.

| Folder | What | Key docs |
|---|---|---|
| [`campaign/`](campaign/) | What is happening now and the full history | [CURRENT](campaign/CURRENT.md) (plan, done, open, gates), [FINDINGS](campaign/FINDINGS.md) (dated lab notebook, newest first), [PHYSICAL_TODO](campaign/PHYSICAL_TODO_20261030.md) (rig checklist) |
| [`research/`](research/) | Why: question, hypothesis, task plan, prior evidence | [RESEARCH_DIRECTION](research/RESEARCH_DIRECTION.md), [key-box slides](research/keybox_sim2real.pptx), [KEYBOX_LESSONS_FOR_USBC](research/KEYBOX_LESSONS_FOR_USBC.md), [USB_C_SIM2REAL_PLAN](research/USB_C_SIM2REAL_PLAN.md), [paper PDF](research/bronars_blind_precise_assembly_2610.23849.pdf) |
| [`real2sim/`](real2sim/) | How the sim matches the rig, with numbers | [README](real2sim/README.md), scene, gaps, data contract, physics audit, measurement plans |
| [`rl/`](rl/) | Runtime, task layout, policy interface, training status | [README](rl/README.md), [BIMANUAL_POLICY_INTERFACE](rl/BIMANUAL_POLICY_INTERFACE.md), [BIMANUAL_TRAINING_HANDOFF](rl/BIMANUAL_TRAINING_HANDOFF.md) |
| [`operations/`](operations/) | Setup, branches, code map, server | [SETUP](operations/SETUP.md), [BRANCHES](operations/BRANCHES.md), [CODE_MAP](operations/CODE_MAP.md), [SERVER](operations/SERVER.md) |
| [`archive/`](archive/) | Superseded plans and handoffs, kept for provenance | [README](archive/README.md) |

Writing rules: dated entries go to FINDINGS (newest first); CURRENT stays short and is
updated whenever a gate or the plan changes; results live in `results/` with the script that
made them; text follows ASD-STE100 (short active sentences).
