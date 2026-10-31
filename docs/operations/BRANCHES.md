# Branches

| Branch | Purpose | Runs on |
|---|---|---|
| `main` | The bimanual simulator and policy training: the current, working state. Docs here are the source of truth. | mason (training), any dev machine |
| `real/duchamp3` | Real-rig work: deploy/test runners, safety layer, calibration and measurement tools, rig-session docs. Based on `main`. | duchamp3 (YAM PC) |
| `sim/fidelity` | Making the simulator more accurate: system identification, camera/fingertip alignment, backdrop and contact fits, replays against real data. | mason / dev |
| `codex/bimanual-calibrated` | Live branch of the codex agent working on the bimanual pick/insert audit; merges into `main`. | mason |
| `exp/<topic>` | Exploratory work that may never merge (e.g. `exp/latency-ms`). | anywhere |
| `data/<name>` | Large raw captures kept off the code branches (e.g. `data/gripper-sweep-20261030`). Fetch with `git checkout origin/data/<name> -- <path>`. | — |

## Rules

- Work on `sim/fidelity`, `real/duchamp3` or `exp/*`; merge into `main` only validated work:
  the checks pass and the numbers are in the docs (`docs/campaign/CURRENT.md`, the relevant
  `docs/*/README.md`, a dated entry in `docs/campaign/FINDINGS.md`).
- Shared code (scenes, envs, configs) is fixed on `main` or `sim/fidelity` first, then merged
  into `real/duchamp3` (`git merge main`), never edited only on the rig branch.
- Rig-only code (`deploy/`, rig runners, CAN/camera tooling) lives on `real/duchamp3`. Small
  data (calibration JSON, measurement results) is committed there and merged into `main`
  when the sim uses it; large raw data goes to a `data/*` branch.
- Name branches by purpose, not by agent. Agents may use `<agent>/<topic>` for a short-lived
  branch and merge it into one of the branches above.

## Retired branches (archived as tags)

Each retired branch has a tag `archive/<name>` at its last commit; restore with
`git branch <name> archive/<name>`.

| Old branch | Now in |
|---|---|
| `main` @ dc535aa (single-arm phases A–H) | `main` history (tag `archive/main-2026-10-25`) |
| `mason-runs`, `real-yam-setup`, `codex/physical-yam-scene` | `main` history |
| `codex/bimanual-contact-20261028`, `codex/physical-validate-duchamp3`, `claude/physical-validate` | `real/duchamp3` |
| `codex/usbc-lab-handoff-20261027` | `real/duchamp3` |
