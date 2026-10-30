# Joint backlash, both YAM arms (duchamp3, 2026-10-30)

Measured with `scripts/real/measure_backlash.py` (encoder on the motor side vs. wrist-camera
rotation about a taped checkerboard on the link side). One board, centred between the arms,
0.30 m in front of the base line (`--hint 0.30 0.30` right, `--hint 0.30 -0.30` left). Camera
noise is 0.45-0.8 arcmin per stair, so values below about 1 arcmin are **not resolved**:
read them as "under ~1 arcmin", not as a number.

| Folder | What |
|---|---|
| `right_20261030/` | right arm, all 6 joints, closed-loop stairs, 26 stairs each (joint 5: 23) |
| `left_20261030/` | left arm, joints 1, 2, 4, 6 complete. Joints 3 and 5 are cut short (6 and 3 stairs): do not use them |
| `left_20261030_j35/` | left arm joints 3 and 5 rerun with 4 cycles (50 stairs each) |
| `right_20261030_openloop/` | first right-arm attempt, open loop (commanded offset only). Keep as stiction evidence, not for backlash |

## Backlash (arcmin; `analysis.json` per folder)

| Joint | Right | Left | Notes |
|---|---|---|---|
| 1 | 1.11 | 0.51 | |
| 2 | 2.32 | 0.57 | fit residual 3.3 / 2.2 arcmin: stick-slip scatter, weakest numbers |
| 3 | 0.14 | 0.09 | |
| 4 | 0.83 | 0.04 | |
| 5 | 0.17 | 0.56 | |
| 6 | 0.15 | 0.69 | |

Everything is at most ~1-2 arcmin, and most joints are below the camera noise: this is the low
end of (or under) the 1-5 arcmin range assumed for the sim. Sensible sim range: 0 to ~2 arcmin.
`link_per_motor_slope` is 0.92-1.01 (right arm 0.92-0.98, left 0.97-1.01): the camera sees up to
8% less rotation than the encoder, mostly on the right arm. Likely hand-eye / joint-axis
calibration, not transmission. Not investigated.

## Static friction (matters more than backlash)

With the commanded stair alone (open loop, 3 mrad), the right arm's joint 3 did not move at all
(0 of 18 mrad encoder travel), joint 2 moved 9.5, joint 6 5.1, joint 5 7.7, joint 4 14.8, joint 1
13.3. The PD hold cannot break static friction with 3 mrad of error. With a closed-loop push
(below) the joints move, but in stick-slip jumps of about 6-12 mrad on joints 2, 3 and 5 (largest push
needed: 41 mrad of extra command on the left joint 3, 31 on the right joint 3). Consistent with the
3-5 N m breakaway friction noted for the elbow.

## Method change (script, same commit)

`servo_stair` drives each stair to a target **encoder** value: open-loop stair plus a slowly
ramping push (cap 80 mrad, ramp 0.03 rad/s), which unwinds when the encoder moves toward the
target over a 0.2 s window (one count is 0.38 mrad, so per-cycle differences are noise). Every
command still passes `SafeArm.command`; the run aborts if an encoder strays more than 15 mrad
beyond the +/-9 mrad range. Stairs are commanded from the accepted viewing-pose target, not the
measured pose: the wrist sags under its command, which put `tip_left` on the 90 mm planning
limit and stopped the open-loop run on its first stair. A viewing pose rejected before moving now
skips to the next candidate. `--open-loop` restores the old behaviour.
