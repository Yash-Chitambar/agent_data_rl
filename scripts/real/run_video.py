"""Three-camera video of one logged rig run, with the chain's stages drawn as a graph above the views.

    .venv/bin/python scripts/real/run_video.py data/real_runs/<date>/<run> <run>/demo_3cam.mp4 \
        $(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

Made for full-chain runs that reach the insertion (it reads align_press_on, 5 insert and seated_detected events).

Frames are the run's own saved raw frames (about 2 per second). Playback is real time; pauses at operator prompts are cut.
"""
import bisect
import json
import subprocess
import sys

import cv2
import numpy as np

run, out, ffmpeg = sys.argv[1:4]
FPS, GAP, H = 10, 2.5, 449
W_WRIST, W_TOP = 561, 797
W, BAR = W_WRIST * 2 + W_TOP + 3, 151
FONT = cv2.FONT_HERSHEY_SIMPLEX

ev = [json.loads(x) for x in open(f'{run}/events.jsonl')]
frames = [json.loads(x) for x in open(f'{run}/frames.jsonl')]
tk = np.load(f'{run}/ticks.npz', allow_pickle=True)
t_tick, grip = tk['t'].astype(float), tk['grip'][:, 0].astype(float)
pol = np.load(f'{run}/policy.npz', allow_pickle=True)
k_tick = tk['k'].astype(int)
t_pol = t_tick[np.clip(np.searchsorted(k_tick, pol['k'].astype(int)), 0, len(t_tick) - 1)]
depth = pol['seat_depth_mm'].astype(float)

cams = {}
for f in frames:
    if f['kind'] == 'raw' and f.get('t_capture'):
        cams.setdefault(f['cam'], []).append((f['t_capture'], f['file']))
for c in cams:
    cams[c].sort()
all_t = sorted(t for c in cams.values() for t, _ in c)

# stage timeline: the runner logs no stage row for the pick, so it starts at the operator's go for it
GRAPH = [('capture', 'capture'), ('pick', 'pick'), ('2 hold', 'left hold'), ('3 transport', 'transport'), ('4 approach', 'approach'),
         ('align', 'align'), ('5 insert', 'insert'), ('6 release', 'release'), ('7 home', 'home')]
marks = [(min(all_t) - 1, 'capture')]
for e in ev:
    if e['kind'] == 'answer' and e['what'].startswith('pick'):
        marks.append((e['t'], 'pick'))
    if e['kind'] == 'stage' and e['stage'] != '0 capture':
        marks.append((e['t'], e['stage']))
marks.sort()
mark_t = [m[0] for m in marks]
WHAT = {'capture': 'ZED locates the PC and the reader; arms at the capture pose', 'pick': 'Right arm servos down and grasps the reader',
        '2 hold': 'Left hand latches onto the PC', '3 transport': 'Right arm carries the reader toward the port',
        '4 approach': 'Wrist nets servo the plug to the stop', '6 release': 'Jaws open and retreat; the reader stays in the port',
        '7 home': 'Arms return home'}
measures = [e for e in ev if e['kind'] == 'align_measure']
m_t = [e['t'] for e in measures]
press_on = next(e['t'] for e in ev if e['kind'] == 'align_press_on')
seat_t = next(e['t'] for e in ev if e['kind'] == 'seated_detected')
ins_t = next(e['t'] for e in ev if e['kind'] == 'stage' and e['stage'] == '5 insert')


def detail(t, stage):
    if stage == 'align':
        i = bisect.bisect_right(m_t, t) - 1
        if i < 0:
            return 'Port vision measures the grasp, arm still'
        m = measures[i]
        kind = str(m.get('decision_reason') or m['decision'])
        word = {'probe': 'probing: is the camera following the hand?', 'correct': 'correcting in small steps', 'confirm': 'in tolerance, confirming', 'within tolerance': 'in tolerance'}.get(kind, kind)
        press = 'left press on' if t >= press_on else 'left press off'
        r = m['rv_deg']
        return f"Vision: across {m['across']:+.1f}  up {m['up']:+.1f}  gap {m['gap']:.1f} mm  |  pitch {r[0]:+.1f}  yaw {r[1]:+.1f} deg  |  {word}  |  {press}"
    if stage == '5 insert':
        if t >= seat_t:
            return f'Insertion policy: SEATED, {seat_t - ins_t:.2f} s after the hand-over'
        i = int(np.clip(np.searchsorted(t_pol, t), 0, len(depth) - 1))
        return f'Insertion policy pushes and searches  |  depth past the port face {depth[i]:+.1f} mm'
    return WHAT.get(stage, '')


def bar(t, stage, clock, note):
    im = np.full((BAR, W, 3), 28, np.uint8)
    n, x0, gap = len(GRAPH), 16, 22
    bw = (W - 2 * x0 - gap * (n - 1)) // n
    cur = [k for k, _ in GRAPH].index(stage)
    for i, (_, name) in enumerate(GRAPH):
        x = x0 + i * (bw + gap)
        col = (60, 170, 70) if i == cur else (92, 92, 92) if i < cur else (58, 58, 58)
        cv2.rectangle(im, (x, 14), (x + bw, 62), col, -1)
        if i == cur:
            cv2.rectangle(im, (x, 14), (x + bw, 62), (255, 255, 255), 2)
        (tw, _), _ = cv2.getTextSize(name, FONT, .68, 2)
        cv2.putText(im, name, (x + (bw - tw) // 2, 46), FONT, .68, (255, 255, 255) if i <= cur else (165, 165, 165), 2, cv2.LINE_AA)
        if i < n - 1:
            cv2.arrowedLine(im, (x + bw + 2, 38), (x + bw + gap - 2, 38), (170, 170, 170), 2, cv2.LINE_AA, tipLength=.45)
    cv2.putText(im, detail(t, stage), (x0, 100), FONT, .78, (255, 255, 255), 2, cv2.LINE_AA)
    i = min(int(np.searchsorted(t_tick, t)), len(grip) - 1)
    tail = f't {clock:5.1f} s  (real time, operator pauses cut)'
    if t >= t_tick[0]:
        tail += f'   |   right grip reading {grip[i]:.3f}'
    if note:
        tail += f'   |   {note}'
    cv2.putText(im, tail, (x0, 135), FONT, .58, (190, 190, 190), 1, cv2.LINE_AA)
    return im


def panel(name, path, w):
    if path is None:
        im = np.zeros((H, w, 3), np.uint8)
    else:
        im = cv2.resize(cv2.imread(f'{run}/{path}'), (w, H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (len(name) * 13 + 16, 32), (0, 0, 0), -1)
    cv2.putText(im, name, (8, 23), FONT, .68, (255, 255, 255), 2, cv2.LINE_AA)
    return im


proc = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H + BAR}', '-r', str(FPS), '-i', '-',
                         '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
LAYOUT = [('left wrist', 'wrist_left', W_WRIST), ('top', 'top', W_TOP), ('right wrist', 'wrist_right', W_WRIST)]
last, cache = {}, {}
t, clock, n_out, note_until, note, skipped = all_t[0], 0., 0, -1., '', 0.
sep = np.full((H, 1, 3), 80, np.uint8)
while t <= all_t[-1]:
    for name, cam, w in LAYOUT:
        i = bisect.bisect_right([x[0] for x in cams[cam]], t) - 1
        path = cams[cam][i][1] if i >= 0 else None
        if last.get(cam, 0) != path:
            last[cam], cache[cam] = path, panel(name, path, w)
    stage = marks[bisect.bisect_right(mark_t, t) - 1][1]
    views = np.hstack([cache['wrist_left'], sep, cache['top'], sep, cache['wrist_right']])
    pad = np.zeros((H, W - views.shape[1], 3), np.uint8)
    proc.stdin.write(np.vstack([bar(t, stage, clock, note if clock < note_until else ''), np.hstack([views, pad])]).tobytes())
    n_out += 1
    nxt = t + 1 / FPS
    j = bisect.bisect_left(all_t, nxt)
    if j < len(all_t) and all_t[j] - nxt > GAP:      # nothing was recorded here: waiting at a prompt
        note, note_until = f'operator prompt, {all_t[j] - nxt:.0f} s cut', clock + 2.5
        skipped += all_t[j] - nxt
        nxt = all_t[j]
    t = nxt
    clock += 1 / FPS
for _ in range(FPS * 2):
    proc.stdin.write(np.vstack([bar(all_t[-1], stage, clock, ''), np.hstack([views, pad])]).tobytes())
    n_out += 1
proc.stdin.close()
proc.wait()
print(dict(frames=n_out, seconds=round(n_out / FPS, 1), skipped_s=round(skipped, 1), size=(W, H + BAR), first_stage=marks[0], returncode=proc.returncode,
           frames_per_cam={c: len(v) for c, v in cams.items()}, t_tick0_vs_first=round(float(t_tick[0] - all_t[0]), 1), seat_clock_note=round(seat_t - ins_t, 2)))
