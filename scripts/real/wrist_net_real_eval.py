"""Wrist net on real frames, on a fixed hold-out. No motion, no training.

    PYTHONPATH=. python scripts/real/wrist_net_real_eval.py --net runs/perception/wrist_v18ws.pt --out results/real/wrist_nets/v18ws.json

Data: data/real_training/port_v1/manifest.jsonl (scripts/real/build_port_dataset.py). The split is fixed here, not by a seed:

  collect sweep (150 frames in visiting order): cut into 10 contiguous blocks of 15. Each of the sweep's three parts (stop,
  fine, far in collect.sample_plan, visited as nearest-neighbour chains) gives held-out blocks; the frame on each side of a
  held-out block is a guard and is used by neither side. Random-frame splits are not allowed: nearby frames leak.
  full_A, full_A2 (`same_spot`), ai4 (`self`): never in training. The ai1, ai2, ai3, ai5 runs (`unlabeled`) may train on
  differences within a run only.

Reports, for any net:
  - slope (least squares, net output per mm of label) of across, up, gap on the held-out collect frames, by gap band, with a
    bootstrap interval over frames, and over the held-out blocks;
  - A and A2 height and sideways difference at their stops, against the FK difference and the vision tip difference (no independent
    truth separates them: vision sees a different plug position in the frame between the runs);
  - residual SD after removing a constant per axis.
A slope is not sensitive to a constant label offset. Absolute y is not used: docs/real/SESSION_20261102_duchamp3.md, "Label quality".
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MANIFEST = 'data/real_training/port_v1/manifest.jsonl'
BLOCK = 15
HOLDOUT_BLOCKS = (1, 4, 7, 9)      # of the 10 blocks of the collect sweep; see block_table() and docs/real/WRIST_NETS_REAL.md
GUARD = 1
GAP_BANDS = ((0., 10.), (10., 25.), (25., 70.), (0., 70.))
AXES = ('across', 'up', 'gap')
A, A2 = 'full_174850_full_A', 'full_175742_full_A2'
# real/port-vision d1eb4d0, results/real/port_vision/label_offset.json, docs/real/PORT_VISION.md (an estimate: about 0.6 mm 1 sigma, a guess)
COLLECT_OFFSET_MM = (-.54, -2.29, -1.67)          # true = manifest + offset, collect run; y range -3.21 to -1.58
VISION_STOP = {A: (-.3, 5., 8.4), A2: (-5.2, 8.9, 8.)}      # vision plug tip at the stops, across, up, gap
VISION_AI4_STOP = (4.3, -2.4, 9.9)
STOP_MM = .6                       # stop frames of A, A2: within this of the run's last frame, per axis


def rows():
    return [json.loads(x) for x in open(ROOT / MANIFEST)]


def collect_rows(rs=None):
    return [r for r in (rs or rows()) if r['source'] == 'collect']


def split_collect(n=150):
    """(train idx, hold-out idx, block id of each held-out frame) in sweep order. Same result every call."""
    block = np.arange(n) // BLOCK
    hold = np.isin(block, HOLDOUT_BLOCKS)
    guard = np.zeros(n, bool)
    for i in range(n):
        if not hold[i] and ((i > 0 and hold[i - 1]) or (i < n - 1 and hold[i + 1])):
            guard[i] = True
    train = ~hold & ~(guard if GUARD else np.zeros(n, bool))
    return np.where(train)[0], np.where(hold)[0], block[hold]


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(ROOT / path)), cv2.COLOR_BGR2RGB)


def load_net(path, dev):
    from blind_insertion.perception.nets import PoseNet
    ck = torch.load(ROOT / path if not str(path).startswith('/') else path, map_location=dev, weights_only=False)
    net = PoseNet(ck.get('n_out', len(ck['scale'])), ck['size'], width=ck.get('width', 32)).to(dev)
    net.load_state_dict(ck['net'])
    net.eval()
    return net, ck['scale'].to(dev), ck


def net_input(im, ck=None):
    """(B, 320, 320, 3) uint8 -> the net's input, as chain.py and eval_chain do."""
    mode = (ck or {}).get('wrist_mode', 'resize')
    if mode == 'full':
        return im
    if mode == 'crop':
        x0, y0 = (ck or {}).get('wrist_crop', (64, 40))
        s = (ck or {}).get('wrist_size', 192)
        return im[:, y0:y0 + s, x0:x0 + s].contiguous()
    return F.interpolate(im.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)


@torch.no_grad()
def predict(net, scale, ck, imgs, dev, bs=64):
    """imgs: (N, 320, 320, 3) uint8 numpy RGB -> (N, 6) mm / deg."""
    out = []
    k = len(scale)
    for i in range(0, len(imgs), bs):
        im = net_input(torch.as_tensor(imgs[i:i + bs], device=dev), ck)
        out.append((net(im)[:, :k] * scale).cpu().numpy())
    return np.concatenate(out)


def slope(y, p):
    return float(np.polyfit(y, p, 1)[0]) if len(y) > 2 and np.ptp(y) > 1e-6 else float('nan')


def boot(y, p, groups=None, n=4000, seed=0):
    """95 % interval of the slope: resampling frames, or whole groups if given."""
    g = np.random.default_rng(seed)
    v = []
    if groups is None:
        for _ in range(n):
            i = g.integers(0, len(y), len(y))
            v.append(slope(y[i], p[i]))
    else:
        u = np.unique(groups)
        idx = {b: np.where(groups == b)[0] for b in u}
        for _ in range(n):
            i = np.concatenate([idx[b] for b in g.choice(u, len(u))])
            v.append(slope(y[i], p[i]))
    return [round(float(x), 3) for x in np.nanpercentile(v, [2.5, 97.5])]


def band_report(Y, P, groups=None):
    out = {}
    for lo, hi in GAP_BANDS:
        m = (Y[:, 2] >= lo) & (Y[:, 2] < hi)
        if m.sum() < 4:
            continue
        e = P[m] - Y[m]
        r = dict(n=int(m.sum()), gap_range=[round(float(Y[m, 2].min()), 1), round(float(Y[m, 2].max()), 1)])
        for i, ax in enumerate(AXES):
            s = slope(Y[m, i], P[m, i])
            r[ax] = dict(slope=round(s, 3), ci_frames=boot(Y[m, i], P[m, i]), label_sd=round(float(Y[m, i].std()), 2), resid_sd=round(float(e[:, i].std()), 2),
                         mean_err=round(float(e[:, i].mean()), 2))
            if groups is not None and len(np.unique(groups[m])) > 1:
                r[ax]['ci_blocks'] = boot(Y[m, i], P[m, i], groups[m])
        out[f'gap {lo:.0f}-{hi:.0f}'] = r
    return out


def stops(rs):
    """Frames at the stop of A, A2: from the end of the approach back, while the label stays within STOP_MM of the last frame."""
    out = {}
    for run in (A, A2):
        fr = [r for r in rs if r['run'] == run and 'p_mm' in r]
        last, sel = np.array(fr[-1]['p_mm']), []
        for r in reversed(fr):
            if np.abs(np.array(r['p_mm']) - last).max() > STOP_MM:
                break
            sel.append(r)
        out[run] = sel[::-1]
    return out


def ab_report(rs, predict_rows):
    """Height and sideways difference of A2 against A at their stops: net against the FK label."""
    st = stops(rs)
    Pa, Pb = predict_rows(st[A]), predict_rows(st[A2])
    Ya, Yb = (np.array([r['p_mm'] for r in st[k]]) for k in (A, A2))
    d_net, d_fk = Pb.mean(0)[:3] - Pa.mean(0)[:3], Yb.mean(0) - Ya.mean(0)
    va, vb = np.array(VISION_STOP[A]), np.array(VISION_STOP[A2])
    g = np.random.default_rng(0)
    bd = [Pb[g.integers(0, len(Pb), len(Pb))].mean(0)[:3] - Pa[g.integers(0, len(Pa), len(Pa))].mean(0)[:3] for _ in range(2000)]
    return dict(n=[len(st[A]), len(st[A2])], label_diff_A2_minus_A=dict(zip(AXES, d_fk.round(2).tolist())), net_diff_A2_minus_A=dict(zip(AXES, d_net.round(2).tolist())),
                net_diff_ci_frames=np.percentile(bd, [2.5, 97.5], axis=0).T.round(2).tolist(),
                vision_tip_diff_A2_minus_A=dict(zip(AXES, (vb - va).round(2).tolist())), vision_at_A=va.tolist(), vision_at_A2=vb.tolist(),
                net_minus_vision_at_A=(Pa.mean(0)[:3] - va).round(2).tolist(), net_minus_vision_at_A2=(Pb.mean(0)[:3] - vb).round(2).tolist(),
                net_at_A=Pa.mean(0)[:3].round(2).tolist(), net_at_A2=Pb.mean(0)[:3].round(2).tolist(), label_at_A=Ya.mean(0).round(2).tolist(), label_at_A2=Yb.mean(0).round(2).tolist(),
                net_sd_at_A=Pa.std(0)[:3].round(2).tolist(), net_sd_at_A2=Pb.std(0)[:3].round(2).tolist())


def evaluate(net_path, dev='cuda:0', rs=None, tf=None):
    """tf: optional function on (N, 320, 320, 3) uint8 numpy (sweeps). Returns the report and raw arrays."""
    rs = rs or rows()
    net, scale, ck = load_net(net_path, dev)

    def pr(sel):
        im = np.stack([read_rgb(r['image']) for r in sel])
        return predict(net, scale, ck, tf(im) if tf else im, dev)
    co = collect_rows(rs)
    Y = np.array([r['p_mm'] for r in co])
    P = pr(co)
    tr, ho, gb = split_collect(len(co))
    rep = dict(net=str(net_path), split=dict(train=len(tr), holdout=len(ho), blocks=list(HOLDOUT_BLOCKS)),
               holdout_collect=band_report(Y[ho], P[ho], gb),
               holdout_mean_err_vs_offset_labels=(P[ho, :3] - (Y[ho] + np.array(COLLECT_OFFSET_MM))).mean(0).round(2).tolist(),
               train_collect=band_report(Y[tr], P[tr]), all_collect=band_report(Y, P), ab=ab_report(rs, pr))
    other = {}
    for src, nm in (('self', 'ai4'), ('same_spot', 'A+A2')):
        s = [r for r in rs if r['source'] == src and 'p_mm' in r]
        Ys, Ps = np.array([r['p_mm'] for r in s]), pr(s)
        other[nm] = dict(n=len(s), **{ax: dict(slope=round(slope(Ys[:, i], Ps[:, i]), 3), ci_frames=boot(Ys[:, i], Ps[:, i]), label_sd=round(float(Ys[:, i].std()), 2),
                                               resid_sd=round(float((Ps[:, i] - Ys[:, i]).std()), 2)) for i, ax in enumerate(AXES)})
    ai = [r for r in rs if r['source'] == 'self' and 'p_mm' in r]
    Pi = pr(ai[-5:])
    other['ai4']['net_at_last_5_frames'] = Pi.mean(0)[:3].round(2).tolist()
    other['ai4']['vision_at_stop'] = list(VISION_AI4_STOP)
    rep['other_runs'] = other
    return rep, dict(Y=Y, P=P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', required=True)
    ap.add_argument('--out')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    rep, _ = evaluate(a.net, a.device)
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
