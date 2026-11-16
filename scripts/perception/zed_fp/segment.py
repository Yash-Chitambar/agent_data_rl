"""First-frame object masks for FoundationPose with SAM 3 (facebook/sam3, image model, facebookresearch/sam3 code).

Runs in the separate FoundationPose env (`fp`), never in the rig venv. Prompts:
  * 'text': a text prompt ("mini PC", "SD card reader"); among the detections, the one whose box best matches the
    prior box (when given) or the highest score.
  * 'box':  a positive box prompt from the prior pose (net / edge fit projected), no text.
  * 'text+box': text prompt, then the prior box as an extra positive geometric prompt.
The image is cropped to `roi` (the table area around the objects) before SAM 3 resizes it to 1008 px, so a 100 mm PC
stays about 180 px wide instead of 95 px.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

DEFAULT_CKPT = Path(os.environ.get('SAM3_CKPT', '/home/yashchitambar/fp_work/sam3_ckpt/sam3.pt'))
# Prompt lists, tried in order on one image encoding. On real ZED frames 'mini PC' and 'SD card reader' give no detection;
# 'small silver box' finds the PC (score 0.88 on zed_gt spot1). The sim render (flat white lid) needs 'white box'.
TEXT = {'pc': ['small silver box', 'silver box', 'white box', 'computer', 'small white object', 'box'],
        'reader': ['small white object', 'white plastic part', 'small white box']}
MIN_PRIOR_IOU = 0.3


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Sam3Segmenter:
    def __init__(self, ckpt=DEFAULT_CKPT, device='cuda', threshold=0.3):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        t0 = time.perf_counter()
        self.torch = torch
        self.model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False, device=device)
        self.proc = Sam3Processor(self.model, device=device, confidence_threshold=threshold)
        self.load_s = time.perf_counter() - t0

    def segment(self, rgb, obj='pc', mode='text', roi=None, prior_box=None, text=None, stop_score=0.6):
        """rgb: HxWx3 uint8 RGB (full frame). roi, prior_box: (u0, v0, u1, v1) full-frame pixels.
        Returns (mask uint8 HxW 0/1 or None, info dict)."""
        from PIL import Image
        t0 = time.perf_counter()
        H, W = rgb.shape[:2]
        u0, v0, u1, v1 = (0, 0, W, H) if roi is None else [int(v) for v in roi]
        crop = rgb[v0:v1, u0:u1]
        ch, cw = crop.shape[:2]
        pb = None if prior_box is None else np.array(prior_box, float) - [u0, v0, u0, v0]
        prompts = [text] if isinstance(text, str) else list(text or TEXT[obj])
        cands = []            # (masks, boxes, scores, prompt)
        with self.torch.autocast('cuda', dtype=self.torch.float16):      # V100: fp16 (the model has bf16 casts inside)
            st = self.proc.set_image(Image.fromarray(crop))
            if mode == 'box':
                if pb is None:
                    raise ValueError('box prompt needs prior_box')
                bx = [(pb[0] + pb[2]) / 2 / cw, (pb[1] + pb[3]) / 2 / ch, (pb[2] - pb[0]) / cw, (pb[3] - pb[1]) / ch]
                out = self.proc.add_geometric_prompt(box=bx, label=True, state=st)
                cands.append(({k: out.get(k) for k in ('masks', 'boxes', 'scores')}, 'box'))
            else:
                for pr in prompts:
                    out = self.proc.set_text_prompt(state=st, prompt=pr)
                    out = {k: (None if out.get(k) is None else out[k].clone()) for k in ('masks', 'boxes', 'scores')}
                    cands.append((out, pr))              # copies: the processor reuses its state dict
                    sc = out['scores']
                    if sc is not None and len(sc) and pb is not None:
                        bb = out['boxes'].float().cpu().numpy().reshape(-1, 4)
                        ok = [float(x) for x, b in zip(sc.float().cpu().numpy().reshape(-1), bb) if box_iou(b, pb) >= MIN_PRIOR_IOU]
                        if ok and max(ok) >= stop_score:
                            break
                    elif sc is not None and len(sc) and float(sc.max()) >= stop_score:
                        break
        best = None           # (key, mask, box, score, prompt, prior_iou)
        n_all = 0
        for out, pr in cands:
            if out['masks'] is None or len(out['masks']) == 0:
                continue
            n = len(out['masks']); n_all += n
            mks = out['masks'].float().cpu().numpy().reshape(n, ch, cw) > 0.5
            bxs = out['boxes'].float().cpu().numpy().reshape(n, 4)
            scs = out['scores'].float().cpu().numpy().reshape(n)
            for mk, b, sc in zip(mks, bxs, scs):
                piou = box_iou(b, pb) if pb is not None else None
                if piou is not None and piou < MIN_PRIOR_IOU:
                    continue
                key = float(sc) if piou is None else float(sc) * piou
                if best is None or key > best[0]:
                    best = (key, mk, b, float(sc), pr, piou)
        info = dict(mode=mode, prompts_tried=[pr for _, pr in cands], n_det=int(n_all), roi=[u0, v0, u1, v1])
        if best is None:
            info['time_s'] = time.perf_counter() - t0
            return None, info
        _, mk, b, sc, pr, piou = best
        masks, boxes, scores, k = mk[None], b[None], np.array([sc]), 0
        info['prompt'] = pr
        if piou is not None:
            info['prior_box_iou'] = float(piou)
        full = np.zeros((H, W), np.uint8)
        full[v0:v1, u0:u1] = masks[k]
        info.update(score=float(scores[k]),
                    box=[float(boxes[k][0] + u0), float(boxes[k][1] + v0), float(boxes[k][2] + u0), float(boxes[k][3] + v0)],
                    area_px=int(masks[k].sum()), time_s=time.perf_counter() - t0)
        return full, info
