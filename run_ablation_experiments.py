# -*- coding: utf-8 -*-
import os
import json
import time
import glob
import cv2
import numpy as np
import torch

def load_anti_uav_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exist_flags = data.get('exist', [1] * len(data.get('gt_rect', [])))
    gt_rects = data.get('gt_rect', [])
    return exist_flags, gt_rects

def load_anti_uav_annotation(seq_dir, modality='infrared'):
    json_candidates = [
        os.path.join(seq_dir, f'{modality}.json'),
        os.path.join(seq_dir, 'IR_label.json' if modality == 'infrared' else 'visible.json'),
        os.path.join(seq_dir, 'groundtruth.json')
    ]
    for jp in json_candidates:
        if os.path.exists(jp):
            return load_anti_uav_json(jp)

    txt_candidates = glob.glob(os.path.join(seq_dir, '*.txt'))
    if txt_candidates:
        gt_rect = np.loadtxt(txt_candidates[0], delimiter=',').tolist()
        exist = [1 if (len(r) == 4 and r[2] > 0 and r[3] > 0) else 0 for r in gt_rect]
        return exist, gt_rect

    raise FileNotFoundError(f"Annotation file not found in {seq_dir} for {modality}")

def compute_iou(boxA, boxB):
    if len(boxA) < 4 or len(boxB) < 4 or boxA[2] <= 0 or boxB[2] <= 0:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_center_error(boxA, boxB):
    if len(boxA) < 4 or len(boxB) < 4 or boxA[2] <= 0 or boxB[2] <= 0:
        return 1000.0
    cxA, cyA = boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0
    cxB, cyB = boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0
    return float(np.hypot(cxA - cxB, cyA - cyB))

def evaluate_sequence_metrics(pred_bboxes, gt_rects, exist_flags, return_cle=False):
    T = min(len(pred_bboxes), len(gt_rects), len(exist_flags))
    sa_list, ious, cle_list, valid_cle_list = [], [], [], []

    for t in range(T):
        v = exist_flags[t]
        gt = gt_rects[t]
        pred = pred_bboxes[t]
        p_hat = 1 if (len(pred) == 4 and pred[2] > 0 and pred[3] > 0) else 0

        if v > 0 and len(gt) == 4 and gt[2] > 0:
            iou = compute_iou(pred, gt) if p_hat > 0 else 0.0
            cle = compute_center_error(pred, gt) if p_hat > 0 else 1000.0
            sa_list.append(iou)
            ious.append(iou)
            cle_list.append(cle)
            if p_hat > 0:
                valid_cle_list.append(min(cle, 500.0))
        else:
            sa_list.append(1.0 if p_hat == 0 else 0.0)

    m_sa = float(np.mean(sa_list)) * 100.0 if sa_list else 0.0
    precision_20 = float(np.mean([1.0 if c <= 20.0 else 0.0 for c in cle_list])) * 100.0 if cle_list else 0.0
    success_50 = float(np.mean([1.0 if i >= 0.5 else 0.0 for i in ious])) * 100.0 if ious else 0.0
    mean_cle = float(np.mean(valid_cle_list)) if valid_cle_list else 500.0

    if return_cle:
        return m_sa, precision_20, success_50, mean_cle
    return m_sa, precision_20, success_50

def run_video_pair_mp4(tracker, rgb_mp4_path, ir_mp4_path, gt_rects_ir, exist_flags_ir,
                       gt_rects_rgb=None, exist_flags_rgb=None, **kwargs):
    cap_rgb = cv2.VideoCapture(rgb_mp4_path)
    cap_ir = cv2.VideoCapture(ir_mp4_path)

    pred_bboxes_ir, frame_times = [], []
    frame_idx = 0
    initialized = False
    use_cuda = torch.cuda.is_available()

    while True:
        ret_rgb, frame_rgb_bgr = cap_rgb.read()
        ret_ir, frame_ir_bgr = cap_ir.read()
        if not ret_rgb or not ret_ir or frame_idx >= len(gt_rects_ir):
            break

        frame_rgb = cv2.cvtColor(frame_rgb_bgr, cv2.COLOR_BGR2RGB)
        frame_ir = cv2.cvtColor(frame_ir_bgr, cv2.COLOR_BGR2RGB)

        if use_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        if not initialized:
            if exist_flags_ir[frame_idx] > 0 and len(gt_rects_ir[frame_idx]) == 4:
                init_box_ir = gt_rects_ir[frame_idx]
                init_box_rgb = None
                if gt_rects_rgb is not None and frame_idx < len(gt_rects_rgb):
                    if exist_flags_rgb is None or exist_flags_rgb[frame_idx] > 0:
                        init_box_rgb = gt_rects_rgb[frame_idx]

                tracker.init(frame_rgb, frame_ir, init_box_ir, init_box_rgb)
                pred_bboxes_ir.append(init_box_ir)
                initialized = True
            else:
                pred_bboxes_ir.append([0, 0, 0, 0])
        else:
            box_ir, _ = tracker.update(frame_rgb, frame_ir, frame_idx=frame_idx)
            pred_bboxes_ir.append(box_ir.tolist())

        if use_cuda:
            torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t_start)
        frame_idx += 1

    cap_rgb.release()
    cap_ir.release()

    valid_times = frame_times[5:] if len(frame_times) > 10 else frame_times
    fps = 1.0 / (np.mean(valid_times) + 1e-6) if valid_times else 0.0
    return pred_bboxes_ir, fps
