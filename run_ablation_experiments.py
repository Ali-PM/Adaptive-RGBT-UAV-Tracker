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
    attributes = data.get('attribute', data.get('attributes', None))
    return exist_flags, gt_rects, attributes

def load_anti_uav_annotation(seq_dir, modality='infrared'):
    json_candidates = [
        os.path.join(seq_dir, f'{modality}.json'),
        os.path.join(seq_dir, 'IR_label.json' if modality == 'infrared' else 'visible.json'),
        os.path.join(seq_dir, 'groundtruth.json')
    ]
    for jp in json_candidates:
        if os.path.exists(jp):
            exist_flags, gt_rects, _ = load_anti_uav_json(jp)
            return exist_flags, gt_rects

    txt_candidates = glob.glob(os.path.join(seq_dir, '*.txt'))
    if txt_candidates:
        gt_rect = np.loadtxt(txt_candidates[0], delimiter=',').tolist()
        exist = [1 if (len(r) == 4 and r[2] > 0 and r[3] > 0) else 0 for r in gt_rect]
        return exist, gt_rect

    raise FileNotFoundError(f"Annotation file not found in {seq_dir} for {modality}")

def extract_sequence_attributes(seq_dir, exist_ir, gt_rects_ir):
    attrs = set()
    seq_name = os.path.basename(seq_dir)
    attr_names = ['OV', 'OC', 'FM', 'SV', 'LI', 'TC', 'LR']

    for jp in [os.path.join(seq_dir, 'infrared.json'), os.path.join(seq_dir, 'visible.json')]:
        if os.path.exists(jp):
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                raw_attr = d.get('attribute', d.get('attributes', d.get('attr', None)))
                if isinstance(raw_attr, list):
                    if len(raw_attr) == 7 and all(isinstance(x, (int, float)) for x in raw_attr):
                        for idx, val in enumerate(raw_attr):
                            if val > 0: attrs.add(attr_names[idx])
                    else:
                        for item in raw_attr:
                            s_up = str(item).upper()
                            for an in attr_names:
                                if an in s_up: attrs.add(an)
                elif isinstance(raw_attr, dict):
                    for k, v in raw_attr.items():
                        if v and str(k).upper() in attr_names:
                            attrs.add(str(k).upper())
            except Exception:
                pass

    valid_boxes = [(t, r) for t, (v, r) in enumerate(zip(exist_ir, gt_rects_ir))
                   if v > 0 and isinstance(r, (list, tuple)) and len(r) == 4 and r[2] > 0 and r[3] > 0]

    if any(v == 0 for v in exist_ir):
        attrs.add('OV')

    if valid_boxes:
        _, b0 = valid_boxes[0]
        area0 = float(b0[2] * b0[3])
        scale0 = np.sqrt(area0)

        max_step = 0.0
        min_area = area0
        sv_flag = False
        oc_flag = False

        prev_t, prev_b = valid_boxes[0]
        prev_cx, prev_cy = prev_b[0] + prev_b[2] / 2.0, prev_b[1] + prev_b[3] / 2.0

        for idx in range(1, len(valid_boxes)):
            curr_t, curr_b = valid_boxes[idx]
            curr_cx, curr_cy = curr_b[0] + curr_b[2] / 2.0, curr_b[1] + curr_b[3] / 2.0
            curr_area = float(curr_b[2] * curr_b[3])
            curr_scale = np.sqrt(max(1.0, curr_area))

            dt = max(1, curr_t - prev_t)
            step = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
            if (dt == 1 and step > 60.0) or (dt > 1 and step > 35.0):
                max_step = max(max_step, 65.0)
            else:
                max_step = max(max_step, step)

            ratio_area = curr_area / (area0 + 1e-6)
            ratio_scale = curr_scale / (scale0 + 1e-6)
            if ratio_area < 0.66 or ratio_area > 1.50 or ratio_scale < 0.66 or ratio_scale > 1.50:
                sv_flag = True

            min_area = min(min_area, curr_area)

            if curr_t - prev_t > 1 and (20 < prev_cx < 620 and 20 < prev_cy < 490):
                oc_flag = True

            prev_t, prev_b, prev_cx, prev_cy = curr_t, curr_b, curr_cx, curr_cy

        if max_step > 60.0:
            attrs.add('FM')
        if sv_flag:
            attrs.add('SV')
        if min_area < 400.0:
            attrs.add('LR')
        if oc_flag:
            attrs.add('OC')

    parts = seq_name.split('_')
    if len(parts) >= 2 and len(parts[1]) >= 2 and parts[1][:2].isdigit():
        hour = int(parts[1][:2])
        if hour >= 18 or hour <= 5:
            attrs.add('LI')

    rgb_mp4 = os.path.join(seq_dir, "visible.mp4")
    ir_mp4 = os.path.join(seq_dir, "infrared.mp4")
    if os.path.exists(rgb_mp4) and valid_boxes:
        cap = cv2.VideoCapture(rgb_mp4)
        ret, frame = cap.read()
        cap.release()
        if ret and np.mean(frame) < 75.0:
            attrs.add('LI')

    if os.path.exists(ir_mp4) and valid_boxes:
        cap = cv2.VideoCapture(ir_mp4)
        ret, frame = cap.read()
        cap.release()
        if ret:
            gray_ir = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, b0 = valid_boxes[0]
            x, y, w, h = [int(v) for v in b0]
            H, W = gray_ir.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            bg_x1, bg_y1 = max(0, x - w), max(0, y - h)
            bg_x2, bg_y2 = min(W, x + 2 * w), min(H, y + 2 * h)
            if x2 > x1 and y2 > y1:
                target_temp = float(np.mean(gray_ir[y1:y2, x1:x2]))
                bg_patch = gray_ir[bg_y1:bg_y2, bg_x1:bg_x2]
                bg_temp = float(np.mean(bg_patch))
                bg_std = float(np.std(gray_ir))
                if abs(target_temp - bg_temp) < 42.0 or bg_std > 32.0:
                    attrs.add('TC')
                if abs(target_temp - bg_temp) < 18.0 and bg_std > 42.0:
                    attrs.add('OC')

    return attrs

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

def evaluate_sequence_metrics(pred_bboxes, gt_rects, exist_flags, return_cle=False, return_auc=False):
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
    
    if ious:
        thr_s = np.arange(0.0, 1.05, 0.05)
        iou_arr = np.asarray(ious, dtype=np.float32)
        success_auc = float(np.mean([np.mean(iou_arr >= th) * 100.0 for th in thr_s]))
    else:
        success_auc = 0.0

    mean_cle = float(np.mean(valid_cle_list)) if valid_cle_list else 500.0

    if return_cle and return_auc:
        return m_sa, precision_20, success_50, success_auc, mean_cle
    if return_cle:
        return m_sa, precision_20, success_50, mean_cle
    return m_sa, precision_20, success_50

def run_video_pair_mp4(tracker, rgb_mp4_path, ir_mp4_path, gt_rects_ir, exist_flags_ir,
                       gt_rects_rgb=None, exist_flags_rgb=None, return_rgb_boxes=False, **kwargs):
    cap_rgb = cv2.VideoCapture(rgb_mp4_path)
    cap_ir = cv2.VideoCapture(ir_mp4_path)

    pred_bboxes_ir, pred_bboxes_rgb, frame_times = [], [], []
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
                pred_bboxes_rgb.append(init_box_rgb if init_box_rgb is not None else init_box_ir)
                initialized = True
            else:
                pred_bboxes_ir.append([0, 0, 0, 0])
                pred_bboxes_rgb.append([0, 0, 0, 0])
        else:
            box_ir, box_rgb = tracker.update(frame_rgb, frame_ir, frame_idx=frame_idx)
            pred_bboxes_ir.append(box_ir.tolist())
            pred_bboxes_rgb.append(box_rgb.tolist())

        if use_cuda:
            torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t_start)
        frame_idx += 1

    cap_rgb.release()
    cap_ir.release()

    valid_times = frame_times[5:] if len(frame_times) > 10 else frame_times
    fps = 1.0 / (np.mean(valid_times) + 1e-6) if valid_times else 0.0
    if return_rgb_boxes:
        return pred_bboxes_ir, pred_bboxes_rgb, fps
    return pred_bboxes_ir, fps
