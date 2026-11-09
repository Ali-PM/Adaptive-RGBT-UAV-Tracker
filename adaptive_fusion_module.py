# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageQualityAssessor:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def compute_rgb_quality(self, rgb_patch):
        if rgb_patch is None or rgb_patch.size == 0: return 0.5
        gray = cv2.cvtColor(rgb_patch.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(rgb_patch.shape) == 3 else rgb_patch.astype(np.uint8)
        gray_norm = gray.astype(np.float32) / 255.0
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = np.clip(laplacian_var / 250.0, 0.2, 1.0)
        contrast = np.clip(np.std(gray_norm) * 4.0, 0.2, 1.0)
        return float(0.6 * sharpness + 0.4 * contrast)

    def compute_thermal_quality(self, ir_patch):
        if ir_patch is None or ir_patch.size == 0: return 0.5
        gray = cv2.cvtColor(ir_patch.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(ir_patch.shape) == 3 else ir_patch.astype(np.uint8)
        gray_norm = gray.astype(np.float32) / 255.0
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = np.clip(laplacian_var / 250.0, 0.2, 1.0)
        contrast = np.clip(np.std(gray_norm) * 4.0, 0.2, 1.0)
        return float(0.5 * sharpness + 0.5 * contrast)

class DualFlowSemanticModulator:
    def __init__(self, csm_cross_ratio=0.20):
        self.csm_cross_ratio = csm_cross_ratio

    def compute_csm_responses(self, feat_rgb, feat_ir, kernel_rgb, kernel_ir):
        resp_rgb_same = F.conv2d(feat_rgb, kernel_rgb) * 0.001
        resp_ir_same = F.conv2d(feat_ir, kernel_ir) * 0.001
        resp_rgb_cross = F.conv2d(feat_rgb, kernel_ir) * 0.001
        resp_ir_cross = F.conv2d(feat_ir, kernel_rgb) * 0.001
        r = self.csm_cross_ratio
        return (1.0 - r) * resp_rgb_same + r * resp_rgb_cross, (1.0 - r) * resp_ir_same + r * resp_ir_cross, resp_rgb_same, resp_ir_same

    @staticmethod
    def compute_ism_score(feat_search, kernel_template, response_map):
        with torch.no_grad():
            if response_map.shape[-1] <= 2 or response_map.shape[-2] <= 2: return 0.5
            mid_scale = feat_search.shape[0] // 2
            res_2d = response_map[mid_scale, 0]
            peak_idx = torch.argmax(res_2d)
            h_res, w_res = res_2d.shape
            py, px = int(peak_idx // w_res), int(peak_idx % w_res)
            kh, kw = kernel_template.shape[2], kernel_template.shape[3]
            y1, x1 = max(0, min(py, feat_search.shape[2] - kh)), max(0, min(px, feat_search.shape[3] - kw))
            x_k = feat_search[mid_scale:mid_scale + 1, :, y1:y1 + kh, x1:x1 + kw]
            if x_k.shape != kernel_template.shape: x_k = F.interpolate(x_k, size=(kh, kw), mode='bilinear', align_corners=False)
            ism_sim = float(torch.sum(F.normalize(kernel_template, p=2, dim=1) * F.normalize(x_k, p=2, dim=1), dim=1).mean().item())
            return float(np.clip((ism_sim + 0.1) / 0.8, 0.1, 1.0))

class LastLayerAdaptiveFusionModule(nn.Module):
    def __init__(self, smoothing_lambda=0.70, mode='full_adaptive', fixed_alpha=0.5, alpha_min=0.05, alpha_max=0.95):
        super(LastLayerAdaptiveFusionModule, self).__init__()
        self.quality_assessor = ImageQualityAssessor()
        self.dfsc_modulator = DualFlowSemanticModulator(csm_cross_ratio=0.20)
        self.smoothing_lambda, self.mode = smoothing_lambda, mode
        self.fixed_alpha, self.alpha_min, self.alpha_max = fixed_alpha, alpha_min, alpha_max
        self.eps = 1e-6
        self.reset_state()

    def reset_state(self):
        self.alpha_smooth = None
        self.last_alpha_q, self.last_alpha_conf = 0.5, 0.5
        self.prev_confidence_rgb, self.prev_confidence_ir, self.prev_confidence_fused = 0.5, 0.5, 0.5
        self.ism_score_rgb, self.ism_score_ir = 0.5, 0.5

    @staticmethod
    def compute_psr_confidence(response_map):
        res = np.asarray(response_map, dtype=np.float32)
        if res.shape[-1] <= 3 or res.shape[-2] <= 3: return 0.5, 5.0
        max_val = float(np.max(res))
        max_loc = np.unravel_index(np.argmax(res), res.shape)
        h, w = res.shape[-2], res.shape[-1]
        y, x = max_loc[-2], max_loc[-1]
        mask = np.ones((h, w), dtype=bool)
        y1, y2, x1, x2 = max(0, y - 2), min(h, y + 3), max(0, x - 2), min(w, x + 3)
        mask[y1:y2, x1:x2] = False
        sidelobe = res[mask]
        psr = (max_val - float(np.mean(sidelobe))) / (float(np.std(sidelobe)) + 1e-6) if sidelobe.size > 0 else 5.0
        return float(np.clip((psr - 1.5) / 7.5, 0.05, 1.0)), float(psr)

    def compute_adaptive_alpha(self, rgb_patch, ir_patch):
        q_rgb = self.quality_assessor.compute_rgb_quality(rgb_patch)
        q_ir = self.quality_assessor.compute_thermal_quality(ir_patch)
        alpha_q = q_rgb / (q_rgb + q_ir + self.eps)
        self.last_alpha_q = float(alpha_q)

        if self.mode == 'rgb_only': return 1.0, q_rgb, q_ir
        if self.mode == 'thermal_only': return 0.0, q_rgb, q_ir
        if self.mode == 'fixed_fusion': return self.fixed_alpha, q_rgb, q_ir
        if self.mode == 'quality_only': return alpha_q, q_rgb, q_ir

        c_rgb = float(self.prev_confidence_rgb)
        c_ir = float(self.prev_confidence_ir)
        score_rgb = c_rgb * float(self.ism_score_rgb) * (0.8 + 0.2 * q_rgb)
        score_ir = c_ir * float(self.ism_score_ir) * (0.8 + 0.2 * q_ir)
        alpha_t = score_rgb / (score_rgb + score_ir + self.eps)

        if score_rgb > 1.3 * score_ir and c_rgb > 0.2: alpha_t = min(alpha_t * 1.2, self.alpha_max)
        elif score_ir > 1.3 * score_rgb and c_ir > 0.2: alpha_t = max(alpha_t * 0.8, self.alpha_min)

        alpha_t = float(np.clip(alpha_t, self.alpha_min, self.alpha_max))
        self.last_alpha_conf = alpha_t

        if self.mode == 'quality_confidence': return alpha_t, q_rgb, q_ir
        if self.alpha_smooth is None:
            self.alpha_smooth = alpha_t
        else:
            lam = 0.20 if abs(alpha_t - self.alpha_smooth) > 0.35 else self.smoothing_lambda
            self.alpha_smooth = lam * self.alpha_smooth + (1.0 - lam) * alpha_t
        return float(self.alpha_smooth), q_rgb, q_ir

    @staticmethod
    def _global_zscore(resp):
        return (resp - torch.mean(resp)) / (torch.std(resp, unbiased=False) + 1e-6)

    def forward(self, feat_rgb, feat_thermal, kernel_rgb, kernel_thermal, rgb_patch=None, ir_patch=None):
        resp_rgb_csm, resp_ir_csm, resp_rgb_same, resp_ir_same = self.dfsc_modulator.compute_csm_responses(feat_rgb, feat_thermal, kernel_rgb, kernel_thermal)
        if resp_rgb_same.shape[-1] > 3:
            mid_s = resp_rgb_same.shape[0] // 2
            self.prev_confidence_rgb, _ = self.compute_psr_confidence(resp_rgb_same[mid_s, 0].detach().cpu().numpy())
            self.prev_confidence_ir, _ = self.compute_psr_confidence(resp_ir_same[mid_s, 0].detach().cpu().numpy())

        if self.mode not in ('rgb_only', 'thermal_only', 'fixed_fusion', 'quality_only'):
            self.ism_score_rgb = self.dfsc_modulator.compute_ism_score(feat_rgb, kernel_rgb, resp_rgb_same)
            self.ism_score_ir = self.dfsc_modulator.compute_ism_score(feat_thermal, kernel_thermal, resp_ir_same)
            resp_rgb_use, resp_ir_use = (resp_rgb_csm, resp_ir_csm) if (self.prev_confidence_rgb > 0.15 and self.prev_confidence_ir > 0.15) else (resp_rgb_same, resp_ir_same)
        else:
            resp_rgb_use, resp_ir_use = resp_rgb_same, resp_ir_same

        alpha_tilde, q_rgb, q_ir = self.compute_adaptive_alpha(rgb_patch, ir_patch)
        feat_fusion = alpha_tilde * feat_rgb + (1.0 - alpha_tilde) * feat_thermal
        resp_fused = alpha_tilde * self._global_zscore(resp_rgb_use) + (1.0 - alpha_tilde) * self._global_zscore(resp_ir_use)
        return feat_fusion, resp_fused, resp_rgb_use, resp_ir_use, alpha_tilde, q_rgb, q_ir


class TargetKalmanPredictor:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=3e-2):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.diag([1e-2, 1e-2, process_noise, process_noise]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self.initialized = False
        self.stable_steps = 0
        self.vel_yx = np.zeros(2, dtype=np.float32)

    def init_state(self, center_yx, sz_hw=None):
        self.kf.statePost = np.array([[center_yx[1]], [center_yx[0]], [0.0], [0.0]], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        self.stable_steps = 0
        self.vel_yx = np.zeros(2, dtype=np.float32)
        self.initialized = True

    def update(self, measured_center_yx, prev_center_yx, confidence, conf_threshold=0.25):
        if not self.initialized:
            self.init_state(measured_center_yx)
            c = np.array(measured_center_yx, dtype=np.float32)
            return c, c.copy()

        self.kf.predict()
        meas_cy, meas_cx = float(measured_center_yx[0]), float(measured_center_yx[1])
        prev_cy, prev_cx = float(prev_center_yx[0]), float(prev_center_yx[1])
        
        step_y, step_x = meas_cy - prev_cy, meas_cx - prev_cx
        step_dist = float(np.hypot(step_y, step_x))
        vel_mag = float(np.hypot(self.vel_yx[0], self.vel_yx[1]))

        if confidence >= 0.18 and step_dist <= 28.0:
            self.stable_steps += 1
            corrected = self.kf.correct(np.array([[meas_cx], [meas_cy]], dtype=np.float32))
            self.vel_yx = 0.75 * self.vel_yx + 0.25 * np.array([step_y, step_x], dtype=np.float32)
            
            if self.stable_steps >= 8 and confidence < 0.225 and step_dist < 12.0 and vel_mag < 3.5:
                search_cy = meas_cy + np.clip(0.06 * float(self.vel_yx[0]), -0.8, 0.8)
                search_cx = meas_cx + np.clip(0.06 * float(self.vel_yx[1]), -0.8, 0.8)
            else:
                search_cy, search_cx = meas_cy, meas_cx

            corr_dy = np.clip(float(corrected[1, 0]) - search_cy, -1.2, 1.2)
            corr_dx = np.clip(float(corrected[0, 0]) - search_cx, -1.2, 1.2)
            box_cy = search_cy + 0.08 * corr_dy
            box_cx = search_cx + 0.08 * corr_dx

        elif self.stable_steps >= 8 and confidence < 0.12 and step_dist > 32.0:
            self.stable_steps = 0
            search_cy = prev_cy + float(np.clip(self.vel_yx[0], -12.0, 12.0))
            search_cx = prev_cx + float(np.clip(self.vel_yx[1], -12.0, 12.0))
            box_cy, box_cx = search_cy, search_cx
            self.kf.statePost = np.array([[search_cx], [search_cy], [0.0], [0.0]], dtype=np.float32)
        else:
            self.stable_steps = 0
            self.vel_yx *= 0.5
            search_cy, search_cx = meas_cy, meas_cx
            box_cy, box_cx = meas_cy, meas_cx
            self.kf.statePost = np.array([[meas_cx], [meas_cy], [0.0], [0.0]], dtype=np.float32)

        return np.array([search_cy, search_cx], dtype=np.float32), np.array([box_cy, box_cx], dtype=np.float32)
