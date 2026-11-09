# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
import os
from collections import namedtuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from adaptive_fusion_module import LastLayerAdaptiveFusionModule, TargetKalmanPredictor

class SiamFCBackbone(nn.Module):
    def __init__(self):
        super(SiamFCBackbone, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2), nn.BatchNorm2d(96, eps=1e-6, momentum=0.05), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2), nn.BatchNorm2d(256, eps=1e-6, momentum=0.05), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384, eps=1e-6, momentum=0.05), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2), nn.BatchNorm2d(384, eps=1e-6, momentum=0.05), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
    def forward(self, x): return self.feature(x)

class TrackerAdaptiveSiamFC_RGBT(object):
    def __init__(self, net_path, fusion_mode='full_adaptive', use_kalman=True, smoothing_lambda=0.70, num_cpu_threads=4, **kargs):
        self.name = f'AdaptiveSiamFC_{fusion_mode}' + ('_KF' if use_kalman else '')
        self.cfg = self.parse_args(**kargs)
        self.use_kalman = use_kalman
        self.fusion_mode = fusion_mode
        torch.set_num_threads(num_cpu_threads)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.net = SiamFCBackbone().to(self.device).eval()
        self._load_pretrained_weights(net_path)
        self.fusion_module = LastLayerAdaptiveFusionModule(smoothing_lambda=smoothing_lambda, mode=fusion_mode).to(self.device)
        self.kalman_ir = TargetKalmanPredictor()
        self.kalman_rgb = TargetKalmanPredictor()
        self.frame_logs, self.is_lost, self.low_count, self.search_expand = [], False, 0, 1.0

    def _load_pretrained_weights(self, net_path):
        if not os.path.exists(net_path): return
        try: state_dict = torch.load(net_path, map_location='cpu', weights_only=False)
        except: state_dict = torch.load(net_path, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        model_dict = self.net.state_dict()
        clean_state = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace('net.', '').replace('backbone.', '')
            if not new_k.startswith('feature.'): new_k = 'feature.' + new_k
            v = torch.as_tensor(np.asarray(v))
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                clean_state[new_k] = v
        self.net.load_state_dict(clean_state, strict=False)

    def parse_args(self, **kargs):
        cfg = {'exemplar_sz': 127, 'instance_sz': 255, 'context': 0.85, 'scale_num': 3, 'scale_step': 1.0375,
               'scale_lr': 0.52, 'scale_penalty': 0.9745, 'window_influence': 0.10, 'response_sz': 17, 'response_up': 16,
               'total_stride': 8, 'lost_conf': 0.03, 'lost_patience': 10, 'recover_conf': 0.1, 'max_expand': 2.0, 'expand_rate': 1.05}
        cfg.update(kargs)
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, img_rgb, img_ir, init_bbox_ir, init_bbox_rgb=None):
        img_rgb, img_ir = np.asarray(img_rgb), np.asarray(img_ir)
        self.fusion_module.reset_state()
        self.frame_logs, self.is_lost, self.low_count, self.search_expand = [], False, 0, 1.0
        
        h_ir, w_ir = img_ir.shape[:2]
        h_rgb, w_rgb = img_rgb.shape[:2]
        self.img_shape_ir, self.img_shape_rgb = (h_ir, w_ir), (h_rgb, w_rgb)
        self.scale_ir_to_rgb = np.array([h_rgb/h_ir, w_rgb/w_ir], dtype=np.float32)

        if not init_bbox_rgb or len(init_bbox_rgb) < 4:
            init_bbox_rgb = [init_bbox_ir[0]*self.scale_ir_to_rgb[1], init_bbox_ir[1]*self.scale_ir_to_rgb[0], 
                             init_bbox_ir[2]*self.scale_ir_to_rgb[1], init_bbox_ir[3]*self.scale_ir_to_rgb[0]]

        self.center_ir = np.array([init_bbox_ir[1]-1+(init_bbox_ir[3]-1)/2.0, init_bbox_ir[0]-1+(init_bbox_ir[2]-1)/2.0], dtype=np.float32)
        self.target_sz_ir = np.array([init_bbox_ir[3], init_bbox_ir[2]], dtype=np.float32)
        self.center_rgb = np.array([init_bbox_rgb[1]-1+(init_bbox_rgb[3]-1)/2.0, init_bbox_rgb[0]-1+(init_bbox_rgb[2]-1)/2.0], dtype=np.float32)
        self.target_sz_rgb = np.array([init_bbox_rgb[3], init_bbox_rgb[2]], dtype=np.float32)
        self.offset_rgb_from_ir = self.center_rgb - (self.center_ir * self.scale_ir_to_rgb)

        self.kalman_ir.init_state(self.center_ir)
        self.kalman_rgb.init_state(self.center_rgb)

        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        self.scale_factors = self.cfg.scale_step ** np.linspace(-(self.cfg.scale_num//2), self.cfg.scale_num//2, self.cfg.scale_num)

        self.z_sz_ir = max(60.0, np.sqrt(np.prod(self.target_sz_ir + self.cfg.context * np.sum(self.target_sz_ir))))
        self.x_sz_ir = self.z_sz_ir * self.cfg.instance_sz / self.cfg.exemplar_sz
        self.z_sz_rgb = max(60.0, np.sqrt(np.prod(self.target_sz_rgb + self.cfg.context * np.sum(self.target_sz_rgb))))
        self.x_sz_rgb = self.z_sz_rgb * self.cfg.instance_sz / self.cfg.exemplar_sz

        self.avg_color_rgb, self.avg_color_ir = np.mean(img_rgb, axis=(0,1)), np.mean(img_ir, axis=(0,1))
        z_rgb = self._crop_and_resize(img_rgb, self.center_rgb, self.z_sz_rgb, self.cfg.exemplar_sz, self.avg_color_rgb)
        z_ir = self._crop_and_resize(img_ir, self.center_ir, self.z_sz_ir, self.cfg.exemplar_sz, self.avg_color_ir)

        with torch.no_grad():
            self.kernel_rgb = self.net(torch.from_numpy(z_rgb).to(self.device).permute(2,0,1).unsqueeze(0).float())
            self.kernel_ir = self.net(torch.from_numpy(z_ir).to(self.device).permute(2,0,1).unsqueeze(0).float())
        
        self.fusion_module.alpha_smooth = None
        self._save_good_state()
        self._log(0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, False)

    def _save_good_state(self):
        self.good_state = {k: getattr(self, k).copy() if hasattr(getattr(self, k), 'copy') else getattr(self, k) 
                           for k in ['center_ir', 'center_rgb', 'target_sz_ir', 'target_sz_rgb', 'z_sz_ir', 'z_sz_rgb']}

    def _restore_good_state(self):
        for k, v in self.good_state.items(): setattr(self, k, v.copy() if hasattr(v, 'copy') else v)
        self.x_sz_ir = self.z_sz_ir * self.cfg.instance_sz / self.cfg.exemplar_sz
        self.x_sz_rgb = self.z_sz_rgb * self.cfg.instance_sz / self.cfg.exemplar_sz
        self.kalman_ir.init_state(self.center_ir)
        self.kalman_rgb.init_state(self.center_rgb)

    def _localize_on_response_stack(self, responses_3d):
        upsampled = np.stack([cv2.resize(t, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC) for t in responses_3d], axis=0)
        upsampled -= upsampled.min()
        upsampled[:self.cfg.scale_num//2] *= self.cfg.scale_penalty
        upsampled[self.cfg.scale_num//2+1:] *= self.cfg.scale_penalty
        scale_id = int(np.argmax(np.amax(upsampled, axis=(1, 2))))
        response = upsampled[scale_id].copy()
        response /= (response.sum() + 1e-16)
        response = (1.0 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        disp_in_instance = (np.array(loc, dtype=np.float32) - (self.upscale_sz // 2)) * self.cfg.total_stride / self.cfg.response_up
        return disp_in_instance, scale_id

    def update(self, img_rgb, img_ir, frame_idx=1):
        prev_ir, prev_rgb = self.center_ir.copy(), self.center_rgb.copy()
        x_sz_ir, x_sz_rgb = self.x_sz_ir * self.search_expand, self.x_sz_rgb * self.search_expand

        x_ir = [self._crop_and_resize(img_ir, self.center_ir, x_sz_ir * f, self.cfg.instance_sz, self.avg_color_ir) for f in self.scale_factors]
        x_rgb = [self._crop_and_resize(img_rgb, self.center_rgb, x_sz_rgb * f, self.cfg.instance_sz, self.avg_color_rgb) for f in self.scale_factors]

        mid = self.cfg.scale_num // 2
        p_ir, p_rgb = self._center_crop(x_ir[mid]), self._center_crop(x_rgb[mid])

        with torch.no_grad():
            f_ir = self.net(torch.from_numpy(np.stack(x_ir)).to(self.device).permute(0,3,1,2).float())
            f_rgb = self.net(torch.from_numpy(np.stack(x_rgb)).to(self.device).permute(0,3,1,2).float())
            _, resp_fused, resp_rgb, resp_ir, alpha, q_rgb, q_ir = self.fusion_module(f_rgb, f_ir, self.kernel_rgb, self.kernel_ir, p_rgb, p_ir)

        conf_rgb, conf_ir = self.fusion_module.prev_confidence_rgb, self.fusion_module.prev_confidence_ir
        conf_fused, _ = self.fusion_module.compute_psr_confidence(resp_fused.squeeze(1).cpu().numpy()[mid])
        eff_conf = max(conf_fused, alpha * conf_rgb + (1.0 - alpha) * conf_ir)
        self.fusion_module.prev_confidence_fused = eff_conf

        if self.is_lost:
            if eff_conf >= self.cfg.recover_conf: self.is_lost, self.low_count = False, 0
            else:
                self.search_expand = min(self.cfg.max_expand, self.search_expand * self.cfg.expand_rate)
                self._log(frame_idx, alpha, q_rgb, q_ir, conf_rgb, conf_ir, eff_conf, True)
                return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
        else:
            self.low_count = self.low_count + 1 if eff_conf < self.cfg.lost_conf else 0
            if self.low_count >= self.cfg.lost_patience:
                self.is_lost, self.search_expand = True, 1.5
                self._restore_good_state()
                self._log(frame_idx, alpha, q_rgb, q_ir, conf_rgb, conf_ir, eff_conf, True)
                return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)

        if alpha >= 0.85 and self.fusion_mode not in ('fixed_fusion', 'thermal_only'):
            disp, sid = self._localize_on_response_stack(resp_rgb.squeeze(1).cpu().numpy())
            self.center_rgb += disp * x_sz_rgb * self.scale_factors[sid] / self.cfg.instance_sz
            if conf_rgb > 0.2: self.center_ir = 0.9 * ((self.center_rgb - self.offset_rgb_from_ir) / self.scale_ir_to_rgb) + 0.1 * self.center_ir
        elif alpha <= 0.15 and self.fusion_mode not in ('fixed_fusion', 'rgb_only'):
            disp, sid = self._localize_on_response_stack(resp_ir.squeeze(1).cpu().numpy())
            self.center_ir += disp * x_sz_ir * self.scale_factors[sid] / self.cfg.instance_sz
            if conf_ir > 0.2: self.center_rgb = 0.9 * (self.center_ir * self.scale_ir_to_rgb + self.offset_rgb_from_ir) + 0.1 * self.center_rgb
        else:
            disp_r, sid_r = self._localize_on_response_stack(resp_rgb.squeeze(1).cpu().numpy())
            disp_i, sid_i = self._localize_on_response_stack(resp_ir.squeeze(1).cpu().numpy())
            self.center_rgb += disp_r * x_sz_rgb * self.scale_factors[sid_r] / self.cfg.instance_sz
            self.center_ir += disp_i * x_sz_ir * self.scale_factors[sid_i] / self.cfg.instance_sz
            sid = sid_r if alpha > 0.5 else sid_i

        if self.use_kalman:
            self.center_ir, out_center_ir = self.kalman_ir.update(self.center_ir, prev_ir, eff_conf)
            self.center_rgb, out_center_rgb = self.kalman_rgb.update(self.center_rgb, prev_rgb, eff_conf)
        else:
            out_center_ir = self.center_ir.copy()
            out_center_rgb = self.center_rgb.copy()

        self.center_ir = np.clip(self.center_ir, [0,0], [self.img_shape_ir[0]-1, self.img_shape_ir[1]-1])
        self.center_rgb = np.clip(self.center_rgb, [0,0], [self.img_shape_rgb[0]-1, self.img_shape_rgb[1]-1])
        out_center_ir = np.clip(out_center_ir, [0,0], [self.img_shape_ir[0]-1, self.img_shape_ir[1]-1])
        out_center_rgb = np.clip(out_center_rgb, [0,0], [self.img_shape_rgb[0]-1, self.img_shape_rgb[1]-1])

        scale = (1.0 - self.cfg.scale_lr) + self.cfg.scale_lr * self.scale_factors[sid] if eff_conf >= 0.15 else 1.0
        self.target_sz_ir = np.clip(self.target_sz_ir * scale, 8.0, 300.0)
        self.z_sz_ir = np.clip(self.z_sz_ir * scale, 48.0, 450.0)
        self.x_sz_ir = self.z_sz_ir * self.cfg.instance_sz / self.cfg.exemplar_sz

        self.target_sz_rgb = np.clip(self.target_sz_rgb * scale, 16.0, 600.0)
        self.z_sz_rgb = np.clip(self.z_sz_rgb * scale, 96.0, 900.0)
        self.x_sz_rgb = self.z_sz_rgb * self.cfg.instance_sz / self.cfg.exemplar_sz

        self.search_expand = 1.0
        if eff_conf >= self.cfg.recover_conf: self._save_good_state()

        box_ir = np.array([out_center_ir[1]+1-(self.target_sz_ir[1]-1)/2., out_center_ir[0]+1-(self.target_sz_ir[0]-1)/2., self.target_sz_ir[1], self.target_sz_ir[0]], dtype=np.float32)
        box_rgb = np.array([out_center_rgb[1]+1-(self.target_sz_rgb[1]-1)/2., out_center_rgb[0]+1-(self.target_sz_rgb[0]-1)/2., self.target_sz_rgb[1], self.target_sz_rgb[0]], dtype=np.float32)

        self._log(frame_idx, alpha, q_rgb, q_ir, conf_rgb, conf_ir, eff_conf, False)
        return box_ir, box_rgb

    @staticmethod
    def _center_crop(p): h,w=p.shape[:2]; d=int(h*0.25); return p[d:h-d, d:w-d]

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        size = max(4, int(round(size)))
        c = np.concatenate((np.round(center-(size-1)/2.), np.round(center-(size-1)/2.)+size)).astype(int)
        pads = np.concatenate((-c[:2], c[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0: image = cv2.copyMakeBorder(image, npad, npad, npad, npad, cv2.BORDER_CONSTANT, value=pad_color.tolist())
        c += npad
        return cv2.resize(image[c[0]:c[2], c[1]:c[3]], (out_size, out_size))
        
    def _log(self, f_idx, alpha, q_r, q_i, c_r, c_i, c_f, lost):
        self.frame_logs.append({'frame': f_idx, 'alpha_smooth': alpha, 'q_rgb': q_r, 'q_ir': q_i, 'conf_rgb': c_r, 'conf_ir': c_i, 'confidence': c_f, 'lost': lost})
