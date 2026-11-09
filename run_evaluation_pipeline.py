# -*- coding: utf-8 -*-
import os, json, csv
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from siamfc_adaptive_rgbt import TrackerAdaptiveSiamFC_RGBT
from run_ablation_experiments import (
    load_anti_uav_annotation,
    evaluate_sequence_metrics,
    run_video_pair_mp4,
    compute_iou,
    compute_center_error
)

def warmup_gpu(model_path):
    if torch.cuda.is_available():
        dummy_tracker = TrackerAdaptiveSiamFC_RGBT(net_path=model_path, fusion_mode='rgb_only', use_kalman=False)
        dummy_img = np.zeros((255, 255, 3), dtype=np.uint8)
        dummy_tracker.init(dummy_img, dummy_img, [50, 50, 40, 40])
        for _ in range(5): dummy_tracker.update(dummy_img, dummy_img, frame_idx=1)
        torch.cuda.synchronize()

def calc_success_curve(ious):
    thresholds = np.arange(0, 1.05, 0.05)
    return thresholds, [np.mean([1.0 if iou >= t else 0.0 for iou in ious]) * 100 for t in thresholds]

def calc_precision_curve(cles):
    thresholds = np.arange(0, 51, 1)
    return thresholds, [np.mean([1.0 if cle <= t else 0.0 for cle in cles]) * 100 for t in thresholds]

def save_archive_plots(archive_dir, summary_rows, seq_frame_logs, all_modes_raw_data):
    try:
        modes = [r['mode_title'] for r in summary_rows]
        msa_vals = [r['mSA'] for r in summary_rows]
        prec_vals = [r['Precision'] for r in summary_rows]
        succ_vals = [r['Success'] for r in summary_rows]

        x = np.arange(len(modes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, msa_vals, width, label='mSA (%)', color='#2b5c8f')
        ax.bar(x, prec_vals, width, label='Precision (%)', color='#388e3c')
        ax.bar(x + width, succ_vals, width, label='Success (%)', color='#d95f02')
        ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax.set_title('Ablation Study Comparison on Anti-UAV Dataset', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(archive_dir, 'ablation_comparison_chart.png'), dpi=200)
        plt.close(fig)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-']
        fig_succ, ax_succ = plt.subplots(figsize=(8, 6))
        fig_prec, ax_prec = plt.subplots(figsize=(8, 6))
        
        for idx, mode in enumerate(all_modes_raw_data.keys()):
            if not all_modes_raw_data[mode]['ious']: continue
            c, ls = colors[idx % len(colors)], linestyles[idx % len(linestyles)]
            thr_s, succ = calc_success_curve(all_modes_raw_data[mode]['ious'])
            ax_succ.plot(thr_s, succ, label=f"{mode} [{np.mean(succ):.1f}%]", color=c, linestyle=ls, linewidth=2)
            thr_p, prec = calc_precision_curve(all_modes_raw_data[mode]['cles'])
            p20 = prec[20] if len(prec) > 20 else 0
            ax_prec.plot(thr_p, prec, label=f"{mode} [{p20:.1f}%]", color=c, linestyle=ls, linewidth=2)

        ax_succ.set_xlabel('Overlap Threshold'); ax_succ.set_ylabel('Success Rate (%)')
        ax_succ.set_title('Success Plot (OPE)'); ax_succ.set_xlim(0, 1); ax_succ.set_ylim(0, 100)
        ax_succ.grid(True, linestyle='--', alpha=0.6); ax_succ.legend(loc='lower left')
        fig_succ.tight_layout(); fig_succ.savefig(os.path.join(archive_dir, 'ope_success_plot.png'), dpi=200); plt.close(fig_succ)

        ax_prec.set_xlabel('Location Error Threshold (Pixels)'); ax_prec.set_ylabel('Precision (%)')
        ax_prec.set_title('Precision Plot (OPE)'); ax_prec.set_xlim(0, 50); ax_prec.set_ylim(0, 100)
        ax_prec.grid(True, linestyle='--', alpha=0.6); ax_prec.legend(loc='lower right')
        fig_prec.tight_layout(); fig_prec.savefig(os.path.join(archive_dir, 'ope_precision_plot.png'), dpi=200); plt.close(fig_prec)

        for seq_name, logs in seq_frame_logs.items():
            if not logs: continue
            frames = [item['frame'] for item in logs]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.plot(frames, [item['q_rgb'] for item in logs], label='Q_RGB,t (Visible Quality)', color='#1f77b4', alpha=0.85)
            ax1.plot(frames, [item['q_ir'] for item in logs], label='Q_Thermal,t (IR Quality)', color='#ff7f0e', alpha=0.85)
            ax1.set_ylabel('Modality Quality', fontweight='bold'); ax1.set_title(f'Temporal Analysis - Sequence: {seq_name}', fontweight='bold')
            ax1.legend(loc='upper right'); ax1.grid(True, linestyle='--', alpha=0.4)

            ax2.plot(frames, [item['alpha_smooth'] for item in logs], label='Smoothed Adaptive Weight ~alpha_t', color='#2ca02c', linewidth=1.8)
            ax2.plot(frames, [item['confidence'] for item in logs], label='Tracker Confidence c_t', color='#9467bd', linestyle=':', alpha=0.75)
            ax2.set_xlabel('Frame Number (t)', fontweight='bold'); ax2.set_ylabel('Weight / Confidence', fontweight='bold')
            ax2.set_ylim(0, 1.05); ax2.legend(loc='upper right'); ax2.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout(); plt.savefig(os.path.join(archive_dir, f'temporal_analysis_{seq_name}.png'), dpi=200); plt.close(fig)
    except Exception as e:
        print(f"[!] Warning plotting: {e}")

def run_full_evaluation(model_path, dataset_val_root, selected_sequences=None, output_root="results_archive"):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_archive_dir = os.path.join(output_root, f"run_{run_timestamp}")
    os.makedirs(current_archive_dir, exist_ok=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    ABLATION_MODES = [
        ("1. RGB Only",                  "rgb_only",           False),
        ("2. Thermal Only",              "thermal_only",       False),
        ("3. Fixed RGB-T Fusion",        "fixed_fusion",       False),
        ("4. Adaptive (Quality Only)",   "quality_only",       False),
        ("5. Adaptive (Quality + Conf)", "quality_confidence", False),
        ("6. Adaptive + Temp Smoothing", "full_adaptive",      False),
        ("7. Proposed Full (+ Kalman)",  "full_adaptive",      True),
    ]

    all_folders = sorted([f.path for f in os.scandir(dataset_val_root) if f.is_dir()])
    seq_folders = [p for p in all_folders if os.path.basename(p) in set(selected_sequences)] if selected_sequences else all_folders

    print("=" * 94)
    print(f"شروع ارزیابی جامع ردیاب تطبیقی RGB-T روی پردازنده: [{device_name}]")
    print(f"[*] تعداد ویدیوهای انتخاب شده برای این تست: {len(seq_folders)} ویدیو")
    print("=" * 94)
    warmup_gpu(model_path)

    all_modes_metrics = {title: [] for title, _, _ in ABLATION_MODES}
    all_modes_raw_data = {title: {'ious': [], 'cles': []} for title, _, _ in ABLATION_MODES}
    proposed_frame_logs = {}

    for seq_idx, test_seq in enumerate(seq_folders, 1):
        seq_name = os.path.basename(test_seq)
        print(f"\n[{seq_idx}/{len(seq_folders)}] ویدئوی در حال تست: {seq_name}")
        rgb_mp4, ir_mp4 = os.path.join(test_seq, "visible.mp4"), os.path.join(test_seq, "infrared.mp4")

        try:
            exist_ir, gt_rects_ir = load_anti_uav_annotation(test_seq, modality='infrared')
            exist_rgb, gt_rects_rgb = load_anti_uav_annotation(test_seq, modality='visible')
        except Exception as e:
            print(f"[!] خطا در برچسب {seq_name}: {e}"); continue

        for title, mode, use_kf in ABLATION_MODES:
            print(f"  -> در حال اجرای حالت: {title} ...")
            tracker = TrackerAdaptiveSiamFC_RGBT(net_path=model_path, fusion_mode=mode, use_kalman=use_kf, num_cpu_threads=4)
            pred_boxes_ir, fps = run_video_pair_mp4(
                tracker, rgb_mp4, ir_mp4, gt_rects_ir=gt_rects_ir, exist_flags_ir=exist_ir,
                gt_rects_rgb=gt_rects_rgb, exist_flags_rgb=exist_rgb
            )
            
            seq_ious, seq_cles = [], []
            for t in range(min(len(pred_boxes_ir), len(gt_rects_ir), len(exist_ir))):
                if exist_ir[t] > 0 and len(gt_rects_ir[t]) == 4 and gt_rects_ir[t][2] > 0:
                    pred, gt = pred_boxes_ir[t], gt_rects_ir[t]
                    p_hat = 1 if (len(pred) == 4 and pred[2] > 0 and pred[3] > 0) else 0
                    seq_ious.append(compute_iou(pred, gt) if p_hat > 0 else 0.0)
                    seq_cles.append(compute_center_error(pred, gt) if p_hat > 0 else 500.0)
                else:
                    seq_ious.append(1.0 if (len(pred_boxes_ir[t]) != 4 or pred_boxes_ir[t][2] <= 0) else 0.0)
            
            all_modes_raw_data[title]['ious'].extend(seq_ious)
            all_modes_raw_data[title]['cles'].extend([c for c in seq_cles if c < 500])

            m_sa, prec, succ, mean_cle = evaluate_sequence_metrics(pred_boxes_ir, gt_rects_ir, exist_ir, return_cle=True)
            print(f"     نتیجه: mSA = {m_sa:.2f}% | Prec = {prec:.2f}% | Succ = {succ:.2f}% | CLE = {mean_cle:.1f}px | Speed = {fps:.1f} FPS")
            all_modes_metrics[title].append((m_sa, prec, succ, mean_cle, fps))
            if title.startswith("7."): proposed_frame_logs[seq_name] = tracker.frame_logs

    base_records = all_modes_metrics.get("3. Fixed RGB-T Fusion", [])
    base_ms = 1000.0 / (sum(r[4] for r in base_records) / len(base_records) + 1e-6) if base_records else 6.0

    summary_rows = []
    table_lines = [
        "=" * 94, "خلاصه نتایج نهایی Ablation Study", "=" * 94,
        f"{'Method / Mode':<30} | {'mSA (%)':<8} | {'Prec (%)':<8} | {'Succ (%)':<8} | {'CLE (px)':<8} | {'FPS':<8} | {'Overhead'}", "-" * 94
    ]

    for title, _, _ in ABLATION_MODES:
        records = all_modes_metrics[title]
        if records:
            avg_msa, avg_prec, avg_succ, avg_cle, avg_fps = [sum(r[i] for r in records)/len(records) for i in range(5)]
            curr_ms = 1000.0 / (avg_fps + 1e-6)
            overhead_ms = max(0.0, curr_ms - base_ms) if not title.startswith(("1.", "2.", "3.")) else 0.0
            table_lines.append(f"{title:<30} | {avg_msa:7.2f}% | {avg_prec:7.2f}% | {avg_succ:7.2f}% | {avg_cle:6.1f}px | {avg_fps:5.1f} FPS | +{overhead_ms:4.2f} ms")
            summary_rows.append({"mode_title": title, "mSA": round(avg_msa, 2), "Precision": round(avg_prec, 2), "Success": round(avg_succ, 2), "CLE_px": round(avg_cle, 1), "FPS": round(avg_fps, 1)})

    table_text = "\n".join(table_lines) + "\n" + "=" * 94
    print("\n" + table_text)

    with open(os.path.join(current_archive_dir, "summary_table.txt"), "w", encoding="utf-8") as f: f.write(table_text)
    with open(os.path.join(current_archive_dir, "ablation_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["mode_title", "mSA", "Precision", "Success", "CLE_px", "FPS"])
        writer.writeheader(); writer.writerows(summary_rows)

    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_excel(os.path.join(current_archive_dir, "ablation_results.xlsx"), index=False)
    except: pass

    save_archive_plots(current_archive_dir, summary_rows, proposed_frame_logs, all_modes_raw_data)
    return current_archive_dir
