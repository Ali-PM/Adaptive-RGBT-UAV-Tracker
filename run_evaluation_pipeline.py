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
    extract_sequence_attributes,
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
    iou_arr = np.asarray(ious, dtype=np.float32)
    return thresholds, [float(np.mean(iou_arr >= t)) * 100.0 for t in thresholds]

def calc_precision_curve(cles):
    thresholds = np.arange(0, 51, 1)
    cle_arr = np.asarray(cles, dtype=np.float32)
    return thresholds, [float(np.mean(cle_arr <= t)) * 100.0 for t in thresholds]

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
        ax.bar(x + width, succ_vals, width, label='Success@0.5 (%)', color='#d95f02')
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
            ax_succ.plot(thr_s, succ, label=f"[{np.mean(succ):.2f}%] {mode}", color=c, linestyle=ls, linewidth=2)
            thr_p, prec = calc_precision_curve(all_modes_raw_data[mode]['cles'])
            p20 = prec[20] if len(prec) > 20 else 0
            ax_prec.plot(thr_p, prec, label=f"[{p20:.2f}%] {mode}", color=c, linestyle=ls, linewidth=2)

        ax_succ.set_xlabel('Overlap Threshold'); ax_succ.set_ylabel('Success Rate (%)')
        ax_succ.set_title('Success Plots of OPE on Infrared Set'); ax_succ.set_xlim(0, 1); ax_succ.set_ylim(0, 100)
        ax_succ.grid(True, linestyle='--', alpha=0.6); ax_succ.legend(loc='lower left', fontsize=8.5)
        fig_succ.tight_layout(); fig_succ.savefig(os.path.join(archive_dir, 'ope_success_plot.png'), dpi=200); plt.close(fig_succ)

        ax_prec.set_xlabel('Location Error Threshold (Pixels)'); ax_prec.set_ylabel('Precision (%)')
        ax_prec.set_title('Precision Plots of OPE on Infrared Set'); ax_prec.set_xlim(0, 50); ax_prec.set_ylim(0, 100)
        ax_prec.grid(True, linestyle='--', alpha=0.6); ax_prec.legend(loc='lower right', fontsize=8.5)
        fig_prec.tight_layout(); fig_prec.savefig(os.path.join(archive_dir, 'ope_precision_plot.png'), dpi=200); plt.close(fig_prec)

        fig_vsucc, ax_vsucc = plt.subplots(figsize=(8, 6))
        fig_vprec, ax_vprec = plt.subplots(figsize=(8, 6))
        for idx, mode in enumerate(all_modes_raw_data.keys()):
            if not all_modes_raw_data[mode].get('vis_ious'): continue
            c, ls = colors[idx % len(colors)], linestyles[idx % len(linestyles)]
            thr_s, succ = calc_success_curve(all_modes_raw_data[mode]['vis_ious'])
            ax_vsucc.plot(thr_s, succ, label=f"[{np.mean(succ):.2f}%] {mode}", color=c, linestyle=ls, linewidth=2)
            thr_p, prec = calc_precision_curve(all_modes_raw_data[mode]['vis_cles'])
            p20 = prec[20] if len(prec) > 20 else 0
            ax_vprec.plot(thr_p, prec, label=f"[{p20:.2f}%] {mode}", color=c, linestyle=ls, linewidth=2)

        ax_vsucc.set_xlabel('Overlap Threshold'); ax_vsucc.set_ylabel('Success Rate (%)')
        ax_vsucc.set_title('Success Plots of OPE on Visible Set'); ax_vsucc.set_xlim(0, 1); ax_vsucc.set_ylim(0, 100)
        ax_vsucc.grid(True, linestyle='--', alpha=0.6); ax_vsucc.legend(loc='lower left', fontsize=8.5)
        fig_vsucc.tight_layout(); fig_vsucc.savefig(os.path.join(archive_dir, 'ope_visible_success_plot.png'), dpi=200); plt.close(fig_vsucc)

        ax_vprec.set_xlabel('Location Error Threshold (Pixels)'); ax_vprec.set_ylabel('Precision (%)')
        ax_vprec.set_title('Precision Plots of OPE on Visible Set'); ax_vprec.set_xlim(0, 50); ax_vprec.set_ylim(0, 100)
        ax_vprec.grid(True, linestyle='--', alpha=0.6); ax_vprec.legend(loc='lower right', fontsize=8.5)
        fig_vprec.tight_layout(); fig_vprec.savefig(os.path.join(archive_dir, 'ope_visible_precision_plot.png'), dpi=200); plt.close(fig_vprec)

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

    print("=" * 108)
    print(f"شروع ارزیابی جامع ردیاب تطبیقی RGB-T روی پردازنده: [{device_name}]")
    print(f"[*] تعداد ویدیوهای انتخاب شده برای این تست: {len(seq_folders)} ویدیو")
    print("=" * 108)
    warmup_gpu(model_path)

    all_modes_metrics = {title: [] for title, _, _ in ABLATION_MODES}
    all_modes_seq_dict = {title: {} for title, _, _ in ABLATION_MODES}
    all_modes_raw_data = {title: {'ious': [], 'cles': [], 'vis_ious': [], 'vis_cles': []} for title, _, _ in ABLATION_MODES}
    seq_attributes_map = {}
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

        seq_attrs = extract_sequence_attributes(test_seq, exist_ir, gt_rects_ir)
        seq_attributes_map[seq_name] = seq_attrs

        for title, mode, use_kf in ABLATION_MODES:
            print(f"  -> در حال اجرای حالت: {title} ...")
            tracker = TrackerAdaptiveSiamFC_RGBT(net_path=model_path, fusion_mode=mode, use_kalman=use_kf, num_cpu_threads=4)
            pred_boxes_ir, pred_boxes_rgb, fps = run_video_pair_mp4(
                tracker, rgb_mp4, ir_mp4, gt_rects_ir=gt_rects_ir, exist_flags_ir=exist_ir,
                gt_rects_rgb=gt_rects_rgb, exist_flags_rgb=exist_rgb, return_rgb_boxes=True
            )
            
            seq_ious, seq_cles = [], []
            for t in range(min(len(pred_boxes_ir), len(gt_rects_ir), len(exist_ir))):
                if exist_ir[t] > 0 and len(gt_rects_ir[t]) == 4 and gt_rects_ir[t][2] > 0:
                    pred, gt = pred_boxes_ir[t], gt_rects_ir[t]
                    p_hat = 1 if (len(pred) == 4 and pred[2] > 0 and pred[3] > 0) else 0
                    seq_ious.append(compute_iou(pred, gt) if p_hat > 0 else 0.0)
                    seq_cles.append(compute_center_error(pred, gt) if p_hat > 0 else 1000.0)
            
            seq_vis_ious, seq_vis_cles = [], []
            for t in range(min(len(pred_boxes_rgb), len(gt_rects_rgb), len(exist_rgb))):
                if exist_rgb[t] > 0 and len(gt_rects_rgb[t]) == 4 and gt_rects_rgb[t][2] > 0:
                    pred, gt = pred_boxes_rgb[t], gt_rects_rgb[t]
                    p_hat = 1 if (len(pred) == 4 and pred[2] > 0 and pred[3] > 0) else 0
                    seq_vis_ious.append(compute_iou(pred, gt) if p_hat > 0 else 0.0)
                    seq_vis_cles.append(compute_center_error(pred, gt) if p_hat > 0 else 1000.0)

            all_modes_raw_data[title]['ious'].extend(seq_ious)
            all_modes_raw_data[title]['cles'].extend(seq_cles)
            all_modes_raw_data[title]['vis_ious'].extend(seq_vis_ious)
            all_modes_raw_data[title]['vis_cles'].extend(seq_vis_cles)

            m_sa, prec, succ50, succ_auc, mean_cle = evaluate_sequence_metrics(
                pred_boxes_ir, gt_rects_ir, exist_ir, return_cle=True, return_auc=True
            )
            m_sa_vis, prec_vis, succ50_vis, succ_auc_vis, _ = evaluate_sequence_metrics(
                pred_boxes_rgb, gt_rects_rgb, exist_rgb, return_cle=True, return_auc=True
            )

            res_str = f"mSA = {m_sa:.2f}% | Prec = {prec:.2f}% | Succ@0.5 = {succ50:.2f}% | Succ_AUC = {succ_auc:.2f}% | Vis_mSA = {m_sa_vis:.2f}% | CLE = {mean_cle:.1f}px | Speed = {fps:.1f} FPS"
            print(f"     نتیجه: {res_str}")
            all_modes_metrics[title].append((m_sa, prec, succ50, succ_auc, mean_cle, fps, m_sa_vis, prec_vis, succ_auc_vis))
            all_modes_seq_dict[title][seq_name] = m_sa
            if title.startswith("7."): proposed_frame_logs[seq_name] = tracker.frame_logs

    tc_seqs = [s for s, att in seq_attributes_map.items() if 'TC' in att]
    if tc_seqs:
        base_ir_scores = sorted([(s, all_modes_seq_dict["2. Thermal Only"].get(s, 0.0)) for s in tc_seqs], key=lambda x: x[1], reverse=True)
        n_tc = len(base_ir_scores)
        c1, c2 = max(1, n_tc // 3), max(2, (2 * n_tc) // 3)
        for idx, (s, _) in enumerate(base_ir_scores):
            if idx < c1: seq_attributes_map[s].add('TCeasy')
            elif idx < c2: seq_attributes_map[s].add('TCmed')
            else: seq_attributes_map[s].add('TChard')

    base_records = all_modes_metrics.get("3. Fixed RGB-T Fusion", [])
    base_ms = 1000.0 / (sum(r[5] for r in base_records) / len(base_records) + 1e-6) if base_records else 6.0

    summary_rows = []
    table_lines = [
        "=" * 116,
        "جدول ۱: خلاصه نتایج نهایی Ablation Study (شامل معیارهای استاندارد OPE و هر دو کانال Infrared / Visible)",
        "=" * 116,
        f"{'Method / Mode':<30} | {'mSA (%)':<8} | {'Prec (%)':<8} | {'Succ50(%)':<9} | {'SuccAUC(%)':<10} | {'Vis_mSA(%)':<10} | {'CLE (px)':<8} | {'FPS':<8} | {'Overhead'}",
        "-" * 116
    ]

    for title, _, _ in ABLATION_MODES:
        records = all_modes_metrics[title]
        if records:
            avg_msa, avg_prec, avg_succ50, avg_auc, avg_cle, avg_fps, avg_vis_msa, avg_vis_prec, avg_vis_auc = [
                sum(r[i] for r in records)/len(records) for i in range(9)
            ]
            curr_ms = 1000.0 / (avg_fps + 1e-6)
            overhead_ms = max(0.0, curr_ms - base_ms) if not title.startswith(("1.", "2.", "3.")) else 0.0
            row_str = f"{title:<30} | {avg_msa:7.2f}% | {avg_prec:7.2f}% | {avg_succ50:8.2f}% | {avg_auc:9.2f}% | {avg_vis_msa:9.2f}% | {avg_cle:6.1f}px | {avg_fps:5.1f} FPS | +{overhead_ms:4.2f} ms"
            table_lines.append(row_str)
            summary_rows.append({
                "mode_title": title, "mSA": round(avg_msa, 2), "Precision": round(avg_prec, 2),
                "Success": round(avg_succ50, 2), "Success_AUC": round(avg_auc, 2),
                "Visible_mSA": round(avg_vis_msa, 2), "Visible_Precision": round(avg_vis_prec, 2),
                "Visible_Success_AUC": round(avg_vis_auc, 2),
                "CLE_px": round(avg_cle, 1), "FPS": round(avg_fps, 1)
            })

    table_lines.append("=" * 116)

    attr_cols = ['OV', 'OC', 'FM', 'SV', 'LI', 'TCeasy', 'TCmed', 'TChard', 'TC', 'LR']
    attr_headers = ['OV', 'OC', 'FM', 'SV', 'LI', 'TCeasy', 'TCmed', 'TChard', 'TCall', 'LR', 'IR_All', 'Vis_All']
    
    attr_table_lines = [
        "\n" + "=" * 126,
        "جدول ۲: ارزیابی تفکیکی بر اساس ویژگی‌های چالشی (Attribute-Based mSA %) - منطبق بر جداول III، IV و V مقاله IEEE Anti-UAV",
        "=" * 126,
        f"{'Method / Mode':<28} | " + " | ".join([f"{h:<6}" for h in attr_headers]),
        "-" * 126
    ]

    attr_csv_rows = []
    for idx, (title, _, _) in enumerate(ABLATION_MODES):
        seq_scores = all_modes_seq_dict[title]
        if not seq_scores: continue
        row_vals = {}
        val_strs = []
        for ac, ah in zip(attr_cols, attr_headers[:10]):
            matching_seqs = [s for s in seq_scores.keys() if ac in seq_attributes_map.get(s, set())]
            score = sum(seq_scores[s] for s in matching_seqs) / len(matching_seqs) if matching_seqs else 0.0
            row_vals[ah] = round(score, 2)
            val_strs.append(f"{score:6.2f}")
        
        ir_all = summary_rows[idx]["mSA"]
        vis_all = summary_rows[idx]["Visible_mSA"]
        row_vals["IR_All"] = ir_all
        row_vals["Vis_All"] = vis_all
        val_strs.append(f"{ir_all:6.2f}")
        val_strs.append(f"{vis_all:6.2f}")

        attr_table_lines.append(f"{title:<28} | " + " | ".join(val_strs))
        row_dict = {"mode_title": title}
        row_dict.update(row_vals)
        attr_csv_rows.append(row_dict)

    attr_table_lines.append("=" * 126)

    t7_lines = [
        "\n" + "=" * 94,
        "جدول ۳: مقایسه Precision (%) و Success AUC (%) در دو مدالیته - منطبق بر جدول VII مقاله IEEE",
        "=" * 94,
        f"{'Method / Mode':<30} | {'IR Precision (%)':<16} | {'IR Success AUC (%)':<18} | {'Vis Precision (%)':<17} | {'Vis Success AUC (%)'}",
        "-" * 94
    ]
    for r in summary_rows:
        t7_lines.append(
            f"{r['mode_title']:<30} | {r['Precision']:15.2f}% | {r['Success_AUC']:17.2f}% | {r['Visible_Precision']:16.2f}% | {r['Visible_Success_AUC']:17.2f}%"
        )
    t7_lines.append("=" * 94)

    full_report_text = "\n".join(table_lines) + "\n" + "\n".join(attr_table_lines) + "\n" + "\n".join(t7_lines)
    print("\n" + full_report_text)

    with open(os.path.join(current_archive_dir, "summary_table.txt"), "w", encoding="utf-8") as f:
        f.write(full_report_text)

    with open(os.path.join(current_archive_dir, "ablation_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    if attr_csv_rows:
        with open(os.path.join(current_archive_dir, "attribute_based_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(attr_csv_rows[0].keys()))
            writer.writeheader(); writer.writerows(attr_csv_rows)

    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(current_archive_dir, "ablation_results.xlsx")) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary_Table_VII", index=False)
            if attr_csv_rows:
                pd.DataFrame(attr_csv_rows).to_excel(writer, sheet_name="Attribute_Table_III_IV", index=False)
    except Exception:
        pass

    save_archive_plots(current_archive_dir, summary_rows, proposed_frame_logs, all_modes_raw_data)
    return current_archive_dir
