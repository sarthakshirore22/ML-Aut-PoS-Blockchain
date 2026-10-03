"""
Reproducible Script: Validator Trust Distribution Visualization
================================================================
This script generates the authoritative 'Validator Trust Distribution' 
visualizations for Slide 10 of the final presentation:
"ML-Based Trust-Aware Validator Selection for Blockchain Consensus"

Data Source:
- Raw Training Dataset: validator_behavior_log_TRAINING.csv (100,000 records, 100 validators)
- Authoritative ML Model: trust_classifier.pkl (trained RandomForestClassifier)
- Prediction Output: validator_model_output.csv (100 validators, aggregated P(HONEST))

Ground Truth Separation:
- 80 Honest Validators (true_label = 1)
- 20 Malicious Validators (true_label = 0)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib

def load_or_compute_validator_trust(handoff_dir="ROLE 2 HANDOFF"):
    if not os.path.exists(handoff_dir):
        handoff_dir = "."
    
    model_output_csv = os.path.join(handoff_dir, "validator_model_output.csv")
    training_log_csv = os.path.join(handoff_dir, "validator_behavior_log_TRAINING.csv")
    if not os.path.exists(training_log_csv):
        training_log_csv = "validator_behavior_log_TRAINING.csv"
    
    # 1. Load Ground Truth from training behavior log
    if not os.path.exists(training_log_csv):
        raise FileNotFoundError(f"Cannot find training dataset: {training_log_csv}")
    
    print(f"[1/4] Reading ground-truth labels from {training_log_csv}...")
    df_raw = pd.read_csv(training_log_csv)
    gt_df = df_raw.groupby('validator_id')['true_label'].agg(['first', 'nunique']).reset_index()
    if (gt_df['nunique'] > 1).any():
        raise ValueError("Inconsistent ground-truth labels found within a validator!")
    gt_df = gt_df.rename(columns={'first': 'true_label'})[['validator_id', 'true_label']]
    
    # 2. Load or Compute Model Output
    if os.path.exists(model_output_csv):
        print(f"[2/4] Reading pre-computed model outputs from {model_output_csv}...")
        df_pred = pd.read_csv(model_output_csv)
    else:
        model_pkl = os.path.join(handoff_dir, "trust_classifier.pkl")
        if not os.path.exists(model_pkl):
            model_pkl = os.path.join("role2_final_handoff", "trust_classifier_FINAL.pkl")
        print(f"[2/4] Computing validator trust dynamically using {model_pkl}...")
        model = joblib.load(model_pkl)
        features = ['uptime', 'response_time_ms', 'verification_correct', 'transaction_outcome', 'suspicious_flag']
        X = df_raw[features]
        df_raw['P(HONEST)'] = model.predict_proba(X)[:, 1]
        df_pred = df_raw.groupby('validator_id')['P(HONEST)'].mean().reset_index()
    
    # 3. Merge Ground Truth with Trust Scores
    df_merged = pd.merge(df_pred, gt_df, on='validator_id')
    return df_merged

def generate_visualizations(df_merged, output_dirs=["ROLE 2 HANDOFF", "."]):
    print("[3/4] Generating publication-quality distribution visualizations...")
    
    honest_df = df_merged[df_merged['true_label'] == 1].sort_values('validator_id')
    malicious_df = df_merged[df_merged['true_label'] == 0].sort_values('validator_id')
    
    honest_trust = honest_df['P(HONEST)'].values
    malicious_trust = malicious_df['P(HONEST)'].values
    
    # Statistics
    h_mean, h_median, h_std = np.mean(honest_trust), np.median(honest_trust), np.std(honest_trust)
    h_min, h_max = np.min(honest_trust), np.max(honest_trust)
    
    m_mean, m_median, m_std = np.mean(malicious_trust), np.median(malicious_trust), np.std(malicious_trust)
    m_min, m_max = np.min(malicious_trust), np.max(malicious_trust)
    
    sep_gap = h_min - m_max
    
    # Validator 55 specific data
    v55_data = df_merged[df_merged['validator_id'] == 55]
    v55_trust = v55_data['P(HONEST)'].values[0] if len(v55_data) > 0 else None
    
    print(f"      Honest Validators (N={len(honest_trust)}):")
    print(f"        Mean: {h_mean:.6f}, Median: {h_median:.6f}, Min: {h_min:.6f}, Max: {h_max:.6f}")
    print(f"      Malicious Validators (N={len(malicious_trust)}):")
    print(f"        Mean: {m_mean:.6f}, Median: {m_median:.6f}, Min: {m_min:.6f}, Max: {m_max:.6f}")
    print(f"      Separation Gap: {sep_gap:.6f}")
    if v55_trust is not None:
        print(f"      Validator 55 Trust: {v55_trust:.6f}")
    
    # Styling configuration
    c_honest = '#2e7d32'          # Forest green
    c_honest_fill = '#e8f5e9'     # Very light green
    c_malicious = '#c62828'       # Dark crimson
    c_malicious_fill = '#ffebee'  # Very light red
    c_coldstart = '#ef6c00'       # Amber / Orange
    
    # =========================================================================
    # VISUALIZATION 1: COMPREHENSIVE MULTI-PANEL FIGURE (Presentation Slide 10)
    # =========================================================================
    fig = plt.figure(figsize=(14.0, 8.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.45, wspace=0.28,
                          left=0.08, right=0.96, top=0.78, bottom=0.09)
    
    # --- PANEL A: Full Global Distribution [0.0, 1.0] ---
    ax_global = fig.add_subplot(gs[:, 0])
    
    bp_data = [honest_trust, malicious_trust]
    box_positions = [1, 2]
    
    bp = ax_global.boxplot(
        bp_data, positions=box_positions, widths=0.45, patch_artist=True,
        showmeans=True, meanline=False,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#ffeb3b', markersize=7),
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(color='#333333', linewidth=1.5),
        capprops=dict(color='#333333', linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )
    
    # Style boxes
    bp['boxes'][0].set(facecolor=c_honest_fill, edgecolor=c_honest, linewidth=2.0)
    bp['boxes'][1].set(facecolor=c_malicious_fill, edgecolor=c_malicious, linewidth=2.0)
    
    # Overlay jittered validator scatter points
    np.random.seed(42)
    jitter_h = np.random.normal(1.0, 0.035, size=len(honest_trust))
    jitter_m = np.random.normal(2.0, 0.035, size=len(malicious_trust))
    
    ax_global.scatter(jitter_h, honest_trust, color=c_honest, alpha=0.55, s=30, edgecolors='none', zorder=4)
    ax_global.scatter(jitter_m, malicious_trust, color=c_malicious, alpha=0.65, s=34, edgecolors='none', zorder=4)
    
    # Cold-Start threshold reference line
    ax_global.axhline(0.80, color=c_coldstart, linestyle='--', linewidth=1.8, zorder=3,
                      label='Cold-Start Baseline (0.80)')
    
    # Separation Gap bracket / annotation
    ax_global.annotate(
        f'Bimodal Separation Gap\n$\\Delta = {sep_gap:.4f}$\n(Zero Overlap / Purity 100%)',
        xy=(1.5, 0.48), xytext=(1.5, 0.48),
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#9e9e9e', linewidth=1.2),
        fontsize=9.5, fontweight='bold', color='#212121'
    )
    ax_global.annotate('', xy=(1.5, 0.96), xytext=(1.5, 0.61),
                        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2))
    ax_global.annotate('', xy=(1.5, 0.04), xytext=(1.5, 0.35),
                        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2))
    
    # Global panel labels and limits
    ax_global.set_title('Global Consensus Scale [0.0, 1.0] (All 100 Validators)',
                        fontsize=11.5, fontweight='bold', pad=8)
    ax_global.set_ylabel('Validator Trust Score: P(HONEST)', fontsize=11, fontweight='bold')
    ax_global.set_xticks(box_positions)
    ax_global.set_xticklabels([f'Honest Validators\n(N = {len(honest_trust)})',
                               f'Malicious Validators\n(N = {len(malicious_trust)})'],
                              fontsize=10.5, fontweight='bold')
    ax_global.set_ylim(-0.06, 1.06)
    ax_global.grid(axis='y', linestyle=':', alpha=0.6)
    ax_global.legend(loc='upper right', bbox_to_anchor=(0.98, 0.74), framealpha=0.92, fontsize=9.0)
    
    # Stats callouts on Global panel
    ax_global.text(1, 0.88, f'Mean: {h_mean:.4f}\nMin: {h_min:.4f}', ha='center', fontsize=8.5,
                   bbox=dict(facecolor='#ffffff', edgecolor=c_honest, boxstyle='round,pad=0.3', alpha=0.9))
    ax_global.text(2, 0.12, f'Mean: {m_mean:.4f}\nMax: {m_max:.4f}', ha='center', fontsize=8.5,
                   bbox=dict(facecolor='#ffffff', edgecolor=c_malicious, boxstyle='round,pad=0.3', alpha=0.9))
    
    # --- PANEL B: Magnified Granular View - Honest Group ---
    ax_h = fig.add_subplot(gs[0, 1])
    bph = ax_h.boxplot(
        honest_trust, orientation='horizontal', widths=0.45, patch_artist=True,
        showmeans=True, meanline=False,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#ffeb3b', markersize=6),
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(color=c_honest, linewidth=1.5),
        capprops=dict(color=c_honest, linewidth=1.5)
    )
    bph['boxes'][0].set(facecolor=c_honest_fill, edgecolor=c_honest, linewidth=1.8)
    
    # Jitter on Honest zoom
    jitter_hy = np.random.normal(1.0, 0.08, size=len(honest_trust))
    ax_h.scatter(honest_trust, jitter_hy, color=c_honest, alpha=0.55, s=24, zorder=4)
    
    ax_h.set_title(f'Honest Validators Detail: N=80 (Mean: {h_mean:.6f})',
                   fontsize=11, fontweight='bold', color=c_honest, pad=8)
    ax_h.set_xlabel('Trust Score: P(HONEST)', fontsize=9.5)
    ax_h.set_yticks([])
    ax_h.set_xlim(0.9970, 1.0001)
    ax_h.set_ylim(0.55, 1.60)
    ax_h.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Quartiles annotation
    ax_h.text(0.9971, 1.35, f'IQR: [{np.percentile(honest_trust, 25):.5f}, {np.percentile(honest_trust, 75):.5f}]\nMedian: {h_median:.5f}',
              fontsize=8.5, bbox=dict(facecolor='#ffffff', edgecolor=c_honest, boxstyle='round,pad=0.25', alpha=0.9))
    
    # --- PANEL C: Magnified Granular View - Malicious Group ---
    ax_m = fig.add_subplot(gs[1, 1])
    bpm = ax_m.boxplot(
        malicious_trust, orientation='horizontal', widths=0.45, patch_artist=True,
        showmeans=True, meanline=False,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#ffeb3b', markersize=6),
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(color=c_malicious, linewidth=1.5),
        capprops=dict(color=c_malicious, linewidth=1.5)
    )
    bpm['boxes'][0].set(facecolor=c_malicious_fill, edgecolor=c_malicious, linewidth=1.8)
    
    # Jitter on Malicious zoom
    jitter_my = np.random.normal(1.0, 0.08, size=len(malicious_trust))
    ax_m.scatter(malicious_trust, jitter_my, color=c_malicious, alpha=0.65, s=28, zorder=4)
    
    # Highlight Validator 55
    if v55_trust is not None:
        ax_m.scatter([v55_trust], [1.0], color='#ff9800', s=95, edgecolors='black', linewidth=1.5, zorder=6,
                     label='Validator 55')
        ax_m.annotate('Validator 55\n(35.7% denials: 0.00234)',
                      xy=(v55_trust, 1.0), xytext=(v55_trust + 0.00035, 1.30),
                      arrowprops=dict(arrowstyle='->', color='#d84315', lw=1.2),
                      fontsize=8.5, fontweight='bold', color='#bf360c',
                      bbox=dict(facecolor='#fff3e0', edgecolor='#ff9800', boxstyle='round,pad=0.25', alpha=0.95))
    
    ax_m.set_title(f'Malicious Validators Detail: N=20 (Mean: {m_mean:.6f})',
                   fontsize=11, fontweight='bold', color=c_malicious, pad=8)
    ax_m.set_xlabel('Trust Score: P(HONEST)', fontsize=9.5)
    ax_m.set_yticks([])
    ax_m.set_xlim(0.0000, 0.0042)
    ax_m.set_ylim(0.55, 1.65)
    ax_m.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Overall Title with ample top margin
    fig.suptitle('Role 2: Validator Trust Distribution by Ground-Truth Category\n'
                 'Authoritative ML Output: RandomForest P(HONEST) Aggregated over 1,000 Records/Node',
                 fontsize=13, fontweight='bold', y=0.93)
    
    # =========================================================================
    # VISUALIZATION 2: CLEAN SINGLE BOX PLOT (Ready for 1-Box Slide Insert)
    # =========================================================================
    fig_single, ax_s = plt.subplots(figsize=(9.0, 6.0), dpi=300)
    
    bp_s = ax_s.boxplot(
        bp_data, positions=box_positions, widths=0.42, patch_artist=True,
        showmeans=True, meanline=False,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#ffeb3b', markersize=8),
        medianprops=dict(color='black', linewidth=2.2),
        whiskerprops=dict(color='#212121', linewidth=1.6),
        capprops=dict(color='#212121', linewidth=1.6)
    )
    bp_s['boxes'][0].set(facecolor=c_honest_fill, edgecolor=c_honest, linewidth=2.2)
    bp_s['boxes'][1].set(facecolor=c_malicious_fill, edgecolor=c_malicious, linewidth=2.2)
    
    ax_s.scatter(jitter_h, honest_trust, color=c_honest, alpha=0.55, s=36, edgecolors='none', zorder=4)
    ax_s.scatter(jitter_m, malicious_trust, color=c_malicious, alpha=0.65, s=40, edgecolors='none', zorder=4)
    
    # Cold-Start threshold reference line
    ax_s.axhline(0.80, color=c_coldstart, linestyle='--', linewidth=2.0, zorder=3,
                 label='Cold-Start Baseline (Trust = 0.80)')
    
    # Legend placed in upper right area cleanly
    legend_elements = [
        Patch(facecolor=c_honest_fill, edgecolor=c_honest, label=f'Honest (N={len(honest_trust)})'),
        Patch(facecolor=c_malicious_fill, edgecolor=c_malicious, label=f'Malicious (N={len(malicious_trust)})'),
        plt.Line2D([0], [0], color=c_coldstart, linestyle='--', linewidth=2.0, label='Cold-Start Baseline (0.80)'),
        plt.Line2D([0], [0], marker='D', color='w', markeredgecolor='black', markerfacecolor='#ffeb3b',
                   markersize=8, label='Group Mean')
    ]
    ax_s.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.76), framealpha=0.94, fontsize=9.5)
    
    ax_s.set_title('Validator Trust Distribution\nComparing Honest vs. Malicious Nodes (N=100)',
                   fontsize=13.5, fontweight='bold', pad=12)
    ax_s.set_ylabel('Calculated Trust Score: P(HONEST)', fontsize=12, fontweight='bold')
    ax_s.set_xticks(box_positions)
    ax_s.set_xticklabels([f'Honest Validators\n(N = {len(honest_trust)})\nMean = {h_mean:.4f}',
                          f'Malicious Validators\n(N = {len(malicious_trust)})\nMean = {m_mean:.4f}'],
                         fontsize=11, fontweight='bold')
    ax_s.set_ylim(-0.06, 1.08)
    ax_s.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Annotation on separation with arrows
    ax_s.annotate(
        f'Bimodal Separation Gap: {sep_gap:.4f}\n(Zero Overlap / Perfect Separation)',
        xy=(1.5, 0.50), xytext=(1.40, 0.50), ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#fafafa', edgecolor='#757575', linewidth=1.2),
        fontsize=10.0, fontweight='bold', color='#212121'
    )
    ax_s.annotate('', xy=(1.40, 0.96), xytext=(1.40, 0.62),
                  arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2))
    ax_s.annotate('', xy=(1.40, 0.04), xytext=(1.40, 0.38),
                  arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2))
    
    fig_single.tight_layout()
    
    # Save files to specified output directories
    saved_files = []
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        path_multi = os.path.join(out_dir, "validator_trust_distribution.png")
        path_single = os.path.join(out_dir, "validator_trust_distribution_single.png")
        
        fig.savefig(path_multi, dpi=300)
        fig_single.savefig(path_single, dpi=300)
        saved_files.extend([path_multi, path_single])
    
    plt.close(fig)
    plt.close(fig_single)
    
    print("[4/4] Successfully saved visualization files:")
    for f in saved_files:
        print(f"      - {os.path.abspath(f)} ({os.path.getsize(f):,} bytes)")
    
    return saved_files

if __name__ == '__main__':
    df = load_or_compute_validator_trust()
    generate_visualizations(df)
