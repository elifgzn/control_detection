"""
5_TFR_permutation.py
====================
Alpha-Band ROI Time-Frequency Permutation & Stat Maps
-----------------------------------------------------

PURPOSE:
    Loads the condition-averaged TFR data, runs a 2D cluster-based permutation test
    (Time × Frequency), generates statistical contour maps, and saves results to a text file.
    
    This functionally replicates the FieldTrip steps from A2_5_permuatation_TFmaps_feedbacktheta.m:
        - Constructing ANOVA contrasts (Main Effects and Interactions)
        - ft_freqstatistics (method='montecarlo', correctm='cluster')
        - Plotting difference heatmap with contour overlays
        - Plotting test statistic heatmap with contour overlays
        - Calculating Effect Size (Cohen's d) for significant clusters
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked"
os.makedirs(figures_path, exist_ok=True)

plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25]

# Permutation test parameters (matches FieldTrip cfg.numrandomization=1000)
N_PERMUTATIONS = 1000
TAIL = 0           # two-sided t-test
SEED = 2025

Z_LIMITS_DIFF = [-1.5, 1.5]  # Z-limits for difference heatmap
Z_LIMITS_STAT = [0, 25]      # FieldTrip scripts occasionally use [0 25] for F-stats/T-stats

# Time and Frequency ranges to test (like FieldTrip cfgPermut.latency / .frequency)
TEST_TIME = (0.0, 3.0)
TEST_FREQ = (2.0, 20.0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & PREPARE CONTRASTS FOR STATS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading per-subject condition averages...")

conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']

# We store per-subject data in a dictionary mapping subject ID to their condition data,
# to ensure we only include subjects with complete data (all 4 conditions) for the paired contrasts.
subject_roi_data = {f"{sub:04d}": {} for sub in plist}

times, freqs, ch_names, roi_channels = None, None, None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    
    if times is None:
        times_all = saved['times']
        freqs_all = saved['freqs']
        ch_names = saved['ch_names'].tolist()
        roi_channels = saved['roi_channels'].tolist()
        roi_idx = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]
        
        # Apply time and frequency restrictions for the permutation test
        time_mask = (times_all >= TEST_TIME[0]) & (times_all <= TEST_TIME[1])
        times = times_all[time_mask]
        
        freq_mask = (freqs_all >= TEST_FREQ[0]) & (freqs_all <= TEST_FREQ[1])
        freqs = freqs_all[freq_mask]
        
    for cond in conditions:
        if cond in saved:
            # Average across ROI channels (FieldTrip cfgPermut.avgoverchan = 'yes')
            # Shape before: (n_channels, n_freqs_all, n_times_all)
            roi_avg = np.nanmean(saved[cond][roi_idx, :, :], axis=0) # (n_freqs_all, n_times_all)
            # Restrict time and frequency windows
            subject_roi_data[sub_id][cond] = roi_avg[np.ix_(freq_mask, time_mask)]

# Find subjects that have all 4 conditions for the paired ANOVA contrasts
valid_subs = [sub for sub in subject_roi_data if len(subject_roi_data[sub]) == 4]
print(f"Found {len(valid_subs)} subjects with all 4 conditions.")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run paired contrasts. Exiting.")
    exit()

# Extract paired subject arrays
low_rec = np.array([subject_roi_data[sub]['low_recalled'] for sub in valid_subs])
low_not = np.array([subject_roi_data[sub]['low_not_recalled'] for sub in valid_subs])
high_rec = np.array([subject_roi_data[sub]['high_recalled'] for sub in valid_subs])
high_not = np.array([subject_roi_data[sub]['high_not_recalled'] for sub in valid_subs])

contrasts = {}

# 1. Main Effect of Memory: (Low Recalled + High Recalled)/2 - (Low Not Recalled + High Not Recalled)/2
# This mirrors the FieldTrip 'main_valence' variable creation.
main_rec = (low_rec + high_rec) / 2
main_not = (low_not + high_not) / 2
contrasts['Main_Effect_Memory'] = main_rec - main_not

# 2. Interaction (Memory x Control): (Low Recalled - Low Not Recalled) - (High Recalled - High Not Recalled)
# This mirrors the FieldTrip 'interaction' variable differences.
low_diff = low_rec - low_not
high_diff = high_rec - high_not
contrasts['Interaction_Memory_x_Control'] = low_diff - high_diff

# 3. OLD ANALYSIS: Simple Effect of Memory in Low Control
contrasts['OLD_ANALYSIS__Low_Control_Memory_Effect'] = low_diff

# 4. OLD ANALYSIS: Simple Effect of Memory in High Control
contrasts['OLD_ANALYSIS__High_Control_Memory_Effect'] = high_diff

descriptions = {
    'Main_Effect_Memory': '(Recalled vs Not Recalled across all conditions)',
    'Interaction_Memory_x_Control': '(Difference in Memory Effect between Low and High Control)',
    'OLD_ANALYSIS__Low_Control_Memory_Effect': '(Recalled vs Not Recalled WITHIN Low Control only)',
    'OLD_ANALYSIS__High_Control_Memory_Effect': '(Recalled vs Not Recalled WITHIN High Control only)'
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. RUN CLUSTER PERMUTATION TEST (ft_freqstatistics)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning 2D (Time x Freq) cluster permutation tests ({N_PERMUTATIONS} permutations)...")

report_lines = []
report_lines.append("TFR PERMUTATION TEST RESULTS")
report_lines.append("=" * 70)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s")
report_lines.append("=" * 70)

for comp_name, X_diff in contrasts.items():
    X_diff_clean = np.nan_to_num(X_diff, nan=0.0)
    
    # Run 2D Permutation Test
    T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
        X_diff_clean,
        n_permutations=N_PERMUTATIONS,
        tail=TAIL,
        n_jobs=-1,
        seed=SEED
    )
    
    # Compute Grand Average Difference
    ga_diff = np.nanmean(X_diff_clean, axis=0)
    
    sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
    n_sig = len(sig_clusters)
    
    desc = descriptions.get(comp_name, '')
    msg_header = f"\n{comp_name.replace('_', ' ').upper()} {desc}\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 70)
    
    # Calculate Cohen's d (Effect Size) for each cluster
    # FieldTrip Step: calculate average Cohen's d for time-frequency points with significant differences
    for i, (c, p) in enumerate(zip(clusters, cluster_p)):
        if p < 0.05:
            # Create boolean mask for the cluster
            mask = np.zeros_like(ga_diff, dtype=bool)
            mask[c] = True
            
            # Extract time and freq ranges
            freq_inds, time_inds = c
            f_start, f_end = freqs[freq_inds.min()], freqs[freq_inds.max()]
            t_start, t_end = times[time_inds.min()], times[time_inds.max()]
            
            # Calculate Cohen's d
            # Mean difference / std of difference across subjects, then average over the cluster points
            mean_diff = np.nanmean(X_diff_clean, axis=0)
            std_diff = np.nanstd(X_diff_clean, axis=0, ddof=1)
            cohens_d_map = mean_diff / np.where(std_diff == 0, 1e-10, std_diff)
            cluster_cohens_d = np.mean(np.abs(cohens_d_map[mask]))
            
            msg = (f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, "
                   f"p={p:.4f} ★ | Cohen's d: {cluster_cohens_d:.2f}")
            print(msg)
            report_lines.append(msg)
        else:
            freq_inds, time_inds = c
            f_start, f_end = freqs[freq_inds.min()], freqs[freq_inds.max()]
            t_start, t_end = times[time_inds.min()], times[time_inds.max()]
            msg = f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, p={p:.4f}"
            print(msg)
            report_lines.append(msg)
            
    # ══════════════════════════════════════════════════════════════════════════════
    # 3. PLOT DIFFERENCE HEATMAP WITH CONTOURS
    # ══════════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        ga_diff,
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='RdBu_r', vmin=Z_LIMITS_DIFF[0], vmax=Z_LIMITS_DIFF[1]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power Difference (dB)')
    
    # Overlay contours
    for c, p in zip(clusters, cluster_p):
        if p < 0.05:
            mask = np.zeros_like(ga_diff, dtype=bool)
            mask[c] = True
            ax.contour(times, freqs, mask, levels=[0.5], colors='black', linewidths=2)
            
    ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax.set_title(f"{comp_name.replace('_', ' ').upper()}", fontsize=16, fontname='Times New Roman')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'01_tf_permut_{comp_name}_diff.png'), dpi=300)
    plt.close(fig)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 4. PLOT TEST STATISTIC HEATMAP WITH CONTOURS
    # ══════════════════════════════════════════════════════════════════════════════
    # Use absolute T-values for the stat map (similar to F-values)
    stat_map = np.abs(T_obs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        stat_map,
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='Reds', vmin=Z_LIMITS_STAT[0], vmax=Z_LIMITS_STAT[1]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Test Statistic (T-value)')
    
    # Overlay contours
    for c, p in zip(clusters, cluster_p):
        if p < 0.05:
            mask = np.zeros_like(ga_diff, dtype=bool)
            mask[c] = True
            ax.contour(times, freqs, mask, levels=[0.5], colors='black', linewidths=2)
            
    ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax.set_title(f"{comp_name.replace('_', ' ').upper()} (Stat Map)", fontsize=16, fontname='Times New Roman')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'01_stat_permut_{comp_name}.png'), dpi=300)
    plt.close(fig)

# Save text report to file
report_file = os.path.join(figures_path, 'TFR_permutation_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
    print(f"\n* Statistical report saved to {report_file}")
