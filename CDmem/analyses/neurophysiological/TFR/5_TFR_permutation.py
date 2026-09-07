"""
5_TFR_permutation.py
====================
Alpha-Band Time-Frequency Permutation & Stat Maps (ROI-based)
-------------------------------------------------------------

PURPOSE:
    Loads the condition-averaged TFR data, averages over a Parieto-Occipital ROI, 
    and runs a 2D cluster-based permutation test (Frequency x Time) on a predefined 
    TEST frequency range (e.g. 2-20 Hz). Generates TFR heatmaps of the t-values.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t as t_dist

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked"
os.makedirs(figures_path, exist_ok=True)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

# Permutation test parameters
N_PERMUTATIONS = 1000
TAIL = 0           # two-sided t-test
SEED = 2025
CLUSTER_ALPHA = 0.05

TEST_TIME = (0.0, 3.0)
TEST_FREQ = (2.0, 20.0)
PLOT_FREQ = (2.0, 40.0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & PREPARE CONTRASTS FOR STATS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading per-subject condition averages (All Items)...")

conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']
subject_data = {f"{sub:04d}": {} for sub in plist}
times, freqs_plot, freqs_test, test_f_inds, ch_names = None, None, None, None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    
    if times is None:
        times_all = saved['times']
        freqs_all = saved['freqs']
        ch_names = saved['ch_names'].tolist()
        
        time_mask = (times_all >= TEST_TIME[0]) & (times_all <= TEST_TIME[1])
        times = times_all[time_mask]
        
        freq_mask_plot = (freqs_all >= PLOT_FREQ[0]) & (freqs_all <= PLOT_FREQ[1])
        freqs_plot = freqs_all[freq_mask_plot]
        
        # Identify indices within the PLOT array that correspond to the TEST range
        test_f_inds = np.where((freqs_plot >= TEST_FREQ[0]) & (freqs_plot <= TEST_FREQ[1]))[0]
        freqs_test = freqs_plot[test_f_inds]
        
    for cond in conditions:
        if cond in saved:
            # Extract data and slice it for plot frequencies and test times
            # Shape: (n_channels, n_freqs_plot, n_times)
            subject_data[sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

valid_subs = [sub for sub in subject_data if len(subject_data[sub]) == 4]
print(f"Found {len(valid_subs)} subjects with all 4 conditions.")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run paired contrasts. Exiting.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACT ROI DATA
# ══════════════════════════════════════════════════════════════════════════════
roi_channels = [ch for ch in ch_names if ch.startswith('P') or ch.startswith('O')]
print(f"\nAveraging over ROI channels ({len(roi_channels)}): {roi_channels}\n")
roi_idx = [ch_names.index(ch) for ch in roi_channels]

# Shape of resulting arrays: (n_subjects, n_freqs_plot, n_times)
low_rec = np.array([subject_data[sub]['low_recalled'][roi_idx, :, :].mean(axis=0) for sub in valid_subs])
low_not = np.array([subject_data[sub]['low_not_recalled'][roi_idx, :, :].mean(axis=0) for sub in valid_subs])
high_rec = np.array([subject_data[sub]['high_recalled'][roi_idx, :, :].mean(axis=0) for sub in valid_subs])
high_not = np.array([subject_data[sub]['high_not_recalled'][roi_idx, :, :].mean(axis=0) for sub in valid_subs])

contrasts = {}

# 1. Main Effect of Memory
main_rec = (low_rec + high_rec) / 2
main_not = (low_not + high_not) / 2
contrasts['Main_Effect_Memory'] = main_rec - main_not

# 2. Interaction Memory x Control
low_diff = low_rec - low_not
high_diff = high_rec - high_not
contrasts['Interaction_Memory_x_Control'] = low_diff - high_diff

descriptions = {
    'Main_Effect_Memory': '(Recalled vs Not Recalled, collapsed across control conditions)',
    'Interaction_Memory_x_Control': '(Difference in Memory Effect between Low and High Control)'
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN CLUSTER PERMUTATION TEST (permutation_cluster_1samp_test)
# ══════════════════════════════════════════════════════════════════════════════
print(f"Running 2D (Freq x Time) cluster permutation tests ({N_PERMUTATIONS} permutations)...")

df = len(valid_subs) - 1
t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)
print(f"Cluster-forming threshold: t = ±{t_threshold:.3f} (p < {CLUSTER_ALPHA})")

report_lines = []
report_lines.append("TFR PERMUTATION TEST RESULTS")
report_lines.append("=" * 70)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s")
report_lines.append(f"Test Frequency window: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz")
report_lines.append(f"Cluster threshold: t = ±{t_threshold:.3f}")
report_lines.append("=" * 70)

for comp_name, X_diff_plot in contrasts.items():
    # Replace NaNs with 0 to prevent issues with permutation test
    X_diff_plot = np.nan_to_num(X_diff_plot, nan=0.0)
    
    # Extract only the test frequencies for statistics
    X_diff_test = X_diff_plot[:, test_f_inds, :]
    
    # Run test
    T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
        X_diff_test,
        n_permutations=N_PERMUTATIONS,
        threshold=t_threshold,
        tail=TAIL,
        seed=SEED,
        n_jobs=-1,
        out_type='mask',
        verbose=True
    )
    
    sig_clusters = [i for i, p in enumerate(cluster_p) if p < 0.05]
    n_sig = len(sig_clusters)
    
    desc = descriptions.get(comp_name, '')
    msg_header = f"\n{comp_name.replace('_', ' ').upper()} {desc}\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 70)
    
    # Combined mask for plotting
    mask_tf = np.zeros(T_obs.shape, dtype=bool)
    
    for i, (mask, pval) in enumerate(zip(clusters, cluster_p)):
        freq_in = np.any(mask, axis=1)
        time_in = np.any(mask, axis=0)
        
        t_start, t_end = times[np.where(time_in)[0][0]], times[np.where(time_in)[0][-1]]
        f_low, f_high = freqs_test[np.where(freq_in)[0][0]], freqs_test[np.where(freq_in)[0][-1]]
        
        if pval < 0.05:
            mask_tf |= mask
            sig_marker = " ★ SIGNIFICANT"
        else:
            sig_marker = ""
            
        mean_t = T_obs[mask].mean()
        
        msg = (f"  Cluster {i+1}: {t_start:.3f}-{t_end:.3f}s, {f_low:.1f}-{f_high:.1f}Hz, "
               f"p={pval:.4f}{sig_marker} | Mean T: {mean_t:.2f}")
        print(msg)
        report_lines.append(msg)

    # ══════════════════════════════════════════════════════════════════════════════
    # 4. PLOTTING (TFR Heatmap)
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"Creating plots for {comp_name}...")
    
    fig_tfr, ax_tfr = plt.subplots(figsize=(10, 6))
    vmin, vmax = -2.5, 2.5
    
    # pcolormesh edges
    time_edges = np.concatenate([times - np.diff(times[:2])[0]/2, [times[-1] + np.diff(times[:2])[0]/2]])
    freq_edges = np.concatenate([freqs_test - np.diff(freqs_test[:2])[0]/2, [freqs_test[-1] + np.diff(freqs_test[:2])[0]/2]])
    
    im = ax_tfr.pcolormesh(time_edges, freq_edges, T_obs, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='flat')
    
    if mask_tf.any():
        ax_tfr.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
        
    cb = fig_tfr.colorbar(im, ax=ax_tfr, label='t value')
    ax_tfr.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax_tfr.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax_tfr.set_title(f"{comp_name.replace('_', ' ').upper()}\n(ROI: Parieto-Occipital)", fontsize=16, fontname='Times New Roman')
    ax_tfr.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig_tfr.savefig(os.path.join(figures_path, f'01_tf_permut_{comp_name}_TFR.png'), dpi=300)
    plt.close(fig_tfr)

report_file = os.path.join(figures_path, 'TFR_permutation_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Statistical report saved to {report_file}")
