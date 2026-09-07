"""
5_TFR_permutation.py
====================
Alpha-Band Whole-Scalp Time-Frequency Permutation & Stat Maps
-------------------------------------------------------------

PURPOSE:
    Loads the condition-averaged TFR data (all channels), runs a 3D cluster-based 
    permutation test (Time × Frequency × Channel) on a predefined TEST frequency 
    range (e.g. 2-20 Hz). Generates TFR heatmaps (averaged over significant channels)
    and topographic maps of significant clusters.
    
DEVIATION NOTE (minnbchan):
    MNE does not have a direct equivalent of FieldTrip's `minnbchan=3`. Clusters 
    are formed purely based on adjacency without a minimum neighbor constraint 
    for initial candidate selection. With a reasonable cluster-forming threshold 
    (p < 0.05) and whole-scalp adjacency, isolated single-channel clusters are 
    unlikely to survive permutation correction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency
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

info_epochs = None

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz")
    info_file = os.path.join(input_path, f"CDmem_{sub_id}_info.fif")
    
    if not os.path.exists(data_file):
        continue
        
    if info_epochs is None and os.path.exists(info_file):
        info_epochs = mne.io.read_info(info_file, verbose=False)
        
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
            # We keep all channels now
            subject_data[sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

valid_subs = [sub for sub in subject_data if len(subject_data[sub]) == 4]
print(f"Found {len(valid_subs)} subjects with all 4 conditions.")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run paired contrasts. Exiting.")
    sys.exit(0)

low_rec = np.array([subject_data[sub]['low_recalled'] for sub in valid_subs])
low_not = np.array([subject_data[sub]['low_not_recalled'] for sub in valid_subs])
high_rec = np.array([subject_data[sub]['high_recalled'] for sub in valid_subs])
high_not = np.array([subject_data[sub]['high_not_recalled'] for sub in valid_subs])

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
# 2. BUILD ADJACENCY MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding 3D adjacency matrix (channel x frequency)...")

if info_epochs is None:
    print("ERROR: Could not find any _info.fif file to build adjacency matrix. Exiting.")
    sys.exit(1)

# Montage file to ensure exact spatial layout
bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"
if os.path.exists(bvef_path):
    montage = mne.channels.read_custom_montage(bvef_path)
    montage.rename_channels({'REF': 'FCz'})
    info_epochs.set_montage(montage, on_missing='ignore')

ch_adjacency, ch_adj_names = mne.channels.find_ch_adjacency(info_epochs, ch_type='eeg')
n_channels = len(ch_names)
n_freqs_test = len(freqs_test)
n_times = len(times)

# Combined adjacency: Kronecker product of channel adjacency and frequency grid
tfr_adjacency = combine_adjacency(ch_adjacency, n_freqs_test)

# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN CLUSTER PERMUTATION TEST (spatio_temporal_cluster_1samp_test)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning 3D (Time x Freq x Channel) cluster permutation tests ({N_PERMUTATIONS} permutations)...")

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

for comp_name, X_diff in contrasts.items():
    # X_diff shape: (n_subjects, n_channels, n_freqs_plot, n_times)
    X_diff_clean_plot = np.nan_to_num(X_diff, nan=0.0)
    
    # Subset data specifically for the statistical test (frequencies)
    X_diff_clean_test = X_diff_clean_plot[:, :, test_f_inds, :]
    
    # Reshape for spatio_temporal_cluster_1samp_test:
    # Requires (n_observations, n_times, n_vertices)
    # where n_vertices = n_channels * n_freqs
    X_for_test = X_diff_clean_test.transpose(0, 3, 1, 2).reshape(
        len(valid_subs), n_times, n_channels * n_freqs_test
    )
    
    T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
        X_for_test,
        adjacency=tfr_adjacency,
        n_permutations=N_PERMUTATIONS,
        threshold=t_threshold,
        tail=TAIL,
        seed=SEED,
        n_jobs=-1,
        out_type='mask',
        verbose=True
    )
    
    # Reshape T_obs back to 3D: (n_times, n_ch*n_freq) -> (n_ch, n_freq, n_times)
    T_obs_3d = T_obs.reshape(n_times, n_channels, n_freqs_test).transpose(1, 2, 0)
    
    cluster_masks_3d = []
    for clust_mask in clusters:
        mask_3d = clust_mask.reshape(n_times, n_channels, n_freqs_test).transpose(1, 2, 0)
        cluster_masks_3d.append(mask_3d)
        
    sig_clusters = [i for i, p in enumerate(cluster_p) if p < 0.05]
    n_sig = len(sig_clusters)
    
    desc = descriptions.get(comp_name, '')
    msg_header = f"\n{comp_name.replace('_', ' ').upper()} {desc}\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 70)
    
    # Combined mask for plotting
    combined_mask = np.zeros((n_channels, n_freqs_test, n_times), dtype=bool)
    
    for i, (mask, pval) in enumerate(zip(cluster_masks_3d, cluster_p)):
        ch_in = np.any(mask, axis=(1, 2))
        freq_in = np.any(mask, axis=(0, 2))
        time_in = np.any(mask, axis=(0, 1))
        
        n_ch = ch_in.sum()
        t_start, t_end = times[np.where(time_in)[0][0]], times[np.where(time_in)[0][-1]]
        f_low, f_high = freqs_test[np.where(freq_in)[0][0]], freqs_test[np.where(freq_in)[0][-1]]
        
        if pval < 0.05:
            combined_mask |= mask
            sig_marker = " ★ SIGNIFICANT"
        else:
            sig_marker = ""
            
        mean_t = T_obs_3d[mask].mean()
        ch_list = [ch_names[j] for j in np.where(ch_in)[0]]
        
        msg = (f"  Cluster {i+1}: {t_start:.3f}-{t_end:.3f}s, {f_low:.1f}-{f_high:.1f}Hz, "
               f"Channels: {n_ch}, p={pval:.4f}{sig_marker} | Mean T: {mean_t:.2f}")
        print(msg)
        report_lines.append(msg)

    # ══════════════════════════════════════════════════════════════════════════════
    # 4. PLOTTING (TFR Heatmap & Topoplot)
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"Creating plots for {comp_name}...")
    
    sig_channel_mask = np.any(combined_mask, axis=(1, 2))
    n_sig_ch = sig_channel_mask.sum()
    
    # -- TFR Heatmap (Averaged over significant channels) --
    if n_sig_ch > 0:
        T_avg = T_obs_3d[sig_channel_mask].mean(axis=0)
        mask_tf = np.any(combined_mask, axis=0) # collapse over channels
    else:
        T_avg = T_obs_3d.mean(axis=0)
        mask_tf = np.zeros((n_freqs_test, n_times), dtype=bool)
        
    fig_tfr, ax_tfr = plt.subplots(figsize=(10, 6))
    vmin, vmax = -2.5, 2.5
    
    # pcolormesh needs edges
    time_edges = np.concatenate([times - np.diff(times[:2])[0]/2, [times[-1] + np.diff(times[:2])[0]/2]])
    freq_edges = np.concatenate([freqs_test - np.diff(freqs_test[:2])[0]/2, [freqs_test[-1] + np.diff(freqs_test[:2])[0]/2]])
    
    im = ax_tfr.pcolormesh(time_edges, freq_edges, T_avg, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='flat')
    
    if mask_tf.any():
        ax_tfr.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
        
    cb = fig_tfr.colorbar(im, ax=ax_tfr, label='t value')
    ax_tfr.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax_tfr.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax_tfr.set_title(f"{comp_name.replace('_', ' ').upper()}", fontsize=16, fontname='Times New Roman')
    ax_tfr.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig_tfr.savefig(os.path.join(figures_path, f'01_tf_permut_{comp_name}_TFR.png'), dpi=300)
    plt.close(fig_tfr)
    
    # -- Topoplot --
    if mask_tf.any():
        topo_data = np.zeros(n_channels)
        for ch_idx in range(n_channels):
            topo_data[ch_idx] = T_obs_3d[ch_idx][mask_tf].mean()
    else:
        topo_data = T_obs_3d.mean(axis=(1, 2))
        
    fig_topo, ax_topo = plt.subplots(figsize=(6, 6))
    im_topo, _ = mne.viz.plot_topomap(
        topo_data, info_epochs,
        axes=ax_topo,
        cmap='RdBu_r',
        vlim=(vmin, vmax),
        show=False,
        contours=6,
        mask=sig_channel_mask if n_sig_ch > 0 else None,
        mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=8) if n_sig_ch > 0 else None
    )
    
    cb_topo = fig_topo.colorbar(im_topo, ax=ax_topo, label='t value', shrink=0.8)
    ax_topo.set_title(f'{comp_name.replace("_", " ")}\n(mean t at sig. TF points)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_topo.savefig(os.path.join(figures_path, f'02_tf_permut_{comp_name}_topo.png'), dpi=300)
    plt.close(fig_topo)

report_file = os.path.join(figures_path, 'TFR_permutation_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Statistical report saved to {report_file}")
