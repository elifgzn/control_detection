"""
4_TFR_permutation.py
====================
Alpha-Band ROI Time-Frequency Permutation & Heatmaps
----------------------------------------------------

PURPOSE:
    Loads the ROI-averaged TFR data, runs a 2D cluster-based permutation test
    (Time × Frequency), and generates contour-highlighted heatmaps.
    
    This functionally replicates the FieldTrip steps:
        `ft_freqstatistics` (method='montecarlo', correctm='cluster')
        `ft_singleplotTFR` (with contour overlays)
    as seen in the provided MATLAB script `A2_5_permuatation_TFmaps_stimonset.m`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from mne.stats import permutation_cluster_1samp_test
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
input_path     = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_alphabeta_static"
eeg_path       = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_alphabeta_static"
figures_path   = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_alphabeta_static"
os.makedirs(figures_path, exist_ok=True)

# Permutation test parameters (matches FieldTrip cfg.numrandomization=1000)
N_PERMUTATIONS = 1000
TAIL = 0    # two-sided t-test (Recalled vs Not-recalled)
SEED = 2025

plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15,16, 18, 19, 20, 21, 22,23,24]

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
group_recalled = {'low': [], 'high': []}
group_not_recalled = {'low': [], 'high': []}
topo_recalled = {'low': [], 'high': []}
topo_not_recalled = {'low': [], 'high': []}
times = None
freqs = None
ch_names = None
info = None

print("Loading per-subject ROI TFR data and topoplot data...")
for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ROI.npz")
    
    if not os.path.exists(data_file):
        print(f"  Missing {data_file}")
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    data_roi = saved['data_roi']  # (n_epochs, n_freqs, n_times)
    
    if times is None:
        times = saved['times']
        freqs = saved['freqs']
        # Find indices for 0.0 to 3.5 seconds
        time_mask = (times >= 0.0) & (times <= 3.5)
        times = times[time_mask]
        
        # Load info object for topoplots
        epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
        if os.path.exists(epo_file):
            import mne
            info = mne.io.read_info(epo_file, verbose=False)
            if 'ch_names' in saved:
                ch_names = info.ch_names  # Use the info object's channels as standard
    
    if 'topo_data' in saved:
        topo_data_raw = saved['topo_data'] # (n_epochs, n_sub_channels)
        sub_ch_names = saved['ch_names'].tolist() if 'ch_names' in saved else []
        
        # Map subject channels to standard info channels
        if info is not None and len(sub_ch_names) > 0:
            topo_data = np.full((topo_data_raw.shape[0], len(info.ch_names)), np.nan)
            for i, ch in enumerate(info.ch_names):
                if ch in sub_ch_names:
                    idx = sub_ch_names.index(ch)
                    topo_data[:, i] = topo_data_raw[:, idx]
        else:
            topo_data = topo_data_raw
    else:
        print(f"  Warning: topo_data not found in {data_file}. Please re-run 3_TFR_calculation.py")
        topo_data = None
        
    cond_arr = saved['trial_info_condition']
    rec_arr = saved['trial_info_recalled']
        
    data_roi = data_roi[:, :, time_mask]
        
    for condition in ['low', 'high']:
        cond_mask = (cond_arr == condition)
        recalled_mask = (rec_arr == 'yes')
        not_recalled_mask = (rec_arr == 'no')
        
        recalled_idx = cond_mask & recalled_mask
        not_recalled_idx = cond_mask & not_recalled_mask
        
        if recalled_idx.sum() < 2 or not_recalled_idx.sum() < 2:
            continue
            
        # Average across trials for this subject
        avg_recalled = np.nanmean(data_roi[recalled_idx], axis=0)
        avg_not_recalled = np.nanmean(data_roi[not_recalled_idx], axis=0)
        
        group_recalled[condition].append(avg_recalled)
        group_not_recalled[condition].append(avg_not_recalled)
        
        if topo_data is not None:
            avg_topo_rec = np.nanmean(topo_data[recalled_idx], axis=0)
            avg_topo_not_rec = np.nanmean(topo_data[not_recalled_idx], axis=0)
            topo_recalled[condition].append(avg_topo_rec)
            topo_not_recalled[condition].append(avg_topo_not_rec)

n_subs = len(group_recalled.get('low', []))

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════════════
# Equivalent to FieldTrip ft_freqstatistics
print(f"\nRunning 2D (Time x Freq) cluster permutation tests ({N_PERMUTATIONS} permutations)...")

report_lines = []
report_lines.append("TFR ROI ANALYSIS: RECALLED vs NOT-RECALLED")
report_lines.append("=" * 70)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"Number of participants: {n_subs}")
report_lines.append("=" * 70)

cluster_results = {}

for condition in ['low', 'high']:
    recalled_arr = np.array(group_recalled[condition])       # (n_subs, n_freqs, n_times)
    not_recalled_arr = np.array(group_not_recalled[condition])  # (n_subs, n_freqs, n_times)

    if len(recalled_arr) < 2:
        continue

    # Difference: recalled - not_recalled
    X_diff = recalled_arr - not_recalled_arr  # (n_subs, n_freqs, n_times)
    
    # Handle NaNs from masks
    X_diff_clean = np.nan_to_num(X_diff, nan=0.0)
    
    # 2D Permutation test (Time x Freq)
    T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
        X_diff_clean,
        n_permutations=N_PERMUTATIONS,
        tail=TAIL,
        n_jobs=-1,
        seed=SEED
    )
    
    # Identify significant clusters
    sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
    n_sig = len(sig_clusters)
    
    msg_header = f"\n{condition.upper()} CONTROL: {len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 60)
    
    for i, (c, p) in enumerate(zip(clusters, cluster_p)):
        freq_inds, time_inds = c
        f_start, f_end = freqs[freq_inds.min()], freqs[freq_inds.max()]
        t_start, t_end = times[time_inds.min()], times[time_inds.max()]
        
        sig_marker = " ★ SIGNIFICANT" if p < 0.05 else ""
        msg = f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, p={p:.4f}{sig_marker}"
        print(msg)
        report_lines.append(msg)
        
    cluster_results[condition] = {
        'T_obs': T_obs,
        'clusters': clusters,
        'cluster_p': cluster_p,
        'recalled_mean': np.nanmean(recalled_arr, axis=0),
        'not_recalled_mean': np.nanmean(not_recalled_arr, axis=0),
        'diff_mean': np.nanmean(X_diff_clean, axis=0)
    }

# Save text report
report_file = os.path.join(figures_path, 'TFR_recalled_vs_not_recalled_stats.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n✓ Statistical report saved to {report_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT HEATMAPS (Matches FieldTrip ft_singleplotTFR)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating 2D Heatmaps...")

for condition in ['low', 'high']:
    if condition not in cluster_results: continue
    
    res = cluster_results[condition]
    
    # ──────────────────────────────────────────────────────────────────
    # PLOT 1: The Difference Heatmap (with significance contours)
    # ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot difference map
    vmax = np.max(np.abs(res['diff_mean']))
    im = ax.imshow(
        res['diff_mean'],
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='RdBu_r', vmin=-vmax, vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power Difference (dB)')
    
    # Overlay contours for significant clusters
    for c, p in zip(res['clusters'], res['cluster_p']):
        if p < 0.05:
            # Create a 2D boolean mask for this cluster
            mask = np.zeros_like(res['diff_mean'], dtype=bool)
            mask[c] = True
            
            # Draw contour (like in A2_5_permuatation_TFmaps_stimonset.m)
            ax.contour(
                times, freqs, mask,
                levels=[0.5], colors='black', linewidths=2
            )
            
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.set_title(f'{condition.upper()} Control: Recalled - Not Recalled', fontsize=16)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'TFR_Difference_Heatmap_{condition}.png'), dpi=300)
    plt.close(fig)
    
    # ──────────────────────────────────────────────────────────────────
    # PLOT 2: The Recalled and Not-Recalled absolute heatmaps (Separate)
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Find common scale
    vmax_abs = max(np.max(np.abs(res['recalled_mean'])), np.max(np.abs(res['not_recalled_mean'])))
    
    im1 = axes[0].imshow(
        res['recalled_mean'], aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs
    )
    axes[0].set_title(f'Recalled (Hits)', fontsize=14)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    
    im2 = axes[1].imshow(
        res['not_recalled_mean'], aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs
    )
    axes[1].set_title(f'Not Recalled (Misses)', fontsize=14)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist())
    cbar.set_label('Power (dB)')
    
    fig.suptitle(f'{condition.upper()} Control: Absolute TFR Power', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'TFR_Absolute_Heatmaps_{condition}.png'), dpi=300)
    plt.close(fig)
    
    # ──────────────────────────────────────────────────────────────────
    # PLOT 3: Topoplots (2-20Hz, 0.0-3.5s)
    # ──────────────────────────────────────────────────────────────────
    if len(topo_recalled[condition]) > 0 and info is not None:
        print(f"Generating Topoplots for {condition.upper()} control...")
        
        # Average across subjects
        # Shape: (n_channels,)
        grand_topo_rec = np.nanmean(topo_recalled[condition], axis=0)
        grand_topo_not_rec = np.nanmean(topo_not_recalled[condition], axis=0)
        
        # Convert to dB: 10 * log10(power)
        # Note: raw power from tfr_morlet is positive
        topo_rec_db = 10 * np.log10(np.maximum(grand_topo_rec, 1e-15))
        topo_not_rec_db = 10 * np.log10(np.maximum(grand_topo_not_rec, 1e-15))
        topo_diff_db = topo_rec_db - topo_not_rec_db
        
        import mne
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Recalled
        im_rec, _ = mne.viz.plot_topomap(topo_rec_db, info, axes=axes[0], show=False, sphere='eeglab')
        axes[0].set_title('Recalled')
        cbar_rec = plt.colorbar(im_rec, ax=axes[0], orientation='vertical')
        cbar_rec.set_label('Power (dB)')
        
        # Not Recalled
        im_not_rec, _ = mne.viz.plot_topomap(topo_not_rec_db, info, axes=axes[1], show=False, sphere='eeglab')
        axes[1].set_title('Not Recalled')
        cbar_not_rec = plt.colorbar(im_not_rec, ax=axes[1], orientation='vertical')
        cbar_not_rec.set_label('Power (dB)')
        
        # Difference
        im_diff, _ = mne.viz.plot_topomap(topo_diff_db, info, axes=axes[2], show=False, sphere='eeglab', cmap='RdBu_r')
        axes[2].set_title('Difference (Recalled - Not Recalled)')
        cbar_diff = plt.colorbar(im_diff, ax=axes[2], orientation='vertical')
        cbar_diff.set_label('Power Difference (dB)')
        
        fig.suptitle(f'{condition.upper()} Control: Alpha-Beta (2-20Hz) Topography (0.0-3.5s)', fontsize=16)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_path, f'TFR_Topoplots_{condition}.png'), dpi=300)
        plt.close(fig)

print("✓ All figures saved!")
