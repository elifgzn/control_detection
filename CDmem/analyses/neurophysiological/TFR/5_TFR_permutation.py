"""
5_TFR_permutation.py
====================
Alpha-Band ROI Time-Frequency Permutation & Stat Maps
-----------------------------------------------------

PURPOSE:
    Loads the condition-averaged TFR data, runs a 2D cluster-based permutation test
    (Time × Frequency) on a predefined TEST frequency range (e.g. 2-20 Hz) while 
    plotting the full range up to a PLOT frequency (e.g. 2-40 Hz) and generates 
    statistical contour maps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked"
os.makedirs(figures_path, exist_ok=True)

plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30, 31]

# Permutation test parameters
N_PERMUTATIONS = 1000
TAIL = 0           # two-sided t-test
SEED = 2025

Z_LIMITS_DIFF = [-1.5, 1.5]  # Z-limits for difference heatmap
# Z_LIMITS_DIFF = [-2.5, 2.5]  # Z-limits for difference heatmap (matches Wu et al)
Z_LIMITS_STAT = [0, 25]      # FieldTrip scripts occasionally use [0 25] for F-stats/T-stats

def _get_2d_cluster_inds(cluster):
    """Return (freq_inds, time_inds) as 1-D integer arrays for a 2D cluster."""
    if isinstance(cluster, tuple) and len(cluster) == 2:
        freq_part, time_part = cluster
        return np.asarray(freq_part).ravel(), np.asarray(time_part).ravel()
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.ndim == 2:
        freq_inds, time_inds = np.where(cluster)
        return freq_inds, time_inds
    raise ValueError(f"Unexpected 2D cluster format: type={type(cluster)}")

TEST_TIME = (0.0, 3.0)
TEST_FREQ = (2.0, 20.0)
PLOT_FREQ = (2.0, 40.0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & PREPARE CONTRASTS FOR STATS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading per-subject condition averages...")

conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']
subject_roi_data = {f"{sub:04d}": {} for sub in plist}
times, freqs_plot, freqs_test, test_f_inds, ch_names, roi_channels = None, None, None, None, None, None

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
        
        time_mask = (times_all >= TEST_TIME[0]) & (times_all <= TEST_TIME[1])
        times = times_all[time_mask]
        
        freq_mask_plot = (freqs_all >= PLOT_FREQ[0]) & (freqs_all <= PLOT_FREQ[1])
        freqs_plot = freqs_all[freq_mask_plot]
        
        # Identify indices within the PLOT array that correspond to the TEST range
        test_f_inds = np.where((freqs_plot >= TEST_FREQ[0]) & (freqs_plot <= TEST_FREQ[1]))[0]
        freqs_test = freqs_plot[test_f_inds]
        
    for cond in conditions:
        if cond in saved:
            roi_avg = np.nanmean(saved[cond][roi_idx, :, :], axis=0)
            subject_roi_data[sub_id][cond] = roi_avg[np.ix_(freq_mask_plot, time_mask)]

valid_subs = [sub for sub in subject_roi_data if len(subject_roi_data[sub]) == 4]
print(f"Found {len(valid_subs)} subjects with all 4 conditions.")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run paired contrasts. Exiting.")
    exit()

low_rec = np.array([subject_roi_data[sub]['low_recalled'] for sub in valid_subs])
low_not = np.array([subject_roi_data[sub]['low_not_recalled'] for sub in valid_subs])
high_rec = np.array([subject_roi_data[sub]['high_recalled'] for sub in valid_subs])
high_not = np.array([subject_roi_data[sub]['high_not_recalled'] for sub in valid_subs])

contrasts = {}
main_rec = (low_rec + high_rec) / 2
main_not = (low_not + high_not) / 2
contrasts['Main_Effect_Memory'] = main_rec - main_not

low_diff = low_rec - low_not
high_diff = high_rec - high_not
contrasts['Interaction_Memory_x_Control'] = low_diff - high_diff
contrasts['OLD_ANALYSIS__Low_Control_Memory_Effect'] = low_diff
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
report_lines.append(f"Test Frequency window: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz")
report_lines.append("=" * 70)

for comp_name, X_diff in contrasts.items():
    X_diff_clean_plot = np.nan_to_num(X_diff, nan=0.0)
    # Subset data specifically for the statistical test
    X_diff_clean_test = X_diff_clean_plot[:, test_f_inds, :]
    
    # Run 2D Permutation Test on TEST subset
    T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
        X_diff_clean_test,
        n_permutations=N_PERMUTATIONS,
        tail=TAIL,
        n_jobs=-1,
        seed=SEED
    )
    
    # Compute Grand Average Difference on FULL plot array
    ga_diff_plot = np.nanmean(X_diff_clean_plot, axis=0)
    
    sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
    n_sig = len(sig_clusters)
    
    desc = descriptions.get(comp_name, '')
    msg_header = f"\n{comp_name.replace('_', ' ').upper()} {desc}\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 70)
    
    for i, (c, p) in enumerate(zip(clusters, cluster_p)):
        if p < 0.05:
            mask_test = np.zeros_like(ga_diff_plot[test_f_inds, :], dtype=bool)
            mask_test[c] = True
            
            freq_inds, time_inds = _get_2d_cluster_inds(c)
            f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
            t_start, t_end = times[time_inds.min()], times[time_inds.max()]
            
            mean_diff = np.nanmean(X_diff_clean_test, axis=0)
            std_diff = np.nanstd(X_diff_clean_test, axis=0, ddof=1)
            cohens_d_map = mean_diff / np.where(std_diff == 0, 1e-10, std_diff)
            cluster_cohens_d = np.mean(np.abs(cohens_d_map[mask_test]))
            
            msg = (f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, "
                   f"p={p:.4f} ★ | Cohen's d: {cluster_cohens_d:.2f}")
            print(msg)
            report_lines.append(msg)
        else:
            freq_inds, time_inds = _get_2d_cluster_inds(c)
            f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
            t_start, t_end = times[time_inds.min()], times[time_inds.max()]
            msg = f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, p={p:.4f}"
            print(msg)
            report_lines.append(msg)
            
    # ══════════════════════════════════════════════════════════════════════════════
    # 3. PLOT DIFFERENCE HEATMAP WITH CONTOURS
    # ══════════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        ga_diff_plot,
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs_plot[0], freqs_plot[-1]],
        cmap='RdBu_r', vmin=Z_LIMITS_DIFF[0], vmax=Z_LIMITS_DIFF[1]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power Difference (dB)')
    
    # Overlay contours mapped back to the PLOT dimensions
    for c, p in zip(clusters, cluster_p):
        if p < 0.05:
            mask_plot = np.zeros_like(ga_diff_plot, dtype=bool)
            freq_inds, time_inds = _get_2d_cluster_inds(c)
            freq_inds_plot = freq_inds + test_f_inds[0]
            mask_plot[freq_inds_plot, time_inds] = True
            ax.contour(times, freqs_plot, mask_plot, levels=[0.5], colors='black', linewidths=2)
            
    ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax.set_ylim(0, 40)
    ax.set_title(f"{comp_name.replace('_', ' ').upper()}", fontsize=16, fontname='Times New Roman')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'01_tf_permut_{comp_name}_diff.png'), dpi=300)
    plt.close(fig)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 4. PLOT TEST STATISTIC HEATMAP WITH CONTOURS
    # ══════════════════════════════════════════════════════════════════════════════
    # Use absolute T-values for the stat map. Embed it back into the full plot shape.
    stat_map_plot = np.zeros_like(ga_diff_plot)
    stat_map_plot[test_f_inds, :] = np.abs(T_obs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        stat_map_plot,
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs_plot[0], freqs_plot[-1]],
        cmap='Reds', vmin=Z_LIMITS_STAT[0], vmax=Z_LIMITS_STAT[1]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Test Statistic (T-value)')
    
    for c, p in zip(clusters, cluster_p):
        if p < 0.05:
            mask_plot = np.zeros_like(ga_diff_plot, dtype=bool)
            freq_inds, time_inds = _get_2d_cluster_inds(c)
            freq_inds_plot = freq_inds + test_f_inds[0]
            mask_plot[freq_inds_plot, time_inds] = True
            ax.contour(times, freqs_plot, mask_plot, levels=[0.5], colors='black', linewidths=2)
            
    ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax.set_ylim(0, 40)
    ax.set_title(f"{comp_name.replace('_', ' ').upper()} (Stat Map)", fontsize=16, fontname='Times New Roman')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'01_stat_permut_{comp_name}.png'), dpi=300)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SUPPLEMENTARY ANALYSIS (H5): All Items (Collapsed across item type)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUNNING SUPPLEMENTARY ANALYSIS (ALL ITEMS)")
print("=" * 70)

supp_conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']
supp_subject_roi_data = {f"{sub:04d}": {} for sub in plist}

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    for cond in supp_conditions:
        if cond in saved:
            roi_avg = np.nanmean(saved[cond][roi_idx, :, :], axis=0)
            supp_subject_roi_data[sub_id][cond] = roi_avg[np.ix_(freq_mask_plot, time_mask)]

valid_subs_supp = [sub for sub in supp_subject_roi_data if len(supp_subject_roi_data[sub]) == 4]
print(f"Found {len(valid_subs_supp)} subjects with all 4 supplementary conditions.")

if len(valid_subs_supp) >= 2:
    low_rec_s = np.array([supp_subject_roi_data[sub]['low_recalled'] for sub in valid_subs_supp])
    low_not_s = np.array([supp_subject_roi_data[sub]['low_not_recalled'] for sub in valid_subs_supp])
    high_rec_s = np.array([supp_subject_roi_data[sub]['high_recalled'] for sub in valid_subs_supp])
    high_not_s = np.array([supp_subject_roi_data[sub]['high_not_recalled'] for sub in valid_subs_supp])
    
    main_rec_s = (low_rec_s + high_rec_s) / 2
    main_not_s = (low_not_s + high_not_s) / 2
    
    supp_contrasts = {
        'SUPPLEMENTARY__Main_Effect_Memory': main_rec_s - main_not_s
    }
    
    for comp_name, X_diff in supp_contrasts.items():
        X_diff_clean_plot = np.nan_to_num(X_diff, nan=0.0)
        X_diff_clean_test = X_diff_clean_plot[:, test_f_inds, :]
        
        T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
            X_diff_clean_test, n_permutations=N_PERMUTATIONS, tail=TAIL, n_jobs=-1, seed=SEED
        )
        ga_diff_plot = np.nanmean(X_diff_clean_plot, axis=0)
        sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
        n_sig = len(sig_clusters)
        
        msg_header = f"\n{comp_name.replace('_', ' ').upper()} (Collapsed across item types)\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
        print(msg_header)
        report_lines.append(msg_header)
        report_lines.append("-" * 70)
        
        for i, (c, p) in enumerate(zip(clusters, cluster_p)):
            if p < 0.05:
                mask_test = np.zeros_like(ga_diff_plot[test_f_inds, :], dtype=bool)
                mask_test[c] = True
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
                t_start, t_end = times[time_inds.min()], times[time_inds.max()]
                mean_diff = np.nanmean(X_diff_clean_test, axis=0)
                std_diff = np.nanstd(X_diff_clean_test, axis=0, ddof=1)
                cohens_d_map = mean_diff / np.where(std_diff == 0, 1e-10, std_diff)
                cluster_cohens_d = np.mean(np.abs(cohens_d_map[mask_test]))
                msg = (f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, "
                       f"p={p:.4f} ★ | Cohen's d: {cluster_cohens_d:.2f}")
                print(msg)
                report_lines.append(msg)
            else:
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
                t_start, t_end = times[time_inds.min()], times[time_inds.max()]
                msg = f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, p={p:.4f}"
                print(msg)
                report_lines.append(msg)
                
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(ga_diff_plot, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs_plot[0], freqs_plot[-1]],
                       cmap='RdBu_r', vmin=Z_LIMITS_DIFF[0], vmax=Z_LIMITS_DIFF[1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power Difference (dB)')
        for c, p in zip(clusters, cluster_p):
            if p < 0.05:
                mask_plot = np.zeros_like(ga_diff_plot, dtype=bool)
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                freq_inds_plot = freq_inds + test_f_inds[0]
                mask_plot[freq_inds_plot, time_inds] = True
                ax.contour(times, freqs_plot, mask_plot, levels=[0.5], colors='black', linewidths=2)
        ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
        ax.set_ylim(0, 40)
        ax.set_title(f"{comp_name.replace('_', ' ').upper()}", fontsize=16, fontname='Times New Roman')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_path, f'02_tf_permut_{comp_name}_diff.png'), dpi=300)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPLORATORY ANALYSIS: Item Type Split (NOT PREREGISTERED)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUNNING EXPLORATORY ANALYSIS (ITEM TYPE SPLIT) - NOT PREREGISTERED")
print("=" * 70)

exp_conditions = [
    'low_controlled_recalled', 'low_controlled_not_recalled',
    'low_uncontrolled_recalled', 'low_uncontrolled_not_recalled',
    'high_controlled_recalled', 'high_controlled_not_recalled',
    'high_uncontrolled_recalled', 'high_uncontrolled_not_recalled'
]
exp_subject_roi_data = {f"{sub:04d}": {} for sub in plist}

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ItemTypeAverages.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    for cond in exp_conditions:
        if cond in saved:
            roi_avg = np.nanmean(saved[cond][roi_idx, :, :], axis=0)
            exp_subject_roi_data[sub_id][cond] = roi_avg[np.ix_(freq_mask_plot, time_mask)]

valid_subs_exp = [sub for sub in exp_subject_roi_data if len(exp_subject_roi_data[sub]) == 8]
print(f"Found {len(valid_subs_exp)} subjects with all 8 item-type conditions.")

if len(valid_subs_exp) >= 2:
    low_ctrl_rec = np.array([exp_subject_roi_data[sub]['low_controlled_recalled'] for sub in valid_subs_exp])
    low_ctrl_not = np.array([exp_subject_roi_data[sub]['low_controlled_not_recalled'] for sub in valid_subs_exp])
    low_unctrl_rec = np.array([exp_subject_roi_data[sub]['low_uncontrolled_recalled'] for sub in valid_subs_exp])
    low_unctrl_not = np.array([exp_subject_roi_data[sub]['low_uncontrolled_not_recalled'] for sub in valid_subs_exp])
    
    high_ctrl_rec = np.array([exp_subject_roi_data[sub]['high_controlled_recalled'] for sub in valid_subs_exp])
    high_ctrl_not = np.array([exp_subject_roi_data[sub]['high_controlled_not_recalled'] for sub in valid_subs_exp])
    high_unctrl_rec = np.array([exp_subject_roi_data[sub]['high_uncontrolled_recalled'] for sub in valid_subs_exp])
    high_unctrl_not = np.array([exp_subject_roi_data[sub]['high_uncontrolled_not_recalled'] for sub in valid_subs_exp])
    
    Dm_low_ctrl = low_ctrl_rec - low_ctrl_not
    Dm_low_unctrl = low_unctrl_rec - low_unctrl_not
    Dm_high_ctrl = high_ctrl_rec - high_ctrl_not
    Dm_high_unctrl = high_unctrl_rec - high_unctrl_not
    
    exp_contrasts = {
        'EXPLORATORY__Main_Effect_ItemType_on_Dm': ((Dm_low_ctrl + Dm_high_ctrl)/2) - ((Dm_low_unctrl + Dm_high_unctrl)/2),
        'EXPLORATORY__Interaction_Control_x_ItemType_on_Dm': (Dm_high_ctrl - Dm_high_unctrl) - (Dm_low_ctrl - Dm_low_unctrl)
    }
    
    for comp_name, X_diff in exp_contrasts.items():
        X_diff_clean_plot = np.nan_to_num(X_diff, nan=0.0)
        X_diff_clean_test = X_diff_clean_plot[:, test_f_inds, :]
        
        T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
            X_diff_clean_test, n_permutations=N_PERMUTATIONS, tail=TAIL, n_jobs=-1, seed=SEED
        )
        ga_diff_plot = np.nanmean(X_diff_clean_plot, axis=0)
        sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
        n_sig = len(sig_clusters)
        
        msg_header = f"\n{comp_name.replace('_', ' ').upper()} (NOT PREREGISTERED)\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
        print(msg_header)
        report_lines.append(msg_header)
        report_lines.append("-" * 70)
        
        for i, (c, p) in enumerate(zip(clusters, cluster_p)):
            if p < 0.05:
                mask_test = np.zeros_like(ga_diff_plot[test_f_inds, :], dtype=bool)
                mask_test[c] = True
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
                t_start, t_end = times[time_inds.min()], times[time_inds.max()]
                mean_diff = np.nanmean(X_diff_clean_test, axis=0)
                std_diff = np.nanstd(X_diff_clean_test, axis=0, ddof=1)
                cohens_d_map = mean_diff / np.where(std_diff == 0, 1e-10, std_diff)
                cluster_cohens_d = np.mean(np.abs(cohens_d_map[mask_test]))
                msg = (f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, "
                       f"p={p:.4f} ★ | Cohen's d: {cluster_cohens_d:.2f}")
                print(msg)
                report_lines.append(msg)
            else:
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                f_start, f_end = freqs_test[freq_inds.min()], freqs_test[freq_inds.max()]
                t_start, t_end = times[time_inds.min()], times[time_inds.max()]
                msg = f"  Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, {f_start:.1f}–{f_end:.1f}Hz, p={p:.4f}"
                print(msg)
                report_lines.append(msg)
                
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(ga_diff_plot, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs_plot[0], freqs_plot[-1]],
                       cmap='RdBu_r', vmin=Z_LIMITS_DIFF[0], vmax=Z_LIMITS_DIFF[1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power Difference (dB)')
        for c, p in zip(clusters, cluster_p):
            if p < 0.05:
                mask_plot = np.zeros_like(ga_diff_plot, dtype=bool)
                freq_inds, time_inds = _get_2d_cluster_inds(c)
                freq_inds_plot = freq_inds + test_f_inds[0]
                mask_plot[freq_inds_plot, time_inds] = True
                ax.contour(times, freqs_plot, mask_plot, levels=[0.5], colors='black', linewidths=2)
        ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
        ax.set_ylim(0, 40)
        ax.set_title(f"{comp_name.replace('_', ' ').upper()}", fontsize=16, fontname='Times New Roman')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_path, f'03_tf_permut_{comp_name}_diff.png'), dpi=300)
        plt.close(fig)

# Save text report to file
report_file = os.path.join(figures_path, 'TFR_permutation_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
    print(f"\n* Statistical report saved to {report_file}")
