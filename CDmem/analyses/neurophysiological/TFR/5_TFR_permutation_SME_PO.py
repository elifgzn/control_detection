"""
5_TFR_permutation.py
====================
Alpha-Band Time-Frequency Permutation & Stat Maps (ROI-based)
-------------------------------------------------------------

PURPOSE:
    Loads condition-averaged TFR data and runs 2D cluster-based permutation
    tests (Frequency x Time) on a predefined TEST frequency range.

    Generates standalone figures for:
      1. Condition Grand-Average TFRs (Power in dB relative to baseline)
      2. Difference TFRs (Power difference in dB relative to baseline)
      3. Statistical t-maps with significant cluster contours

    Contrasts computed:
      A. Main Effect of Memory (collapsed across control):
         - Controlled items only
         - All items (either item recalled)
      B. Simple Effects of Memory within each control level (from 4-cell AllItems):
         - Low Control: recalled vs not recalled
         - High Control: recalled vs not recalled
         - Interaction: Memory x Control
      C. Simple Effects of Memory within each control level (from 4-cell ControlledOnly):
         - Low Control: recalled vs not recalled
         - High Control: recalled vs not recalled
         - Interaction: Memory x Control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t as t_dist

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
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
TEST_FREQ = (2.0, 40.0)
PLOT_FREQ = (2.0, 40.0)

# Colorbar limits for standalone figures
POWER_LIMITS = (-1.5, 1.5)  # dB limits for condition and difference spectrograms
STAT_LIMITS = (-2.5, 2.5)   # t-value limits for permutation stat maps

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

# --- A. Collapsed main-effect files (recalled vs not_recalled) ---
main_effect_files = {
    'ControlledOnly': '_TFR_MainEffect_ControlledOnly.npz',
    'AllItems': '_TFR_MainEffect_AllItems.npz',
}

collapsed_conditions = ['recalled', 'not_recalled']
subject_data_collapsed = {ft: {f"{sub:04d}": {} for sub in plist} for ft in main_effect_files}
subject_ch_names = {}
times, freqs_plot, freqs_test, test_f_inds = None, None, None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    for ft_key, ft_suffix in main_effect_files.items():
        data_file = os.path.join(input_path, f"CDmem_{sub_id}{ft_suffix}")
        
        if not os.path.exists(data_file):
            continue
            
        saved = np.load(data_file, allow_pickle=True)
        
        if sub_id not in subject_ch_names and 'ch_names' in saved:
            subject_ch_names[sub_id] = saved['ch_names'].tolist()
            
        if times is None:
            times_all = saved['times']
            freqs_all = saved['freqs']
            
            time_mask = (times_all >= TEST_TIME[0]) & (times_all <= TEST_TIME[1])
            times = times_all[time_mask]
            
            freq_mask_plot = (freqs_all >= PLOT_FREQ[0]) & (freqs_all <= PLOT_FREQ[1])
            freqs_plot = freqs_all[freq_mask_plot]
            
            test_f_inds = np.where((freqs_plot >= TEST_FREQ[0]) & (freqs_plot <= TEST_FREQ[1]))[0]
            freqs_test = freqs_plot[test_f_inds]
            
        for cond in collapsed_conditions:
            if cond in saved:
                subject_data_collapsed[ft_key][sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

# --- B. 4-cell AllItems file (low/high x recalled/not_recalled) ---
print("Loading per-subject 4-cell condition averages (All Items)...")
four_cell_conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']
subject_data_4cell = {f"{sub:04d}": {} for sub in plist}

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    
    if sub_id not in subject_ch_names and 'ch_names' in saved:
        subject_ch_names[sub_id] = saved['ch_names'].tolist()
        
    for cond in four_cell_conditions:
        if cond in saved:
            subject_data_4cell[sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

# --- C. 4-cell ControlledOnly file (low/high x recalled/not_recalled) ---
print("Loading per-subject 4-cell condition averages (Controlled Only)...")
subject_data_4cell_ctrl = {f"{sub:04d}": {} for sub in plist}

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    
    if sub_id not in subject_ch_names and 'ch_names' in saved:
        subject_ch_names[sub_id] = saved['ch_names'].tolist()
        
    for cond in four_cell_conditions:
        if cond in saved:
            subject_data_4cell_ctrl[sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

# Determine valid subjects
valid_subs_collapsed = {}
for ft_key in main_effect_files:
    valid_subs_collapsed[ft_key] = [sub for sub in subject_data_collapsed[ft_key] 
                                     if all(c in subject_data_collapsed[ft_key][sub] for c in collapsed_conditions)]
    print(f"[Main Effect - {ft_key}] Found {len(valid_subs_collapsed[ft_key])} subjects with both conditions")

valid_subs_4cell = [sub for sub in subject_data_4cell if all(c in subject_data_4cell[sub] for c in four_cell_conditions)]
print(f"[4-cell AllItems] Found {len(valid_subs_4cell)} subjects with all 4 conditions")

valid_subs_4cell_ctrl = [sub for sub in subject_data_4cell_ctrl if all(c in subject_data_4cell_ctrl[sub] for c in four_cell_conditions)]
print(f"[4-cell ControlledOnly] Found {len(valid_subs_4cell_ctrl)} subjects with all 4 conditions")

if all(len(v) < 2 for v in valid_subs_collapsed.values()) and len(valid_subs_4cell) < 2 and len(valid_subs_4cell_ctrl) < 2:
    print("Not enough complete subjects to run contrasts. Exiting.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════
# 2. EXTRACT ROI DATA & BUILD CONTRASTS
# ══════════════════════════════════════════════════════════════════
# ROI Option 1: all channels starting with 'P' or 'O' (Parieto-Occipital)
sample_chs = next(iter(subject_ch_names.values()))
roi_channels = [ch for ch in sample_chs if ch.startswith(('P', 'O'))]

# # ROI Option 2: Frontocentral
# roi_channels = ['Fz', 'FCz', 'FC1', 'FC2']

print(f"\nAveraging over ROI channels ({len(roi_channels)}): {roi_channels}\n")

def extract_roi(data_arr, sub_id):
    chs = subject_ch_names[sub_id]
    idx = [chs.index(ch) for ch in roi_channels if ch in chs]
    if len(idx) == 0:
        raise ValueError(f"Participant {sub_id} has none of the requested ROI channels {roi_channels}!")
    return data_arr[idx, :, :].mean(axis=0)

contrasts = {}
descriptions = {}
contrast_n_subs = {}
condition_grand_averages = {}

# A. Collapsed main-effect contrasts
for ft_key in main_effect_files:
    subs = valid_subs_collapsed[ft_key]
    if len(subs) < 2:
        continue
    
    rec = np.array([extract_roi(subject_data_collapsed[ft_key][sub]['recalled'], sub) for sub in subs])
    not_rec = np.array([extract_roi(subject_data_collapsed[ft_key][sub]['not_recalled'], sub) for sub in subs])
    
    name = f"Main_Effect_Memory_{ft_key}"
    contrasts[name] = rec - not_rec
    descriptions[name] = f"(Recalled vs Not Recalled, collapsed across control - {ft_key})"
    
    condition_grand_averages[f"MainEffect_{ft_key}_Recalled"] = {
        'data': np.nanmean(rec, axis=0),
        'title': f"Grand Average TFR: Recalled\n(Main Effect - {ft_key}, N={len(subs)})",
        'filename': f"00_TFR_power_MainEffect_{ft_key}_recalled.png"
    }
    condition_grand_averages[f"MainEffect_{ft_key}_NotRecalled"] = {
        'data': np.nanmean(not_rec, axis=0),
        'title': f"Grand Average TFR: Not Recalled\n(Main Effect - {ft_key}, N={len(subs)})",
        'filename': f"00_TFR_power_MainEffect_{ft_key}_not_recalled.png"
    }

# B. Within-control simple effects + interaction (from 4-cell AllItems)
if len(valid_subs_4cell) >= 2:
    subs = valid_subs_4cell
    low_rec = np.array([extract_roi(subject_data_4cell[sub]['low_recalled'], sub) for sub in subs])
    low_not = np.array([extract_roi(subject_data_4cell[sub]['low_not_recalled'], sub) for sub in subs])
    high_rec = np.array([extract_roi(subject_data_4cell[sub]['high_recalled'], sub) for sub in subs])
    high_not = np.array([extract_roi(subject_data_4cell[sub]['high_not_recalled'], sub) for sub in subs])
    
    low_diff = low_rec - low_not
    high_diff = high_rec - high_not
    
    contrasts['Simple_Effect_Low_Control_AllItems'] = low_diff
    descriptions['Simple_Effect_Low_Control_AllItems'] = '(Recalled vs Not Recalled WITHIN Low Control only)'
    
    contrasts['Simple_Effect_High_Control_AllItems'] = high_diff
    descriptions['Simple_Effect_High_Control_AllItems'] = '(Recalled vs Not Recalled WITHIN High Control only)'
    
    contrasts['Interaction_Memory_x_Control_AllItems'] = low_diff - high_diff
    descriptions['Interaction_Memory_x_Control_AllItems'] = '(Difference in Memory Effect between Low and High Control - All Items)'
    
    for cond_name, arr in [('Low_Control_Recalled', low_rec),
                           ('Low_Control_Not_Recalled', low_not),
                           ('High_Control_Recalled', high_rec),
                           ('High_Control_Not_Recalled', high_not)]:
        condition_grand_averages[f"AllItems_{cond_name}"] = {
            'data': np.nanmean(arr, axis=0),
            'title': f"Grand Average TFR: {cond_name.replace('_', ' ')}\n(All Items, N={len(subs)})",
            'filename': f"00_TFR_power_AllItems_{cond_name.lower()}.png"
        }

# C. Within-control simple effects + interaction (from 4-cell Controlled Only)
if len(valid_subs_4cell_ctrl) >= 2:
    subs = valid_subs_4cell_ctrl
    low_rec_c = np.array([extract_roi(subject_data_4cell_ctrl[sub]['low_recalled'], sub) for sub in subs])
    low_not_c = np.array([extract_roi(subject_data_4cell_ctrl[sub]['low_not_recalled'], sub) for sub in subs])
    high_rec_c = np.array([extract_roi(subject_data_4cell_ctrl[sub]['high_recalled'], sub) for sub in subs])
    high_not_c = np.array([extract_roi(subject_data_4cell_ctrl[sub]['high_not_recalled'], sub) for sub in subs])
    
    low_diff_c = low_rec_c - low_not_c
    high_diff_c = high_rec_c - high_not_c
    
    contrasts['Simple_Effect_Low_Control_CtrlOnly'] = low_diff_c
    descriptions['Simple_Effect_Low_Control_CtrlOnly'] = '(Recalled vs Not Recalled WITHIN Low Control - Controlled items only)'
    
    contrasts['Simple_Effect_High_Control_CtrlOnly'] = high_diff_c
    descriptions['Simple_Effect_High_Control_CtrlOnly'] = '(Recalled vs Not Recalled WITHIN High Control - Controlled items only)'
    
    contrasts['Interaction_Memory_x_Control_CtrlOnly'] = low_diff_c - high_diff_c
    descriptions['Interaction_Memory_x_Control_CtrlOnly'] = '(Difference in Memory Effect between Low and High Control - Controlled items only)'
    
    for cond_name, arr in [('Low_Control_Recalled', low_rec_c),
                           ('Low_Control_Not_Recalled', low_not_c),
                           ('High_Control_Recalled', high_rec_c),
                           ('High_Control_Not_Recalled', high_not_c)]:
        condition_grand_averages[f"ControlledOnly_{cond_name}"] = {
            'data': np.nanmean(arr, axis=0),
            'title': f"Grand Average TFR: {cond_name.replace('_', ' ')}\n(Controlled Only, N={len(subs)})",
            'filename': f"00_TFR_power_ControlledOnly_{cond_name.lower()}.png"
        }

# ══════════════════════════════════════════════════════════════════
# 3. PLOT CONDITION GRAND AVERAGES (Power in dB)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating Condition Grand-Average TFR Power Plots (dB)...")
time_edges = np.concatenate([times - np.diff(times[:2])[0]/2, [times[-1] + np.diff(times[:2])[0]/2]])
freq_edges_plot = np.concatenate([freqs_plot - np.diff(freqs_plot[:2])[0]/2, [freqs_plot[-1] + np.diff(freqs_plot[:2])[0]/2]])
freq_edges_test = np.concatenate([freqs_test - np.diff(freqs_test[:2])[0]/2, [freqs_test[-1] + np.diff(freqs_test[:2])[0]/2]])

for cond_key, item in condition_grand_averages.items():
    fig_cond, ax_cond = plt.subplots(figsize=(10, 6))
    im = ax_cond.pcolormesh(time_edges, freq_edges_plot, item['data'], 
                            cmap='RdBu_r', vmin=POWER_LIMITS[0], vmax=POWER_LIMITS[1], shading='flat')
    cb = fig_cond.colorbar(im, ax=ax_cond, label='Power (dB)')
    ax_cond.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax_cond.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax_cond.set_title(item['title'], fontsize=15, fontname='Times New Roman')
    ax_cond.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    out_path = os.path.join(figures_path, item['filename'])
    fig_cond.savefig(out_path, dpi=300)
    plt.close(fig_cond)
    print(f"  Saved condition TFR: {item['filename']}")

# ══════════════════════════════════════════════════════════════════
# 4. RUN CLUSTER PERMUTATION TEST & PLOT STAT/DIFF MAPS
# ══════════════════════════════════════════════════════════════════
print(f"\nRunning 2D (Freq x Time) cluster permutation tests ({N_PERMUTATIONS} permutations)...")

report_lines = []
report_lines.append("TFR PERMUTATION TEST RESULTS")
report_lines.append("=" * 70)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
for ft_key in main_effect_files:
    report_lines.append(f"N (Main Effect {ft_key}): {len(valid_subs_collapsed[ft_key])}")
report_lines.append(f"N (4-cell AllItems): {len(valid_subs_4cell)}")
report_lines.append(f"N (4-cell ControlledOnly): {len(valid_subs_4cell_ctrl)}")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s")
report_lines.append(f"Test Frequency window: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz")
report_lines.append("=" * 70)

for comp_name, X_diff_plot in contrasts.items():
    # Replace NaNs with 0 to prevent issues with permutation test
    X_diff_plot = np.nan_to_num(X_diff_plot, nan=0.0)
    
    # Compute per-contrast threshold
    n_subs = X_diff_plot.shape[0]
    df = n_subs - 1
    t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)
    print(f"\n  {comp_name}: N={n_subs}, df={df}, cluster threshold t = +/-{t_threshold:.3f}")
    
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
    msg_header = (f"\n{comp_name.replace('_', ' ').upper()} {desc}"
                  f"\nN={n_subs}, df={df}, cluster threshold=+/-{t_threshold:.3f}"
                  f"\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)")
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
            sig_marker = " ** SIGNIFICANT"
        else:
            sig_marker = ""
            
        mean_t = T_obs[mask].mean()
        
        msg = (f"  Cluster {i+1}: {t_start:.3f}-{t_end:.3f}s, {f_low:.1f}-{f_high:.1f}Hz, "
               f"p={pval:.4f}{sig_marker} | Mean T: {mean_t:.2f}")
        print(msg)
        report_lines.append(msg)

    # -------------------------------------------------------------
    # 4B. PLOTTING (Option 2: Standalone Difference & Stat Maps)
    # -------------------------------------------------------------
    print(f"Creating standalone TFR plots for {comp_name}...")
    
    roi_str = "/".join(roi_channels) if len(roi_channels) <= 6 else f"Parieto-Occipital ({len(roi_channels)} channels)"
    # 1) Standalone Difference TFR (Power in dB)
    ga_diff = np.nanmean(X_diff_plot, axis=0)
    fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
    im_diff = ax_diff.pcolormesh(time_edges, freq_edges_plot, ga_diff, cmap='RdBu_r', 
                                 vmin=POWER_LIMITS[0], vmax=POWER_LIMITS[1], shading='flat')
    if mask_tf.any():
        ax_diff.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
    cb_diff = fig_diff.colorbar(im_diff, ax=ax_diff, label='Power Difference (dB)')
    ax_diff.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax_diff.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax_diff.set_title(f"DIFFERENCE TFR: {comp_name.replace('_', ' ').upper()}\n(ROI: {roi_str}, N={n_subs})", 
                      fontsize=15, fontname='Times New Roman')
    ax_diff.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    fig_diff.savefig(os.path.join(figures_path, f'01_TFR_diff_{comp_name}.png'), dpi=300)
    plt.close(fig_diff)
    print(f"  Saved difference TFR: 01_TFR_diff_{comp_name}.png")
    
    # 2) Standalone Statistical t-map (with cluster contours)
    fig_stat, ax_stat = plt.subplots(figsize=(10, 6))
    im_stat = ax_stat.pcolormesh(time_edges, freq_edges_test, T_obs, cmap='RdBu_r', 
                                 vmin=STAT_LIMITS[0], vmax=STAT_LIMITS[1], shading='flat')
    if mask_tf.any():
        ax_stat.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
    cb_stat = fig_stat.colorbar(im_stat, ax=ax_stat, label='t value')
    ax_stat.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax_stat.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax_stat.set_title(f"STAT MAP: {comp_name.replace('_', ' ').upper()}\n(ROI: {roi_str}, N={n_subs})", 
                      fontsize=15, fontname='Times New Roman')
    ax_stat.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    fig_stat.savefig(os.path.join(figures_path, f'02_TFR_stat_{comp_name}.png'), dpi=300)
    # Also save as 01_tf_permut_{comp_name}_TFR.png for backward compatibility
    fig_stat.savefig(os.path.join(figures_path, f'01_tf_permut_{comp_name}_TFR.png'), dpi=300)
    plt.close(fig_stat)
    print(f"  Saved stat map: 02_TFR_stat_{comp_name}.png")

report_file = os.path.join(figures_path, 'TFR_permutation_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Statistical report saved to {report_file}")