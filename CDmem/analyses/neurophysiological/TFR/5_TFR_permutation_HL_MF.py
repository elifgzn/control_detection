"""
5_TFR_permutation_HL_MF.py
==========================
Main Effect of Control (High vs Low) TFR Permutation & Stat Maps (ROI-based)
----------------------------------------------------------------------------

PURPOSE:
    Loads condition-averaged TFR data for the Main Effect of Control
    (High Control vs. Low Control, computed across all trials in 3_TFR_calculation_HL_MF.py)
    and runs a 2D cluster-based permutation test (Frequency x Time).

    Generates standalone figures for:
      1. Condition Grand-Average TFRs (High Control, Low Control)
      2. Difference TFR (High Control - Low Control)
      3. Statistical t-map with significant cluster contours
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

# Colorbar limits
POWER_LIMITS = (-1.5, 1.5)  # dB limits for spectrograms
STAT_LIMITS  = (-2.5, 2.5)  # t-value limits for stat maps

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading per-subject Main Effect of Control data...")
conditions = ['high_control', 'low_control']
subject_data = {f"{sub:04d}": {} for sub in plist}
subject_ch_names = {}
times, freqs_plot, freqs_test, test_f_inds = None, None, None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_MainEffect_Control.npz")

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

    for cond in conditions:
        if cond in saved:
            subject_data[sub_id][cond] = saved[cond][:, freq_mask_plot, :][:, :, time_mask]

# Determine valid subjects
valid_subs = [sub for sub in subject_data if all(c in subject_data[sub] for c in conditions)]
print(f"[Main Effect - Control] Found {len(valid_subs)} subjects with both conditions")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run contrasts. Exiting.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════
# 2. EXTRACT ROI DATA & BUILD CONTRAST
# ══════════════════════════════════════════════════════════════════
# ROI Option 1: all channels starting with 'P' or 'O' (Parieto-Occipital)
sample_chs = next(iter(subject_ch_names.values()))
roi_channels = [ch for ch in sample_chs if ch.startswith(('P', 'O'))]

# ROI Option 2: Frontocentral
# roi_channels = ['Fz', 'FCz', 'FC1', 'FC2']

print(f"\nAveraging over ROI channels ({len(roi_channels)}): {roi_channels}\n")

def extract_roi(data_arr, sub_id):
    chs = subject_ch_names[sub_id]
    idx = [chs.index(ch) for ch in roi_channels if ch in chs]
    if len(idx) == 0:
        raise ValueError(f"Participant {sub_id} has none of the requested ROI channels {roi_channels}!")
    return data_arr[idx, :, :].mean(axis=0)

high_arr = np.array([extract_roi(subject_data[sub]['high_control'], sub) for sub in valid_subs])
low_arr  = np.array([extract_roi(subject_data[sub]['low_control'], sub) for sub in valid_subs])

contrast_name = "Main_Effect_Control"
contrast_desc = "(High Control vs Low Control, all trials)"
X_diff_plot = high_arr - low_arr

condition_grand_averages = {
    "High_Control": {
        'data': np.nanmean(high_arr, axis=0),
        'title': f"Grand Average TFR: High Control\n(N={len(valid_subs)})",
        'filename': "00_TFR_power_MainEffect_Control_high.png"
    },
    "Low_Control": {
        'data': np.nanmean(low_arr, axis=0),
        'title': f"Grand Average TFR: Low Control\n(N={len(valid_subs)})",
        'filename': "00_TFR_power_MainEffect_Control_low.png"
    }
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
print(f"\nRunning 2D (Freq x Time) cluster permutation test ({N_PERMUTATIONS} permutations)...")

report_lines = []
report_lines.append("MAIN EFFECT OF CONTROL (HIGH VS LOW) TFR PERMUTATION TEST RESULTS")
report_lines.append("=" * 70)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"N subjects: {len(valid_subs)}")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s")
report_lines.append(f"Test Frequency window: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz")
report_lines.append(f"ROI channels: {roi_channels}")
report_lines.append("=" * 70)

X_diff_plot = np.nan_to_num(X_diff_plot, nan=0.0)

n_subs = X_diff_plot.shape[0]
df = n_subs - 1
t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)
print(f"\n  {contrast_name}: N={n_subs}, df={df}, cluster threshold t = +/-{t_threshold:.3f}")

X_diff_test = X_diff_plot[:, test_f_inds, :]

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

msg_header = (f"\n{contrast_name.replace('_', ' ').upper()} {contrast_desc}"
              f"\nN={n_subs}, df={df}, cluster threshold=+/-{t_threshold:.3f}"
              f"\n{len(clusters)} clusters found, {n_sig} significant (p < 0.05)")
print(msg_header)
report_lines.append(msg_header)
report_lines.append("-" * 70)

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
# PLOTTING
# -------------------------------------------------------------
print(f"Creating standalone TFR plots for {contrast_name}...")

roi_str = "/".join(roi_channels) if len(roi_channels) <= 6 else f"Parieto-Occipital ({len(roi_channels)} channels)"

# 1) Standalone Difference TFR
ga_diff = np.nanmean(X_diff_plot, axis=0)
fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
im_diff = ax_diff.pcolormesh(time_edges, freq_edges_plot, ga_diff, cmap='RdBu_r', 
                             vmin=POWER_LIMITS[0], vmax=POWER_LIMITS[1], shading='flat')
if mask_tf.any():
    ax_diff.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
cb_diff = fig_diff.colorbar(im_diff, ax=ax_diff, label='Power Difference (dB)')
ax_diff.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
ax_diff.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
ax_diff.set_title(f"DIFFERENCE TFR: HIGH VS LOW CONTROL\n(ROI: {roi_str}, N={n_subs})", 
                  fontsize=15, fontname='Times New Roman')
ax_diff.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
fig_diff.savefig(os.path.join(figures_path, f'01_TFR_diff_{contrast_name}.png'), dpi=300)
plt.close(fig_diff)
print(f"  Saved difference TFR: 01_TFR_diff_{contrast_name}.png")

# 2) Standalone Statistical t-map (with cluster contours)
fig_stat, ax_stat = plt.subplots(figsize=(10, 6))
im_stat = ax_stat.pcolormesh(time_edges, freq_edges_test, T_obs, cmap='RdBu_r', 
                             vmin=STAT_LIMITS[0], vmax=STAT_LIMITS[1], shading='flat')
if mask_tf.any():
    ax_stat.contour(times, freqs_test, mask_tf.astype(float), levels=[0.5], colors='black', linewidths=2)
cb_stat = fig_stat.colorbar(im_stat, ax=ax_stat, label='t value')
ax_stat.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
ax_stat.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
ax_stat.set_title(f"STAT MAP: HIGH VS LOW CONTROL\n(ROI: {roi_str}, N={n_subs})", 
                  fontsize=15, fontname='Times New Roman')
ax_stat.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
fig_stat.savefig(os.path.join(figures_path, f'02_TFR_stat_{contrast_name}.png'), dpi=300)
plt.close(fig_stat)
print(f"  Saved stat map: 02_TFR_stat_{contrast_name}.png")

report_file = os.path.join(figures_path, 'TFR_permutation_HL_MF_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Statistical report saved to {report_file}")
