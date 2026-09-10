"""
5_TFR_permutation_HL_full3D.py
==============================
Main Effect of Control (High vs Low) Full 3D Spatio-Temporal-Spectral Permutation Test
--------------------------------------------------------------------------------------

PURPOSE:
    Runs a hypothesis-free, whole-brain, broadband 3D cluster-based permutation test
    across Channels x Frequencies x Times (no a priori ROI electrode or band pre-selection).

    Adjacency:
      - Spatial adjacency: Delaunay triangulation across 65 scalp EEG electrodes
      - Spectral adjacency: Regular 1D lattice along frequency bins (2 - 40 Hz)
      - Temporal adjacency: Regular 1D lattice along time points (0 - 3 s)
      Combined via mne.stats.combine_adjacency(ch_adj, n_freqs, n_times).

    Outputs (generated IF AND ONLY IF a cluster is significant, p < 0.05):
      1. Condition Topoplot: High Control (participating cluster electrodes starred)
      2. Condition Topoplot: Low Control (participating cluster electrodes starred)
      3. Difference Topoplot: High - Low Control (participating cluster electrodes starred)
      4. Combined 3-Panel Figure: [High Control | Low Control | Difference]
      5. Cluster-Averaged TFR Spectrogram: Spectrogram averaged across participating
         electrodes with significant cluster contours
      6. Comprehensive statistical report text file
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne.stats.cluster_level as cl
from mne.stats import permutation_cluster_1samp_test, combine_adjacency
from scipy.stats import t as t_dist

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked_HL_MF"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked_HL_full3D"
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

# Spatial clustering & pruning parameters (mirroring FieldTrip's cfg.minnbchan)
# At each (freq, time) bin, an electrode is only retained if at least
# MIN_ADJACENT_NEIGHBORS spatial neighbors also exceed the univariate t-threshold.
MIN_ADJACENT_NEIGHBORS = 2  # 2 requires >= 3 total electrodes in spatial clique (set to 3 if strictly 3 neighbors required)
MIN_CLUSTER_CHANNELS   = 3  # Overall cluster must span at least 3 distinct channels

# Colorbar limits
POWER_LIMITS = (-1.5, 1.5)  # dB limits for spectrograms
STAT_LIMITS  = (-2.5, 2.5)  # t-value limits for stat maps

# ══════════════════════════════════════════════════════════════════
# 1. LOAD SENSOR INFO & ALIGNMENT
# ══════════════════════════════════════════════════════════════════
print("Loading canonical sensor layout and computing spatial adjacency...")
ref_sub = 4
ref_file = os.path.join(eeg_path, f"CDmem_{ref_sub:04d}-epo.fif")
if not os.path.exists(ref_file):
    for s in plist:
        candidate = os.path.join(eeg_path, f"CDmem_{s:04d}-epo.fif")
        if os.path.exists(candidate):
            ref_file = candidate
            break

info = mne.read_epochs(ref_file, preload=False, verbose=False).info
canonical_ch_names = info.ch_names
n_channels = len(canonical_ch_names)

ch_adj, _ = mne.channels.find_ch_adjacency(info, 'eeg')
ch_adj_no_diag = ch_adj.copy()
ch_adj_no_diag.setdiag(0)
print(f"  [OK] Loaded {n_channels} canonical channels; computed spatial adjacency matrix ({ch_adj.shape[0]}x{ch_adj.shape[1]})")

def align_channels(sub_data, sub_ch_names, canonical_ch_names):
    """Align individual subject channels to canonical 65-channel layout."""
    aligned = np.full((len(canonical_ch_names), sub_data.shape[1], sub_data.shape[2]), np.nan, dtype=np.float64)
    for i, ch in enumerate(canonical_ch_names):
        if ch in sub_ch_names:
            orig_idx = sub_ch_names.index(ch)
            aligned[i] = sub_data[orig_idx]
    return aligned

# ══════════════════════════════════════════════════════════════════
# 2. LOAD PER-SUBJECT DATA
# ══════════════════════════════════════════════════════════════════
print("\nLoading per-subject Main Effect of Control data across all channels...")
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
print(f"  [OK] Found {len(valid_subs)} subjects with both conditions")

if len(valid_subs) < 2:
    print("Not enough complete subjects to run contrasts. Exiting.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════
# 3. BUILD 3D DATA ARRAYS (Channels x Frequencies x Times)
# ══════════════════════════════════════════════════════════════════
print("\nAligning 3D arrays across all 65 channels...")
high_all = np.array([
    align_channels(subject_data[sub]['high_control'], subject_ch_names[sub], canonical_ch_names)
    for sub in valid_subs
])
low_all = np.array([
    align_channels(subject_data[sub]['low_control'], subject_ch_names[sub], canonical_ch_names)
    for sub in valid_subs
])

# Extract test frequency range
high_test = high_all[:, :, test_f_inds, :]
low_test  = low_all[:, :, test_f_inds, :]
X_diff_test = high_test - low_test
X_diff_test = np.nan_to_num(X_diff_test, nan=0.0)

# Grand averages across subjects
ga_high_all = np.nanmean(high_all, axis=0)  # (65, n_freqs, n_times)
ga_low_all  = np.nanmean(low_all, axis=0)   # (65, n_freqs, n_times)
ga_diff_all = ga_high_all - ga_low_all      # (65, n_freqs, n_times)

n_subs, n_chs, n_freqs, n_times = X_diff_test.shape
df = n_subs - 1
t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)
print(f"  3D Data shape: {n_subs} subjects x {n_chs} channels x {n_freqs} frequencies x {n_times} time points")
print(f"  Cluster threshold t = +/-{t_threshold:.3f} (df={df}, alpha={CLUSTER_ALPHA})")

# ══════════════════════════════════════════════════════════════════
# 4. CONSTRUCT 3D ADJACENCY MATRIX & SPATIAL PRUNING STAT FUN
# ══════════════════════════════════════════════════════════════════
print("\nCombining spatial (channels), spectral (freqs), and temporal (times) adjacency...")
adj_3d = combine_adjacency(ch_adj, n_freqs, n_times)
print(f"  [OK] 3D Adjacency shape: {adj_3d.shape[0]} x {adj_3d.shape[1]}")

def make_spatial_pruned_stat_fun(n_chs, n_freqs, n_times, t_thresh, ch_adj_no_diag, min_neighbors):
    """
    Computes 1-sample t-statistics across subjects, and prunes any sample where
    fewer than `min_neighbors` adjacent spatial neighbors simultaneously exceed
    the univariate t-threshold (mirroring FieldTrip's cfg.minnbchan).
    """
    def stat_fun_pruned(X_in):
        var = np.var(X_in, axis=0, ddof=1)
        t = np.mean(X_in, axis=0) / np.sqrt(var / X_in.shape[0])
        t_3d = t.reshape(n_chs, n_freqs, n_times)
        
        pos = t_3d >= t_thresh
        neg = t_3d <= -t_thresh
        
        pos_p = pos & ((ch_adj_no_diag @ pos.reshape(n_chs, -1)).reshape(pos.shape) >= min_neighbors)
        neg_p = neg & ((ch_adj_no_diag @ neg.reshape(n_chs, -1)).reshape(neg.shape) >= min_neighbors)
        
        t_pruned = np.where(pos_p | neg_p, t_3d, 0.0)
        return t_pruned.ravel()
    return stat_fun_pruned

stat_fun = make_spatial_pruned_stat_fun(n_chs, n_freqs, n_times, t_threshold, ch_adj_no_diag, MIN_ADJACENT_NEIGHBORS)

# ══════════════════════════════════════════════════════════════════
# 5. PRE-CHECK CANDIDATE CLUSTERS IN OBSERVED DATA
# ══════════════════════════════════════════════════════════════════
print(f"\nEvaluating candidate clusters in observed data with spatial pruning (>= {MIN_ADJACENT_NEIGHBORS} adjacent neighbors)...")
t_obs_pruned = stat_fun(X_diff_test)
initial_clusters, initial_stats = cl._find_clusters(
    t_obs_pruned, threshold=t_threshold, tail=TAIL, adjacency=adj_3d
)

valid_initial = []
for c_idx in initial_clusters:
    c_mask = np.zeros((n_chs, n_freqs, n_times), dtype=bool)
    c_mask.ravel()[c_idx] = True
    n_ch = np.sum(np.any(c_mask, axis=(1, 2)))
    if n_ch >= MIN_CLUSTER_CHANNELS:
        valid_initial.append((c_idx, n_ch))

print(f"  Total spatially-pruned initial clusters formed: {len(initial_clusters)}")
print(f"  Clusters meeting >= {MIN_CLUSTER_CHANNELS} channels: {len(valid_initial)}")
print(f"  Excluded {len(initial_clusters) - len(valid_initial)} clusters (spanned only 1 or 2 channels)")

report_lines = []
report_lines.append("MAIN EFFECT OF CONTROL: FULL 3D CLUSTER PERMUTATION TEST RESULTS")
report_lines.append("=" * 75)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"N subjects: {n_subs} (df={df})")
report_lines.append(f"Channels: {n_chs} (whole scalp, Delaunay adjacency)")
report_lines.append(f"Spatial neighbor constraint (FieldTrip minnbchan): >= {MIN_ADJACENT_NEIGHBORS} adjacent neighbors")
report_lines.append(f"Minimum cluster channels required: {MIN_CLUSTER_CHANNELS} (1- and 2-channel clusters excluded)")
report_lines.append(f"Frequency range: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz ({n_freqs} bins)")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s ({n_times} bins)")
report_lines.append(f"Total 3D data bins: {n_chs * n_freqs * n_times:,}")
report_lines.append(f"Cluster inclusion threshold: t = +/-{t_threshold:.3f}")
report_lines.append("=" * 75)

if len(valid_initial) == 0:
    msg_skip = (f"\n[INFO] No candidate clusters found with spatial pruning and >= {MIN_CLUSTER_CHANNELS} participating channels. "
                f"Skipping permutation test.")
    print(msg_skip)
    report_lines.append(msg_skip)
    report_file = os.path.join(figures_path, 'TFR_permutation_HL_full3D_statistics_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"* Statistical report saved to {report_file}")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════
# 6. RUN 3D CLUSTER PERMUTATION TEST WITH SPATIAL PRUNING
# ══════════════════════════════════════════════════════════════════
print(f"\nRunning 3D cluster permutation test ({N_PERMUTATIONS} permutations with spatial pruning)...")
contrast_name = "Main_Effect_Control_full3D"

T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
    X_diff_test,
    threshold=t_threshold,
    adjacency=adj_3d,
    n_permutations=N_PERMUTATIONS,
    tail=TAIL,
    seed=SEED,
    stat_fun=stat_fun,
    buffer_size=None,
    n_jobs=1,
    out_type='mask',
    verbose=True
)

t_raw = mne.stats.ttest_1samp_no_p(X_diff_test)

# Filter clusters: exclude single-channel and 2-channel clusters
valid_cluster_indices = [
    idx for idx, c in enumerate(clusters) 
    if np.sum(np.any(c, axis=(1, 2))) >= MIN_CLUSTER_CHANNELS
]
sig_clusters = [idx for idx in valid_cluster_indices if cluster_p[idx] < 0.05]
n_sig = len(sig_clusters)

msg_header = (f"\nMAIN EFFECT CONTROL (HIGH VS LOW) FULL 3D PERMUTATION"
              f"\n{len(valid_cluster_indices)} valid clusters with >= {MIN_CLUSTER_CHANNELS} channels found "
              f"({len(clusters) - len(valid_cluster_indices)} 1-2 channel clusters excluded), "
              f"{n_sig} significant (p < 0.05)")
print(msg_header)
report_lines.append(msg_header)
report_lines.append("-" * 75)

star_kwargs = dict(marker='*', markerfacecolor='yellow', markeredgecolor='black', markersize=13)

# ══════════════════════════════════════════════════════════════════
# 7. PARSE CLUSTERS & GENERATE VISUALIZATIONS (IF SIGNIFICANT)
# ══════════════════════════════════════════════════════════════════
time_edges = np.concatenate([times - np.diff(times[:2])[0]/2, [times[-1] + np.diff(times[:2])[0]/2]])
freq_edges_plot = np.concatenate([freqs_plot - np.diff(freqs_plot[:2])[0]/2, [freqs_plot[-1] + np.diff(freqs_plot[:2])[0]/2]])
freq_edges_test = np.concatenate([freqs_test - np.diff(freqs_test[:2])[0]/2, [freqs_test[-1] + np.diff(freqs_test[:2])[0]/2]])

cluster_rank = 0
for i in valid_cluster_indices:
    mask = clusters[i]
    pval = cluster_p[i]
    cluster_rank += 1
    # mask has shape: (n_channels, n_freqs, n_times)
    ch_in   = np.any(mask, axis=(1, 2))
    freq_in = np.any(mask, axis=(0, 2))
    time_in = np.any(mask, axis=(0, 1))

    sig_channels = [canonical_ch_names[idx] for idx in np.where(ch_in)[0]]
    t_start = times[np.where(time_in)[0][0]]
    t_end   = times[np.where(time_in)[0][-1]]
    f_low   = freqs_test[np.where(freq_in)[0][0]]
    f_high  = freqs_test[np.where(freq_in)[0][-1]]

    mean_t = t_raw[mask].mean()
    cluster_mass = np.sum(t_raw[mask])

    if pval < 0.05:
        sig_marker = " ** SIGNIFICANT"
    else:
        sig_marker = ""

    msg = (f"  Cluster {cluster_rank} (raw #{i+1}): p={pval:.4f}{sig_marker} | Mass: {cluster_mass:.1f} | Mean T: {mean_t:.2f}\n"
           f"    Time: {t_start:.3f}-{t_end:.3f} s | Freq: {f_low:.1f}-{f_high:.1f} Hz\n"
           f"    Participating Channels ({len(sig_channels)}): {', '.join(sig_channels)}")
    print(msg)
    report_lines.append(msg)

    # --- IF AND ONLY IF SIGNIFICANT: GENERATE TOPOPLOTS & TFR SPECTROGRAM ---
    if pval < 0.05:
        print(f"\n  >>> Generating 3D Cluster {cluster_rank} figures (p = {pval:.4f})...")
        c_time_mask = (times >= t_start) & (times <= t_end)
        c_freq_mask = (freqs_plot >= f_low) & (freqs_plot <= f_high)

        # Average over cluster time and frequency across all channels
        topo_high = np.nanmean(ga_high_all[:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
        topo_low  = np.nanmean(ga_low_all[:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
        topo_diff = np.nanmean(ga_diff_all[:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))

        roi_mask = ch_in  # The participating cluster channels are the mask!

        v_cond = max(0.5, float(np.nanpercentile(np.abs(np.concatenate([topo_high, topo_low])), 99)))
        v_cond = round(v_cond, 2)
        v_diff = max(0.5, float(np.nanpercentile(np.abs(topo_diff), 99)))
        v_diff = round(v_diff, 2)

        ch_summary = f"{len(sig_channels)} channels" if len(sig_channels) > 5 else ", ".join(sig_channels)

        # 1) Topoplot: High Control
        fig_h, ax_h = plt.subplots(figsize=(6, 6))
        im_h, _ = mne.viz.plot_topomap(topo_high, info, axes=ax_h, mask=roi_mask, mask_params=star_kwargs,
                                       cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
        cb_h = plt.colorbar(im_h, ax=ax_h, orientation='horizontal', pad=0.08, shrink=0.7)
        cb_h.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
        ax_h.set_title(f"High Control\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = Cluster Channels: {ch_summary})",
                       fontsize=12, fontname='Times New Roman', pad=12)
        plt.tight_layout()
        fname_h = f"03_TFR_topo_{contrast_name}_cluster_{cluster_rank}_high_control.png"
        fig_h.savefig(os.path.join(figures_path, fname_h), dpi=300)
        plt.close(fig_h)
        print(f"    [OK] Saved: {fname_h}")

        # 2) Topoplot: Low Control
        fig_l, ax_l = plt.subplots(figsize=(6, 6))
        im_l, _ = mne.viz.plot_topomap(topo_low, info, axes=ax_l, mask=roi_mask, mask_params=star_kwargs,
                                       cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
        cb_l = plt.colorbar(im_l, ax=ax_l, orientation='horizontal', pad=0.08, shrink=0.7)
        cb_l.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
        ax_l.set_title(f"Low Control\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = Cluster Channels: {ch_summary})",
                       fontsize=12, fontname='Times New Roman', pad=12)
        plt.tight_layout()
        fname_l = f"03_TFR_topo_{contrast_name}_cluster_{cluster_rank}_low_control.png"
        fig_l.savefig(os.path.join(figures_path, fname_l), dpi=300)
        plt.close(fig_l)
        print(f"    [OK] Saved: {fname_l}")

        # 3) Topoplot: Difference
        fig_d, ax_d = plt.subplots(figsize=(6, 6))
        im_d, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax_d, mask=roi_mask, mask_params=star_kwargs,
                                       cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
        cb_d = plt.colorbar(im_d, ax=ax_d, orientation='horizontal', pad=0.08, shrink=0.7)
        cb_d.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=12)
        ax_d.set_title(f"Difference (High - Low Control)\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\np = {pval:.4f} (* = Cluster Channels: {ch_summary})",
                       fontsize=12, fontname='Times New Roman', pad=12)
        plt.tight_layout()
        fname_d = f"03_TFR_topo_{contrast_name}_cluster_{cluster_rank}_diff.png"
        fig_d.savefig(os.path.join(figures_path, fname_d), dpi=300)
        plt.close(fig_d)
        print(f"    [OK] Saved: {fname_d}")

        # 4) Combined 3-Panel Topoplot
        fig_p, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        im1, _ = mne.viz.plot_topomap(topo_high, info, axes=ax1, mask=roi_mask, mask_params=star_kwargs,
                                      cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
        cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.7)
        cb1.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
        ax1.set_title("High Control", fontsize=14, fontname='Times New Roman')

        im2, _ = mne.viz.plot_topomap(topo_low, info, axes=ax2, mask=roi_mask, mask_params=star_kwargs,
                                      cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
        cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.7)
        cb2.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
        ax2.set_title("Low Control", fontsize=14, fontname='Times New Roman')

        im3, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax3, mask=roi_mask, mask_params=star_kwargs,
                                      cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
        cb3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.7)
        cb3.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=11)
        ax3.set_title(f"Difference (High - Low)\np = {pval:.4f}", fontsize=14, fontname='Times New Roman')

        fig_p.suptitle(f"3D Cluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s (* = {ch_summary})",
                       fontsize=15, fontname='Times New Roman', y=0.98)
        plt.tight_layout()
        fname_p = f"03_TFR_topo_{contrast_name}_cluster_{cluster_rank}_panel.png"
        fig_p.savefig(os.path.join(figures_path, fname_p), dpi=300)
        plt.close(fig_p)
        print(f"    [OK] Saved 3-panel topoplot: {fname_p}")

        # 5) Cluster-Averaged TFR Spectrogram
        # Average spectrogram over the cluster's significant channels
        sig_ch_indices = np.where(ch_in)[0]
        cluster_tfr_diff = np.nanmean(ga_diff_all[sig_ch_indices, :, :], axis=0)  # (n_freqs, n_times)
        # Cluster mask collapsed over participating channels
        cluster_tf_mask = np.any(mask[sig_ch_indices, :, :], axis=0)  # (n_freqs_test, n_times)

        fig_tfr, ax_tfr = plt.subplots(figsize=(10, 6))
        im_tfr = ax_tfr.pcolormesh(time_edges, freq_edges_plot, cluster_tfr_diff, cmap='RdBu_r',
                                   vmin=POWER_LIMITS[0], vmax=POWER_LIMITS[1], shading='flat')
        if cluster_tf_mask.any():
            ax_tfr.contour(times, freqs_test, cluster_tf_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        cb_tfr = fig_tfr.colorbar(im_tfr, ax=ax_tfr, label='Power Difference (dB)')
        ax_tfr.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
        ax_tfr.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
        ax_tfr.set_title(f"CLUSTER {cluster_rank} SPECTROGRAM (Averaged over {len(sig_channels)} participating electrodes)\np = {pval:.4f}",
                         fontsize=14, fontname='Times New Roman')
        ax_tfr.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        fname_tfr = f"03_TFR_spectrogram_{contrast_name}_cluster_{cluster_rank}.png"
        fig_tfr.savefig(os.path.join(figures_path, fname_tfr), dpi=300)
        plt.close(fig_tfr)
        print(f"    [OK] Saved cluster TFR spectrogram: {fname_tfr}")

report_file = os.path.join(figures_path, 'TFR_permutation_HL_full3D_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Full 3D Statistical report saved to {report_file}")
print("Done.")
