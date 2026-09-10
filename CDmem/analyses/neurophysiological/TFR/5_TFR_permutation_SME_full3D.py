"""
5_TFR_permutation_SME_full3D.py
===============================
Subsequent Memory Effects (SME) Full 3D Spatio-Temporal-Spectral Permutation Tests
----------------------------------------------------------------------------------

PURPOSE:
    Runs hypothesis-free, whole-brain, broadband 3D cluster-based permutation tests
    across Channels x Frequencies x Times (no a priori ROI electrode or band pre-selection)
    for subsequent memory effects and interactions with control.

    Adjacency:
      - Spatial adjacency: Delaunay triangulation across 65 scalp EEG electrodes
      - Spectral adjacency: Regular 1D lattice along frequency bins (2 - 40 Hz)
      - Temporal adjacency: Regular 1D lattice along time points (0 - 3 s)
      Combined via mne.stats.combine_adjacency(ch_adj, n_freqs, n_times).

    Contrasts Evaluated:
      A. Main Effect of Memory (ControlledOnly, AllItems)
      B. Simple Effects of Memory within Low and High Control (AllItems)
      C. Interaction: Memory x Control (AllItems)
      D. Simple Effects of Memory within Low and High Control (ControlledOnly)
      E. Interaction: Memory x Control (ControlledOnly)

    Outputs (generated IF AND ONLY IF a cluster is significant, p < 0.05):
      1. Condition Topoplot: Condition 1 (participating cluster electrodes starred)
      2. Condition Topoplot: Condition 2 (participating cluster electrodes starred)
      3. Difference Topoplot: Condition 1 - Condition 2 (participating cluster electrodes starred)
      4. Combined 3-Panel Figure: [Condition 1 | Condition 2 | Difference]
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
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked_SME_full3D"
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
# 2. LOAD DATA
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

valid_subs_collapsed = {}
for ft_key in main_effect_files:
    valid_subs_collapsed[ft_key] = [sub for sub in subject_data_collapsed[ft_key] 
                                     if all(c in subject_data_collapsed[ft_key][sub] for c in collapsed_conditions)]
    print(f"  [Main Effect - {ft_key}] Found {len(valid_subs_collapsed[ft_key])} complete subjects")

valid_subs_4cell = [sub for sub in subject_data_4cell if all(c in subject_data_4cell[sub] for c in four_cell_conditions)]
print(f"  [4-cell AllItems] Found {len(valid_subs_4cell)} complete subjects")

valid_subs_4cell_ctrl = [sub for sub in subject_data_4cell_ctrl if all(c in subject_data_4cell_ctrl[sub] for c in four_cell_conditions)]
print(f"  [4-cell ControlledOnly] Found {len(valid_subs_4cell_ctrl)} complete subjects")

# ══════════════════════════════════════════════════════════════════
# 3. BUILD 3D CONTRASTS ACROSS ALL CHANNELS
# ══════════════════════════════════════════════════════════════════
print("\nBuilding 3D contrast arrays across all 65 channels...")
contrasts_3d = {}
contrast_conditions_all = {}
descriptions = {}

# A. Collapsed Main Effects
for ft_key in main_effect_files:
    subs = valid_subs_collapsed[ft_key]
    if len(subs) < 2:
        continue
    rec_all = np.array([align_channels(subject_data_collapsed[ft_key][s]['recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    not_all = np.array([align_channels(subject_data_collapsed[ft_key][s]['not_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    diff_all = rec_all - not_all
    name = f"Main_Effect_Memory_{ft_key}"
    contrasts_3d[name] = diff_all[:, :, test_f_inds, :]
    descriptions[name] = f"(Recalled vs Not Recalled - {ft_key})"
    contrast_conditions_all[name] = {
        'cond1_name': 'Recalled',
        'cond2_name': 'Not_Recalled',
        'cond1_ga': np.nanmean(rec_all, axis=0),
        'cond2_ga': np.nanmean(not_all, axis=0),
        'diff_ga':  np.nanmean(diff_all, axis=0)
    }

# B. 4-cell AllItems Simple Effects & Interaction
if len(valid_subs_4cell) >= 2:
    subs = valid_subs_4cell
    l_rec = np.array([align_channels(subject_data_4cell[s]['low_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    l_not = np.array([align_channels(subject_data_4cell[s]['low_not_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    h_rec = np.array([align_channels(subject_data_4cell[s]['high_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    h_not = np.array([align_channels(subject_data_4cell[s]['high_not_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])

    l_diff = l_rec - l_not
    h_diff = h_rec - h_not
    inter  = l_diff - h_diff

    # Simple Effect Low
    name = "Simple_Effect_Low_Control_AllItems"
    contrasts_3d[name] = l_diff[:, :, test_f_inds, :]
    descriptions[name] = "(Recalled vs Not Recalled WITHIN Low Control - All Items)"
    contrast_conditions_all[name] = {
        'cond1_name': 'Low_Control_Recalled',
        'cond2_name': 'Low_Control_Not_Recalled',
        'cond1_ga': np.nanmean(l_rec, axis=0),
        'cond2_ga': np.nanmean(l_not, axis=0),
        'diff_ga':  np.nanmean(l_diff, axis=0)
    }

    # Simple Effect High
    name = "Simple_Effect_High_Control_AllItems"
    contrasts_3d[name] = h_diff[:, :, test_f_inds, :]
    descriptions[name] = "(Recalled vs Not Recalled WITHIN High Control - All Items)"
    contrast_conditions_all[name] = {
        'cond1_name': 'High_Control_Recalled',
        'cond2_name': 'High_Control_Not_Recalled',
        'cond1_ga': np.nanmean(h_rec, axis=0),
        'cond2_ga': np.nanmean(h_not, axis=0),
        'diff_ga':  np.nanmean(h_diff, axis=0)
    }

    # Interaction
    name = "Interaction_Memory_x_Control_AllItems"
    contrasts_3d[name] = inter[:, :, test_f_inds, :]
    descriptions[name] = "(Memory Effect: Low Control vs High Control - All Items)"
    contrast_conditions_all[name] = {
        'cond1_name': 'Memory_Effect_Low_Control',
        'cond2_name': 'Memory_Effect_High_Control',
        'cond1_ga': np.nanmean(l_diff, axis=0),
        'cond2_ga': np.nanmean(h_diff, axis=0),
        'diff_ga':  np.nanmean(inter, axis=0)
    }

# C. 4-cell ControlledOnly Simple Effects & Interaction
if len(valid_subs_4cell_ctrl) >= 2:
    subs = valid_subs_4cell_ctrl
    l_rec_c = np.array([align_channels(subject_data_4cell_ctrl[s]['low_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    l_not_c = np.array([align_channels(subject_data_4cell_ctrl[s]['low_not_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    h_rec_c = np.array([align_channels(subject_data_4cell_ctrl[s]['high_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])
    h_not_c = np.array([align_channels(subject_data_4cell_ctrl[s]['high_not_recalled'], subject_ch_names[s], canonical_ch_names) for s in subs])

    l_diff_c = l_rec_c - l_not_c
    h_diff_c = h_rec_c - h_not_c
    inter_c  = l_diff_c - h_diff_c

    # Simple Effect Low
    name = "Simple_Effect_Low_Control_CtrlOnly"
    contrasts_3d[name] = l_diff_c[:, :, test_f_inds, :]
    descriptions[name] = "(Recalled vs Not Recalled WITHIN Low Control - Controlled items only)"
    contrast_conditions_all[name] = {
        'cond1_name': 'Low_Control_Recalled',
        'cond2_name': 'Low_Control_Not_Recalled',
        'cond1_ga': np.nanmean(l_rec_c, axis=0),
        'cond2_ga': np.nanmean(l_not_c, axis=0),
        'diff_ga':  np.nanmean(l_diff_c, axis=0)
    }

    # Simple Effect High
    name = "Simple_Effect_High_Control_CtrlOnly"
    contrasts_3d[name] = h_diff_c[:, :, test_f_inds, :]
    descriptions[name] = "(Recalled vs Not Recalled WITHIN High Control - Controlled items only)"
    contrast_conditions_all[name] = {
        'cond1_name': 'High_Control_Recalled',
        'cond2_name': 'High_Control_Not_Recalled',
        'cond1_ga': np.nanmean(h_rec_c, axis=0),
        'cond2_ga': np.nanmean(h_not_c, axis=0),
        'diff_ga':  np.nanmean(h_diff_c, axis=0)
    }

    # Interaction
    name = "Interaction_Memory_x_Control_CtrlOnly"
    contrasts_3d[name] = inter_c[:, :, test_f_inds, :]
    descriptions[name] = "(Memory Effect: Low Control vs High Control - Controlled items only)"
    contrast_conditions_all[name] = {
        'cond1_name': 'Memory_Effect_Low_Control',
        'cond2_name': 'Memory_Effect_High_Control',
        'cond1_ga': np.nanmean(l_diff_c, axis=0),
        'cond2_ga': np.nanmean(h_diff_c, axis=0),
        'diff_ga':  np.nanmean(inter_c, axis=0)
    }

# ══════════════════════════════════════════════════════════════════
# 4. CONSTRUCT 3D ADJACENCY MATRIX & SPATIAL PRUNING STAT FUN
# ══════════════════════════════════════════════════════════════════
n_chs = n_channels
n_freqs = len(freqs_test)
n_times = len(times)
print(f"\nCombining spatial ({n_chs} chs), spectral ({n_freqs} freqs), and temporal ({n_times} times) adjacency...")
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

# ══════════════════════════════════════════════════════════════════
# 5. RUN 3D PERMUTATION TESTS ACROSS ALL CONTRASTS
# ══════════════════════════════════════════════════════════════════
report_lines = []
report_lines.append("SUBSEQUENT MEMORY EFFECTS (SME): FULL 3D CLUSTER PERMUTATION TEST RESULTS")
report_lines.append("=" * 75)
report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
report_lines.append(f"Channels: {n_chs} (whole scalp, Delaunay spatial adjacency)")
report_lines.append(f"Spatial neighbor constraint (FieldTrip minnbchan): >= {MIN_ADJACENT_NEIGHBORS} adjacent neighbors")
report_lines.append(f"Frequency range: {TEST_FREQ[0]} - {TEST_FREQ[1]} Hz ({n_freqs} bins)")
report_lines.append(f"Time window: {TEST_TIME[0]} - {TEST_TIME[1]} s ({n_times} bins)")
report_lines.append(f"Total 3D data bins per subject: {n_chs * n_freqs * n_times:,}")
report_lines.append(f"Minimum cluster channels required: {MIN_CLUSTER_CHANNELS} (1- and 2-channel clusters excluded)")
report_lines.append("=" * 75)

time_edges = np.concatenate([times - np.diff(times[:2])[0]/2, [times[-1] + np.diff(times[:2])[0]/2]])
freq_edges_plot = np.concatenate([freqs_plot - np.diff(freqs_plot[:2])[0]/2, [freqs_plot[-1] + np.diff(freqs_plot[:2])[0]/2]])
freq_edges_test = np.concatenate([freqs_test - np.diff(freqs_test[:2])[0]/2, [freqs_test[-1] + np.diff(freqs_test[:2])[0]/2]])
star_kwargs = dict(marker='*', markerfacecolor='yellow', markeredgecolor='black', markersize=13)

for comp_name, X_diff_test in contrasts_3d.items():
    X_diff_test = np.nan_to_num(X_diff_test, nan=0.0)
    n_subs = X_diff_test.shape[0]
    df = n_subs - 1
    t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)

    print(f"\n{'='*70}")
    print(f"  CONTRAST: {comp_name}")
    print(f"  N={n_subs}, df={df}, cluster threshold t = +/-{t_threshold:.3f}")
    print(f"{'='*70}")

    stat_fun = make_spatial_pruned_stat_fun(n_chs, n_freqs, n_times, t_threshold, ch_adj_no_diag, MIN_ADJACENT_NEIGHBORS)

    # Pre-check candidate clusters in observed data with spatial pruning
    print(f"  Pre-checking candidate clusters with spatial pruning (>= {MIN_ADJACENT_NEIGHBORS} adjacent neighbors)...")
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

    desc = descriptions.get(comp_name, '')
    if len(valid_initial) == 0:
        msg_skip = (f"  [SKIP] No candidate clusters with spatial pruning and >= {MIN_CLUSTER_CHANNELS} channels for {comp_name}. "
                    f"Skipping {N_PERMUTATIONS} permutations.")
        print(msg_skip)
        report_lines.append(f"\n{comp_name.upper()} {desc}")
        report_lines.append(f"N={n_subs}, df={df}, cluster threshold=+/-{t_threshold:.3f}")
        report_lines.append(f"0 valid clusters found with >= {MIN_CLUSTER_CHANNELS} channels. Permutations skipped.")
        report_lines.append("-" * 75)
        continue

    print(f"\n  Running 3D permutation test ({N_PERMUTATIONS} permutations with spatial pruning)...")
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

    msg_header = (f"\n{comp_name.upper()} {desc}"
                  f"\nN={n_subs}, df={df}, cluster threshold=+/-{t_threshold:.3f}"
                  f"\n{len(valid_cluster_indices)} valid clusters with >= {MIN_CLUSTER_CHANNELS} channels found "
                  f"({len(clusters) - len(valid_cluster_indices)} 1-2 channel clusters excluded), "
                  f"{n_sig} significant (p < 0.05)")
    print(msg_header)
    report_lines.append(msg_header)
    report_lines.append("-" * 75)

    c_info = contrast_conditions_all.get(comp_name, None)

    cluster_rank = 0
    for i in valid_cluster_indices:
        mask = clusters[i]
        pval = cluster_p[i]
        cluster_rank += 1

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
        if pval < 0.05 and c_info is not None:
            print(f"\n    >>> Generating 3D Cluster {cluster_rank} figures for {comp_name} (p = {pval:.4f})...")
            c_time_mask = (times >= t_start) & (times <= t_end)
            c_freq_mask = (freqs_plot >= f_low) & (freqs_plot <= f_high)

            cond1_name = c_info['cond1_name']
            cond2_name = c_info['cond2_name']
            topo_c1 = np.nanmean(c_info['cond1_ga'][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
            topo_c2 = np.nanmean(c_info['cond2_ga'][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
            topo_diff = np.nanmean(c_info['diff_ga'][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))

            roi_mask = ch_in

            v_cond = max(0.5, float(np.nanpercentile(np.abs(np.concatenate([topo_c1, topo_c2])), 99)))
            v_cond = round(v_cond, 2)
            v_diff = max(0.5, float(np.nanpercentile(np.abs(topo_diff), 99)))
            v_diff = round(v_diff, 2)

            ch_summary = f"{len(sig_channels)} channels" if len(sig_channels) > 5 else ", ".join(sig_channels)

            # 1) Condition 1 Topoplot
            fig_1, ax_1 = plt.subplots(figsize=(6, 6))
            im_1, _ = mne.viz.plot_topomap(topo_c1, info, axes=ax_1, mask=roi_mask, mask_params=star_kwargs,
                                           cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
            cb_1 = plt.colorbar(im_1, ax=ax_1, orientation='horizontal', pad=0.08, shrink=0.7)
            cb_1.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
            ax_1.set_title(f"{cond1_name.replace('_', ' ')}\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = {ch_summary})",
                           fontsize=12, fontname='Times New Roman', pad=12)
            plt.tight_layout()
            fname_1 = f"03_TFR_topo_{comp_name}_cluster_{cluster_rank}_{cond1_name.lower()}.png"
            fig_1.savefig(os.path.join(figures_path, fname_1), dpi=300)
            plt.close(fig_1)
            print(f"      [OK] Saved: {fname_1}")

            # 2) Condition 2 Topoplot
            fig_2, ax_2 = plt.subplots(figsize=(6, 6))
            im_2, _ = mne.viz.plot_topomap(topo_c2, info, axes=ax_2, mask=roi_mask, mask_params=star_kwargs,
                                           cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
            cb_2 = plt.colorbar(im_2, ax=ax_2, orientation='horizontal', pad=0.08, shrink=0.7)
            cb_2.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
            ax_2.set_title(f"{cond2_name.replace('_', ' ')}\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = {ch_summary})",
                           fontsize=12, fontname='Times New Roman', pad=12)
            plt.tight_layout()
            fname_2 = f"03_TFR_topo_{comp_name}_cluster_{cluster_rank}_{cond2_name.lower()}.png"
            fig_2.savefig(os.path.join(figures_path, fname_2), dpi=300)
            plt.close(fig_2)
            print(f"      [OK] Saved: {fname_2}")

            # 3) Difference Topoplot
            fig_d, ax_d = plt.subplots(figsize=(6, 6))
            im_d, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax_d, mask=roi_mask, mask_params=star_kwargs,
                                           cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
            cb_d = plt.colorbar(im_d, ax=ax_d, orientation='horizontal', pad=0.08, shrink=0.7)
            cb_d.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=12)
            ax_d.set_title(f"Difference ({cond1_name.replace('_', ' ')} - {cond2_name.replace('_', ' ')})\nCluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\np = {pval:.4f} (* = {ch_summary})",
                           fontsize=12, fontname='Times New Roman', pad=12)
            plt.tight_layout()
            fname_d = f"03_TFR_topo_{comp_name}_cluster_{cluster_rank}_diff.png"
            fig_d.savefig(os.path.join(figures_path, fname_d), dpi=300)
            plt.close(fig_d)
            print(f"      [OK] Saved: {fname_d}")

            # 4) Combined 3-Panel Topoplot
            fig_p, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            im1, _ = mne.viz.plot_topomap(topo_c1, info, axes=ax1, mask=roi_mask, mask_params=star_kwargs,
                                          cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
            cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.7)
            cb1.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
            ax1.set_title(cond1_name.replace('_', ' '), fontsize=14, fontname='Times New Roman')

            im2, _ = mne.viz.plot_topomap(topo_c2, info, axes=ax2, mask=roi_mask, mask_params=star_kwargs,
                                          cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
            cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.7)
            cb2.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
            ax2.set_title(cond2_name.replace('_', ' '), fontsize=14, fontname='Times New Roman')

            im3, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax3, mask=roi_mask, mask_params=star_kwargs,
                                          cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
            cb3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.7)
            cb3.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=11)
            ax3.set_title(f"Difference\np = {pval:.4f}", fontsize=14, fontname='Times New Roman')

            fig_p.suptitle(f"{comp_name.replace('_', ' ')} - 3D Cluster {cluster_rank}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s (* = {ch_summary})",
                           fontsize=15, fontname='Times New Roman', y=0.98)
            plt.tight_layout()
            fname_p = f"03_TFR_topo_{comp_name}_cluster_{cluster_rank}_panel.png"
            fig_p.savefig(os.path.join(figures_path, fname_p), dpi=300)
            plt.close(fig_p)
            print(f"      [OK] Saved 3-panel topoplot: {fname_p}")

            # 5) Cluster-Averaged TFR Spectrogram
            sig_ch_indices = np.where(ch_in)[0]
            cluster_tfr_diff = np.nanmean(c_info['diff_ga'][sig_ch_indices, :, :], axis=0)
            cluster_tf_mask = np.any(mask[sig_ch_indices, :, :], axis=0)

            fig_tfr, ax_tfr = plt.subplots(figsize=(10, 6))
            im_tfr = ax_tfr.pcolormesh(time_edges, freq_edges_plot, cluster_tfr_diff, cmap='RdBu_r',
                                       vmin=POWER_LIMITS[0], vmax=POWER_LIMITS[1], shading='flat')
            if cluster_tf_mask.any():
                ax_tfr.contour(times, freqs_test, cluster_tf_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
            cb_tfr = fig_tfr.colorbar(im_tfr, ax=ax_tfr, label='Power Difference (dB)')
            ax_tfr.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
            ax_tfr.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
            ax_tfr.set_title(f"{comp_name.replace('_', ' ')} - 3D CLUSTER {cluster_rank} SPECTROGRAM\n(Averaged over {len(sig_channels)} participating electrodes, p = {pval:.4f})",
                             fontsize=13, fontname='Times New Roman')
            ax_tfr.axvline(0, color='black', linestyle='--', linewidth=1)
            plt.tight_layout()
            fname_tfr = f"03_TFR_spectrogram_{comp_name}_cluster_{cluster_rank}.png"
            fig_tfr.savefig(os.path.join(figures_path, fname_tfr), dpi=300)
            plt.close(fig_tfr)
            print(f"      [OK] Saved cluster TFR spectrogram: {fname_tfr}")

report_file = os.path.join(figures_path, 'TFR_permutation_SME_full3D_statistics_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"\n* Full 3D Statistical report saved to {report_file}")
print("Done.")
