r"""
5_TFR_permutation_SME.py   (from_15.10 version of 5_TFR_permutation_SME_PO.py)
================================================================================
Subsequent Memory Effect: TFR Permutation & Stat Maps (ROI-based)
-----------------------------------------------------------------

PURPOSE:
    Loads condition-averaged TFR data and runs a 2D cluster-based permutation
    test (Frequency x Time) on a predefined TEST frequency range.

    Generates standalone figures for:
      1. Condition Grand-Average TFRs (Power in dB relative to baseline)
      2. Difference TFR (Power difference in dB relative to baseline)
      3. Statistical t-map with significant cluster contours

    Contrast computed:
      Main Effect of Memory (collapsed across control), CONTROLLED items only:
      recalled (controlled item later recognised) vs not recalled.

WHAT CHANGED COMPARED TO THE before_15.10 VERSION (agreed on 15.10):
    0. Only the Controlled-items main effect is tested. Removed: Main Effect
       AllItems, the simple effects within Low / High control and the
       Memory x Control interactions (AllItems and ControlledOnly).
    1. dB BASELINE CORRECTION NOW HAPPENS HERE (the calculation script saves
       raw power). Switch BASELINE_MODE:
         'pooled'       -> recalled and not-recalled are divided by the SAME
                           baseline (mean of both cell baselines; equal trial
                           numbers after balancing). It cancels exactly in the
                           difference: dB_rec - dB_not = 10*log10(P_rec/P_not).
                           (CHOSEN on 15.10.)
         'percondition' -> each cell divided by its OWN baseline (old method).
    2. Switch LOCKING: 'stim' (0 = stimulus onset, as before) or 'move'
       (0 = movement onset, NEW). Baseline stays pre-STIMULUS.
    3. Switch ROI ('PO' or 'MF') instead of commenting lines in/out.
    4. 25-TRIAL RULE: participants with fewer than MIN_TRIALS trials in either
       cell (counted before balancing, for the chosen LOCKING) are excluded.
       Printed in the console and written to the report.
    5. Output folder name contains the settings:
         from_15.10\eeg5_TFR_SME_<LOCKING>_<ROI>_<BASELINE_MODE>
    6. x-axis label says what time 0 is.

WHAT DID NOT CHANGE:
    Permutation settings (1000 permutations, two-sided, seed 2025, t threshold
    at p < .05 with df = N-1), test window 0-3 s, test frequencies 2-40 Hz,
    PLOT_FREQ, all plots, colour limits, topoplots, Cohen's d, participant list.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t as t_dist

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# ── NEW SWITCHES ─────────────────────────────────────────────────
BASELINE_MODE = 'pooled'   # 'pooled' (chosen) or 'percondition' (old method)
LOCKING       = 'stim'     # 'stim' = stimulus-locked, 'move' = movement-onset-locked
ROI           = 'PO'       # 'PO' = all P*/O* channels, 'MF' = Fz/FCz/FC1/FC2 (frontocentral)
MIN_TRIALS    = 25         # 25-trial rule (per condition, before balancing)

input_path   = r"H:\PHD\control_detection\main_data\eeg\from_15.10\eeg4_TFR_SME"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
figures_path = (r"H:\PHD\control_detection\main_data\eeg\from_15.10"
                + rf"\eeg5_TFR_SME_{LOCKING}_{ROI}_{BASELINE_MODE}")
os.makedirs(figures_path, exist_ok=True)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

# x-axis label for all TFR plots (NEW: says what time 0 is)
TIME_LABEL = 'Time from stimulus onset (s)' if LOCKING == 'stim' else 'Time from movement onset (s)'

# Load canonical sensor layout for whole-brain topoplots
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

def align_channels(sub_data, sub_ch_names, canonical_ch_names):
    """Align individual subject channels to canonical 65-channel layout."""
    aligned = np.full((len(canonical_ch_names), sub_data.shape[1], sub_data.shape[2]), np.nan, dtype=np.float64)
    for i, ch in enumerate(canonical_ch_names):
        if ch in sub_ch_names:
            orig_idx = sub_ch_names.index(ch)
            aligned[i] = sub_data[orig_idx]
    return aligned

# Permutation test parameters (UNCHANGED)
N_PERMUTATIONS = 1000
TAIL = 0           # two-sided t-test
SEED = 2025
CLUSTER_ALPHA = 0.05

TEST_TIME = (0.0, 3.0)
TEST_FREQ = (2.0, 40.0)
PLOT_FREQ = (2.0, 60.0)

# Colorbar limits for standalone figures
POWER_LIMITS = (-1.5, 1.5)  # dB limits for spectrograms
STAT_LIMITS  = (-2.5, 2.5)  # t-value limits for stat maps


def apply_db_baseline(power_A, power_B, bl_A, bl_B, mode):
    """
    NEW. Convert raw condition-averaged power to dB relative to a baseline.
        power_A, power_B : raw power (channels x freqs x times)
        bl_A, bl_B       : baseline power of the same trials (channels x freqs)
        mode             : 'pooled' or 'percondition' (see header)
    Formula (same as the old baseline_correct_db): 10 * log10(power / baseline)
    """
    if mode == 'pooled':
        ref_A = ref_B = (bl_A + bl_B) / 2.0
    elif mode == 'percondition':
        ref_A, ref_B = bl_A, bl_B
    else:
        raise ValueError(f"Unknown BASELINE_MODE '{mode}' (use 'pooled' or 'percondition')")
    ref_A = np.maximum(ref_A, 1e-30)[:, :, np.newaxis]
    ref_B = np.maximum(ref_B, 1e-30)[:, :, np.newaxis]
    db_A = 10 * np.log10(np.maximum(power_A, 1e-30) / ref_A)
    db_B = 10 * np.log10(np.maximum(power_B, 1e-30) / ref_B)
    return db_A, db_B

print(f"SETTINGS: baseline = {BASELINE_MODE}, locking = {LOCKING}, ROI = {ROI}, "
      f"min. trials = {MIN_TRIALS}")
print(f"Output folder: {figures_path}")

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

# --- Collapsed main-effect file (recalled vs not_recalled) ---
# CHANGED: only ControlledOnly is left (AllItems removed)
main_effect_files = {
    'ControlledOnly': '_TFR_MainEffect_ControlledOnly.npz',
}

collapsed_conditions = ['recalled', 'not_recalled']
subject_data_collapsed = {ft: {f"{sub:04d}": {} for sub in plist} for ft in main_effect_files}
subject_ch_names = {}
trial_counts = {}          # NEW: {sub_id: (n_recalled, n_not_recalled)} before balancing
excluded_min_trials = []   # NEW: participants removed by the 25-trial rule
times, freqs_plot, freqs_test, test_f_inds = None, None, None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    for ft_key, ft_suffix in main_effect_files.items():
        data_file = os.path.join(input_path, f"CDmem_{sub_id}{ft_suffix}")

        if not os.path.exists(data_file):
            continue

        saved = np.load(data_file, allow_pickle=True)

        # NEW: the movement-locked keys are missing if the .vmrk check failed
        if f"{LOCKING}_recalled" not in saved:
            print(f"  Participant {sub}: no {LOCKING}-locked data in file - skipped")
            continue

        # NEW: 25-trial rule (counts before balancing)
        n_rec = int(saved[f"n_{LOCKING}_recalled"])
        n_not = int(saved[f"n_{LOCKING}_not_recalled"])
        trial_counts[sub_id] = (n_rec, n_not)
        if n_rec < MIN_TRIALS or n_not < MIN_TRIALS:
            excluded_min_trials.append((sub_id, n_rec, n_not))
            continue

        if sub_id not in subject_ch_names and 'ch_names' in saved:
            subject_ch_names[sub_id] = saved['ch_names'].tolist()

        if times is None:
            times_all = saved[f"times_{LOCKING}"]
            freqs_all = saved['freqs']

            time_mask = (times_all >= TEST_TIME[0] - 1e-9) & (times_all <= TEST_TIME[1] + 1e-9)
            times = times_all[time_mask]

            freq_mask_plot = (freqs_all >= PLOT_FREQ[0]) & (freqs_all <= PLOT_FREQ[1])
            freqs_plot = freqs_all[freq_mask_plot]

            test_f_inds = np.where((freqs_plot >= TEST_FREQ[0]) & (freqs_plot <= TEST_FREQ[1]))[0]
            freqs_test = freqs_plot[test_f_inds]

        # NEW: baseline correction (raw power -> dB) with the chosen BASELINE_MODE
        db_rec, db_not = apply_db_baseline(
            saved[f"{LOCKING}_recalled"].astype(np.float64),
            saved[f"{LOCKING}_not_recalled"].astype(np.float64),
            saved[f"bl_{LOCKING}_recalled"], saved[f"bl_{LOCKING}_not_recalled"],
            BASELINE_MODE)

        # From here on the data look exactly like before (dB, ch x freq x time)
        subject_data_collapsed[ft_key][sub_id]['recalled'] = db_rec[:, freq_mask_plot, :][:, :, time_mask]
        subject_data_collapsed[ft_key][sub_id]['not_recalled'] = db_not[:, freq_mask_plot, :][:, :, time_mask]

# NEW: print the 25-trial rule result
print(f"\n25-TRIAL RULE (>= {MIN_TRIALS} trials per cell, {LOCKING}-locked):")
if excluded_min_trials:
    for sid, nr, nn in excluded_min_trials:
        print(f"  EXCLUDED participant {sid}: Recalled={nr}, Not recalled={nn}")
    print(f"  -> {len(excluded_min_trials)} participant(s) excluded")
else:
    print("  No participant excluded.")

# Determine valid subjects
valid_subs_collapsed = {}
for ft_key in main_effect_files:
    valid_subs_collapsed[ft_key] = [sub for sub in subject_data_collapsed[ft_key]
                                     if all(c in subject_data_collapsed[ft_key][sub] for c in collapsed_conditions)]
    print(f"[Main Effect - {ft_key}] Found {len(valid_subs_collapsed[ft_key])} subjects with both conditions")

if all(len(v) < 2 for v in valid_subs_collapsed.values()):
    print("Not enough complete subjects to run contrasts. Exiting.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════
# 2. EXTRACT ROI DATA & BUILD CONTRASTS
# ══════════════════════════════════════════════════════════════════
# CHANGED: chosen with the ROI switch above (same two options as before)
if ROI == 'PO':
    # ROI Option 1: all channels starting with 'P' or 'O' (Parieto-Occipital)
    sample_chs = next(iter(subject_ch_names.values()))
    roi_channels = [ch for ch in sample_chs if ch.startswith(('P', 'O'))]
elif ROI == 'MF':
    # ROI Option 2: Frontocentral
    roi_channels = ['Fz', 'FCz', 'FC1', 'FC2']
else:
    raise ValueError(f"Unknown ROI '{ROI}' (use 'PO' or 'MF')")


print(f"\nAveraging over ROI channels ({len(roi_channels)}): {roi_channels}\n")

def extract_roi(data_arr, sub_id):
    chs = subject_ch_names[sub_id]
    idx = [chs.index(ch) for ch in roi_channels if ch in chs]
    if len(idx) == 0:
        raise ValueError(f"Participant {sub_id} has none of the requested ROI channels {roi_channels}!")
    return data_arr[idx, :, :].mean(axis=0)

contrasts = {}
contrasts_all_chs = {}
contrast_conditions_all_chs = {}
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

    rec_all = np.array([align_channels(subject_data_collapsed[ft_key][sub]['recalled'], subject_ch_names[sub], canonical_ch_names) for sub in subs])
    not_rec_all = np.array([align_channels(subject_data_collapsed[ft_key][sub]['not_recalled'], subject_ch_names[sub], canonical_ch_names) for sub in subs])
    ga_rec_all = np.nanmean(rec_all, axis=0)
    ga_not_all = np.nanmean(not_rec_all, axis=0)
    contrasts_all_chs[name] = ga_rec_all - ga_not_all
    contrast_conditions_all_chs[name] = {
        'cond1_name': 'Recalled',
        'cond2_name': 'Not_Recalled',
        'cond1_data': ga_rec_all,
        'cond2_data': ga_not_all
    }
    
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
    ax_cond.set_xlabel(TIME_LABEL, fontsize=14, fontname='Times New Roman')
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
    report_lines.append(f"Participants: {valid_subs_collapsed[ft_key]}")
report_lines.append(f"ROI channels: {roi_channels}")
report_lines.append(f"Baseline mode: {BASELINE_MODE} | Locking: {LOCKING} (0 = {'stimulus' if LOCKING == 'stim' else 'movement'} onset)")
report_lines.append(f"25-trial rule (>= {MIN_TRIALS} per cell): excluded {[s for s, _, _ in excluded_min_trials] or 'none'}")
for _sid, _nr, _nn in excluded_min_trials:
    report_lines.append(f"  excluded {_sid}: Recalled={_nr}, Not recalled={_nn}")
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

            # --- Generate Whole-Brain Topoplots for Significant Cluster Period ---
            if comp_name in contrasts_all_chs and comp_name in contrast_conditions_all_chs:
                c_time_mask = (times >= t_start) & (times <= t_end)
                c_freq_mask = (freqs_plot >= f_low) & (freqs_plot <= f_high)

                c_info = contrast_conditions_all_chs[comp_name]
                cond1_label = c_info['cond1_name']
                cond2_label = c_info['cond2_name']
                topo_cond1 = np.nanmean(c_info['cond1_data'][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
                topo_cond2 = np.nanmean(c_info['cond2_data'][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))
                topo_diff  = np.nanmean(contrasts_all_chs[comp_name][:, c_freq_mask, :][:, :, c_time_mask], axis=(1, 2))

                roi_mask = np.array([ch in roi_channels for ch in canonical_ch_names])
                roi_label = "/".join(roi_channels) if len(roi_channels) <= 6 else f"{len(roi_channels)} ROI channels"

                v_cond = max(0.5, float(np.nanpercentile(np.abs(np.concatenate([topo_cond1, topo_cond2])), 99)))
                v_cond = round(v_cond, 2)
                v_diff = max(0.5, float(np.nanpercentile(np.abs(topo_diff), 99)))
                v_diff = round(v_diff, 2)

                star_kwargs = dict(marker='*', markerfacecolor='yellow', markeredgecolor='black', markersize=14)

                # 1) Topoplot: Condition 1
                fig_c1, ax_c1 = plt.subplots(figsize=(6, 6))
                im_c1, _ = mne.viz.plot_topomap(topo_cond1, info, axes=ax_c1, mask=roi_mask, mask_params=star_kwargs,
                                                cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
                cb_c1 = plt.colorbar(im_c1, ax=ax_c1, orientation='horizontal', pad=0.08, shrink=0.7)
                cb_c1.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
                ax_c1.set_title(f"{cond1_label.replace('_', ' ')}\nCluster {i+1}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = ROI: {roi_label})",
                                fontsize=13, fontname='Times New Roman', pad=12)
                plt.tight_layout()
                fname_c1 = f"03_TFR_topo_{comp_name}_cluster_{i+1}_{cond1_label.lower()}.png"
                fig_c1.savefig(os.path.join(figures_path, fname_c1), dpi=300)
                plt.close(fig_c1)
                print(f"  [OK] Saved condition topoplot: {fname_c1}")

                # 2) Topoplot: Condition 2
                fig_c2, ax_c2 = plt.subplots(figsize=(6, 6))
                im_c2, _ = mne.viz.plot_topomap(topo_cond2, info, axes=ax_c2, mask=roi_mask, mask_params=star_kwargs,
                                                cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
                cb_c2 = plt.colorbar(im_c2, ax=ax_c2, orientation='horizontal', pad=0.08, shrink=0.7)
                cb_c2.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
                ax_c2.set_title(f"{cond2_label.replace('_', ' ')}\nCluster {i+1}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\n(* = ROI: {roi_label})",
                                fontsize=13, fontname='Times New Roman', pad=12)
                plt.tight_layout()
                fname_c2 = f"03_TFR_topo_{comp_name}_cluster_{i+1}_{cond2_label.lower()}.png"
                fig_c2.savefig(os.path.join(figures_path, fname_c2), dpi=300)
                plt.close(fig_c2)
                print(f"  [OK] Saved condition topoplot: {fname_c2}")

                # 3) Topoplot: Difference
                fig_d, ax_d = plt.subplots(figsize=(6, 6))
                im_d, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax_d, mask=roi_mask, mask_params=star_kwargs,
                                               cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
                cb_d = plt.colorbar(im_d, ax=ax_d, orientation='horizontal', pad=0.08, shrink=0.7)
                cb_d.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=12)
                ax_d.set_title(f"Difference ({cond1_label.replace('_', ' ')} - {cond2_label.replace('_', ' ')})\nCluster {i+1}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s\np = {pval:.4f} (* = ROI: {roi_label})",
                               fontsize=13, fontname='Times New Roman', pad=12)
                plt.tight_layout()
                fname_d = f"03_TFR_topo_{comp_name}_cluster_{i+1}_diff.png"
                fig_d.savefig(os.path.join(figures_path, fname_d), dpi=300)
                plt.close(fig_d)
                print(f"  [OK] Saved difference topoplot: {fname_d}")

                # 4) Combined 3-Panel Figure
                fig_p, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                im1, _ = mne.viz.plot_topomap(topo_cond1, info, axes=ax1, mask=roi_mask, mask_params=star_kwargs,
                                              cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
                cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.7)
                cb1.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
                ax1.set_title(cond1_label.replace('_', ' '), fontsize=14, fontname='Times New Roman')

                im2, _ = mne.viz.plot_topomap(topo_cond2, info, axes=ax2, mask=roi_mask, mask_params=star_kwargs,
                                              cmap='RdBu_r', sphere='eeglab', vlim=(-v_cond, v_cond), show=False)
                cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.7)
                cb2.set_label('Power (dB)', fontname='Times New Roman', fontsize=11)
                ax2.set_title(cond2_label.replace('_', ' '), fontsize=14, fontname='Times New Roman')

                im3, _ = mne.viz.plot_topomap(topo_diff, info, axes=ax3, mask=roi_mask, mask_params=star_kwargs,
                                              cmap='RdBu_r', sphere='eeglab', vlim=(-v_diff, v_diff), show=False)
                cb3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.7)
                cb3.set_label('Power Difference (dB)', fontname='Times New Roman', fontsize=11)
                ax3.set_title(f"Difference\np = {pval:.4f}", fontsize=14, fontname='Times New Roman')

                fig_p.suptitle(f"{comp_name.replace('_', ' ')} - Cluster {i+1}: {f_low:.1f}-{f_high:.1f} Hz, {t_start:.3f}-{t_end:.3f} s (* = ROI: {roi_label})",
                               fontsize=15, fontname='Times New Roman', y=0.98)
                plt.tight_layout()
                fname_p = f"03_TFR_topo_{comp_name}_cluster_{i+1}_panel.png"
                fig_p.savefig(os.path.join(figures_path, fname_p), dpi=300)
                plt.close(fig_p)
                print(f"  [OK] Saved 3-panel comparison topoplot: {fname_p}")
        else:
            sig_marker = ""
            
        mean_t = T_obs[mask].mean()

        # Compute Cohen's d (paired, d_z) for every cluster
        participant_mean_diff = X_diff_test[:, mask].mean(axis=1)
        mean_diff = np.mean(participant_mean_diff)
        std_diff = np.std(participant_mean_diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

        msg = (f"  Cluster {i+1}: {t_start:.3f}-{t_end:.3f}s, {f_low:.1f}-{f_high:.1f}Hz, "
               f"p={pval:.4f}{sig_marker} | Mean T: {mean_t:.2f}"
               f"\n    Mean difference: {mean_diff:8.4f} dB"
               f"\n    SD difference:   {std_diff:8.4f} dB"
               f"\n    Cohen's d:       {cohens_d:8.4f}")
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
    ax_diff.set_xlabel(TIME_LABEL, fontsize=14, fontname='Times New Roman')
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
    ax_stat.set_xlabel(TIME_LABEL, fontsize=14, fontname='Times New Roman')
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