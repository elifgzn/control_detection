"""
3_hilbert_alphabeta.py
=======================
Alpha/Beta Hilbert Envelope Analysis: Recalled vs Not-Recalled
--------------------------------------------------------------

PURPOSE:
    Compares alpha/beta (10-20 Hz) oscillatory amplitude during encoding
    between images that were SUBSEQUENTLY RECALLED vs NOT RECALLED in
    the surprise memory test.

    This implements the "subsequent memory effect" (SME) or "Dm effect"
    (difference due to memory) analysis for oscillatory power.

ANALYSIS PIPELINE (step by step):
    1. Load cleaned stimulus-locked epochs (from 2b_remove_noise_alphabeta.py)
    2. Load behavioral data (encoding CSV + recognition CSV)
    3. Identify the CONTROLLED image per trial and look up its memory outcome
    4. Bandpass filter to alpha/beta (10-20 Hz) using IIR Butterworth
    5. Apply Hilbert transform to get the amplitude envelope
    6. Select posterior channels (P*, O*) and average across them
    7. Crop to analysis window (stimulus onset → response onset per trial)
    8. Baseline correction (-500 to -200 ms pre-stimulus)
    9. Average per participant per condition (low/high control × recalled/not-recalled)
    10. Cluster-based permutation test (separate for each control condition)
    11. Publication-ready figures

TRIAL CLASSIFICATION:
    Each encoding trial shows TWO images. We classify each trial based on
    whether the CONTROLLED image (the one the participant moved with their
    mouse) was subsequently recalled (hit) or not recalled (miss).

    "Recalled" = the controlled image appeared in the recognition test
                 (mem_ground_truth == 'seen') AND the participant said "yes"
                 (mem_response == 'yes') → this is a HIT.

    "Not recalled" = the controlled image appeared in the recognition test
                     (mem_ground_truth == 'seen') AND the participant said "no"
                     (mem_response == 'no') → this is a MISS.

    NOTE: A 2×2 trial classification is also possible:
          controlled vs uncontrolled × recalled vs not-recalled.
          This is not implemented here but can be added later.

FIELDTRIP MATCHING:
    This script is designed to match the FieldTrip workflow as closely as
    possible in terms of the signal processing steps:
    - IIR Butterworth bandpass (same as cfg.bpfilter with default settings)
    - Hilbert envelope = abs(hilbert(x)) (same as cfg.hilbert = 'abs')
    - Cluster permutation test (same as ft_timelockstatistics with maxsum)

INPUTS:
    - Cleaned epochs: eeg3_clean_stimlocked/CDmem_XXXX-epo.fif
    - Encoding CSV:   behavioral/CDmem_1_X.csv
    - Recognition CSV: behavioral/CDmem_1_X_recognition.csv

OUTPUTS:
    - Analysis results: eeg4_alphabeta/ (numpy arrays per participant)
    - Figures:          eeg5_figures/ (SVG + PNG)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.stats import permutation_cluster_1samp_test

def _get_cluster_inds(cluster):
    """Return a 1-D integer array of time-point indices for *cluster*.
    
    MNE 1.11+ returns clusters as tuples of slices, e.g. (slice(32, 45),),
    regardless of out_type. This helper handles all formats.
    """
    if isinstance(cluster, tuple):
        obj = cluster[0]
        if isinstance(obj, slice):
            return np.arange(obj.start, obj.stop)
        else:
            return np.asarray(obj).ravel()
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return np.flatnonzero(cluster)
    return np.flatnonzero(np.asarray(cluster))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Which participant(s) to analyze?
# ──────────────────────────────────────────────────────────────
plist = [4,6,7,8,9,10,12,13,14,15,17,18,19,20,21,22]  # <-- change this as needed

# ──────────────────────────────────────────────────────────────
# Frequency band of interest
# Alpha/Beta band: 10-20 Hz
# ──────────────────────────────────────────────────────────────
FREQ_LOW  = 10  # Hz, lower edge of bandpass
FREQ_HIGH = 20  # Hz, upper edge of bandpass

# ──────────────────────────────────────────────────────────────
# Baseline window (in seconds, relative to stimulus onset)
# Used for baseline-correcting the Hilbert envelope.
# -500 to -200 ms is before any visual processing starts,
# providing a clean reference period.
# ──────────────────────────────────────────────────────────────
BASELINE_START = -0.5   # seconds
BASELINE_END   = -0.2   # seconds

# ──────────────────────────────────────────────────────────────
# Cluster permutation test parameters
# ──────────────────────────────────────────────────────────────
N_PERMUTATIONS = 1000   # number of random permutations
TAIL           = 0      # 0 = two-sided test
SEED           = 42     # for reproducibility

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
eeg_path       = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path    = r"H:\PHD\control_detection\main_data\eeg\eeg4_alphabeta"
figures_path   = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_alphabeta"

os.makedirs(output_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavioral_data(sub):
    """
    Load and merge encoding + recognition behavioral data for one participant.

    The encoding CSV contains one row per trial (calibration + test).
    The recognition CSV contains one row per recognition item (seen + unseen).

    We filter for:
    - Test phase trials only (from encoding CSV)
    - Seen items only (from recognition CSV) — we only care about old items

    For each test trial, we identify the CONTROLLED image and look up whether
    the participant recognized it (hit) or not (miss).

    Parameters
    ----------
    sub : int
        Participant number.

    Returns
    -------
    trial_info : pd.DataFrame
        One row per test trial, with columns:
        - control_condition: 'low' or 'high'
        - controlled_img: filename of the controlled image
        - mem_response: 'yes' (recalled/hit) or 'no' (not recalled/miss)
        - trigger_stim_onset: stimulus onset trigger value (11 or 13)
    """
    # ── Load encoding CSV ─────────────────────────────────────────────────
    # This CSV is generated by CDmem_1.py during the experiment.
    # It uses comma separation (standard CSV).
    enc_file = os.path.join(behavioral_path, f"CDmem_1_{sub}.csv")
    enc_df = pd.read_csv(enc_file)

    # Filter for test phase only (exclude calibration trials)
    # Calibration trials don't have stimulus onset triggers (S 11/S 13).
    test_df = enc_df[enc_df['phase'] == 'test'].copy().reset_index(drop=True)
    print(f"    Encoding CSV: {len(test_df)} test trials")

    # ── Identify the controlled image per trial ──────────────────────────
    # Each trial has two images: img_A and img_B.
    # 'true_controlled' tells us which one the participant actually controlled:
    #   - 'img_A' → the controlled image filename is in img_A_name
    #   - 'img_B' → the controlled image filename is in img_B_name
    controlled_imgs = []
    for _, row in test_df.iterrows():
        if row['true_controlled'] == 'img_A':
            controlled_imgs.append(row['img_A_name'])
        else:
            controlled_imgs.append(row['img_B_name'])
    test_df['controlled_img'] = controlled_imgs

    # ── Load recognition CSV ──────────────────────────────────────────────
    # This CSV contains the memory test results: one row per recognition item.
    # We only care about "seen" items (old items from the encoding phase).
    rec_file = os.path.join(behavioral_path, f"CDmem_1_{sub}_recognition.csv")
    rec_df = pd.read_csv(rec_file)

    # Filter: only SEEN items that were the CONTROLLED image
    # (mem_ground_truth == 'seen' AND controlled == 'yes')
    rec_controlled = rec_df[
        (rec_df['mem_ground_truth'] == 'seen') &
        (rec_df['controlled'] == 'yes')
    ][['mem_filename', 'mem_response']].copy()

    print(f"    Recognition CSV: {len(rec_controlled)} controlled seen items")

    # ── Merge: look up each controlled image's memory outcome ────────────
    # Match by image filename.
    # After merge, each test trial has a 'mem_response' column:
    #   - 'yes' = the controlled image was correctly recognized (HIT)
    #   - 'no'  = the controlled image was not recognized (MISS)
    trial_info = test_df.merge(
        rec_controlled,
        left_on='controlled_img',
        right_on='mem_filename',
        how='left',
        suffixes=('', '_rec')
    )

    # Check for unmatched trials (shouldn't happen, but just in case)
    n_unmatched = trial_info['mem_response'].isna().sum()
    if n_unmatched > 0:
        print(f"    WARNING: {n_unmatched} trials could not be matched to recognition data")

    # Keep only the columns we need
    trial_info = trial_info[['control_condition', 'controlled_img',
                             'mem_response', 'trigger_stim_onset']].copy()

    # Print summary
    for cond in ['low', 'high']:
        cond_mask = trial_info['control_condition'] == cond
        n_recalled = ((cond_mask) & (trial_info['mem_response'] == 'yes')).sum()
        n_not_recalled = ((cond_mask) & (trial_info['mem_response'] == 'no')).sum()
        print(f"    {cond.upper()} control: {n_recalled} recalled, "
              f"{n_not_recalled} not recalled")

    return trial_info


def compute_response_onset_times(epochs, sub):
    """
    Compute the time from stimulus onset to response onset for each epoch.

    We do this by loading the RAW events from the original EEG file and
    finding pairs of stimulus onset (S 11/13) and response onset (S 31-34)
    triggers. The time difference gives us the encoding duration per trial.

    Parameters
    ----------
    epochs : mne.Epochs
        The cleaned stimulus-locked epochs.
    sub : int
        Participant number (to load the raw events).

    Returns
    -------
    resp_latencies : np.array
        Array of response onset times (in seconds) relative to stimulus onset,
        one per epoch. NaN if no matching response trigger was found.
    """
    # Load the raw events from the original BrainVision file
    raw_path = os.path.join(
        r"H:\PHD\control_detection\main_data\eeg\eeg1_raweeg",
        f"CDmem_{sub:04d}.vhdr"
    )
    raw = mne.io.read_raw_brainvision(raw_path, preload=False, verbose=False)
    all_events, all_event_id = mne.events_from_annotations(raw, verbose=False)

    # Get the event IDs for stimulus onset and response onset triggers
    stim_ids = []
    for t in ['Stimulus/S 11', 'Stimulus/S 13']:
        if t in all_event_id:
            stim_ids.append(all_event_id[t])

    resp_ids = []
    for t in ['Stimulus/S 31', 'Stimulus/S 32', 'Stimulus/S 33', 'Stimulus/S 34']:
        if t in all_event_id:
            resp_ids.append(all_event_id[t])

    # Find all stimulus onset events and response onset events
    stim_events = all_events[np.isin(all_events[:, 2], stim_ids)]
    resp_events = all_events[np.isin(all_events[:, 2], resp_ids)]

    # For each stimulus onset, find the NEXT response onset event
    sfreq = raw.info['sfreq']  # sampling rate of the raw data (1000 Hz)
    resp_latencies = np.full(len(stim_events), np.nan)

    for i, stim_ev in enumerate(stim_events):
        stim_sample = stim_ev[0]
        # Find response events that come AFTER this stimulus onset
        later_resp = resp_events[resp_events[:, 0] > stim_sample]
        if len(later_resp) > 0:
            # Take the first one (nearest response onset after this stimulus)
            resp_sample = later_resp[0, 0]
            resp_latencies[i] = (resp_sample - stim_sample) / sfreq

    del raw  # free memory

    # The epochs may have had some trials dropped (artifact rejection).
    # We need to match the remaining epochs to the correct latencies.
    # epochs.selection gives the original indices of the kept epochs.
    if hasattr(epochs, 'selection') and epochs.selection is not None:
        resp_latencies = resp_latencies[epochs.selection]

    print(f"    Response onset latencies: mean={np.nanmean(resp_latencies):.2f}s, "
          f"min={np.nanmin(resp_latencies):.2f}s, max={np.nanmax(resp_latencies):.2f}s")

    return resp_latencies




# ── LOAD ALPHABETA DATA FROM DISK ──────────────────────────────────────────
# Since this script is now standalone, we must load the individual
# participant '.npz' files saved by 3_hilbert_calculation.py back into memory.
group_recalled = {'low': [], 'high': []}
group_not_recalled = {'low': [], 'high': []}
times = None

print("  Loading per-subject Hilbert envelope data...")
for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(output_path, f"CDmem_{sub_id}_alphabeta.npz")
    
    if not os.path.exists(data_file):
        print(f"    Warning: Missing {data_file}")
        continue
        
    saved = np.load(data_file, allow_pickle=True)
    
    # We need to recreate the conditions averages
    # The .npz file contains: data_bc, trial_info_condition, trial_info_recalled
    data_bc = saved['data_bc']
    cond_arr = saved['trial_info_condition']
    rec_arr = saved['trial_info_recalled']
    if times is None:
        times = saved['times']
        
    for condition in ['low', 'high']:
        cond_mask = (cond_arr == condition)
        recalled_mask = (rec_arr == 'yes')
        not_recalled_mask = (rec_arr == 'no')
        
        recalled_idx = cond_mask & recalled_mask
        not_recalled_idx = cond_mask & not_recalled_mask
        
        if recalled_idx.sum() < 2 or not_recalled_idx.sum() < 2:
            continue
            
        avg_recalled = np.nanmean(data_bc[recalled_idx], axis=0)
        avg_not_recalled = np.nanmean(data_bc[not_recalled_idx], axis=0)
        
        group_recalled[condition].append(avg_recalled)
        group_not_recalled[condition].append(avg_not_recalled)

# ══════════════════════════════════════════════════════════════════════════════
# GROUP-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GROUP-LEVEL ANALYSIS")
print(f"{'='*70}")

# Check if we have enough participants
n_subs = len(group_recalled.get('low', []))
if n_subs < 2:
    print(f"\n  Only {n_subs} participant(s) processed. Need at least 2 for "
          f"group statistics. Skipping group analysis.")
    print("  Individual participant results are saved in eeg4_alphabeta/.")
    print("  Run more participants and re-run this script to get group results.\n")
else:
    # ══════════════════════════════════════════════════════════════════════
    # STEP 10: Cluster-based permutation test
    # ══════════════════════════════════════════════════════════════════════
    # We run one test per control condition (low, high).
    # The test compares recalled vs not-recalled using a paired design
    # (within-participant difference).
    #
    # This matches FieldTrip:
    #   cfg.method = 'montecarlo';
    #   cfg.statistic = 'ft_statfun_depsamplesT';  (dependent samples t-test)
    #   cfg.correctm = 'cluster';                  (cluster correction)
    #   cfg.clusterstatistic = 'maxsum';            (sum of t-values in cluster)
    #   cfg.alpha = 0.025;                          (two-sided, so 0.05/2)
    #   cfg.numrandomization = 1000;
    #   cfg.tail = 0;                               (two-sided)
    print(f"\n  Running cluster permutation tests ({N_PERMUTATIONS} permutations)...")

    report_lines = []
    report_lines.append("ALPHA/BETA HILBERT ANALYSIS: RECALLED vs NOT-RECALLED")
    report_lines.append("=" * 70)
    report_lines.append(f"Number of permutations: {N_PERMUTATIONS}")
    report_lines.append(f"Number of participants: {n_subs}")
    report_lines.append("=" * 70)

    cluster_results = {}

    for condition in ['low', 'high']:
        recalled_arr = np.array(group_recalled[condition])       # (n_subs, n_times)
        not_recalled_arr = np.array(group_not_recalled[condition])  # (n_subs, n_times)

        if len(recalled_arr) < 2:
            print(f"    {condition.upper()}: Not enough data. Skipping.")
            continue

        # Compute the within-participant difference
        # Positive = recalled > not-recalled
        X_diff = recalled_arr - not_recalled_arr  # (n_subs, n_times)

        # Handle NaN columns: some time points may be NaN if all trials were
        # masked at that time point. Replace NaN with 0 for the permutation test.
        nan_cols = np.all(np.isnan(X_diff), axis=0)
        X_diff_clean = np.nan_to_num(X_diff, nan=0.0)

        # Run the cluster-based permutation test
        # This performs a one-sample t-test on the difference (recalled - not_recalled)
        # at each time point, clusters adjacent significant time points, and
        # corrects for multiple comparisons using the max-sum cluster statistic.
        T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
            X_diff_clean,
            n_permutations=N_PERMUTATIONS,
            tail=TAIL,
            n_jobs=-1,      # use all CPU cores
            seed=SEED
        )

        # Find significant clusters (p < 0.05)
        sig_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
        n_sig = len(sig_clusters)

        msg_header = f"\n    {condition.upper()}: {len(clusters)} clusters found, {n_sig} significant (p < 0.05)"
        print(msg_header)
        report_lines.append(msg_header)
        print(f"    {'─'*56}")
        report_lines.append(f"    {'-'*56}")

        for i, (c, p) in enumerate(zip(clusters, cluster_p)):
            # Extract time indices using the robust helper
            cluster_t_indices = _get_cluster_inds(c)
            t_start = times[cluster_t_indices[0]]
            t_end = times[cluster_t_indices[-1]]
            # Compute cluster-level statistics
            mean_T = np.mean(T_obs[cluster_t_indices])
            sum_T  = np.sum(T_obs[cluster_t_indices])
            n_timepoints = len(cluster_t_indices)
            sig_marker = " ★ SIGNIFICANT" if p < 0.05 else ""
            
            msg_cluster = (f"      Cluster {i+1}: {t_start:.3f}–{t_end:.3f}s, "
                           f"p={p:.4f}, mean T={mean_T:.3f}, sum T={sum_T:.3f}, "
                           f"n_timepoints={n_timepoints}{sig_marker}")
            print(msg_cluster)
            report_lines.append(msg_cluster)

        cluster_results[condition] = {
            'T_obs': T_obs,
            'clusters': clusters,
            'cluster_p': cluster_p,
            'H0': H0
        }

    # Save the text report
    report_file = os.path.join(figures_path, 'alphabeta_recalled_vs_not_recalled_stats.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"  ✓ Statistical report saved → {report_file}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 11: Publication-ready figures
    # ══════════════════════════════════════════════════════════════════════
    # Two panels (one per control condition).
    # Each panel shows:
    #   - Mean Hilbert envelope for recalled trials (with SEM shading)
    #   - Mean Hilbert envelope for not-recalled trials (with SEM shading)
    #   - Grey shaded regions for significant clusters
    print("\n  Creating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, condition in enumerate(['low', 'high']):
        ax = axes[ax_idx]

        recalled_arr = np.array(group_recalled[condition])
        not_recalled_arr = np.array(group_not_recalled[condition])
        n = len(recalled_arr)

        # Compute mean and SEM (standard error of the mean) across participants
        # Use nanmean/nanstd to handle NaN-masked time points
        mean_recalled = np.nanmean(recalled_arr, axis=0)
        mean_not_recalled = np.nanmean(not_recalled_arr, axis=0)
        sem_recalled = np.nanstd(recalled_arr, axis=0) / np.sqrt(n)
        sem_not_recalled = np.nanstd(not_recalled_arr, axis=0) / np.sqrt(n)

        # Plot time courses with SEM shading
        ax.plot(times, mean_recalled, color='#E91E63', linewidth=2,
                label='Recalled (hits)')
        ax.fill_between(times,
                        mean_recalled - sem_recalled,
                        mean_recalled + sem_recalled,
                        color='#E91E63', alpha=0.2)

        ax.plot(times, mean_not_recalled, color='#FFC107', linewidth=2,
                label='Not recalled (misses)')
        ax.fill_between(times,
                        mean_not_recalled - sem_not_recalled,
                        mean_not_recalled + sem_not_recalled,
                        color='#FFC107', alpha=0.2)

        # Shade significant clusters in grey
        if condition in cluster_results:
            for cluster, p_val in zip(cluster_results[condition]['clusters'],
                                       cluster_results[condition]['cluster_p']):
                if p_val < 0.05:
                    c_inds = _get_cluster_inds(cluster)
                    ax.axvspan(times[c_inds[0]], times[c_inds[-1]],
                              color='grey', alpha=0.3, label=f'p={p_val:.3f}')

        # Formatting
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_title(f'{condition.upper()} control', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
                  label='Stimulus onset')
        ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Alpha/Beta amplitude\n(baseline-corrected)', fontsize=12)
    fig.suptitle(f'Alpha/Beta ({FREQ_LOW}–{FREQ_HIGH} Hz) Envelope: '
                 f'Recalled vs Not-Recalled\n'
                 f'Posterior channels (P*, O*), N={n_subs}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    fig_base = 'alphabeta_recalled_vs_not_recalled'
    fig.savefig(os.path.join(figures_path, f'{fig_base}.svg'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(figures_path, f'{fig_base}.png'),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ Figures saved → {figures_path}")

    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Save group-level results
    # ══════════════════════════════════════════════════════════════════════
    np.savez(
        os.path.join(output_path, 'group_alphabeta_results.npz'),
        times=times,
        recalled_low=np.array(group_recalled['low']),
        recalled_high=np.array(group_recalled['high']),
        not_recalled_low=np.array(group_not_recalled['low']),
        not_recalled_high=np.array(group_not_recalled['high']),
        n_participants=n_subs
    )
    print(f"  ✓ Group results saved → {output_path}")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
