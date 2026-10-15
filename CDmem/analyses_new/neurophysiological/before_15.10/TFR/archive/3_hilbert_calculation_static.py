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
    - Cleaned epochs: eeg3_clean_alphabeta_static/CDmem_XXXX-epo.fif
    - Encoding CSV:   behavioral/CDmem_1_X.csv
    - Recognition CSV: behavioral/CDmem_1_X_recognition.csv

OUTPUTS:
    - Analysis results: eeg4_alphabeta_static/ (numpy arrays per participant)
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
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 19, 20, 21, 22]  # <-- change this as needed

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
eeg_path       = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_alphabeta_static"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path    = r"H:\PHD\control_detection\main_data\eeg\eeg4_alphabeta_static"
figures_path   = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_alphabeta_static"

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





# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# Storage for group-level data (across participants)
# Structure: {condition: [participant_averages]}
group_recalled     = {'low': [], 'high': []}
group_not_recalled = {'low': [], 'high': []}
times = None  # will be set from the first participant

print("=" * 70)
print("  ALPHA/BETA HILBERT ANALYSIS: RECALLED vs NOT-RECALLED")
print("=" * 70)

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: file not found")
        continue

    print(f"\n{'─'*60}")
    print(f"  Participant {sub}")
    print(f"{'─'*60}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Load cleaned stimulus-locked epochs
    # ══════════════════════════════════════════════════════════════════════
    print("  Step 1: Loading cleaned epochs...")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    print(f"    {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz, "
          f"{len(epochs.ch_names)} channels")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Load behavioral data and match to EEG epochs
    # ══════════════════════════════════════════════════════════════════════
    print("  Step 2: Loading behavioral data...")
    trial_info = load_behavioral_data(sub)

    # The behavioral data has one row per test trial (120 trials).
    # The EEG epochs also correspond to test trials (one epoch per S 11/S 13 trigger).
    # After artifact rejection, some epochs may have been dropped.
    # epochs.selection gives the original indices of the kept epochs.
    if hasattr(epochs, 'selection') and epochs.selection is not None:
        kept_indices = epochs.selection
    else:
        kept_indices = np.arange(len(epochs))

    # Select only the behavioral rows that correspond to kept EEG epochs
    trial_info_matched = trial_info.iloc[kept_indices].reset_index(drop=True)

    # Sanity check: number of epochs should match behavioral rows
    assert len(epochs) == len(trial_info_matched), (
        f"Mismatch: {len(epochs)} epochs vs {len(trial_info_matched)} behavioral rows"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Compute per-trial response onset times
    # ══════════════════════════════════════════════════════════════════════
    print("  Step 3: Computing response onset times from EEG events...")
    resp_latencies = compute_response_onset_times(epochs, sub)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Bandpass filter to alpha/beta (10-20 Hz)
    # ══════════════════════════════════════════════════════════════════════
    # We use the same IIR Butterworth filter type as in the preprocessing
    # (1_import_raw.py, 1b_import_raw_alphabeta.py) and in the original
    # FieldTrip workflow.
    #
    # This narrowband filter extracts only the alpha/beta oscillations.
    # IMPORTANT: We filter BEFORE applying Hilbert, as per standard practice
    # (and matching FieldTrip's cfg.bpfilter → cfg.hilbert order).
    print(f"  Step 4: Bandpass filtering ({FREQ_LOW}-{FREQ_HIGH} Hz)...")
    epochs_filt = epochs.copy().filter(
        l_freq=FREQ_LOW,
        h_freq=FREQ_HIGH,
        method='iir',
        iir_params=dict(order=4, ftype='butter'),
        verbose=False
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Apply Hilbert transform to get amplitude envelope
    # ══════════════════════════════════════════════════════════════════════
    # The Hilbert transform converts the bandpass-filtered signal into its
    # analytic signal. Taking the absolute value (envelope=True) gives us
    # the instantaneous amplitude envelope — a smooth, positive-only signal
    # that represents moment-to-moment oscillatory power.
    #
    # MNE equivalent of FieldTrip's cfg.hilbert = 'abs':
    #   - FieldTrip: abs(hilbert(x)) = amplitude envelope
    #   - MNE:       apply_hilbert(envelope=True) = same thing
    print("  Step 5: Applying Hilbert transform (amplitude envelope)...")
    epochs_filt.apply_hilbert(envelope=True)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Select posterior channels and average across them
    # ══════════════════════════════════════════════════════════════════════
    # We focus on posterior channels (P* and O*) because alpha/beta power
    # is strongest over parietal and occipital cortex.
    #
    # This matches the FieldTrip script:
    #   cfg.channel = {'P*','O*'};
    #   cfg.avgoverchan = 'yes';
    print("  Step 6: Selecting posterior channels (P*, O*)...")
    # Match channels starting with P or O
    picks = [ch for ch in epochs_filt.ch_names if ch.startswith('P') or ch.startswith('O')]
    epochs_post = epochs_filt.copy().pick(picks=picks)
    post_channels = epochs_post.ch_names
    print(f"    Selected {len(post_channels)} channels: {post_channels}")

    # Get data and average across channels
    # Shape: (n_epochs, n_channels, n_times) → mean over axis=1 → (n_epochs, n_times)
    data = epochs_post.get_data()  # (n_epochs, n_channels, n_times)
    data = data.mean(axis=1)       # (n_epochs, n_times)

    # Get the time axis (same for all epochs)
    if times is None:
        times = epochs_post.times

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Mask data after each trial's response onset
    # ══════════════════════════════════════════════════════════════════════
    # The analysis window is from stimulus onset (t=0) to response onset
    # (which varies per trial). We set all data AFTER the response onset
    # to NaN so it doesn't contribute to the average.
    #
    # NOTE (COMMENTED-OUT ALTERNATIVE):
    # To use the full encoding period instead (0 to +5.5s), simply skip
    # this masking step. Uncomment the lines below and comment out the
    # masking loop:
    #
    # # --- FULL ENCODING PERIOD (0 to +5.5s) ---
    # # No per-trial masking needed. All time points from 0 to +5.5s
    # # are included in the analysis. This captures the complete encoding
    # # period from stimulus onset through agency rating.
    # pass  # keep all data as-is
    # # --- END FULL ENCODING PERIOD ---
    #
    print("  Step 7: Masking data after response onset (per trial)...")
    data_masked = data.copy()
    n_masked = 0
    for i in range(len(data_masked)):
        if not np.isnan(resp_latencies[i]):
            # Find the time index corresponding to the response onset
            mask_after = times > resp_latencies[i]
            data_masked[i, mask_after] = np.nan
            n_masked += 1
    print(f"    Masked {n_masked}/{len(data_masked)} trials at their response onset time")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: Baseline correction on the Hilbert envelope
    # ══════════════════════════════════════════════════════════════════════
    # We subtract the mean envelope amplitude during the pre-stimulus
    # baseline window (-500 to -200 ms) from each trial.
    #
    # Why this window?
    #   - It's before any visual processing (stimulus appears at t=0)
    #   - It avoids the fixation cross onset (~1s before stimulus)
    #   - It provides a clean reference for resting-state alpha/beta levels
    #   - This is standard practice for DM (difference due to memory) analyses
    #
    # Note: This is DIFFERENT from the voltage baseline correction done in
    # 2b_remove_noise_alphabeta.py (which was on the raw signal). Here we
    # baseline-correct the HILBERT ENVELOPE specifically.
    print(f"  Step 8: Baseline correction ({BASELINE_START} to {BASELINE_END}s)...")
    bc_idx = (times >= BASELINE_START) & (times <= BASELINE_END)
    # Use nanmean to handle the NaN-masked data correctly
    baseline_mean = np.nanmean(data_masked[:, bc_idx], axis=1, keepdims=True)
    data_bc = data_masked - baseline_mean

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9: Split by condition and memory outcome, then average
    # ══════════════════════════════════════════════════════════════════════
    # We analyze LOW and HIGH control conditions separately.
    # Within each condition, we split into recalled (hits) vs not-recalled (misses).
    print("  Step 9: Splitting by condition and memory outcome...")

    for condition in ['low', 'high']:
        # Boolean masks for this condition
        cond_mask = (trial_info_matched['control_condition'] == condition).values

        # Boolean masks for recalled (hit) vs not-recalled (miss)
        recalled_mask = (trial_info_matched['mem_response'] == 'yes').values
        not_recalled_mask = (trial_info_matched['mem_response'] == 'no').values

        # Combined masks: condition AND memory outcome
        recalled_idx = cond_mask & recalled_mask
        not_recalled_idx = cond_mask & not_recalled_mask

        n_recalled = recalled_idx.sum()
        n_not_recalled = not_recalled_idx.sum()

        print(f"    {condition.upper()}: {n_recalled} recalled, "
              f"{n_not_recalled} not recalled")

        if n_recalled < 2 or n_not_recalled < 2:
            print(f"    WARNING: Too few trials in {condition} condition for "
                  f"participant {sub}. Skipping this condition.")
            continue

        # Average over trials within each memory category
        # Use nanmean because some time points are NaN (masked after response onset)
        avg_recalled = np.nanmean(data_bc[recalled_idx], axis=0)
        avg_not_recalled = np.nanmean(data_bc[not_recalled_idx], axis=0)

        # Store for group-level analysis
        group_recalled[condition].append(avg_recalled)
        group_not_recalled[condition].append(avg_not_recalled)

    # ══════════════════════════════════════════════════════════════════════
    # Save per-participant results
    # ══════════════════════════════════════════════════════════════════════
    np.savez(
        os.path.join(output_path, f"CDmem_{sub_id}_alphabeta.npz"),
        times=times,
        data_bc=data_bc,
        trial_info_condition=trial_info_matched['control_condition'].values,
        trial_info_recalled=trial_info_matched['mem_response'].values,
        resp_latencies=resp_latencies,
        posterior_channels=post_channels
    )
    print(f"  ✓ Saved per-participant results → {output_path}")

    # Clean up
    del epochs, epochs_filt, epochs_post, data, data_masked, data_bc

