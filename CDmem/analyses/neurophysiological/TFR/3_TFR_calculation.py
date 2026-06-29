"""
3_TFR_calculation.py
====================
Alpha-Band ROI Time-Frequency Calculation
-----------------------------------------

PURPOSE:
    Computes Time-Frequency Representation (TFR) via Morlet wavelets, 
    applies baseline correction, selects a posterior/occipital Region of 
    Interest (ROI), and averages across the spatial dimension.
    
    This functionally replicates the FieldTrip step:
        `ft_freqanalysis` followed by `cfg.avgoverchan = 'yes'`
    as seen in the provided MATLAB script `A2_5_permuatation_TFmaps_stimonset.m`.

    By explicitly averaging over the posterior electrodes here, we collapse
    the spatial dimension, massively increasing statistical sensitivity for 
    the targeted permutation test in the next script.
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_morlet

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
eeg_path       = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_alphabeta_static"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path    = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_alphabeta_static"

os.makedirs(output_path, exist_ok=True)

# TFR Parameters
FREQS = np.arange(2, 41, 1)  # 2 to 40 Hz (broadband, so we have all data available)
N_CYCLES = FREQS * 0.5       # Fixed 0.5s effective window (matches FieldTrip t_ftimwin=0.5)
DECIM = 10                   # Downsample TFR to 25 Hz to save memory/disk space
BASELINE = (-0.5, -0.2)      # Baseline correction window (-500 to -200 ms)
BASELINE_MODE = 'logratio'   # Log-ratio (dB) is standard for TFR baseline correction

plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 19, 20, 21, 22]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavioral_data(sub):
    """Load and merge encoding + recognition behavioral data."""
    enc_file = os.path.join(behavioral_path, f"CDmem_1_{sub}.csv")
    enc_df = pd.read_csv(enc_file)
    test_df = enc_df[enc_df['phase'] == 'test'].copy().reset_index(drop=True)

    controlled_imgs = []
    for _, row in test_df.iterrows():
        if row['true_controlled'] == 'img_A':
            controlled_imgs.append(row['img_A_name'])
        else:
            controlled_imgs.append(row['img_B_name'])
    test_df['controlled_img'] = controlled_imgs

    rec_file = os.path.join(behavioral_path, f"CDmem_1_{sub}_recognition.csv")
    rec_df = pd.read_csv(rec_file)

    rec_controlled = rec_df[
        (rec_df['mem_ground_truth'] == 'seen') &
        (rec_df['controlled'] == 'yes')
    ][['mem_filename', 'mem_response']].copy()

    trial_info = test_df.merge(
        rec_controlled, left_on='controlled_img', right_on='mem_filename', how='left'
    )
    return trial_info[['control_condition', 'controlled_img', 'mem_response', 'trigger_stim_onset']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  PART 1: ROI-AVERAGED TFR COMPUTATION")
print("=" * 70)

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    out_file = os.path.join(output_path, f"CDmem_{sub_id}_TFR_ROI.npz")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: epoch file not found")
        continue

    print(f"\n{'-'*60}\n  Participant {sub}\n{'-'*60}")

    # 1. Load Epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # 2. Load Behavioral Data
    trial_info = load_behavioral_data(sub)
    kept_indices = epochs.selection if hasattr(epochs, 'selection') and epochs.selection is not None else np.arange(len(epochs))
    trial_info = trial_info.iloc[kept_indices].reset_index(drop=True)

    # 3. Compute TFR (Morlet Wavelets)
    # Equivalent to ft_freqanalysis
    print("  Step 1: Computing TFR (Morlet wavelets)...")
    tfr = tfr_morlet(
        epochs,
        freqs=FREQS,
        n_cycles=N_CYCLES,
        return_itc=False,
        average=False,       # Keep individual trials
        decim=DECIM,
        n_jobs=-1,
        verbose=False
    )

    # 4. Skip Baseline Correction
    print(f"  Step 2: No baseline correction applied (raw oscillatory power).")

    # 5. Select ROI and Average Over Channels
    # Equivalent to cfgPermut.avgoverchan = 'yes' in FieldTrip
    print("  Step 3: Selecting posterior ROI and averaging across channels...")
    roi_channels = [ch for ch in tfr.ch_names if ch.startswith('P') or ch.startswith('O')]
    tfr_roi = tfr.copy().pick(picks=roi_channels)
    
    # Data shape is (n_epochs, n_channels, n_freqs, n_times)
    # We mean over axis 1 (channels) to get (n_epochs, n_freqs, n_times)
    data_roi = tfr_roi.data.mean(axis=1)

    # Calculate Topoplot data: all channels, alpha-beta (2-20Hz), 0.0-3.5s
    print("  Step 4: Extracting topoplot data (2-20Hz, 0.0-3.5s)...")
    freq_mask = (FREQS >= 2) & (FREQS <= 20)
    time_mask = (tfr.times >= 0.0) & (tfr.times <= 3.5)
    
    # shape: (n_epochs, n_channels, n_freqs_masked, n_times_masked)
    topo_data = tfr.data[:, :, freq_mask, :][:, :, :, time_mask]
    # Average over freqs and times -> shape: (n_epochs, n_channels)
    topo_data_mean = topo_data.mean(axis=(2, 3))

    # 6. Save data
    print("  Step 5: Saving per-subject ROI and topoplot data...")
    np.savez(
        out_file,
        data_roi=data_roi,                                   # (n_epochs, n_freqs, n_times)
        topo_data=topo_data_mean,                            # (n_epochs, n_channels)
        ch_names=tfr.ch_names,                               # list of all channel names
        trial_info_condition=trial_info['control_condition'].values,
        trial_info_recalled=trial_info['mem_response'].values,
        times=tfr.times,
        freqs=FREQS,
        roi_channels=roi_channels
    )
    print(f"  ✓ Saved to {out_file}")

print("\n" + "=" * 70)
print("  TFR CALCULATION COMPLETE")
print("=" * 70)
