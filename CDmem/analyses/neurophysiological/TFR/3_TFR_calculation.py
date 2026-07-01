"""
3_TFR_calculation.py
====================
Alpha-Band Time-Frequency Calculation & Baseline Correction
-----------------------------------------------------------

PURPOSE:
    Computes Time-Frequency Representation (TFR) via Morlet wavelets, 
    averages power per condition across trials, and applies baseline correction 
    to the condition-averaged power.
    
    This functionally replicates the FieldTrip steps from A2_3_freqCalc_feedback.m:
        - ft_freqanalysis (for baseline across all trials)
        - ft_freqdescriptives (to get grand average baseline over time & trials)
        - ft_freqanalysis (with keeptrials='no' for condition averages)
        - Baseline correction: 10 * log10(condition_power / grand_baseline)
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_morlet

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
eeg_path       = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path    = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"

os.makedirs(output_path, exist_ok=True)

# TFR Parameters
FREQS = np.arange(2, 41, 1)  # 2 to 40 Hz (broadband, so we have all data available)
N_CYCLES = FREQS * 0.5       # Fixed 0.5s effective window (matches FieldTrip t_ftimwin=0.5)
DECIM = 10                   # Downsample TFR to 25 Hz to save memory/disk space
BASELINE = (-0.5, -0.2)      # Baseline correction window (-500 to -200 ms)

# plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
plist = [25]

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
print("  PART 1: TFR COMPUTATION & BASELINE CORRECTION")
print("=" * 70)

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    out_file = os.path.join(output_path, f"CDmem_{sub_id}_TFR_ConditionAverages.npz")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: epoch file not found")
        continue

    print(f"\n{'-'*60}\n  Participant {sub}\n{'-'*60}")

    # 1. Load Epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Apply Spatial Laplacian (matches FieldTrip ft_scalpcurrentdensity)
    print("  Step 1: Applying Surface Laplacian (CSD)...")
    epochs = mne.preprocessing.compute_current_source_density(epochs)
    
    # 2. Load Behavioral Data
    trial_info = load_behavioral_data(sub)
    kept_indices = epochs.selection if hasattr(epochs, 'selection') and epochs.selection is not None else np.arange(len(epochs))
    trial_info = trial_info.iloc[kept_indices].reset_index(drop=True)

    # ── Trigger sanity check ──────────────────────────────────────────────────
    # Verify that the EEG epoch triggers (S 11 = low control, S 13 = high control)
    # align perfectly with the behavioral CSV trigger_stim_onset values.
    # This is the TFR equivalent of the check in 3_ERP_calculation.py (lines 131–146).
    #
    # epoch.events[:, 2] → integer MNE event codes
    # epochs.event_id    → maps 'Stimulus/S 11' / 'Stimulus/S 13' to those codes
    event_id_rev_check = {v: int(k.split('S ')[1]) for k, v in epochs.event_id.items()}
    eeg_triggers = np.array([event_id_rev_check[e] for e in epochs.events[:, 2]])
    log_triggers  = trial_info['trigger_stim_onset'].values   # should be 11 or 13

    if len(eeg_triggers) != len(log_triggers):
        print(f"  ERROR: EEG epoch count ({len(eeg_triggers)}) ≠ "
              f"behavioral rows ({len(log_triggers)}) after selection. "
              f"Check trigger alignment!")
    else:
        mismatch_found = False
        for i in range(len(eeg_triggers)):
            if eeg_triggers[i] != log_triggers[i]:
                print(f"  ERROR: Trigger mismatch at trial index {i}: "
                      f"EEG={eeg_triggers[i]}, Log={log_triggers[i]}")
                mismatch_found = True
                break
        if not mismatch_found:
            print(f"  ✓ Sanity check passed: all {len(eeg_triggers)} EEG triggers "
                  f"match behavioral CSV trigger_stim_onset values (S 11 / S 13).")

    cond_arr = trial_info['control_condition'].values
    rec_arr = trial_info['mem_response'].values
    
    conditions_dict = {
        'low_recalled': (cond_arr == 'low') & (rec_arr == 'yes'),
        'low_not_recalled': (cond_arr == 'low') & (rec_arr == 'no'),
        'high_recalled': (cond_arr == 'high') & (rec_arr == 'yes'),
        'high_not_recalled': (cond_arr == 'high') & (rec_arr == 'no')
    }

    # 3. Compute TFR (Morlet Wavelets) for ALL trials
    # Equivalent to ft_freqanalysis(keeptrials='yes')
    print("  Step 2: Computing TFR (Morlet wavelets) for all trials...")
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
    
    # Data shape is (n_epochs, n_channels, n_freqs, n_times)
    
    # 4. Calculate Grand-Average Baseline Power
    # FieldTrip Step 15: average power during baseline time window per trial,
    # then compute grand average of baseline over all trials.
    print(f"  Step 3: Calculating grand-average baseline ({BASELINE[0]} to {BASELINE[1]} s) across all trials...")
    baseline_mask = (tfr.times >= BASELINE[0]) & (tfr.times <= BASELINE[1])
    
    # Average across time (axis=3) and trials (axis=0)
    # This gives a single baseline power value per channel and frequency
    grand_baseline = np.nanmean(tfr.data[:, :, :, baseline_mask], axis=(0, 3))
    
    # 5. Average Trials Per Condition and Apply Baseline Correction
    # FieldTrip Step 16: ft_freqanalysis(keeptrials='no') -> condition average
    # Then: 10 * log10(bsxfun(@rdivide,powavg.powspctrm,ga_base))
    print("  Step 4: Calculating condition averages and applying baseline correction...")
    
    out_data = {}
    epsilon = 1e-15 # small constant to avoid divide by zero or log of zero
    
    for cond_name, mask in conditions_dict.items():
        if mask.sum() < 2:
            print(f"    Warning: Not enough trials for {cond_name}. Skipping condition.")
            continue
            
        # Get raw power for trials in this condition
        cond_trials_power = tfr.data[mask]
        
        # Average across trials (axis=0) -> shape: (n_channels, n_freqs, n_times)
        cond_avg_power = np.nanmean(cond_trials_power, axis=0)
        
        # Apply baseline correction (dB = 10 * log10(signal / baseline))
        cond_db = 10 * np.log10(
            np.maximum(cond_avg_power, epsilon) / 
            np.maximum(grand_baseline[:, :, np.newaxis], epsilon)
        )
        
        out_data[cond_name] = cond_db

    # 6. Save Data
    # We save all channels, allowing subsequent scripts to pick ROIs for heatmaps or run topoplots.
    print("  Step 5: Saving per-subject condition-averaged data...")
    roi_channels = [ch for ch in tfr.ch_names if ch.startswith('P') or ch.startswith('O')]
    
    np.savez(
        out_file,
        ch_names=tfr.ch_names,
        times=tfr.times,
        freqs=FREQS,
        roi_channels=roi_channels,
        **out_data
    )
    print(f"  ✓ Saved condition averages to {out_file}")

print("\n" + "=" * 70)
print("  TFR CALCULATION COMPLETE")
print("=" * 70)
