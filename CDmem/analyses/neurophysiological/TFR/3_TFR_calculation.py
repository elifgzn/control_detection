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

import sys
sys.stdout.reconfigure(encoding='utf-8')
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
plist = list(range(40,51))

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavioral_data(sub):
    """Load and merge encoding + recognition behavioral data."""
    enc_file = os.path.join(behavioral_path, f"CDmem_1_{sub}.csv")
    enc_df = pd.read_csv(enc_file)
    test_df = enc_df[enc_df['phase'] == 'test'].copy().reset_index(drop=True)

    controlled_imgs = []
    uncontrolled_imgs = []
    for _, row in test_df.iterrows():
        if row['true_controlled'] == 'img_A':
            controlled_imgs.append(row['img_A_name'])
            uncontrolled_imgs.append(row['img_B_name'])
        else:
            controlled_imgs.append(row['img_B_name'])
            uncontrolled_imgs.append(row['img_A_name'])
    test_df['controlled_img'] = controlled_imgs
    test_df['uncontrolled_img'] = uncontrolled_imgs

    rec_file = os.path.join(behavioral_path, f"CDmem_1_{sub}_recognition.csv")
    rec_df = pd.read_csv(rec_file)

    rec_controlled = rec_df[
        (rec_df['mem_ground_truth'] == 'seen') &
        (rec_df['controlled'] == 'yes')
    ][['mem_filename', 'mem_response']].copy()
    rec_controlled = rec_controlled.rename(columns={'mem_response': 'ctrl_mem_response'})

    rec_uncontrolled = rec_df[
        (rec_df['mem_ground_truth'] == 'seen') &
        (rec_df['controlled'] == 'no')
    ][['mem_filename', 'mem_response']].copy()
    rec_uncontrolled = rec_uncontrolled.rename(columns={'mem_response': 'unctrl_mem_response'})

    trial_info = test_df.merge(
        rec_controlled, left_on='controlled_img', right_on='mem_filename', how='left'
    ).drop(columns=['mem_filename'], errors='ignore')
    trial_info = trial_info.merge(
        rec_uncontrolled, left_on='uncontrolled_img', right_on='mem_filename', how='left'
    ).drop(columns=['mem_filename'], errors='ignore')

    return trial_info[['control_condition', 'controlled_img', 'uncontrolled_img', 'ctrl_mem_response', 'unctrl_mem_response', 'trigger_stim_onset']].copy()

def balance_trials(tfr_A, tfr_B, n_iterations=100, rng=None):
    """
    Balance trial counts between two conditions by random subsampling.
    The condition with MORE trials is randomly subsampled (without replacement)
    to match the trial count of the condition with FEWER trials. This is
    repeated n_iterations times, and the result is averaged across all
    iterations. This produces a stable estimate that is not biased by
    different trial counts.
    """
    n_A, n_B = len(tfr_A), len(tfr_B)
    n_min = min(n_A, n_B)

    if rng is None:
        rng = np.random.default_rng()

    # Accumulate averages across iterations
    sum_A = np.zeros(tfr_A.shape[1:], dtype=np.float64)
    sum_B = np.zeros(tfr_B.shape[1:], dtype=np.float64)

    for _ in range(n_iterations):
        idx_A = rng.choice(n_A, size=n_min, replace=False)
        idx_B = rng.choice(n_B, size=n_min, replace=False)
        sum_A += tfr_A[idx_A].mean(axis=0)
        sum_B += tfr_B[idx_B].mean(axis=0)

    # Average across all iterations
    avg_A = sum_A / n_iterations
    avg_B = sum_B / n_iterations

    return avg_A, avg_B, n_min

def baseline_correct_db(power, times, baseline):
    """Apply decibel (dB) baseline correction to TFR power."""
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    # Mean power during baseline, per channel and frequency
    bl_mean = power[:, :, bl_mask].mean(axis=-1, keepdims=True)
    bl_mean = np.maximum(bl_mean, 1e-30)
    # dB conversion
    return 10 * np.log10(np.maximum(power, 1e-30) / bl_mean)

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
    
    # Save the info file for adjacency matrix reconstruction in permutation tests
    info_file = os.path.join(output_path, f"CDmem_{sub_id}_info.fif")
    if not os.path.exists(info_file):
        epochs.info.save(info_file, overwrite=True)
    
    # Apply Spatial Laplacian (matches FieldTrip ft_scalpcurrentdensity)
    # print("  Step 1: Applying Surface Laplacian (CSD)...")
    # epochs = mne.preprocessing.compute_current_source_density(epochs)
    
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
    ctrl_rec = trial_info['ctrl_mem_response'].values
    unctrl_rec = trial_info['unctrl_mem_response'].values
    
    # ── PREREGISTERED: Controlled items only ──
    conditions_dict = {
        'low_recalled': (cond_arr == 'low') & (ctrl_rec == 'yes'),
        'low_not_recalled': (cond_arr == 'low') & (ctrl_rec == 'no'),
        'high_recalled': (cond_arr == 'high') & (ctrl_rec == 'yes'),
        'high_not_recalled': (cond_arr == 'high') & (ctrl_rec == 'no')
    }

    # ── SUPPLEMENTARY (H5): All items (collapsed across item type) ──
    # If either item was recalled, trial is recalled; if both not recalled, trial is not recalled.
    any_recalled = (ctrl_rec == 'yes') | (unctrl_rec == 'yes')
    both_not_recalled = (ctrl_rec == 'no') & (unctrl_rec == 'no')
    conditions_dict_all_items = {
        'low_recalled': (cond_arr == 'low') & any_recalled,
        'low_not_recalled': (cond_arr == 'low') & both_not_recalled,
        'high_recalled': (cond_arr == 'high') & any_recalled,
        'high_not_recalled': (cond_arr == 'high') & both_not_recalled
    }

    # ── EXPLORATORY (NOT PREREGISTERED): Item Type split ──
    conditions_dict_itemtype = {
        'low_controlled_recalled': (cond_arr == 'low') & (ctrl_rec == 'yes'),
        'low_controlled_not_recalled': (cond_arr == 'low') & (ctrl_rec == 'no'),
        'low_uncontrolled_recalled': (cond_arr == 'low') & (unctrl_rec == 'yes'),
        'low_uncontrolled_not_recalled': (cond_arr == 'low') & (unctrl_rec == 'no'),
        'high_controlled_recalled': (cond_arr == 'high') & (ctrl_rec == 'yes'),
        'high_controlled_not_recalled': (cond_arr == 'high') & (ctrl_rec == 'no'),
        'high_uncontrolled_recalled': (cond_arr == 'high') & (unctrl_rec == 'yes'),
        'high_uncontrolled_not_recalled': (cond_arr == 'high') & (unctrl_rec == 'no')
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
    
    # 4 & 5. Trial balancing and Baseline Correction
    print("  Step 3 & 4: Trial balancing (100 iterations), condition averaging, and baseline correction...")
    
    file_configs = [
        (conditions_dict, f"CDmem_{sub_id}_TFR_ConditionAverages.npz", "Preregistered (Controlled only)", 
         [('low_recalled', 'low_not_recalled'), ('high_recalled', 'high_not_recalled')]),
         
        (conditions_dict_all_items, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz", "Supplementary (All items)", 
         [('low_recalled', 'low_not_recalled'), ('high_recalled', 'high_not_recalled')]),
         
        (conditions_dict_itemtype, f"CDmem_{sub_id}_TFR_ItemTypeAverages.npz", "EXPLORATORY - NOT PREREGISTERED (Item Type)", 
         [('low_controlled_recalled', 'low_controlled_not_recalled'), 
          ('low_uncontrolled_recalled', 'low_uncontrolled_not_recalled'),
          ('high_controlled_recalled', 'high_controlled_not_recalled'),
          ('high_uncontrolled_recalled', 'high_uncontrolled_not_recalled')])
    ]

    roi_channels = [ch for ch in tfr.ch_names if ch.startswith('P') or ch.startswith('O')]
    rng = np.random.default_rng(2026)
    
    for conds, filename, desc, balance_pairs in file_configs:
        print(f"    Processing: {desc}")
        out_data = {}
        
        for pair_A, pair_B in balance_pairs:
            mask_A = conds[pair_A]
            mask_B = conds[pair_B]
            
            if mask_A.sum() < 2 or mask_B.sum() < 2:
                print(f"      Warning: Not enough trials for {pair_A}/{pair_B}. Skipping.")
                continue
                
            # Get raw power for trials
            power_A = tfr.data[mask_A]
            power_B = tfr.data[mask_B]
            
            # Balance trials and average
            avg_A, avg_B, n_min = balance_trials(power_A, power_B, n_iterations=100, rng=rng)
            
            # Apply baseline correction (condition-specific)
            db_A = baseline_correct_db(avg_A, tfr.times, BASELINE)
            db_B = baseline_correct_db(avg_B, tfr.times, BASELINE)
            
            out_data[pair_A] = db_A
            out_data[pair_B] = db_B

        # 6. Save Data
        if out_data:
            out_path_file = os.path.join(output_path, filename)
            np.savez(
                out_path_file,
                ch_names=tfr.ch_names,
                times=tfr.times,
                freqs=FREQS,
                roi_channels=roi_channels,
                **out_data
            )
            print(f"      ✓ Saved to {filename}")

print("\n" + "=" * 70)
print("  TFR CALCULATION COMPLETE")
print("=" * 70)
