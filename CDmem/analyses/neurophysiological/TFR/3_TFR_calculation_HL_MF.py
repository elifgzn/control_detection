"""
3_TFR_calculation_HL_MF.py
==========================
Main Effect of Control (High vs Low) TFR Calculation & Baseline Correction
--------------------------------------------------------------------------

PURPOSE:
    Computes Time-Frequency Representation (TFR) via Morlet wavelets across
    ALL trials for High Control vs. Low Control (regardless of subsequent memory).
    Balances trials between High and Low control (100 iterations), applies decibel
    baseline correction, and saves per-participant condition averages to:
        CDmem_XXXX_TFR_MainEffect_Control.npz
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
eeg_path        = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path     = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"

os.makedirs(output_path, exist_ok=True)

# TFR Parameters
FREQS = np.arange(2, 41, 1)  # 2 to 40 Hz broadband
N_CYCLES = FREQS * 0.5       # 0.5s effective window
DECIM = 10                   # Downsample TFR to 25 Hz
BASELINE = (-0.5, -0.2)      # Baseline window (-500 to -200 ms)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

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

    return trial_info[['control_condition', 'controlled_img', 'uncontrolled_img', 'ctrl_mem_response', 'unctrl_mem_response', 'detection_accuracy', 'trigger_stim_onset']].copy()

def balance_trials(tfr_A, tfr_B, n_iterations=100, rng=None):
    """
    Balance trial counts between two conditions by random subsampling.
    The condition with MORE trials is randomly subsampled (without replacement)
    to match the trial count of the condition with FEWER trials.
    """
    n_A, n_B = len(tfr_A), len(tfr_B)
    n_min = min(n_A, n_B)

    if rng is None:
        rng = np.random.default_rng()

    sum_A = np.zeros(tfr_A.shape[1:], dtype=np.float64)
    sum_B = np.zeros(tfr_B.shape[1:], dtype=np.float64)

    for _ in range(n_iterations):
        idx_A = rng.choice(n_A, size=n_min, replace=False)
        idx_B = rng.choice(n_B, size=n_min, replace=False)
        sum_A += tfr_A[idx_A].mean(axis=0)
        sum_B += tfr_B[idx_B].mean(axis=0)

    avg_A = sum_A / n_iterations
    avg_B = sum_B / n_iterations

    return avg_A, avg_B, n_min

def baseline_correct_db(power, times, baseline):
    """Apply decibel (dB) baseline correction to TFR power."""
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    bl_mean = power[:, :, bl_mask].mean(axis=-1, keepdims=True)
    bl_mean = np.maximum(bl_mean, 1e-30)
    return 10 * np.log10(np.maximum(power, 1e-30) / bl_mean)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MAIN EFFECT OF CONTROL: TFR COMPUTATION & BASELINE CORRECTION")
print("=" * 70)

rng = np.random.default_rng(2026)

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    out_file = os.path.join(output_path, f"CDmem_{sub_id}_TFR_MainEffect_Control.npz")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: epoch file not found")
        continue

    print(f"\n{'-'*60}\n  Participant {sub}\n{'-'*60}")

    # 1. Load Epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Apply Spatial Laplacian (CSD)
    print("  Step 1: Applying Surface Laplacian (CSD)...")
    epochs = mne.preprocessing.compute_current_source_density(epochs)
    
    # 2. Load Behavioral Data & Align
    trial_info = load_behavioral_data(sub)
    kept_indices = epochs.selection if hasattr(epochs, 'selection') and epochs.selection is not None else np.arange(len(epochs))
    trial_info = trial_info.iloc[kept_indices].reset_index(drop=True)

    event_id_rev_check = {v: int(k.split('S ')[1]) for k, v in epochs.event_id.items()}
    eeg_triggers = np.array([event_id_rev_check[e] for e in epochs.events[:, 2]])
    log_triggers  = trial_info['trigger_stim_onset'].values

    if len(eeg_triggers) != len(log_triggers):
        print(f"  ERROR: EEG epoch count ({len(eeg_triggers)}) ≠ behavioral rows ({len(log_triggers)}).")
    else:
        mismatch_found = False
        for i in range(len(eeg_triggers)):
            if eeg_triggers[i] != log_triggers[i]:
                print(f"  ERROR: Trigger mismatch at trial index {i}")
                mismatch_found = True
                break
        if not mismatch_found:
            print(f"  ✓ Sanity check passed: {len(eeg_triggers)} EEG triggers match behavioral CSV.")

    cond_arr = trial_info['control_condition'].values
    
    # ── Main Effect of Control: All High Control vs All Low Control ──
    mask_high = (cond_arr == 'high')
    mask_low  = (cond_arr == 'low')
    
    n_high, n_low = mask_high.sum(), mask_low.sum()
    print(f"  Trial counts: High Control={n_high}, Low Control={n_low}")
    
    if n_high < 2 or n_low < 2:
        print(f"  Warning: Not enough trials for High/Low control in participant {sub}. Skipping.")
        continue

    # 3. Compute TFR (Morlet Wavelets) for all trials
    print("  Step 2: Computing TFR (Morlet wavelets) for all trials...")
    tfr = tfr_morlet(
        epochs,
        freqs=FREQS,
        n_cycles=N_CYCLES,
        return_itc=False,
        average=False,
        decim=DECIM,
        n_jobs=-1,
        verbose=False
    )
    
    # 4. Trial balancing & Baseline correction
    print("  Step 3: Trial balancing (100 iterations) & baseline correction...")
    power_high = tfr.data[mask_high]
    power_low  = tfr.data[mask_low]
    
    avg_high, avg_low, n_min = balance_trials(power_high, power_low, n_iterations=100, rng=rng)
    
    db_high = baseline_correct_db(avg_high, tfr.times, BASELINE)
    db_low  = baseline_correct_db(avg_low, tfr.times, BASELINE)
    
    roi_channels = [ch for ch in tfr.ch_names if ch.startswith(('P', 'O'))]
    
    # 5. Save Data
    np.savez(
        out_file,
        ch_names=tfr.ch_names,
        times=tfr.times,
        freqs=FREQS,
        roi_channels=roi_channels,
        high_control=db_high,
        low_control=db_low
    )
    print(f"  ✓ Saved to CDmem_{sub_id}_TFR_MainEffect_Control.npz (balanced to {n_min} trials/condition)")

print("\n" + "=" * 70)
print("  MAIN EFFECT OF CONTROL TFR CALCULATION COMPLETE")
print("=" * 70)
