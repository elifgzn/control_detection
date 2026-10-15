"""
3_TFR_calculation_SME.py   (from_15.10 version)
================================================
Subsequent Memory Effect (SME) TFR Calculation
----------------------------------------------

PURPOSE:
    Computes Time-Frequency Representation (TFR) via Morlet wavelets, splits
    the trials by subsequent memory of the CONTROLLED item, balances trials
    between the two cells (100 iterations) and saves per-participant
    condition averages to:
        CDmem_XXXX_TFR_MainEffect_ControlledOnly.npz

    "Subsequently remembered" (recalled)   = the controlled item of this encoding
                                              trial was later answered "old" (hit)
    "Subsequently forgotten" (not_recalled) = it was later answered "new" (miss)
    Collapsed across High and Low control.

WHAT CHANGED COMPARED TO THE before_15.10 VERSION (agreed on 15.10):
    0. ONLY the collapsed Controlled-items SME is computed (agreed: this is the
       SME analysis). Removed: the 4-cell files (ConditionAverages,
       ConditionAverages_AllItems), MainEffect_AllItems, ItemTypeAverages and
       DetectionAverages.
    1. NO BASELINE CORRECTION IS APPLIED HERE ANYMORE.
       We now save the RAW (not dB) condition-averaged power. The dB baseline
       correction is done in the permutation script, where a switch chooses
       between a POOLED baseline (one baseline shared by both conditions) and
       a PER-CONDITION baseline (each condition divided by its own baseline,
       = the old method). To make that possible we additionally save the
       baseline power of each condition (per channel x frequency).
    2. MOVEMENT-ONSET-LOCKED VERSION ADDED (in addition to stimulus-locked).
       Wavelets are still computed on the stimulus-locked epochs (-0.8 to 5.5 s).
       Afterwards every trial is cut out around ITS OWN movement onset
       (S 21-24, read from the raw .vmrk file), window -0.52 to +3.00 s.
       The baseline of each trial stays its PRE-STIMULUS baseline (-0.5 to
       -0.2 s before stimulus onset). Trials whose movement started more than
       2.25 s after stimulus onset are excluded from the movement-locked data
       (otherwise +3 s would reach the wavelet edge at the end of the epoch).
    3. Trial counts (before balancing) are saved in the output file and in
       TFR_trial_counts_SME.csv. The 25-TRIAL RULE (participants need >= 25
       trials in BOTH cells) is applied, printed and reported in the
       permutation script; here the counts are only printed and flagged.

WHAT DID NOT CHANGE:
    Frequencies (2-40 Hz, 1 Hz steps), wavelet cycles (f * 0.5), decimation (10),
    baseline window (-0.5 to -0.2 s), Surface Laplacian (CSD), behavioural
    matching + trigger sanity check, the recalled / not-recalled definition,
    trial balancing (100 iterations, seed 2026), saving the info file,
    participant list.

OUTPUT FILE CONTENTS (CDmem_XXXX_TFR_MainEffect_ControlledOnly.npz):
    ch_names, freqs, times_stim, times_move
    stim_recalled, stim_not_recalled          RAW power (ch x freqs x times)
    move_recalled, move_not_recalled
    bl_stim_recalled, bl_stim_not_recalled    baseline power (ch x freqs)
    bl_move_recalled, bl_move_not_recalled
    n_stim_recalled, n_stim_not_recalled      trials per cell BEFORE balancing
    n_move_recalled, n_move_not_recalled
    nbal_stim, nbal_move                      trials per cell AFTER balancing
    n_move_excluded_recalled, n_move_excluded_not_recalled
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_morlet

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
eeg_path        = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"   # unchanged input
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
raw_path        = r"H:\PHD\control_detection\main_data\eeg\eeg1_raweeg"             # NEW: for .vmrk marker files
# NEW: all from_15.10 results go into main_data\eeg\from_15.10\
output_path     = r"H:\PHD\control_detection\main_data\eeg\from_15.10\eeg4_TFR_SME"

os.makedirs(output_path, exist_ok=True)

# TFR Parameters (UNCHANGED)
FREQS = np.arange(2, 41, 1)  # 2 to 40 Hz broadband
N_CYCLES = FREQS * 0.5       # 0.5s effective window
DECIM = 10                   # Downsample TFR to 25 Hz (= one time point every 40 ms)
BASELINE = (-0.5, -0.2)      # Baseline window (-500 to -200 ms before STIMULUS onset)

# NEW: minimum number of trials per cell (the 25-trial rule). Only used for
# flagging here; the rule itself is applied in 5_TFR_permutation_SME.py.
MIN_TRIALS = 25

# ── NEW: movement-locked settings ───────────────────────────────────────────
# Window around movement onset, given in TFR samples (1 sample = 40 ms):
#   13 samples before = -0.52 s,  75 samples after = +3.00 s
MOVE_N_PRE  = 13
MOVE_N_POST = 75
# Trials whose movement started later than this (s after stimulus onset) are
# excluded from the movement-locked data (see header, point 2).
MAX_MOVE_LATENCY = 2.25
# Wavelet edge margin: data within this distance (s) of the epoch start/end is
# considered contaminated. Used only for a safety check below.
EDGE_MARGIN = 0.25

# ── DIAGNOSTIC SWITCH (NOT FOR REPORTING) ── (UNCHANGED, kept from before) ──
# True  = crop the new epochs back to start at -0.5 s, as in the OLD pipeline,
#         so the baseline (-0.5 to -0.2 s) again sits at the epoch edge.
#         Everything else (ICA, exclusions, rejection) stays new.
#         Output goes to a separate folder; normal results are not touched.
# False = normal analysis (use this for everything you report).
DIAG_OLD_EDGE = False
if DIAG_OLD_EDGE:
    output_path = output_path + "_DIAG_oldedge"
    os.makedirs(output_path, exist_ok=True)
    print(f"*** DIAGNOSTIC MODE: epochs cropped to -0.5 s, output -> {output_path} ***")

# Participants: all, minus the exclusions (1, 5, 28 = technical; 2, 3, 11, 24, 26,
# 43, 45, 46, 47 = data-based). Reset to the FULL list for the new pipeline.
plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})
# plist = [4]   # <- for a quick test with one participant

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavioral_data(sub):
    """Load and merge encoding + recognition behavioral data. (UNCHANGED)"""
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


def read_movement_latencies(sub_id):
    """
    NEW. Read the raw BrainVision marker file (.vmrk) and return, for EVERY
    stimulus-onset trigger (S 11 / S 13) in recording order:
        codes     : the stimulus trigger code (11 or 13)
        latencies : time (s) from stimulus onset to movement onset, where
                    movement onset = the next S 21/22/23/24 trigger
                    (NaN if no movement trigger follows before the next stimulus)

    Why this order matches the epochs:
        1_import_raw_ab_whole.py kept ONLY the S 11/S 13 events before epoching,
        so epochs.selection (the original index of each surviving epoch) is an
        index into exactly this list of stimulus triggers.
    """
    vmrk_file = os.path.join(raw_path, f"CDmem_{sub_id}.vmrk")
    vhdr_file = os.path.join(raw_path, f"CDmem_{sub_id}.vhdr")

    # Sampling rate of the RAW recording (marker positions are in raw samples).
    # Read from the header; SamplingInterval is given in microseconds.
    sfreq_raw = 1000.0
    if os.path.exists(vhdr_file):
        with open(vhdr_file, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.startswith('SamplingInterval='):
                    sfreq_raw = 1e6 / float(line.split('=')[1])
                    break

    # Parse lines like: Mk4=Stimulus,S 13,1257284,1,0  -> (code=13, sample=1257284)
    markers = []
    with open(vmrk_file, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = re.match(r"Mk\d+=Stimulus,S\s*(\d+),(\d+)", line)
            if m:
                markers.append((int(m.group(1)), int(m.group(2))))

    codes, latencies = [], []
    for i, (code, sample) in enumerate(markers):
        if code not in (11, 13):
            continue
        lat = np.nan
        # Walk forward to the next movement trigger, but stop if the next
        # stimulus onset comes first (= no movement recorded for this trial).
        for code2, sample2 in markers[i + 1:]:
            if code2 in (11, 13):
                break
            if code2 in (21, 22, 23, 24):
                lat = (sample2 - sample) / sfreq_raw
                break
        codes.append(code)
        latencies.append(lat)
    return np.array(codes), np.array(latencies)


def balance_trials(arrays_A, arrays_B, n_iterations=100, rng=None):
    """
    Balance trial counts between two conditions by random subsampling.
    The condition with MORE trials is randomly subsampled (without replacement)
    to match the trial count of the condition with FEWER trials.
    (Same procedure as before.)

    CHANGED (only technically): instead of one array per condition, this now
    takes a LIST of arrays per condition that all have trials on axis 0 (e.g.
    [movement-locked power, per-trial baseline power]). In every iteration the
    SAME randomly chosen trials are used for all arrays of a condition, so the
    baseline always belongs to exactly the trials that were averaged.
    Returns: list of averages for A, list of averages for B, n_min
    """
    n_A, n_B = len(arrays_A[0]), len(arrays_B[0])
    n_min = min(n_A, n_B)

    if rng is None:
        rng = np.random.default_rng()

    sum_A = [np.zeros(a.shape[1:], dtype=np.float64) for a in arrays_A]
    sum_B = [np.zeros(b.shape[1:], dtype=np.float64) for b in arrays_B]

    for _ in range(n_iterations):
        idx_A = rng.choice(n_A, size=n_min, replace=False)
        idx_B = rng.choice(n_B, size=n_min, replace=False)
        for k, a in enumerate(arrays_A):
            sum_A[k] += a[idx_A].mean(axis=0)
        for k, b in enumerate(arrays_B):
            sum_B[k] += b[idx_B].mean(axis=0)

    avg_A = [s / n_iterations for s in sum_A]
    avg_B = [s / n_iterations for s in sum_B]

    return avg_A, avg_B, n_min


def baseline_power(power, times, baseline):
    """
    NEW (replaces baseline_correct_db, which now lives in the permutation script).
    Mean RAW power in the baseline window, per channel x frequency.
    Works on averaged data (ch x f x t) and on single trials (trials x ch x f x t):
    it always averages over the LAST axis (time).
    """
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    return power[..., bl_mask].mean(axis=-1)


def cut_movement_locked(power_trials, times, latencies, epoch_limits):
    """
    NEW. Re-align single-trial TFR power to movement onset.
        power_trials : (trials x channels x freqs x times) stimulus-locked power
        times        : stimulus-locked time axis (s)
        latencies    : (trials,) movement onset in s after stimulus onset
        epoch_limits : (first, last) time of the EPOCH in s (e.g. -0.8, 5.5);
                       used for the edge safety check
    For each trial, finds the TFR sample closest to its movement onset and cuts
    MOVE_N_PRE samples before to MOVE_N_POST samples after it.
    (Latencies are rounded to the 40 ms TFR grid: max. +/-20 ms error.)
    Returns: (trials x channels x freqs x (MOVE_N_PRE + MOVE_N_POST + 1)) array
    """
    dt = times[1] - times[0]
    n_win = MOVE_N_PRE + MOVE_N_POST + 1
    out = np.empty(power_trials.shape[:3] + (n_win,), dtype=power_trials.dtype)
    for i, lat in enumerate(latencies):
        i0 = int(np.argmin(np.abs(times - lat)))      # sample of movement onset
        start, stop = i0 - MOVE_N_PRE, i0 + MOVE_N_POST + 1
        # Safety check: the window must lie inside the epoch AND outside the
        # wavelet edge zones. This can only fail if MAX_MOVE_LATENCY or the
        # window settings above are changed.
        if (start < 0 or stop > len(times) or
                times[start] < epoch_limits[0] + EDGE_MARGIN - 0.03 or
                times[stop - 1] > epoch_limits[1] - EDGE_MARGIN + 0.03):
            # (0.03 s tolerance = the latency rounding to the 40 ms grid)
            raise ValueError(f"Movement-locked window of trial {i} (latency {lat:.2f}s) "
                             f"reaches into the epoch edge. Check MAX_MOVE_LATENCY.")
        out[i] = power_trials[i, :, :, start:stop]
    return out

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  SME (CONTROLLED ITEMS): TFR COMPUTATION (RAW POWER, STIM + MOVE LOCKED)")
print("=" * 70)

rng = np.random.default_rng(2026)        # UNCHANGED: stimulus-locked balancing
# NEW: a SEPARATE random generator for the movement-locked balancing, so that
# the stimulus-locked balancing draws exactly the same random trials as in the
# old script (the old script only had one generator, used once per participant).
rng_move = np.random.default_rng(2027)

count_rows = []   # NEW: one row per participant for the trial-count CSV

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    out_file = os.path.join(output_path, f"CDmem_{sub_id}_TFR_MainEffect_ControlledOnly.npz")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: epoch file not found")
        continue

    print(f"\n{'-'*60}\n  Participant {sub}\n{'-'*60}")

    # 1. Load Epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    if DIAG_OLD_EDGE:
        epochs.crop(tmin=-0.5)   # diagnostic: restore old epoch start (edge at baseline)

    # Save the info file for adjacency matrix reconstruction in permutation tests
    # (UNCHANGED from the old SME script)
    info_file = os.path.join(output_path, f"CDmem_{sub_id}_info.fif")
    if not os.path.exists(info_file):
        epochs.info.save(info_file, overwrite=True)

    # Apply Spatial Laplacian (CSD)
    print("  Step 1: Applying Surface Laplacian (CSD)...")
    epochs = mne.preprocessing.compute_current_source_density(epochs)

    # 2. Load Behavioral Data & Align (UNCHANGED)
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

    # 2b. NEW: movement-onset latencies from the .vmrk file, aligned to the epochs
    vmrk_codes, vmrk_lat = read_movement_latencies(sub_id)
    lat = vmrk_lat[kept_indices]
    # Check that the .vmrk stimulus triggers line up with the epochs. If they do
    # not, the latencies would belong to the wrong trials, so the movement-locked
    # data is NOT computed for this participant (stimulus-locked still is).
    move_ok = (len(vmrk_codes) > kept_indices.max()) and np.array_equal(vmrk_codes[kept_indices], eeg_triggers)
    if move_ok:
        print(f"  ✓ Movement latencies: {np.sum(~np.isnan(lat))} trials, "
              f"median {np.nanmedian(lat):.2f} s, max {np.nanmax(lat):.2f} s")
    else:
        print(f"  ERROR: .vmrk stimulus triggers do not match the epochs - "
              f"movement-locked data SKIPPED for participant {sub}.")

    if DIAG_OLD_EDGE and move_ok:
        # The diagnostic crop (epochs start at -0.5 s) leaves no room for the
        # movement-locked window, so the diagnostic is stimulus-locked only.
        move_ok = False
        print("  (Diagnostic mode: movement-locked data not computed)")

    ctrl_rec = trial_info['ctrl_mem_response'].values

    # ── MAIN EFFECT: Recalled vs. Not Recalled (collapsed across control) ──
    # Controlled items only: trial is "recalled" if the controlled item was recalled
    # (UNCHANGED definition, = conditions_dict_collapsed_ctrl of the old script)
    mask_A = (ctrl_rec == 'yes')     # recalled
    mask_B = (ctrl_rec == 'no')      # not_recalled
    n_A, n_B = int(mask_A.sum()), int(mask_B.sum())
    print(f"  Trial counts: Recalled={n_A}, Not recalled={n_B}")
    if n_A < MIN_TRIALS or n_B < MIN_TRIALS:
        print(f"  NOTE: fewer than {MIN_TRIALS} trials in at least one cell -> this participant "
              f"will be EXCLUDED by the 25-trial rule in the permutation script.")

    if n_A < 2 or n_B < 2:
        print(f"  Warning: Not enough trials for recalled/not_recalled in participant {sub}. Skipping.")
        continue

    # 3. Compute TFR (Morlet Wavelets) for all trials (UNCHANGED)
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

    # 4a. Stimulus-locked: trial balancing (UNCHANGED), but NO dB correction here
    print("  Step 3: Trial balancing (100 iterations), stimulus-locked...")
    (avg_A,), (avg_B,), n_min = balance_trials([tfr.data[mask_A]], [tfr.data[mask_B]],
                                                    n_iterations=100, rng=rng)
    out = dict(
        ch_names=tfr.ch_names,
        freqs=FREQS,
        times_stim=tfr.times,
        stim_recalled=avg_A.astype(np.float32),
        stim_not_recalled=avg_B.astype(np.float32),
        bl_stim_recalled=baseline_power(avg_A, tfr.times, BASELINE),
        bl_stim_not_recalled=baseline_power(avg_B, tfr.times, BASELINE),
        n_stim_recalled=int(n_A),
        n_stim_not_recalled=int(n_B),
        nbal_stim=int(n_min),
    )
    row = dict(sub=sub, n_stim_recalled=int(n_A), n_stim_not_recalled=int(n_B))

    # 4b. NEW: Movement-locked
    if move_ok:
        print("  Step 4: Movement-locked re-alignment + trial balancing...")
        # Only trials with a valid movement onset that is not too late
        valid = ~np.isnan(lat) & (lat <= MAX_MOVE_LATENCY)
        mA, mB = mask_A & valid, mask_B & valid
        n_excl_A = int((mask_A & ~valid).sum())
        n_excl_B  = int((mask_B & ~valid).sum())
        print(f"    Excluded (movement > {MAX_MOVE_LATENCY} s after stimulus or missing): "
              f"Recalled={n_excl_A}, Not recalled={n_excl_B}")

        # Per-trial pre-stimulus baseline power (trials x ch x f)
        bl_trials = baseline_power(tfr.data, tfr.times, BASELINE)
        epoch_limits = (epochs.times[0], epochs.times[-1])   # -0.8, 5.5 s
        # Re-aligned power (trials x ch x f x times_move)
        move_A = cut_movement_locked(tfr.data[mA], tfr.times, lat[mA], epoch_limits)
        move_B  = cut_movement_locked(tfr.data[mB], tfr.times, lat[mB], epoch_limits)

        (mavg_A, mbl_A), (mavg_B, mbl_B), n_min_move = balance_trials(
            [move_A, bl_trials[mA]], [move_B, bl_trials[mB]],
            n_iterations=100, rng=rng_move)

        dt = tfr.times[1] - tfr.times[0]
        out.update(
            times_move=np.arange(-MOVE_N_PRE, MOVE_N_POST + 1) * dt,
            move_recalled=mavg_A.astype(np.float32),
            move_not_recalled=mavg_B.astype(np.float32),
            bl_move_recalled=mbl_A,
            bl_move_not_recalled=mbl_B,
            n_move_recalled=int(mA.sum()),
            n_move_not_recalled=int(mB.sum()),
            nbal_move=int(n_min_move),
            n_move_excluded_recalled=n_excl_A,
            n_move_excluded_not_recalled=n_excl_B,
        )
        row.update(n_move_recalled=int(mA.sum()), n_move_not_recalled=int(mB.sum()),
                   n_move_excluded_recalled=n_excl_A, n_move_excluded_not_recalled=n_excl_B)
        del move_A, move_B, bl_trials

    # 5. Save Data
    np.savez(out_file, **out)
    count_rows.append(row)
    print(f"  ✓ Saved to CDmem_{sub_id}_TFR_MainEffect_ControlledOnly.npz "
          f"(stim: balanced to {n_min} trials/condition"
          + (f"; move: balanced to {out['nbal_move']})" if move_ok else "; move: SKIPPED)"))
    del tfr, epochs

# ── NEW: trial-count summary across participants ────────────────────────────
# Rows of participants run now replace their old rows; others are kept, so
# running the script for one participant does not erase the rest.
if count_rows:
    count_file = os.path.join(output_path, "TFR_trial_counts_SME.csv")
    new_df = pd.DataFrame(count_rows)
    if os.path.exists(count_file):
        old_df = pd.read_csv(count_file)
        new_df = pd.concat([old_df[~old_df['sub'].isin(new_df['sub'])], new_df])
    new_df.sort_values('sub').to_csv(count_file, index=False)
    print(f"\n  Trial counts saved to {count_file}")

print("\n" + "=" * 70)
print("  SME TFR CALCULATION COMPLETE")
print("=" * 70)
