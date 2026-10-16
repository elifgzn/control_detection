"""
3_TFR_calculation_HL_MF_pooledbaseline.py
=========================================
Time-Frequency Calculation for the Main Effect of Control (High vs Low), 2-30 Hz
-- POOLED-BASELINE VERSION --
-------------------------------------------------------------------------------

PURPOSE:
    Copy of 3_TFR_calculation_HL_MF.py with ONE difference: the baseline.
    Instead of a separate baseline for each condition, ONE pooled baseline is
    computed from ALL trials (High + Low Control together) and applied to both
    conditions, as in the original FieldTrip pipeline
    (A2_3_freqCalc_feedback.m). Everything else (TFR settings, 25-trial rule,
    trial balancing, seeds, trigger check) is identical.

    For each participant: applies a surface Laplacian, computes single-trial
    time-frequency power with Morlet wavelets (2-30 Hz: theta, alpha, beta) for
    ALL trials, compares High Control vs. Low Control (regardless of
    subsequent memory), balances trial counts, applies dB baseline correction
    with the pooled baseline, and saves the condition averages to:
        CDmem_XXXX_TFR_MainEffect_Control.npz
    in a SEPARATE output folder (..._HL_MF_pooledbaseline), so the results of
    the per-condition-baseline version are not overwritten.

PIPELINE (per participant):
    1. Load cleaned stimulus-locked epochs (output of 2_remove_noise_ab_whole.py;
       -2.5 to 5.5 s, 250 Hz, 1-40 Hz filtered, average reference, ICA-cleaned).
    2. Surface Laplacian / current source density (MNE spherical splines,
       Perrin et al., 1989; MNE defaults: m = 4, lambda = 1e-5).
    3. Load behavioural data and align it to the surviving epochs
       (epochs.selection). Trigger check: S 11 = low control, S 13 = high control.
    4. Minimum-trial rule: the participant is excluded if High or Low Control
       has fewer than 25 trials. An old output file is deleted.
    5. Morlet wavelet power for every trial (power only, no phase):
         - 2-30 Hz in 1 Hz steps
         - cycles = max(0.5 x f, 3): fixed time window from 6 Hz up
           (Wu-like), at least 3 cycles at 2-5 Hz (Cheng et al., 2023)
         - output downsampled to 25 Hz (40 ms steps)
    6. Trial balancing: the larger condition is randomly subsampled to the size
       of the smaller one, 100 times, and the subsample averages are averaged
       (random generator seeded from (RANDOM_SEED = 2026, participant number),
       so results are identical on every run).
    7. Baseline correction in dB with ONE POOLED BASELINE for both conditions:
           pooled_baseline = mean power in -500 to -200 ms over ALL trials
                             (High + Low Control), per channel and frequency
           10 * log10(condition_power / pooled_baseline)
       as in the original FieldTrip pipeline (A2_3_freqCalc_feedback.m:
       ft_freqanalysis over all trials -> ft_freqdescriptives -> mean over
       baseline time points -> 10*log10(condition power / grand baseline)).
       Pre-stimulus power differences between High and Low Control (e.g.
       block-level state differences) therefore REMAIN visible in the contrast.

OUTPUT:
    - Per participant: .npz with high_control / low_control dB arrays
      (channels x 29 frequencies x time points), channel names, times, freqs,
      posterior ROI channels (P*/O*), raw trial counts per cell, balanced
      trial count, and the minimum-trial rule used.
    - trial_counts_HL.csv (all participants x cells, included yes/no).
    - Output folder: main_data/eeg/TFR/eeg4_TFR_stimlocked_HL_MF_pooledbaseline
      (the 5_/6_ HL scripts read eeg4_TFR_stimlocked_HL_MF; point their
      input_path to this folder to analyse the pooled-baseline results).
    - At the end of the run: trial-count summary and a trigger-check summary
      listing every participant with an EEG/behaviour trigger mismatch.
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
eeg_path        = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path     = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg4_TFR_stimlocked_HL_MF_pooledbaseline"   # separate folder for the pooled-baseline version

os.makedirs(output_path, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# TFR PARAMETERS — SETTINGS AND REASONING
# ──────────────────────────────────────────────────────────────────────────────
#
# (1) FREQUENCY RANGE: 2-30 Hz in 1 Hz steps (29 frequencies)
#     - Covers the three bands of interest: theta (~4-7 Hz), alpha (~8-12 Hz)
#       and beta (~13-30 Hz).
#     - 2-30 Hz follows Wu et al. (2025), whose main interest was alpha/beta.
#       Cheng et al. (2023) analysed theta (4-7 Hz; their main focus) and also
#       reported alpha (8-12 Hz), low beta (13-17 Hz) and high beta (18-30 Hz)
#       effects, all of which fall inside this range.
#     - Nothing above 30 Hz is calculated (earlier versions went up to 40/60 Hz).
#     - Steps: Wu et al. used 2 Hz steps, which corresponds to the frequency
#       resolution of a 0.5 s window (1 / 0.5 s = 2 Hz). We use 1 Hz steps.
#       Neighbouring 1 Hz bins overlap (see frequency smoothing below); this
#       is harmless and only gives a smoother frequency axis.
#
# (2) NUMBER OF WAVELET CYCLES: max(0.5 x f, 3)
#     Background (Cohen, 2014, ch. 12-13): the number of cycles of a Morlet
#     wavelet sets the trade-off between time and frequency precision. In MNE,
#     sigma_t = n_cycles / (2*pi*f) and sigma_f = f / n_cycles; FWHM = 2.355 x sigma.
#     More cycles -> sharper in frequency, blurrier in time, and vice versa.
#     Wavelets with fewer than ~3 cycles are no longer well-defined oscillations
#     and give poor frequency specificity (Cohen, 2014).
#
#     (a) 6-30 Hz: 0.5 x f cycles (3 to 15 cycles)
#         - With n_cycles proportional to f, the time window is the SAME at
#           every frequency: time FWHM ~0.19 s, frequency FWHM ~4.7 Hz
#           (e.g. 10 Hz is smoothed over ~7.6-12.4 Hz, 20 Hz over ~17.6-22.4 Hz).
#         - This follows the fixed-time-window approach of Wu et al. (2025),
#           who used a 0.5 s time window at all frequencies (FieldTrip-style),
#           which suits alpha/beta, the main focus of this analysis.
#         - NOTE: this is a Morlet approximation of that approach, NOT an exact
#           0.5 s window. A 0.5 s Hanning window has a time FWHM of ~0.25 s and
#           a frequency FWHM of ~2.9 Hz, so our estimates are somewhat finer in
#           time and broader in frequency than Wu et al.'s.
#
#     (b) 2-5 Hz: raised to a minimum of 3 cycles
#         - 0.5 x f alone would give only 1, 1.5, 2 and 2.5 cycles at 2, 3, 4
#           and 5 Hz. At 4 Hz this would smooth over ~1.6-6.4 Hz, i.e. well
#           into delta, so the lower half of theta would be poorly estimated.
#           (A 0.5 s window, as in Wu et al., has the same limitation; it
#           mattered less there because theta was not their focus.)
#         - Because theta is also a band of interest here, these frequencies
#           use 3 cycles, the setting used by Cheng et al. (2023) for theta
#           (Morlet wavelets with 3 cycles, 1-40 Hz; following Cohen, 2014).
#           With 3 cycles: 4 Hz is smoothed over ~2.4-5.6 Hz (time FWHM
#           ~0.28 s), 5 Hz over ~3.0-7.0 Hz (time FWHM ~0.22 s).
#         - This gives the same NUMBER of cycles as Cheng et al. at 2-5 Hz. It
#           is not necessarily an identical wavelet: toolboxes (MNE, FieldTrip,
#           EEGLAB) differ slightly in how the wavelet is defined and truncated.
#         - At 6 Hz both rules give 3 cycles, so 6-30 Hz is unchanged by the
#           minimum, and alpha/beta keep the Wu-like fixed window.
#
#     (c) Alternatives considered (not used)
#         - Fixed 3 cycles at all frequencies (exactly as Cheng et al., 2023):
#           rejected because alpha/beta would be strongly smeared in frequency
#           (10 Hz over ~6-14 Hz, 20 Hz over ~12-28 Hz), which is a poor fit
#           for alpha/beta as the main interest.
#         - Cycles rising linearly from 3 (2 Hz) to 8 (20 Hz), as in the
#           original FieldTrip pipeline (A2_3_freqCalc_feedback.m; a common
#           approach, Cohen, 2014): slightly sharper theta, practically
#           identical alpha, slightly broader beta. A valid alternative; not
#           chosen in order to keep the Wu-like fixed time window for alpha/beta.
#         - 0.67 x f cycles (closer to the time smoothing of a 0.5 s Hanning
#           window): not chosen.
#
#     (d) Caveats for interpretation
#         - Theta and low alpha overlap in frequency (e.g. 7 Hz is smoothed
#           over ~4.6-9.4 Hz), so theta effects should be described as
#           "around 4-7 Hz", not as a sharply separated band.
#         - At 2-3 Hz the 3-cycle wavelet is long (time FWHM ~0.37-0.56 s), so
#           power at the end of the baseline window (-0.2 s) already contains
#           some post-stimulus activity. Effects below ~4 Hz should be
#           interpreted with caution.
#         - Power is total power (evoked + induced). Shortly after stimulus
#           onset, theta in particular can be partly driven by the ERP.
#
# (3) TIME RESOLUTION OF THE OUTPUT: DECIM = 10
#     Every 10th sample is kept: 250 Hz -> 25 Hz (40 ms steps). Power changes
#     slowly at the time scales above (time FWHM >= ~0.19 s), so nothing is
#     lost; it only saves memory and disk space. (Cheng et al., 2023, used
#     50 ms bins.)
#
# (4) BASELINE: -500 to -200 ms, dB, ONE POOLED BASELINE for both conditions
#     (this is the only difference from 3_TFR_calculation_HL_MF.py)
#     - Pooled baseline = mean power in the baseline window over ALL trials
#       (High + Low Control), per channel and frequency. Both conditions are
#       then expressed as 10 * log10(condition power / pooled baseline).
#     - This follows the original FieldTrip pipeline (A2_3_freqCalc_feedback.m),
#       which computes the baseline over all trials (ft_freqanalysis with
#       keeptrials -> ft_freqdescriptives -> mean over baseline time points)
#       and divides every condition by this grand baseline.
#     - Why this version exists: the per-condition baseline of
#       3_TFR_calculation_HL_MF.py (following Wu et al., 2025) removes any
#       pre-stimulus power difference between conditions from the contrast.
#       High and Low Control appear to be run in blocks, so sustained
#       block-level (pre-stimulus) differences are plausible; with a pooled
#       baseline they remain visible.
#     - Consequence for the High - Low contrast: because both conditions share
#       the same baseline, it cancels out of the difference:
#           dB_high - dB_low = 10 * log10(power_high / power_low)
#       i.e. the contrast equals the raw power ratio, including any
#       pre-stimulus difference. (With the per-condition baseline the
#       contrast is 10*log10(power_high/power_low) - 10*log10(baseline_high/
#       baseline_low).) This is equivalent, for the contrast, to using no
#       frequency-domain baseline, as Cheng et al. (2023) did.
#     - The pooled baseline is computed from all trials BEFORE balancing (as in
#       the FieldTrip script), so the condition with more trials contributes
#       more to it. It only affects the dB values of the single conditions,
#       not the High - Low contrast (see above).
#     - The baseline WINDOW is kept at -500 to -200 ms, as in
#       3_TFR_calculation_HL_MF.py. (The FieldTrip script used -300 to
#       -100 ms; only the pooling was taken over.)
#
# REFERENCES
#   Cheng, S., Ding, Z., Chen, C., Sun, W., Jiang, T., Liu, X., & Zhang, M.
#     (2023). The effect of choice on memory: The role of theta oscillations.
#     Psychophysiology, 60(12), e14390. https://doi.org/10.1111/psyp.14390
#   Cohen, M. X. (2014). Analyzing neural time series data: Theory and
#     practice. MIT Press. https://doi.org/10.7551/mitpress/9609.001.0001
#   Wu et al. (2025). [TODO: add full reference]
# ──────────────────────────────────────────────────────────────────────────────
FREQS = np.arange(2, 31, 1)            # 2 to 30 Hz in 1 Hz steps (29 frequencies): theta, alpha and beta; see (1)
N_CYCLES = np.maximum(FREQS * 0.5, 3)  # 0.5 x f cycles (Wu-like fixed window), but at least 3 cycles (as in Cheng et al., 2023); see (2)
DECIM = 10                             # Keep every 10th sample: 250 Hz -> 25 Hz (40 ms steps); see (3)
BASELINE = (-0.5, -0.2)                # Baseline window (-500 to -200 ms), ONE POOLED baseline over all trials; see (4)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

# ── Minimum-trial rule ────────────────────────────────────────────────────────
# A participant is EXCLUDED from this analysis if either cell (High Control or
# Low Control) has fewer than MIN_TRIALS clean trials. The count is taken
# BEFORE trial balancing.
MIN_TRIALS = 25

# ── Random seed for trial balancing ───────────────────────────────────────────
# Trial balancing draws random subsamples. To make results IDENTICAL on every
# run of this script, each random generator is seeded from fixed numbers:
#   (RANDOM_SEED, participant number)
# Because every participant gets its own seed, its random draws do not depend on
# which other participants were processed before it (e.g. when a
# participant is added, removed or excluded by the 25-trial rule).
RANDOM_SEED = 2026

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

def pooled_baseline(power_all_trials, times, baseline):
    """
    POOLED baseline over ALL trials (as in A2_3_freqCalc_feedback.m).
    power_all_trials: (n_trials, n_channels, n_freqs, n_times), all trials of
    both conditions. Returns the mean power in the baseline window, averaged
    over trials AND baseline time points: shape (n_channels, n_freqs, 1).
    """
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    bl_mean = power_all_trials[:, :, :, bl_mask].mean(axis=(0, 3))[:, :, np.newaxis]
    return np.maximum(bl_mean, 1e-30)

def baseline_correct_db(power, bl_mean):
    """dB baseline correction of condition-averaged power with a GIVEN baseline:
    10 * log10(power / bl_mean). Here bl_mean is the pooled baseline."""
    return 10 * np.log10(np.maximum(power, 1e-30) / bl_mean)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MAIN EFFECT OF CONTROL: TFR COMPUTATION & BASELINE CORRECTION (POOLED BASELINE)")
print("=" * 70)

# One row per participant x cell (trial counts, included yes/no).
# Printed as a summary and saved as a CSV at the end of the script.
count_rows = []

# Collects trigger problems as (participant, message); printed at the very end.
trigger_problems = []

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
        msg = f"EEG epoch count ({len(eeg_triggers)}) ≠ behavioral rows ({len(log_triggers)})."
        print(f"  ERROR: {msg}")
        trigger_problems.append((sub, msg))
    else:
        mismatch_found = False
        for i in range(len(eeg_triggers)):
            if eeg_triggers[i] != log_triggers[i]:
                msg = (f"Trigger mismatch at trial index {i}: "
                       f"EEG={eeg_triggers[i]}, Log={log_triggers[i]}")
                print(f"  ERROR: {msg}")
                trigger_problems.append((sub, msg))
                mismatch_found = True
                break
        if not mismatch_found:
            print(f"  ✓ Sanity check passed: {len(eeg_triggers)} EEG triggers match behavioral CSV.")

    cond_arr = trial_info['control_condition'].values
    
    # ── Main Effect of Control: All High Control vs All Low Control ──
    mask_high = (cond_arr == 'high')
    mask_low  = (cond_arr == 'low')
    
    n_high, n_low = int(mask_high.sum()), int(mask_low.sum())
    print(f"  Trial counts: High Control={n_high}, Low Control={n_low}")

    # ── Minimum-trial rule (MIN_TRIALS) ──────────────────────────────────────
    included = (n_high >= MIN_TRIALS) and (n_low >= MIN_TRIALS)
    count_rows.append({'participant': sub, 'cell': 'high_control', 'n_trials': n_high, 'included': included})
    count_rows.append({'participant': sub, 'cell': 'low_control',  'n_trials': n_low,  'included': included})

    if not included:
        print(f"  EXCLUDED: fewer than {MIN_TRIALS} trials in at least one cell.")
        # A file saved by an older run would let the later scripts include
        # this participant anyway, so delete it.
        if os.path.exists(out_file):
            os.remove(out_file)
            print(f"  Removed old file from a previous run: {os.path.basename(out_file)}")
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
    print("  Step 3: Trial balancing (100 iterations) & pooled baseline correction...")
    power_high = tfr.data[mask_high]
    power_low  = tfr.data[mask_low]
    
    # Fixed seed for this participant (see RANDOM_SEED above)
    rng = np.random.default_rng([RANDOM_SEED, sub])
    avg_high, avg_low, n_min = balance_trials(power_high, power_low, n_iterations=100, rng=rng)
    
    # POOLED baseline: one baseline from ALL trials (High + Low together, before
    # balancing), applied to both conditions (see (4) above).
    # tfr.data contains every trial of this participant (only S 11 / S 13 epochs).
    bl_pooled = pooled_baseline(tfr.data, tfr.times, BASELINE)
    db_high = baseline_correct_db(avg_high, bl_pooled)
    db_low  = baseline_correct_db(avg_low,  bl_pooled)
    
    roi_channels = [ch for ch in tfr.ch_names if ch.startswith(('P', 'O'))]
    
    # 5. Save Data
    # Also stores the trial counts (raw per cell + balanced) and the rule used;
    # the later scripts (5_, 6_) use cell_n_trials as a safety check.
    np.savez(
        out_file,
        ch_names=tfr.ch_names,
        times=tfr.times,
        freqs=FREQS,
        roi_channels=roi_channels,
        high_control=db_high,
        low_control=db_low,
        cell_names=np.array(['high_control', 'low_control']),
        cell_n_trials=np.array([n_high, n_low]),
        n_balanced=np.array([n_min]),
        min_trials=MIN_TRIALS,
        baseline_type='pooled'   # one baseline from all trials (this script)
    )
    print(f"  ✓ Saved to CDmem_{sub_id}_TFR_MainEffect_Control.npz (balanced to {n_min} trials/condition)")

# ══════════════════════════════════════════════════════════════════════════════
# TRIAL-COUNT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  TRIAL-COUNT SUMMARY (rule: at least {MIN_TRIALS} trials in every cell)")
print("=" * 70)

counts_df = pd.DataFrame(count_rows)
if not counts_df.empty:
    per_sub = counts_df.groupby('participant')['included'].first()
    incl = sorted(per_sub[per_sub].index.tolist())
    excl = sorted(per_sub[~per_sub].index.tolist())
    print(f"  Included: N = {len(incl)}  -> {incl}")
    print(f"  Excluded: N = {len(excl)}  -> {excl}")
    for s in excl:
        small = counts_df[(counts_df['participant'] == s) & (counts_df['n_trials'] < MIN_TRIALS)]
        details = ", ".join(f"{r.cell}={r.n_trials}" for r in small.itertuples())
        print(f"    px {s}: {details}")

    csv_file = os.path.join(output_path, "trial_counts_HL.csv")
    counts_df.to_csv(csv_file, index=False)
    print(f"  ✓ Trial counts saved to {csv_file}")

# ══════════════════════════════════════════════════════════════════════════════
# TRIGGER-CHECK SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
# All trigger problems found above, collected in one place so they do not have
# to be searched for in the per-participant output. The script does NOT stop
# on a mismatch: affected participants are still analysed, so check them!
print("\n" + "=" * 70)
print("  TRIGGER-CHECK SUMMARY (EEG triggers vs. behavioural CSV)")
print("=" * 70)
if trigger_problems:
    bad_px = sorted({p for p, _ in trigger_problems})
    print(f"  !! Problems found for {len(bad_px)} participant(s): {bad_px}")
    for p, msg in trigger_problems:
        print(f"    px {p}: {msg}")
    print("  These participants were still analysed. Check their alignment!")
else:
    print("  ✓ No trigger problems: EEG and behavioural triggers matched for all participants.")

print("\n" + "=" * 70)
print("  MAIN EFFECT OF CONTROL TFR CALCULATION (POOLED BASELINE) COMPLETE")
print("=" * 70)
