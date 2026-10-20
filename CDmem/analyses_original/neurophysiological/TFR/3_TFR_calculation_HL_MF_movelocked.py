"""
3_TFR_calculation_HL_MF_movelocked.py
=====================================
Time-Frequency Calculation for the Main Effect of Control (High vs Low), 2-30 Hz
-------------------------------------------------------------------------------
-- MOVEMENT-LOCKED VERSION --

    Copy of 3_TFR_calculation_HL_MF.py with ONE difference: the output is
    locked to MOVEMENT onset (trigger S 21-24) instead of stimulus onset. The TFR is
    computed exactly as before on the stimulus-locked epochs; then, for every
    trial, the power from -1.0 to +2.0 s around that trial's own movement
    onset is cut out (stimulus -> movement latency read from the raw .vmrk
    file). Baseline: pre-stimulus window (-500 to -200 ms before STIMULUS
    onset), separately for each condition, as in the stimulus-locked
    per-condition script. Trials with movement onset later than 2.3 s
    after the stimulus are left out (window would reach the epoch edge); the
    25-trial rule is applied afterwards. Full reasoning: section (5) below.
    Output time axis: 0 = movement onset. Everything else is identical.

PURPOSE:
    For each participant: applies a surface Laplacian, computes single-trial
    time-frequency power with Morlet wavelets (2-30 Hz: theta, alpha, beta) for
    ALL trials, compares High Control vs. Low Control (regardless of
    subsequent memory), balances trial counts, applies dB baseline correction
    per condition, and saves the condition averages to:
        CDmem_XXXX_TFR_MainEffect_Control.npz

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
    7. Baseline correction in dB, SEPARATELY FOR EACH CONDITION:
           10 * log10(condition_power / mean condition_power in -500 to -200 ms)
       per channel and frequency (per-condition baseline, following Wu et al.).
       Any pre-stimulus power difference between High and Low Control
       (e.g. block-level state differences) is therefore removed from the
       contrast.

OUTPUT:
    - Output folder: main_data/eeg/TFR/eeg4_TFR_movelocked_HL
      (to analyse these results, point the input_path of the 5_/6_ scripts
       to this folder and set their analysis time window to the
       movement-locked window, -1.0 to +2.0 s).
    - Per participant: .npz with high_control / low_control dB arrays
      (channels x 29 frequencies x time points), channel names, times, freqs,
      posterior ROI channels (P*/O*), raw trial counts per cell, balanced
      trial count, and the minimum-trial rule used.
    - trial_counts_HL.csv (all participants x cells, included yes/no).
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
output_path     = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg4_TFR_movelocked_HL"
raw_path        = r"H:\PHD\control_detection\main_data\eeg\ERP\eeg1_raweeg"   # raw .vmrk/.vhdr files: ONLY used to read each trial's stimulus -> movement latency

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
# (4) BASELINE: -500 to -200 ms, dB, separately for each condition
#     10 * log10(power / mean baseline power), per channel and frequency,
#     computed for each condition's own average (following Wu et al., 2025).
#     Consequence: pre-stimulus power differences between conditions are
#     removed from the contrast. Alternatives: one shared ("grand") baseline
#     over all trials (original FieldTrip pipeline, A2_3_freqCalc_feedback.m),
#     or no frequency-domain baseline (Cheng et al., 2023, who compared raw
#     power between conditions); both keep pre-stimulus differences visible.
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
BASELINE = (-0.5, -0.2)                # Baseline window (-500 to -200 ms), applied per condition; see (4)

# ──────────────────────────────────────────────────────────────────────────────
# (5) MOVEMENT-LOCKING — SETTINGS AND REASONING
#     (the only difference from the stimulus-locked version of this script)
# ──────────────────────────────────────────────────────────────────────────────
#   WHY
#     - Movement onset (trigger S 21-24, "motion start") was self-paced: its
#       latency after stimulus onset (S 11 / S 13) was not fixed by the task.
#       Measured from the raw .vmrk files (38 participants x 120 test trials):
#       median 0.60 s, middle 50% of trials 0.42-0.88 s, 95% below 1.6 s,
#       longest 8.4 s; it varies strongly within each participant.
#     - In stimulus-locked averages, activity tied to movement onset therefore
#       falls at a different time in every trial and is smeared out. Locking
#       to movement onset lines it up.
#     - Movement started on average ~30 ms earlier in High than in Low Control
#       (paired t-test on participant medians, p = .06), so stimulus-locked and
#       movement-locked contrasts are not simply time-shifted copies.
#
#   HOW
#     - The TFR is computed on the stimulus-locked epochs EXACTLY as in the
#       stimulus-locked script (same epochs, CSD, wavelets, decimation,
#       trial balancing, seeds, 25-trial rule).
#     - Each trial's stimulus -> movement latency is read from the raw .vmrk
#       file (movement trigger S 21-24 -> preceding stimulus trigger S 11 / S 13;
#       same logic as read_stimulus_to_movement() in 3_ERP_calculation.py).
#     - For every trial, the power from MOVE_WINDOW = -1.0 to +2.0 s around ITS
#       OWN movement onset is cut out of the stimulus-locked TFR. The time axis
#       of the output ('times') is movement-locked: 0 = movement onset.
#     - Baseline: the PRE-STIMULUS window (BASELINE, -500 to -200 ms before
#       STIMULUS onset), as in the stimulus-locked scripts, NOT a window before
#       movement onset: the period before movement onset already contains the
#       response to the stimulus, so it is not a neutral baseline.
#       Per-condition baseline (as in the stimulus-locked per-condition
#       script): each condition's own average power in that window, taken
#       from exactly the same (balanced) trials as its movement-locked power.
#
#   WHICH TRIALS FIT
#     - Wavelet power near the edges of an epoch is distorted. MNE's Morlet
#       wavelets extend +/- 5 sigma_t; at 2 Hz with 3 cycles this is
#       5 x 3 / (2*pi*2) = ~1.2 s. The stimulus-locked epochs end at +5.5 s, so
#       power is usable up to ~4.3 s after stimulus onset.
#     - The window must end (movement onset + 2.0 s) before that, so trials with
#       a stimulus -> movement latency above MAX_STIM_TO_MOVE = 2.3 s are left
#       out (~1.6% of all trials; at most 10 of 60 per condition in any
#       participant; equally often in High and Low Control). The script checks
#       this limit against the actual epoch length and wavelets.
#     - The start of the window and the pre-stimulus baseline are always far
#       enough from the epoch start (-2.5 s), because latencies are >= 0.
#     - The 25-trial rule is applied AFTER this exclusion.
#     - Very fast movements are KEPT: ~11% of trials have a latency below
#       100 ms (mostly 30-40 ms), concentrated in a few participants (most in
#       px 10, 18 and 9). In these trials the participant was probably already
#       moving when the stimulus appeared. Decision: keep them. Every trial's
#       latency is saved in the output file, so this can be checked later.
#
#   TIME RESOLUTION
#     - With DECIM = 10 the TFR has 40 ms steps, so each trial's movement onset
#       is placed on the nearest step (error at most +/- 20 ms). This is small
#       compared with the time smoothing of the wavelets (time FWHM >= ~0.19 s)
#       and random across trials. Avoiding it would require computing the TFR
#       at full time resolution (~10x more memory per participant).
# ──────────────────────────────────────────────────────────────────────────────
MOVE_WINDOW = (-1.0, 2.0)     # s around MOVEMENT onset (0 = movement onset); see (5)
MAX_STIM_TO_MOVE = 2.3        # s; trials with a later movement onset do not fit the window; see (5)
MOVE_TRIGGERS = (21, 22, 23, 24)   # movement-onset ("motion start") triggers in the .vmrk files
STIM_TRIGGERS = (11, 13)           # stimulus-onset triggers (S 11 = low, S 13 = high control)

# Half-length of the longest wavelet (MNE truncates Morlet wavelets at +/- 5 sigma_t);
# used to check that MAX_STIM_TO_MOVE keeps every window clear of the epoch edge.
WAVELET_HALF_LENGTH = float(np.max(5 * N_CYCLES / (2 * np.pi * FREQS)))   # ~1.19 s (2 Hz, 3 cycles)

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

def baseline_correct_db(power_locked, n_move):
    """
    dB baseline correction for MOVEMENT-LOCKED power (see (5)).
    power_locked: condition-averaged power (n_channels, n_freqs, n_move + n_baseline),
    as returned by movement_lock(..., baseline=BASELINE) and averaged by
    balance_trials: the first n_move samples are the movement-locked window,
    the rest are the PRE-STIMULUS baseline samples of the SAME trials.
    Returns 10 * log10(power / mean baseline power) for the movement-locked
    window only: (n_channels, n_freqs, n_move). Per-condition baseline.
    """
    bl_mean = power_locked[:, :, n_move:].mean(axis=-1, keepdims=True)
    bl_mean = np.maximum(bl_mean, 1e-30)
    return 10 * np.log10(np.maximum(power_locked[:, :, :n_move], 1e-30) / bl_mean)

# ── Movement-locking helpers (see (5)) ────────────────────────────────────────

def read_stimulus_to_movement(sub_id):
    """
    Read the raw .vmrk marker file and return, for EVERY test trial in
    recording order (= the order of the stimulus-locked epochs before artifact
    rejection, i.e. the indices used by epochs.selection):
        stim_codes : stimulus trigger of the trial (11 / 13)
        latencies  : time (s) from stimulus onset (S 11 / S 13) to the
                     following movement onset (S 21-24)
    Same logic as read_stimulus_to_movement() in 3_ERP_calculation.py, but
    organised per STIMULUS (the stimulus-locked epochs are the starting point).
    """
    import re
    vmrk_file = os.path.join(raw_path, f"CDmem_{sub_id}.vmrk")
    vhdr_file = os.path.join(raw_path, f"CDmem_{sub_id}.vhdr")
    sfreq_raw = 1000.0                       # marker positions are in raw samples
    if os.path.exists(vhdr_file):
        with open(vhdr_file, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.startswith('SamplingInterval='):
                    sfreq_raw = 1e6 / float(line.split('=')[1])   # interval in microseconds
                    break
    markers = []
    with open(vmrk_file, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = re.match(r"Mk\d+=Stimulus,S\s*(\d+),(\d+)", line)
            if m:
                markers.append((int(m.group(1)), int(m.group(2))))
    stim_codes, latencies = [], []
    for k, (code, sample) in enumerate(markers):
        if code in STIM_TRIGGERS:
            lat = np.nan
            # first marker after this stimulus that is a movement onset,
            # unless the next stimulus comes first
            for code2, sample2 in markers[k + 1:]:
                if code2 in STIM_TRIGGERS:
                    break
                if code2 in MOVE_TRIGGERS:
                    lat = (sample2 - sample) / sfreq_raw
                    break
            stim_codes.append(code)
            latencies.append(lat)
    return np.array(stim_codes), np.array(latencies)

def movement_lock(power, times, latencies, window, baseline=None):
    """
    Cut a movement-locked window out of stimulus-locked single-trial power.
    power     : (n_trials, n_channels, n_freqs, n_times), stimulus-locked
    times     : stimulus-locked time axis (s) of power
    latencies : stimulus -> movement latency (s) per trial
    window    : (start, end) in s relative to MOVEMENT onset
    baseline  : optional (start, end) in s relative to STIMULUS onset. If given,
                the baseline-window samples of each trial are appended after
                the movement-locked samples (last axis), so that trial
                balancing averages both from exactly the same trials.
    Returns (locked, move_times): locked has shape
        (n_trials, n_channels, n_freqs, n_move [+ n_baseline]);
    move_times is the movement-locked time axis (n_move,).
    Each movement onset is placed on the nearest time step (+/- dt/2).
    """
    dt = times[1] - times[0]
    k0 = int(round(window[0] / dt))
    k1 = int(round(window[1] / dt))
    offsets = np.arange(k0, k1 + 1)
    move_times = np.round(offsets * dt, 6)
    # movement onset on the stimulus-locked axis, rounded to the nearest step
    # (exact halves always rounded UP, so ties are handled the same way in every trial)
    onset_idx = np.floor((latencies - times[0]) / dt + 0.5 + 1e-9).astype(int)
    idx = onset_idx[:, None] + offsets[None, :]
    if idx.min() < 0 or idx.max() >= len(times):
        raise ValueError("Movement-locked window runs outside the epoch - check MAX_STIM_TO_MOVE.")
    locked = np.stack([power[k][..., idx[k]] for k in range(len(latencies))])
    if baseline is not None:
        bl_mask = (times >= baseline[0]) & (times <= baseline[1])
        locked = np.concatenate([locked, power[..., bl_mask]], axis=-1)
    return locked, move_times

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MAIN EFFECT OF CONTROL: TFR COMPUTATION & BASELINE CORRECTION")
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
    n_test_trials = len(trial_info)   # all test trials, before artifact rejection (for the .vmrk check)
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

    # ── 2b. Stimulus -> movement latency of every surviving epoch (see (5)) ──
    vm_stim, vm_lat = read_stimulus_to_movement(sub_id)
    if len(vm_stim) != n_test_trials or len(kept_indices) != len(eeg_triggers) \
            or not np.array_equal(vm_stim[kept_indices], eeg_triggers):
        msg = (f".vmrk stimulus triggers ({len(vm_stim)}) do not match the test trials "
               f"({n_test_trials}) / surviving epochs. Movement-locking impossible: "
               f"participant SKIPPED.")
        print(f"  ERROR: {msg}")
        trigger_problems.append((sub, msg))
        continue
    lat = vm_lat[kept_indices]                              # latency per surviving epoch (s)
    lat_ok = ~np.isnan(lat) & (lat <= MAX_STIM_TO_MOVE)     # trials whose window fits (see (5))
    lat_for_lock = np.where(lat_ok, lat, 0.0)               # left-out trials get a dummy value (never used)
    print(f"  Stimulus -> movement latency: median {np.nanmedian(lat):.2f} s, "
          f"max {np.nanmax(lat):.2f} s; {int((~lat_ok).sum())} of {len(lat)} trials left out "
          f"(> {MAX_STIM_TO_MOVE} s); {int((lat < 0.1).sum())} trials < 0.1 s (kept)")
    # Safety check: the latest window end must stay clear of the epoch edge.
    if MAX_STIM_TO_MOVE + MOVE_WINDOW[1] > epochs.tmax - WAVELET_HALF_LENGTH + 0.01:
        raise ValueError(f"MAX_STIM_TO_MOVE + window end ({MAX_STIM_TO_MOVE + MOVE_WINDOW[1]:.2f} s) "
                         f"exceeds the usable part of the epoch ({epochs.tmax - WAVELET_HALF_LENGTH:.2f} s).")

    cond_arr = trial_info['control_condition'].values
    
    # ── Main Effect of Control: All High Control vs All Low Control ──
    # Only trials whose movement-locked window fits (lat_ok, see (5))
    mask_high = (cond_arr == 'high') & lat_ok
    mask_low  = (cond_arr == 'low') & lat_ok
    n_out_high = int(((cond_arr == 'high') & ~lat_ok).sum())   # left out by the latency limit
    n_out_low  = int(((cond_arr == 'low') & ~lat_ok).sum())
    
    n_high, n_low = int(mask_high.sum()), int(mask_low.sum())
    print(f"  Trial counts (after latency limit): High Control={n_high}, Low Control={n_low} "
          f"(left out: {n_out_high} / {n_out_low})")

    # ── Minimum-trial rule (MIN_TRIALS) ──────────────────────────────────────
    included = (n_high >= MIN_TRIALS) and (n_low >= MIN_TRIALS)
    count_rows.append({'participant': sub, 'cell': 'high_control', 'n_trials': n_high,
                       'n_left_out_latency': n_out_high, 'included': included})
    count_rows.append({'participant': sub, 'cell': 'low_control',  'n_trials': n_low,
                       'n_left_out_latency': n_out_low, 'included': included})

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
    print("  Step 3: Trial balancing (100 iterations) & baseline correction...")
    # Movement-locked single-trial power (see (5)): -1.0 to +2.0 s around each
    # trial's own movement onset, cut out of the stimulus-locked TFR, followed
    # by the trial's PRE-STIMULUS baseline samples (so balancing averages both
    # from exactly the same trials).
    locked, move_times = movement_lock(tfr.data, tfr.times, lat_for_lock, MOVE_WINDOW, baseline=BASELINE)
    n_move = len(move_times)
    power_high = locked[mask_high]
    power_low  = locked[mask_low]
    
    # Fixed seed for this participant (see RANDOM_SEED above)
    rng = np.random.default_rng([RANDOM_SEED, sub])
    avg_high, avg_low, n_min = balance_trials(power_high, power_low, n_iterations=100, rng=rng)
    
    db_high = baseline_correct_db(avg_high, n_move)   # per-condition pre-stimulus baseline
    db_low  = baseline_correct_db(avg_low,  n_move)
    
    roi_channels = [ch for ch in tfr.ch_names if ch.startswith(('P', 'O'))]
    
    # 5. Save Data
    # Also stores the trial counts (raw per cell + balanced) and the rule used;
    # the later scripts (5_, 6_) use cell_n_trials as a safety check.
    np.savez(
        out_file,
        ch_names=tfr.ch_names,
        times=move_times,                     # MOVEMENT-locked time axis (0 = movement onset)
        freqs=FREQS,
        roi_channels=roi_channels,
        high_control=db_high,
        low_control=db_low,
        cell_names=np.array(['high_control', 'low_control']),
        cell_n_trials=np.array([n_high, n_low]),
        n_balanced=np.array([n_min]),
        min_trials=MIN_TRIALS,
        baseline_type='per_condition',
        time_lock='movement_onset',          # 0 = movement onset (this script); see (5)
        move_window=np.array(MOVE_WINDOW),
        max_stim_to_move=MAX_STIM_TO_MOVE,
        trial_latencies=lat,                 # stimulus -> movement latency (s) of every surviving epoch
        trial_used=lat_ok,                   # True = window fits (used); False = left out
    )
    print(f"  ✓ Saved to CDmem_{sub_id}_TFR_MainEffect_Control.npz (movement-locked, balanced to {n_min} trials/condition)")

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
print("  MAIN EFFECT OF CONTROL TFR CALCULATION (MOVEMENT-LOCKED) COMPLETE")
print("=" * 70)
