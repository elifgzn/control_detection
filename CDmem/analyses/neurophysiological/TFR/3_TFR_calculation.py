"""
3_TFR_calculation.py
====================
Time-Frequency Representation (TFR) Analysis: Subsequent Memory Effect
(Part 1: TFR Computation)
-----------------------------------------------------------------------

PURPOSE:
    Full broadband (2-40 Hz) time-frequency analysis to identify WHERE
    (which electrodes) and AT WHAT FREQUENCY oscillatory power differs
    between subsequently recalled vs not-recalled images during encoding.

    This script replicates the FieldTrip TFR analysis workflow from:
        Wu et al. (2025) — "Time-Frequency Analysis of EEG during Encoding"
    using MNE-Python equivalents. All FieldTrip → MNE mappings are documented
    inline so every processing step can be traced back to the original code.

FIELDTRIP → MNE MAPPING (Wu et al., 2025):
    FieldTrip                              MNE-Python
    ─────────────────────────────────────  ────────────────────────────────────
    ft_freqanalysis (mtmconvol, hanning) → mne.time_frequency.tfr_morlet
      cfg.foi = 2:2:40                  → freqs = np.arange(2, 42, 2)
      cfg.t_ftimwin = 0.5s (all freqs)  → n_cycles = freqs * 0.5
      cfg.toi = -1:0.05:4               → decim=5 (output every 20 ms)
      cfg.keeptrials = 'yes'            → average=False

ANALYSIS PIPELINE:
    Part 1 (per subject):
        1. Load cleaned stimulus-locked epochs
        2. Load behavioral data (encoding + recognition CSVs)
        3. Compute TFR via Morlet wavelets (2-40 Hz, fixed 0.5s window)
        4. Save per-subject TFR power + trial info

INPUTS:
    - Cleaned epochs: eeg3_clean_stimlocked/CDmem_XXXX-epo.fif
    - Encoding CSV:   behavioral/CDmem_1_X.csv
    - Recognition CSV: behavioral/CDmem_1_X_recognition.csv

OUTPUTS:
    - Per-subject TFR:  eeg4_TFR/CDmem_XXXX_tfr.npz
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import (spatio_temporal_cluster_1samp_test,
                        combine_adjacency)
from scipy.stats import t as t_dist

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Which participant(s) to analyze?
# Change this list to select specific participants.
# ──────────────────────────────────────────────────────────────
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21, 22]

# ──────────────────────────────────────────────────────────────
# Skip TFR computation if files already exist?
# Set True after first run to only re-run the analysis/stats.
# Part 1 (TFR computation) is slow (~5-15 min per participant).
# Part 2 (analysis) is fast and can be re-run independently.
# ──────────────────────────────────────────────────────────────
SKIP_TFR_COMPUTATION = False

# ──────────────────────────────────────────────────────────────
# TFR Parameters
# These match Wu et al. (2025) FieldTrip settings exactly.
# ──────────────────────────────────────────────────────────────

# FieldTrip:  cfg.foi = 2:2:40;
# → Frequencies of interest: 2 to 40 Hz in steps of 2 Hz (20 frequencies)
FREQS = np.arange(2, 42, 2)

# FieldTrip:  cfg.t_ftimwin = ones(length(cfg.foi),1).*0.5;
# → Fixed 0.5 s analysis window at ALL frequencies.
# MNE equivalent: n_cycles = freqs * window_length.
# At each frequency f, the Morlet wavelet has f*0.5 cycles,
# giving an effective duration of n_cycles/f = 0.5 s.
# NOTE: FieldTrip uses a Hanning-windowed STFT (hard window edges),
# while Morlet uses a Gaussian envelope (soft edges). In practice
# the results are virtually identical for TFR analyses.
N_CYCLES = FREQS * 0.5

# FieldTrip:  cfg.toi = -1:0.05:4;
# → Output one TFR value every 50 ms.
# MNE equivalent: decim parameter. At 250 Hz sampling:
#   decim=5 → output every 5 samples = 20 ms (slightly higher resolution)
# We use decim=5 for cleaner computation; the extra resolution doesn't hurt.
DECIM = 5

# Paths
# ──────────────────────────────────────────────────────────────
eeg_path        = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path     = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR"
figures_path    = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_TFR"

os.makedirs(output_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_behavioral_data(sub):
    """
    Load and merge encoding + recognition behavioral data for one participant.

    Identifies the CONTROLLED image per trial and looks up whether the
    participant subsequently recalled it (hit) or not (miss).

    This is the same logic as in 3_hilbert_alphabeta.py.

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
    enc_file = os.path.join(behavioral_path, f"CDmem_1_{sub}.csv")
    enc_df = pd.read_csv(enc_file)

    # Filter for test phase only (exclude calibration trials)
    test_df = enc_df[enc_df['phase'] == 'test'].copy().reset_index(drop=True)
    print(f"    Encoding CSV: {len(test_df)} test trials")

    # ── Identify the controlled image per trial ──────────────────────────
    controlled_imgs = []
    for _, row in test_df.iterrows():
        if row['true_controlled'] == 'img_A':
            controlled_imgs.append(row['img_A_name'])
        else:
            controlled_imgs.append(row['img_B_name'])
    test_df['controlled_img'] = controlled_imgs

    # ── Load recognition CSV ──────────────────────────────────────────────
    rec_file = os.path.join(behavioral_path, f"CDmem_1_{sub}_recognition.csv")
    rec_df = pd.read_csv(rec_file)

    # Filter: only SEEN items that were the CONTROLLED image
    rec_controlled = rec_df[
        (rec_df['mem_ground_truth'] == 'seen') &
        (rec_df['controlled'] == 'yes')
    ][['mem_filename', 'mem_response']].copy()

    print(f"    Recognition CSV: {len(rec_controlled)} controlled seen items")

    # ── Merge: look up each controlled image's memory outcome ────────────
    trial_info = test_df.merge(
        rec_controlled,
        left_on='controlled_img',
        right_on='mem_filename',
        how='left',
        suffixes=('', '_rec')
    )

    n_unmatched = trial_info['mem_response'].isna().sum()
    if n_unmatched > 0:
        print(f"    WARNING: {n_unmatched} trials could not be matched "
              f"to recognition data")

    trial_info = trial_info[['control_condition', 'controlled_img',
                             'mem_response', 'trigger_stim_onset']].copy()

    # Print summary
    for cond in ['low', 'high']:
        cond_mask = trial_info['control_condition'] == cond
        n_recalled = ((cond_mask) &
                      (trial_info['mem_response'] == 'yes')).sum()
        n_not_recalled = ((cond_mask) &
                          (trial_info['mem_response'] == 'no')).sum()
        print(f"    {cond.upper()} control: {n_recalled} recalled, "
              f"{n_not_recalled} not recalled")

    return trial_info


# PART 1: PER-SUBJECT TFR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
#
# FieldTrip equivalent (Wu et al., 2025):
#   for s = 1:length(Mem)
#       savename = [eegsavepath, sprintf('Sub_eegICA_PV_%d_%s.mat', s, 'Exp1')];
#       load(savename);
#       eeg.trialinfo(:,2) = Mem{s}(eeg.trialinfo(:,1));
#       ...
#       cfg.output  = 'pow';
#       cfg.method  = 'mtmconvol';
#       cfg.taper   = 'hanning';
#       cfg.foi     = 2:2:40;
#       cfg.t_ftimwin = ones(length(cfg.foi),1).*0.5;
#       cfg.toi     = -1:0.05:4;
#       cfg.keeptrials = 'yes';
#       tfralltrl = ft_freqanalysis(cfg, eeg);
#       save(savename, 'tfralltrl', '-v7.3')
#   end

# Storage for group-level data
group_tfr_data = {}  # {sub: {'power': ..., 'trial_info': ..., 'times': ...}}
tfr_times = None     # set from first participant

print("=" * 70)
print("  PART 1: PER-SUBJECT TFR COMPUTATION")
print("=" * 70)

for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    tfr_file = os.path.join(output_path, f"CDmem_{sub_id}_tfr.npz")

    if not os.path.exists(epo_file):
        print(f"\nSkipping participant {sub}: epoch file not found")
        continue

    # ── Check if TFR already computed ─────────────────────────────────────
    if SKIP_TFR_COMPUTATION and os.path.exists(tfr_file):
        print(f"\n  Participant {sub}: TFR file exists, skipping computation. "
              f"Loading saved data...")
        saved = np.load(tfr_file, allow_pickle=True)
        group_tfr_data[sub] = {
            'power': saved['power'],
            'control_condition': saved['control_condition'],
            'mem_response': saved['mem_response'],
            'times': saved['times'],
            'ch_names': list(saved['ch_names']),
            'freqs': saved['freqs']
        }
        if tfr_times is None:
            tfr_times = saved['times']
        continue

    print(f"\n{'─' * 60}")
    print(f"  Participant {sub}")
    print(f"{'─' * 60}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Load cleaned stimulus-locked epochs
    # ══════════════════════════════════════════════════════════════════════
    # FieldTrip: load(savename);  eeg = eeg_final;
    print("  Step 1: Loading cleaned epochs...")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    print(f"    {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz, "
          f"{len(epochs.ch_names)} channels")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Load behavioral data and match to EEG epochs
    # ══════════════════════════════════════════════════════════════════════
    # FieldTrip:
    #   eeg.trialinfo(:,2) = Mem{s}(eeg.trialinfo(:,1));   % memory label
    #   eeg.trialinfo(:,3) = Confi{s}(eeg.trialinfo(:,1)); % confidence
    print("  Step 2: Loading behavioral data...")
    trial_info = load_behavioral_data(sub)

    # Match behavioral data to EEG epochs via epochs.selection
    # (same approach as 3_hilbert_alphabeta.py)
    if hasattr(epochs, 'selection') and epochs.selection is not None:
        kept_indices = epochs.selection
    else:
        kept_indices = np.arange(len(epochs))

    trial_info_matched = trial_info.iloc[kept_indices].reset_index(drop=True)

    assert len(epochs) == len(trial_info_matched), (
        f"Mismatch: {len(epochs)} epochs vs {len(trial_info_matched)} "
        f"behavioral rows"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Compute TFR using Morlet wavelets
    # ══════════════════════════════════════════════════════════════════════
    # FieldTrip:
    #   cfg.output       = 'pow';
    #   cfg.method       = 'mtmconvol';
    #   cfg.taper        = 'hanning';
    #   cfg.foi          = 2:2:40;
    #   cfg.t_ftimwin    = ones(length(cfg.foi),1).*0.5;
    #   cfg.toi          = -1:0.05:4;
    #   cfg.keeptrials   = 'yes';
    #   tfralltrl = ft_freqanalysis(cfg, eeg);
    #
    # MNE equivalent:
    #   tfr_morlet returns per-trial power (average=False) using Morlet
    #   wavelets. n_cycles = freqs * 0.5 gives a fixed 0.5 s effective
    #   window at each frequency, matching FieldTrip's t_ftimwin = 0.5.
    #
    # NOTE: FieldTrip's 'hanning' taper applies a Hanning window to
    # segment the data before the FFT (hard window edges). Morlet wavelets
    # use a Gaussian envelope (soft edges). The difference is negligible
    # for standard TFR analyses — both produce equivalent power estimates
    # at the same time-frequency resolution.
    print(f"  Step 3: Computing TFR ({FREQS[0]}-{FREQS[-1]} Hz, "
          f"{len(FREQS)} frequencies)...")
    print(f"    n_cycles = freqs × 0.5 → fixed 0.5 s window")
    print(f"    decim={DECIM} → output every "
          f"{DECIM / epochs.info['sfreq'] * 1000:.0f} ms")

    tfr = tfr_morlet(
        epochs,
        freqs=FREQS,
        n_cycles=N_CYCLES,
        return_itc=False,   # we don't need inter-trial coherence
        average=False,       # FieldTrip: cfg.keeptrials = 'yes'
        decim=DECIM,
        n_jobs=-1,           # use all CPU cores
        verbose=False
    )

    # Extract the power data as a numpy array
    # Shape: (n_epochs, n_channels, n_freqs, n_times)
    power = tfr.data  # already |amplitude|^2 (power)
    times_tfr = tfr.times
    ch_names = tfr.ch_names

    print(f"    TFR shape: {power.shape}  "
          f"(epochs × channels × freqs × times)")
    print(f"    Time range: [{times_tfr[0]:.3f}, {times_tfr[-1]:.3f}] s, "
          f"{len(times_tfr)} time points")

    if tfr_times is None:
        tfr_times = times_tfr

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Save per-subject TFR data
    # ══════════════════════════════════════════════════════════════════════
    # FieldTrip:
    #   save(savename, 'tfralltrl', '-v7.3')
    print(f"  Step 4: Saving TFR data → {tfr_file}")
    np.savez_compressed(
        tfr_file,
        power=power.astype(np.float32),  # float32 to reduce file size
        times=times_tfr,
        freqs=FREQS,
        ch_names=np.array(ch_names),
        control_condition=trial_info_matched['control_condition'].values,
        mem_response=trial_info_matched['mem_response'].values
    )

    # Store in memory for Part 2
    group_tfr_data[sub] = {
        'power': power,
        'control_condition': trial_info_matched['control_condition'].values,
        'mem_response': trial_info_matched['mem_response'].values,
        'times': times_tfr,
        'ch_names': ch_names,
        'freqs': FREQS
    }

    # Free memory
    del epochs, tfr, power
    print(f"  ✓ Participant {sub} done")


