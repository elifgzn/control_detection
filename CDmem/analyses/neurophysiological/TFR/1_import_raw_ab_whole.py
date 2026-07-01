"""
1b_import_raw_alphabeta.py
==========================
Preprocessing Script for Alpha/Beta Hilbert Analysis
-----------------------------------------------------

PURPOSE:
    This script creates STIMULUS-LOCKED epochs for the alpha/beta (10-20 Hz)
    time-frequency analysis. It epochs around STIMULUS ONSET (S 11 / S 13)
    rather than movement onset (S 21-24), because we want to analyze oscillatory
    activity during the full encoding period.

    This is a companion to 1_import_raw.py, which epochs around movement onset
    for the ERP analysis. Both scripts share the same preprocessing pipeline.

WHAT IT DOES (step by step):
    1. Loads raw BrainVision EEG data (.vhdr/.vmrk/.eeg)
    2. Applies the actiCAP electrode montage (CACS-64_REF_new.bvef)
    3. Drops bad channels (if any are specified)
    4. Bandpass filters 1-40 Hz using IIR Butterworth (matches FieldTrip default)
    5. Adds back the implicit reference channel (FCz) and re-references to average
    6. Epochs around STIMULUS ONSET triggers (S 11 = low control, S 13 = high control)
       with a window of -0.5 to +5.5 seconds (covers pre-stimulus baseline + full encoding)
    6e. Computes per-trial encoding duration from S 91/S 92 triggers (agency rating
        onset = encoding end) and stores in epoch metadata
    7. Downsamples to 250 Hz
    8. Fits ICA (Picard algorithm) for later artifact removal
    9. Saves epochs + ICA to eeg2_ica_stimlocked/

INPUTS:
    - Raw EEG files: H:/PHD/control_detection/main_data/eeg/eeg1_raweeg/CDmem_XXXX.vhdr
    - Montage file: CACS-64_REF_new.bvef (in the same folder as this script)

OUTPUTS:
    - Epoch files: H:/PHD/control_detection/main_data/eeg/eeg2_ica_stimlocked/CDmem_XXXX-epo.fif
    - ICA files:   H:/PHD/control_detection/main_data/eeg/eeg2_ica_stimlocked/CDmem_XXXX-ica.fif

DIFFERENCES FROM 1_import_raw.py:
    - Epochs locked to S 11 / S 13 (stimulus onset) instead of S 21-24 (movement onset)
    - Longer epoch window: -0.5 to +5.5s instead of -2.5 to +2.5s
    - Output folder: eeg2_ica_stimlocked/ instead of eeg2_ica/

NOTE:
    The epoch window (-0.5 to +5.5s) is deliberately long to capture:
    - Pre-stimulus baseline (-0.5 to 0s)
    - Stimulus viewing (0 to ~0.5s, before mouse movement starts)
    - Movement phase (~0.5 to ~3.5s, 3 seconds of active motion)
    - Response screen (~3.5 to ~4.5s)
    - Agency rating (~4.5 to ~5.5s)
    The analysis script (3_hilbert_alphabeta.py) will select the specific
    time window of interest from these long epochs.
"""

import os
import gc
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Which participant(s) to process?
# Use a single-element list (e.g. [2]) to test one participant,
# or list(range(2, 21)) to run all of them.
# ──────────────────────────────────────────────────────────────
# plist = [4,6,7,8,9,10,12,13,14,15,16,17,19,20,21,22]

plist = [25]  # <-- change this as needed

# ──────────────────────────────────────────────────────────────
# Bad channels per participant
# Identified after visual inspection. If a channel is extremely
# noisy, add it here and re-run. Empty list = no bad channels.
# These should be the SAME bad channels as in 1_import_raw.py.
# ──────────────────────────────────────────────────────────────
bad_channels = {
    7: ['P2'],
    8: ['F3'],
    17: ['O1'],
    19: ['P3', 'TP10', 'T8', 'CP6', 'T7'],
    20: ['P3', 'T8'],
    21: ['T8'],
    22: ['TP10'],
    25: ['AF4']
}

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
input_path  = r"H:\PHD\control_detection\main_data\eeg\eeg1_raweeg"
output_path = r"H:\PHD\control_detection\main_data\eeg\eeg2_ica_stimlocked"

# Montage file — the actual actiCAP electrode layout (.bvef format)
# This is the same montage used in 1_import_raw.py.
bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

for sub in plist:

    sub_id   = f"{sub:04d}"  # e.g. 2 -> "0002", 10 -> "0010"
    filename = f"CDmem_{sub_id}.vhdr"
    filepath = os.path.join(input_path, filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue

    print(f"\n{'='*60}")
    print(f"  Processing participant {sub} — stimulus-locked epochs")
    print(f"{'='*60}")

    try:
        # ── STEP 1: Load raw BrainVision data ────────────────────────────────
        # read_raw_brainvision reads the .vhdr header, .vmrk markers, and
        # .eeg binary data together as a single continuous Raw object.
        # preload=True loads all data into RAM (needed for filtering).
        print(f"  Loading {filename}...")
        raw = mne.io.read_raw_brainvision(filepath, preload=True)

        # ── STEP 2: Apply electrode montage ──────────────────────────────────
        # The actiCAP .bvef file contains the real 3D electrode positions
        # for our specific cap layout (CACS-64). This is more accurate than
        # the generic standard_1020 template.
        montage = mne.channels.read_custom_montage(bvef_path)
        # The .bvef file labels the online reference electrode as "REF",
        # but we call it "FCz" in our data (added back later). Renaming
        # ensures FCz gets a valid position instead of NaN.
        montage.rename_channels({'REF': 'FCz'})
        raw.set_montage(montage, on_missing='warn')

        # ── STEP 3: Drop bad channels ────────────────────────────────────────
        # Bad channels are identified by visual inspection and should be the
        # same ones used in 1_import_raw.py. They will be interpolated later
        # in 2b_remove_noise_alphabeta.py using spherical spline interpolation.
        bads = bad_channels.get(sub, [])
        if bads:
            raw.drop_channels(bads)
            print(f"  Dropped bad channels: {bads}")

        # ── STEP 4: Bandpass filter (1–40 Hz) ────────────────────────────────
        # We use an IIR Butterworth filter to match FieldTrip's default.
        # This is the SAME filter used in 1_import_raw.py and in the
        # MNE_learn reference scripts.
        #
        # Why IIR Butterworth?
        #   - Matches FieldTrip's default filter type exactly
        #   - More computationally efficient than MNE's default FIR
        #   - 4th order = -80 dB/decade roll-off, good attenuation
        #
        # Why 1-40 Hz?
        #   - 1 Hz high-pass removes slow drifts
        #   - 40 Hz low-pass removes line noise (50 Hz) and muscle artifacts
        #   - Our alpha/beta band of interest (10-20 Hz) sits well within this
        raw.filter(
            l_freq=1.,
            h_freq=40.,
            method='iir',
            iir_params=dict(order=4, ftype='butter')
        )

        # ── STEP 5: Re-reference to average ──────────────────────────────────
        # Step 5a: Add back the implicit reference channel (FCz).
        #   During recording, FCz was the online reference electrode.
        #   It doesn't appear in the raw data because it's always 0 by definition.
        #   Adding it back as a flat (all zeros) channel restores it,
        #   and the re-referencing step below will recover its actual signal.
        raw.add_reference_channels('FCz')
        raw.set_montage(montage, on_missing='warn')  # re-apply so FCz gets its position

        # Step 5b: Re-reference all channels to the common average.
        #   After this, each channel = original – mean(all 65 channels).
        #   FCz goes from zeros to –mean(all), recovering its true signal.
        raw.set_eeg_reference('average')

        # ── STEP 6: Create epochs around STIMULUS ONSET ──────────────────────
        # This is the KEY DIFFERENCE from 1_import_raw.py:
        # We epoch around S 11 (stimulus onset, low control) and
        # S 13 (stimulus onset, high control) instead of S 21-24 (movement onset).

        # Step 6a: Extract all events from the BrainVision annotations.
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Step 6b: Select only stimulus onset triggers.
        #   S 11 = stimulus onset, LOW control condition
        #   S 13 = stimulus onset, HIGH control condition
        #   These triggers fire only during the TEST phase (not calibration),
        #   so we automatically exclude calibration trials.
        wanted_triggers = ['Stimulus/S 11', 'Stimulus/S 13']
        triggers = {}
        for t in wanted_triggers:
            if t in event_id:
                triggers[t] = event_id[t]
            else:
                print(f"  WARNING: trigger '{t}' not found in {filename}")

        if not triggers:
            print(f"  SKIPPED — no matching stimulus onset triggers found")
            continue

        # Step 6c: Filter events to ONLY include our target triggers.
        # This is CRITICAL so that epochs.selection (used in later matching
        # with behavioral data) corresponds to the trial index (0, 1, 2...)
        # rather than the index in the full list of all EEG markers.
        mask = np.isin(events[:, 2], list(triggers.values()))
        events = events[mask]

        # Step 6d: Create epochs with a LONG window.
        #   -0.5 s before stimulus onset: pre-stimulus baseline
        #   +5.5 s after stimulus onset: covers the full encoding period
        #     (stimulus viewing + motion phase + response + agency rating)
        #
        #   baseline=None: we do NOT apply automatic baseline correction here.
        #   Baseline correction will be applied explicitly in the analysis script
        #   (3_hilbert_alphabeta.py) to match FieldTrip conventions.
        epochs = mne.Epochs(
            raw, events, event_id=triggers,
            tmin=-0.5, tmax=5.5,
            preload=True, verbose=False,
            baseline=None  # applied later in analysis script
        )

        # ── STEP 6e: Compute per-trial encoding duration from S 91/S 92 ──────
        # S 91 = agency rating onset, LOW control condition
        # S 92 = agency rating onset, HIGH control condition
        # These triggers mark the moment stimuli go OFF-SCREEN, i.e., the end
        # of the encoding period. The encoding duration for each trial is:
        #     encoding_duration = (S 91/S 92 sample) - (S 11/S 13 sample) / sfreq
        #
        # This is stored as epoch metadata so downstream analysis scripts
        # (3_TFR.py, 3_hilbert_alphabeta.py) can use it to:
        #   - Set informed statistical time windows
        #   - Mask data after encoding ends per trial
        #   - Quality-check trial durations
        #
        # We need the FULL event list (not the filtered one) to find S 91/S 92.
        all_events_full, all_event_id_full = mne.events_from_annotations(
            raw, verbose=False
        )

        # Find S 91/S 92 event codes
        encoding_end_triggers = ['Stimulus/S 91', 'Stimulus/S 92']
        encoding_end_ids = []
        for t in encoding_end_triggers:
            if t in all_event_id_full:
                encoding_end_ids.append(all_event_id_full[t])
            else:
                print(f"  WARNING: trigger '{t}' not found — encoding duration "
                      f"will be NaN for some trials")

        encoding_end_events = all_events_full[
            np.isin(all_events_full[:, 2], encoding_end_ids)
        ]

        # For each stimulus onset event, find the NEXT S 91/S 92 event
        # and compute the time difference (= encoding duration in seconds).
        sfreq_raw = raw.info['sfreq']  # original sampling rate (1000 Hz)
        stim_onset_events = events  # these are the filtered S 11/S 13 events
        encoding_durations = np.full(len(stim_onset_events), np.nan)

        for i, stim_ev in enumerate(stim_onset_events):
            stim_sample = stim_ev[0]
            # Find encoding-end events that come AFTER this stimulus onset
            later_ends = encoding_end_events[
                encoding_end_events[:, 0] > stim_sample
            ]
            if len(later_ends) > 0:
                # Take the first one (nearest encoding end after this stimulus)
                end_sample = later_ends[0, 0]
                encoding_durations[i] = (end_sample - stim_sample) / sfreq_raw

        # Build a reverse map for trigger IDs (for metadata)
        event_id_rev = {v: int(k.split('S ')[1]) for k, v in triggers.items()}

        # Attach encoding duration as epoch metadata (pandas DataFrame).
        # This survives epoch dropping (artifact rejection in 2b) and is
        # automatically saved/loaded with the .fif file.
        epochs.metadata = pd.DataFrame({
            'encoding_duration': encoding_durations,
            'trigger_id': [event_id_rev.get(e, 0) for e in epochs.events[:, 2]]
        })

        # Print encoding duration summary
        valid_dur = encoding_durations[~np.isnan(encoding_durations)]
        if len(valid_dur) > 0:
            print(f"  Encoding durations: "
                  f"mean={np.mean(valid_dur):.2f}s, "
                  f"min={np.min(valid_dur):.2f}s, "
                  f"max={np.max(valid_dur):.2f}s")
        else:
            print(f"  WARNING: No S 91/S 92 triggers found — "
                  f"encoding durations are all NaN")

        # ── STEP 7: Downsample to 250 Hz ────────────────────────────────────
        # Reduces data size by 4x (1000 Hz → 250 Hz).
        # MNE automatically applies an anti-aliasing lowpass filter before
        # downsampling. Since we already filtered at 40 Hz (well below the
        # new Nyquist frequency of 125 Hz), this is safe.
        epochs.resample(250, verbose=False)

        # ── Print summary ────────────────────────────────────────────────────
        counts = {k: (epochs.events[:, 2] == v).sum() for k, v in triggers.items()}
        print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz")
        for trig_name, count in counts.items():
            print(f"    {trig_name}: {count} epochs")

        # ── STEP 8: Fit ICA ──────────────────────────────────────────────────
        # ICA decomposes the data into statistically independent components.
        # Components capturing eye blinks and eye movements will be identified
        # and removed in a later step (2b_remove_noise_alphabeta.py).
        #
        # We use Picard, which is mathematically equivalent to Infomax (the
        # algorithm used by FieldTrip's 'runica') but converges much faster.
        # ortho=False, extended=False = standard Infomax behavior.
        ica = ICA(
            method='picard',
            fit_params=dict(ortho=False, extended=False),
            random_state=42  # for reproducibility
        )
        ica.fit(epochs, verbose=False)
        print(f"  ICA fitted: {ica.n_components_} components")

        # ── STEP 9: Save epochs and ICA ──────────────────────────────────────
        # Save to eeg2_ica_stimlocked/ (separate from movement-onset epochs).
        # fmt='single' saves as float32 to reduce file size (sufficient precision
        # for EEG data).
        epo_name = f"CDmem_{sub_id}-epo.fif"
        ica_name = f"CDmem_{sub_id}-ica.fif"
        epochs.save(os.path.join(output_path, epo_name), overwrite=True, fmt='single')
        ica.save(os.path.join(output_path, ica_name), overwrite=True)

        print(f"  ✓ Saved {epo_name} + {ica_name} → {output_path}")

        # ── Clean up RAM ─────────────────────────────────────────────────────
        del raw, epochs, ica
        gc.collect()

    except Exception as e:
        print(f"  FAILED {filename}: {e}")
        import traceback
        traceback.print_exc()

print("\nAll done!")
