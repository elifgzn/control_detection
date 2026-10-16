"""
2_remove_noise_ab_whole.py
==========================
Noise Removal for the Stimulus-Locked Time-Frequency Analysis
-------------------------------------------------------------

PURPOSE:
    Removes ICA components (eye blinks, eye movements, muscle) and applies
    artifact rejection to the stimulus-locked epochs created by
    1_import_raw_ab_whole.py. The cleaned epochs are the input of
    3_TFR_calculation_SME_PO.py and 3_TFR_calculation_HL_MF.py.

    Companion of 2_remove_noise.py (ERP pipeline, movement-onset epochs).

WORKFLOW (two passes):
    Pass 1 - INSPECTION MODE:
        For participants without an entry (or with an empty list) in
        component_exclusions, the script shows inspection plots: ICA
        topomaps (first 20 components), component time courses, and the
        epochs before cleaning. Close the plots, decide which components to
        exclude, and fill in component_exclusions below.
        NOTE: the percentages in the topomap titles are computed from the
        component time courses, which MNE scales to roughly equal variance,
        so every component shows ~1/(number of components). They do NOT show
        how much of the EEG a component explains (see README.md, section 3).

    Pass 2 - CLEANING MODE (participants with exclusions defined):
        1. Removes the listed ICA components
        2. Applies a voltage baseline correction (-200 to 0 ms)
        3. Rejects epochs with amplitudes beyond +/-100 uV on any channel
        4. Adds back the channels dropped in 1_import_raw_ab_whole.py and
           interpolates them (spherical splines)
        5. Saves the cleaned epochs and updates the exclusion-rate log

COMPONENT SELECTION:
    See README.md, section 3, for the criteria discussed (eye vs. muscle
    components, keeping components with an alpha peak, checking the
    spectrum before/after removal, consistency across participants).

IMPORTANT:
    The ICA used here is fitted on the stimulus-locked epochs in
    1_import_raw_ab_whole.py, i.e. it is a SEPARATE decomposition from the
    ERP pipeline's ICA (2_remove_noise.py). The same TYPES of components
    (blinks, eye movements) are expected, but component NUMBERS are not
    guaranteed to match: check the exclusions on this ICA's own plots rather
    than copying indices from 2_remove_noise.py.

INPUTS:
    - Epochs:   main_data/eeg/TFR/eeg2_ica_stimlocked/CDmem_XXXX-epo.fif
    - ICA:      main_data/eeg/TFR/eeg2_ica_stimlocked/CDmem_XXXX-ica.fif
    - Montage:  CDmem/analyses_original/neurophysiological/CACS-64_REF_new.bvef

OUTPUTS:
    - Cleaned epochs: main_data/eeg/TFR/eeg3_clean_stimlocked/CDmem_XXXX-epo.fif
    - Exclusion log:  main_data/eeg/TFR/eeg3_clean_stimlocked/exclusionrate.json
      (per participant: % rejected epochs, excluded components, bad channels)
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import read_ica

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Which participant(s) to process?
# ──────────────────────────────────────────────────────────────
plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})
# plist = [8,10,14,15,17,19,20,23,25,30,34,36,37,38,39,40,41]
# ──────────────────────────────────────────────────────────────
# Component exclusions — fill these in AFTER inspecting the plots.
# Keys are participant numbers (integers).
# Values are lists of component indices to remove (0-based).
# Leave the list EMPTY ([]) if you haven't decided yet; the
# script will then show the inspection plots.
#
# TIP: The same TYPES of components as in 2_remove_noise.py are
#      expected (blinks/eye movements look the same regardless of the
#      trigger), but this ICA is a separate decomposition, so the
#      component NUMBERS can differ. Check them on this ICA's plots.
# ──────────────────────────────────────────────────────────────
component_exclusions = {
   
    4: [0,1,2,5,11,12,14,15,16],
    6: [0,1,8,18],
    7: [0,1,2,13,15,16,17],

    8: [0,1,2,16],
    9: [0,1,2,15],
    10: [0,1,4],
    12: [0,1,9,11,12,15,19],
    13: [0,1,12,13,17,18],
    14: [0,1,2,6,7,10,15,18,19],
    15: [0,1,2,3],
    16: [0,1,2],
    17: [0,1],
    18: [0,1,2,4,5,14,16],
    19: [0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19],
    20: [0,1,2,9,13,14,16,19],
    21: [0,1,2,5,9,12,15],   
    22: [0,1,3,8,18],
    23: [0,1,6,8,9,15,18],
    
    25: [0,1,2,3,4,11,16,17,19],
    
    27: [0,1,9],
    29: [0,1,2,3,4,7,8,17,18],
    30: [0,1,10,11,18],
    31: [0,1,2,6,7,8],
    32: [0,1,2,11,15],
    33: [0,1,2,11,17,19],
    34: [0,1,13,16,19],
    35: [0,1,15,17,19],
    36: [0,1,2,3],
    37: [0,1,2,3,4,8],
    38: [0,1,2,3,5,18,19],
    39: [0,1,2,3,4,6,16],
    40: [0,1,6,10,12,16,19],
    41: [0,1,2,3,10,15,16,18],
    42: [0,1,2,3,12],
    44: [0,1,2,3,11,18],
    48: [0,1,2,3,4,9,12,13,17,18],
    49: [0,1,2,19],
    50: [0,1,2,3,4,12,15,16,19]
    



}

# ──────────────────────────────────────────────────────────────
# Bad channels per participant (must be the same as in 1_import_raw_ab_whole.py)
# These will be interpolated using spherical spline interpolation
# after ICA cleaning.
# ──────────────────────────────────────────────────────────────
bad_channels = {
   7: ['P2'],
    8: ['F3','F7'],
    10: ['Fp2'],
    14: ['Fp2'],
    15: ['P3'],
    17: ['O1','Iz'],
    18: ['Fp1','Fp2'],
    19: ['P3', 'TP9', 'TP10', 'T8', 'CP6', 'T7'],
    20: ['P3', 'T8','P1','Fz','Iz'],
    21: ['T8'],
    22: ['TP10'],
    23: ['FT9'],
    25: ['AF4','FP1', 'FP2'],
    27: ['C3'],
    29: ['AFz','T7'],
    31: ['Fp1','Fp2'],
    32: ['F2','FT8'],
    34: ['AFz','F1','FT8'],
    36: ['FT8','F6'],
    38: ['O1'],
    39: ['O1'],
    40: ['O1'],
    41: ['O1','AFz'],
    42: ['C5','FC2','FC6', 'TP7'],
    43: ['C5'],
    46:['C5', 'TP7'],
    48: ['AFz','F1'],
    50: ['F6','FT8']







}

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
input_path  = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg2_ica_stimlocked"
output_path = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg3_clean_stimlocked"
os.makedirs(output_path, exist_ok=True)

# Montage file — needed for spherical spline interpolation of bad channels
bvef_path = r"H:\PHD\control_detection\CDmem\analyses_original\neurophysiological\CACS-64_REF_new.bvef"

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

for sub in plist:

    sub_id   = f"{sub:04d}"
    epo_file = os.path.join(input_path, f"CDmem_{sub_id}-epo.fif")
    ica_file = os.path.join(input_path, f"CDmem_{sub_id}-ica.fif")

    # Check that both files exist
    if not os.path.exists(epo_file):
        print(f"Epochs file not found: {epo_file}")
        continue
    if not os.path.exists(ica_file):
        print(f"ICA file not found: {ica_file}")
        continue

    print(f"\n{'='*60}")
    print(f"  Participant {sub}  (sub_id = {sub_id})")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    ica    = read_ica(ica_file)

    print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz, "
          f"{ica.n_components_} ICA components")

    # ── Check whether exclusions have been defined ────────────────────────
    excl = component_exclusions.get(sub, None)
    needs_inspection = (excl is None) or (len(excl) == 0)

    if needs_inspection:
        # ──────────────────────────────────────────────────────────────────
        # INSPECTION MODE
        # Show plots so you can decide which components to exclude.
        # After closing the plots, fill in component_exclusions above
        # and re-run the script.
        # ──────────────────────────────────────────────────────────────────
        print(f"  No exclusions defined yet → showing inspection plots.")
        print(f"  Close all plots, fill in component_exclusions[{sub}], "
              f"then re-run.\n")

        # Compute variance explained by each component (for labeling)
        sources   = ica.get_sources(epochs)
        src_data  = sources.get_data()
        comp_var  = np.var(src_data, axis=(0, 2))
        var_pct   = (comp_var / comp_var.sum()) * 100

        # 1) ICA topoplots — show spatial patterns of each component
        #    Blink components typically show frontal focus (Fp1/Fp2).
        #    Eye movement components show lateral frontal patterns.
        n_plot = min(20, ica.n_components_)
        figs   = ica.plot_components(picks=range(n_plot), show=False,
                                     title=f"ICA topomaps — participant {sub}")
        if not isinstance(figs, list):
            figs = [figs]
        for fig in figs:
            for ax in fig.axes:
                title = ax.get_title()
                if title.startswith('ICA'):
                    comp_idx = int(title[3:])
                    ax.set_title(f"{title} ({var_pct[comp_idx]:.1f}%)")
        for fig in figs:
            fig.show()

        # 2) Component time courses — scrollable view of each component
        ica.plot_sources(epochs, title=f"ICA sources — participant {sub}")

        # 3) Raw EEG — for visual comparison before/after cleaning
        epochs.plot(
            n_channels=20,
            scalings=dict(eeg=100e-6),
            title=f"EEG data BEFORE component removal — participant {sub}"
        )

        plt.show()

    else:
        # ──────────────────────────────────────────────────────────────────
        # CLEANING MODE
        # Apply the component exclusions and save cleaned data.
        # ──────────────────────────────────────────────────────────────────
        print(f"  Excluding components: {excl}")

        # ── Remove ICA components ─────────────────────────────────────────
        # This subtracts the contribution of the specified components from
        # the data, effectively removing blink/eye movement artifacts while
        # preserving brain signals.
        ica.exclude = excl
        epochs_clean = ica.apply(epochs.copy())

        # ── Baseline correction ───────────────────────────────────────────
        # Subtracts the mean of the -200 ms to 0 ms window from every epoch.
        # This removes any DC offset that varies across trials.
        # Applied after ICA (before artifact rejection).
        #
        # Note: the time-frequency analysis applies a SECOND, separate baseline
        # in 3_TFR_calculation_*.py (-500 to -200 ms, dB, per condition, on
        # wavelet power). This initial baseline correction here is on the raw
        # voltage signal and is standard practice.
        epochs_clean.apply_baseline(baseline=(-0.20, 0.0))
        print(f"  ✓ Baseline correction applied (−200 ms to 0 ms)")

        # ── Artifact rejection (amplitude threshold) ──────────────────────
        # Reject any epoch where ANY channel at ANY time point exceeds ±100 µV.
        # This catches residual muscle artifacts, electrode pops, and other
        # transient noise that ICA didn't fully remove.
        #
        # MNE works in Volts internally, so 100 µV = 100e-6 V.
        n_before = len(epochs_clean)

        data = epochs_clean.get_data(copy=False)  # (epochs, channels, times)
        is_bad = np.any((data > 100e-6) | (data < -100e-6), axis=(1, 2))
        bad_indices = np.where(is_bad)[0]
        epochs_clean.drop(bad_indices)

        n_after   = len(epochs_clean)
        n_dropped = n_before - n_after
        pct_dropped = round((n_dropped / n_before) * 100, 2)

        # Quality guide for epoch rejection rates:
        #   0–5 %   → very good
        #   5–10 %  → good
        #   10–15 % → acceptable
        #   > 15 %  → consider removing more ICA components or a channel
        if pct_dropped <= 5:
            quality = "very good"
        elif pct_dropped <= 10:
            quality = "good"
        elif pct_dropped <= 15:
            quality = "acceptable"
        else:
            quality = "HIGH — consider removing more components or a bad channel"

        print(f"  Noise exclusions: {n_dropped} / {n_before} trials "
              f"({pct_dropped} %)  [{quality}]")

        # ── Optional: browse post-ICA data ────────────────────────────────
        show_browser = False
        if show_browser:
            epochs_clean.plot(
                n_channels=20,
                scalings=dict(eeg=100e-6),
                title=f"EEG data AFTER component removal — participant {sub}"
            )
            plt.show()

        # ── Spherical spline interpolation ──────────────────────────────────
        # Replaces channels that were dropped during preprocessing (1b script).
        
        # ──────────────────────────────────────────────────────────
        # CRITICAL: Add back missing channels FIRST, then apply montage 
        # so that the newly added channels get their correct 3D positions!
        # Without positions, interpolate_bads() interpolates them at [0,0,0].
        # ──────────────────────────────────────────────────────────
        bads = bad_channels.get(sub, [])
        if bads:
            print(f"  Adding back dropped channel(s): {bads}")
            epochs_clean.add_reference_channels(bads)

        montage = mne.channels.read_custom_montage(bvef_path)
        montage.rename_channels({'REF': 'FCz'})
        epochs_clean.set_montage(montage, on_missing='ignore')

        if bads:
            epochs_clean.info['bads'] = bads
            epochs_clean.interpolate_bads(reset_bads=True)
            print(f"  ✓ Spherical spline interpolation done")
        else:
            print(f"  No bad channels to interpolate")

        # ── Save cleaned epochs ───────────────────────────────────────────
        out_file = os.path.join(output_path, f"CDmem_{sub_id}-epo.fif")
        epochs_clean.save(out_file, overwrite=True)
        print(f"  ✓ Cleaned epochs saved → {out_file}")

        # ── Exclusion-rate log ────────────────────────────────────────────
        # Keeps a running JSON file with per-participant stats.
        # Each run updates the entry for the current participant;
        # entries for other participants are preserved.
        log_file = os.path.join(output_path, "exclusionrate.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                noiserate = json.load(f)
        else:
            noiserate = {}
        noiserate[str(sub)] = {
            "exclusion_rate": pct_dropped,
            "excluded_components": excl,
            "bad_channels": bad_channels.get(sub, [])
        }
        with open(log_file, 'w') as f:
            json.dump(noiserate, f, indent=2)
        print(f"  ✓ Exclusion rate logged → {log_file}")

        print(f"\n  DONE! Participant {sub} fully processed.")
