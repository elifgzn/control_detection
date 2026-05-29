"""
2b_remove_noise_alphabeta.py
=============================
Noise Removal Script for Stimulus-Locked Epochs (Alpha/Beta Analysis)
----------------------------------------------------------------------

PURPOSE:
    Removes ICA components (eye blinks, eye movements) and applies artifact
    rejection to the stimulus-locked epochs created by 1b_import_raw_alphabeta.py.

    This is the companion to 2_remove_noise.py, which cleans movement-onset epochs.
    The workflow is identical — only the input/output paths differ.

WORKFLOW (two-pass):
    Pass 1 — INSPECTION MODE:
        Run the script with empty component_exclusions for a participant.
        Inspection plots will appear (ICA topomaps, time courses, raw EEG).
        Close the plots, decide which components to exclude, and fill in
        the component_exclusions dictionary below.

    Pass 2 — CLEANING MODE:
        Run the script again. Now it applies the exclusions automatically:
        1. Removes the specified ICA components
        2. Applies baseline correction (-200 ms to 0 ms)
        3. Rejects epochs with amplitudes > ±100 µV
        4. Interpolates bad channels (spherical spline)
        5. Saves cleaned epochs to eeg3_clean_stimlocked/

IMPORTANT:
    The component exclusion decisions should be the SAME as in 2_remove_noise.py,
    because the ICA decomposition captures the same types of artifacts
    (blinks, eye movements) regardless of the epoching trigger. You can
    typically copy the exclusions directly from 2_remove_noise.py.

INPUTS:
    - Epoch files: eeg2_ica_stimlocked/CDmem_XXXX-epo.fif
    - ICA files:   eeg2_ica_stimlocked/CDmem_XXXX-ica.fif

OUTPUTS:
    - Cleaned epochs: eeg3_clean_stimlocked/CDmem_XXXX-epo.fif
    - Exclusion log:  eeg3_clean_stimlocked/exclusionrate.json
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
plist = [2]

# ──────────────────────────────────────────────────────────────
# Component exclusions — fill these in AFTER inspecting the plots.
# Keys are participant numbers (integers).
# Values are lists of component indices to remove (0-based).
# Leave the list EMPTY ([]) if you haven't decided yet; the
# script will then show the inspection plots.
#
# TIP: These are typically the SAME components as in 2_remove_noise.py,
#      because blinks/eye movements look the same regardless of
#      which trigger the epochs are locked to.
# ──────────────────────────────────────────────────────────────
component_exclusions = {
    2: [0, 1],
    3: [0, 1],
}

# ──────────────────────────────────────────────────────────────
# Bad channels per participant (same as in 1b_import_raw_alphabeta.py)
# These will be interpolated using spherical spline interpolation
# after ICA cleaning.
# ──────────────────────────────────────────────────────────────
bad_channels = {
    # Example: 3: ['P2', 'TP8'],
}

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
input_path  = r"H:\PHD\control_detection\main_data\eeg\eeg2_ica_stimlocked"
output_path = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
os.makedirs(output_path, exist_ok=True)

# Montage file — needed for spherical spline interpolation of bad channels
bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"

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
        # Note: For the Hilbert analysis, we apply a SECOND baseline correction
        # in the analysis script (3_hilbert_alphabeta.py) using -500 to -200 ms,
        # specifically on the Hilbert envelope. This initial baseline correction
        # here is on the raw voltage signal and is standard practice.
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

        # ── Spherical spline interpolation of bad channels ────────────────
        # Channels that were dropped in the preprocessing step are added back
        # as flat (zero) channels, marked as 'bad', and interpolated from
        # surrounding electrodes using spherical spline interpolation.
        bads = bad_channels.get(sub, [])
        if bads:
            print(f"  Adding back and interpolating dropped channel(s): {bads}")
            epochs_clean.add_reference_channels(bads)

            # Re-apply montage so the added channels get electrode positions.
            # Without valid positions, interpolate_bads() would fail.
            montage = mne.channels.read_custom_montage(bvef_path)
            montage.rename_channels({'REF': 'FCz'})
            epochs_clean.set_montage(montage, on_missing='ignore')

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
