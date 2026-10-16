"""
4_TFR_graphs.py
===============
Time-Frequency Heatmaps and Topoplots (2-30 Hz)
-----------------------------------------------

PURPOSE:
    Loads the preregistered 4-cell condition averages (controlled items only:
    low/high control x recalled/not recalled; *_TFR_ConditionAverages.npz from
    3_TFR_calculation_SME_PO.py), computes the grand average across
    participants, and generates standard visualizations:
        - Heatmaps (2-30 Hz x time), averaged over posterior ROI channels (P*/O*)
        - Topoplots, averaged over 2-30 Hz and 0.0-3.0 s

    Only participants that passed the 25-trial rule and have all four cells
    are used, so every grand average is based on the same participants.
    
    This functionally replicates the FieldTrip steps from A2_4_freqGraphs_feedbacktheta.m:
        - ft_freqgrandaverage (averaging across subjects)
        - ft_singleplotTFR (heatmaps averaged over ROI channels)
        - ft_topoplotTFR (topographies averaged over a time-frequency window)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg4_TFR_stimlocked"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg3_clean_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\TFR\eeg5_figures_stimlocked"
os.makedirs(figures_path, exist_ok=True)

# Same participant list as the other TFR scripts (the old hard-coded list
# [4, 6, ..., 24] was replaced; it still contained participant 24, who is
# excluded everywhere else).
plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})
# plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]   # old list

# ── Minimum-trial rule (safety check) ─────────────────────────────────────────
# The 25-trial rule is applied in the 3_TFR_calculation scripts: a participant
# with fewer than MIN_TRIALS trials in any cell of an analysis gets no output
# file for that analysis. As a second line of defence, every file loaded here
# is checked again:
#   - files without stored trial counts are from an OLD run (made before the
#     rule existed) and are skipped -> rerun the 3_TFR_calculation script;
#   - files where any cell has fewer than MIN_TRIALS trials are skipped.
# Participants without a file at all are reported as "no file" (most often
# because the calculation script excluded them).
MIN_TRIALS = 25

def passes_trial_rule(saved, sub_id, label, skipped):
    """Return True if this file passes the minimum-trial rule.
    Failures are recorded in skipped[label] as (participant, reason)."""
    if 'cell_n_trials' not in saved:
        skipped.setdefault(label, []).append((int(sub_id), "old file without trial counts"))
        return False
    names = [str(c) for c in saved['cell_names']]
    counts = [int(n) for n in saved['cell_n_trials']]
    low = [f"{c}={n}" for c, n in zip(names, counts) if n < MIN_TRIALS]
    if low:
        skipped.setdefault(label, []).append((int(sub_id), ", ".join(low)))
        return False
    return True

def participant_lines(label, used_subs, skipped):
    """Text lines listing the included / skipped participants of one analysis."""
    used = sorted(int(s) for s in used_subs)
    no_file = [s for s in plist if s not in used
               and s not in [p for p, _ in skipped.get(label, [])]]
    lines = [f"[{label}] Included participants (N={len(used)}): {used}"]
    for p, reason in sorted(skipped.get(label, [])):
        lines.append(f"[{label}]   skipped px {p}: {reason}")
    if no_file:
        lines.append(f"[{label}]   no file (excluded by the {MIN_TRIALS}-trial rule "
                     f"in 3_TFR_calculation, or missing data): {no_file}")
    return lines

trial_rule_skipped = {}   # label -> list of (participant, reason)

# Parameters matching MATLAB script `A2_4_freqGraphs_feedbacktheta.m`
Z_LIMITS = [-1.5, 1.5]  # Fixed z-limits for heatmaps and topoplots
# Z_LIMITS = [-2.5, 2.5]  # Fixed z-limits for heatmaps and topoplots (matches Wu et al)
TOPO_FREQ = (2, 30)     # Hz (full calculated range, 2-30 Hz)
TOPO_TIME = (0.0, 3.0)  # seconds (0-3 s, same window as the statistics and the other topo scripts)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & COMPUTE GRAND AVERAGES
# ══════════════════════════════════════════════════════════════════════════════
print("Loading per-subject condition averages...")

conditions = ['low_recalled', 'low_not_recalled', 'high_recalled', 'high_not_recalled']
group_data = {cond: [] for cond in conditions}
used_subs = []   # participant numbers that end up in the grand averages

times, freqs, ch_names, roi_channels = None, None, None, None
info = None

for sub in plist:
    sub_id = f"{sub:04d}"
    data_file = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages.npz")
    
    if not os.path.exists(data_file):
        continue
        
    saved = np.load(data_file, allow_pickle=True)

    # Minimum-trial rule safety check (see MIN_TRIALS above)
    if not passes_trial_rule(saved, sub_id, "4-cell ControlledOnly", trial_rule_skipped):
        continue

    # Only use participants that have ALL four cells, so that every grand
    # average below is computed over the same set of participants.
    if not all(cond in saved for cond in conditions):
        continue
    
    if times is None:
        times = saved['times']
        freqs = saved['freqs']
        ch_names = saved['ch_names'].tolist()
        roi_channels = saved['roi_channels'].tolist()
        
        # Load info object for topoplots
        epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
        if os.path.exists(epo_file):
            info = mne.io.read_info(epo_file, verbose=False)
            
    for cond in conditions:
        group_data[cond].append(saved[cond])
    used_subs.append(sub)

print("\n".join(participant_lines("4-cell ControlledOnly", used_subs, trial_rule_skipped)))

# Compute Grand Averages (ft_freqgrandaverage equivalent)
print("Computing Grand Averages...")
grand_averages = {}
for cond in conditions:
    if len(group_data[cond]) > 0:
        # Average across subjects (axis 0)
        grand_averages[cond] = np.nanmean(np.array(group_data[cond]), axis=0)
    else:
        grand_averages[cond] = None

# ══════════════════════════════════════════════════════════════════════════════
# 2. PLOT HEATMAPS (ft_singleplotTFR equivalent)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Heatmaps (ROI averaged)...")

# Find indices for ROI channels
roi_idx = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]

for cond in conditions:
    ga = grand_averages[cond]
    if ga is None: continue
    
    # Average across ROI channels (axis 0) -> shape: (n_freqs, n_times)
    heatmap_data = np.nanmean(ga[roi_idx, :, :], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        heatmap_data, 
        aspect='auto', origin='lower',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap='RdBu_r', 
        vmin=Z_LIMITS[0], vmax=Z_LIMITS[1]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontname='Times New Roman')
    ax.set_ylim(2, 30)   # calculated range: 2-30 Hz
    ax.set_title(cond.replace('_', ' ').title(), fontsize=16, fontname='Times New Roman')
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_path, f'00_heatmap_{cond}.png'), dpi=300)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT TOPOPLOTS (ft_topoplotTFR equivalent)
# ══════════════════════════════════════════════════════════════════════════════
if info is not None:
    print(f"Generating Topoplots ({TOPO_FREQ[0]}-{TOPO_FREQ[1]} Hz, {TOPO_TIME[0]}-{TOPO_TIME[1]} s)...")
    
    freq_mask = (freqs >= TOPO_FREQ[0]) & (freqs <= TOPO_FREQ[1])
    time_mask = (times >= TOPO_TIME[0]) & (times <= TOPO_TIME[1])
    
    for cond in conditions:
        ga = grand_averages[cond]
        if ga is None: continue
        
        # Extract window and average over frequencies and time -> shape: (n_channels,)
        topo_data = np.nanmean(ga[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        im, _ = mne.viz.plot_topomap(
            topo_data, info, axes=ax, show=False, 
            cmap='RdBu_r', sphere='eeglab', vlim=Z_LIMITS
        )
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Power (dB)', fontname='Times New Roman')
        
        ax.set_title(cond.replace('_', ' ').title(), fontsize=16, fontname='Times New Roman')
        
        fig.savefig(os.path.join(figures_path, f'00_topo_{cond}.png'), dpi=300)
        plt.close(fig)

print("✓ All graphs generated and saved!")
