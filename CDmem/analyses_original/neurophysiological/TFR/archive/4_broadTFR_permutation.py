"""
4_TFR_permutation.py
====================
Time-Frequency Representation (TFR) Analysis: Subsequent Memory Effect
(Part 2: Group-Level Permutation Test)
-----------------------------------------------------------------------

PURPOSE:
    Full broadband (2-40 Hz) time-frequency analysis to identify WHERE
    (which electrodes) and AT WHAT FREQUENCY oscillatory power differs
    between subsequently recalled vs not-recalled images during encoding.

    This script replicates the FieldTrip TFR analysis workflow from:
        Wu et al. (2025) — "Time-Frequency Analysis of EEG during Encoding"
    using MNE-Python equivalents. All FieldTrip → MNE mappings are documented
    inline so every processing step can be traced back to the original code.

    Three statistical contrasts:
    1. Recalled vs Not-recalled (collapsed across control conditions)
    2. Recalled vs Not-recalled (LOW motor control only)
    3. Recalled vs Not-recalled (HIGH motor control only)

FIELDTRIP → MNE MAPPING (Wu et al., 2025):
    FieldTrip                              MNE-Python
    ─────────────────────────────────────  ────────────────────────────────────
    ft_freqdescriptives                  → np.nanmean(power, axis=0)
    ft_freqbaseline (db)                 → 10 * log10(power / baseline_mean)
    TFR_balance_trial(A, B, 100)         → balance_trials() [custom]
    ft_freqstatistics (cluster)          → spatio_temporal_cluster_1samp_test
      cfg.method = 'montecarlo'         → n_permutations=1000
      cfg.statistic = 'depsamplesT'     → one-sample t on within-sub diff
      cfg.correctm = 'cluster'          → adjacency via combine_adjacency
      cfg.clusteralpha = 0.05           → threshold = t_crit at p=0.05
      cfg.clusterstatistic = 'maxsum'   → default (sum of t in cluster)
      cfg.alpha = 0.025                 → p < 0.05 (MNE two-tailed p-values)
      cfg.tail = 0                      → tail=0
      cfg.minnbchan = 3                 → (no direct equivalent; noted below)
      cfg.neighbours                    → find_ch_adjacency + combine_adjacency
    ft_singleplotTFR                     → plt.pcolormesh
    ft_topoplotTFR                       → mne.viz.plot_topomap

ANALYSIS PIPELINE:
    Part 2 (group level, per contrast):
        1. Load per-subject TFR
        2. Split trials by memory outcome (± control condition)
        3. Trial-balance conditions (100 random subsamples)
        4. Average across trials per condition
        5. Baseline correct in dB (-0.5 to -0.2 s)
        6. Cluster-based permutation test (channel × frequency × time)
        7. Publication-ready figures (TFR maps + topoplots)

INPUTS:
    - Per-subject TFR:  eeg4_TFR/CDmem_XXXX_tfr.npz

OUTPUTS:
    - Group statistics:  eeg4_TFR/stat_*.npz
    - Figures:           eeg5_figures_TFR/*.png, *.svg
    - Text reports:      eeg5_figures_TFR/*.txt
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
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]
# TFR Parameters
# These match Wu et al. (2025) FieldTrip settings exactly.
# ──────────────────────────────────────────────────────────────

# FieldTrip:  cfg.foi = 2:2:40;
# → Frequencies of interest: 2 to 40 Hz in steps of 2 Hz (20 frequencies)
FREQS = np.arange(2, 42, 2)
# Baseline correction parameters
# ──────────────────────────────────────────────────────────────
# FieldTrip:  cfg.baseline = [-1, -0.5]; cfg.baselinetype = 'db';
#
# ADAPTATION FOR CDmem:
# Wu et al. (2025) use [-1, -0.5] s as the baseline window. However,
# our stimulus-locked epochs start at -0.5 s (the fixation cross period
# before stimulus onset). Going earlier than -0.5 s would bleed into
# the PREVIOUS TRIAL's agency rating response period — a cognitively
# active period that is NOT suitable as a neutral baseline.
# Therefore we use [-0.5, -0.2] s: a clean pre-stimulus window within
# the fixation period, before any visual processing starts.
BASELINE = (-0.5, -0.2)

# ──────────────────────────────────────────────────────────────
# Cluster-based permutation test parameters
# ──────────────────────────────────────────────────────────────

# FieldTrip:  cfg.numrandomization = 1000;
N_PERMUTATIONS = 1000

# FieldTrip:  cfg.tail = 0; cfg.clustertail = 0;
# → Two-sided test (positive AND negative clusters)
TAIL = 0

# FieldTrip:  cfg.alpha = 0.025;
# → With tail=0 in FieldTrip, alpha=0.025 per tail → total α=0.05.
# MNE returns two-tailed p-values when tail=0, so clusters are
# significant at p < 0.05. This is equivalent.
CLUSTER_P_THRESHOLD = 0.05

# FieldTrip:  cfg.clusteralpha = 0.05;
# → Uncorrected p-value threshold for forming clusters.
# We convert this to a t-value threshold based on degrees of freedom.
CLUSTER_ALPHA = 0.05

# FieldTrip:  cfg.latency = [0, 4];
# → Time window for the statistical test.
# For CDmem, the encoding period is from stimulus onset (0 s) to
# approximately 3.5-4.5 s (agency rating onset). We use [0, 4] s
# to match Wu et al. (2025).
STAT_TMIN = 0.0
STAT_TMAX = 4.0

# FieldTrip:  cfg.frequency = [2, 40];
# → Frequency range for the statistical test (all frequencies in our range).
STAT_FMIN = 2
STAT_FMAX = 40

# FieldTrip:  cfg.minnbchan = 3;
# → Minimum number of neighboring channels required for a candidate
#   point to be included in a cluster. This prevents spurious clusters
#   from isolated noisy channels.
# NOTE: MNE does not have a direct equivalent of minnbchan. Clusters
# are formed purely based on adjacency. This is a minor difference
# that does not significantly affect results in practice, especially
# with a reasonable cluster-forming threshold.
# (Documented here for transparency; no action needed.)

# FieldTrip:  rng(2024);
SEED = 2024

# FieldTrip:  TFR_balance_trial(A, B, 100);
# → Number of random subsamples for trial balancing.
N_BALANCE_ITER = 100

# Paths
# ──────────────────────────────────────────────────────────────
eeg_path        = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
behavioral_path = r"H:\PHD\control_detection\main_data\behavioral"
output_path     = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR"
figures_path    = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_TFR"

os.makedirs(output_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Montage file — needed for channel adjacency in the cluster test
bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"

# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def balance_trials(tfr_A, tfr_B, n_iterations=100, rng=None):
    """
    Balance trial counts between two conditions by random subsampling.

    FieldTrip equivalent: TFR_balance_trial(A, B, 100)

    The condition with MORE trials is randomly subsampled (without replacement)
    to match the trial count of the condition with FEWER trials. This is
    repeated n_iterations times, and the result is averaged across all
    iterations. This produces a stable estimate that is not biased by
    different trial counts.

    Parameters
    ----------
    tfr_A : np.ndarray, shape (n_A, n_channels, n_freqs, n_times)
        Per-trial TFR power for condition A.
    tfr_B : np.ndarray, shape (n_B, n_channels, n_freqs, n_times)
        Per-trial TFR power for condition B.
    n_iterations : int
        Number of random subsamples (default: 100).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    avg_A : np.ndarray, shape (n_channels, n_freqs, n_times)
        Trial-balanced average for condition A.
    avg_B : np.ndarray, shape (n_channels, n_freqs, n_times)
        Trial-balanced average for condition B.
    n_balanced : tuple (int, int)
        Number of trials used per condition in each subsample.
    """
    n_A, n_B = len(tfr_A), len(tfr_B)
    n_min = min(n_A, n_B)

    if rng is None:
        rng = np.random.default_rng()

    # Accumulate averages across iterations
    sum_A = np.zeros(tfr_A.shape[1:], dtype=np.float64)
    sum_B = np.zeros(tfr_B.shape[1:], dtype=np.float64)

    for _ in range(n_iterations):
        # Subsample each condition to n_min trials
        idx_A = rng.choice(n_A, size=n_min, replace=False)
        idx_B = rng.choice(n_B, size=n_min, replace=False)

        # Average within this subsample and accumulate
        sum_A += tfr_A[idx_A].mean(axis=0)
        sum_B += tfr_B[idx_B].mean(axis=0)

    # Average across all iterations
    avg_A = sum_A / n_iterations
    avg_B = sum_B / n_iterations

    return avg_A, avg_B, (n_min, n_min)


def baseline_correct_db(power, times, baseline):
    """
    Apply decibel (dB) baseline correction to TFR power.

    FieldTrip equivalent:
        cfg.baseline     = [-1, -0.5];
        cfg.baselinetype = 'db';
        tfr = ft_freqbaseline(cfg, tfr);

    The dB conversion is:
        power_db = 10 * log10(power / mean(power_in_baseline))

    This expresses power as a ratio relative to the baseline period,
    in decibels. Positive values = power increase relative to baseline.
    Negative values = power decrease (desynchronization).

    Parameters
    ----------
    power : np.ndarray, shape (n_channels, n_freqs, n_times)
        Absolute TFR power (NOT yet baseline-corrected).
    times : np.ndarray, shape (n_times,)
        Time axis in seconds.
    baseline : tuple (float, float)
        Baseline window (start, end) in seconds.

    Returns
    -------
    power_db : np.ndarray, same shape as power
        Baseline-corrected TFR power in decibels.
    """
    # Find time indices within the baseline window
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])

    # Mean power during baseline, per channel and frequency
    # Shape: (n_channels, n_freqs, 1) for broadcasting
    bl_mean = power[:, :, bl_mask].mean(axis=-1, keepdims=True)

    # Avoid division by zero / log of zero
    bl_mean = np.maximum(bl_mean, 1e-30)

    # dB conversion: 10 * log10(power / baseline_mean)
    power_db = 10 * np.log10(power / bl_mean)

    return power_db


def _get_cluster_inds(cluster):
    """
    Return a 1-D integer array of indices for *cluster*.

    MNE 1.11+ returns clusters as tuples of slices, e.g. (slice(32, 45),),
    regardless of out_type. This helper handles all formats.

    Reused from 5_ERP_permutation.py and 3_hilbert_alphabeta.py.
    """
    if isinstance(cluster, tuple):
        # Unpack the tuple — each element corresponds to one dimension
        # For out_type='mask', cluster is a boolean array
        # For out_type='indices', cluster may be a tuple of arrays/slices
        if len(cluster) == 1:
            obj = cluster[0]
            if isinstance(obj, slice):
                return np.arange(obj.start, obj.stop)
            else:
                return np.asarray(obj).ravel()
        else:
            # Multi-dimensional cluster — return as tuple of arrays
            return tuple(
                np.arange(s.start, s.stop) if isinstance(s, slice)
                else np.asarray(s).ravel()
                for s in cluster
            )
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return np.flatnonzero(cluster)
    return np.flatnonzero(np.asarray(cluster))


# PART 2: GROUP-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# FieldTrip equivalent (Wu et al., 2025):
#   rng(2024);
#   for s = 1:20
#       load(savename, 'tfralltrl')
#       ...
#       % Sep by Memory
#       cfg.trials = find(tfrfinal.trialinfo(:,2)==1);  A = ft_selectdata(...);
#       cfg.trials = find(tfrfinal.trialinfo(:,2)==0);  B = ft_selectdata(...);
#       [tfr_C1, tfr_C2, n_trl] = TFR_balance_trial(A, B, 100);
#       ...
#       % Baseline correction
#       cfg.baseline     = [-1, -0.5];
#       cfg.baselinetype = 'db';
#       tfr_C1_ave{s} = ft_freqbaseline(cfg, tfr_C1_raw{s});
#       ...
#   end

print(f"\n{'=' * 70}")
print("  PART 2: GROUP-LEVEL ANALYSIS")
print(f"{'=' * 70}")

group_tfr_data = {}
tfr_times = None

# ── LOAD TFR DATA FROM DISK ──────────────────────────────────────────
# In the original combined script (3_TFR.py), the 'group_tfr_data' dictionary
# was populated during Part 1 and held in memory for Part 2.
# Since this script is now standalone, we must load the individual
# participant '.npz' files saved by 3_TFR_calculation.py back into memory.
print("  Loading per-subject TFR data...")
for sub in plist:
    sub_id = f"{sub:04d}"
    tfr_file = os.path.join(output_path, f"CDmem_{sub_id}_tfr.npz")
    
    if not os.path.exists(tfr_file):
        print(f"    Warning: Missing {tfr_file}")
        continue
        
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

n_subs = len(group_tfr_data)
if n_subs < 2:
    print(f"\n  Only {n_subs} participant(s) processed. Need at least 2 for "
          f"group statistics. Exiting.")
    sys.exit(0)

print(f"  {n_subs} participants loaded: {list(group_tfr_data.keys())}")

# ── Get channel info for adjacency matrix ─────────────────────────────────
# We need the epoch Info object for find_ch_adjacency.
# Load one participant's epochs to get the channel layout.
first_sub = list(group_tfr_data.keys())[0]
first_sub_id = f"{first_sub:04d}"
info_epochs = mne.read_epochs(
    os.path.join(eeg_path, f"CDmem_{first_sub_id}-epo.fif"),
    preload=False, verbose=False
).info
stat_ch_names = group_tfr_data[first_sub]['ch_names']
n_channels_stat = len(stat_ch_names)
n_freqs = len(FREQS)
n_times_full = len(tfr_times)

print(f"  Channels: {n_channels_stat}, Frequencies: {n_freqs}, "
      f"Time points: {n_times_full}")

# ── Build adjacency matrix for cluster test ───────────────────────────────
# FieldTrip:
#   load([datapath, 'saved_edited_neighbours.mat'])
#   cfg.neighbours = neighbours;
#
# MNE equivalent:
#   1. find_ch_adjacency() → channel adjacency from electrode positions
#   2. combine_adjacency() → combine channel adj. with frequency grid adj.
#   The result is a single adjacency matrix over the channel×frequency
#   product space. Time adjacency is handled automatically by
#   spatio_temporal_cluster_1samp_test (adjacent time points are neighbors).
print("  Building adjacency matrix (channel × frequency)...")


bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"
montage = mne.channels.read_custom_montage(bvef_path)
montage.rename_channels({'REF': 'FCz'})
info_epochs.set_montage(montage, on_missing='ignore')

# Channel adjacency from the EEG montage
# This creates a sparse matrix where entry (i,j) = 1 if channels i and j
# are spatial neighbors (based on electrode positions + Delaunay triangulation).
ch_adjacency, ch_adj_names = mne.channels.find_ch_adjacency(
    info_epochs, ch_type='eeg'
)
ch_adj_names = list(ch_adj_names)
print(f"    Channel adjacency: {ch_adjacency.shape}, "
      f"{ch_adjacency.nnz} connections")

# ── Align channels between adjacency and TFR data ────────────────────────
ch_names_list = list(stat_ch_names) if not isinstance(stat_ch_names, list) else stat_ch_names
stat_ch_names = [ch for ch in ch_names_list if ch in ch_adj_names]
n_channels_stat_stat = len(stat_ch_names)

if n_channels_stat_stat < n_channels_stat:
    excluded = set(ch_names_list) - set(stat_ch_names)
    print(f"    NOTE: {n_channels_stat - n_channels_stat_stat} channel(s) excluded "
          f"from cluster test (no adjacency info): {excluded}")
    # Reorder adjacency rows/cols to match stat_ch_names order
    adj_pick = [ch_adj_names.index(ch) for ch in stat_ch_names]
    ch_adjacency = ch_adjacency[adj_pick][:, adj_pick]

# Build pick indices to subset TFR data → only adjacency-matched channels
tfr_ch_pick = np.array([ch_names_list.index(ch) for ch in stat_ch_names])

# Combine channel adjacency with frequency adjacency
# Frequency adjacency is grid-based: each frequency is adjacent to its
# immediate neighbors (e.g., 8 Hz is adjacent to 6 Hz and 10 Hz).
# combine_adjacency(ch_adj, n_freqs) creates the Kronecker product:
#   adj_combined[i*n_freqs + f1, j*n_freqs + f2] = 1
#   if channels i,j are neighbors AND frequencies f1,f2 are neighbors.
tfr_adjacency = combine_adjacency(ch_adjacency, n_freqs)
print(f"    Combined adjacency (ch × freq): {tfr_adjacency.shape}, "
      f"{tfr_adjacency.nnz} connections")

# ── Compute cluster-forming threshold ─────────────────────────────────────
# FieldTrip:
#   cfg.clusteralpha = 0.05;
# → The t-value threshold at which uncorrected p < clusteralpha.
# For a two-tailed test, this is t_crit for p/2 with df = n_subs - 1.
df = n_subs - 1
t_threshold = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df)
print(f"  Cluster-forming threshold: t = ±{t_threshold:.3f} "
      f"(uncorrected p < {CLUSTER_ALPHA}, df = {df})")

# ── Find time indices for the statistical test window ─────────────────────
# FieldTrip:  cfg.latency = [0, 4];
stat_time_mask = (tfr_times >= STAT_TMIN) & (tfr_times <= STAT_TMAX)
stat_time_idx = np.where(stat_time_mask)[0]
n_times_stat = len(stat_time_idx)
print(f"  Statistical time window: [{STAT_TMIN}, {STAT_TMAX}] s → "
      f"{n_times_stat} time points")

# ── Find frequency indices for the statistical test ───────────────────────
# FieldTrip:  cfg.frequency = [2, 40];
stat_freq_mask = (FREQS >= STAT_FMIN) & (FREQS <= STAT_FMAX)
stat_freq_idx = np.where(stat_freq_mask)[0]
n_freqs_stat = len(stat_freq_idx)
print(f"  Statistical freq range: [{STAT_FMIN}, {STAT_FMAX}] Hz → "
      f"{n_freqs_stat} frequencies")

# ══════════════════════════════════════════════════════════════════════════════
# RUN THREE CONTRASTS
# ══════════════════════════════════════════════════════════════════════════════
#
# FieldTrip (Wu et al., 2025):
#   [stat_mem] = ft_freqstatistics(cfg, tfr_C1_ave{subset}, tfr_C2_ave{subset});
#
# We extend this to three contrasts:
#   1. Recalled vs Not-recalled (all trials)
#   2. Recalled vs Not-recalled (LOW control only)
#   3. Recalled vs Not-recalled (HIGH control only)

rng = np.random.default_rng(SEED)

contrasts = [
    {
        'name': 'all',
        'title': 'Recalled vs Not-Recalled (All Trials)',
        'filter_condition': None,  # no filtering — use all trials
        'fig_base': 'TFR_recalled_vs_not_recalled_all'
    },
    {
        'name': 'low',
        'title': 'Recalled vs Not-Recalled (LOW Control)',
        'filter_condition': 'low',
        'fig_base': 'TFR_recalled_vs_not_recalled_low'
    },
    {
        'name': 'high',
        'title': 'Recalled vs Not-Recalled (HIGH Control)',
        'filter_condition': 'high',
        'fig_base': 'TFR_recalled_vs_not_recalled_high'
    }
]

for contrast in contrasts:
    cname = contrast['name']
    ctitle = contrast['title']
    cfilt = contrast['filter_condition']
    cfig = contrast['fig_base']

    print(f"\n{'═' * 70}")
    print(f"  CONTRAST: {ctitle}")
    print(f"{'═' * 70}")

    # ── Per-subject: split, balance, average, baseline-correct ────────────
    # FieldTrip:
    #   cfg.trials = find(tfrfinal.trialinfo(:,2)==1);
    #   A = ft_selectdata(cfg, tfrfinal);
    #   cfg.trials = find(tfrfinal.trialinfo(:,2)==0);
    #   B = ft_selectdata(cfg, tfrfinal);
    #   [tfr_C1, tfr_C2, n_trl] = TFR_balance_trial(A, B, 100);
    #   tfr_C1_raw{s} = tfr_C1;
    #   tfr_C2_raw{s} = tfr_C2;
    #   ...
    #   cfg.baseline     = baseline_t;
    #   cfg.baselinetype = 'db';
    #   tfr_C1_ave{s} = ft_freqbaseline(cfg, tfr_C1_raw{s});

    sub_recalled_db = []     # per-subject baseline-corrected TFR (recalled)
    sub_not_recalled_db = [] # per-subject baseline-corrected TFR (not recalled)
    sub_list_valid = []      # participants with enough trials

    for sub in group_tfr_data:
        d = group_tfr_data[sub]
        power_all = d['power']        # (n_epochs, n_chans, n_freqs, n_times)
        cond_all = d['control_condition']
        mem_all = d['mem_response']

        # ── Filter by control condition (if applicable) ──────────────────
        if cfilt is not None:
            trial_mask = (cond_all == cfilt)
        else:
            trial_mask = np.ones(len(cond_all), dtype=bool)

        # ── Split by memory outcome ──────────────────────────────────────
        recalled_mask = trial_mask & (mem_all == 'yes')
        not_recalled_mask = trial_mask & (mem_all == 'no')

        n_recalled = recalled_mask.sum()
        n_not_recalled = not_recalled_mask.sum()

        print(f"  Sub {sub}: {n_recalled} recalled, "
              f"{n_not_recalled} not recalled", end="")

        if n_recalled < 2 or n_not_recalled < 2:
            print(" → SKIPPED (too few trials)")
            continue
        print()

        # ── Trial balancing ──────────────────────────────────────────────
        # FieldTrip: [tfr_C1, tfr_C2, n_trl] = TFR_balance_trial(A, B, 100);
        power_recalled = power_all[recalled_mask]
        power_not_recalled = power_all[not_recalled_mask]

        avg_recalled, avg_not_recalled, n_balanced = balance_trials(
            power_recalled, power_not_recalled,
            n_iterations=N_BALANCE_ITER,
            rng=rng
        )
        # avg_recalled shape: (n_chans, n_freqs, n_times)

        # ── Baseline correction in dB ────────────────────────────────────
        # FieldTrip:
        #   cfg.baseline     = [-1, -0.5];   (adapted: [-0.5, -0.2])
        #   cfg.baselinetype = 'db';
        #   tfr_C1_ave{s} = ft_freqbaseline(cfg, tfr_C1_raw{s});
        avg_recalled_db = baseline_correct_db(
            avg_recalled, tfr_times, BASELINE
        )
        avg_not_recalled_db = baseline_correct_db(
            avg_not_recalled, tfr_times, BASELINE
        )

        sub_recalled_db.append(avg_recalled_db)
        sub_not_recalled_db.append(avg_not_recalled_db)
        sub_list_valid.append(sub)

    n_valid = len(sub_list_valid)
    if n_valid < 2:
        print(f"\n  Only {n_valid} valid participant(s) for contrast "
              f"'{cname}'. Skipping statistics.\n")
        continue

    print(f"\n  {n_valid} participants included: {sub_list_valid}")

    # Stack into arrays: (n_subjects, n_channels_stat, n_freqs, n_times)
    X_recalled = np.array(sub_recalled_db)
    X_not_recalled = np.array(sub_not_recalled_db)

    # ══════════════════════════════════════════════════════════════════════
    # CLUSTER-BASED PERMUTATION TEST
    # ══════════════════════════════════════════════════════════════════════
    # FieldTrip:
    #   cfg.method           = 'montecarlo';
    #   cfg.statistic        = 'ft_statfun_depsamplesT';
    #   cfg.correctm         = 'cluster';
    #   cfg.clusteralpha     = 0.05;
    #   cfg.clusterstatistic = 'maxsum';
    #   cfg.tail             = 0;
    #   cfg.clustertail      = 0;
    #   cfg.alpha            = 0.025;
    #   cfg.numrandomization = 1000;
    #   cfg.minnbchan        = 3;
    #   cfg.neighbours       = neighbours;
    #   cfg.latency          = [0, 4];
    #   cfg.frequency        = [2, 40];
    #   cfg.avgoverfreq      = 'no';
    #   [stat] = ft_freqstatistics(cfg, tfr_C1_ave{:}, tfr_C2_ave{:});
    #
    # MNE equivalent:
    #   Compute within-subject difference (recalled − not_recalled),
    #   then run a one-sample cluster permutation test on the difference.
    #   This is mathematically equivalent to FieldTrip's dependent-samples
    #   t-test (depsamplesT).
    #
    #   spatio_temporal_cluster_1samp_test expects:
    #     X shape: (n_subjects, n_times, n_vertices)
    #   where n_vertices = n_channels_stat × n_freqs (the spatial dimension).
    #   Time adjacency is handled automatically (neighboring time points
    #   are always adjacent). Channel × frequency adjacency is provided
    #   via the combined adjacency matrix.

    print(f"\n  Running cluster permutation test...")
    print(f"    N = {n_valid} participants")
    print(f"    {N_PERMUTATIONS} permutations, tail = {TAIL}")
    print(f"    Threshold: t = ±{t_threshold:.3f}")

    # Compute within-subject difference
    X_diff = X_recalled - X_not_recalled
    # Shape: (n_subjects, n_channels_stat, n_freqs, n_times)

    # Subset to adjacency-matched channels only
    X_diff_stat = X_diff[:, tfr_ch_pick, :, :]

    # Crop to statistical time window
    # FieldTrip: cfg.latency = [0, 4];
    X_diff_stat = X_diff_stat[:, :, :, stat_time_idx]
    # Shape: (n_subjects, n_channels_stat, n_freqs, n_times_stat)

    # Crop to statistical frequency range
    # FieldTrip: cfg.frequency = [2, 40];
    X_diff_stat = X_diff_stat[:, :, stat_freq_idx, :]
    # Shape: (n_subjects, n_channels_stat, n_freqs_stat, n_times_stat)

    # Reshape for spatio_temporal_cluster_1samp_test:
    # Input must be (n_observations, n_times, n_vertices)
    # where n_vertices = n_channels_stat × n_freqs
    # We reshape (n_subs, n_ch, n_freq, n_time) → (n_subs, n_time, n_ch × n_freq)
    X_for_test = X_diff_stat.transpose(0, 3, 1, 2).reshape(
        n_valid, n_times_stat, n_channels_stat * n_freqs_stat
    )

    # Build the adjacency for the stat frequencies only
    if n_freqs_stat < n_freqs:
        tfr_adjacency_stat = combine_adjacency(ch_adjacency, n_freqs_stat)
    else:
        tfr_adjacency_stat = tfr_adjacency

    # Run the cluster test
    T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
        X_for_test,
        adjacency=tfr_adjacency_stat,
        n_permutations=N_PERMUTATIONS,
        threshold=t_threshold,
        tail=TAIL,
        seed=SEED,
        n_jobs=-1,
        out_type='mask',
        verbose=True
    )

    # T_obs shape: (n_times_stat, n_channels_stat × n_freqs_stat)
    # Reshape back to (n_times_stat, n_channels_stat, n_freqs_stat)
    T_obs_3d = T_obs.reshape(n_times_stat, n_channels_stat, n_freqs_stat)
    # Transpose to (n_channels_stat, n_freqs_stat, n_times_stat)
    T_obs_3d = T_obs_3d.transpose(1, 2, 0)

    # Reshape cluster masks back to 3D
    cluster_masks_3d = []
    for clust_mask in clusters:
        # clust_mask shape: (n_times_stat, n_ch × n_freq_stat)
        mask_3d = clust_mask.reshape(n_times_stat, n_channels_stat, n_freqs_stat)
        mask_3d = mask_3d.transpose(1, 2, 0)  # → (n_ch, n_freq, n_time)
        cluster_masks_3d.append(mask_3d)

    # ── Report results ────────────────────────────────────────────────────
    sig_clusters = [i for i, p in enumerate(cluster_pv) if p < CLUSTER_P_THRESHOLD]
    n_sig = len(sig_clusters)

    stat_times = tfr_times[stat_time_idx]
    stat_freqs = FREQS[stat_freq_idx]

    txt_lines = []
    txt_lines.append(f"CLUSTER PERMUTATION TEST: {ctitle}")
    txt_lines.append(f"{'─' * 60}")
    txt_lines.append(f"N = {n_valid} participants: {sub_list_valid}")
    txt_lines.append(f"Time window: [{STAT_TMIN}, {STAT_TMAX}] s")
    txt_lines.append(f"Freq range: [{STAT_FMIN}, {STAT_FMAX}] Hz")
    txt_lines.append(f"Baseline: [{BASELINE[0]}, {BASELINE[1]}] s (dB)")
    txt_lines.append(f"Permutations: {N_PERMUTATIONS}, tail: {TAIL}")
    txt_lines.append(f"Cluster threshold: t = ±{t_threshold:.3f} "
                     f"(p < {CLUSTER_ALPHA})")
    txt_lines.append(f"Trial balancing: {N_BALANCE_ITER} iterations")
    txt_lines.append(f"")
    txt_lines.append(f"Total clusters: {len(clusters)}")
    txt_lines.append(f"Significant clusters (p < {CLUSTER_P_THRESHOLD}): "
                     f"{n_sig}")
    txt_lines.append(f"{'─' * 60}")

    for i, (mask, pval) in enumerate(zip(cluster_masks_3d, cluster_pv)):
        # Find the extent of this cluster
        ch_in = np.any(mask, axis=(1, 2))   # channels involved
        freq_in = np.any(mask, axis=(0, 2))  # frequencies involved
        time_in = np.any(mask, axis=(0, 1))  # time points involved

        n_ch = ch_in.sum()
        t_start = stat_times[np.where(time_in)[0][0]]
        t_end = stat_times[np.where(time_in)[0][-1]]
        f_low = stat_freqs[np.where(freq_in)[0][0]]
        f_high = stat_freqs[np.where(freq_in)[0][-1]]

        # Mean t-value within cluster
        mean_t = T_obs_3d[mask].mean()
        sum_t = T_obs_3d[mask].sum()

        sig_marker = " ★ SIGNIFICANT" if pval < CLUSTER_P_THRESHOLD else ""
        direction = "Positive" if mean_t > 0 else "Negative"

        ch_list = [stat_ch_names[j] for j in np.where(ch_in)[0]]

        txt_lines.append(f"")
        txt_lines.append(f"  Cluster {i + 1}: {direction}{sig_marker}")
        txt_lines.append(f"    p-value:     {pval:.4f}")
        txt_lines.append(f"    Time:        {t_start:.3f} – {t_end:.3f} s")
        txt_lines.append(f"    Frequency:   {f_low:.0f} – {f_high:.0f} Hz")
        txt_lines.append(f"    Channels:    {n_ch} ({', '.join(ch_list[:10])}"
                         f"{'...' if n_ch > 10 else ''})")
        txt_lines.append(f"    Mean T:      {mean_t:.3f}")
        txt_lines.append(f"    Sum T:       {sum_t:.3f}")
        txt_lines.append(f"    Voxels:      {mask.sum()}")

    if len(clusters) == 0:
        txt_lines.append("  No clusters found.")

    report_text = "\n".join(txt_lines)
    print(f"\n{report_text}")

    # Save text report
    txt_path = os.path.join(figures_path, f"{cfig}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  ✓ Report saved → {txt_path}")

    # Save statistical results
    stat_path = os.path.join(output_path, f"stat_{cname}.npz")
    np.savez_compressed(
        stat_path,
        T_obs=T_obs_3d,
        cluster_pv=cluster_pv,
        stat_times=stat_times,
        stat_freqs=stat_freqs,
        ch_names=np.array(stat_ch_names),
        n_clusters=len(clusters),
        n_sig_clusters=n_sig,
        # Save cluster masks for significant clusters only
        **{f"cluster_mask_{i}": cluster_masks_3d[i]
           for i in range(len(cluster_masks_3d))}
    )
    print(f"  ✓ Statistics saved → {stat_path}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    #
    # Figure 1: TFR heatmap (t-values averaged over significant channels)
    # FieldTrip:
    #   sig_channel = sum(sum(stat_mem.mask,3),2) > 0;
    #   stat_mem_ave.stat = mean(stat_mem.stat(sig_channel,:,:), 1);
    #   cfg.parameter = 'stat';
    #   cfg.maskparameter = 'maskTF';
    #   cfg.maskstyle = 'outline';
    #   ft_singleplotTFR(cfg, stat_mem_ave);

    print("\n  Creating figures...")

    # ── Build combined significance mask ──────────────────────────────────
    # Combine all significant cluster masks into one mask
    combined_mask = np.zeros((n_channels_stat, n_freqs_stat, n_times_stat),
                             dtype=bool)
    for i in sig_clusters:
        combined_mask |= cluster_masks_3d[i]

    # ── Figure 1: TFR heatmap ────────────────────────────────────────────
    # FieldTrip: sig_channel = sum(sum(stat.mask,3),2) > 0;
    sig_channel_mask = np.any(combined_mask, axis=(1, 2))  # channels in sig clusters
    n_sig_ch = sig_channel_mask.sum()

    if n_sig_ch > 0:
        # Average t-values over significant channels
        # FieldTrip: stat_mem_ave.stat = mean(stat_mem.stat(sig_channel,:,:), 1);
        T_avg = T_obs_3d[sig_channel_mask].mean(axis=0)  # (n_freqs, n_times)

        # Significance mask collapsed over channels (any channel significant)
        # FieldTrip: stat_mem_ave.maskTF = (sum(stat_mem.mask,1)) > 0;
        mask_tf = np.any(combined_mask, axis=0)  # (n_freqs, n_times)
    else:
        # No significant clusters — show all-channel average anyway
        T_avg = T_obs_3d.mean(axis=0)
        mask_tf = np.zeros((n_freqs_stat, n_times_stat), dtype=bool)

    # Plot
    fig_tfr, ax_tfr = plt.subplots(figsize=(12, 6))

    # FieldTrip: cfg.zlim = [-2.5, 2.5];
    vmin, vmax = -2.5, 2.5

    # pcolormesh expects edges, not centers
    time_edges = np.concatenate([
        stat_times - np.diff(stat_times[:2])[0] / 2,
        [stat_times[-1] + np.diff(stat_times[:2])[0] / 2]
    ])
    freq_edges = np.concatenate([
        stat_freqs - np.diff(stat_freqs[:2])[0] / 2,
        [stat_freqs[-1] + np.diff(stat_freqs[:2])[0] / 2]
    ])

    # FieldTrip: colormap(flipud(cbrewer2('RdBu','div')));
    im = ax_tfr.pcolormesh(
        time_edges, freq_edges, T_avg,
        cmap='RdBu_r',  # RdBu_r = reversed RdBu (red=positive, blue=negative)
        vmin=vmin, vmax=vmax,
        shading='flat'
    )

    # FieldTrip: cfg.maskparameter = 'maskTF'; cfg.maskstyle = 'outline';
    # Outline significant clusters with black contour
    if mask_tf.any():
        ax_tfr.contour(
            stat_times, stat_freqs, mask_tf.astype(float),
            levels=[0.5], colors='black', linewidths=2
        )

    # Formatting (matching Wu et al., 2025)
    cb = fig_tfr.colorbar(im, ax=ax_tfr, label='t value')
    ax_tfr.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax_tfr.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax_tfr.set_title(f'TFR: {ctitle}', fontsize=16, fontweight='bold')
    ax_tfr.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_tfr.tick_params(labelsize=12)
    ax_tfr.set_xlim([stat_times[0], stat_times[-1]])
    ax_tfr.set_ylim([stat_freqs[0], stat_freqs[-1]])

    fig_tfr.patch.set_facecolor('white')
    ax_tfr.set_facecolor('white')
    plt.tight_layout()

    # Save
    for ext in ['png', 'svg']:
        fig_tfr.savefig(
            os.path.join(figures_path, f"{cfig}.{ext}"),
            format=ext, dpi=300, bbox_inches='tight', facecolor='white'
        )
    print(f"  ✓ TFR heatmap saved → {cfig}.png/.svg")
    plt.close(fig_tfr)

    # ── Figure 2: Topoplot ───────────────────────────────────────────────
    # FieldTrip:
    #   for chan = 1:size(stat_mem.stat,1)
    #       chan_t = stat_mem.stat(chan,:,:);
    #       topo_t(chan,1) = mean(chan_t(sum(stat_mem.mask,1) > 0));
    #   end
    #   ft_topoplotTFR(cfg, topo_stat_mem);
    #
    # For each channel, compute the mean t-value at time-frequency points
    # that are part of ANY significant cluster.

    # Build topoplot data
    if mask_tf.any():
        # mask_tf is (n_freqs, n_times), same for all channels
        # For each channel, get t-values at significant TF points
        topo_data = np.zeros(n_channels_stat)
        for ch_idx in range(n_channels_stat):
            ch_t = T_obs_3d[ch_idx]  # (n_freqs, n_times)
            topo_data[ch_idx] = ch_t[mask_tf].mean()
    else:
        # No significant clusters — show overall mean
        topo_data = T_obs_3d.mean(axis=(1, 2))

    # Create an Info object for the topoplot
    # Use the epochs Info (has channel positions from the montage)
    topo_info = mne.pick_info(info_epochs,
                              mne.pick_channels(info_epochs.ch_names,
                                                list(stat_ch_names)))

    fig_topo, ax_topo = plt.subplots(figsize=(6, 6))

    # FieldTrip:
    #   cfg.zlim = [-2.5, 2.5];
    #   cfg.colormap = '*RdBu';
    #   cfg.highlight = 'on';
    #   cfg.highlightchannel = {stat_mem.label{sig_channel}};
    im_topo, _ = mne.viz.plot_topomap(
        topo_data, topo_info,
        axes=ax_topo,
        cmap='RdBu_r',
        vlim=(vmin, vmax),
        show=False,
        contours=6,
        # Highlight significant channels
        mask=sig_channel_mask if n_sig_ch > 0 else None,
        mask_params=dict(marker='o', markerfacecolor='black',
                         markeredgecolor='black', markersize=8,
                         zorder=10) if n_sig_ch > 0 else None
    )

    cb_topo = fig_topo.colorbar(im_topo, ax=ax_topo, label='t value',
                                 shrink=0.8)
    ax_topo.set_title(f'{ctitle}\n(mean t at significant TF points)',
                      fontsize=12, fontweight='bold')

    fig_topo.patch.set_facecolor('white')
    plt.tight_layout()

    for ext in ['png', 'svg']:
        fig_topo.savefig(
            os.path.join(figures_path, f"{cfig}_topo.{ext}"),
            format=ext, dpi=300, bbox_inches='tight', facecolor='white'
        )
    print(f"  ✓ Topoplot saved → {cfig}_topo.png/.svg")
    plt.close(fig_topo)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE GROUP-LEVEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print("  SAVING GROUP-LEVEL RESULTS")
print(f"{'═' * 70}")

# Save the group-level averaged TFR arrays for future re-analysis
group_path = os.path.join(output_path, 'group_tfr_summary.npz')
np.savez_compressed(
    group_path,
    times=tfr_times,
    freqs=FREQS,
    ch_names=np.array(stat_ch_names),
    baseline=np.array(BASELINE),
    plist=np.array(list(group_tfr_data.keys())),
    n_participants=n_subs
)
print(f"  ✓ Group summary saved → {group_path}")

print(f"\n{'═' * 70}")
print("  ANALYSIS COMPLETE")
print(f"{'═' * 70}")
print(f"  Figures saved to: {figures_path}")
print(f"  Statistics saved to: {output_path}")
