#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem Analysis Script
=====================

Analysis script for the CDmem (Control Detection + Memory) experiment.
Loads all subject data files and performs various analyses.

Data file naming convention: CDmem_1_{participant}.csv
Located in: CDmem/data/subjects/

TODO:
- add cd accuracy analysis - check simon's code?
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymer4.models import Lmer

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "subjects"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(data_dir=DATA_DIR):
    """
    Load all CDmem subject data files and combine into a single DataFrame.

    File naming pattern: CDmem_1_{participant}.csv
    (e.g. CDmem_1_99.csv for participant 99)

    The function deliberately excludes:
      - CDmem_test.csv style files (intermediate output)
      - CDmem_*_kinematics.csv  (kinematics data â€” separate analysis)
      - CDmem_*_recognition.csv (recognition data â€” separate analysis)

    As new participants complete the study and their CSV is placed in the
    data folder, they are automatically included on the next run.
    No code changes are required.

    Parameters
    ----------
    data_dir : Path
        Directory to search for CSV files.

    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame of all loaded participants, or None if no files found.
    """

    # Glob all CSVs that start with "CDmem_" then apply strict regex so we
    # only match files with exactly two underscore-separated numeric parts
    # where the first part is '1' (e.g. CDmem_1_99.csv).
    pattern = str(data_dir / "CDmem_*.csv")
    all_files = glob.glob(pattern)

    # Regex breakdown:
    #   ^          â€” start of filename
    #   CDmem_1_   â€” literal prefix
    #   \d+        â€” participant ID
    #   \.csv$     â€” ends with .csv
    strict_pattern = re.compile(r"^CDmem_1_\d+\.csv$")
    all_files = [
        f for f in all_files
        if strict_pattern.match(os.path.basename(f))
    ]

    if not all_files:
        print(f"[ERROR] No matching data files found in: {data_dir}")
        return None

    print(f"Found {len(all_files)} data file(s):")
    for f in sorted(all_files):
        print(f"  {os.path.basename(f)}")

    # Load each file individually and tag with its source filename so we can
    # trace back which file any given row came from (useful for debugging).
    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARNING] Could not load {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No files could be loaded.")
        return None

    # Stack all individual participant DataFrames into one master DataFrame.
    # ignore_index=True resets the index so it runs 0 â†’ N-1 across all rows.
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(dfs)} file(s) â†’ {len(data)} total rows combined.\n")
    return data


# ============================================================================
# EXCLUSION CRITERION 1 â€” TIMEOUT RATE
# ============================================================================
#
# Criterion (Haridi et al., 2025):
#   A participant is excluded if 50% or more of their TEST-PHASE trials in
#   EITHER control condition (low or high) are timeout trials.
#
# Rationale:
#   A very high timeout rate in any one condition indicates that the participant
#   stopped engaging with the task. Their data are therefore considered
#   unreliable and removed entirely from the dataset.
#
# Scope:
#   Only test-phase trials are evaluated. Calibration-phase timeouts reflect
#   the staircase adaptation process and are handled separately.
#
# Relevant columns:
#   'phase'             : 'calibration' or 'test'
#   'control_condition' : 'low' or 'high'
#   'is_timeout'        : True / False (or 'True' / 'False' strings)
#   'participant'       : unique participant identifier
# ============================================================================

TIMEOUT_THRESHOLD = 0.50  # 50% Haridi et al., 2025


def exclude_timeout_participants(data):
    """
    Exclude participants whose timeout rate in the test phase reaches or
    exceeds TIMEOUT_THRESHOLD in either the 'low' or 'high' control condition.

    Designed to work with any number of participants: it groups the full
    combined DataFrame by participant and condition, so no code changes
    are needed as the sample grows.

    Parameters
    ----------
    data : pd.DataFrame
        Full combined dataset (all participants, all phases).

    Returns
    -------
    data_clean : pd.DataFrame
        Dataset with flagged participants fully removed (all phases).
    excluded_ids : list
        List of participant IDs excluded by this criterion.
    """

    # --- Step 1: Isolate test-phase trials ---
    # The timeout criterion is applied only to the test phase. During
    # calibration, QUEST adjusts difficulty trial by trial, so occasional
    # timeouts are expected and do not warrant exclusion.
    test_data = data[data["phase"] == "test"].copy()

    # --- Step 2: Normalise the is_timeout column to proper booleans ---
    # PsychoPy can serialise booleans as Python True/False or as the strings
    # 'True'/'False'. We standardise to bool so .sum() counts them correctly.
    test_data["is_timeout"] = test_data["is_timeout"].map(
        lambda x: str(x).strip().lower() == "true"
    )

    # --- Step 3: Compute timeout rate per participant Ã-- condition ---
    # Group by participant and control_condition.
    # 'count' gives the total number of trials; 'sum' counts True (== 1).
    timeout_stats = (
        test_data
        .groupby(["participant", "control_condition"])["is_timeout"]
        .agg(
            total_trials="count",    # total test-phase trials in this condition
            timeout_trials="sum"     # how many of those were timeouts
        )
        .reset_index()
    )

    # timeout_rate = proportion of trials that timed out
    timeout_stats["timeout_rate"] = (
        timeout_stats["timeout_trials"] / timeout_stats["total_trials"]
    )

    # --- Step 4: Flag participants exceeding the threshold in ANY condition ---
    # "either condition" means: timeout_rate >= threshold in 'low' OR in 'high'
    # triggers removal of ALL of that participant's data.
    flagged = timeout_stats[timeout_stats["timeout_rate"] >= TIMEOUT_THRESHOLD]
    excluded_ids = flagged["participant"].unique().tolist()

    # --- Step 5: Print a transparent, detailed report ---
    print("=" * 60)
    print("EXCLUSION CRITERION 1: Timeout Rate")
    print(f"  Threshold : >= {TIMEOUT_THRESHOLD * 100:.0f}% timeouts in either condition")
    print(f"  Reference : Haridi et al., 2025")
    print("=" * 60)

    if len(excluded_ids) == 0:
        print("  â†’ No participants excluded by this criterion.\n")
    else:
        print(f"  â†’ {len(excluded_ids)} participant(s) excluded:\n")
        # Show the specific condition and rate that triggered each exclusion
        for _, row in flagged.iterrows():
            print(
                f"     Participant {row['participant']} | "
                f"Condition: {row['control_condition']} | "
                f"Timeout rate: {row['timeout_rate']:.1%} "
                f"({int(row['timeout_trials'])}/{int(row['total_trials'])} trials)"
            )
        print()

    # --- Step 6: Remove flagged participants from the FULL dataset ---
    # We drop all rows for those participants â€” including calibration trials â€”
    # because an excluded participant should not contribute any data.
    data_clean = data[~data["participant"].isin(excluded_ids)].copy()

    return data_clean, excluded_ids


# ============================================================================
# EXCLUSION CRITERION 2: ACCURACY OUTLIERS
# ============================================================================
#
# Criterion:
#   Expected mean accuracy is 55% for the LOW control condition and 85% for
#   the HIGH control condition (reflecting the QUEST staircase targets).
#   A participant is excluded if their accuracy in EITHER condition is more
#   than 2.5 standard deviations away from the GROUP mean.
#
# Rationale:
#   Although the QUEST staircase converges each participant toward their
#   personal threshold, group-level outliers indicate that calibration failed
#   (e.g., threshold not reached) or that the participant misunderstood the
#   task. A relative SD criterion (rather than fixed bounds) keeps the cutoff
#   adaptive to the actual spread in the sample.
#
# Important note on sample size:
#   The group mean and SD are estimated from all currently-loaded participants
#   (after Criterion 1 has already been applied). With small samples the
#   estimates are noisy; the criterion becomes more precise as N grows.
#   This is expected behaviour and consistent with typical psychophysics
#   exclusion pipelines.
#
# Relevant columns:
#   'phase'              : we look only at test-phase trials
#   'control_condition'  : 'low' or 'high'
#   'detection_accuracy' : 0 = incorrect, 1 = correct
#   'participant'        : unique participant identifier
# ============================================================================

# Expected (target) accuracy per condition â€” defined by QUEST staircase targets
EXPECTED_ACCURACY = {
    "low":  0.55,   # 55% low control condition
    "high": 0.85,   # 85% high control condition
}

# Outlier cutoff: participants more than this many SDs from the group mean
# are excluded. Applied separately for each condition.
ACCURACY_SD_THRESHOLD = 2.5


def exclude_accuracy_outliers(data):
    """
    Exclude participants whose mean test-phase accuracy in either control
    condition deviates more than ACCURACY_SD_THRESHOLD SDs from the group mean.

    The criterion is computed separately for the 'low' and 'high' conditions.
    Exclusion in either condition removes the participant's data entirely.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset after previous exclusion steps (all phases, all participants
        remaining after Criterion 1).

    Returns
    -------
    data_clean : pd.DataFrame
        Dataset with accuracy-outlier participants fully removed.
    excluded_ids : list
        Participant IDs excluded by this criterion.
    report_df : pd.DataFrame
        Full accuracy table (all participants Ã-- conditions) with z-scores â€”
        useful for reporting in the methods section or supplementary materials.
    """

    # --- Step 1: Isolate test-phase trials ---
    # Accuracy is only meaningful for the test phase. During calibration,
    # QUEST intentionally varies difficulty, so accuracy is expected to
    # fluctuate and should not be used for exclusion.
    test_data = data[data["phase"] == "test"].copy()

    # --- Step 2: Compute per-participant mean accuracy per condition ---
    # detection_accuracy is coded as 1 (correct) or 0 (incorrect).
    # .mean() on a 0/1 column = proportion correct = accuracy.
    acc_per_px = (
        test_data
        .groupby(["participant", "control_condition"])["detection_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"detection_accuracy": "mean_accuracy"})
    )

    # --- Step 3: Compute group-level mean and SD per condition ---
    # These statistics describe the distribution of accuracies across all
    # participants currently in the dataset. pandas .std() uses ddof=1 by
    # default, giving the unbiased sample standard deviation.
    group_stats = (
        acc_per_px
        .groupby("control_condition")["mean_accuracy"]
        .agg(group_mean="mean", group_sd="std")
        .reset_index()
    )

    # Merge group stats back onto the per-participant table so each row
    # carries its condition's group_mean and group_sd alongside mean_accuracy.
    acc_per_px = acc_per_px.merge(group_stats, on="control_condition")

    # --- Step 4: Compute z-score for each participant Ã-- condition cell ---
    # z = (participant_accuracy - group_mean) / group_SD
    #   z > +2.5  â†’ unusually high accuracy (e.g., ceiling, possible cheating)
    #   z < -2.5  â†’ unusually low accuracy (e.g., calibration failure)
    # Either direction triggers exclusion.
    acc_per_px["z_score"] = (
        (acc_per_px["mean_accuracy"] - acc_per_px["group_mean"])
        / acc_per_px["group_sd"]
    )

    # --- Step 5: Flag participants with |z| > threshold in ANY condition ---
    # Being an outlier in one condition is sufficient to exclude the participant.
    flagged = acc_per_px[acc_per_px["z_score"].abs() > ACCURACY_SD_THRESHOLD]
    excluded_ids = flagged["participant"].unique().tolist()

    # --- Step 6: Print a detailed, human-readable report ---
    print("=" * 60)
    print("EXCLUSION CRITERION 2: Accuracy Outliers")
    print(f"  Threshold : |z| > {ACCURACY_SD_THRESHOLD} SD from group mean")
    print(f"  Expected accuracies: "
          f"low = {EXPECTED_ACCURACY['low']*100:.0f}%, "
          f"high = {EXPECTED_ACCURACY['high']*100:.0f}%")
    print("=" * 60)

    # Print the full table so the researcher can inspect every participant's
    # accuracy and z-score. Excluded participants are marked with ***.
    print("\n  Per-participant accuracy by condition (with group z-scores):\n")
    print(f"  {'Participant':<15} {'Condition':<10} {'Accuracy':>10} "
          f"{'Group Mean':>12} {'Group SD':>10} {'z-score':>9}")
    print("  " + "-" * 70)
    for _, row in acc_per_px.sort_values(["control_condition", "participant"]).iterrows():
        flag = " *** EXCLUDED" if row["participant"] in excluded_ids else ""
        print(
            f"  {str(row['participant']):<15} {row['control_condition']:<10} "
            f"{row['mean_accuracy']:>10.3f} {row['group_mean']:>12.3f} "
            f"{row['group_sd']:>10.3f} {row['z_score']:>9.2f}{flag}"
        )

    print()
    if len(excluded_ids) == 0:
        print("  â†’ No participants excluded by this criterion.\n")
    else:
        print(f"  â†’ {len(excluded_ids)} participant(s) excluded: {excluded_ids}\n")

    # --- Step 7: Remove flagged participants from the full dataset ---
    # All rows (calibration and test) for the flagged participants are removed.
    assert data is not None
    data_clean = data[~data["participant"].isin(excluded_ids)].copy()

    return data_clean, excluded_ids, acc_per_px


# ============================================================================
# RECOGNITION DATA LOADING
# ============================================================================

def load_recognition_data(data_dir=DATA_DIR):
    """
    Load all CDmem recognition CSV files and combine into a single DataFrame.

    File naming pattern: CDmem_{participant}_{session}_{block}_recognition.csv
    (e.g. CDmem_1_1_1_recognition.csv, CDmem_1_2_1_recognition.csv)

    Recognition data is kept in a SEPARATE DataFrame (recog_data) from the
    main task data because the two have different trial structures, columns,
    and exclusion criteria.

    As new participants complete the study and their recognition CSV is placed
    in the data folder, they are automatically included on the next run.

    Parameters
    ----------
    data_dir : Path
        Directory to search for CSV files (same folder as main data).

    Returns
    -------
    pd.DataFrame or None
        Combined recognition DataFrame across all participants, or None.
    """

    # Glob for files ending in _recognition.csv.
    # The * before _recognition captures the full CDmem_px_session_block prefix.
    pattern = str(data_dir / "CDmem_*_recognition.csv")
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"[ERROR] No recognition files found in: {data_dir}")
        return None

    print(f"Found {len(all_files)} recognition file(s):")
    for f in sorted(all_files):
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f)
            # Tag with source filename for traceability
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARNING] Could not load {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No recognition files could be loaded.")
        return None

    recog_data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(dfs)} recognition file(s) â†’ {len(recog_data)} total rows.\n")
    return recog_data


# ============================================================================
# RECOGNITION EXCLUSION LONG RT TRIALS
# ============================================================================
#
# Criterion (Haridi et al., 2025):
#   Recognition trials with a response time (mem_rt) longer than 20 seconds
#   are excluded at the TRIAL level (not the participant level).
#
# Rationale:
#   Extremely long RTs in a recognition task typically reflect mind-wandering,
#   distraction, or task disengagement on that specific trial rather than a
#   systematic problem with the participant. Removing these trials prevents
#   them from distorting condition means without unnecessarily discarding the
#   participant's remaining valid data.
#
# Relevant columns:
#   'mem_rt'      : response time in seconds for each recognition trial
#   'participant' : unique participant identifier (for the report)
# ============================================================================

RT_CUTOFF_SECONDS = 20.0   # Haridi et al., 2025


def exclude_long_rt_trials(recog_data):
    """
    Remove individual recognition trials where mem_rt exceeds RT_CUTOFF_SECONDS.

    This is a trial-level exclusion (not participant-level): only the outlier
    trials are dropped; the participant's remaining trials are kept.

    Designed to work with any number of participants â€” it operates on the
    full combined recognition DataFrame, so no code changes are needed as
    the sample grows.

    Parameters
    ----------
    recog_data : pd.DataFrame
        Full recognition dataset (all participants, all trials).

    Returns
    -------
    recog_clean : pd.DataFrame
        Recognition dataset with long-RT trials removed.
    n_excluded_trials : int
        Number of individual trials removed.
    exclusion_details : pd.DataFrame
        Table of each removed trial (participant, trial number, RT) for
        reporting and inspection.
    """

    # --- Step 1: Convert mem_rt to numeric ---
    # Read from CSV, the column should already be float, but we coerce
    # defensively in case any non-numeric values crept in (e.g., empty cells).
    recog_data = recog_data.copy()
    recog_data["mem_rt"] = pd.to_numeric(recog_data["mem_rt"], errors="coerce")

    # --- Step 2: Identify long-RT trials ---
    # A trial is flagged if mem_rt > RT_CUTOFF_SECONDS OR if mem_rt is NaN
    # (missing RT is also not a valid response and should be dropped).
    long_rt_mask = recog_data["mem_rt"] > RT_CUTOFF_SECONDS
    missing_rt_mask = recog_data["mem_rt"].isna()
    exclude_mask = long_rt_mask | missing_rt_mask

    # --- Step 3: Store details of excluded trials for the report ---
    exclusion_details = recog_data[exclude_mask][["participant", "overall_trial_num", "mem_rt"]].copy()
    n_excluded_trials = len(exclusion_details)

    # --- Step 4: Print a detailed report ---
    print("=" * 60)
    print("RECOGNITION EXCLUSION: Long RT Trials")
    print(f"  RT cutoff : > {RT_CUTOFF_SECONDS:.0f} s")
    print(f"  Reference : Haridi et al., 2025")
    print("=" * 60)

    if n_excluded_trials == 0:
        print("  â†’ No trials excluded by this criterion.\n")
    else:
        print(f"  â†’ {n_excluded_trials} trial(s) excluded:\n")
        # Show each removed trial so the researcher can verify
        for _, row in exclusion_details.iterrows():
            print(
                f"     Participant {row['participant']} | "
                f"Trial {int(row['overall_trial_num'])} | "
                f"RT = {row['mem_rt']:.2f} s"
            )
        print()

    # --- Step 5: Remove the flagged trials ---
    # We keep ~mask so that only valid (short-RT) trials remain.
    recog_clean = recog_data[~exclude_mask].copy()

    return recog_clean, n_excluded_trials, exclusion_details


# ============================================================================
# CROSS-DATASET PARTICIPANT SYNCHRONISATION
# ============================================================================
#
# After all exclusions have been applied independently to the main task data
# and the recognition data, the two datasets may contain different participant
# sets. For example:
#   - A participant excluded from main data (timeout / accuracy) should also
#     be dropped from recog_data.
#   - A participant missing from recog_data entirely (file never collected, or
#     corrupted) should also be removed from main data.
#
# This function computes the INTERSECTION of participant IDs in both datasets
# and retains only those participants in both DataFrames.
# ============================================================================

def sync_participant_ids(data, recog_data):
    """
    Ensure both DataFrames contain exactly the same set of participant IDs.

    Any participant present in one DataFrame but not the other is removed
    from the DataFrame that does contain them, and the reason is reported.

    Applied as a final cross-dataset consistency step, AFTER all independent
    exclusion criteria have been run on each dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Main task data (already filtered by timeout / accuracy criteria).
    recog_data : pd.DataFrame
        Recognition data (already filtered by RT criterion).

    Returns
    -------
    data_synced : pd.DataFrame
    recog_synced : pd.DataFrame
    """

    # --- Step 1: Get participant sets from each dataset ---
    main_pxs  = set(data['participant'].unique())
    recog_pxs = set(recog_data['participant'].unique())

    # --- Step 2: Compute the intersection ---
    common_pxs = main_pxs & recog_pxs

    # --- Step 3: Identify asymmetric participants ---
    # In main but not in recognition -> drop from main data
    only_in_main  = main_pxs  - recog_pxs
    # In recognition but not in main -> drop from recog_data
    only_in_recog = recog_pxs - main_pxs

    # --- Step 4: Print synchronisation report ---
    print("=" * 60)
    print("CROSS-DATASET PARTICIPANT SYNCHRONISATION")
    print("=" * 60)

    if not only_in_main and not only_in_recog:
        print("  Both datasets already contain the same participant IDs.")
        print(f"  Participants in both: {sorted(common_pxs)}\n")
    else:
        if only_in_main:
            print(f"  In main data but NOT in recognition (removed from main data):")
            print(f"    {sorted(only_in_main)}")
        if only_in_recog:
            print(f"  In recognition data but NOT in main data (removed from recog_data):")
            print(f"    {sorted(only_in_recog)}")
        print(f"  Participants retained in both: {sorted(common_pxs)}\n")

    # --- Step 5: Filter both DataFrames to the common participant set ---
    data_synced  = data[data['participant'].isin(common_pxs)].copy()
    recog_synced = recog_data[recog_data['participant'].isin(common_pxs)].copy()

    return data_synced, recog_synced


def calc_dprime(hr, far, clip_val=0.01):
    """
    Calculate d-prime from hit rate (hr) and false alarm rate (far).
    Includes clipping to avoid infinity when hr=1.0 or far=0.0.
    """
    hr_clipped = np.clip(hr, clip_val, 1.0 - clip_val)
    far_clipped = np.clip(far, clip_val, 1.0 - clip_val)
    return norm.ppf(hr_clipped) - norm.ppf(far_clipped)

def analyze_recognition(main_data, recog_data):
    """
    Calculate recognition performance (HR, FA, d-prime) per participant
    and per control condition.
    
    Returns
    -------
    pd.DataFrame
        Participant-level summary.
    """
    # 1. Calculate overall False Alarm rate per participant
    # Foils have mem_ground_truth == 'unseen'
    # Responses are in 'mem_response' ('yes'/'no')
    recog_data = recog_data.copy()
    recog_data['said_old'] = recog_data['mem_response'].str.lower() == 'yes'
    
    foils = recog_data[recog_data['mem_ground_truth'] == 'unseen']
    if len(foils) == 0:
        print("[WARNING] No foil ('unseen') trials found in recognition data.")
        return None
        
    fa_rates = foils.groupby('participant')['said_old'].mean().reset_index()
    fa_rates.rename(columns={'said_old': 'FA_rate'}, inplace=True)
    
    # 2. Extract Hit rates per condition
    # Targets have mem_ground_truth == 'seen'.
    # We must merge with main_data to know which condition ('low'/'high') the target came from.
    # The linking variable is the image name. 
    # Recognition targets are in `mem_filename`. Main data has `img_A_name` and `img_B_name`.
    # Let's create a lookup table mapping each participant + image name to its control_condition.
    
    # Isolate test phase of main data
    if 'phase' in main_data.columns:
        test_data = main_data[main_data['phase'] == 'test'].copy()
    else:
        test_data = main_data.copy()
        
    if 'control_condition' not in test_data.columns:
        print("[WARNING] control_condition not found in main data, cannot split recognition by condition.")
        return None
        
    # We need to reshape the main data to have one row per image (img_A and img_B)
    img_a_df = test_data[['participant', 'control_condition', 'img_A_name']].copy()
    img_a_df.rename(columns={'img_A_name': 'mem_filename'}, inplace=True)
    
    img_b_df = test_data[['participant', 'control_condition', 'img_B_name']].copy()
    img_b_df.rename(columns={'img_B_name': 'mem_filename'}, inplace=True)
    
    img_lookup = pd.concat([img_a_df, img_b_df], ignore_index=True)
    img_lookup.dropna(subset=['mem_filename'], inplace=True)
    # Deduplicate just in case 
    img_lookup.drop_duplicates(subset=['participant', 'mem_filename'], inplace=True)
    
    # 3. Merge targets with their condition
    targets = recog_data[recog_data['mem_ground_truth'] == 'seen'].copy()
    targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')
    
    # Verify merge success
    unmatched = targets['control_condition'].isna().sum()
    if unmatched > 0:
        print(f"[WARNING] {unmatched} recognition targets could not be matched to a control condition.")
        
    # Calculate hit rate per participant x condition
    hit_rates = targets.groupby(['participant', 'control_condition'])['said_old'].mean().reset_index()
    hit_rates.rename(columns={'said_old': 'Hit_rate'}, inplace=True)
    
    # 4. Combine HR and FA to calculate d-prime
    results = hit_rates.merge(fa_rates, on='participant', how='left')
    results['d_prime'] = results.apply(lambda row: calc_dprime(row['Hit_rate'], row['FA_rate']), axis=1)
    
    return results, targets

# ==============================================================================
# RECOGNITION MEMORY ANALYSES
# ==============================================================================

# ==============================================================================
# PRIMARY ANALYSES (High vs. Low Control)
# ==============================================================================

# ------------------------------------------------------------------------------
# ANALYSIS 1: Paired t-test on d-prime
# ------------------------------------------------------------------------------

def run_analysis_1_dprime_ttest(mem_results):
    """
    Perform a paired t-test comparing d-prime between high and low control conditions.
    """
    pivoted = mem_results.pivot(index='participant', columns='control_condition', values='d_prime')
    
    # Check if we have both conditions for at least some participants
    if 'high' not in pivoted.columns or 'low' not in pivoted.columns:
        print("[WARNING] Missing condition data for d-prime t-test.")
        return None
        
    # Drop participants who don't have both conditions
    pivoted = pivoted.dropna(subset=['high', 'low'])
    
    t_stat, p_val = stats.ttest_rel(pivoted['high'], pivoted['low'])
    
    results = {
        'analysis': 'Analysis 1: d-prime paired t-test',
        't_stat': t_stat,
        'p_val': p_val,
        'mean_high': pivoted['high'].mean(),
        'mean_low': pivoted['low'].mean(),
        'sd_high': pivoted['high'].std(),
        'sd_low': pivoted['low'].std(),
        'n': len(pivoted)
    }
    return results

# ------------------------------------------------------------------------------
# ANALYSIS 2: Paired t-test on hit rates
# ------------------------------------------------------------------------------

def run_analysis_2_hitrate_ttest(mem_results):
    """
    Perform a paired t-test comparing hit rates between high and low control conditions.
    """
    pivoted = mem_results.pivot(index='participant', columns='control_condition', values='Hit_rate')
    
    if 'high' not in pivoted.columns or 'low' not in pivoted.columns:
        print("[WARNING] Missing condition data for hit rate t-test.")
        return None
        
    pivoted = pivoted.dropna(subset=['high', 'low'])
    
    t_stat, p_val = stats.ttest_rel(pivoted['high'], pivoted['low'])
    
    results = {
        'analysis': 'Analysis 2: Hit rate paired t-test',
        't_stat': t_stat,
        'p_val': p_val,
        'mean_high': pivoted['high'].mean(),
        'mean_low': pivoted['low'].mean(),
        'sd_high': pivoted['high'].std(),
        'sd_low': pivoted['low'].std(),
        'n': len(pivoted)
    }
    return results

# ------------------------------------------------------------------------------
# ANALYSIS 3: GLMM on target trials only
# ------------------------------------------------------------------------------

def run_analysis_3_glmm(targets):
    """
    Perform a trial-level GLMM (Binomial, logit link) on target (old) trials only.
    Model: said_old ~ control_condition + (1 | participant)
    
    Uses pymer4 (wraps R's lme4::glmer) for proper GLMM estimation.
    """
    if targets is None or len(targets) == 0:
        print("[WARNING] No target data for GLMM.")
        return None

    df = targets.copy()
    df['said_old_int'] = df['said_old'].astype(int)
    
    # Contrast-code: +0.5 = high, -0.5 = low
    df['control_numeric'] = df['control_condition'].map({'high': 0.5, 'low': -0.5})
    df = df.dropna(subset=['said_old_int', 'control_numeric', 'participant'])

    if len(df) == 0:
        print("[WARNING] No valid trials after preprocessing.")
        return None

    print("\nAnalysis 3: GLMM (Binomial, logit link) on target trials...")
    print(f"  Trials: {len(df)} | Participants: {df['participant'].nunique()}")

    try:
        model = Lmer(
            "said_old_int ~ control_numeric + (1 | participant)",
            data=df,
            family="binomial"
        )
        result = model.fit()
        print(result)
        return model

    except Exception as e:
        print(f"  [ERROR] GLMM failed: {e}")
        return None

# ------------------------------------------------------------------------------
# ANALYSIS 4: GEE with item_type x control interaction
# ------------------------------------------------------------------------------
# [PLACEHOLDER] To be implemented (requires statsmodels)


# ==============================================================================
# SUPPLEMENTARY ANALYSES (Including Uncontrolled Items)
# ==============================================================================

# ------------------------------------------------------------------------------
# SUPPLEMENTARY 1: One-way repeated measures ANOVA on d-prime
# ------------------------------------------------------------------------------
# (Conditions: high control, low control, uncontrolled)
# [PLACEHOLDER] To be implemented

# ------------------------------------------------------------------------------
# SUPPLEMENTARY 2: One-way repeated measures ANOVA on hit rates
# ------------------------------------------------------------------------------
# (Conditions: high control, low control, uncontrolled)
# [PLACEHOLDER] To be implemented

# ------------------------------------------------------------------------------
# SUPPLEMENTARY 3: GEE on all old items (3-level control_type predictor)
# ------------------------------------------------------------------------------
# [PLACEHOLDER] To be implemented

# ------------------------------------------------------------------------------
# SUPPLEMENTARY 4: GEE with item_type x control interaction
# ------------------------------------------------------------------------------
# (Three-level control, no foil dummy assignment needed)
# [PLACEHOLDER] To be implemented


def print_stat_results(res):
    """Helper to print formatted statistical results."""
    if res is None: return
    print(f"\n{res['analysis']}:")
    print(f"  t({res['n']-1}) = {res['t_stat']:.3f}, p = {res['p_val']:.4f}")
    print(f"  Mean (SD) High: {res['mean_high']:.3f} ({res['sd_high']:.3f})")
    print(f"  Mean (SD) Low:  {res['mean_low']:.3f} ({res['sd_low']:.3f})")

def run_recognition_stats(mem_results, targets):
    """Run all recognition statistical analyses."""
    print("\n" + "=" * 60)
    print("RECOGNITION STATISTICAL ANALYSES")
    print("=" * 60)
    
    # Primary Analysis 1
    res1 = run_analysis_1_dprime_ttest(mem_results)
    print_stat_results(res1)
    
    # Primary Analysis 2
    res2 = run_analysis_2_hitrate_ttest(mem_results)
    print_stat_results(res2)
    
    # Primary Analysis 3 - GLMM
    run_analysis_3_glmm(targets)
    
    print("\n" + "=" * 60)


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def print_descriptives(data):
    """
    Calculate and print demographic descriptives for the final sample.
    (N, gender counts/percentages, age mean, SD, min, max)
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset containing 'participant', 'age', and 'gender'.
    """
    # Isolate one row per participant
    demographics = data.drop_duplicates(subset=["participant"])[["participant", "age", "gender"]].copy()
    
    n_total = len(demographics)
    
    # -- Gender Statistics --
    # Normalise gender strings (e.g. 'f' -> 'female' if needed, here we just strip/lower)
    demographics["gender"] = demographics["gender"].astype(str).str.strip().str.lower()
    gender_counts = demographics["gender"].value_counts()
    gender_pcts = demographics["gender"].value_counts(normalize=True) * 100
    
    # -- Age Statistics --
    demographics["age"] = pd.to_numeric(demographics["age"], errors="coerce")
    age_mean = demographics["age"].mean()
    age_sd = demographics["age"].std()
    age_min = demographics["age"].min()
    age_max = demographics["age"].max()
    
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS (Final Sample)")
    print("=" * 60)
    print(f"  Total N = {n_total}\n")
    
    print("  Gender:")
    for g in gender_counts.index:
        count = gender_counts[g]
        pct = gender_pcts[g]
        # Just neatening up 'f' and 'm' if they are used
        label = "Female" if g == "f" else "Male" if g == "m" else g.capitalize()
        print(f"    {label:<10} : {count:>3} ({pct:>4.1f}%)")
        
    print("\n  Age:")
    if pd.notna(age_mean):
        print(f"    Mean (SD)  : {age_mean:.1f} ({age_sd:.1f})")
        print(f"    Range      : {age_min:.0f} - {age_max:.0f}")
    else:
        print("    No valid age data found.")
    print("=" * 60 + "\n")


# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================

def check_image_uniqueness(data):
    """
    Confirm that all images shown during the test phase were unique
    within and across the img_A_name and img_B_name columns.
    """
    if 'phase' not in data.columns or 'img_A_name' not in data.columns or 'img_B_name' not in data.columns:
        print("[WARNING] Missing columns for image uniqueness check (img_A_name or img_B_name).")
        return

    test_data = data[data['phase'] == 'test'].copy()
    if len(test_data) == 0:
        print("[WARNING] No test phase trials found for image uniqueness check.")
        return

    # Collect all image names from both columns across all selected participants
    img_a = test_data['img_A_name'].dropna().tolist()
    img_b = test_data['img_B_name'].dropna().tolist()
    all_images = img_a + img_b

    n_total = len(all_images)
    n_unique = len(set(all_images))

    print("\n" + "=" * 60)
    print("IMAGE UNIQUENESS CHECK (Test Phase)")
    print("=" * 60)
    print(f"  Total image presentations : {n_total}")
    print(f"  Unique image names         : {n_unique}")

    if n_total == n_unique:
        print("  â†’ SUCCESS: All images shown during the test phase were different.")
    else:
        n_dupes = n_total - n_unique
        print(f"  â†’ FAILURE: Found {n_dupes} duplicate image presentation(s)!")
        
        # Identify the duplicates
        from collections import Counter
        counts = Counter(all_images)
        duplicates = [img for img, count in counts.items() if count > 1]
        for d in duplicates:
            print(f"     Duplicate: {d} (shown {counts[d]} times)")
    print("=" * 60)


# ============================================================================
# EXTENDED SANITY CHECKS & PLOTS
# ============================================================================

def sanity_check(df):
    """Compute sanity check statistics for performance and agency ratings."""
    
    results = {}
    
    # Overall accuracy by control_condition
    if 'control_condition' in df.columns:
        accuracy_by_condition = df.groupby('control_condition')['detection_accuracy'].agg(['mean', 'std', 'count'])
        results['accuracy_by_condition'] = accuracy_by_condition
    else:
        results['accuracy'] = df['detection_accuracy'].agg(['mean', 'std', 'count'])
    
    # Agency ratings by control_condition
    if 'agency_rating' in df.columns and df['agency_rating'].notna().any():
        if 'control_condition' in df.columns:
            agency_by_condition = df.groupby('control_condition')['agency_rating'].agg(['mean', 'std', 'count'])
            results['agency_by_condition'] = agency_by_condition
            
            # Agency by accuracy (correct vs incorrect)
            agency_by_accuracy = df.groupby(['control_condition', 'detection_accuracy'])['agency_rating'].agg(['mean', 'std', 'count'])
            results['agency_by_accuracy'] = agency_by_accuracy
        else:
            results['agency'] = df['agency_rating'].agg(['mean', 'std', 'count'])
            agency_by_accuracy = df.groupby('detection_accuracy')['agency_rating'].agg(['mean', 'std', 'count'])
            results['agency_by_accuracy'] = agency_by_accuracy
    
    # Psychometric function: accuracy as function of control level
    df_copy = df.copy()
    df_copy['prop_used'] = pd.to_numeric(df_copy['prop_used'], errors='coerce')
    # Bins of control levels
    if df_copy['prop_used'].nunique() > 10:
        df_copy['prop_bin'] = pd.cut(df_copy['prop_used'], bins=10, duplicates='drop')
    else:
        df_copy['prop_bin'] = df_copy['prop_used']
    
    if 'control_condition' in df_copy.columns:
        psychometric = df_copy.groupby(['prop_bin', 'control_condition'], observed=False)['detection_accuracy'].agg(['mean', 'std', 'count'])
    else:
        psychometric = df_copy.groupby('prop_bin', observed=False)['detection_accuracy'].agg(['mean', 'std', 'count'])
    results['psychometric'] = psychometric
    
    return results

def plot_sanity_check(df, output_dir):
    """Create plots for sanity check statistics."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    df = df.copy()
    
    # Ensure variables are numeric
    df['detection_accuracy'] = pd.to_numeric(df['detection_accuracy'], errors='coerce')
    df['prop_used'] = pd.to_numeric(df['prop_used'], errors='coerce')
    if 'agency_rating' in df.columns:
        df['agency_rating'] = pd.to_numeric(df['agency_rating'], errors='coerce')
    if 'rt_choice' in df.columns:
        df['rt_choice'] = pd.to_numeric(df['rt_choice'], errors='coerce')
    
    has_condition = 'control_condition' in df.columns
    
    # 1. Psychometric function
    ax = axes[0, 0]
    if has_condition:
        for condition in df['control_condition'].dropna().unique():
            subset = df[df['control_condition'] == condition].copy()
            if len(subset) == 0: continue
            
            # Match MTI binning precisely: Use identical bins calculation where min != max
            if subset['prop_used'].min() != subset['prop_used'].max():
                bins = np.linspace(subset['prop_used'].min(), subset['prop_used'].max(), 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                subset['prop_bin'] = pd.cut(subset['prop_used'], bins=bins, labels=bin_centers)
                psychometric = subset.groupby('prop_bin', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                psychometric['prop_bin'] = psychometric['prop_bin'].astype(float)
                ax.errorbar(psychometric['prop_bin'], psychometric['mean'], 
                           yerr=psychometric['sem'], label=f'{condition}', marker='o', capsize=3)
            else:
                # If all threshold values are exactly identical (which crashes pd.cut), calculate directly
                psychometric = subset.groupby('prop_used', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                ax.errorbar(psychometric['prop_used'], psychometric['mean'], 
                           yerr=psychometric['sem'], label=f'{condition}', marker='o', capsize=3)
        ax.legend()
    else:
        if df['prop_used'].min() != df['prop_used'].max():
            bins = np.linspace(df['prop_used'].min(), df['prop_used'].max(), 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            df['prop_bin'] = pd.cut(df['prop_used'], bins=bins, labels=bin_centers)
            psychometric = df.groupby('prop_bin', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
            psychometric['prop_bin'] = psychometric['prop_bin'].astype(float)
            ax.errorbar(psychometric['prop_bin'], psychometric['mean'], 
                       yerr=psychometric['sem'], marker='o', capsize=3)
        else:
            psychometric = df.groupby('prop_used', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
            ax.errorbar(psychometric['prop_used'], psychometric['mean'], 
                       yerr=psychometric['sem'], marker='o', capsize=3)
            
    ax.set_xlabel('Control Level (prop self-motion)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Psychometric Function')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Accuracy
    ax = axes[0, 1]
    if has_condition:
        accuracy_data = df.groupby('control_condition')['detection_accuracy'].agg(['mean', 'sem']).reset_index()
        colors = ['#1f77b4', '#ff7f0e'] if len(accuracy_data) <= 2 else None
        bars = ax.bar(accuracy_data['control_condition'].astype(str), accuracy_data['mean'], 
                      yerr=accuracy_data['sem'], capsize=5, color=colors)
        ax.set_xlabel('Control Condition')
    else:
        mean_acc = df['detection_accuracy'].mean()
        sem_acc = df['detection_accuracy'].sem()
        ax.bar(['Overall'], [mean_acc], yerr=[sem_acc], capsize=5)
    
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim([0, 1])
    
    # 3. Agency ratings by accuracy
    ax = axes[1, 0]
    if 'agency_rating' in df.columns and df['agency_rating'].notna().any():
        if has_condition:
            sns.barplot(data=df, x='control_condition', y='agency_rating', hue='detection_accuracy', 
                       ax=ax, errorbar='se')
            ax.set_xlabel('Control Condition')
        else:
            df['Overall'] = 'Overall'
            sns.barplot(data=df, x='Overall', y='agency_rating', hue='detection_accuracy', 
                       ax=ax, errorbar='se')
            ax.set_xlabel('')
            
        ax.set_ylabel('Agency Rating (1-7)')
        ax.set_title('Agency Ratings by Accuracy')
        ax.legend(title='Correct (1) / Incorrect (0)')
    else:
        ax.text(0.5, 0.5, 'No agency rating data available', 
               transform=ax.transAxes, ha='center')
    
    # 4. RT distribution
    ax = axes[1, 1]
    if 'rt_choice' in df.columns and df['rt_choice'].notna().any():
        df_rt = df[df['rt_choice'] > 0]
        if has_condition and len(df_rt) > 0:
            for condition in df_rt['control_condition'].dropna().unique():
                subset = df_rt[df_rt['control_condition'] == condition]['rt_choice']
                if len(subset) > 0:
                    ax.hist(subset, bins=30, alpha=0.5, label=str(condition), density=True)
            ax.legend()
        elif len(df_rt) > 0:
            ax.hist(df_rt['rt_choice'], bins=30, alpha=0.5, density=True)
        ax.set_xlabel('Reaction Time (s)')
        ax.set_ylabel('Density')
        ax.set_title('RT Distribution')
    else:
        ax.text(0.5, 0.5, 'No RT data available', 
               transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'sanity_check.png', dpi=150)
    plt.close()
    
    print("Saved plot: " + str(out_dir / 'sanity_check.png'))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    # Reads every CDmem_{px}_{session}_{block}.csv from data/subjects/ and
    # concatenates them into a single DataFrame.
    # Adding a new participant's file to the folder is sufficient to include
    # them on the next run â€” no code changes needed.
    data = load_all_data()

    if data is None:
        print("No data loaded. Exiting.")
        sys.exit(1)
    
    assert data is not None

    # ------------------------------------------------------------------
    # 2. Sanity check: print column names and overall shape
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(f"Total columns: {len(data.columns)}\n")
    for i, col in enumerate(data.columns, start=1):
        print(f"  {i:>3}. {col}")

    # ------------------------------------------------------------------
    # 2b. Image Uniqueness Check
    # ------------------------------------------------------------------
    # Verifies that every image shown in 'test' phase (img_A_name and img_B_name)
    # is different within and across trials.
    check_image_uniqueness(data)

    print("\n" + "=" * 60)
    print("DATAFRAME SHAPE  (raw, before exclusions)")
    print("=" * 60)
    print(f"  Rows         : {data.shape[0]}")
    print(f"  Columns      : {data.shape[1]}")
    print(f"  Participants : {data['participant'].nunique()}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 3. Exclusion criterion 1 â€” Timeout rate
    # ------------------------------------------------------------------
    # Participants with >= 50% timeouts in either condition are removed.
    # This is applied first so that disengaged participants do not distort
    # the group accuracy statistics used in criterion 2.
    # Reference: Haridi et al., 2025
    data, excluded_timeout = exclude_timeout_participants(data)

    # ------------------------------------------------------------------
    # 4. Exclusion criterion 2 â€” Accuracy outliers
    # ------------------------------------------------------------------
    # Participants whose accuracy is > 2.5 SD from the group mean in
    # either the low (expected 55%) or high (expected 85%) condition are
    # removed. Applied after criterion 1 so group stats are not skewed
    # by non-engaged participants.
    data, excluded_accuracy, accuracy_report = exclude_accuracy_outliers(data)

    # ------------------------------------------------------------------
    # 5. Load recognition data (separate DataFrame: recog_data)
    # ------------------------------------------------------------------
    # Recognition trials have a different structure from the main task
    # (different columns, no control_condition, etc.) so they live in
    # their own DataFrame throughout the analysis pipeline.
    print("\n" + "=" * 60)
    print("LOADING RECOGNITION DATA")
    print("=" * 60)
    recog_data = load_recognition_data()

    if recog_data is None:
        print("No recognition data loaded. Skipping recognition analyses.")
    else:
        # ---------------------------------------------------------------
        # 6. Recognition exclusion â€” long RTs (> 20 s)
        # ---------------------------------------------------------------
        # This is a trial-level exclusion: only the individual outlier
        # trials are dropped; the participant's other trials are kept.
        # Reference: Haridi et al., 2025
        recog_data, n_rt_excluded, rt_exclusion_details = exclude_long_rt_trials(recog_data)

    # ------------------------------------------------------------------
    # 7. Cross-dataset synchronisation
    # ------------------------------------------------------------------
    # After all independent per-dataset exclusions are done, make sure
    # both DataFrames share exactly the same participant IDs.
    # Any participant only in one dataset is removed from that dataset.
    if recog_data is not None:
        data, recog_data = sync_participant_ids(data, recog_data)
    else:
        print("Skipping synchronisation (no recognition data loaded).")

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FINAL EXCLUSION SUMMARY")
    print("=" * 60)
    print("  [Main task data]")
    print(f"    Excluded -- timeout     : {excluded_timeout if excluded_timeout else 'none'}")
    print(f"    Excluded -- accuracy    : {excluded_accuracy if excluded_accuracy else 'none'}")
    print(f"    Participants remaining  : {data['participant'].nunique()}")
    print(f"    Rows remaining          : {len(data)}")
    if recog_data is not None:
        print("  [Recognition data]")
        print(f"    Trials excluded (RT>20s): {n_rt_excluded}")
        print(f"    Trials remaining         : {len(recog_data)}")
        print(f"    Participants remaining    : {recog_data['participant'].nunique()}")
    print("  [Final participant IDs in both datasets]")
    final_ids = sorted(data['participant'].unique().tolist())
    print(f"    {final_ids}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 9. Descriptive Statistics
    # ------------------------------------------------------------------
    # Calculated on the final, cleaned dataset so excluded participants
    # are automatically excluded from the descriptives as well.
    print_descriptives(data)

    # ------------------------------------------------------------------
    # 10. Sanity Checks & Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECKS & PLOTS")
    print("=" * 60)
    
    # We only want to plot the test phase data since calibration is dynamic/different
    if 'phase' in data.columns:
        test_data = data[data['phase'] == 'test'].copy()
    else:
        test_data = data.copy()
        
    # Ensure variables are numeric
    test_data['detection_accuracy'] = pd.to_numeric(test_data['detection_accuracy'], errors='coerce')
    test_data['prop_used'] = pd.to_numeric(test_data['prop_used'], errors='coerce')
    
    s_results = sanity_check(test_data)
    if 'accuracy_by_condition' in s_results:
        print("\nAccuracy by Condition:")
        print(s_results['accuracy_by_condition'].to_string())
        print()
    
    plot_sanity_check(test_data, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 11. Recognition Memory Analysis
    # ------------------------------------------------------------------
    if recog_data is not None:
        print("\n" + "=" * 60)
        print("RECOGNITION MEMORY (D-PRIME)")
        print("=" * 60)
        
        mem_results, targets = analyze_recognition(data, recog_data)
        if mem_results is not None:
            print("\nPer-participant Hit Rate, FA Rate, and D-prime:")
            print("-" * 60)
            print(mem_results.to_string(index=False))
            print("-" * 60)
            
            # Group level summary
            print("\nGroup Summary:")
            summary = mem_results.groupby('control_condition')[['Hit_rate', 'FA_rate', 'd_prime']].agg(['mean', 'std'])
            print(summary.to_string())
        # 11b. Statistical Analyses
        run_recognition_stats(mem_results, targets)



