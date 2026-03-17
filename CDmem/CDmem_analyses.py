#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem Analysis Script
=====================

Analysis script for the CDmem (Control Detection + Memory) experiment.
Loads all subject data files and performs various analyses.

Data file naming convention: CDmem_{participant}_{session}_{block}.csv
Located in: CDmem/data/subjects/
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "subjects"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(data_dir=DATA_DIR):
    """
    Load all CDmem subject data files and combine into a single DataFrame.

    File naming pattern: CDmem_{participant}_{session}_{block}.csv
    (e.g. CDmem_1_1_1.csv for participant 1, CDmem_1_2_1.csv for participant 2)

    The function deliberately excludes:
      - CDmem_1_1.csv style files (only 2 numeric parts â€” intermediate output)
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
    # only match files with exactly 3 underscore-separated numeric parts.
    pattern = str(data_dir / "CDmem_*.csv")
    all_files = glob.glob(pattern)

    # Regex breakdown:
    #   ^          â€” start of filename
    #   CDmem_     â€” literal prefix
    #   \d+_\d+_\d+ â€” exactly three groups of digits separated by underscores
    #   \.csv$     â€” ends with .csv
    strict_pattern = re.compile(r"^CDmem_\d+_\d+_\d+\.csv$")
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
        exit(1)

    # ------------------------------------------------------------------
    # 2. Sanity check: print column names and overall shape
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(f"Total columns: {len(data.columns)}\n")
    for i, col in enumerate(data.columns, start=1):
        print(f"  {i:>3}. {col}")

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
