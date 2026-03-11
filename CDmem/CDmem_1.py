#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem_1.py
==========
Control Detection and Memory Experiment — Version 1

Overview
--------
This script runs a PsychoPy experiment investigating how people detect
control over moving shapes, and how this detection is influenced by memory
across trials (serial dependence / history effects).

Experiment structure:
   1. CALIBRATION PHASE
     A QUEST+ Bayesian adaptive procedure estimates each participant's
     psychometric function and finds the "self-proportion" (prop) at which
     they can correctly identify the controlled shape ~75% of the time.

  2. TEST PHASE (4 blocks)
     Four difficulty levels are derived from the calibrated threshold.
     Each level is presented as its own block (all trials in a block share
     the same control level). Block order is counterbalanced across
     participants using all 24 permutations of the 4 levels.

Trial mechanics (same as MT Inference.py):
  - Two shapes appear on screen: a Square and a Circle.
  - One shape (the "target") has its direction partially influenced by the
    participant's mouse movement. The other (the "distractor") moves
    independently along a pre-recorded trajectory.
  - The proportion of mouse influence is controlled by `prop` (0 = fully
    autonomous, 1 = fully mouse-driven).
  - After the motion phase, participants identify which shape they controlled
    and rate their confidence and sense of agency.

Adapted from: MT Inference.py (Multi-Timescale Inference experiment)
Author note: The 4-shape complexity condition has been removed. All trials
             use 2 shapes. The test phase is restructured into 4 separate
             blocks (one per difficulty level) with counterbalanced order.
"""

import os
import sys
import math
import random
import pathlib
import datetime
import atexit
import hashlib
import json
import subprocess

import serial #for sending triggers via triggerbox
import time 


# ─────────────────────────────────────────────────────────────────────────────
#  PYTHON INTERPRETER CHECK
#  PsychoPy requires specific packages (numpy, pandas, psychopy). If they are
#  not available in the current Python environment, this block tries to find
#  a compatible Python installation and re-launches the script with it.
# ─────────────────────────────────────────────────────────────────────────────

def check_and_run_with_correct_python():
    """
    Checks whether the required packages are available. If not, searches
    for a Python interpreter that has them and re-launches this script.
    Returns False if packages are available (script should continue normally).
    """
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # All good — continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Searching for a compatible Python interpreter...")

        # Common locations for Python with PsychoPy installed
        python_paths = [
            "C:/Program Files/PsychoPy/python.exe",           # Standalone PsychoPy
            "C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe",
            "C:/Users/knogl/Miniconda3/python.exe",
            "/opt/anaconda3/bin/python",                       # macOS Anaconda
            "/usr/bin/python3",                                # Linux
        ]

        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)

        print("Error: No compatible Python found. Please install psychopy, numpy, and pandas.")
        sys.exit(1)


# Run the check — if it returns True, the script was re-launched and we exit.
if check_and_run_with_correct_python():
    sys.exit(0)

# ---------------------------------------------------------------------------
# INITIALIZE TRIGGERBOX
USE_TRIGGERS = False  # Set to False to disable EEG triggers manually

# Replace 'COM3' with the actual port found in Device Manager
try:
    if USE_TRIGGERS:
        port = serial.Serial('COM3') 
        port.write(b'\x00')  # Ensure it starts at zero
        TRIGGERBOX_READY = True
    else:
        print("EEG triggers are DISABLED via USE_TRIGGERS flag.")
        TRIGGERBOX_READY = False
        port = None
except Exception as e:
    print(f"WARNING: Could not connect to TriggerBox on COM3: {e}")
    print("Experiment will continue without EEG triggers.")
    TRIGGERBOX_READY = False
    port = None

def send_trigger(val):
    """Sends a trigger byte and resets it after a short delay."""
    if not TRIGGERBOX_READY or port is None:
        return
    try:
        port.write(bytes([val]))      # Set the trigger lines
        time.sleep(0.01)              # Wait 10ms (standard pulse width)
        port.write(b'\x00')           # Reset all lines to zero
    except Exception as e:
        print(f"Error sending trigger {val}: {e}")

# ---------------------------------------------------------------------------
#  IMPORTS (available after interpreter check)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL KINEMATICS STORAGE
#  Frame-by-frame mouse and shape position data are collected during each trial
#  and saved to a separate kinematics CSV at the end of the experiment.
# ─────────────────────────────────────────────────────────────────────────────

kinematics_data = []       # List of dicts, one per frame across all trials
kinematics_csv_path = ""   # Will be set after participant dialog


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-SAVE ON QUIT
#  If the experiment is closed unexpectedly (e.g. crash or Escape key),
#  this function saves whatever data has been collected so far.
# ─────────────────────────────────────────────────────────────────────────────

_saved = False  # Flag to prevent saving twice

def _save():
    """Save main trial data and kinematics data to CSV files."""
    global _saved
    if not _saved:
        if 'thisExp' in globals() and thisExp is not None:
            thisExp.saveAsWideText(csv_path)
            print("Main data auto-saved ->", csv_path)
            if kinematics_data:
                kinematics_df = pd.DataFrame(kinematics_data)
                kinematics_df.to_csv(kinematics_csv_path, index=False)
                print("Kinematics data auto-saved ->", kinematics_csv_path)
        else:
            print("Experiment not initialized — no data to save.")
        _saved = True

atexit.register(_save)  # Register _save to run automatically when Python exits


# ─────────────────────────────────────────────────────────────────────────────
#  PARTICIPANT DIALOG & EXPERIMENT SETTINGS
#  A dialog box collects participant ID and session number.
#  Two special modes are available:
#    - simulate: runs the experiment with a virtual mouse (no real input needed)
#    - check_mode: uses fewer trials for quick testing of the script
# ─────────────────────────────────────────────────────────────────────────────

expName = "CDmem_1"

# AUTO_TEST mode: skip the dialog entirely and run a fast simulation.
# Activated by setting the environment variable CDT_AUTO_TEST=true or
# passing --autotest as a command-line argument.
AUTO_TEST = os.environ.get('CDT_AUTO_TEST', '').lower() == 'true' or '--autotest' in sys.argv

if AUTO_TEST:
    print("AUTO-TEST MODE: Skipping dialog, running simulation")
    expInfo = {"participant": "AUTO_TEST", "session": "001"}
    SIMULATE = True
    CHECK_MODE = True
else:
    expInfo = {"participant": "", "session": "001", "age": "", "gender": "", "handedness": "", "simulate": False, "check_mode": False}
    dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "age", "gender", "handedness", "simulate", "check_mode"], title=expName)
    if not dlg.OK:
        core.quit()  # User pressed Cancel
    SIMULATE = bool(expInfo.pop("simulate"))
    CHECK_MODE = bool(expInfo.pop("check_mode"))
    if SIMULATE:
        expInfo["participant"] = "SIM"

# Set trial counts depending on mode.
# CHECK_MODE uses minimal trials so the experimenter can quickly verify
# that the script runs correctly end-to-end.
if CHECK_MODE:
    CHECK_CALIBRATION_TRIALS = 6     # Minimum trials for QUEST+ (matches MTI check mode)
    CHECK_TEST_TRIALS_PER_LEVEL = 5  # Trials per difficulty level per block
else:
    CHECK_CALIBRATION_TRIALS = 60    # Full QUEST+ calibration (matches MTI)
    CHECK_TEST_TRIALS_PER_LEVEL = 20 # 20 trials per miniblock (6 miniblocks × 20 = 120 total)

if CHECK_MODE:
    print("=" * 60)
    print("** CHECK MODE ENABLED — Running minimal trials **")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/miniblock × 6 miniblocks = {CHECK_TEST_TRIALS_PER_LEVEL * 6} total")
    print("=" * 60)
else:
    print("Running FULL experiment mode")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/miniblock × 6 miniblocks = {CHECK_TEST_TRIALS_PER_LEVEL * 6} total")


# ─────────────────────────────────────────────────────────────────────────────
#  SCREENSHOT SETTINGS
#  Screenshots are saved automatically at the first frame of each phase.
#  Saved to: CDmem/screenshots/
# ─────────────────────────────────────────────────────────────────────────────

SCREENSHOTS_DIR = pathlib.Path(__file__).parent / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Tracks which phases have already had a screenshot saved this session.
# Keys: e.g. 'calibration_frame001', 'test_frame001', 'memory_test'
_screenshots_saved = set()


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION LIBRARY
#  Pre-recorded cursor movement snippets are stored as velocity arrays in
#  core_pool.npy. Each snippet is a sequence of (dx, dy) displacements.
#  These are used to drive the autonomous movement of shapes on screen.
#
#  Additional files:
#    core_pool_feats.npy  — feature vectors for each snippet (used for clustering)
#    core_pool_labels.npy — cluster labels (k=4 clusters of movement styles)
#    scaler_params.json   — mean/std used to normalize features
#    cluster_centroids.json — centroid coordinates for each cluster
# ─────────────────────────────────────────────────────────────────────────────

script_dir = pathlib.Path(__file__).parent  # Directory containing this script

LIB_NAME    = script_dir / "Motion_Library" / "core_pool.npy"
FEATS_NAME  = script_dir / "Motion_Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir / "Motion_Library" / "core_pool_labels.npy"

motion_pool      = np.load(LIB_NAME)    # Shape: (n_snippets, snippet_length, 2)
snippet_features = np.load(FEATS_NAME)  # Shape: (n_snippets, n_features)
snippet_labels   = np.load(LABELS_NAME) # Shape: (n_snippets,) — cluster IDs

SNIP_LEN    = motion_pool.shape[1]  # Number of frames per snippet
TOTAL_SNIPS = motion_pool.shape[0]  # Total number of snippets
K_CLUST     = 4                     # Number of movement clusters

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

# Load feature scaler parameters (used to normalize snippet features)
with open(script_dir / "Motion_Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"],  dtype=np.float32)
scaler_std  = np.array(scp["scale"], dtype=np.float32)

# Load cluster centroids (used for trajectory quality scoring)
with open(script_dir / "Motion_Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

# Create a participant-specific random number generator seeded by participant ID.
# Using a deterministic seed means the same participant always gets the same
# random sequence — useful for reproducibility.
participant_clusters = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY QUALITY FUNCTIONS
#  Before using snippets in the experiment, we filter out low-quality ones
#  (e.g. snippets with too little movement, too much jitter, or erratic speed).
#  Valid snippets are then normalized to a consistent speed range.
# ─────────────────────────────────────────────────────────────────────────────

def get_trajectory_signature(trajectory):
    """
    Compute key movement statistics for a trajectory (position array).
    Used to score trajectory quality and select the best snippets.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 2)
        Array of (x, y) positions over time.

    Returns
    -------
    dict with keys: mean_speed, speed_variability, path_length,
                    net_displacement, speed_percentiles
    """
    velocities = np.diff(trajectory, axis=0)  # Frame-to-frame displacements
    if len(velocities) == 0:
        return {'mean_speed': 0, 'speed_variability': 0, 'path_length': 0,
                'net_displacement': 0, 'speed_percentiles': np.array([0, 0, 0])}
    speeds = np.linalg.norm(velocities, axis=1)  # Scalar speed per frame
    return {
        'mean_speed':        np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length':       np.sum(speeds),
        'net_displacement':  np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }


def analyze_trajectory_quality(trajectory):
    """
    Compute detailed quality metrics for a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 2)
        Position array (cumulative sum of velocity snippet).

    Returns
    -------
    dict with quality metrics used by is_trajectory_valid().
    """
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed  = np.std(speeds)
    max_speed  = np.max(speeds)
    min_speed  = np.min(speeds)

    # Fraction of frames where the shape is nearly stationary
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)

    # Fraction of frames with unusually high speed (outliers)
    high_jitter_ratio = np.sum(speeds > mean_speed + 3 * std_speed) / len(speeds)

    # Compute direction changes (jerkiness) from unit velocity vectors
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        angle_changes = np.arccos(
            np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1)
        )
        mean_angle_change = np.mean(angle_changes)
        jerkiness = np.std(angle_changes)
    else:
        mean_angle_change = 0
        jerkiness = 0

    return {
        'mean_speed':          mean_speed,
        'std_speed':           std_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio':   high_jitter_ratio,
        'mean_angle_change':   mean_angle_change,
        'jerkiness':           jerkiness,
        'speed_range':         max_speed - min_speed
    }


def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3,
                         max_jitter_ratio=0.1, max_jerkiness=1.5):
    """
    Decide whether a trajectory meets quality standards.

    A trajectory is rejected if:
      - Mean speed is too low (shape barely moves)
      - Too many near-zero-speed frames (shape stalls)
      - Too many high-jitter frames (speed spikes)
      - Too jerky (erratic direction changes)

    Returns
    -------
    (bool, str) — (is_valid, reason_if_invalid)
    """
    quality = analyze_trajectory_quality(trajectory)
    if quality['mean_speed'] < min_speed:
        return False, "mean_speed_too_low"
    if quality['zero_movement_ratio'] > max_zero_ratio:
        return False, "too_much_zero_movement"
    if quality['high_jitter_ratio'] > max_jitter_ratio:
        return False, "too_much_jitter"
    if quality['jerkiness'] > max_jerkiness:
        return False, "too_jerky"
    return True, "valid"


def normalize_trajectory(trajectory, target_speed_range=(5.0, 15.0), smooth_factor=0.35):
    """
    Rescale and smooth a trajectory so all snippets have comparable speed.

    Steps:
      1. Scale all velocities so mean speed matches the target range midpoint.
      2. Apply exponential smoothing to reduce abrupt speed changes.

    Parameters
    ----------
    trajectory    : np.ndarray, shape (T, 2) — position array
    target_speed_range : (min, max) target mean speed in pixels/frame
    smooth_factor : weight given to the previous frame's velocity (0–1)

    Returns
    -------
    np.ndarray, shape (T, 2) — normalized position array
    """
    if len(trajectory) < 2:
        return trajectory
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    current_mean_speed = np.mean(speeds)
    if current_mean_speed > 0:
        target_mean_speed = np.mean(target_speed_range)
        speed_scale = target_mean_speed / current_mean_speed
        velocities = velocities * speed_scale

    # Exponential smoothing: each frame's velocity is a blend of the
    # previous smoothed velocity and the current raw velocity.
    smoothed_velocities = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed_velocities[i] = (smooth_factor * smoothed_velocities[i - 1]
                                  + (1 - smooth_factor) * velocities[i])

    # Reconstruct position array from smoothed velocities
    normalized_trajectory = [trajectory[0]]
    for vel in smoothed_velocities:
        normalized_trajectory.append(normalized_trajectory[-1] + vel)
    return np.array(normalized_trajectory)


def preprocess_motion_pool():
    """
    Filter and normalize all snippets in the motion pool.

    For each snippet:
      1. Convert velocity array to position array (cumulative sum).
      2. Check quality — discard if it fails any criterion.
      3. Normalize speed and smoothness.
      4. Convert back to velocity array (differences of positions).

    Updates the global motion_pool, snippet_features, and snippet_labels
    in-place, keeping only valid snippets.

    Returns
    -------
    list of int — indices of valid snippets (0 to N-1 after filtering)
    """
    global motion_pool, snippet_features, snippet_labels, SNIP_LEN
    print("Preprocessing motion pool for quality control...")
    initial_count = len(motion_pool)

    processed_snippets  = []
    processed_features  = []
    processed_labels    = []

    for i, snippet in enumerate(motion_pool):
        # Convert velocity snippet to position trajectory for quality checks
        trajectory = np.cumsum(snippet, axis=0)
        is_valid, reason = is_trajectory_valid(trajectory)

        if is_valid:
            normalized_trajectory = normalize_trajectory(trajectory)
            # Convert back to velocities (differences between consecutive positions)
            velocities = np.diff(normalized_trajectory, axis=0)
            processed_snippets.append(velocities)
            processed_features.append(snippet_features[i])
            processed_labels.append(snippet_labels[i])
        else:
            print(f"  Removed snippet {i}: {reason}")

    # Replace global arrays with filtered versions
    motion_pool      = np.array(processed_snippets)
    snippet_features = np.array(processed_features)
    snippet_labels   = np.array(processed_labels)
    SNIP_LEN = motion_pool.shape[1] if len(motion_pool) > 0 else 0

    print(f"Motion pool preprocessed: kept {len(processed_snippets)}/{initial_count} snippets")
    return list(range(len(processed_snippets)))


# Run preprocessing immediately at startup
valid_snippet_indices = preprocess_motion_pool()


# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSAL TRAJECTORY SET
#  To ensure all participants see trajectories of comparable quality, we
#  pre-select a fixed set of the best snippets (ranked by a quality score).
#  This set is the same for every participant (seeded with 42, not the
#  participant seed), ensuring cross-participant comparability.
#
#  Primary set:  1,240 best snippets (used first)
#  Overflow set: next 40 snippets (used if primary runs out)
# ─────────────────────────────────────────────────────────────────────────────

# These globals are populated by select_universal_trajectory_set()
universal_trajectory_set_primary  = []
universal_trajectory_set_overflow = []
universal_trajectory_set          = []
used_trajectory_indices           = set()   # Tracks which snippets have been used
trajectory_usage_stats            = {"used_count": 0, "total_needed": 2000}

# Global trial counter — increments across all phases for consistent numbering
global_trial_counter = 0


def select_universal_trajectory_set():
    """
    Score all valid snippets by movement quality and select the top 1,240
    as the primary set (plus up to 40 overflow).

    Quality score = speed_score × variability_score × length_score
      - speed_score:       how close mean speed is to 8 px/frame
      - variability_score: how close speed std is to 3 px/frame
      - length_score:      path length relative to 100 px (capped at 1.0)

    Uses a fixed random seed (42) so the selection is identical for all
    participants.
    """
    global valid_snippet_indices
    global universal_trajectory_set_primary, universal_trajectory_set_overflow
    global universal_trajectory_set

    total_valid = len(valid_snippet_indices)
    if total_valid < 1240:
        print(f"Warning: Only {total_valid} valid trajectories (fewer than 1,240 needed)")
        universal_trajectory_set_primary  = valid_snippet_indices.copy()
        universal_trajectory_set_overflow = []
        universal_trajectory_set          = universal_trajectory_set_primary.copy()
        return universal_trajectory_set.copy()

    selection_rng = np.random.default_rng(42)  # Fixed seed — same for all participants
    print("Selecting universal trajectory sets (Primary 1,240 + Overflow 40)...")

    trajectory_scores = []
    for idx in valid_snippet_indices:
        trajectory = motion_pool[idx]
        traj_cumsum = np.cumsum(trajectory, axis=0)
        sig = get_trajectory_signature(traj_cumsum)

        speed_score       = 1.0 / (1.0 + abs(sig['mean_speed'] - 8.0))
        variability_score = 1.0 / (1.0 + abs(sig['speed_variability'] - 3.0))
        length_score      = min(1.0, sig['path_length'] / 100.0)
        overall_score     = speed_score * variability_score * length_score
        trajectory_scores.append((overall_score, idx))

    trajectory_scores.sort(reverse=True)  # Best snippets first

    primary_indices  = [idx for score, idx in trajectory_scores[:1240]]
    overflow_indices = ([idx for score, idx in trajectory_scores[1240:1280]]
                        if total_valid >= 1280 else [])

    universal_trajectory_set_primary  = primary_indices
    universal_trajectory_set_overflow = overflow_indices
    universal_trajectory_set          = primary_indices + overflow_indices

    print(f"  Primary: {len(primary_indices)} snippets (best score={trajectory_scores[0][0]:.3f})")
    if overflow_indices:
        print(f"  Overflow: {len(overflow_indices)} snippets")
    else:
        print("  No overflow set (valid < 1,280)")

    return universal_trajectory_set.copy()


# Initialize the universal set after preprocessing
universal_trajectory_set = select_universal_trajectory_set()
print(f"Universal set: Primary={len(universal_trajectory_set_primary)}, "
      f"Overflow={len(universal_trajectory_set_overflow)}, "
      f"Total={len(universal_trajectory_set)}")


def get_trajectory_indices(n_trajectories):
    """
    Select n unique trajectory indices for a single trial.

    Preference order:
      1. Unused snippets from the primary set
      2. Unused snippets from the overflow set (if primary exhausted)
      3. Any snippet from the combined set (if both exhausted)
      4. Emergency fallback: any valid snippet

    Parameters
    ----------
    n_trajectories : int — how many snippets to select (2 for this experiment)

    Returns
    -------
    list of int — selected snippet indices
    """
    global used_trajectory_indices, trajectory_usage_stats

    available_primary  = [i for i in universal_trajectory_set_primary
                          if i not in used_trajectory_indices]
    available_overflow = [i for i in universal_trajectory_set_overflow
                          if i not in used_trajectory_indices]
    available_indices  = (available_primary if len(available_primary) >= n_trajectories
                          else available_primary + available_overflow)

    if len(available_indices) >= n_trajectories:
        selected = rng.choice(available_indices, size=n_trajectories, replace=False)
        for idx in selected:
            used_trajectory_indices.add(idx)
        trajectory_usage_stats["used_count"] += n_trajectories
        return list(selected)
    else:
        # Fallback: reuse from combined set
        combined = universal_trajectory_set_primary + universal_trajectory_set_overflow
        if len(combined) >= n_trajectories:
            return list(rng.choice(combined, size=n_trajectories, replace=False))
        else:
            # Emergency: use any valid snippet
            return list(rng.choice(valid_snippet_indices, size=n_trajectories, replace=False))


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY SMOOTHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_consistent_smoothing(trajectory1, trajectory2):
    """
    Apply a moving-average window to two trajectories simultaneously.
    This ensures both the target and distractor have similar smoothness,
    preventing participants from using smoothness as a cue.

    Parameters
    ----------
    trajectory1, trajectory2 : np.ndarray, shape (T, 2) — velocity arrays

    Returns
    -------
    (vel1, vel2) — smoothed velocity arrays
    """
    def smooth_trajectory(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end   = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed

    # Convert velocity → position → smooth → back to velocity
    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    return vel1, vel2


def smooth_single_trajectory(trajectory):
    """Apply moving-average smoothing to a single velocity array."""
    def smooth_traj(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end   = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed

    pos = np.cumsum(trajectory, axis=0)
    smooth_pos = smooth_traj(pos)
    vel = np.diff(smooth_pos, axis=0)
    return vel


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION MIXING FUNCTION
#  This is the core of the control manipulation.
#
#  The target shape moves at the same speed as the distractor (trajectory
#  speed), but its DIRECTION is a weighted blend of:
#    - The participant's mouse direction (weight = prop)
#    - The pre-recorded trajectory direction (weight = 1 - prop)
#
#  By keeping speed constant and only varying direction, the task difficulty
#  is controlled purely by how much the mouse direction "leaks" into the
#  target's movement — making it harder to detect at low prop values.
# ─────────────────────────────────────────────────────────────────────────────

def mix_direction_only(mouse_dx, mouse_dy, traj_dx, traj_dy, prop):
    """
    Blend mouse and trajectory directions while preserving trajectory speed.

    Parameters
    ----------
    mouse_dx, mouse_dy : float — mouse displacement this frame
    traj_dx, traj_dy   : float — trajectory velocity this frame
    prop               : float in [0, 1] — proportion of mouse influence
                         (0 = fully autonomous, 1 = fully mouse-driven)

    Returns
    -------
    (vx, vy) : float — blended velocity with trajectory magnitude
    """
    traj_speed = math.hypot(traj_dx, traj_dy)
    if traj_speed < 0.01:
        return traj_dx, traj_dy  # Shape is nearly stationary — no mixing needed

    mouse_mag = math.hypot(mouse_dx, mouse_dy)
    if mouse_mag < 0.01:
        return traj_dx, traj_dy  # Mouse is stationary — follow trajectory only

    # Compute unit direction vectors
    m_dir_x, m_dir_y = mouse_dx / mouse_mag, mouse_dy / mouse_mag
    t_dir_x, t_dir_y = traj_dx / traj_speed, traj_dy / traj_speed

    # Blend directions: prop controls how much mouse direction is used
    mix_x = prop * m_dir_x + (1 - prop) * t_dir_x
    mix_y = prop * m_dir_y + (1 - prop) * t_dir_y

    # Normalize the blended direction and scale to trajectory speed
    mix_mag = math.hypot(mix_x, mix_y)
    if mix_mag > 0.01:
        return (mix_x / mix_mag) * traj_speed, (mix_y / mix_mag) * traj_speed
    return traj_dx, traj_dy


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OFFSET_X         = 300    # Horizontal distance from center to shape start position (px)
OFFSET_Y         = 150    # Vertical distance from center to shape start position (px)
LOWPASS          = 0.2    # Low-pass filter weight: lower = less smoothing, more responsive
SPEED_MULTIPLIER = 1.3    # Multiply trajectory velocities to make shapes move faster

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGINE DATASET CONSTANTS
#  Familiar images are sampled from CARA_prep chosen_stimuli dataset.
#  They are paired (e.g., alpaca_03s.jpg and alpaca_07s.jpg). We use one from
#  each pair in the test phase and the other in the memory test phase.
# ─────────────────────────────────────────────────────────────────────────────

IMAGINE_DIR  = pathlib.Path(r"C:\Users\elifg\Desktop\PHD\CARA_prep\stimulus_prep\chosen_stimuli")
IMAGE_SIZE   = (200, 200)   # Display size in pixels — matches the 40×40 px shapes
# Number of unique pairs needed for TEST phase = 120 trials (full) or 30 trials (check mode)
N_IMAGES     = (CHECK_TEST_TRIALS_PER_LEVEL * 6) if CHECK_MODE else 120
IMAGE_SEED   = 42         # Fixed seed for reproducible sampling across runs
IMAGE_LOG    = pathlib.Path(__file__).parent / "image_stimuli_log.json"


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE SAMPLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_and_log_images(img_dir, n, seed):
    """
    Randomly sample `n` unique pairs from `img_dir` and split them.
    
    Images in `chosen_stimuli` are named like `alpaca_03s.jpg`, `alpaca_07s.jpg`.
    We group them by the stem before the last underscore, randomly assign one 
    to test_group and one to recognition_group. Then we sample `n` objects from 
    these to use in the experiment, leaving the extras unused.
    
    Returns:
       test_images: list of dicts for the test phase
       recognition_images: list of dicts for the recognition phase (as foils)
    """
    all_jpgs = sorted(img_dir.glob("*.jpg"))  # Sort for determinism
    
    # Group by object name (everything before the last underscore)
    from collections import defaultdict
    pairs = defaultdict(list)
    for p in all_jpgs:
        # e.g. "alpaca_03s.jpg" -> stem "alpaca_03s" -> split "_" -> "alpaca"
        object_name = "_".join(p.stem.split("_")[:-1])
        pairs[object_name].append(p)
        
    # Keep only valid pairs (exactly 2 images)
    valid_pairs = {k: v for k, v in pairs.items() if len(v) == 2}
    
    rng_img  = random.Random(seed)            # Isolated RNG so experiment RNG is unaffected
    sampled_object_keys = rng_img.sample(list(valid_pairs.keys()), n)
    
    test_images = []
    recognition_images = []
    
    for obj_key in sampled_object_keys:
        img_list = valid_pairs[obj_key]
        # Randomly assign one to test, one to recognition
        if rng_img.choice([True, False]):
            test_images.append({'filename': img_list[0].stem, 'path': str(img_list[0])})
            recognition_images.append({'filename': img_list[1].stem, 'path': str(img_list[1])})
        else:
            test_images.append({'filename': img_list[1].stem, 'path': str(img_list[1])})
            recognition_images.append({'filename': img_list[0].stem, 'path': str(img_list[0])})

    records = {
        'seed': seed, 
        'n_pairs_sampled': n, 
        'test_images': test_images,
        'recognition_images': recognition_images
    }

    with open(IMAGE_LOG, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print(f"[IMAGINE] Sampled {n} object pairs. Log saved to: {IMAGE_LOG}")
    return test_images, recognition_images
    """
    Randomly sample `n` PNG images from `img_dir` and write a JSON log.

    The log is saved to IMAGE_LOG and contains, for each sampled image:
      - 'filename'  : stem name (no extension)
      - 'path'      : absolute path to the PNG file

    Returns a list of dicts with keys 'filename' and 'path'.
    """
    all_pngs = sorted(img_dir.glob("*.png"))  # Sort for determinism
    rng_img  = random.Random(seed)            # Isolated RNG so experiment RNG is unaffected
    sampled  = rng_img.sample(all_pngs, n)

    records = [
        {'filename': p.stem, 'path': str(p)}
        for p in sampled
    ]

    with open(IMAGE_LOG, 'w', encoding='utf-8') as f:
        json.dump({'seed': seed, 'n_images': n, 'images': records}, f, indent=2)

    print(f"[IMAGINE] Sampled {n} images. Log saved to: {IMAGE_LOG}")
    return records


def make_image_pairs(images, seed):
    """
    Pair `test_images` into non-overlapping pairs for simultaneous presentation.
    
    Since each object concept only appears once in `test_images`, any random 
    pairing of two different list elements will naturally pair two DIFFERENT objects.
    
    Returns a list of tuples: [(dict_A, dict_B), ...]
    """
    assert len(images) % 2 == 0, "Need an even number of images to form pairs"
    
    # Shuffle locally to create random pairings
    rng_img = random.Random(seed)
    shuffled = list(images)
    rng_img.shuffle(shuffled)
    
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  FILE PATHS & EXPERIMENT HANDLER
#  PsychoPy's ExperimentHandler manages trial-by-trial data logging.
#  We also define paths for the main CSV and the kinematics CSV.
# ─────────────────────────────────────────────────────────────────────────────

root         = pathlib.Path(__file__).parent / "data"
subjects_dir = root / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

participant_id = expInfo['participant']
base_filename  = f"CDmem_1_{participant_id}"
csv_path            = subjects_dir / f"{base_filename}.csv"
kinematics_csv_path = subjects_dir / f"{base_filename}_kinematics.csv"

# If a file with this name already exists, append a number to avoid overwriting
i = 1
while csv_path.exists():
    new_filename        = f"CDmem_1_{participant_id}_{i}"
    csv_path            = subjects_dir / f"{new_filename}.csv"
    kinematics_csv_path = subjects_dir / f"{new_filename}_kinematics.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename)
)


# ─────────────────────────────────────────────────────────────────────────────
#  PSYCHOPY WINDOW & STIMULI
# ─────────────────────────────────────────────────────────────────────────────

# Open a fullscreen window (or windowed in simulate mode for easier debugging)
win = visual.Window((1920, 1080), fullscr=not SIMULATE, color=[0.5] * 3,
                    units="pix", allowGUI=True)
win.setMouseVisible(False)

# The two shapes used in CALIBRATION trials
square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, 20, fillColor="black", lineColor="black")

# ── Sample IMAGINE images and build trial pairs ───────────────────────────────
# Done here (after the window is open) so the log is always written before
# the experiment starts. The participant dialog has already run at this point.
sampled_test_images, foil_images = sample_and_log_images(IMAGINE_DIR, N_IMAGES, IMAGE_SEED)

# Build non-overlapping pairs for the test phase trials
# Use participant seed so the simultaneous pairings are random but reproducible per participant
_pair_seed = int(hashlib.md5(expInfo['participant'].encode()).hexdigest(), 16) % (2**32)
image_pairs = make_image_pairs(sampled_test_images, _pair_seed)

# Global index into image_pairs — incremented by each test trial
pair_index = 0

# ── 200 FOIL images for the memory test ───────────────────────────────
# We already populated `foil_images` from the paired subset sampled above.
# The `foil_images` list contains exactly N_IMAGES (120) unseen images that are 
# the paired counterparts to the `sampled_test_images`.

# Fixation cross shown at the start of each trial
fix = visual.TextStim(win, "+", color="white", height=60)

# General-purpose message text (instructions, response prompts, etc.)
msg = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)

# Feedback text shown after calibration trials ("Right" / "Wrong")
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

# Helper: confine a position to a circle of radius l around the screen center.
# Prevents shapes from flying off-screen.
confine = lambda p, l=400: p if (r := math.hypot(*p)) <= l else (p[0] * l / r, p[1] * l / r)

# Helper: rotate a 2D vector (vx, vy) by angle a (degrees).
# Used to apply an angle bias to mouse input (not used in this version, kept for compatibility).
rotate = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS
#  When SIMULATE=True, a virtual mouse replaces real mouse input and key
#  presses are generated automatically. This allows the script to be tested
#  without a human participant.
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedMouse:
    """A fake mouse that drifts randomly, used in simulation mode."""
    def __init__(self):
        self._pos = np.array([0.0, 0.0], dtype=float)

    def setPos(self, pos=(0, 0)):
        self._pos = np.array(pos, dtype=float)

    def getPos(self):
        # Add small random noise to simulate natural hand movement
        self._pos += rng.normal(0, 3, 2)
        return self._pos.tolist()


def wait_keys(keys=None):
    """
    Wait for a key press. In simulation mode, returns a random valid key
    immediately without waiting for real input.
    """
    if SIMULATE:
        if keys is None:
            core.wait(0.2)
            return ["space"]
        allowed = [k for k in keys if k != "escape"] or ["space"]
        return [rng.choice(allowed)]
    return event.waitKeys(keyList=keys)


# ─────────────────────────────────────────────────────────────────────────────
#  BREAK SCREEN
#  Shown automatically every 50 trials within a phase, and between blocks.
# ─────────────────────────────────────────────────────────────────────────────

def show_break_screen(trials_completed, total_trials_in_block, block_label):
    """
    Display a 30-second countdown break screen.

    Parameters
    ----------
    trials_completed      : int — how many trials done so far in this phase
    total_trials_in_block : int — total trials in this phase
    block_label           : str — descriptive label for the current block
    """
    break_msg = visual.TextStim(
        win=win,
        text=f"BREAK TIME\n\nCompleted {trials_completed} trials.\n"
             f"Progress: {trials_completed}/{total_trials_in_block} ({block_label})\n\n"
             f"Break time remaining: 30 seconds",
        pos=(0, 50), color='white', height=30, wrapWidth=800
    )
    countdown_text = visual.TextStim(win=win, text='30', pos=(0, -100), color='yellow', height=60)

    break_clock = core.Clock()
    while break_clock.getTime() < 30.0:
        remaining = 30 - int(break_clock.getTime())
        countdown_text.text = str(remaining)
        break_msg.text = (f"BREAK TIME\n\nCompleted {trials_completed} trials.\n"
                          f"Progress: {trials_completed}/{total_trials_in_block} ({block_label})\n\n"
                          f"Break time remaining: {remaining} seconds")
        break_msg.draw()
        countdown_text.draw()
        win.flip()

        if not SIMULATE:
            if event.getKeys(['escape']):
                _save(); core.quit()
        core.wait(0.1)

    # After countdown: show "press space to continue"
    visual.TextStim(
        win=win,
        text=f"BREAK COMPLETE\n\nCompleted {trials_completed}/{total_trials_in_block} trials.\n\nPress SPACE to continue.",
        pos=(0, 0), color='white', height=30, wrapWidth=800
    ).draw()
    win.flip()
    wait_keys(['space', 'escape'])


# ─────────────────────────────────────────────────────────────────────────────
#  QUEST+ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def logit(x):
    """Convert a probability to logit (log-odds) scale."""
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return float(np.log(x / (1 - x)))


def inv_logit(z):
    """Convert a logit value back to probability."""
    return float(1.0 / (1.0 + np.exp(-z)))


def clamp_prop(s):
    """Clamp a prop value to the valid experiment range [0.02, 0.90]."""
    return float(np.clip(s, 0.02, 0.90))


# ─────────────────────────────────────────────────────────────────────────────
#  QUEST+ STAIRCASE
#  Bayesian adaptive procedure to estimate the psychometric function.
#  Uses entropy-minimising stimulus selection.
#  Adapted from: MT Inference.py (SimonKnogler / GitHub)
# ─────────────────────────────────────────────────────────────────────────────

class QuestPlusStaircase:
    def __init__(self, target_type):
        """
        QUEST+ implementation with entropy-based stimulus selection.

        Parameters
        ----------
        target_type : str — 'high' (80% target), 'low' (60% target),
                            or 'neutral' (calibration, ~40% prior mean)
        """
        self.s_grid      = np.linspace(logit(0.05), logit(0.90), 61)
        self.alpha_grid  = np.linspace(logit(0.05), logit(0.90), 61)
        self.beta_grid   = np.geomspace(1.0, 12.0, 25)
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])
        self.gamma = 0.5
        self.target_type = target_type

        if target_type == "high":
            alpha_mu = logit(0.48)
        elif target_type == "low":
            alpha_mu = logit(0.33)
        else:  # 'neutral' — used for calibration
            alpha_mu = logit(0.40)
        alpha_sd = 1.0

        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
        self.prior_alpha /= self.prior_alpha.sum()

        beta_mean, beta_gsd = 2.5, 2.0
        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - np.log(beta_mean)) / np.log(beta_gsd)) ** 2)
        self.prior_beta /= self.prior_beta.sum()

        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

        self.post_alpha  = self.prior_alpha.copy()
        self.post_beta   = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        self.trial_count = 0
        self.responses   = []

    def psychometric(self, s_logit, alpha, beta, lapse):
        """p(correct | s; α, β, λ) = γ + (1 − γ − λ) σ(β [s − α])"""
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid

    def compute_entropy(self, posterior):
        """Shannon entropy of a probability distribution."""
        posterior = posterior + 1e-12
        return -np.sum(posterior * np.log(posterior))

    def select_stimulus_entropy_fast(self):
        """Select next stimulus by expected-entropy minimisation (subsampled grid for speed)."""
        s_grid_subset   = self.s_grid[::3]
        current_entropy = self.compute_entropy(self.post_alpha)
        best_stimulus, max_info_gain = None, -np.inf

        alpha_mean  = np.sum(self.alpha_grid  * self.post_alpha)
        beta_mean   = np.sum(self.beta_grid   * self.post_beta)
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)

        for s_logit in s_grid_subset:
            p_correct   = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue

            post_c = np.zeros_like(self.post_alpha)
            post_i = np.zeros_like(self.post_alpha)
            for idx, alpha in enumerate(self.alpha_grid):
                lc = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                post_c[idx] = self.post_alpha[idx] * lc
                post_i[idx] = self.post_alpha[idx] * (1.0 - lc)
            post_c /= (post_c.sum() + 1e-12)
            post_i /= (post_i.sum() + 1e-12)

            expected_entropy = (p_correct * self.compute_entropy(post_c) +
                                p_incorrect * self.compute_entropy(post_i))
            info_gain = current_entropy - expected_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit

        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid) // 2]
        return clamp_prop(inv_logit(best_stimulus))

    def select_stimulus_entropy(self):
        """Select stimulus using fast entropy approximation."""
        return self.select_stimulus_entropy_fast()

    def update(self, stimulus_prop, correct):
        """Bayesian update of the posterior after observing a response."""
        s_logit  = logit(clamp_prop(stimulus_prop))
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))
        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    w = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    p = self.psychometric(s_logit, alpha, beta, lapse)
                    new_post[i, j, k] = w * (p if correct else 1.0 - p)
        new_post /= (new_post.sum() + 1e-12)
        self.post_alpha  = new_post.sum(axis=(1, 2))
        self.post_beta   = new_post.sum(axis=(0, 2))
        self.post_lambda = new_post.sum(axis=(0, 1))
        self.trial_count += 1
        self.responses.append((s_logit, correct))

    def get_threshold_sd(self):
        """Standard deviation of the alpha (threshold) posterior in logits."""
        mu  = np.sum(self.alpha_grid * self.post_alpha)
        var = np.sum(self.post_alpha * (self.alpha_grid - mu) ** 2)
        return float(np.sqrt(var))

    def get_threshold_mean(self):
        """Mean of the alpha (threshold) posterior in logits."""
        return float(np.sum(self.alpha_grid * self.post_alpha))

    def posterior_summary(self):
        """Summary statistics of alpha, beta, and lambda posteriors."""
        a_mu  = np.sum(self.alpha_grid  * self.post_alpha)
        a_sd  = np.sqrt(np.sum(self.post_alpha  * (self.alpha_grid  - a_mu) ** 2))
        b_mu  = np.sum(self.beta_grid   * self.post_beta)
        b_sd  = np.sqrt(np.sum(self.post_beta   * (self.beta_grid   - b_mu) ** 2))
        l_mu  = np.sum(self.lambda_grid * self.post_lambda)
        l_sd  = np.sqrt(np.sum(self.post_lambda * (self.lambda_grid - l_mu) ** 2))
        return {
            'alpha_mean':  float(a_mu),  'alpha_sd':  float(a_sd),
            'beta_mean':   float(b_mu),  'beta_sd':   float(b_sd),
            'lambda_mean': float(l_mu),  'lambda_sd': float(l_sd),
        }

    def threshold_for_target(self, p_target):
        """Find the prop value that yields p_target% correct under the current posterior."""
        lam_hat = np.sum(self.lambda_grid * self.post_lambda)
        if p_target > 1.0 - lam_hat:
            p_target = min(0.85, 1.0 - lam_hat - 0.02)
        best_diff, best_s = float('inf'), 0.5
        for s_logit in self.s_grid:
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        w = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += w * self.psychometric(s_logit, alpha, beta, lapse)
            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s    = inv_logit(s_logit)
        return clamp_prop(best_s)


# ─────────────────────────────────────────────────────────────────────────────
#  DIFFICULTY LEVEL CALCULATOR
#  After QUEST+ calibration, derive the 2 test difficulty levels from the
#  fitted psychometric function using threshold_for_target():
#    level_1 (hardest):     threshold that yields ~55% correct
#    level_3 (medium-hard): threshold that yields ~85% correct
# ─────────────────────────────────────────────────────────────────────────────

def calculate_difficulty_levels(quest):
    """
    Compute the 2 test-phase difficulty levels from the QUEST+ posterior.

    Parameters
    ----------
    quest : QuestPlusStaircase — fitted calibration object

    Returns
    -------
    dict with keys 'level_1' (~55% correct, hardest) and
                   'level_3' (~85% correct, medium-hard)
    """
    return {
        'level_1': quest.threshold_for_target(0.55),
        'level_3': quest.threshold_for_target(0.85),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TRIAL FUNCTION: run_trial_2shapes
#  This function runs a single experimental trial with 2 shapes.
#
#  Trial timeline:
#    1. Fixation cross (1 second, or random interval if cue_dur_range given)
#    2. Shapes appear; wait for mouse movement to begin
#    3. Motion phase (3 seconds): shapes move, mouse influences target
#    4. Response phase: participant presses A (square) or S (circle)
#    5. Feedback ("Right"/"Wrong") — calibration trials only
#    6. Agency rating (1–7) — test trials only
#
#  Returns a dict of trial results for logging.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum mouse speed (px/frame) to consider the mouse "moving".
# Below this threshold, the target follows its trajectory without mouse mixing.
MOUSE_MOVE_THRESHOLD = 0.5

# Fixed duration of the motion phase in seconds.
MOTION_DURATION = 3.0


def run_trial_2shapes(trial_num, phase, angle_bias, mode, block_num=1,
                      prop_override=None, cue_dur_range=None, motion_dur=None,
                      response_window=None, difficulty_level=None,
                      image_pair=None):
    # image_pair : tuple of two dicts {'filename': str, 'path': str}, or None.
    #              When provided (test phase), images replace the shapes.
    #              When None (calibration phase), the original shapes are used.
    """
    Run a single 2-shape trial.

    Parameters
    ----------
    trial_num       : int   — trial number within the current phase
    phase           : str   — 'calibration' or 'test'
    angle_bias      : int   — rotation applied to mouse input (degrees).
                              0 = no rotation; 90 = perpendicular (not used here)
    mode            : str   — 'staircase' or 'test' (informational only)
    block_num       : int   — current block number (for logging)
    prop_override   : float — self-proportion to use (overrides staircase)
    cue_dur_range   : tuple — (min, max) seconds for fixation duration
    motion_dur      : float — motion phase duration (not used; MOTION_DURATION is fixed)
    response_window : float — response time limit (not used; unlimited response time)
    difficulty_level: str   — level label for logging (e.g. 'level_1')

    Returns
    -------
    dict — trial results including accuracy, RT, ratings, and evidence metrics
    """

    # ── Set the self-proportion for this trial ──────────────────────────────
    # `prop` controls how much the participant's mouse direction influences
    # the target shape. Range: 0.02 (nearly autonomous) to 0.90 (mostly mouse).
    if prop_override is not None:
        prop = float(np.clip(prop_override, 0.02, 0.90))
    else:
        prop = 0.40  # Default fallback (should not normally be used)

    # ── Decide whether this is an image trial or a shape trial ─────────────
    use_images = (image_pair is not None)

    if use_images:
        # Load the two images for this trial as PsychoPy ImageStim objects.
        # Both are scaled to IMAGE_SIZE (40×40 px) to match the shape sizes.
        img_A_info, img_B_info = image_pair
        stim_A = visual.ImageStim(win, image=img_A_info['path'], size=IMAGE_SIZE)
        stim_B = visual.ImageStim(win, image=img_B_info['path'], size=IMAGE_SIZE)
        # Use 'A' and 'B' as internal labels (analogous to 'square' and 'dot')
        stim_left_label  = 'img_A'
        stim_right_label = 'img_B'
    else:
        # Calibration phase: use the original shapes
        stim_A = square
        stim_B = dot
        stim_left_label  = 'square'
        stim_right_label = 'dot'

    # ── Fixation cross ───────────────────────────────────────────────────────
    fix.color = "white"
    if not use_images:
        square.fillColor = square.lineColor = "black"
        dot.fillColor    = dot.lineColor    = "black"

    fix.draw(); win.flip()
    if cue_dur_range is not None:
        core.wait(float(rng.uniform(cue_dur_range[0], cue_dur_range[1])))
    else:
        core.wait(1.0)

    # ── Position stimuli ─────────────────────────────────────────────────────
    # Randomly place stim_A on left or right; stim_B takes the other side.
    left_stim = random.choice(['A', 'B'])
    if left_stim == 'A':
        stim_A.pos = (-OFFSET_X, 0); stim_B.pos = (OFFSET_X, 0)
        left_label  = stim_left_label
        right_label = stim_right_label
    else:
        stim_A.pos = (OFFSET_X, 0);  stim_B.pos = (-OFFSET_X, 0)
        left_label  = stim_right_label
        right_label = stim_left_label

    # Save starting positions so we can reset the images at response time
    start_pos_A = tuple(stim_A.pos)
    start_pos_B = tuple(stim_B.pos)

    stim_A.draw(); stim_B.draw(); win.flip()

    # ── EEG Triggers: Stimulus Onset ────────────────────────────────────────
    trigger_stim_onset = np.nan
    if phase == "test" and difficulty_level:
        try:
            level_idx = int(difficulty_level.split('_')[-1])
            trigger_stim_onset = 10 + level_idx
            send_trigger(trigger_stim_onset)
        except (ValueError, IndexError):
            pass

    # ── Wait for mouse movement to start ────────────────────────────────────
    # The trial doesn't begin until the participant moves the mouse.
    # This ensures the motion phase always starts with active mouse input.
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        stim_A.draw(); stim_B.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE:
            break
        if not SIMULATE and event.getKeys(["escape"]):
            _save(); core.quit()

    # ── Select target and trajectories ──────────────────────────────────────
    # Randomly decide which stimulus is the target (the one the mouse influences).
    # We keep the labels 'square'/'dot' for calibration, 'img_A'/'img_B' for test.
    target = random.choice([stim_left_label, stim_right_label])

    # Get 2 unique trajectory snippets: one for target, one for distractor.
    trajectory_indices = get_trajectory_indices(2)
    target_snippet_idx, distractor_snippet_idx = trajectory_indices[0], trajectory_indices[1]

    # Load snippets and apply consistent smoothing to both.
    # Smoothing is applied to both shapes equally so smoothness is not a cue.
    target_snippet     = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    target_snippet, distractor_snippet = apply_consistent_smoothing(
        target_snippet, distractor_snippet
    )

    # ── Motion phase setup ───────────────────────────────────────────────────
    trial_kinematics = []            # Frame-by-frame data for this trial
    clk   = core.Clock()             # Timer for the motion phase
    frame = 0                        # Frame counter (used to index into snippets)
    vt    = np.zeros(2, np.float32)  # Low-pass filtered velocity for target
    vd    = np.zeros(2, np.float32)  # Low-pass filtered velocity for distractor

    event.clearEvents(eventType='keyboard')

    # Angle bias: 90 means randomly rotate mouse input left or right.
    # In this experiment, angle_bias=0 (no rotation), kept for compatibility.
    applied_angle = angle_bias
    if angle_bias == 90:
        applied_angle = int(rng.choice([90, -90]))

    # ── MOTION PHASE: exactly 3 seconds, no response allowed ────────────────
    # EEG Triggers: Motion Start
    trigger_motion_start = np.nan
    if phase == "test" and difficulty_level:
        try:
            level_idx = int(difficulty_level.split('_')[-1])
            trigger_motion_start = 20 + level_idx
            send_trigger(trigger_motion_start)
        except (ValueError, IndexError):
            pass

    # Two screenshots per phase: frame 1 (starting positions) + frame 30 (~0.5 s in, mid-motion)
    _SCREENSHOT_FRAMES = {1: 'frame001', 30: 'frame030'}  # frames to capture per phase
    while clk.getTime() < MOTION_DURATION:
        # Get current mouse position and compute displacement from last frame
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)

        # Apply angle rotation to mouse input (0 here = no change)
        dx, dy = rotate(dx, dy, applied_angle)

        # Get trajectory velocities for this frame.
        # Snippets are looped if the trial is longer than the snippet.
        target_traj_dx,     target_traj_dy     = target_snippet[frame % len(target_snippet)]
        distractor_traj_dx, distractor_traj_dy = distractor_snippet[frame % len(distractor_snippet)]

        # Scale up trajectory speed so shapes move visibly on screen
        target_traj_dx     *= SPEED_MULTIPLIER
        target_traj_dy     *= SPEED_MULTIPLIER
        distractor_traj_dx *= SPEED_MULTIPLIER
        distractor_traj_dy *= SPEED_MULTIPLIER

        frame += 1

        # Cap mouse speed to prevent extreme jumps from fast mouse movements
        mouse_speed = math.hypot(dx, dy)
        MAX_SPEED = 20.0
        if mouse_speed > MAX_SPEED:
            scale_factor = MAX_SPEED / mouse_speed
            dx *= scale_factor
            dy *= scale_factor
            mouse_speed = MAX_SPEED

        # Only mix mouse input when the mouse is actually moving.
        # When stationary, the target follows its trajectory autonomously.
        mouse_is_moving = mouse_speed > MOUSE_MOVE_THRESHOLD

        # ── Direction mixing (core of the control manipulation) ──────────────
        # Target: blend mouse direction with trajectory direction (prop controls blend)
        # Distractor: always follows its own trajectory, unaffected by mouse
        if mouse_is_moving:
            tdx, tdy = mix_direction_only(dx, dy, target_traj_dx, target_traj_dy, prop)
        else:
            tdx, tdy = target_traj_dx, target_traj_dy

        ddx, ddy = distractor_traj_dx, distractor_traj_dy

        # ── Low-pass filter ──────────────────────────────────────────────────
        # Smooths out frame-to-frame velocity changes for more natural motion.
        # LOWPASS=0.2 means 20% of previous velocity + 80% of new velocity.
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

        # ── Evidence calculation ─────────────────────────────────────────────
        # Measures how aligned the mouse movement is with the target vs distractor.
        # Positive evidence = mouse aligns more with target (correct signal).
        # This is logged per frame for later analysis.
        vm       = np.array([dx, dy], dtype=float)
        vm_speed = np.linalg.norm(vm) + 1e-9  # Avoid division by zero

        vt_disp = np.array(vt, dtype=float)
        vd_disp = np.array(vd, dtype=float)

        # Unit vectors in the direction of each shape's movement
        ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
        ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)

        # Cosine similarity between mouse direction and each shape's direction
        cos_T = np.dot(vm, ut) / vm_speed
        cos_D = np.dot(vm, ud) / vm_speed

        # Evidence = difference in alignment, weighted by mouse speed
        evidence = (cos_T - cos_D) * mouse_speed if mouse_is_moving else 0.0

        # ── Update stimulus positions ────────────────────────────────────────
        # stim_A is the target when target == stim_left_label (set at trial start)
        if target == stim_left_label:
            stim_A.pos = confine(tuple(np.array(stim_A.pos) + vt))
            stim_B.pos = confine(tuple(np.array(stim_B.pos) + vd))
        else:
            stim_B.pos = confine(tuple(np.array(stim_B.pos) + vt))
            stim_A.pos = confine(tuple(np.array(stim_A.pos) + vd))

        # ── Log kinematics for this frame ────────────────────────────────────
        trial_kinematics.append({
            'timestamp':       clk.getTime(),
            'frame':           frame,
            'mouse_x':         x,
            'mouse_y':         y,
            'mouse_speed':     mouse_speed,
            'mouse_is_moving': mouse_is_moving,
            'stim_A_x':        stim_A.pos[0],
            'stim_A_y':        stim_A.pos[1],
            'stim_B_x':        stim_B.pos[0],
            'stim_B_y':        stim_B.pos[1],
            'evidence':        evidence
        })

        # Check for Escape key (no response allowed during motion phase)
        if not SIMULATE:
            if event.getKeys(['escape']):
                _save(); core.quit()

        stim_A.draw(); stim_B.draw(); win.flip()

        # ── Screenshots: frame 1 (start) + frame 30 (~0.5 s, mid-motion) ────────
        if frame in _SCREENSHOT_FRAMES:
            key = f"{phase}_{_SCREENSHOT_FRAMES[frame]}"
            if key not in _screenshots_saved:
                fname = SCREENSHOTS_DIR / f"screenshot_{key}.png"
                win.getMovieFrame(buffer='front')
                win.saveMovieFrames(str(fname))
                _screenshots_saved.add(key)
                print(f"[Screenshot] Saved: {fname}")

    # ── RESPONSE PHASE ───────────────────────────────────────────────────────
    # After the motion phase ends, a response screen appears.
    # Calibration: A = Square, S = Circle
    # Test:        Images are reset to their starting positions.
    #              A key label is shown below the left image, S below the right.
    event.clearEvents(eventType='keyboard')

    # Key mapping: always based on which image is on which side
    key_to_label = {'a': left_label, 's': right_label}

    CHOICE_DURATION = 5.0  # Fixed response window (seconds)

    if use_images:
        # Reset images to their starting positions
        stim_A.pos = start_pos_A
        stim_B.pos = start_pos_B

        # ── Build A / S key labels positioned below each image ────────────────
        # Determine which image is on the left side
        if start_pos_A[0] < 0:   # stim_A is on the left
            left_img_x, right_img_x = start_pos_A[0], start_pos_B[0]
        else:                     # stim_B is on the left
            left_img_x, right_img_x = start_pos_B[0], start_pos_A[0]

        img_y = 0        # vertical centre of images
        img_half_h = IMAGE_SIZE[1] / 2  # half height of image stimulus
        label_y = img_y - img_half_h - 40  # 40 px below the bottom edge

        key_label_A_stim = visual.TextStim(
            win, text="A", pos=(left_img_x, label_y),
            height=36, color='white', bold=True, alignText='center'
        )
        key_label_S_stim = visual.TextStim(
            win, text="S", pos=(right_img_x, label_y),
            height=36, color='white', bold=True, alignText='center'
        )
        choice_question = visual.TextStim(
            win, text="Which image did you control?",
            pos=(0, 380), height=30, color='white', wrapWidth=1200
        )

        def draw_choice_screen():
            stim_A.draw()
            stim_B.draw()
            key_label_A_stim.draw()
            key_label_S_stim.draw()
            choice_question.draw()
            win.flip()
    else:
        # Calibration: plain text prompt
        msg.text = "Which shape did you control?\n\nA = Square          S = Circle"

        def draw_choice_screen():
            msg.draw()
            win.flip()

    # EEG Triggers: Response Screen Onset
    trigger_resp_onset = np.nan
    if phase == "test" and difficulty_level:
        try:
            level_idx = int(difficulty_level.split('_')[-1])
            trigger_resp_onset = 30 + level_idx
            send_trigger(trigger_resp_onset)
        except (ValueError, IndexError):
            pass

    resp_shape = None
    rt_choice  = np.nan
    response_clock = core.Clock()  # Starts now; used for both RT and 5-s window

    draw_choice_screen()
    response_start_time = response_clock.getTime()  # ≈ 0 — used for RT

    if SIMULATE:
        # Simulate an early response, then wait out the remaining window
        core.wait(0.3)
        resp_shape = rng.choice([stim_left_label, stim_right_label])
        rt_choice  = response_clock.getTime()
        remaining  = CHOICE_DURATION - rt_choice
        if remaining > 0:
            core.wait(remaining)
    else:
        # Run until the full 5 s are up
        while response_clock.getTime() < CHOICE_DURATION:
            elapsed = response_clock.getTime()
            draw_choice_screen()
            keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == 'escape':
                    _save(); core.quit()
                elif key in key_to_label and resp_shape is None:
                    resp_shape = key_to_label[key]
                    rt_choice  = elapsed
                    # Response recorded — keep drawing the screen until 5 s done
            core.wait(0.01)

        # Timed out — show "Please answer faster" for 2 s
        if resp_shape is None:
            resp_shape = 'timeout'
            timeout_msg = visual.TextStim(
                win, text="Please answer faster!",
                pos=(0, 0), height=40, color='yellow', bold=True
            )
            timeout_msg.draw(); win.flip()
            core.wait(2.0)

    # Accuracy: 1 if participant identified the correct target, 0 otherwise.
    # Timeouts count as incorrect.
    correct = int(resp_shape == target)

    # EEG Triggers: Response Value
    trigger_resp_val = np.nan
    if phase == "test":
        trigger_resp_val = 41 if correct else 42
        send_trigger(trigger_resp_val)


    # ── FEEDBACK (calibration trials only) ───────────────────────────────────
    # During calibration, participants receive immediate feedback so the
    # staircase can converge to their threshold.
    if phase == "calibration":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # ── AGENCY RATING (test trials only) ─────────────────────────────────────
    # Participants rate how much control they felt over the shape's movement.
    # This is the subjective sense of agency (SoA), rated on a 1–7 scale.
    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
            core.wait(0.5)
        else:
            event.clearEvents(eventType='keyboard')
            msg.text = "How much control did you feel over the shape's movement?"

            # Create the 7-point rating scale as individual text stimuli
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100),
                                (150, -100), (300, -100), (450, -100)]
            scale_labels = ["1\nVery weak", "2\nWeak", "3\nSomewhat weak", "4\nModerate",
                            "5\nSomewhat strong", "6\nStrong", "7\nVery strong"]
            scale_stimuli = [
                visual.TextStim(win, text=label, pos=pos, height=18,
                                color='white', alignText='center')
                for pos, label in zip(scale_positions, scale_labels)
            ]

            rating = None
            while rating is None:
                msg.draw()
                for stim in scale_stimuli:
                    stim.draw()
                win.flip()
                keys = event.getKeys(['1', '2', '3', '4', '5', '6', '7', 'escape'])
                if keys:
                    if 'escape' in keys:
                        _save(); core.quit()
                    else:
                        rating = int(keys[0])
                core.wait(0.01)
            agency_rating = rating

    # ── COMPUTE SUMMARY EVIDENCE METRICS ─────────────────────────────────────
    # Aggregate the per-frame evidence values into summary statistics.
    # These capture how consistently the mouse aligned with the target.
    frame_evidence = [d['evidence'] for d in trial_kinematics]
    mean_evidence  = np.mean(frame_evidence) if frame_evidence else np.nan
    sum_evidence   = np.sum(frame_evidence)  if frame_evidence else np.nan
    var_evidence   = np.var(frame_evidence)  if frame_evidence else np.nan

    # ── ADD TRIAL METADATA TO KINEMATICS ─────────────────────────────────────
    # Tag each frame with trial-level information for later analysis.
    for frame_data in trial_kinematics:
        frame_data.update({
            'trial_num':        trial_num,
            'phase':            phase,
            'n_shapes':         2,
            'target':           target,
            'prop_used':        prop,
            'block_num':        block_num,
            'difficulty_level': difficulty_level
        })
    kinematics_data.extend(trial_kinematics)

    # Return all trial results as a dict for logging by the caller
    return dict(
        n_shapes=2,
        target_snippet_id=target_snippet_idx,
        distractor_snippet_ids=[distractor_snippet_idx],
        phase=phase,
        block_num=block_num,
        angle_bias=angle_bias,
        applied_angle_bias=applied_angle,
        true_shape=target,
        resp_shape=resp_shape,
        accuracy=correct,
        rt_choice=rt_choice,
        agency_rating=agency_rating,
        prop_used=prop,
        early_response=False,
        mean_evidence=mean_evidence,
        sum_evidence=sum_evidence,
        var_evidence=var_evidence,
        difficulty_level=difficulty_level,
        # Image info (NaN for calibration trials, filenames for test trials)
        img_A_name=img_A_info['filename'] if use_images else np.nan,
        img_B_name=img_B_info['filename'] if use_images else np.nan,
        # Trigger values for logging
        trigger_stim_onset=trigger_stim_onset,
        trigger_motion_start=trigger_motion_start,
        trigger_resp_onset=trigger_resp_onset,
        trigger_resp_val=trigger_resp_val
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION PHASE RUNNER (QUEST+)
#  Runs a QUEST+ Bayesian adaptive staircase to estimate the participant's
#  psychometric function (and hence their 75% accuracy threshold).
#
#  Adaptive stopping (same as MT Inference.py):
#    • Run at least CHECK_CALIBRATION_TRIALS trials
#    • Stop when posterior SD < 0.20 AND trials ≥ min
#    • Hard cap at CHECK_CALIBRATION_TRIALS + 20 trials
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_quest(num_trials, angle_bias=0, block_num=0):
    """
    Run the QUEST+ calibration phase.

    Parameters
    ----------
    num_trials  : int — minimum number of calibration trials (hard cap = +20)
    angle_bias  : int — rotation applied to mouse input (0 = none)
    block_num   : int — block number for logging (0 = calibration)

    Returns
    -------
    quest     : QuestPlusStaircase — fitted object (use for difficulty levels)
    threshold : float — estimated 75% accuracy prop
    """
    global global_trial_counter

    quest        = QuestPlusStaircase("neutral")
    min_trials   = num_trials
    max_trials   = num_trials + 20
    sd_threshold = 0.20

    print(f"Starting QUEST+ calibration (min={min_trials}, max={max_trials} trials, "
          f"SD threshold={sd_threshold})")

    trial_num = 0
    while trial_num < max_trials:
        trial_num            += 1
        global_trial_counter += 1

        s_candidate = quest.select_stimulus_entropy()

        res = run_trial_2shapes(
            trial_num, "calibration", angle_bias=angle_bias, mode="staircase",
            prop_override=s_candidate, cue_dur_range=(0.5, 0.8),
            difficulty_level="calibration", block_num=block_num
        )

        # Update QUEST only for valid (non-timeout) responses
        if res.get('resp_shape') != 'timeout':
            quest.update(s_candidate, int(res.get('accuracy', 0)))

        # Compute QUEST summary every 10 trials (expensive); otherwise fast SD
        if trial_num % 10 == 0 or trial_num < 10:
            summ           = quest.posterior_summary()
            quest_alpha_sd = summ['alpha_sd']
        else:
            quest_alpha_sd = quest.get_threshold_sd()
            summ           = None

        # ── Log trial data ────────────────────────────────────────────────────
        thisExp.addData('trial_num',              global_trial_counter)
        thisExp.addData('participant',             expInfo['participant'])
        thisExp.addData('session',                 expInfo['session'])
        thisExp.addData('phase',                   'calibration')
        thisExp.addData('n_shapes',                2)
        thisExp.addData('block_num',               block_num)
        thisExp.addData('staircase_trial',         trial_num)
        thisExp.addData('prop_used',               s_candidate)
        thisExp.addData('stimulus_logit',          logit(s_candidate))
        thisExp.addData('accuracy',                res.get('accuracy', 0))
        thisExp.addData('is_timeout',              res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice',               res.get('rt_choice', np.nan))
        thisExp.addData('early_response',          res.get('early_response', False))
        thisExp.addData('true_shape',              res.get('true_shape', ''))
        thisExp.addData('resp_shape',              res.get('resp_shape', ''))
        thisExp.addData('angle_bias',              angle_bias)
        thisExp.addData('applied_angle_bias',      res.get('applied_angle_bias', angle_bias))
        thisExp.addData('quest_alpha_sd',          quest_alpha_sd)
        thisExp.addData('mean_evidence',           res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence',            res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence',            res.get('var_evidence', np.nan))
        thisExp.addData('target_snippet_id',       res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids',  str(res.get('distractor_snippet_ids', [])))

        if summ is not None:
            thisExp.addData('quest_alpha_mean',   summ['alpha_mean'])
            thisExp.addData('quest_beta_mean',    summ['beta_mean'])
            thisExp.addData('quest_beta_sd',      summ['beta_sd'])
            thisExp.addData('quest_lambda_mean',  summ['lambda_mean'])
            thisExp.addData('quest_lambda_sd',    summ['lambda_sd'])
        else:
            thisExp.addData('quest_alpha_mean',   np.nan)
            thisExp.addData('quest_beta_mean',    np.nan)
            thisExp.addData('quest_beta_sd',      np.nan)
            thisExp.addData('quest_lambda_mean',  np.nan)
            thisExp.addData('quest_lambda_sd',    np.nan)

        thisExp.nextEntry()

        if trial_num % 10 == 0:
            print(f"  Trial {trial_num}: prop={s_candidate:.3f}, "
                  f"alpha_sd={quest_alpha_sd:.4f}")

        # Mid-way break
        halfway = min_trials // 2
        if trial_num == halfway:
            msg.text = (f"Short Break\n\n"
                        f"You have completed {trial_num} practice trials.\n\n"
                        f"Take a moment to rest if needed.\n\n"
                        f"Press SPACE to continue.")
            msg.draw(); win.flip()
            wait_keys(['space', 'escape'])

        # Adaptive stopping: converge after min_trials if posterior SD is small
        if trial_num >= min_trials and quest_alpha_sd < sd_threshold:
            print(f"  QUEST+ converged after {trial_num} trials "
                  f"(alpha_sd={quest_alpha_sd:.4f} < {sd_threshold})")
            break

    threshold = quest.threshold_for_target(0.75)
    summary   = quest.posterior_summary()
    print(f"\nCalibration complete:")
    print(f"  Trials: {trial_num}, alpha_sd: {summary['alpha_sd']:.4f}")
    print(f"  QUEST+ threshold (75% correct): {threshold:.3f}")

    return quest, threshold


# ─────────────────────────────────────────────────────────────────────────────
#  TEST BLOCK RUNNER
#  Runs all trials for a single difficulty level (one block).
#  Unlike the original MT Inference.py (which shuffled all 4 levels together),
#  here each block contains only one control level — enabling analysis of
#  within-block history effects at a fixed difficulty.
# ─────────────────────────────────────────────────────────────────────────────

def run_test_block_for_level(threshold_75, level_name, prop_value,
                              num_trials, block_num, angle_bias=0):
    """
    Run a test block where all trials use the same difficulty level.

    Parameters
    ----------
    threshold_75 : float — calibrated 75% threshold (logged for reference)
    level_name   : str   — difficulty level label (e.g. 'level_1')
    prop_value   : float — self-proportion for all trials in this block
    num_trials   : int   — number of trials in this block
    block_num    : int   — block number (1–4) for logging
    angle_bias   : int   — rotation applied to mouse input (0 = none)
    """
    global global_trial_counter, pair_index

    print(f"\nTest Block {block_num}: {level_name} (prop={prop_value:.3f}, "
          f"{num_trials} trials)")

    for trial_num in range(1, num_trials + 1):
        global_trial_counter += 1

        # Fetch the next unique image pair for this trial.
        # pair_index is a global counter so pairs never repeat across blocks.
        current_pair = image_pairs[pair_index % len(image_pairs)]
        pair_index += 1

        # Run one trial at the fixed prop_value for this block, using images
        res = run_trial_2shapes(
            trial_num, "test", angle_bias=angle_bias, mode="test",
            prop_override=prop_value, cue_dur_range=(0.5, 0.8),
            difficulty_level=level_name, block_num=block_num,
            image_pair=current_pair
        )

        # ── Log trial data ────────────────────────────────────────────────────
        thisExp.addData('trial_num',              global_trial_counter)
        thisExp.addData('participant',             expInfo['participant'])
        thisExp.addData('session',                 expInfo['session'])
        thisExp.addData('phase',                   'test')
        thisExp.addData('n_shapes',                2)
        thisExp.addData('block_num',               block_num)
        thisExp.addData('difficulty_level',        level_name)
        thisExp.addData('prop_used',               prop_value)
        thisExp.addData('threshold_75',            threshold_75)
        thisExp.addData('accuracy',                res.get('accuracy', 0))
        thisExp.addData('is_timeout',              res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice',               res.get('rt_choice', np.nan))
        thisExp.addData('agency_rating',           res.get('agency_rating', np.nan))
        thisExp.addData('early_response',          res.get('early_response', False))
        thisExp.addData('true_shape',              res.get('true_shape', ''))
        thisExp.addData('resp_shape',              res.get('resp_shape', ''))
        thisExp.addData('angle_bias',              angle_bias)
        thisExp.addData('applied_angle_bias',      res.get('applied_angle_bias', angle_bias))
        thisExp.addData('mean_evidence',           res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence',            res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence',            res.get('var_evidence', np.nan))
        thisExp.addData('target_snippet_id',       res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids',  str(res.get('distractor_snippet_ids', [])))
        thisExp.addData('img_A_name',              res.get('img_A_name', np.nan))
        thisExp.addData('img_B_name',              res.get('img_B_name', np.nan))
        # Log EEG triggers
        thisExp.addData('trigger_stim_onset',      res.get('trigger_stim_onset', np.nan))
        thisExp.addData('trigger_motion_start',    res.get('trigger_motion_start', np.nan))
        thisExp.addData('trigger_resp_onset',      res.get('trigger_resp_onset', np.nan))
        thisExp.addData('trigger_resp_val',        res.get('trigger_resp_val', np.nan))
        thisExp.nextEntry()

        # Offer a break every 50 trials within a block
        if trial_num % 50 == 0 and trial_num < num_trials:
            show_break_screen(trial_num, num_trials, f"Block {block_num} ({level_name})")


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY TEST
#  After all 4 test blocks, participants complete a yes/no recognition test.
#  400 images are shown one at a time (200 seen during the experiment +
#  200 unseen foils). For each image, participants press:
#    A = Yes (I saw this during the experiment)
#    S = No  (I did not see this during the experiment)
#  Logged per item: filename, seen/unseen ground truth, response, accuracy, RT.
# ─────────────────────────────────────────────────────────────────────────────

def run_memory_test(seen_images, foil_images_list):
    """
    Run the yes/no recognition memory test.

    Parameters
    ----------
    seen_images      : list of dicts {'filename', 'path'} — the 200 images
                       shown during the experiment (ground truth = 'seen')
    foil_images_list : list of dicts {'filename', 'path'} — 200 new images
                       never shown during the experiment (ground truth = 'unseen')

    Each item is shown one at a time. Participant presses:
      A = Yes (seen before)   |   S = No (not seen before)
    No time limit per item.

    Logged per item (appended to the main CSV via thisExp):
      mem_item_num    : item number within the memory test (1–400)
      mem_filename    : image filename stem
      mem_ground_truth: 'seen' or 'unseen'
      mem_response    : 'yes' or 'no'
      mem_accuracy    : 1 if correct, 0 if incorrect
      mem_rt          : response time in seconds
    """
    global global_trial_counter

    # Build the full 400-item list: tag each image with its ground truth
    mem_items = (
        [{'filename': d['filename'], 'path': d['path'], 'ground_truth': 'seen'}
         for d in seen_images] +
        [{'filename': d['filename'], 'path': d['path'], 'ground_truth': 'unseen'}
         for d in foil_images_list]
    )

    # Shuffle with a participant-specific seed for reproducibility
    _mem_rng = random.Random(
        int(hashlib.md5((expInfo['participant'] + '_mem').encode()).hexdigest(), 16) % (2**32)
    )
    _mem_rng.shuffle(mem_items)

    print(f"\nMemory test: {len(mem_items)} items (200 seen + 200 unseen)")

    # Pre-create a large ImageStim; we'll update its image each trial
    mem_img_stim = visual.ImageStim(win, size=(300, 300))

    # Question text sits below the image
    mem_question = visual.TextStim(
        win,
        text="Have you seen this object during the experiment before?",
        pos=(0, -200), color='white', height=26, wrapWidth=900
    )
    # Key labels: A = Old (left), S = New (right)
    mem_key_A = visual.TextStim(
        win, text="A\nOld", pos=(-80, -270),
        height=28, color='white', bold=True, alignText='center'
    )
    mem_key_S = visual.TextStim(
        win, text="S\nNew", pos=(80, -270),
        height=28, color='white', bold=True, alignText='center'
    )

    # Fixation cross for inter-trial interval
    mem_fix_h = visual.TextStim(win, text='─────', pos=(0, 0), color='white',
                                height=30, bold=True)
    mem_fix_v = visual.TextStim(win, text='|', pos=(0, 0), color='white',
                                height=50, bold=True)

    for item_num, item in enumerate(mem_items, start=1):
        global_trial_counter += 1

        # Load image for this item
        mem_img_stim.image = item['path']
        mem_img_stim.pos   = (0, 80)   # Slightly above centre; prompt sits below

        # Fixation cross — random duration 0.5–0.8 s
        fix_dur = random.uniform(0.5, 0.8)
        mem_fix_h.draw(); mem_fix_v.draw(); win.flip()
        core.wait(fix_dur)

        # Draw image + key labels and start timing
        event.clearEvents(eventType='keyboard')
        mem_img_stim.draw()
        mem_question.draw()
        mem_key_A.draw()
        mem_key_S.draw()
        win.flip()
        item_onset = core.getTime()

        # EEG Triggers: Recognition Stimulus Onset
        mem_trigger_onset = 51 if item['ground_truth'] == 'seen' else 52
        send_trigger(mem_trigger_onset)

        # Screenshot: capture the very first memory test item
        if item_num == 1 and 'memory_test' not in _screenshots_saved:
            fname = SCREENSHOTS_DIR / "screenshot_memory_test.png"
            win.getMovieFrame(buffer='front')
            win.saveMovieFrames(str(fname))
            _screenshots_saved.add('memory_test')
            print(f"[Screenshot] Saved: {fname}")

        # Wait for A or S (no time limit)
        mem_response = None
        mem_rt       = np.nan

        if SIMULATE:
            sim_rt = random.uniform(1.0, 5.0)
            core.wait(sim_rt)
            mem_response = _mem_rng.choice(['yes', 'no'])
            mem_rt       = sim_rt
        else:
            while mem_response is None:
                keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
                if keys:
                    key, key_time = keys[0]
                    if key == 'escape':
                        _save(); core.quit()
                    elif key == 'a':
                        mem_response = 'yes'
                        mem_rt       = key_time - item_onset
                    elif key == 's':
                        mem_response = 'no'
                        mem_rt       = key_time - item_onset
                core.wait(0.01)

        # Accuracy: correct if 'yes' for seen, 'no' for unseen
        ground_truth  = item['ground_truth']
        mem_correct   = int(
            (mem_response == 'yes' and ground_truth == 'seen') or
            (mem_response == 'no'  and ground_truth == 'unseen')
        )

        # EEG Triggers: Recognition Participant Response
        # 61 = Correct (Hit or CR), 62 = Incorrect (Miss or FA)
        mem_trigger_resp = 61 if mem_correct else 62
        send_trigger(mem_trigger_resp)

        win.flip()  # clear screen; next trial's fixation cross follows immediately

        # Log to main CSV
        thisExp.addData('trial_num',        global_trial_counter)
        thisExp.addData('participant',      expInfo['participant'])
        thisExp.addData('session',          expInfo['session'])
        thisExp.addData('phase',            'memory_test')
        thisExp.addData('mem_item_num',     item_num)
        thisExp.addData('mem_filename',     item['filename'])
        thisExp.addData('mem_ground_truth', ground_truth)
        thisExp.addData('mem_response',     mem_response)
        thisExp.addData('mem_accuracy',     mem_correct)
        thisExp.addData('mem_rt',           mem_rt)
        # Log EEG triggers
        thisExp.addData('mem_trigger_onset', mem_trigger_onset)
        thisExp.addData('mem_trigger_resp',  mem_trigger_resp)
        thisExp.nextEntry()

        # Optional mid-test break every 100 items
        if item_num % 100 == 0 and item_num < len(mem_items):
            msg.text = (f"Memory test: {item_num} of {len(mem_items)} done.\n\n"
                        f"Take a short break if needed.\n\n"
                        f"Press SPACE to continue.")
            msg.draw(); win.flip()
            wait_keys(['space', 'escape'])

    print(f"Memory test complete. {len(mem_items)} items judged.")


# ─────────────────────────────────────────────────────────────────────────────
#  INSTRUCTION SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def show_initial_instructions():
    """Display the welcome and general task instructions."""
    instructions = [
        """Dear participant, welcome to the study!

This task involves moving objects on the screen, and figuring out which one is under your control. The experiment consists of 6 blocks with 20 trials each.

In each trial, you will see two objects on the screen.
You will use the touchpad to move the objects. Only one of them will be controlled by your touchpad movements. After a certain duration, you will be asked to press [A] or [S] to report which object you were controlling. Please try to respond accurately.

After each trial, you will be asked to indicate your feeling of control (on a scale of 1 to 7, where 1 is no control and 7 is full control). Please answer by pressing the corresponding number on the keyboard.

Please respond as accurately as possible throughout the whole experiment. If unsure, make your best guess.

Please feel free to ask any questions to the experimenter now.

Before the main experiment, you will practice the task with simple shapes. During the practice block, you will receive feedback on whether your response was correct or incorrect. You won't be asked to indicate your feeling of control.

Please press SPACE to start the practice block."""
    ]

    for instruction in instructions:
        msg.text = instruction
        msg.draw(); win.flip()
        keys = wait_keys(['space', 'escape'])
        if 'escape' in keys:
            _save(); core.quit()



# def show_calibration_instructions():
#     """Display instructions for the calibration (practice) phase."""
#     msg.text = """PRACTICE PHASE

# In this phase, you will practice the task.
# After each trial, you will receive feedback: "Right" or "Wrong".

# This helps us calibrate the task difficulty to your individual level.

# Response keys:
#   A = Square
#   S = Circle

# Press SPACE to start the practice..."""
#     msg.draw(); win.flip()
#     wait_keys(['space', 'escape'])


def show_test_phase_instructions():
    """
    Display transition instructions shown ONCE after calibration,
    before the first miniblock of test trials.
    """
    msg.text = """Well done — the practice block is now complete!

You will now see pairs of images on screen.
Use the touchpad to move the images and decide which image was the one you controlled.

Indicate your decision by pressing [A] or [S].
After each trial, rate your feeling of control on a scale of 1 to 7
(1 = no control, 7 = full control) by pressing the corresponding number.

No feedback will be shown during the main experiment.

Please press SPACE to start."""
    msg.draw(); win.flip()
    wait_keys(['space', 'escape'])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Show initial instructions ────────────────────────────────────────
show_initial_instructions()

# ── Step 2: Calibration phase ────────────────────────────────────────────────────────────────
# Run QUEST+ to estimate the participant's psychometric function.
msg.text = ("Practice Block\n\n"
            "In this block, you will practice the task.\n"
            "You will receive feedback after each response.\n\n"
            "Press SPACE to start.")
msg.draw(); win.flip(); wait_keys()

quest, threshold_75 = run_calibration_quest(
    num_trials=CHECK_CALIBRATION_TRIALS,
    angle_bias=0,
    block_num=0  # Block 0 = calibration (test blocks are 1-6)
)

# Save QUEST+ results to expInfo for reference
expInfo['quest_threshold_75'] = threshold_75
expInfo['quest_alpha_sd']     = quest.get_threshold_sd()

# Show calibration completion screen
msg.text = """Practice complete!

You can take a short break now.

Please press SPACE to continue to the main experiment."""
msg.draw(); win.flip(); wait_keys()

# ── Step 3: Calculate difficulty levels ────────────────────────────────────────────────
# Derive level_1 (~55% correct) and level_3 (~85% correct) via QUEST+ posterior.
levels = calculate_difficulty_levels(quest)

print(f"\nDifficulty levels (QUEST+, threshold={threshold_75:.3f}):")
for name, val in levels.items():
    print(f"  {name}: prop={val:.3f}")

# ── Step 4: Determine miniblock order by participant parity ───────────────────
# Odd participant number  → starts with level_1: [L1, L3, L1, L3, L1, L3]
# Even participant number → starts with level_3: [L3, L1, L3, L1, L3, L1]
try:
    participant_num = int(expInfo["participant"])
except ValueError:
    # If participant ID is not a plain number, derive one from its hash
    participant_num = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16)

starts_with_level1 = (participant_num % 2 == 1)   # odd → level_1 first
level_A = 'level_1' if starts_with_level1 else 'level_3'
level_B = 'level_3' if starts_with_level1 else 'level_1'

miniblock_sequence = [level_A, level_B, level_A, level_B, level_A, level_B]

expInfo['miniblock_order'] = str(miniblock_sequence)
expInfo['starts_with']     = level_A

print(f"\nParticipant {expInfo['participant']} (num={participant_num}, "
      f"{'odd' if starts_with_level1 else 'even'}): "
      f"Miniblock order = {miniblock_sequence}")

# ── Step 5: Show test phase instructions (once, after calibration) ────────────
show_test_phase_instructions()

# ── Step 6: Run 6 miniblocks ──────────────────────────────────────────────────
# Miniblocks alternate between level_1 and level_3 (order set by parity above).
# Each miniblock contains CHECK_TEST_TRIALS_PER_LEVEL trials (20 in full mode).
TOTAL_MINIBLOCKS = 6

for mb_idx, level_name in enumerate(miniblock_sequence):
    miniblock_num = mb_idx + 1
    prop_value    = levels[level_name]

    # Run all trials for this miniblock
    run_test_block_for_level(
        threshold_75=threshold_75,
        level_name=level_name,
        prop_value=prop_value,
        num_trials=CHECK_TEST_TRIALS_PER_LEVEL,
        block_num=miniblock_num,
        angle_bias=0
    )

    # Show a break screen between miniblocks (not after the last one)
    if miniblock_num < TOTAL_MINIBLOCKS:
        msg.text = (f"Miniblock {miniblock_num} of {TOTAL_MINIBLOCKS} complete!\n\n"
                    f"You can take a short break.\n\n"
                    f"Press SPACE when you are ready to continue.")
        msg.draw(); win.flip(); wait_keys()


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY TEST PHASE
# ─────────────────────────────────────────────────────────────────────────────

# Show transition instructions before the memory test
msg.text = """

The main experiment is now complete. One last part! 

You will now see a series of objects on the screen, one at a time.
For each object, decide whether you saw it during the experiment (old) or not (new). If you have seen the object before (old), press [A]. If you have not seen the object before (new), press [S].

Please try to respond as accurately as possible. If unsure, make your best guess.

Please press SPACE to start."""
msg.draw(); win.flip()
wait_keys(['space', 'escape'])

# Run the memory test (N_IMAGES seen + N_IMAGES foils, shuffled)
run_memory_test(sampled_test_images, foil_images)


# ─────────────────────────────────────────────────────────────────────────────
#  END OF EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

final_used = len(used_trajectory_indices)
print(f"\nExperiment complete!")
print(f"  Total trajectories used: {final_used}")
print(f"  QUEST+ threshold (75%): {threshold_75:.3f}")
print(f"  Miniblock order: {miniblock_sequence}")

msg.text = f"""Thank you for participating!

Your data have been recorded. 

Press SPACE to exit."""
msg.draw(); win.flip(); wait_keys()

# Save all data and close
_save()
win.close()
core.quit()
