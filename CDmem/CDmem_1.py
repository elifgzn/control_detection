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
     A 3-up-1-down staircase procedure finds each participant's individual
     75% accuracy threshold — the "self-proportion" (prop) at which they
     can correctly identify the controlled shape ~75% of the time.

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
from itertools import permutations  # Used for counterbalancing block order


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

# ─────────────────────────────────────────────────────────────────────────────
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
    expInfo = {"participant": "", "session": "001", "simulate": False, "check_mode": True}
    dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "simulate", "check_mode"], title=expName)
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
    CHECK_CALIBRATION_TRIALS = 20    # Minimum trials for staircase
    CHECK_TEST_TRIALS_PER_LEVEL = 5  # Trials per difficulty level per block
else:
    CHECK_CALIBRATION_TRIALS = 60    # Full calibration
    CHECK_TEST_TRIALS_PER_LEVEL = 25 # Full test (4 levels × 25 = 100 trials per block)

if CHECK_MODE:
    print("=" * 60)
    print("** CHECK MODE ENABLED — Running minimal trials **")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/level × 4 levels × 4 blocks")
    print("=" * 60)
else:
    print("Running FULL experiment mode")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/level × 4 levels × 4 blocks")


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
SPEED_MULTIPLIER = 1.5    # Multiply trajectory velocities to make shapes move faster

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGINE DATASET CONSTANTS
#  200 familiar images are sampled from the IMAGINE dataset and used as
#  moving stimuli in the TEST phase only. Calibration keeps the shapes.
# ─────────────────────────────────────────────────────────────────────────────

IMAGINE_DIR  = pathlib.Path(r"C:\Users\elifg\Desktop\PHD\stimuli_datasets\IMAGINE\familiar\png")
IMAGE_SIZE   = (40, 40)   # Display size in pixels — matches the 40×40 px shapes
N_IMAGES     = 200        # Number of images to sample from the full set
IMAGE_SEED   = 42         # Fixed seed for reproducible sampling across runs
IMAGE_LOG    = pathlib.Path(__file__).parent / "image_stimuli_log.json"


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE SAMPLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_and_log_images(img_dir, n, seed):
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


def make_image_pairs(images):
    """
    Pair `images` into non-overlapping pairs: (img[0], img[1]), (img[2], img[3]), ...

    Each image appears in exactly ONE pair — no image is shared across pairs.
    With N_IMAGES=200 this produces 100 pairs, one per test trial.

    Returns a list of 100 tuples: [(dict_A, dict_B), ...]
    """
    assert len(images) % 2 == 0, "Need an even number of images to form pairs"
    pairs = [(images[i], images[i + 1]) for i in range(0, len(images), 2)]
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
sampled_images = sample_and_log_images(IMAGINE_DIR, N_IMAGES, IMAGE_SEED)

# Shuffle the sampled list with a participant-specific seed so pair order
# differs across participants but is reproducible for the same participant.
_img_rng = random.Random(int(hashlib.md5(expInfo['participant'].encode()).hexdigest(), 16) % (2**32))
_img_rng.shuffle(sampled_images)

# Build 100 non-overlapping pairs from the shuffled list
image_pairs = make_image_pairs(sampled_images)

# Global index into image_pairs — incremented by each test trial
pair_index = 0

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
#  3-UP-1-DOWN STAIRCASE
#  This adaptive procedure finds the "self-proportion" (prop) at which the
#  participant achieves approximately 75% accuracy.
#
#  Rules:
#    - 3 consecutive correct responses → decrease prop (make task harder)
#    - 1 incorrect response            → increase prop (make task easier)
#
#  Mathematically, this converges to ~79.4% correct, which is close to 75%.
#  The threshold estimate is the mean prop over the last 8 trials.
# ─────────────────────────────────────────────────────────────────────────────

class ThreeUpOneDownStaircase:
    """
    3-up-1-down adaptive staircase for estimating the 75% accuracy threshold.

    Attributes
    ----------
    current_prop        : float — current self-proportion value
    step_size           : float — how much to change prop on each step
    consecutive_correct : int   — count of consecutive correct responses
    reversals           : int   — number of direction reversals (used to
                                  assess convergence)
    prop_history        : list  — all prop values presented
    response_history    : list  — all responses (1=correct, 0=incorrect)
    reversal_props      : list  — prop values at each reversal point
    """

    def __init__(self, start_prop=0.25, step_size=0.05, min_prop=0.05, max_prop=0.90):
        """
        Parameters
        ----------
        start_prop : float — initial prop (default 0.25 = harder start)
        step_size  : float — step size for prop adjustments
        min_prop   : float — minimum allowed prop
        max_prop   : float — maximum allowed prop
        """
        self.current_prop        = start_prop
        self.step_size           = step_size
        self.min_prop            = min_prop
        self.max_prop            = max_prop
        self.consecutive_correct = 0
        self.trial_count         = 0
        self.reversals           = 0
        self.last_direction      = None   # 'up' or 'down'
        self.prop_history        = []
        self.response_history    = []
        self.reversal_props      = []

    def get_current_prop(self):
        """Return the prop value to use for the next trial."""
        return self.current_prop

    def update(self, correct):
        """
        Update the staircase based on the participant's response.

        Parameters
        ----------
        correct : int or bool — 1 if correct, 0 if incorrect
        """
        correct = int(correct)
        self.trial_count += 1
        self.prop_history.append(self.current_prop)
        self.response_history.append(correct)

        if correct:
            self.consecutive_correct += 1
            if self.consecutive_correct >= 3:
                # 3 in a row correct → decrease prop (harder)
                self.consecutive_correct = 0
                new_prop = max(self.min_prop, self.current_prop - self.step_size)
                if self.last_direction == 'up':
                    self.reversals += 1
                    self.reversal_props.append(self.current_prop)
                self.current_prop   = new_prop
                self.last_direction = 'down'
        else:
            # 1 incorrect → increase prop (easier)
            self.consecutive_correct = 0
            new_prop = min(self.max_prop, self.current_prop + self.step_size)
            if self.last_direction == 'down':
                self.reversals += 1
                self.reversal_props.append(self.current_prop)
            self.current_prop   = new_prop
            self.last_direction = 'up'

    def get_threshold_estimate(self, n_last_trials=8):
        """
        Estimate the threshold as the mean prop over the last n trials.

        Using the last few trials (rather than all trials) gives a more
        stable estimate because the staircase has converged by then.
        """
        if len(self.prop_history) < n_last_trials:
            return np.mean(self.prop_history) if self.prop_history else self.current_prop
        return np.mean(self.prop_history[-n_last_trials:])

    def get_summary(self):
        """Return a dict of summary statistics for logging."""
        return {
            'trial_count':       self.trial_count,
            'reversals':         self.reversals,
            'final_prop':        self.current_prop,
            'threshold_estimate': self.get_threshold_estimate(),
            'mean_accuracy':     np.mean(self.response_history) if self.response_history else 0,
            'prop_history':      self.prop_history.copy(),
            'response_history':  self.response_history.copy()
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DIFFICULTY LEVEL CALCULATOR
#  After calibration, we derive 4 test difficulty levels around the threshold.
#
#  Level 1 (hardest): threshold − 2 × step  → participant correct ~55% of the time
#  Level 2:           threshold − 1 × step  → ~65%
#  Level 3:           threshold + 1 × step  → ~85%
#  Level 4 (easiest): threshold + 2 × step  → ~95%
#
#  All values are clipped to [min_prop, max_prop] to stay within valid range.
# ─────────────────────────────────────────────────────────────────────────────

def calculate_difficulty_levels(threshold_75, step_size=0.08, min_prop=0.05, max_prop=0.90):
    """
    Compute 4 difficulty levels symmetrically around the 75% threshold.

    Parameters
    ----------
    threshold_75 : float — calibrated prop for ~75% accuracy
    step_size    : float — spacing between levels (default 0.08)
    min_prop     : float — minimum allowed prop
    max_prop     : float — maximum allowed prop

    Returns
    -------
    dict with keys 'level_1' (hardest) to 'level_4' (easiest)
    """
    return {
        'level_1': np.clip(threshold_75 - 2 * step_size, min_prop, max_prop),
        'level_2': np.clip(threshold_75 - 1 * step_size, min_prop, max_prop),
        'level_3': np.clip(threshold_75 + 1 * step_size, min_prop, max_prop),
        'level_4': np.clip(threshold_75 + 2 * step_size, min_prop, max_prop),
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
#    5. Confidence rating (1–4) — test trials only
#    6. Feedback ("Right"/"Wrong") — calibration trials only
#    7. Agency rating (1–7) — test trials only
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
    stim_A.draw(); stim_B.draw(); win.flip()

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

    # ── RESPONSE PHASE ───────────────────────────────────────────────────────
    # After the motion phase ends, a response screen appears.
    # Calibration: A = Square, S = Circle
    # Test:        A = Left image, S = Right image
    event.clearEvents(eventType='keyboard')
    response_start_time = core.getTime()

    if use_images:
        # Show the two images side-by-side as a reminder during the response
        stim_A.draw(); stim_B.draw()
        msg.text = "Which image did you control?\n\nA = Left image          S = Right image"
        msg.draw(); win.flip()
        # Map keys to the label of whichever image is on that side
        key_to_label = {'a': left_label, 's': right_label}
    else:
        msg.text = "Which shape did you control?\n\nA = Square          S = Circle"
        msg.draw(); win.flip()
        key_to_label = {'a': 'square', 's': 'dot'}

    resp_shape = None
    rt_choice  = np.nan

    if SIMULATE:
        core.wait(0.3)
        resp_shape = rng.choice([stim_left_label, stim_right_label])
        rt_choice  = 0.3
    else:
        while resp_shape is None:
            keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == "escape":
                    _save(); core.quit()
                elif key in key_to_label:
                    resp_shape = key_to_label[key]
                    rt_choice  = key_time - response_start_time
            core.wait(0.01)

    # Accuracy: 1 if participant identified the correct target, 0 otherwise
    correct = int(resp_shape == target)

    # ── CONFIDENCE RATING (test trials only) ─────────────────────────────────
    # Participants rate how confident they are in their shape choice (1–4 scale).
    confidence_rating = np.nan
    if phase == "test":
        if SIMULATE:
            confidence_rating = float(rng.integers(1, 5))
        else:
            msg.text = ("How confident are you in your choice?\n\n"
                        "1 = Not at all confident\n"
                        "2 = Slightly confident\n"
                        "3 = Moderately confident\n"
                        "4 = Very confident")
            msg.draw(); win.flip()
            conf_key = wait_keys(['1', '2', '3', '4', 'escape'])[0]
            if conf_key == "escape":
                _save(); core.quit()
            confidence_rating = int(conf_key)
            core.wait(0.2)

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
            core.wait(0.2)

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
        confidence_rating=confidence_rating,
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
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION PHASE RUNNER
#  Runs the 3-up-1-down staircase to find the participant's 75% threshold.
#  Feedback is given after each trial so the staircase can adapt.
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_staircase(num_trials, angle_bias=0, block_num=0):
    """
    Run the 3-up-1-down staircase calibration phase.

    Parameters
    ----------
    num_trials  : int — number of calibration trials to run
    angle_bias  : int — rotation applied to mouse input (0 = none)
    block_num   : int — block number for logging (0 = calibration)

    Returns
    -------
    staircase : ThreeUpOneDownStaircase — completed staircase object
    threshold : float — estimated 75% accuracy threshold (mean of last 8 trials)
    """
    global global_trial_counter

    # Initialize the staircase starting at prop=0.25 (harder start)
    staircase = ThreeUpOneDownStaircase(
        start_prop=0.25,
        step_size=0.05,
        min_prop=0.05,
        max_prop=0.90
    )

    print(f"Starting 3-up-1-down staircase calibration ({num_trials} trials)")
    print(f"  Starting prop: {staircase.current_prop:.2f}, step: {staircase.step_size:.2f}")

    for trial_num in range(1, num_trials + 1):
        global_trial_counter += 1
        current_prop = staircase.get_current_prop()

        # Run one trial at the current staircase prop value
        res = run_trial_2shapes(
            trial_num, "calibration", angle_bias=angle_bias, mode="staircase",
            prop_override=current_prop, cue_dur_range=(0.5, 0.8),
            difficulty_level="calibration", block_num=block_num
        )

        # Update staircase based on response (skip timeouts)
        if res.get('resp_shape') != 'timeout':
            staircase.update(int(res.get('accuracy', 0)))

        # ── Log trial data ────────────────────────────────────────────────────
        thisExp.addData('trial_num',              global_trial_counter)
        thisExp.addData('participant',             expInfo['participant'])
        thisExp.addData('session',                 expInfo['session'])
        thisExp.addData('phase',                   'calibration')
        thisExp.addData('n_shapes',                2)
        thisExp.addData('block_num',               block_num)
        thisExp.addData('staircase_trial',         trial_num)
        thisExp.addData('prop_used',               current_prop)
        thisExp.addData('accuracy',                res.get('accuracy', 0))
        thisExp.addData('is_timeout',              res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice',               res.get('rt_choice', np.nan))
        thisExp.addData('early_response',          res.get('early_response', False))
        thisExp.addData('true_shape',              res.get('true_shape', ''))
        thisExp.addData('resp_shape',              res.get('resp_shape', ''))
        thisExp.addData('angle_bias',              angle_bias)
        thisExp.addData('applied_angle_bias',      res.get('applied_angle_bias', angle_bias))
        thisExp.addData('consecutive_correct',     staircase.consecutive_correct)
        thisExp.addData('reversals',               staircase.reversals)
        thisExp.addData('current_staircase_prop',  staircase.current_prop)
        thisExp.addData('mean_evidence',           res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence',            res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence',            res.get('var_evidence', np.nan))
        thisExp.addData('target_snippet_id',       res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids',  str(res.get('distractor_snippet_ids', [])))
        thisExp.nextEntry()

        if trial_num % 10 == 0:
            print(f"  Trial {trial_num}/{num_trials}: "
                  f"prop={current_prop:.3f}, reversals={staircase.reversals}")

        # Offer a break every 50 trials
        if trial_num % 50 == 0 and trial_num < num_trials:
            show_break_screen(trial_num, num_trials, "Calibration")

    # Compute final threshold estimate
    threshold = staircase.get_threshold_estimate(n_last_trials=8)
    summary   = staircase.get_summary()

    print(f"\nCalibration complete:")
    print(f"  Trials: {summary['trial_count']}, Reversals: {summary['reversals']}")
    print(f"  Mean accuracy: {summary['mean_accuracy']:.1%}")
    print(f"  Threshold estimate (avg last 8 trials): {threshold:.3f}")

    return staircase, threshold


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
        thisExp.addData('confidence_rating',       res.get('confidence_rating', np.nan))
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
        thisExp.nextEntry()

        # Offer a break every 50 trials within a block
        if trial_num % 50 == 0 and trial_num < num_trials:
            show_break_screen(trial_num, num_trials, f"Block {block_num} ({level_name})")


# ─────────────────────────────────────────────────────────────────────────────
#  INSTRUCTION SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def show_initial_instructions():
    """Display the welcome and general task instructions."""
    instructions = [
        """Welcome to the study.

In this session you will see two moving objects on screen.
Move your mouse and try to figure out which object you are controlling.

The session includes a practice phase (with simple shapes) followed by 4 test blocks (with images).

Press SPACE to continue...""",

        """Response instructions:

After each trial, you will be asked:
  1. Which object did you control?
       Practice: A = Square, S = Circle
       Test:     A = Left image, S = Right image
  2. How confident are you in your choice? (1-4)
  3. How much control did you feel? (1-7)

Please respond as accurately as possible. If unsure, make your best guess.

Press SPACE to continue..."""
    ]

    for instruction in instructions:
        msg.text = instruction
        msg.draw(); win.flip()
        keys = wait_keys(['space', 'escape'])
        if 'escape' in keys:
            _save(); core.quit()


def show_calibration_instructions():
    """Display instructions for the calibration (practice) phase."""
    msg.text = """PRACTICE PHASE

In this phase, you will practice the task.
After each trial, you will receive feedback: "Right" or "Wrong".

This helps us calibrate the task difficulty to your individual level.

Response keys:
  A = Square
  S = Circle

Press SPACE to start the practice..."""
    msg.draw(); win.flip()
    wait_keys(['space', 'escape'])


def show_test_block_instructions(block_num, total_blocks, level_name):
    """
    Display instructions before each test block.

    Parameters
    ----------
    block_num    : int — current block number (1-indexed)
    total_blocks : int — total number of test blocks
    level_name   : str — difficulty level label (not shown to participant)
    """
    # Note: we deliberately do NOT tell participants the difficulty level,
    # as that information could bias their responses.
    msg.text = f"""TEST BLOCK {block_num} of {total_blocks}

You will now see pairs of images moving on screen.
Move your mouse and decide which image you controlled.

After each trial:
  1. Press A (Left image) or S (Right image) to indicate which you controlled
  2. Rate your confidence (1-4)
  3. Rate how much control you felt (1-7)

No feedback will be shown in this block.

Press SPACE to start Block {block_num}..."""
    msg.draw(); win.flip()
    wait_keys(['space', 'escape'])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Show initial instructions ────────────────────────────────────────
show_initial_instructions()

# ── Step 2: Calibration phase ────────────────────────────────────────────────
# Run the 3-up-1-down staircase to find the participant's 75% threshold.
show_calibration_instructions()

msg.text = ("Calibration Phase: Practice\n\n"
            "In this block, you will practice the task.\n"
            "You will receive feedback after each response.\n\n"
            "Press SPACE to start.")
msg.draw(); win.flip(); wait_keys()

staircase, threshold_75 = run_calibration_staircase(
    num_trials=CHECK_CALIBRATION_TRIALS,
    angle_bias=0,
    block_num=0  # Block 0 = calibration (test blocks are 1-4)
)

# Save threshold to expInfo for reference
expInfo['threshold_75']          = threshold_75
expInfo['staircase_reversals']   = staircase.reversals

# Show calibration completion screen
msg.text = """Calibration complete!

Your individual difficulty level has been estimated.
You can take a short break now.

Press SPACE to continue to the main experiment."""
msg.draw(); win.flip(); wait_keys()

# ── Step 3: Calculate 4 difficulty levels ────────────────────────────────────
# Derive 4 prop values symmetrically around the calibrated threshold.
# Level 1 = hardest (threshold - 2*step), Level 4 = easiest (threshold + 2*step)
levels = calculate_difficulty_levels(threshold_75, step_size=0.08)

print(f"\nDifficulty levels derived from threshold={threshold_75:.3f}:")
for name, val in levels.items():
    print(f"  {name}: prop={val:.3f}")

# ── Step 4: Determine counterbalanced block order ─────────────────────────────
# We use all 4! = 24 permutations of the 4 difficulty levels.
# Each participant (indexed by their participant number mod 24) gets a unique
# ordering of the 4 test blocks. This ensures that across 24 participants,
# every possible block order appears exactly once.
try:
    participant_num = int(expInfo["participant"])
except ValueError:
    # If participant ID is not a number (e.g. "P01"), derive a number from its hash
    participant_num = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16)

# Generate all 24 permutations of the 4 level names
all_level_orders = list(permutations(['level_1', 'level_2', 'level_3', 'level_4']))

# Select this participant's block order
block_order = list(all_level_orders[participant_num % 24])

# Save the block order to expInfo for logging
expInfo['block_order'] = str(block_order)

print(f"\nParticipant {expInfo['participant']} (num={participant_num}): "
      f"Block order = {block_order}")

# ── Step 5: Run 4 test blocks ─────────────────────────────────────────────────
# Each block presents all trials at a single fixed difficulty level.
# Block order is counterbalanced across participants (see above).
TOTAL_BLOCKS = 4

for block_idx, level_name in enumerate(block_order):
    block_num  = block_idx + 1       # 1-indexed block number
    prop_value = levels[level_name]  # Self-proportion for this block

    # Show block instructions (difficulty level is NOT revealed to participant)
    show_test_block_instructions(block_num, TOTAL_BLOCKS, level_name)

    # Run all trials for this block
    run_test_block_for_level(
        threshold_75=threshold_75,
        level_name=level_name,
        prop_value=prop_value,
        num_trials=CHECK_TEST_TRIALS_PER_LEVEL,
        block_num=block_num,
        angle_bias=0
    )

    # Show a break screen between blocks (not after the last block)
    if block_num < TOTAL_BLOCKS:
        msg.text = (f"Block {block_num} of {TOTAL_BLOCKS} complete!\n\n"
                    f"Take a short break.\n\n"
                    f"Press SPACE when you are ready to continue.")
        msg.draw(); win.flip(); wait_keys()


# ─────────────────────────────────────────────────────────────────────────────
#  END OF EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

final_used = len(used_trajectory_indices)
print(f"\nExperiment complete!")
print(f"  Total trajectories used: {final_used}")
print(f"  Calibrated threshold (75%): {threshold_75:.3f}")
print(f"  Block order: {block_order}")

msg.text = f"""Thank you for participating!

Your data have been recorded.

Summary:
  Calibrated threshold: {threshold_75:.2f}
  Blocks completed: {TOTAL_BLOCKS}

Press SPACE to exit."""
msg.draw(); win.flip(); wait_keys()

# Save all data and close
_save()
win.close()
core.quit()
