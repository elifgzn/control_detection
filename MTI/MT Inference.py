#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_detection_task_staircase.py - 3-up-1-down staircase version
Windows-compatible version with early response capability
No color manipulation - uses 3-up-1-down staircase for calibration

Block 1: 2-shape trials (calibration + test)
Block 2: 4-shape trials (calibration + test)

"""

import os, sys, math, random, pathlib, datetime, atexit, hashlib, json, subprocess

# Check if we're running with the correct Python interpreter
def check_and_run_with_correct_python():
    # If psychopy import fails, try to find anaconda Python
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # Continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Trying to find Python with required packages...")
        
        # Try common Python paths on different systems
        python_paths = [
            "C:/Program Files/PsychoPy/python.exe",  # Standalone PsychoPy (most reliable)
            "C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe",  # Your PsychoPy conda environment
            "C:/Users/knogl/Miniconda3/python.exe",  # Your base Miniconda3
            "/opt/anaconda3/bin/python",  # macOS
            "/usr/bin/python3",  # Linux
        ]
        
        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)
        
        print("Error: Python with required packages not found. Please install psychopy, numpy, and pandas.")
        sys.exit(1)

# Check interpreter and switch if needed
if check_and_run_with_correct_python():
    sys.exit(0)

# Import statements (will work after interpreter check)
import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui

# ───────────────────────────────────────────────────────
#  Global variable for kinematics data
# ───────────────────────────────────────────────────────
kinematics_data = []
kinematics_csv_path = ""

# ───────────────────────────────────────────────────────
#  Auto‐save on quit
# ───────────────────────────────────────────────────────
_saved = False
def _save():
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
            print("Experiment not initialized - no data to save")
        _saved = True
atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "ControlDetection_Staircase_2vs4"

# Check for auto-test mode (skip dialog, run simulation)
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
        core.quit()
    SIMULATE = bool(expInfo.pop("simulate"))
    CHECK_MODE = bool(expInfo.pop("check_mode"))
    if SIMULATE:
        expInfo["participant"] = "SIM"

if CHECK_MODE:
    # Check mode: minimal trials to run through entire experiment quickly
    CHECK_CALIBRATION_TRIALS = 20  # Minimum trials for staircase
    CHECK_TEST_TRIALS_PER_LEVEL = 5  # Trials per difficulty level
else:
    # Full experiment settings
    CHECK_CALIBRATION_TRIALS = 60  # Trials for staircase calibration
    CHECK_TEST_TRIALS_PER_LEVEL = 25  # Trials per difficulty level (4 levels = 100 total)

# Print check mode status
if CHECK_MODE:
    print("=" * 60)
    print("** CHECK MODE ENABLED - Running minimal trials **")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials per block")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/level (4 levels = {CHECK_TEST_TRIALS_PER_LEVEL * 4} total per block)")
    print(f"   Total: 2 blocks x (calibration + test)")
    print("=" * 60)
else:
    print("Running FULL experiment mode")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials per block")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/level (4 levels = {CHECK_TEST_TRIALS_PER_LEVEL * 4} total per block)")
    print(f"   Total: 2 blocks x (calibration + test)")

# ───────────────────────────────────────────────────────
#  Load motion library with cluster information
# ───────────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).parent
LIB_NAME = script_dir / "Motion_Library" / "core_pool.npy"
FEATS_NAME = script_dir / "Motion_Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir / "Motion_Library" / "core_pool_labels.npy"

motion_pool = np.load(LIB_NAME)
snippet_features = np.load(FEATS_NAME)
snippet_labels = np.load(LABELS_NAME)

SNIP_LEN = motion_pool.shape[1]
TOTAL_SNIPS = motion_pool.shape[0]
K_CLUST = 4

print(f"Loaded {TOTAL_SNIPS} snippets x {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

with open(script_dir / "Motion_Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std = np.array(scp["scale"], dtype=np.float32)

with open(script_dir / "Motion_Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

participant_clusters = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(),16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)

# ───────────────────────────────────────────────────────
#  Trajectory signature function (needed for universal selection)
# ───────────────────────────────────────────────────────

def get_trajectory_signature(trajectory):
    """Key movement characteristics for matching"""
    velocities = np.diff(trajectory, axis=0)
    if len(velocities) == 0:
        return {'mean_speed':0,'speed_variability':0,'path_length':0,'net_displacement':0,'speed_percentiles':np.array([0,0,0])}
    speeds = np.linalg.norm(velocities, axis=1)
    return {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }

# ───────────────────────────────────────────────────────
#  Universal trajectory set (same for all participants)
# ───────────────────────────────────────────────────────

def select_universal_trajectory_set():
    """Pre-select trajectories for all participants with deterministic ranking."""
    global valid_snippet_indices, universal_trajectory_set_primary, universal_trajectory_set_overflow, universal_trajectory_set

    total_valid = len(valid_snippet_indices)
    if total_valid < 1240:
        print(f"Warning: Only {total_valid} valid trajectories, fewer than 1,240 required for primary set")
        universal_trajectory_set_primary = valid_snippet_indices.copy()
        universal_trajectory_set_overflow = []
        universal_trajectory_set = universal_trajectory_set_primary.copy()
        return universal_trajectory_set.copy()

    selection_rng = np.random.default_rng(42)

    print("Selecting universal trajectory sets (Primary 1,240 + Overflow 40)...")

    # Score all valid trajectories by quality
    trajectory_scores = []
    for idx in valid_snippet_indices:
        trajectory = motion_pool[idx]
        traj_cumsum = np.cumsum(trajectory, axis=0)
        sig = get_trajectory_signature(traj_cumsum)

        speed_score = 1.0 / (1.0 + abs(sig['mean_speed'] - 8.0))
        variability_score = 1.0 / (1.0 + abs(sig['speed_variability'] - 3.0))
        length_score = min(1.0, sig['path_length'] / 100.0)

        overall_score = speed_score * variability_score * length_score
        trajectory_scores.append((overall_score, idx))

    trajectory_scores.sort(reverse=True)
    primary_indices = [idx for score, idx in trajectory_scores[:1240]]
    overflow_indices = [idx for score, idx in trajectory_scores[1240:1280]] if total_valid >= 1280 else []

    universal_trajectory_set_primary = primary_indices
    universal_trajectory_set_overflow = overflow_indices
    universal_trajectory_set = primary_indices + overflow_indices

    print(f"Selected Primary: {len(primary_indices)} (best={trajectory_scores[0][0]:.3f})")
    if overflow_indices:
        print(f"Selected Overflow: {len(overflow_indices)}")
    else:
        print("No overflow set available (valid < 1,280)")

    return universal_trajectory_set.copy()

# Will be initialized after preprocessing
universal_trajectory_set_primary = []
universal_trajectory_set_overflow = []
universal_trajectory_set = []
used_trajectory_indices = set()
trajectory_usage_stats = {"used_count": 0, "total_needed": 2000}  # Increased for 4-shape trials

# Global trial counter (persists across all phases for consistent trial numbering)
global_trial_counter = 0

# -------------------------------------------------------------------
#  Helpers for simulation mode
# -------------------------------------------------------------------
class SimulatedMouse:
    def __init__(self):
        self._pos = np.array([0.0, 0.0], dtype=float)
    def setPos(self, pos=(0, 0)):
        self._pos = np.array(pos, dtype=float)
    def getPos(self):
        self._pos += rng.normal(0, 3, 2)
        return self._pos.tolist()

def wait_keys(keys=None):
    if SIMULATE:
        if keys is None:
            core.wait(0.2)
            return ["space"]
        allowed = [k for k in keys if k != "escape"] or ["space"]
        return [rng.choice(allowed)]
    return event.waitKeys(keyList=keys)

def show_break_screen(trials_completed, total_trials_in_block, block_num):
    """Show a 30-second break screen with countdown timer."""
    
    break_msg = visual.TextStim(
        win=win,
        text=f"""BREAK TIME
        
Please take a short break. You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Break time remaining: 30 seconds""",
        pos=(0, 50),
        color='white',
        height=30,
        wrapWidth=800
    )
    
    countdown_text = visual.TextStim(
        win=win,
        text='30',
        pos=(0, -100),
        color='yellow',
        height=60
    )
    
    break_clock = core.Clock()
    while break_clock.getTime() < 30.0:
        remaining_time = 30 - int(break_clock.getTime())
        countdown_text.text = str(remaining_time)
        
        break_msg.text = f"""BREAK TIME
        
Please take a short break. You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Break time remaining: {remaining_time} seconds"""
        
        break_msg.draw()
        countdown_text.draw()
        win.flip()
        
        if not SIMULATE:
            keys = event.getKeys(['escape'])
            if keys and 'escape' in keys:
                _save()
                core.quit()
        
        core.wait(0.1)
    
    final_msg = visual.TextStim(
        win=win,
        text=f"""BREAK COMPLETE
        
You have completed {trials_completed} trials.
Progress: {trials_completed}/{total_trials_in_block} trials in Block {block_num}

Press SPACE to continue""",
        pos=(0, 0),
        color='white',
        height=30,
        wrapWidth=800
    )
    
    final_msg.draw()
    win.flip()
    wait_keys(['space', 'escape'])

# ───────────────────────────────────────────────────────
#  Trajectory Quality Control and Preprocessing
# ───────────────────────────────────────────────────────

def analyze_trajectory_quality(trajectory):
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)
    high_jitter_ratio = np.sum(speeds > mean_speed + 3*std_speed) / len(speeds)
    if len(velocities) > 1:
        unit_velocities = velocities / (sps := (speeds.reshape(-1, 1) + 1e-9))
        angle_changes = np.arccos(np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1))
        mean_angle_change = np.mean(angle_changes)
        jerkiness = np.std(angle_changes)
    else:
        mean_angle_change = 0
        jerkiness = 0
    return {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio': high_jitter_ratio,
        'mean_angle_change': mean_angle_change,
        'jerkiness': jerkiness,
        'speed_range': max_speed - min_speed
    }

def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3, max_jitter_ratio=0.1, max_jerkiness=1.5):
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
    if len(trajectory) < 2:
        return trajectory
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    current_mean_speed = np.mean(speeds)
    if current_mean_speed > 0:
        target_mean_speed = np.mean(target_speed_range)
        speed_scale = target_mean_speed / current_mean_speed
        velocities = velocities * speed_scale
    smoothed_velocities = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed_velocities[i] = smooth_factor * smoothed_velocities[i-1] + (1 - smooth_factor) * velocities[i]
    normalized_trajectory = [trajectory[0]]
    for vel in smoothed_velocities:
        next_point = normalized_trajectory[-1] + vel
        normalized_trajectory.append(next_point)
    return np.array(normalized_trajectory)

def preprocess_motion_pool():
    """Preprocess motion pool to ensure quality and consistency"""
    global motion_pool, snippet_features, snippet_labels
    print("Preprocessing motion pool for quality control...")
    initial_count = len(motion_pool)
    processed_snippets = []
    processed_features = []
    processed_labels = []
    for i, snippet in enumerate(motion_pool):
        trajectory = np.cumsum(snippet, axis=0)
        is_valid, reason = is_trajectory_valid(trajectory)
        if is_valid:
            normalized_trajectory = normalize_trajectory(trajectory)
            velocities = np.diff(normalized_trajectory, axis=0)
            processed_snippets.append(velocities)
            processed_features.append(snippet_features[i])
            processed_labels.append(snippet_labels[i])
        else:
            print(f"Removed snippet {i}: {reason}")
    motion_pool = np.array(processed_snippets)
    snippet_features = np.array(processed_features)
    snippet_labels = np.array(processed_labels)
    global SNIP_LEN
    SNIP_LEN = motion_pool.shape[1] if len(motion_pool) > 0 else 0
    print(f"Motion pool preprocessed: kept {len(processed_snippets)}/{initial_count} snippets")
    return list(range(len(processed_snippets)))

valid_snippet_indices = preprocess_motion_pool()

# Now initialize the universal trajectory sets after preprocessing
universal_trajectory_set = select_universal_trajectory_set()
print(
    f"Universal trajectory sets initialized: Primary={len(universal_trajectory_set_primary)}; "
    f"Overflow={len(universal_trajectory_set_overflow)}; Total={len(universal_trajectory_set)}"
)

def get_trajectory_indices(n_trajectories):
    """Get n unique trajectory indices for a trial."""
    global used_trajectory_indices, trajectory_usage_stats
    
    available_primary = [idx for idx in universal_trajectory_set_primary if idx not in used_trajectory_indices]
    available_overflow = [idx for idx in universal_trajectory_set_overflow if idx not in used_trajectory_indices]
    available_indices = available_primary if len(available_primary) >= n_trajectories else (available_primary + available_overflow)
    
    if len(available_indices) >= n_trajectories:
        selected = rng.choice(available_indices, size=n_trajectories, replace=False)
        for idx in selected:
            used_trajectory_indices.add(idx)
        trajectory_usage_stats["used_count"] += n_trajectories
        return list(selected)
    else:
        # Fallback: use any from universal set
        combined = universal_trajectory_set_primary + universal_trajectory_set_overflow
        if len(combined) >= n_trajectories:
            selected = rng.choice(combined, size=n_trajectories, replace=False)
            return list(selected)
        else:
            # Emergency: use from all valid
            return list(rng.choice(valid_snippet_indices, size=n_trajectories, replace=False))

def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to both trajectories"""
    def smooth_trajectory(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed
    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    return vel1, vel2

def smooth_single_trajectory(trajectory):
    """Apply smoothing to a single trajectory"""
    def smooth_traj(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed
    pos = np.cumsum(trajectory, axis=0)
    smooth_pos = smooth_traj(pos)
    vel = np.diff(smooth_pos, axis=0)
    return vel

def mix_direction_only(mouse_dx, mouse_dy, traj_dx, traj_dy, prop):
    """
    Mix mouse and trajectory directions, preserving trajectory speed.
    Returns velocity with mixed direction but trajectory magnitude.
    
    This ensures the target shape moves at the same speed as distractors,
    with only the direction influenced by mouse input.
    """
    traj_speed = math.hypot(traj_dx, traj_dy)
    if traj_speed < 0.01:
        return traj_dx, traj_dy
    
    mouse_mag = math.hypot(mouse_dx, mouse_dy)
    if mouse_mag < 0.01:
        return traj_dx, traj_dy
    
    # Unit directions
    m_dir_x, m_dir_y = mouse_dx / mouse_mag, mouse_dy / mouse_mag
    t_dir_x, t_dir_y = traj_dx / traj_speed, traj_dy / traj_speed
    
    # Mix directions
    mix_x = prop * m_dir_x + (1 - prop) * t_dir_x
    mix_y = prop * m_dir_y + (1 - prop) * t_dir_y
    
    # Normalize and apply trajectory speed
    mix_mag = math.hypot(mix_x, mix_y)
    if mix_mag > 0.01:
        return (mix_x / mix_mag) * traj_speed, (mix_y / mix_mag) * traj_speed
    return traj_dx, traj_dy

# ───────────────────────────────────────────────────────
#  Constants and Parameters
# ───────────────────────────────────────────────────────
OFFSET_X = 300
OFFSET_Y = 150
LOWPASS = 0.2  # Lower = less smoothing, more responsive movement
SPEED_MULTIPLIER = 1.5  # Multiply trajectory velocities to make shapes move faster

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
root = pathlib.Path(__file__).parent / "data"
subjects_dir = root / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

participant_id = expInfo['participant']
base_filename = f"CDT_staircase_2vs4_{participant_id}"
csv_path = subjects_dir / f"{base_filename}.csv"
kinematics_csv_path = subjects_dir / f"{base_filename}_kinematics.csv"

i = 1
while csv_path.exists():
    new_filename = f"CDT_staircase_2vs4_{participant_id}_{i}"
    csv_path = subjects_dir / f"{new_filename}.csv"
    kinematics_csv_path = subjects_dir / f"{new_filename}_kinematics.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename)
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win = visual.Window((1920,1080), fullscr=not SIMULATE, color=[0.5]*3, units="pix", allowGUI=True)
win.setMouseVisible(False)

# 2-shape stimuli
square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot = visual.Circle(win, 20, fillColor="black", lineColor="black")

# 4-shape stimuli (4 distinct shapes)
shape_square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
shape_circle = visual.Circle(win, 20, fillColor="black", lineColor="black")
shape_triangle = visual.Polygon(win, edges=3, radius=25, fillColor="black", lineColor="black")
shape_hexagon = visual.Polygon(win, edges=6, radius=25, fillColor="black", lineColor="black")  # Hexagon for clear visual distinction

# Shape definitions for 4-shape trials
FOUR_SHAPES = {
    'square': shape_square,
    'circle': shape_circle,
    'triangle': shape_triangle,
    'hexagon': shape_hexagon
}
FOUR_SHAPE_NAMES = ['square', 'circle', 'triangle', 'hexagon']
FOUR_SHAPE_KEYS = {'a': 'square', 's': 'circle', 'd': 'triangle', 'f': 'hexagon'}

fix = visual.TextStim(win, "+", color="white", height=60)
msg = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

confine = lambda p, l=400: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)  # Increased from 250 for faster movement
rotate = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)

# ───────────────────────────────────────────────────────
#  3-Up-1-Down Staircase Class
# ───────────────────────────────────────────────────────

class ThreeUpOneDownStaircase:
    """
    3-up-1-down staircase procedure for estimating 75% accuracy threshold.
    
    - After 3 consecutive correct responses: decrease self-prop (make harder)
    - After 1 incorrect response: increase self-prop (make easier)
    
    The staircase converges to approximately 79.4% correct, which is close to 75%.
    Final threshold is estimated as the average of the last 8 trial self-prop values.
    """
    
    def __init__(self, start_prop=0.25, step_size=0.05, min_prop=0.05, max_prop=0.90):
        """
        Initialize the staircase.
        
        Args:
            start_prop: Starting self-proportion (default 0.25 - harder start)
            step_size: Step size for adjustments (default 0.05)
            min_prop: Minimum allowed self-proportion (default 0.05)
            max_prop: Maximum allowed self-proportion (default 0.90)
        """
        self.current_prop = start_prop
        self.step_size = step_size
        self.min_prop = min_prop
        self.max_prop = max_prop
        
        self.consecutive_correct = 0
        self.trial_count = 0
        self.reversals = 0
        self.last_direction = None  # 'up' or 'down'
        
        # History tracking
        self.prop_history = []  # All self-props presented
        self.response_history = []  # All responses (1=correct, 0=incorrect)
        self.reversal_props = []  # Self-props at reversal points
        
    def get_current_prop(self):
        """Get the current self-proportion for the next trial."""
        return self.current_prop
    
    def update(self, correct):
        """
        Update the staircase based on the response.
        
        Args:
            correct: Boolean or int (1=correct, 0=incorrect)
        """
        correct = int(correct)
        self.trial_count += 1
        self.prop_history.append(self.current_prop)
        self.response_history.append(correct)
        
        if correct:
            self.consecutive_correct += 1
            
            # After 3 consecutive correct: make harder (decrease prop)
            if self.consecutive_correct >= 3:
                self.consecutive_correct = 0
                new_prop = self.current_prop - self.step_size
                new_prop = max(self.min_prop, new_prop)
                
                # Check for reversal
                if self.last_direction == 'up':
                    self.reversals += 1
                    self.reversal_props.append(self.current_prop)
                
                self.current_prop = new_prop
                self.last_direction = 'down'
        else:
            # After 1 incorrect: make easier (increase prop)
            self.consecutive_correct = 0
            new_prop = self.current_prop + self.step_size
            new_prop = min(self.max_prop, new_prop)
            
            # Check for reversal
            if self.last_direction == 'down':
                self.reversals += 1
                self.reversal_props.append(self.current_prop)
            
            self.current_prop = new_prop
            self.last_direction = 'up'
    
    def get_threshold_estimate(self, n_last_trials=8):
        """
        Get the threshold estimate as the average of the last n trials.
        
        Args:
            n_last_trials: Number of last trials to average (default 8)
            
        Returns:
            Estimated threshold (self-proportion for ~75% accuracy)
        """
        if len(self.prop_history) < n_last_trials:
            # If not enough trials, use all available
            return np.mean(self.prop_history) if self.prop_history else self.current_prop
        
        return np.mean(self.prop_history[-n_last_trials:])
    
    def get_summary(self):
        """Get summary statistics of the staircase."""
        return {
            'trial_count': self.trial_count,
            'reversals': self.reversals,
            'final_prop': self.current_prop,
            'threshold_estimate': self.get_threshold_estimate(),
            'mean_accuracy': np.mean(self.response_history) if self.response_history else 0,
            'prop_history': self.prop_history.copy(),
            'response_history': self.response_history.copy()
        }


# ───────────────────────────────────────────────────────
#  Difficulty Level Calculator
# ───────────────────────────────────────────────────────

def calculate_difficulty_levels(threshold_75, step_size=0.08, min_prop=0.05, max_prop=0.90):
    """
    Calculate 4 difficulty levels based on the 75% threshold.
    
    Args:
        threshold_75: The self-proportion for 75% accuracy (from staircase)
        step_size: Step between levels (default 0.08)
        min_prop: Minimum allowed proportion
        max_prop: Maximum allowed proportion
    
    Returns:
        Dictionary with 4 difficulty levels:
        - level_1: threshold - 2*step (hardest)
        - level_2: threshold - 1*step
        - level_3: threshold + 1*step
        - level_4: threshold + 2*step (easiest)
    """
    levels = {
        'level_1': np.clip(threshold_75 - 2 * step_size, min_prop, max_prop),  # Hardest
        'level_2': np.clip(threshold_75 - 1 * step_size, min_prop, max_prop),
        'level_3': np.clip(threshold_75 + 1 * step_size, min_prop, max_prop),
        'level_4': np.clip(threshold_75 + 2 * step_size, min_prop, max_prop),  # Easiest
    }
    return levels


# ───────────────────────────────────────────────────────
#  2-SHAPE Trial function
# ───────────────────────────────────────────────────────
# Minimum mouse speed threshold to consider mouse "moving"
MOUSE_MOVE_THRESHOLD = 0.5
# Fixed motion duration (seconds)
MOTION_DURATION = 3.0

def run_trial_2shapes(
    trial_num, phase, angle_bias, mode, block_num=1,
    prop_override=None, cue_dur_range=None, motion_dur=None, response_window=None,
    difficulty_level=None
):
    """Run a single 2-shape trial.
    
    Shapes move independently following their trajectories.
    Mouse input is only mixed into the target when the mouse is moving.
    Motion lasts exactly 3 seconds, then a response screen appears.
    """
    
    if prop_override is not None:
        prop = float(np.clip(prop_override, 0.02, 0.90))
    else:
        prop = 0.40  # Default fallback
    
    # Always use black cue (no color manipulation)
    fix.color = "white"
    square.fillColor = square.lineColor = "black"
    dot.fillColor = dot.lineColor = "black"

    fix.draw(); win.flip()
    if cue_dur_range is not None:
        core.wait(float(rng.uniform(cue_dur_range[0], cue_dur_range[1])))
    else:
        core.wait(1.0)

    left_shape = random.choice(['square', 'dot'])
    if left_shape == 'square':
        square.pos = (-OFFSET_X, 0); dot.pos = (OFFSET_X, 0)
    else:
        square.pos = (OFFSET_X, 0); dot.pos = (-OFFSET_X, 0)
    square.draw(); dot.draw(); win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        square.draw(); dot.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE: break
        if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()

    target = random.choice(["square", "dot"])
    trajectory_indices = get_trajectory_indices(2)
    target_snippet_idx, distractor_snippet_idx = trajectory_indices[0], trajectory_indices[1]
    
    target_snippet = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    target_snippet, distractor_snippet = apply_consistent_smoothing(target_snippet, distractor_snippet)

    trial_kinematics = []
    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)
    
    event.clearEvents(eventType='keyboard')
    
    applied_angle = angle_bias
    if angle_bias == 90:
        applied_angle = int(rng.choice([90, -90]))
    
    # === MOTION PHASE: exactly 3 seconds, no response allowed ===
    while clk.getTime() < MOTION_DURATION:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        dx, dy = rotate(dx, dy, applied_angle)
        
        # Get trajectory velocities for this frame (shapes move independently)
        target_traj_dx, target_traj_dy = target_snippet[frame % len(target_snippet)]
        distractor_traj_dx, distractor_traj_dy = distractor_snippet[frame % len(distractor_snippet)]
        
        # Apply speed multiplier to make shapes move faster
        target_traj_dx *= SPEED_MULTIPLIER
        target_traj_dy *= SPEED_MULTIPLIER
        distractor_traj_dx *= SPEED_MULTIPLIER
        distractor_traj_dy *= SPEED_MULTIPLIER
        
        frame += 1
        
        # Calculate mouse speed
        mouse_speed = math.hypot(dx, dy)
        MAX_SPEED = 20.0
        if mouse_speed > MAX_SPEED:
            scale_factor = MAX_SPEED / mouse_speed
            dx = dx * scale_factor
            dy = dy * scale_factor
            mouse_speed = MAX_SPEED
        
        # Determine if mouse is moving
        mouse_is_moving = mouse_speed > MOUSE_MOVE_THRESHOLD
        
        # Target shape: mix mouse DIRECTION only (preserving trajectory speed)
        # This ensures target moves at same speed as distractor - only direction differs
        if mouse_is_moving:
            tdx, tdy = mix_direction_only(dx, dy, target_traj_dx, target_traj_dy, prop)
        else:
            # When mouse is stationary: follow trajectory only
            tdx = target_traj_dx
            tdy = target_traj_dy
        
        # Distractor: always follows its own trajectory
        ddx = distractor_traj_dx
        ddy = distractor_traj_dy
        
        # Low-pass filter for smooth movement
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

        # Evidence calculation (only meaningful when mouse is moving)
        vm = np.array([dx, dy], dtype=float)
        vm_speed = np.linalg.norm(vm) + 1e-9

        vt_disp = np.array(vt, dtype=float)
        vd_disp = np.array(vd, dtype=float)

        ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
        ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)

        cos_T = np.dot(vm, ut) / vm_speed
        cos_D = np.dot(vm, ud) / vm_speed

        evidence = (cos_T - cos_D) * mouse_speed if mouse_is_moving else 0.0

        if target == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos = confine(tuple(dot.pos + vd))
        else:
            dot.pos = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        trial_kinematics.append({
            'timestamp': clk.getTime(), 'frame': frame, 'mouse_x': x, 'mouse_y': y,
            'mouse_speed': mouse_speed, 'mouse_is_moving': mouse_is_moving,
            'square_x': square.pos[0], 'square_y': square.pos[1], 'dot_x': dot.pos[0], 'dot_y': dot.pos[1],
            'evidence': evidence
        })
        
        # Check for escape only (no response during motion)
        if not SIMULATE:
            keys = event.getKeys(['escape'])
            if keys and 'escape' in keys:
                _save(); core.quit()
        
        square.draw(); dot.draw(); win.flip()
    
    # === RESPONSE PHASE: separate screen, must respond ===
    event.clearEvents(eventType='keyboard')
    response_start_time = core.getTime()
    
    # Show response screen
    msg.text = "Which shape did you control?\n\nA = Square          S = Circle"
    msg.draw()
    win.flip()
    
    # Wait for response (no timeout)
    resp_shape = None
    rt_choice = np.nan
    
    if SIMULATE:
        core.wait(0.3)
        resp_shape = rng.choice(["square", "dot"])
        rt_choice = 0.3
    else:
        while resp_shape is None:
            keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == "escape":
                    _save(); core.quit()
                elif key == "a":
                    resp_shape = "square"
                    rt_choice = key_time - response_start_time
                elif key == "s":
                    resp_shape = "dot"
                    rt_choice = key_time - response_start_time
            core.wait(0.01)
    
    correct = int(resp_shape == target)

    # Confidence rating (1-4 scale) - only in test phase
    confidence_rating = np.nan
    if phase == "test":
        if SIMULATE:
            confidence_rating = float(rng.integers(1, 5))
        else:
            msg.text = "How confident are you in your choice?\n\n1 = Not at all confident\n2 = Slightly confident\n3 = Moderately confident\n4 = Very confident"
            msg.draw(); win.flip()
            
            confidence_keys = ['1', '2', '3', '4']
            conf_key = wait_keys(confidence_keys + ["escape"])[0]
            if conf_key == "escape": 
                _save(); core.quit()
            else:
                confidence_rating = int(conf_key)
            core.wait(0.2)

    # Show feedback for calibration trials
    if phase == "calibration":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # Agency rating (only for test trials)
    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
        else:
            event.clearEvents(eventType='keyboard')
            
            msg.text = "How much control did you feel over the shape's movement?"
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100), (150, -100), (300, -100), (450, -100)]
            scale_labels = ["1\nVery weak","2\nWeak","3\nSomewhat weak","4\nModerate","5\nSomewhat strong","6\nStrong","7\nVery strong"]
            scale_stimuli = [visual.TextStim(win, text=label, pos=pos, height=18, color='white', alignText='center')
                             for pos, label in zip(scale_positions, scale_labels)]
            rating = None
            while rating is None:
                msg.draw()
                for stim in scale_stimuli: stim.draw()
                win.flip()
                keys = event.getKeys(['1','2','3','4','5','6','7','escape'])
                if keys:
                    if 'escape' in keys: _save(); core.quit()
                    else: rating = int(keys[0])
                core.wait(0.01)
            agency_rating = rating
            core.wait(0.2)

    frame_evidence = [d['evidence'] for d in trial_kinematics]
    mean_evidence = np.mean(frame_evidence) if frame_evidence else np.nan
    sum_evidence = np.sum(frame_evidence) if frame_evidence else np.nan
    var_evidence = np.var(frame_evidence) if frame_evidence else np.nan

    # Add trial metadata to each frame and append to global kinematics
    for frame_data in trial_kinematics:
        frame_data.update({
            'trial_num': trial_num,
            'phase': phase,
            'n_shapes': 2,
            'target': target,
            'prop_used': prop
        })
    kinematics_data.extend(trial_kinematics)

    return dict(
        n_shapes=2, target_snippet_id=target_snippet_idx, distractor_snippet_ids=[distractor_snippet_idx],
        phase=phase, block_num=block_num,
        angle_bias=angle_bias, applied_angle_bias=applied_angle, true_shape=target, resp_shape=resp_shape,
        confidence_rating=confidence_rating, accuracy=correct, rt_choice=rt_choice, 
        agency_rating=agency_rating, prop_used=prop, early_response=False,
        mean_evidence=mean_evidence, sum_evidence=sum_evidence, var_evidence=var_evidence,
        difficulty_level=difficulty_level
    )


# ───────────────────────────────────────────────────────
#  4-SHAPE Trial function
# ───────────────────────────────────────────────────────
def run_trial_4shapes(
    trial_num, phase, angle_bias, mode, block_num=1,
    prop_override=None, cue_dur_range=None, motion_dur=None, response_window=None,
    difficulty_level=None
):
    """Run a single 4-shape trial (1 target + 3 distractors).
    
    Shapes move independently following their trajectories.
    Mouse input is only mixed into the target when the mouse is moving.
    Motion lasts exactly 3 seconds, then a response screen appears.
    """
    
    if prop_override is not None:
        prop = float(np.clip(prop_override, 0.02, 0.90))
    else:
        prop = 0.40  # Default fallback
    
    # Reset all shapes to black
    for shape in FOUR_SHAPES.values():
        shape.fillColor = shape.lineColor = "black"
    
    fix.color = "white"
    fix.draw(); win.flip()
    if cue_dur_range is not None:
        core.wait(float(rng.uniform(cue_dur_range[0], cue_dur_range[1])))
    else:
        core.wait(1.0)

    # Position 4 shapes in quadrants
    positions = [
        (-OFFSET_X, OFFSET_Y),   # Top-left
        (OFFSET_X, OFFSET_Y),    # Top-right
        (-OFFSET_X, -OFFSET_Y),  # Bottom-left
        (OFFSET_X, -OFFSET_Y)    # Bottom-right
    ]
    
    # Shuffle positions
    shuffled_positions = positions.copy()
    rng.shuffle(shuffled_positions)
    
    # Assign positions to shapes
    shape_positions = {}
    for i, shape_name in enumerate(FOUR_SHAPE_NAMES):
        shape_positions[shape_name] = shuffled_positions[i]
        FOUR_SHAPES[shape_name].pos = shuffled_positions[i]
    
    # Draw all 4 shapes
    for shape in FOUR_SHAPES.values():
        shape.draw()
    win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        for shape in FOUR_SHAPES.values():
            shape.draw()
        win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE: break
        if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()

    # Select target and get 4 trajectories (1 target + 3 distractors)
    target = rng.choice(FOUR_SHAPE_NAMES)
    trajectory_indices = get_trajectory_indices(4)
    target_snippet_idx = trajectory_indices[0]
    distractor_snippet_indices = trajectory_indices[1:]
    
    # Load and smooth trajectories
    target_snippet = smooth_single_trajectory(motion_pool[target_snippet_idx])
    distractor_snippets = [smooth_single_trajectory(motion_pool[idx]) for idx in distractor_snippet_indices]
    
    # Assign distractors to non-target shapes
    non_target_shapes = [s for s in FOUR_SHAPE_NAMES if s != target]
    shape_to_snippet = {target: target_snippet}
    for i, shape_name in enumerate(non_target_shapes):
        shape_to_snippet[shape_name] = distractor_snippets[i]

    trial_kinematics = []
    clk = core.Clock(); frame = 0
    
    # Velocity states for each shape (for low-pass filtering)
    velocities = {name: np.zeros(2, np.float32) for name in FOUR_SHAPE_NAMES}
    
    event.clearEvents(eventType='keyboard')
    
    applied_angle = angle_bias
    if angle_bias == 90:
        applied_angle = int(rng.choice([90, -90]))
    
    # === MOTION PHASE: exactly 3 seconds, no response allowed ===
    while clk.getTime() < MOTION_DURATION:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        dx, dy = rotate(dx, dy, applied_angle)
        frame += 1
        
        # Calculate mouse speed
        mouse_speed = math.hypot(dx, dy)
        MAX_SPEED = 20.0
        if mouse_speed > MAX_SPEED:
            scale_factor = MAX_SPEED / mouse_speed
            dx = dx * scale_factor
            dy = dy * scale_factor
            mouse_speed = MAX_SPEED
        
        # Determine if mouse is moving
        mouse_is_moving = mouse_speed > MOUSE_MOVE_THRESHOLD
        
        # Update each shape
        for shape_name in FOUR_SHAPE_NAMES:
            snippet = shape_to_snippet[shape_name]
            # Get trajectory velocity for this frame (independent movement)
            traj_dx, traj_dy = snippet[frame % len(snippet)]
            
            # Apply speed multiplier to make shapes move faster
            traj_dx *= SPEED_MULTIPLIER
            traj_dy *= SPEED_MULTIPLIER
            
            if shape_name == target:
                # Target: mix mouse DIRECTION only (preserving trajectory speed)
                # This ensures target moves at same speed as distractors - only direction differs
                if mouse_is_moving:
                    new_vx, new_vy = mix_direction_only(dx, dy, traj_dx, traj_dy, prop)
                else:
                    # When mouse is stationary: follow trajectory only
                    new_vx = traj_dx
                    new_vy = traj_dy
            else:
                # Distractor: always follows its own trajectory
                new_vx = traj_dx
                new_vy = traj_dy
            
            # Low-pass filter for smooth movement
            velocities[shape_name] = LOWPASS * velocities[shape_name] + (1 - LOWPASS) * np.array([new_vx, new_vy])
            
            # Update position
            shape = FOUR_SHAPES[shape_name]
            shape.pos = confine(tuple(np.array(shape.pos) + velocities[shape_name]))
        
        # Calculate evidence (only meaningful when mouse is moving)
        vm = np.array([dx, dy], dtype=float)
        vm_speed = np.linalg.norm(vm) + 1e-9
        
        vt_disp = velocities[target]
        ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
        cos_T = np.dot(vm, ut) / vm_speed
        
        # Average distractor evidence
        cos_D_sum = 0
        for shape_name in non_target_shapes:
            vd_disp = velocities[shape_name]
            ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)
            cos_D_sum += np.dot(vm, ud) / vm_speed
        cos_D = cos_D_sum / len(non_target_shapes)
        
        evidence = (cos_T - cos_D) * mouse_speed if mouse_is_moving else 0.0

        trial_kinematics.append({
            'timestamp': clk.getTime(), 'frame': frame, 'mouse_x': x, 'mouse_y': y,
            'mouse_speed': mouse_speed, 'mouse_is_moving': mouse_is_moving,
            'evidence': evidence,
            **{f'{name}_x': FOUR_SHAPES[name].pos[0] for name in FOUR_SHAPE_NAMES},
            **{f'{name}_y': FOUR_SHAPES[name].pos[1] for name in FOUR_SHAPE_NAMES}
        })
        
        # Check for escape only (no response during motion)
        if not SIMULATE:
            keys = event.getKeys(['escape'])
            if keys and 'escape' in keys:
                _save(); core.quit()
        
        for shape in FOUR_SHAPES.values():
            shape.draw()
        win.flip()
    
    # === RESPONSE PHASE: separate screen, must respond ===
    event.clearEvents(eventType='keyboard')
    response_start_time = core.getTime()
    
    # Show response screen
    msg.text = "Which shape did you control?\n\nA = Square    S = Circle    D = Triangle    F = Hexagon"
    msg.draw()
    win.flip()
    
    # Wait for response (no timeout)
    resp_shape = None
    rt_choice = np.nan
    
    if SIMULATE:
        core.wait(0.3)
        resp_shape = rng.choice(FOUR_SHAPE_NAMES)
        rt_choice = 0.3
    else:
        while resp_shape is None:
            keys = event.getKeys(['a', 's', 'd', 'f', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == "escape":
                    _save(); core.quit()
                elif key in FOUR_SHAPE_KEYS:
                    resp_shape = FOUR_SHAPE_KEYS[key]
                    rt_choice = key_time - response_start_time
            core.wait(0.01)
    
    correct = int(resp_shape == target)

    # Confidence rating (1-4 scale) - only in test phase
    confidence_rating = np.nan
    if phase == "test":
        if SIMULATE:
            confidence_rating = float(rng.integers(1, 5))
        else:
            msg.text = "How confident are you in your choice?\n\n1 = Not at all confident\n2 = Slightly confident\n3 = Moderately confident\n4 = Very confident"
            msg.draw(); win.flip()
            
            confidence_keys = ['1', '2', '3', '4']
            conf_key = wait_keys(confidence_keys + ["escape"])[0]
            if conf_key == "escape": 
                _save(); core.quit()
            else:
                confidence_rating = int(conf_key)
            core.wait(0.2)

    # Show feedback for calibration trials
    if phase == "calibration":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # Agency rating (only for test trials)
    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
        else:
            event.clearEvents(eventType='keyboard')
            
            msg.text = "How much control did you feel over the shape's movement?"
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100), (150, -100), (300, -100), (450, -100)]
            scale_labels = ["1\nVery weak","2\nWeak","3\nSomewhat weak","4\nModerate","5\nSomewhat strong","6\nStrong","7\nVery strong"]
            scale_stimuli = [visual.TextStim(win, text=label, pos=pos, height=18, color='white', alignText='center')
                             for pos, label in zip(scale_positions, scale_labels)]
            rating = None
            while rating is None:
                msg.draw()
                for stim in scale_stimuli: stim.draw()
                win.flip()
                keys = event.getKeys(['1','2','3','4','5','6','7','escape'])
                if keys:
                    if 'escape' in keys: _save(); core.quit()
                    else: rating = int(keys[0])
                core.wait(0.01)
            agency_rating = rating
            core.wait(0.2)

    frame_evidence = [d['evidence'] for d in trial_kinematics]
    mean_evidence = np.mean(frame_evidence) if frame_evidence else np.nan
    sum_evidence = np.sum(frame_evidence) if frame_evidence else np.nan
    var_evidence = np.var(frame_evidence) if frame_evidence else np.nan

    # Add trial metadata to each frame and append to global kinematics
    for frame_data in trial_kinematics:
        frame_data.update({
            'trial_num': trial_num,
            'phase': phase,
            'n_shapes': 4,
            'target': target,
            'prop_used': prop
        })
    kinematics_data.extend(trial_kinematics)

    return dict(
        n_shapes=4, target_snippet_id=target_snippet_idx, distractor_snippet_ids=distractor_snippet_indices,
        phase=phase, block_num=block_num,
        angle_bias=angle_bias, applied_angle_bias=applied_angle, true_shape=target, resp_shape=resp_shape,
        confidence_rating=confidence_rating, accuracy=correct, rt_choice=rt_choice, 
        agency_rating=agency_rating, prop_used=prop, early_response=False,
        mean_evidence=mean_evidence, sum_evidence=sum_evidence, var_evidence=var_evidence,
        difficulty_level=difficulty_level
    )


# ───────────────────────────────────────────────────────
#  Calibration Phase: 3-up-1-down Staircase
# ───────────────────────────────────────────────────────

def run_calibration_staircase(num_trials, n_shapes=2, angle_bias=0, block_num=1):
    """
    Run 3-up-1-down staircase calibration to find 75% accuracy threshold.
    
    Args:
        num_trials: Number of calibration trials to run
        n_shapes: Number of shapes (2 or 4)
        angle_bias: Rotation angle for this condition (0 or 90)
        block_num: Block number for logging
    
    Returns:
        staircase: The completed staircase object
        threshold: Estimated threshold (average of last 8 trials)
    """
    global global_trial_counter
    
    staircase = ThreeUpOneDownStaircase(
        start_prop=0.25,  # Start harder (25% mouse control)
        step_size=0.05,
        min_prop=0.05,
        max_prop=0.90
    )
    
    print(f"Starting 3-up-1-down staircase calibration for {n_shapes}-shape, {angle_bias}° angle")
    print(f"  Starting prop: {staircase.current_prop:.2f}")
    print(f"  Step size: {staircase.step_size:.2f}")
    
    run_trial_func = run_trial_2shapes if n_shapes == 2 else run_trial_4shapes
    
    for trial_num in range(1, num_trials + 1):
        global_trial_counter += 1
        
        current_prop = staircase.get_current_prop()
        
        res = run_trial_func(
            trial_num, "calibration", angle_bias=angle_bias, mode="staircase",
            prop_override=current_prop, cue_dur_range=(0.5, 0.8), motion_dur=5.0,
            difficulty_level="calibration"
        )
        
        # Only update staircase for valid responses
        if res.get('resp_shape') != 'timeout':
            correct = int(res.get('accuracy', 0))
            staircase.update(correct)
        
        # Log trial data
        thisExp.addData('trial_num', global_trial_counter)
        thisExp.addData('participant', expInfo['participant'])
        thisExp.addData('session', expInfo['session'])
        thisExp.addData('phase', 'calibration')
        thisExp.addData('n_shapes', n_shapes)
        thisExp.addData('block_num', block_num)
        thisExp.addData('staircase_trial', trial_num)
        thisExp.addData('prop_used', current_prop)
        thisExp.addData('accuracy', res.get('accuracy', 0))
        thisExp.addData('is_timeout', res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
        thisExp.addData('early_response', res.get('early_response', False))
        thisExp.addData('true_shape', res.get('true_shape', ''))
        thisExp.addData('resp_shape', res.get('resp_shape', ''))
        thisExp.addData('angle_bias', angle_bias)
        thisExp.addData('applied_angle_bias', res.get('applied_angle_bias', angle_bias))
        thisExp.addData('consecutive_correct', staircase.consecutive_correct)
        thisExp.addData('reversals', staircase.reversals)
        thisExp.addData('current_staircase_prop', staircase.current_prop)
        # Evidence metrics
        thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
        # Trajectory IDs
        thisExp.addData('target_snippet_id', res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids', str(res.get('distractor_snippet_ids', [])))
        thisExp.nextEntry()
        
        # Progress update every 10 trials
        if trial_num % 10 == 0:
            print(f"  Trial {trial_num}/{num_trials}: prop={current_prop:.3f}, reversals={staircase.reversals}")
        
        # Show break screen every 50 trials
        if trial_num % 50 == 0 and trial_num < num_trials:
            show_break_screen(trial_num, num_trials, f"Calibration ({n_shapes}-shape)")
    
    # Calculate final threshold
    threshold = staircase.get_threshold_estimate(n_last_trials=8)
    summary = staircase.get_summary()
    
    print(f"\nCalibration complete for {n_shapes}-shape, {angle_bias}° angle:")
    print(f"  Total trials: {summary['trial_count']}")
    print(f"  Reversals: {summary['reversals']}")
    print(f"  Mean accuracy: {summary['mean_accuracy']:.1%}")
    print(f"  Threshold estimate (avg last 8): {threshold:.3f}")
    
    return staircase, threshold


def run_test_phase(threshold_75, num_trials_per_level, n_shapes=2, angle_bias=0, block_num=1):
    """
    Run test phase with 4 difficulty levels based on calibrated threshold.
    
    Args:
        threshold_75: The 75% accuracy threshold from calibration
        num_trials_per_level: Number of trials per difficulty level
        n_shapes: Number of shapes (2 or 4)
        angle_bias: Rotation angle for this condition
        block_num: Block number for logging
    """
    global global_trial_counter
    
    # Calculate 4 difficulty levels
    levels = calculate_difficulty_levels(threshold_75, step_size=0.08)
    
    print(f"\nTest phase for {n_shapes}-shape, {angle_bias}° angle:")
    print(f"  Threshold (75%): {threshold_75:.3f}")
    print(f"  Level 1 (hardest): {levels['level_1']:.3f}")
    print(f"  Level 2: {levels['level_2']:.3f}")
    print(f"  Level 3: {levels['level_3']:.3f}")
    print(f"  Level 4 (easiest): {levels['level_4']:.3f}")
    
    # Create trial list with all 4 levels
    test_trials = []
    for level_name, prop_value in levels.items():
        for _ in range(num_trials_per_level):
            test_trials.append((level_name, prop_value))
    
    # Shuffle trials
    rng.shuffle(test_trials)
    
    total_trials = len(test_trials)
    print(f"  Total test trials: {total_trials}")
    
    run_trial_func = run_trial_2shapes if n_shapes == 2 else run_trial_4shapes
    
    trial_counter = 0
    for level_name, prop_value in test_trials:
        trial_counter += 1
        global_trial_counter += 1
        
        res = run_trial_func(
            trial_counter, "test", angle_bias=angle_bias, mode="test",
            prop_override=prop_value, cue_dur_range=(0.5, 0.8), motion_dur=5.0,
            difficulty_level=level_name
        )
        
        # Log trial data
        thisExp.addData('trial_num', global_trial_counter)
        thisExp.addData('participant', expInfo['participant'])
        thisExp.addData('session', expInfo['session'])
        thisExp.addData('phase', 'test')
        thisExp.addData('n_shapes', n_shapes)
        thisExp.addData('block_num', block_num)
        thisExp.addData('difficulty_level', level_name)
        thisExp.addData('prop_used', prop_value)
        thisExp.addData('threshold_75', threshold_75)
        thisExp.addData('level_1_prop', levels['level_1'])
        thisExp.addData('level_2_prop', levels['level_2'])
        thisExp.addData('level_3_prop', levels['level_3'])
        thisExp.addData('level_4_prop', levels['level_4'])
        thisExp.addData('accuracy', res.get('accuracy', 0))
        thisExp.addData('is_timeout', res.get('resp_shape') == 'timeout')
        thisExp.addData('rt_choice', res.get('rt_choice', np.nan))
        thisExp.addData('confidence_rating', res.get('confidence_rating', np.nan))
        thisExp.addData('agency_rating', res.get('agency_rating', np.nan))
        thisExp.addData('early_response', res.get('early_response', False))
        thisExp.addData('true_shape', res.get('true_shape', ''))
        thisExp.addData('resp_shape', res.get('resp_shape', ''))
        thisExp.addData('angle_bias', angle_bias)
        thisExp.addData('applied_angle_bias', res.get('applied_angle_bias', angle_bias))
        # Evidence metrics
        thisExp.addData('mean_evidence', res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence', res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence', res.get('var_evidence', np.nan))
        # Trajectory IDs
        thisExp.addData('target_snippet_id', res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids', str(res.get('distractor_snippet_ids', [])))
        thisExp.nextEntry()
        
        # Show break screen every 50 trials
        if trial_counter % 50 == 0 and trial_counter < total_trials:
            show_break_screen(trial_counter, total_trials, f"Test ({n_shapes}-shape)")


# ───────────────────────────────────────────────────────
#  Initial Instructions
# ───────────────────────────────────────────────────────

def show_initial_instructions():
    instructions = [
        """Welcome to the study.

On each trial you will see moving shapes. Move your mouse as usual and decide which shape you had more control over.

Please follow the on‑screen prompts. The session includes several blocks with quick breaks in between.

Press SPACE to continue...""",

        """Response instructions:

Please respond as quickly and accurately as possible. If you are unsure, make your best guess.

You have up to 5 seconds to respond on each trial.

Press SPACE to continue..."""
    ]
    
    for instruction in instructions:
        msg.text = instruction
        msg.draw()
        win.flip()
        keys = wait_keys(['space', 'escape'])
        if 'escape' in keys:
            _save()
            core.quit()


def show_2shape_instructions():
    msg.text = """BLOCK 1: Two Shapes

In this block, you will see TWO shapes on each trial:
• A Square and a Circle

One of them follows your mouse movement.
Your task is to identify which one.

Response keys:
• Press A for Square
• Press S for Circle

Press SPACE to continue..."""
    msg.draw()
    win.flip()
    wait_keys(['space', 'escape'])


def show_4shape_instructions():
    msg.text = """BLOCK 2: Four Shapes

In this block, you will see FOUR shapes on each trial:
• Square, Circle, Triangle, and Hexagon

One of them follows your mouse movement.
Your task is to identify which one.

Response keys:
• Press A for Square
• Press S for Circle
• Press D for Triangle
• Press F for Hexagon

Press SPACE to continue..."""
    msg.draw()
    win.flip()
    wait_keys(['space', 'escape'])


# ───────────────────────────────────────────────────────
#  Block Runner Functions
# ───────────────────────────────────────────────────────

def run_2shape_block(block_order_num):
    """Run the complete 2-shape block (calibration + test)."""
    global threshold_2shape, staircase_2shape
    
    show_2shape_instructions()
    
    # Calibration
    msg.text = f"Block {block_order_num}A: Practice (2 shapes)\n\nIn this block, you will practice the task.\nYou will receive feedback after each response.\n\nPress SPACE to start."
    msg.draw(); win.flip(); wait_keys()
    
    staircase_2shape, threshold_2shape = run_calibration_staircase(
        num_trials=CHECK_CALIBRATION_TRIALS,
        n_shapes=2,
        angle_bias=0,
        block_num=block_order_num
    )
    
    # Save staircase summary
    expInfo['threshold_75_2shape'] = threshold_2shape
    expInfo['staircase_reversals_2shape'] = staircase_2shape.reversals
    
    # Break
    msg.text = f"""Calibration complete!

Your threshold has been estimated.

You can take a short break now.

Press SPACE to continue to the main experiment."""
    msg.draw(); win.flip(); wait_keys()
    
    # Test
    msg.text = f"""Block {block_order_num}B: Main Experiment (2 shapes)

In this block, you will complete the main task.

After each trial you will:
1. Rate how confident you are in your response
2. Rate how much control you felt over the shape

Try to use the full scales to answer both questions.
No feedback will be shown in this block.

Press SPACE to start."""
    msg.draw(); win.flip(); wait_keys()
    
    run_test_phase(
        threshold_75=threshold_2shape,
        num_trials_per_level=CHECK_TEST_TRIALS_PER_LEVEL,
        n_shapes=2,
        angle_bias=0,
        block_num=block_order_num
    )
    
    return threshold_2shape


def run_4shape_block(block_order_num):
    """Run the complete 4-shape block (calibration + test)."""
    global threshold_4shape, staircase_4shape
    
    show_4shape_instructions()
    
    # Calibration
    msg.text = f"Block {block_order_num}A: Practice (4 shapes)\n\nIn this block, you will practice with 4 shapes.\nYou will receive feedback after each response.\n\nPress SPACE to start."
    msg.draw(); win.flip(); wait_keys()
    
    staircase_4shape, threshold_4shape = run_calibration_staircase(
        num_trials=CHECK_CALIBRATION_TRIALS,
        n_shapes=4,
        angle_bias=0,
        block_num=block_order_num
    )
    
    # Save staircase summary
    expInfo['threshold_75_4shape'] = threshold_4shape
    expInfo['staircase_reversals_4shape'] = staircase_4shape.reversals
    
    # Break
    msg.text = f"""Calibration complete!

Your threshold has been estimated for the 4-shape task.

You can take a short break now.

Press SPACE to continue to the main experiment."""
    msg.draw(); win.flip(); wait_keys()
    
    # Test
    msg.text = f"""Block {block_order_num}B: Main Experiment (4 shapes)

In this block, you will complete the main task with 4 shapes.

After each trial you will:
1. Rate how confident you are in your response
2. Rate how much control you felt over the shape

Try to use the full scales to answer both questions.
No feedback will be shown in this block.

Press SPACE to start."""
    msg.draw(); win.flip(); wait_keys()
    
    run_test_phase(
        threshold_75=threshold_4shape,
        num_trials_per_level=CHECK_TEST_TRIALS_PER_LEVEL,
        n_shapes=4,
        angle_bias=0,
        block_num=block_order_num
    )
    
    return threshold_4shape


# ───────────────────────────────────────────────────────
#  Main Experiment
# ───────────────────────────────────────────────────────

# Initialize threshold variables
threshold_2shape = None
threshold_4shape = None
staircase_2shape = None
staircase_4shape = None

# Show initial instructions
show_initial_instructions()

# Randomize block order based on participant ID (counterbalanced)
# Even participant numbers: 2-shape first, Odd: 4-shape first
try:
    participant_num = int(expInfo["participant"])
except ValueError:
    # If participant ID is not a number, use hash
    participant_num = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16)

start_with_4shapes = (participant_num % 2) == 1
expInfo['block_order'] = '4shape_first' if start_with_4shapes else '2shape_first'

print(f"Participant {expInfo['participant']}: Block order = {expInfo['block_order']}")

if start_with_4shapes:
    # Run 4-shape block first
    run_4shape_block(block_order_num=1)
    
    # Break between blocks
    msg.text = """Block 1 Complete!

You have completed the 4-shape block.

Take a short break. The next block will have 2 shapes.

Press SPACE when you are ready to continue."""
    msg.draw(); win.flip(); wait_keys()
    
    # Run 2-shape block second
    run_2shape_block(block_order_num=2)
else:
    # Run 2-shape block first
    run_2shape_block(block_order_num=1)
    
    # Break between blocks
    msg.text = """Block 1 Complete!

You have completed the 2-shape block.

Take a short break. The next block will have 4 shapes.

Press SPACE when you are ready to continue."""
    msg.draw(); win.flip(); wait_keys()
    
    # Run 4-shape block second
    run_4shape_block(block_order_num=2)

# ═══════════════════════════════════════════════════════
#  END
# ═══════════════════════════════════════════════════════

# Final summary
final_used = len(used_trajectory_indices)
print(f"\nExperiment complete!")
print(f"  Total trajectories used: {final_used}")
print(f"  2-shape threshold (75%): {threshold_2shape:.3f}")
print(f"  4-shape threshold (75%): {threshold_4shape:.3f}")

msg.text = f"""Thank you for participating in the experiment.

Your data have been recorded.

Summary:
• 2-shape threshold: {threshold_2shape:.2f}
• 4-shape threshold: {threshold_4shape:.2f}

Press SPACE to exit."""
msg.draw(); win.flip(); wait_keys()
win.close(); core.quit()
