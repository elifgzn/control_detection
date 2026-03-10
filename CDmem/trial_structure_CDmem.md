# Trial & Block Structure — CDmem (CDmem_1.py)

## Overview

The experiment is divided into 3 phases: **Calibration**, **Test**, and **Memory Test**. Calibration and Test trials all share the same single-trial structure (described below). The Memory Test phase has a different structure.

---

## Single Trial Structure (Calibration & Test)

Each trial proceeds through the following steps in order.

### 1. Fixation cross — **Timed (0.5–0.8 s, random uniform)**
A white fixation cross (`+`) appears at the centre of a grey screen.
- No input required from the participant.
- Duration is randomly sampled on every trial.

### 2. Motion phase — **Timed (3.0 s, no input required)**
Two stimuli appear on screen: one on the left (`−300 px`) and one on the right (`+300 px`) of centre, vertically centred.

- **Calibration phase**: stimuli are a black **Square** (40 × 40 px) and a black **Circle** (radius 20 px).
- **Test phase**: stimuli are two **object images** (200 × 200 px) from the CARA stimulus set.

One stimulus is the **target** (randomly chosen each trial): its direction is a weighted blend of the participant's **mouse direction** and a pre-recorded **trajectory direction**. The blend weight (`prop`) is set by QUEST+ in calibration, or by the fixed difficulty level in the test phase.

The other stimulus (the **distractor**) follows its pre-recorded trajectory fully independently of the mouse.

> **Direction-only mixing**: speed is always equal to the trajectory speed. Only the *direction* of the target is influenced by the mouse. This keeps speed identical for both stimuli, preventing speed from being used as a cue.

- Mouse movement is tracked every frame; if the mouse is stationary (speed < 0.5 px/frame), the target also follows its trajectory autonomously.
- Stimuli are confined within a circle of radius 400 px around the centre to prevent them flying off-screen.
- **No keypress is required during this phase** — it proceeds automatically for exactly 3 seconds.
- EEG triggers are sent at stimulus onset and motion start (test phase only).

### 3. Response — **Self-paced (5 s window)**
After the motion phase ends, a response screen appears.

- **Calibration**: text prompt — *"Which shape did you control? A = Square   S = Circle"*
- **Test**: images are reset to their starting positions; the question *"Which image did you control?"* appears above them. Key labels **A** and **S** appear below the left and right images respectively.

The participant presses **A** or **S** to indicate which stimulus they controlled. A 5-second response window is enforced; if no response is given, a "Please answer faster!" message is shown and the trial is logged as a timeout.

> **Note**: the response is time-limited to 5 s (unlike the photo task, which is fully self-paced). RT and accuracy are logged.

### 4. Feedback — **Timed (0.8 s) — Calibration only**
Immediately after the response, *"Right"* or *"Wrong"* is displayed for 0.8 s.
- Only shown during the calibration phase.
- No feedback is given during the test phase.

### 5. Agency rating — **Self-paced (no time limit) — Test only**
After the response (test phase only), the question appears:
> *"How much control did you feel over the shape's movement?"*

A 7-point labelled scale is shown:

| Key | Label |
|-----|-------|
| 1 | Very weak |
| 2 | Weak |
| 3 | Somewhat weak |
| 4 | Moderate |
| 5 | Somewhat strong |
| 6 | Strong |
| 7 | Very strong |

Participant presses a **number key 1–7**. After keypress, the next trial's fixation cross appears immediately.

---

## Phase 1 — Calibration (Practice)

**Purpose:** Estimate each participant's psychometric function via a QUEST+ Bayesian adaptive staircase, and find the `prop` value at which they can identify the controlled stimulus ~75% of the time.

**Stimuli:** Plain Square and Circle (no images).

**Feedback:** Given after every trial ("Right" / "Wrong").

**No agency rating** is collected.

### Trial count

| Mode | Minimum | Hard cap |
|------|---------|----------|
| Full | 60 | 80 |
| Check mode | 6 | 26 |

QUEST+ stops early once the posterior SD of the threshold (`alpha_sd`) drops below **0.20**, provided the minimum number of trials has been reached.

### Mid-block break
After half the minimum trials, a self-paced break screen is shown. Participant presses **SPACE** to continue.

### After calibration
The QUEST+ posterior is used to derive two difficulty levels for the test phase. A completion screen is shown; participant presses **SPACE** to continue to the test phase.

---

## Phase 2 — Test

**Purpose:** Measure perceived control and sense of agency across two difficulty conditions, using paired object images as stimuli.

**Stimuli:** Two object images per trial (one target, one distractor), drawn from the CARA stimulus set (200 × 200 px).

**No feedback** is given.

**Agency rating** collected after every trial.

### Difficulty levels

Derived from the QUEST+ calibration posterior via `threshold_for_target()`:

| Level | Target accuracy | Description |
|-------|----------------|-------------|
| `level_1` | ~55 % correct | Hardest |
| `level_3` | ~85 % correct | Medium-hard |

### Image stimuli

- 120 unique object-concept pairs are sampled from the `chosen_stimuli` folder at startup (fixed seed = 42 for reproducibility).
- From each pair, one image is assigned to the **test group** (shown during test trials) and the other to the **recognition group** (used as foils in the memory test).
- Images within the test group are randomly paired (two different object concepts per trial) to create 60 simultaneous-display pairs.
- Each image pair is used exactly once — no repeats across the 6 miniblocks.

### Block structure

- Total: **6 miniblocks × 20 trials = 120 test trials** (full mode), or **6 × 5 = 30** in check mode.
- Miniblocks alternate between `level_1` and `level_3`.
- The starting level is **counterbalanced by participant number parity**:
  - **Odd** participant number → starts with `level_1`: `[L1, L3, L1, L3, L1, L3]`
  - **Even** participant number → starts with `level_3`: `[L3, L1, L3, L1, L3, L1]`

### Breaks between miniblocks
After each miniblock (except the last), a self-paced break screen is shown. Participant presses **SPACE** to continue.

### EEG triggers (test phase only)

| Event | Trigger value |
|-------|--------------|
| Stimulus onset | 10 + level index (11 or 13) |
| Motion start | 20 + level index (21 or 23) |
| Response screen onset | 30 + level index (31 or 33) |
| Correct response | 41 |
| Incorrect response | 42 |

---

## Phase 3 — Memory Test

**Purpose:** Assess recognition memory for the object images seen during the test phase.

Two sets of images are presented in a single shuffled sequence:
- **Seen images** — the exact images shown during the test phase (120 items, ground truth = *old*).
- **Foil (unseen) images** — the paired counterparts that were *not* shown during the test phase (120 items, ground truth = *new*).

Total: **240 items**, presented one at a time in a participant-specific random order.

### Instructions (self-paced screen)
Participant reads instructions and presses **SPACE** to begin.

### Per-item trial structure

#### 1. Fixation cross — **Timed (0.5–0.8 s, random uniform)**
Same as in the navigation phases.

#### 2. Recognition judgement — **Self-paced (no time limit)**
A single object image (300 × 300 px) is shown slightly above screen centre. Below it:
> *"Have you seen this object during the experiment before?"*

Key labels: **A = Old**, **S = New**

- Participant presses **A** (seen before) or **S** (not seen before).
- After keypress, the screen clears and the next trial's fixation cross appears immediately.

### EEG triggers (memory test phase)

| Event | Trigger value |
|-------|--------------|
| Seen image onset | 51 |
| Foil image onset | 52 |
| Correct response (Hit or Correct Rejection) | 61 |
| Incorrect response (Miss or False Alarm) | 62 |

### Optional mid-test break
Every **100 items**, a self-paced break screen is shown. Participant presses **SPACE** to continue.

---

## End of Experiment

A completion screen is shown:
> *"Thank you for participating! Your data have been recorded."*

Participant presses **SPACE** to exit. All data (main CSV + kinematics CSV) is saved automatically.

---

## Output Files

| File | Contents |
|------|----------|
| `data/subjects/CDmem_1_<ID>.csv` | Trial-by-trial responses: accuracy, RT, agency ratings, QUEST+ parameters, image filenames, EEG triggers |
| `data/subjects/CDmem_1_<ID>_kinematics.csv` | Frame-by-frame mouse position, shape positions, and per-frame evidence |
| `image_stimuli_log.json` | Record of which images were sampled and assigned to test vs. recognition groups |

---

## Running Modes

| Mode | How to activate | Effect |
|------|----------------|--------|
| Normal | Run `CDmem_1.py` and fill in the dialog | Full trial counts |
| `check_mode` | Tick checkbox in dialog | Minimal trials (6 calibration, 5/miniblock) |
| `simulate` | Tick checkbox in dialog | Virtual mouse and auto-keypresses — no human needed |
| `AUTO_TEST` | `CDT_AUTO_TEST=true python CDmem_1.py` | Skips dialog, runs full simulation |
