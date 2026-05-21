# CDmem_1 — EEG Trigger Mapping

> **Experiment**: CDmem_1.py (Control Detection & Memory)  
> **Hardware**: BrainProducts TriggerBox Plus via serial (COM6, 2 MHz baud)  
> **Pulse width**: 10 ms (then reset to 0x00)  
> **Flag**: `USE_TRIGGERS` in the script (set `True` for EEG recording)

---

## Test Phase Triggers

### Fixation Onset

Sent when the fixation cross appears at the start of each test trial.

| Trigger | Meaning |
|---------|---------|
| **71** | Fixation onset — **low** control condition |
| **72** | Fixation onset — **high** control condition |

### Stimulus Onset

Sent when the two images first appear on screen (before mouse movement starts).  
Target side is **not yet determined** at this point.

| Trigger | Meaning |
|---------|---------|
| **11** | Stimulus onset — **low** control condition |
| **13** | Stimulus onset — **high** control condition |

### Motion Start

Sent when the motion phase begins (after participant starts moving the mouse).  
Encodes both **condition** and **target side** (which image is controlled).

| Trigger | Condition | Controlled image on... |
|---------|-----------|----------------------|
| **21** | Low control | **Left** side |
| **22** | Low control | **Right** side |
| **23** | High control | **Left** side |
| **24** | High control | **Right** side |

### Response Screen Onset

Sent when the choice screen appears ("Which image did you control?").

| Trigger | Condition | Controlled image on... |
|---------|-----------|----------------------|
| **31** | Low control | **Left** side |
| **32** | Low control | **Right** side |
| **33** | High control | **Left** side |
| **34** | High control | **Right** side |

### Detection Response

Sent immediately after the participant presses A or S.

| Trigger | Meaning |
|---------|---------|
| **41** | Response was **correct** (detected the controlled image) |
| **42** | Response was **incorrect** (chose the wrong image) |


---

## Recognition (Memory Test) Phase Triggers

### Fixation Onset

Sent when the fixation cross appears before each recognition item.  
Encodes the **upcoming item's** ground truth, item type, and condition.

| Trigger | Ground Truth | Item Type | Control Condition |
|---------|-------------|-----------|------------------|
| **81** | Seen | Controlled | Low |
| **82** | Seen | Controlled | High |
| **83** | Seen | Uncontrolled | Low |
| **84** | Seen | Uncontrolled | High |
| **85** | Unseen (foil) | — | — |

### Image Onset

Sent when the recognition image appears on screen.  
Same encoding as fixation triggers.

| Trigger | Ground Truth | Item Type | Control Condition |
|---------|-------------|-----------|------------------|
| **51** | Seen | Controlled | Low |
| **52** | Seen | Controlled | High |
| **53** | Seen | Uncontrolled | Low |
| **54** | Seen | Uncontrolled | High |
| **55** | Unseen (foil) | — | — |

### Recognition Response

Sent immediately after the participant presses Y (yes) or N (no).

| Trigger | Meaning |
|---------|---------|
| **61** | Response was **correct** (Hit or Correct Rejection) |
| **62** | Response was **incorrect** (Miss or False Alarm) |

---

## Trial Counts (Full Mode)

| Phase | Trials |
|-------|--------|
| Calibration (practice) | 2 × 60 = 120 (no triggers sent) |
| Test phase | 6 miniblocks × 20 = 120 trials |
| Recognition test | 240 seen + 240 unseen = 480 items |

### Recognition items per trigger:

- 51/81 (seen, controlled, low): **60 items**
- 52/82 (seen, controlled, high): **60 items**
- 53/83 (seen, uncontrolled, low): **60 items**
- 54/84 (seen, uncontrolled, high): **60 items**
- 55/85 (unseen foils): **240 items**

---

## Quick Reference — All Trigger Values

```
TEST PHASE                          RECOGNITION PHASE
─────────────────────────           ─────────────────────────
71 = Fix, Low                       81 = Fix, Seen/Ctrl/Low
72 = Fix, High                      82 = Fix, Seen/Ctrl/High
                                    83 = Fix, Seen/Unctrl/Low
11 = StimOn, Low                    84 = Fix, Seen/Unctrl/High
13 = StimOn, High                   85 = Fix, Unseen

21 = MotionStart, Low/Left          51 = ImgOn, Seen/Ctrl/Low
22 = MotionStart, Low/Right         52 = ImgOn, Seen/Ctrl/High
23 = MotionStart, High/Left         53 = ImgOn, Seen/Unctrl/Low
24 = MotionStart, High/Right        54 = ImgOn, Seen/Unctrl/High
                                    55 = ImgOn, Unseen
31 = RespScreen, Low/Left
32 = RespScreen, Low/Right          61 = MemResp, Correct
33 = RespScreen, High/Left          62 = MemResp, Incorrect
34 = RespScreen, High/Right

41 = DetResp, Correct
42 = DetResp, Incorrect
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Low control** | Prop calibrated to ~55% detection accuracy (hard) |
| **High control** | Prop calibrated to ~75% detection accuracy (easy) |
| **Controlled** | The image whose movement was partially driven by the mouse |
| **Uncontrolled** | The image that moved along a pre-recorded trajectory |
| **Seen** | Image that appeared during the test phase (old item) |
| **Unseen** | Foil image never shown during test phase (new item) |
| **Hit** | Correctly identifying a seen item as "yes" |
| **CR** | Correctly rejecting an unseen item as "no" |
| **Miss** | Failing to recognize a seen item |
| **FA** | Falsely claiming to have seen an unseen item |
