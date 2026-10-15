# TFR Statistical Contrasts Guide

This document explains each statistical contrast currently programmed into your Time-Frequency (Alpha/Beta) permutation testing script (`5_TFR_permutation.py`) and maps them directly to your behavioral hypotheses and findings.

---

## 1. Control & Memory Contrasts (The "All Items" Analysis)
*These contrasts evaluate how the difficulty of the motor task (Low vs. High Control) influences subsequent memory encoding.*

### Contrast A: Main Effect of Memory (The Baseline SME)
- **Contrast Name**: `Main_Effect_Memory`
- **Calculation**: `(All Recalled) - (All Not Recalled)` *(Collapsed across High & Low Control)*
  $$\text{Contrast} = \frac{\text{Low}_{\text{rec}} + \text{High}_{\text{rec}}}{2} - \frac{\text{Low}_{\text{not}} + \text{High}_{\text{not}}}{2}$$
- **What it shows**: The classic Subsequent Memory Effect (SME). It identifies the general neural signature (e.g., parietal alpha/beta desynchronization) differentiating successful encoding from forgetting.
- **Behavioral Relevance**: Validates the EEG pipeline and establishes baseline memory-related neural dynamics in this task before examining condition splits.

### Contrast B: Interaction (Memory × Control)
- **Contrast Name**: `Interaction_Memory_x_Control`
- **Calculation**: `(Low Recalled - Low Not Recalled) - (High Recalled - High Not Recalled)`
  $$\text{Contrast} = (\text{Low}_{\text{rec}} - \text{Low}_{\text{not}}) - (\text{High}_{\text{rec}} - \text{High}_{\text{not}})$$
- **What it shows**: Tests whether the neural mechanism of memory encoding differs depending on motor control demands.
- **Behavioral Relevance**: Behaviorally, overall recall was higher in Low Control than in High Control. A significant interaction demonstrates whether high motor/cognitive load fundamentally disrupts or attenuates normal memory-encoding oscillations (e.g., failure to sustain posterior alpha suppression under high tracking difficulty).

### Contrast C: Simple Effect of Memory in Low Control (Old Analysis)
- **Contrast Name**: `OLD_ANALYSIS__Low_Control_Memory_Effect`
- **Calculation**: `Low Recalled - Low Not Recalled`
  $$\text{Contrast} = \text{Low}_{\text{rec}} - \text{Low}_{\text{not}}$$
- **What it shows**: Evaluates the Subsequent Memory Effect strictly when motor control demands are minimal.
- **Behavioral Relevance**: Serves as the "benchmark" SME under low-interference conditions. It reveals the robust memory encoding pattern when attention is free to process stimuli rather than being consumed by tracking.

### Contrast D: Simple Effect of Memory in High Control (Old Analysis)
- **Contrast Name**: `OLD_ANALYSIS__High_Control_Memory_Effect`
- **Calculation**: `High Recalled - High Not Recalled`
  $$\text{Contrast} = \text{High}_{\text{rec}} - \text{High}_{\text{not}}$$
- **What it shows**: Evaluates whether a significant Subsequent Memory Effect is still present when the participant is actively engaged in demanding motor tracking.
- **Behavioral Relevance**: Crucial for determining whether high tracking load abolishes the neural SME entirely, or whether encoding still operates but through an altered or weakened oscillatory pattern.

---

## 2. Detection & Memory Contrasts
*These contrasts investigate the link between conscious perceptual detection during encoding and subsequent long-term memory recognition.*

### Contrast E: Main Effect of Detection
- **Contrast Name**: `Main_Effect_Detection`
- **Calculation**: `(All Detected Trials) - (All Not Detected Trials)`
  $$\text{Contrast} = \frac{\text{Det}_{\text{rec}} + \text{Det}_{\text{not}}}{2} - \frac{\text{NotDet}_{\text{rec}} + \text{NotDet}_{\text{not}}}{2}$$
- **What it shows**: Captures the neural signature of conscious detection/perceptual awareness during the task (e.g., transient sensory-attentional alpha desynchronization).
- **Behavioral Relevance**: Behaviorally, correctly detected items were recognized better. This contrast isolates the neural state associated with conscious detection itself.

### Contrast F: SME within Detected Trials
- **Contrast Name**: `SME_Detected_Trials`
- **Calculation**: `Detected Recalled - Detected Not Recalled`
  $$\text{Contrast} = \text{Det}_{\text{rec}} - \text{Det}_{\text{not}}$$
- **What it shows**: Isolates memory encoding exclusively for items that were consciously detected.
- **Behavioral Relevance**: Tests for encoding variability among items that reached conscious awareness, confirming that detection alone does not guarantee encoding.

### Contrast G: SME within Not Detected Trials
- **Contrast Name**: `SME_Not_Detected_Trials`
- **Calculation**: `Not Detected Recalled - Not Detected Not Recalled`
  $$\text{Contrast} = \text{NotDet}_{\text{rec}} - \text{NotDet}_{\text{not}}$$
- **What it shows**: Tests for unconscious or incidental memory encoding signatures when participants fail to detect the target.
- **Behavioral Relevance**: Evaluates whether subliminal or unattended items still leave a distinct oscillatory memory trace.

### Contrast H: Interaction (Memory × Detection)
- **Contrast Name**: `Interaction_Memory_x_Detection`
- **Calculation**: `(SME for Detected) - (SME for Not Detected)`
  $$\text{Contrast} = (\text{Det}_{\text{rec}} - \text{Det}_{\text{not}}) - (\text{NotDet}_{\text{rec}} - \text{NotDet}_{\text{not}})$$
- **What it shows**: Tests whether conscious detection fundamentally alters the mechanism of memory encoding.
- **Behavioral Relevance**: Directly tests whether detection simply amplifies standard encoding or engages qualitatively distinct oscillatory networks.

---

## 3. Summary of Output Files

When running `5_TFR_permutation.py`, the following files are produced in `H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked`:

| Contrast Name | Statistical Test Type | Output Figure Name |
| :--- | :--- | :--- |
| `Main_Effect_Memory` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_Main_Effect_Memory_TFR.png` |
| `Interaction_Memory_x_Control` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_Interaction_Memory_x_Control_TFR.png` |
| `OLD_ANALYSIS__Low_Control_Memory_Effect` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_OLD_ANALYSIS__Low_Control_Memory_Effect_TFR.png` |
| `OLD_ANALYSIS__High_Control_Memory_Effect` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_OLD_ANALYSIS__High_Control_Memory_Effect_TFR.png` |
| `Main_Effect_Detection` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_Main_Effect_Detection_TFR.png` |
| `SME_Detected_Trials` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_SME_Detected_Trials_TFR.png` |
| `SME_Not_Detected_Trials` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_SME_Not_Detected_Trials_TFR.png` |
| `Interaction_Memory_x_Detection` | 2D Cluster Permutation (Freq $\times$ Time) | `01_tf_permut_Interaction_Memory_x_Detection_TFR.png` |
| All Contrasts Summary | Text Report | `TFR_permutation_statistics_report.txt` |
