# CDmem: Time-Frequency (TFR) Analysis Pipeline, Stimulus-Locked

This README describes the stimulus-locked time-frequency analysis in this
folder: what each script does, which settings are used, and why. Where a
setting was chosen deliberately, the reasoning and the literature it follows
are given. Settings that are simply implemented as they are (e.g. in
preprocessing) are only described, not justified.

The individual scripts contain the same information in their docstrings and
comments; the TFR parameter reasoning is written out in full in both
`3_TFR_calculation_*.py` scripts.

---

## 1. Overview

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `1_import_raw_ab_whole.py` | `main_data\eeg\ERP\eeg1_raweeg` | `main_data\eeg\TFR\eeg2_ica_stimlocked` (epochs + ICA) |
| 2 | `2_remove_noise_ab_whole.py` | `main_data\eeg\TFR\eeg2_ica_stimlocked` | `main_data\eeg\TFR\eeg3_clean_stimlocked` (cleaned epochs) |
| 3a | `3_TFR_calculation_SME_PO.py` | `eeg3_clean_stimlocked` + `main_data\behavioral` | `main_data\eeg\TFR\eeg4_TFR_stimlocked` |
| 3b | `3_TFR_calculation_HL_MF.py` | `eeg3_clean_stimlocked` + `main_data\behavioral` | `main_data\eeg\TFR\eeg4_TFR_stimlocked_HL_MF` |
| 4 | `4_TFR_graphs.py` | `eeg4_TFR_stimlocked` | `main_data\eeg\TFR\eeg5_figures_stimlocked` |
| 5 | `5_TFR_permutation_SME_PO.py`, `5_TFR_permutation_SME_full3D.py` | `eeg4_TFR_stimlocked` | `main_data\eeg\TFR\eeg5_figures_stimlocked_*` |
| 5 | `5_TFR_permutation_HL_MF.py`, `5_TFR_permutation_HL_full3D.py` | `eeg4_TFR_stimlocked_HL_MF` | `main_data\eeg\TFR\eeg5_figures_stimlocked_*` |
| 6 | `6_TFR_topo_SME.py`, `6_TFR_topo_HL.py` | `eeg4_*` | `main_data\eeg\TFR\eeg5_figures_stimlocked_topo_*` |

All EEG data folders from step 2 onwards are under `H:\PHD\control_detection\main_data\eeg\TFR\`.
Behavioural data: `H:\PHD\control_detection\main_data\behavioral`.
Montage: `CACS-64_REF_new.bvef` (actiCAP layout).

Run order: 1 → 2 (two passes, see below) → 3a and 3b → 4, 5, 6.
After any change to the calculation scripts, rerun everything from step 3 on.

Participant list (all scripts): participants 1–50, excluding
1, 2, 3, 5, 11, 24, 26, 28, 43, 45, 46, 47.

---

## 2. Preprocessing (`1_import_raw_ab_whole.py`)

As implemented:

1. Load raw BrainVision data; apply the actiCAP montage (`REF` renamed to `FCz`).
2. Drop bad channels listed per participant (they are interpolated in step 2).
3. Band-pass filter 1–40 Hz, IIR Butterworth, 4th order.
4. Add back the online reference channel (FCz) and re-reference to the average.
5. Epoch around stimulus onset: S 11 = low control, S 13 = high control;
   −2.5 to +5.5 s; no baseline at this stage. Only these two triggers are
   kept, so that `epochs.selection` later corresponds to the trial index in
   the behavioural file.
6. Per-trial encoding duration (S 11/S 13 → S 91/S 92) stored as epoch metadata.
7. Resample to 250 Hz.
8. Fit ICA: Picard algorithm with `ortho=False, extended=False` (equivalent to
   standard Infomax, as FieldTrip's `runica`), `random_state=42`.

The ICA is fitted on these stimulus-locked epochs, so it is a separate
decomposition from the ERP pipeline's ICA (`1_import_raw.py` / `2_remove_noise.py`):
component numbers cannot be copied between the two pipelines.

---

## 3. ICA cleaning (`2_remove_noise_ab_whole.py`)

Two-pass workflow:

- **Pass 1 (inspection):** for participants without an exclusion list, the
  script shows ICA topomaps, component time courses and the raw epochs.
- **Pass 2 (cleaning):** removes the listed components, applies a voltage
  baseline (−200 to 0 ms), rejects epochs exceeding ±100 µV on any channel,
  interpolates the dropped bad channels (spherical splines), and saves the
  cleaned epochs plus an exclusion-rate log (`exclusionrate.json`).

### Component selection: points discussed

- **Eye components:** strong frontal / frontal-pole topography; confirm with
  the time course (large, sharp blink deflections, compared against Fp1/Fp2).
- **Muscle components:** typically (1) a spectrum that is flat or rising above
  ~20 Hz with no alpha peak (the most reliable sign), (2) a small, sharp focus
  at the edge of the head (temporal / neck), (3) fast irregular high-frequency
  bursts in the time course. Check with `ica.plot_properties(...)`.
- **Keep** any component with a clear alpha peak (~10 Hz), even if its map
  looks unusual — it likely carries brain signal relevant to the alpha analysis.
- **Why muscle removal matters here:** muscle activity is strongest from about
  20 Hz up and inflates beta power. Because the task involves motor control,
  muscle tension could also differ between conditions (e.g. high vs. low
  control) and create spurious beta differences.
- **Number of components is not a criterion in itself.** A participant with
  many clear muscle components (e.g. 10) can have all of them removed if each
  is clearly muscle.
- **Check after removal:** compare the participant's average spectrum before
  and after removal — the alpha peak should stay, power above ~20 Hz should
  drop. If the alpha peak shrinks, a brain component was removed.
- **Consistency:** apply the same criteria to every participant and record
  the number of removed components per participant (to report mean and range).
  If data remain contaminated after removal, or many epochs fail the ±100 µV
  criterion, excluding the participant is preferable to removing ever more
  components; the 25-trial rule (section 4.4) catches participants who lose
  too many epochs.
- **Posterior components and dropped channels:** when channels (e.g. O1, Iz)
  were dropped before ICA, the topomap colours at those positions are
  interpolated by the plot, not measured. Do not remove posterior components
  based on the map alone.

### Known issue: variance percentages in the topomap titles

The percentages shown in the topomap titles are computed from the variance of
the component time courses (`ica.get_sources`). These are scaled to roughly
equal variance, so every component shows about 1/(number of components)
(~1.5 %). The numbers do **not** indicate how much of the EEG a component
explains. Correct values would come from
`ica.get_explained_variance_ratio(epochs, components=[i], ch_type='eeg')`.
MNE already orders components by explained variance (ICA000 = largest).
*(Not yet changed in the script.)*


---

## 4. Time-frequency calculation (`3_TFR_calculation_SME_PO.py`, `3_TFR_calculation_HL_MF.py`)

Both scripts use identical TFR settings. Per participant:

### 4.1 Surface Laplacian (current source density)

`mne.preprocessing.compute_current_source_density` with MNE defaults
(m = 4, λ = 1e-5). This is the spherical-spline surface Laplacian of
Perrin et al. (1989), based on the spherical spline theory of Freeden (1984).
It sharpens the spatial picture and reduces blurring of activity across
neighbouring electrodes. (The original FieldTrip pipeline used
`ft_scalpcurrentdensity` with `degree = 10`; MNE uses 50 Legendre terms by
default, so maps may be slightly sharper. Minor difference.)

### 4.2 Behavioural alignment and trigger check

Behavioural trials are aligned to the surviving epochs via `epochs.selection`.
The EEG triggers (S 11 / S 13) are compared with `trigger_stim_onset` in the
behavioural file. On a mismatch the script **continues**, but every problem is
collected and printed in a **TRIGGER-CHECK SUMMARY at the very end of the run**,
with participant numbers, so affected participants can be checked.

### 4.3 TFR parameters and reasoning

| Setting | Value |
|---|---|
| Method | Complex Morlet wavelets (MNE `tfr_morlet`), single trials, power only (no phase) |
| Frequencies | 2–30 Hz, 1 Hz steps (29 frequencies) |
| Cycles | `np.maximum(FREQS * 0.5, 3)` |
| Output time resolution | `DECIM = 10`: 250 Hz → 25 Hz (40 ms steps) |
| Baseline | −500 to −200 ms, dB, per condition |

**Frequency range (2–30 Hz).** Covers theta (~4–7 Hz), alpha (~8–12 Hz) and
beta (~13–30 Hz). 2–30 Hz follows Wu et al. (2025), whose main interest was
alpha/beta. Cheng et al. (2023) analysed theta (4–7 Hz; their main focus) and
also reported alpha (8–12 Hz), low beta (13–17 Hz) and high beta (18–30 Hz)
effects, all inside this range. Nothing above 30 Hz is calculated. Wu et al.
used 2 Hz steps, matching the frequency resolution of a 0.5 s window
(1 / 0.5 s = 2 Hz); 1 Hz steps are used here. Neighbouring bins overlap,
which is harmless.

**Cycles: background.** The number of cycles of a Morlet wavelet sets the
trade-off between time and frequency precision (Cohen, 2014). In MNE,
σ_t = n_cycles / (2πf) and σ_f = f / n_cycles (FWHM = 2.355 × σ). More cycles
give sharper frequency and blurrier time, and vice versa. Wavelets with fewer
than ~3 cycles are no longer well-defined oscillations and have poor
frequency specificity (Cohen, 2014).

**6–30 Hz: 0.5 × f cycles (3 to 15).** With cycles proportional to frequency,
the time window is the same at every frequency (time FWHM ≈ 0.19 s,
frequency FWHM ≈ 4.7 Hz; e.g. 10 Hz is smoothed over ~7.6–12.4 Hz). This
follows the fixed-time-window approach of Wu et al. (2025), who used a 0.5 s
window at all frequencies, which suits alpha/beta, the main focus. It is a
Morlet approximation of that approach, **not** an exact 0.5 s window: a 0.5 s
Hanning window has a time FWHM of ~0.25 s and a frequency FWHM of ~2.9 Hz, so
these estimates are somewhat finer in time and broader in frequency.

**2–5 Hz: minimum of 3 cycles.** 0.5 × f alone would give only 1–2.5 cycles
at 2–5 Hz (at 4 Hz: smoothing over ~1.6–6.4 Hz, well into delta), so the lower
half of theta would be poorly estimated. A 0.5 s window, as in Wu et al., has
the same limitation; it mattered less there because theta was not their focus.
Because theta is also of interest here, these frequencies use 3 cycles, the
setting of Cheng et al. (2023) (Morlet wavelets with 3 cycles, 1–40 Hz,
following Cohen, 2014). With 3 cycles, 4 Hz is smoothed over ~2.4–5.6 Hz
(time FWHM ~0.28 s). This gives the same *number* of cycles as Cheng et al.;
the wavelet is not necessarily identical, because toolboxes (MNE, FieldTrip,
EEGLAB) define and truncate wavelets slightly differently. At 6 Hz both rules
give 3 cycles, so 6–30 Hz (alpha/beta) is unchanged by the minimum.

**Alternatives considered (not used):**

| Option | Effect | Why not |
|---|---|---|
| Fixed 3 cycles at all frequencies (exactly Cheng et al., 2023) | Alpha/beta strongly smeared (10 Hz: ~6–14 Hz; 20 Hz: ~12–28 Hz) | Poor fit for alpha/beta as the main interest |
| Linear 3 → 8 cycles over 2–20 Hz (original FieldTrip script, `A2_3_freqCalc_feedback.m`; common approach, Cohen, 2014) | Slightly sharper theta, practically identical alpha, slightly broader beta | Valid alternative; not chosen, to keep the Wu-like fixed window for alpha/beta |
| 0.67 × f cycles (closer to the time smoothing of a 0.5 s Hanning window) | Coarser timing, sharper frequency in alpha/beta | Not chosen |

**Caveats for interpretation:**
- Theta and low alpha overlap in frequency (e.g. 7 Hz is smoothed over
  ~4.6–9.4 Hz), so theta effects should be described as "around 4–7 Hz", not
  as a sharply separated band. This applies to every setting considered.
- At 2–3 Hz the 3-cycle wavelet is long (time FWHM ~0.37–0.56 s), so power at
  the end of the baseline (−0.2 s) already contains some post-stimulus
  activity. Interpret effects below ~4 Hz with caution.
- Power is total power (evoked + induced). Shortly after stimulus onset,
  theta in particular can be partly driven by the ERP. If relevant, induced
  power can be computed by subtracting each condition's ERP from its trials
  before the wavelet step.

**Output time resolution (DECIM = 10).** 25 Hz (40 ms steps). Power changes
slowly at these time scales (time FWHM ≥ ~0.19 s), so nothing is lost; it
saves memory and disk space. (Cheng et al., 2023, used 50 ms bins.) No edge
effects: the longest wavelet reaches ~±1.2 s, well inside the −2.5 to 5.5 s epoch.

### 4.4 Minimum-trial rule (25 trials per cell)

A participant is excluded **from an analysis** (= one output file) if **any**
cell of that analysis has fewer than 25 trials, counted before balancing.
Example: in a 4-cell analysis, 20 trials in `high_not_recalled` excludes the
participant from all four cells of that analysis, but not from other analyses
(e.g. the main-effect files).

- The previous minimum (2 trials) was far too low for stable time-frequency
  estimates. (For comparison: Cheng et al., 2023, excluded participants with
  fewer than 12 trials in any condition.)
- Old output files of excluded participants are deleted, because the later
  scripts load every file they find.
- Each output file stores the raw trial count per cell (`cell_names`,
  `cell_n_trials`), the balanced count (`n_balanced`) and the rule
  (`min_trials`). Scripts 4–6 check these counts again and skip files that
  fail the rule or come from an older run without counts.
- Included/excluded participant numbers (with the cell counts that were too
  small) are printed at the end of each calculation script and saved to
  `trial_counts_SME.csv` / `trial_counts_HL.csv`. Scripts 4–6 print and report
  the included and skipped participants for every analysis.
- Consequence: N can differ between analyses. Main-effect analyses pool low
  and high control (bigger cells), so fewer participants drop out than in the
  4-cell analyses. Each contrast reports its own N and participant list.

### 4.5 Trial balancing

For each compared pair, the larger cell is randomly subsampled to the size of
the smaller one, 100 times, and the subsample averages are averaged.

Note: for power averaged over trials, the mean of random subsamples is almost
the same as the plain mean of all trials (more trials make an average less
noisy, but do not shift it). The balancing is therefore harmless but changes
little; it would matter more for measures that depend on trial count, such as
phase coherence.

**Reproducibility:** each random generator is seeded from fixed numbers:
`(RANDOM_SEED = 2026, participant number)` in the HL script and
`(RANDOM_SEED = 2026, participant number, analysis)` in the SME script (the
analysis number is derived from the analysis name with `zlib.crc32`).
Results are identical on every run, and a participant's result does not depend
on which other participants or analyses were processed.

### 4.6 Baseline: per condition (dB)

`10 × log10(condition power / mean condition power in −500 to −200 ms)`, per
channel and frequency, computed on each condition's own average (following
Wu et al., 2025).

Consequence: any pre-stimulus power difference between conditions is removed
from the contrast (difference in dB = 10·log10(A/B) − 10·log10(baseline_A /
baseline_B)). This matters especially for:
- high vs. low control, which appear to be run in blocks (the HL statistics
  split participants by starting block), so block-level pre-stimulus state
  differences are plausible;
- theta, which can differ before stimulus onset (anticipation).

Alternatives, both keeping pre-stimulus differences visible:
- One shared ("grand") baseline over all trials: the original FieldTrip
  pipeline (`A2_3_freqCalc_feedback.m`, baseline −300 to −100 ms).
- No frequency-domain baseline: Cheng et al. (2023), who compared raw power
  between conditions and analysed a pre-encoding window (−0.8 to 0 s).

### 4.7 Analyses

**`3_TFR_calculation_SME_PO.py`** (subsequent memory; one file per analysis):

| Analysis | Cells | "Recalled" / "not recalled" | File |
|---|---|---|---|
| Preregistered | low/high control × recalled/not recalled | controlled image recognised / not | `*_TFR_ConditionAverages.npz` |
| Supplementary (all items) | low/high control × recalled/not recalled | either image recognised / neither | `*_TFR_ConditionAverages_AllItems.npz` |
| Main effect, controlled items | recalled vs. not recalled (pooled across control) | controlled image | `*_TFR_MainEffect_ControlledOnly.npz` |
| Main effect, all items | recalled vs. not recalled (pooled across control) | either / neither | `*_TFR_MainEffect_AllItems.npz` |

- **Main effect vs. preregistered:** same trials and same memory rule; the main
  effect pools low and high control into one comparison. Caveat: if
  recognition rates differ between low and high control, the recalled and
  not-recalled groups contain different mixes of the two conditions, and the
  main-effect difference can partly reflect control condition. Worth checking
  that the low/high mix is similar in both groups.
- **Exploratory item-type split** (controlled vs. uncontrolled image, within
  low/high control): **commented out**, produces no output. Its controlled-image
  half duplicates the preregistered analysis. The uncontrolled-image half is
  hard to interpret: each trial shows both images, so one EEG recording has
  two memory outcomes. The same trials enter both comparisons (possibly on
  opposite sides), so the effects are not independent, and the EEG cannot tell
  which image the activity belongs to.
- **Detection × memory:** **removed**, because of too few trials per cell
  (non-detected trials are few, and the "neither image recognised" group is
  the smaller memory group).

**`3_TFR_calculation_HL_MF.py`** (main effect of control): High control (S 13)
vs. low control (S 11), all trials regardless of memory →
`*_TFR_MainEffect_Control.npz`.

---

## 5. Graphs (`4_TFR_graphs.py`)

Uses the preregistered 4-cell file. Only participants that passed the 25-trial
rule **and** have all four cells are used, so all four grand averages are
based on the same participants.
- Heatmaps: 2–30 Hz × time, averaged over posterior ROI channels (P*/O*).
- Topoplots: averaged over 2–30 Hz and 0–3 s.

---

## 6. Statistics (`5_TFR_permutation_*.py`)

Common settings (as implemented): cluster-based permutation tests (MNE;
method of Maris & Oostenveld, 2007), 1000 permutations, two-sided
(`TAIL = 0`), cluster-forming threshold from the t distribution at
α = 0.05 (two-sided), seed 2025. Test window: **2–30 Hz, 0–3 s**; plots also
2–30 Hz.

- **`5_TFR_permutation_SME_PO.py`**: 2D (frequency × time) tests on power
  averaged over posterior ROI channels (P*/O*). Contrasts: main effect of
  memory (controlled items; all items); simple effects of memory within low
  and within high control, and the memory × control interaction — separately
  from the all-items and the preregistered (controlled-items) 4-cell files.
- **`5_TFR_permutation_SME_full3D.py`**: same contrasts, whole-scalp 3D tests
  (channels × frequencies × times), Delaunay spatial adjacency; a cluster point
  needs ≥ 2 adjacent significant neighbours (as FieldTrip `minnbchan`), and
  clusters must span ≥ 3 channels.
- **`5_TFR_permutation_HL_MF.py`**: 2D ROI test of high vs. low control; the
  same test separately for participants who started with the high- vs.
  low-control block; and a between-subjects test of whether the high − low
  difference differs between the two starting orders.
- **`5_TFR_permutation_HL_full3D.py`**: whole-scalp 3D test of high vs. low
  control (same 3D settings as above).

---

## 7. Topoplots (`6_TFR_topo_*.py`)

Grand-average topoplots over 0–3 s for:
- Theta: 4–7 Hz (as in Cheng et al., 2023)
- Alpha: 8–12 Hz
- Beta: 13–30 Hz

Gamma bands were removed, since nothing above 30 Hz is calculated.
`6_TFR_topo_SME.py`: main effects (all items; controlled items) and simple
effects / interaction from the all-items 4-cell file.
`6_TFR_topo_HL.py`: high control, low control, and high − low.

---

### Figure colour scales

Colour limits only affect the figures, not the statistics.
- Power (conditions **and** differences; heatmaps, spectrograms, topoplots in
  `4_`, `5_`, `6_`): fixed ±1.5 dB, as in the original FieldTrip pipeline
  (`A2_4_freqGraphs_feedbacktheta.m`, `A2_5_permuatation_TFmaps_feedbacktheta.m`).
- t-maps (`5_TFR_permutation_SME_PO.py`, `5_TFR_permutation_HL_MF.py`):
  `STAT_LIMITS = None` → symmetric ± largest |t| of each map (the data's own
  range, as the original used `zlim = 'maxmin'`); the scale differs between
  maps, so read each colourbar. Set e.g. `STAT_LIMITS = (-4, 4)` for a fixed
  scale. (The previous fixed ±2.5 cut off strong effects.)
- The cluster topoplots inside the `5_` scripts use their own automatic scale
  (99th percentile of the data, separately for conditions and differences).

---

## 8. Reference approaches compared

| | Wu et al. (2025) | Cheng et al. (2023) | Original FieldTrip script (`A2_3_freqCalc_feedback.m`) | This pipeline |
|---|---|---|---|---|
| Main interest | Alpha/beta | Theta (also alpha/beta effects) | Midfrontal theta (feedback-locked) | Alpha/beta, plus theta |
| Method | Fixed 0.5 s window (FieldTrip-style) | Morlet, 3 cycles | Morlet, 3 → 8 cycles | Morlet, max(0.5 × f, 3) cycles |
| Frequencies | 2–30 Hz, 2 Hz steps | 1–40 Hz, 0.5 Hz steps | 2–20 Hz, 1 Hz steps | 2–30 Hz, 1 Hz steps |
| Baseline | Per condition | None in the frequency domain | One grand baseline over all trials (−300 to −100 ms) | Per condition (−500 to −200 ms) |
| Laplacian | — | No (mastoid reference) | Yes (spline, degree 10) | Yes (MNE defaults) |
| Trial balancing | — | No | No | Yes (100 subsamples) |
| Minimum trials | — | 12 per cell | — | 25 per cell |

(Wu et al. details as described by the researcher; "—" = not discussed.)

Suggested methods wording: *Morlet wavelets with 0.5 × f cycles (constant time
window, following Wu et al., 2025), with a minimum of 3 cycles at low
frequencies (as in Cheng et al., 2023).* Cite Wu et al. for the per-condition
baseline; Cheng et al. used no frequency-domain baseline.

---

## 9. Open points

- **Wu et al. (2025): full reference still to be added** (here and in both
  calculation scripts).
- Variance percentages in the ICA topomap titles are not meaningful (section 3).
- `contrasts_guide.md` is outdated (detection contrasts, old file name); this
  README replaces it.

---

## References

- Cheng, S., Ding, Z., Chen, C., Sun, W., Jiang, T., Liu, X., & Zhang, M. (2023).
  The effect of choice on memory: The role of theta oscillations.
  *Psychophysiology, 60*(12), e14390. https://doi.org/10.1111/psyp.14390
- Cohen, M. X. (2014). *Analyzing neural time series data: Theory and practice.*
  MIT Press. https://doi.org/10.7551/mitpress/9609.001.0001
- Freeden, W. (1984). Spherical spline interpolation — basic theory and
  computational aspects. *Journal of Computational and Applied Mathematics, 11*,
  367–375.
- Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG-
  and MEG-data. *Journal of Neuroscience Methods, 164*(1), 177–190.
  https://doi.org/10.1016/j.jneumeth.2007.03.024
- Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989). Spherical
  splines for scalp potential and current density mapping.
  *Electroencephalography and Clinical Neurophysiology, 72*, 184–187.
- Wu et al. (2025). **[TODO: add full reference]**
