# from_15.10 – EEG analysis pipeline (started 15.10.2026)

Everything before this date is archived in `before_15.10/` (scripts) and
`main_data/eeg/before_15.10/` (results). Every new result goes to
`main_data/eeg/from_15.10/`. Nothing in the archive is overwritten.

## What changed (agreed 15.10)

1. **SME** is tested only as: subsequently remembered = the controlled item was later
   recognised, collapsed across High/Low control. All other SME contrasts were removed.
2. **Pooled baseline** for both analyses (High vs Low and SME). The calculation scripts
   save *raw* power; the permutation scripts apply the dB baseline with the switch
   `BASELINE_MODE = 'pooled'` (chosen) or `'percondition'` (the old method).
3. **Movement-onset-locked** analyses for TFR and ERPs (in addition to the old ones):
   - TFR: switch `LOCKING = 'stim'` / `'move'` in the permutation scripts. Movement-locked
     window −0.52 to +3.00 s; baseline = pre-stimulus (−0.5 to −0.2 s) of the same trial;
     trials with movement > 2.25 s after stimulus are excluded from the movement-locked data
     (≈1.7 % of trials).
   - ERP: the ERPs were already movement-locked. New: a second version with a
     pre-stimulus baseline (−200 to 0 ms before stimulus) as a robustness check; the
     pre-movement version (= old analysis) is still computed.
4. **25-trial rule** (≥ 25 trials per condition/cell, before balancing): applied in the TFR
   permutation scripts, printed in the console and written to the report.
5. Frequency range 2–40 Hz (unchanged).

Everything else (filters, ICA, rejection, wavelets, balancing, permutation settings,
ROIs, plots, order analyses, participant exclusions) is unchanged. Each script lists its
changes at the top and marks new or changed lines with `NEW` / `CHANGED`.

## Order of running

| Step | TFR | ERP |
|---|---|---|
| 1–2 | `1_import_raw_ab_whole.py`, `2_remove_noise_ab_whole.py` (**unchanged copies, no need to re-run**) | `1_import_raw.py`, `2_remove_noise.py` (**unchanged copies, no need to re-run**) |
| 3 | `3_TFR_calculation_CTRL.py` and `3_TFR_calculation_SME.py` | `3_ERP_calculation.py` |
| 4 | – (figures are made by step 5) | `4_ERP_graphs.py` (only the folders changed) |
| 5 | `5_TFR_permutation_CTRL.py` and `5_TFR_permutation_SME.py`; set `BASELINE_MODE`, `LOCKING`, `ROI` at the top and run once per combination | `5_ERP_permutation.py` (runs all 4 datasets) |

Output folders of the TFR permutation scripts are named after the settings, e.g.
`main_data/eeg/from_15.10/eeg5_TFR_SME_move_PO_pooled`, so runs never overwrite each other.

## Not ported

`4_TFR_graphs.py`, `6_TFR_topo_HL.py`, `6_TFR_topo_SME.py`, `*_full3D.py`: they read the old
dB files. The permutation scripts already make the condition maps, difference maps, t-maps
and the topoplots of significant clusters.
