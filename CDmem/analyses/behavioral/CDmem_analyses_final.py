#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#hi 

import os
import sys
import io
import re
import glob
import numpy as np
import pandas as pd
import polars as pl
import pingouin as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm, ttest_ind
from pymer4.models import glmer, lmer

# Setup for rpy2/pymer4 on Windows
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION & OUTPUT DIRECTORIES
# ============================================================================
DATA_DIR = Path(r"H:\PHD\control_detection\main_data\behavioral")
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "CDmem_final_output"
POOLED_DIR = OUTPUT_DIR / "pooled"
PER_PARTICIPANT_DIR = OUTPUT_DIR / "per_participant"

for d in [OUTPUT_DIR, POOLED_DIR, PER_PARTICIPANT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "Comprehensive_Report.md"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# CDmem Comprehensive Analysis Report\n\n")

def write_report(text):
    print(text)
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

PARTICIPANT_FILTER = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]
# [7]
TIMEOUT_THRESHOLD = 0.50
ACCURACY_SD_THRESHOLD = 2.5
RT_SD_MULTIPLIER = 3.0

# ============================================================================
# 1. DATA LOADING
# ============================================================================
write_report("## Data Loading & Preprocessing\n")

# Load main experimental data
# Only load files whose participant number is in the PARTICIPANT_FILTER to avoid
# parsing errors on irrelevant files (e.g. CDmem_1_5.csv for participant 5).
main_files = [f for f in glob.glob(str(DATA_DIR / "CDmem_*.csv")) if re.match(r"^CDmem_1_\d+(?:_\d+)?\.csv$", os.path.basename(f))]
dfs = []
for f in main_files:
    # Extract participant number from filename: CDmem_1_<px>.csv or CDmem_1_<px>_<session>.csv
    basename = os.path.basename(f)
    px_match = re.match(r"^CDmem_1_(\d+)", basename)
    if px_match and PARTICIPANT_FILTER:
        px_num = int(px_match.group(1))
        if px_num not in PARTICIPANT_FILTER:
            continue  # Skip files for participants not in the filter
    try:
        df = pd.read_csv(f)
        df["source_file"] = basename
        dfs.append(df)
    except Exception as e:
        write_report(f"[WARNING] Could not load {basename}: {e}")

data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Apply participant filter
if PARTICIPANT_FILTER and not data.empty:
    data = data[data['participant'].isin(PARTICIPANT_FILTER)].copy()

# Load recognition data (same participant filtering logic)
recog_files = [f for f in glob.glob(str(DATA_DIR / "CDmem_*_recognition.csv")) if re.match(r"^CDmem_1_\d+(?:_\d+)?_recognition\.csv$", os.path.basename(f))]
rdfs = []
for f in recog_files:
    basename = os.path.basename(f)
    px_match = re.match(r"^CDmem_1_(\d+)", basename)
    if px_match and PARTICIPANT_FILTER:
        px_num = int(px_match.group(1))
        if px_num not in PARTICIPANT_FILTER:
            continue
    try:
        df = pd.read_csv(f)
        df["source_file"] = basename
        rdfs.append(df)
    except Exception as e:
        write_report(f"[WARNING] Could not load {basename}: {e}")
recog_data = pd.concat(rdfs, ignore_index=True) if rdfs else pd.DataFrame()

if PARTICIPANT_FILTER and not recog_data.empty:
    recog_data = recog_data[recog_data['participant'].isin(PARTICIPANT_FILTER)].copy()

# ============================================================================
# 2. EXCLUSION CRITERIA
# ============================================================================
# Keep a copy of the full dataset BEFORE exclusions for sanity-check plots
data_all = data.copy()

# Criterion 1: Timeout Rate (>= 50% in any condition)
test_data = data[data["phase"] == "test"].copy()
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"
timeout_stats = test_data.groupby(["participant", "control_condition"])["is_timeout"].agg(total="count", timeouts="sum").reset_index()
timeout_stats["rate"] = timeout_stats["timeouts"] / timeout_stats["total"]
excluded_timeout = timeout_stats[timeout_stats["rate"] >= TIMEOUT_THRESHOLD]["participant"].unique().tolist()
data = data[~data["participant"].isin(excluded_timeout)].copy()

# Criterion 2: Accuracy Outliers (> 2.5 SD from group mean)
test_data = data[data["phase"] == "test"].copy()
acc_per_px = test_data.groupby(["participant", "control_condition"])["detection_accuracy"].mean().reset_index()
group_stats = acc_per_px.groupby("control_condition")["detection_accuracy"].agg(group_mean="mean", group_std="std").reset_index()
acc_per_px = acc_per_px.merge(group_stats, on="control_condition")
acc_per_px["z"] = (acc_per_px["detection_accuracy"] - acc_per_px["group_mean"]) / acc_per_px["group_std"]
excluded_acc = acc_per_px[acc_per_px["z"].abs() > ACCURACY_SD_THRESHOLD]["participant"].unique().tolist()
data = data[~data["participant"].isin(excluded_acc)].copy()

# Criterion 3: Calibration failure
# Exclude participants for whom neither QUEST+ staircase converged (posterior SD >= 0.20 after max trials)
calib_data = data[data["phase"] == "calibration"].copy()
calib_data['quest_alpha_sd'] = pd.to_numeric(calib_data['quest_alpha_sd'], errors='coerce')
excluded_calib = []
for px, px_df in calib_data.groupby("participant"):
    converged_any = False
    for target, target_df in px_df.groupby("calib_target"):
        if not target_df.empty:
            final_alpha_sd = target_df["quest_alpha_sd"].iloc[-1]
            if not pd.isna(final_alpha_sd) and final_alpha_sd < 0.20:
                converged_any = True
    if not converged_any:
        excluded_calib.append(px)
data = data[~data["participant"].isin(excluded_calib)].copy()

# Criterion 5: Manipulation failure (NOT preregistered — post-hoc data-driven)
# Ensures the control manipulation worked at the individual participant level.
#
# Logic:
#   (A) Always exclude if both accuracy diff (high-low) <= 0 AND agency diff <= 0.
#       This is a clear manipulation failure regardless of calibration quality.
#   (B) If both QUEST+ staircases converged (final alpha_sd < 0.20) and diffs are in the
#       expected direction -> pass without further tests.
#   (C) If only one staircase converged -> run per-participant independent t-tests on
#       trial-level accuracy and agency (high vs low). Exclude if EITHER test is
#       non-significant (p >= .05) or in the wrong direction (OR logic).
test_data_manip = data[data["phase"] == "test"].copy()
test_data_manip["detection_accuracy"] = pd.to_numeric(test_data_manip["detection_accuracy"], errors="coerce")
test_data_manip["agency_rating"] = pd.to_numeric(test_data_manip["agency_rating"], errors="coerce")

# Determine per-participant convergence count (how many staircases converged)
calib_for_manip = data[data["phase"] == "calibration"].copy()
calib_for_manip['quest_alpha_sd'] = pd.to_numeric(calib_for_manip['quest_alpha_sd'], errors='coerce')
px_convergence = {}
for px, px_df in calib_for_manip.groupby("participant"):
    n_converged = 0
    for _, target_df in px_df.groupby("calib_target"):
        if not target_df.empty:
            final_sd = target_df["quest_alpha_sd"].iloc[-1]
            if not pd.isna(final_sd) and final_sd < 0.20:
                n_converged += 1
    px_convergence[px] = n_converged

excluded_manip = []
manip_details = []
for px in sorted(data["participant"].unique()):
    px_test = test_data_manip[test_data_manip["participant"] == px]
    high = px_test[px_test["control_condition"] == "high"]
    low = px_test[px_test["control_condition"] == "low"]

    if len(high) == 0 or len(low) == 0:
        continue

    acc_diff = high["detection_accuracy"].mean() - low["detection_accuracy"].mean()
    ag_diff = high["agency_rating"].mean() - low["agency_rating"].mean()
    both_converged = px_convergence.get(px, 0) == 2

    # (A) Both diffs in wrong direction -> always exclude
    if acc_diff <= 0 and ag_diff <= 0:
        excluded_manip.append(px)
        manip_details.append(f"  P{int(px)}: EXCLUDED -- both diffs wrong direction "
                             f"(acc={acc_diff:+.3f}, agency={ag_diff:+.3f})")
        continue

    # (B) Both staircases converged and diffs OK -> pass (no report line needed)
    if both_converged:
        continue

    # (C) Partial convergence -> independent-samples t-tests (high vs low)
    t_acc, p_acc = ttest_ind(high["detection_accuracy"].dropna(), low["detection_accuracy"].dropna())
    t_ag, p_ag = ttest_ind(high["agency_rating"].dropna(), low["agency_rating"].dropna())
    df_acc = len(high["detection_accuracy"].dropna()) + len(low["detection_accuracy"].dropna()) - 2
    df_ag = len(high["agency_rating"].dropna()) + len(low["agency_rating"].dropna()) - 2

    acc_ok = acc_diff > 0 and p_acc < 0.05
    ag_ok = ag_diff > 0 and p_ag < 0.05

    if not acc_ok or not ag_ok:
        excluded_manip.append(px)
        # Build per-variable diagnostic
        failures = []
        if not acc_ok:
            if acc_diff <= 0:
                failures.append(f"accuracy wrong direction (diff={acc_diff:+.3f})")
            else:
                failures.append(f"accuracy t-test n.s. (diff={acc_diff:+.3f}, t({df_acc})={t_acc:.3f}, p={p_acc:.4f})")
        if not ag_ok:
            if ag_diff <= 0:
                failures.append(f"agency wrong direction (diff={ag_diff:+.3f})")
            else:
                failures.append(f"agency t-test n.s. (diff={ag_diff:+.3f}, t({df_ag})={t_ag:.3f}, p={p_ag:.4f})")
        manip_details.append(f"  P{int(px)}: EXCLUDED -- partial convergence; {'; '.join(failures)}")

data = data[~data["participant"].isin(excluded_manip)].copy()

# Save pre-exclusion recog data for per-participant plots
recog_data_all = recog_data.copy()

# Synchronize participants between both datasets
valid_px = set(data["participant"].unique()) & set(recog_data["participant"].unique())
data = data[data["participant"].isin(valid_px)].copy()
recog_data = recog_data[recog_data["participant"].isin(valid_px)].copy()

# Recognition RT Outliers (Criterion 4)
def _apply_rt_outlier_removal(rdf, multiplier):
    """Apply per-participant RT outlier removal and return cleaned DataFrame + count removed."""
    rdf = rdf.copy()
    rdf["mem_rt"] = pd.to_numeric(rdf["mem_rt"], errors="coerce")
    px_rt = rdf.groupby("participant")["mem_rt"].agg(mean_rt="mean", std_rt="std").reset_index()
    rdf = rdf.merge(px_rt, on="participant")
    rdf["rt_thresh"] = rdf["mean_rt"] + multiplier * rdf["std_rt"]
    mask = (rdf["mem_rt"] > rdf["rt_thresh"]) | rdf["mem_rt"].isna()
    return rdf[~mask].drop(columns=["mean_rt", "std_rt", "rt_thresh"]).copy(), mask.sum()

recog_data, rt_removed = _apply_rt_outlier_removal(recog_data, RT_SD_MULTIPLIER)
recog_data_all, _ = _apply_rt_outlier_removal(recog_data_all, RT_SD_MULTIPLIER)

write_report(f"- Excluded for timeout: {excluded_timeout}")
write_report(f"- Excluded for accuracy outliers: {excluded_acc}")
write_report(f"- Excluded for calibration failure: {excluded_calib}")
write_report(f"- Excluded for manipulation failure (not preregistered): {excluded_manip}")
for detail in manip_details:
    write_report(detail)
write_report(f"- Total recognition trials excluded for RT outliers: {rt_removed}")
write_report(f"- Final N participants: {len(valid_px)}\n")


# Memory floor/ceiling: Exclude participants whose overall recognition d' < 0.10
# or who respond "yes" to > 95% or < 5% of all items (extreme bias).
#
# recog_data['said_old'] = recog_data['mem_response'].str.lower() == 'yes'
# def calc_dprime_temp(hr, far, clip_val=0.01):
#     hr_clipped = np.clip(hr, clip_val, 1.0 - clip_val)
#     far_clipped = np.clip(far, clip_val, 1.0 - clip_val)
#     return norm.ppf(hr_clipped) - norm.ppf(far_clipped)
#
# mem_summary = recog_data.groupby('participant').apply(lambda x: pd.Series({
#     'yes_rate': x['said_old'].mean(),
#     'hr': x[x['mem_ground_truth']=='seen']['said_old'].mean(),
#     'far': x[x['mem_ground_truth']=='unseen']['said_old'].mean()
# })).reset_index()
# mem_summary['dprime'] = mem_summary.apply(lambda r: calc_dprime_temp(r['hr'], r['far']), axis=1)
# excluded_mem = mem_summary[(mem_summary['dprime'] < 0.10) | (mem_summary['yes_rate'] > 0.95) | (mem_summary['yes_rate'] < 0.05)]['participant'].tolist()
# data = data[~data['participant'].isin(excluded_mem)]
# recog_data = recog_data[~recog_data['participant'].isin(excluded_mem)]


# ============================================================================
# 3. VARIABLE DERIVATION & SANITY CHECKS
# ============================================================================
write_report("## Sanity Checks\n")
test_data = data[data["phase"] == "test"].copy()

test_data["detection_accuracy"] = pd.to_numeric(test_data["detection_accuracy"], errors="coerce")
test_data["agency_rating"] = pd.to_numeric(test_data["agency_rating"], errors="coerce")

# Agency Rating Sanity Check
agency_px = test_data.groupby(["participant", "control_condition"])["agency_rating"].mean().reset_index()
agency_pivot = agency_px.pivot(index="participant", columns="control_condition", values="agency_rating").dropna()
if len(agency_pivot) >= 2:
    res_agency = pg.ttest(agency_pivot["high"], agency_pivot["low"], paired=True)
    write_report("**Sanity Check: Agency Ratings (High vs Low)**")
    write_report("We expect agency ratings to be significantly higher in the 'high' control condition.")
    write_report(f"- High Mean (SD): {agency_pivot['high'].mean():.3f} ({agency_pivot['high'].std():.3f})")
    write_report(f"- Low Mean (SD): {agency_pivot['low'].mean():.3f} ({agency_pivot['low'].std():.3f})")
    write_report(f"- Result: *t*({res_agency['dof'].iloc[0]}) = {res_agency['T'].iloc[0]:.3f}, *p* = {res_agency['p-val'].iloc[0]:.4f}, Cohen's *d* = {res_agency['cohen-d'].iloc[0]:.3f}\n")

# Detection Accuracy Sanity Check
det_px = test_data.groupby(["participant", "control_condition"])["detection_accuracy"].mean().reset_index()
det_pivot = det_px.pivot(index="participant", columns="control_condition", values="detection_accuracy").dropna()
if len(det_pivot) >= 2:
    res_det = pg.ttest(det_pivot["high"], det_pivot["low"], paired=True)
    write_report("**Sanity Check: Detection Accuracy (High vs Low)**")
    write_report("We expect detection accuracy to be significantly higher in the 'high' control condition.")
    write_report(f"- High Mean (SD): {det_pivot['high'].mean():.3f} ({det_pivot['high'].std():.3f})")
    write_report(f"- Low Mean (SD): {det_pivot['low'].mean():.3f} ({det_pivot['low'].std():.3f})")
    write_report(f"- Result: *t*({res_det['dof'].iloc[0]}) = {res_det['T'].iloc[0]:.3f}, *p* = {res_det['p-val'].iloc[0]:.4f}, Cohen's *d* = {res_det['cohen-d'].iloc[0]:.3f}\n")


# ---------------------------------------------------------------------------
# Helper: derive recognition variables from a (main_data, recog_data) pair.
# This is called once with post-exclusion data (for statistics & pooled plots)
# and once with pre-exclusion data (for per-participant plots).
# ---------------------------------------------------------------------------
def calc_dprime(hr, far, clip_val=0.01):
    hr_clipped = np.clip(hr, clip_val, 1.0 - clip_val)
    far_clipped = np.clip(far, clip_val, 1.0 - clip_val)
    return norm.ppf(hr_clipped) - norm.ppf(far_clipped)

def derive_recognition_vars(main_df, recog_df):
    """Derive targets, foils, mem_results_2x2, and fa_rates from raw data."""
    td = main_df[main_df['phase'] == 'test'].copy()
    td['detection_accuracy'] = pd.to_numeric(td['detection_accuracy'], errors='coerce')
    td['agency_rating'] = pd.to_numeric(td['agency_rating'], errors='coerce')

    rd = recog_df.copy()
    rd['said_old'] = rd['mem_response'].str.lower() == 'yes'
    rd['said_old_int'] = rd['said_old'].astype(int)

    tgt = rd[rd['mem_ground_truth'] == 'seen'].copy()
    fl  = rd[rd['mem_ground_truth'] == 'unseen'].copy()

    # Build lookup: (participant, mem_filename) -> agency_rating, detection_accuracy
    img_lk = []
    for _, row in td.iterrows():
        px = row['participant']
        ctrl_img = row['true_controlled']
        ag = row.get('agency_rating', np.nan)
        da = row.get('detection_accuracy', np.nan)
        ctrl_fn   = row['img_A_name'] if ctrl_img == 'img_A' else row['img_B_name']
        unctrl_fn = row['img_B_name'] if ctrl_img == 'img_A' else row['img_A_name']
        img_lk.append({'participant': px, 'mem_filename': ctrl_fn,   'agency_rating': ag, 'detection_accuracy': da})
        img_lk.append({'participant': px, 'mem_filename': unctrl_fn, 'agency_rating': ag, 'detection_accuracy': da})

    if img_lk:
        img_lookup = pd.DataFrame(img_lk).dropna(subset=['mem_filename']).drop_duplicates(subset=['participant', 'mem_filename'])
    else:
        img_lookup = pd.DataFrame(columns=['participant', 'mem_filename', 'agency_rating', 'detection_accuracy'])

    has_tracking = 'item_type' in tgt.columns and 'trial_level' in tgt.columns
    if has_tracking:
        for col in ['agency_rating', 'detection_accuracy']:
            if col in tgt.columns:
                tgt = tgt.drop(columns=[col])
        tgt = tgt.merge(
            img_lookup[['participant', 'mem_filename', 'agency_rating', 'detection_accuracy']],
            on=['participant', 'mem_filename'], how='left'
        )
        tgt = tgt.rename(columns={'trial_level': 'control_level'})
    else:
        img_lk_full = []
        for _, row in td.iterrows():
            px = row['participant']; cond = row['control_condition']; ctrl_img = row['true_controlled']
            ag = row.get('agency_rating', np.nan); da = row.get('detection_accuracy', np.nan)
            ctrl_fn   = row['img_A_name'] if ctrl_img == 'img_A' else row['img_B_name']
            unctrl_fn = row['img_B_name'] if ctrl_img == 'img_A' else row['img_A_name']
            img_lk_full.append({'participant': px, 'mem_filename': ctrl_fn,   'control_level': cond, 'item_type': 'controlled',   'agency_rating': ag, 'detection_accuracy': da})
            img_lk_full.append({'participant': px, 'mem_filename': unctrl_fn, 'control_level': cond, 'item_type': 'uncontrolled', 'agency_rating': ag, 'detection_accuracy': da})
        img_lookup_full = pd.DataFrame(img_lk_full).dropna(subset=['mem_filename']).drop_duplicates(subset=['participant', 'mem_filename'])
        tgt = tgt.merge(img_lookup_full, on=['participant', 'mem_filename'], how='inner')

    # Contrast coding
    tgt['control_level_c'] = tgt['control_level'].map({'high': 0.5, 'low': -0.5})
    tgt['item_type_c'] = tgt['item_type'].map({'controlled': 0.5, 'uncontrolled': -0.5})
    tgt['detection_accuracy_c'] = tgt['detection_accuracy'].map({1: 0.5, 0: -0.5})
    tgt['agency_rating'] = pd.to_numeric(tgt['agency_rating'], errors='coerce')
    tgt['agency_z'] = tgt.groupby('participant')['agency_rating'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    tgt['log_mem_rt'] = np.log(tgt['mem_rt'])

    # d' summary
    fa_r = fl.groupby('participant')['said_old'].mean().reset_index().rename(columns={'said_old': 'FA_rate'})
    m2x2 = tgt.groupby(['participant', 'control_level', 'item_type'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
    m2x2 = m2x2.merge(fa_r, on='participant')
    m2x2['d_prime'] = m2x2.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)

    return tgt, fl, fa_r, m2x2

# --- Derive for post-exclusion data (used in statistics & pooled plots) ---
targets, foils, fa_rates, mem_results_2x2 = derive_recognition_vars(data, recog_data)

# --- Derive for ALL participants (used in per-participant plots) ---
targets_all, foils_all, fa_rates_all, mem_results_2x2_all = derive_recognition_vars(data_all, recog_data_all)


# Helper function to fit and report mixed models
# 
# In pymer4 >= 0.8, the API changed:
#   - model.coefs      -> model.result_fit  (a Polars DataFrame)
#   - Column names:  term, estimate, std_error, conf_low, conf_high, z_stat (glmer) / t_stat (lmer), df, p_value
#
# This function fits the model, prints the result_fit table, and returns it.
# The stat_col will be 'z_stat' for GLMMs and 't_stat' for LMMs.
def fit_print_lmm(formula, df, family="binomial", is_glmer=True):
    try:
        model = glmer(formula, data=pl.from_pandas(df), family=family) if is_glmer else lmer(formula, data=pl.from_pandas(df))
        
        # Suppress R warnings/printouts to keep the console clean
        import rpy2.rinterface_lib.callbacks as _rpy2_cb
        _orig_print, _orig_warn = _rpy2_cb.consolewrite_print, _rpy2_cb.consolewrite_warnerror
        _rpy2_cb.consolewrite_print = _rpy2_cb.consolewrite_warnerror = lambda x: None
        try:
            model.fit()
        finally:
            _rpy2_cb.consolewrite_print, _rpy2_cb.consolewrite_warnerror = _orig_print, _orig_warn
        
        # Access the coefficients via model.result_fit (Polars DataFrame)
        result_fit = model.result_fit
        if result_fit is not None and len(result_fit) > 0:
            write_report("**Fixed Effects:**")
            write_report(str(result_fit) + "\n")
        else:
            write_report("Model fitted but no coefficients returned.\n")
            return None

        # --- Random Effects ---
        # ranef_var: variance-covariance components for each random-effects grouping factor
        # ranef: cluster-level deviations (BLUPs minus fixed effects) per participant
        ranef_var = getattr(model, 'ranef_var', None)
        ranef = getattr(model, 'ranef', None)

        write_report("**Random Effects Variance Components (ranef_var):**")
        if ranef_var is not None:
            # ranef_var can be a dict of DataFrames (one per grouping factor) or a single DataFrame
            if isinstance(ranef_var, dict):
                for grp_name, grp_df in ranef_var.items():
                    write_report(f"  Group: {grp_name}")
                    write_report(str(grp_df) + "\n")
            else:
                write_report(str(ranef_var) + "\n")
        else:
            write_report("  (not available)\n")

        write_report("**Random Effects by Participant (ranef):**")
        if ranef is not None:
            if isinstance(ranef, dict):
                for grp_name, grp_df in ranef.items():
                    write_report(f"  Group: {grp_name}")
                    write_report(str(grp_df) + "\n")
            else:
                write_report(str(ranef) + "\n")
        else:
            write_report("  (not available)\n")

        return result_fit
    except Exception as e:
        write_report(f"Model failed: {e}\n")
        return None

# ============================================================================
# 4. ANALYSES
# ============================================================================
write_report("## ANALYSES\n")

# --- 1) DOES HIGHER MOTOR CONTROL LEAD TO BETTER MEMORY? ---
write_report("### 1) Does higher motor control lead to better memory?\n")

## 1A) Descriptive d' summary: confirm all participants perform above chance
# (The 2x2 RM-ANOVA on d' was removed; d' is used descriptively only.)
write_report("**1A) Descriptive d' Summary (above-chance check)**")
overall_dprime = targets.groupby('participant')['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
overall_dprime = overall_dprime.merge(fa_rates, on='participant')
overall_dprime['d_prime'] = overall_dprime.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)

write_report("**Participant-level d' values:**")
for _, row in overall_dprime.iterrows():
    write_report(f"  - P{int(row['participant'])}: d' = {row['d_prime']:.3f}")
write_report(f"\n- Mean d' = {overall_dprime['d_prime'].mean():.3f} (SD = {overall_dprime['d_prime'].std():.3f})")
write_report(f"- All above chance (d' > 0): {(overall_dprime['d_prime'] > 0).all()}\n")

# One-sample t-test: d' > 0?
if len(overall_dprime) >= 2:
    res_dprime = pg.ttest(overall_dprime['d_prime'], 0)
    write_report(f"- One-sample t-test against chance: t({res_dprime['dof'].iloc[0]}) = {res_dprime['T'].iloc[0]:.3f}, p = {res_dprime['p-val'].iloc[0]:.4f}\n")

# 1B) Binomial GLMM
write_report("**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**")
df_1b = targets.dropna(subset=['said_old_int', 'control_level_c', 'item_type_c']).copy()
fit_print_lmm("said_old_int ~ control_level_c * item_type_c + (1 | participant)", df_1b, is_glmer=True)

# 1C) Gaussian LMM
write_report("**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**")
df_1c = targets[targets['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'control_level_c', 'item_type_c']).copy()
fit_print_lmm("log_mem_rt ~ control_level_c * item_type_c + (1 | participant)", df_1c, is_glmer=False)


# --- 2) DOES THIS MEMORY EFFECT DEPEND ON THE CONSCIOUS DETECTION OF CONTROL? ---
write_report("### 2) Does this memory effect depend on the conscious detection of control?\n")
write_report("*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*\n")
df_controlled = targets[targets['item_type'] == 'controlled'].copy()

# 2D) Binomial GLMM
write_report("**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**")
df_2d = df_controlled.dropna(subset=['said_old_int', 'detection_accuracy_c', 'control_level_c']).copy()
fit_print_lmm("said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2d, is_glmer=True)

# 2E) Gaussian LMM
write_report("**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**")
df_2e = df_controlled[df_controlled['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'detection_accuracy_c', 'control_level_c']).copy()
fit_print_lmm("log_mem_rt ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2e, is_glmer=False)


# --- 3) CAN WE PREDICT THIS EFFECT FROM PARTICIPANTS' SUBJECTIVE AGENCY RATINGS? ---
write_report("### 3) Can we predict this effect from participants' subjective agency ratings?\n")
write_report("*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*\n")

# 3F) Binomial GLMM
write_report("**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**")
df_3f = df_controlled.dropna(subset=['said_old_int', 'agency_z', 'control_level_c']).copy()
fit_print_lmm("said_old_int ~ agency_z * control_level_c + (1 | participant)", df_3f, is_glmer=True)

# 3G) Gaussian LMM
write_report("**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**")
df_3g = df_controlled[df_controlled['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'agency_z', 'control_level_c']).copy()
fit_print_lmm("log_mem_rt ~ agency_z * control_level_c + (1 | participant)", df_3g, is_glmer=False)


# ============================================================================
# 5. PLOTTING
# ============================================================================
# Determine which participants were excluded (by any criterion) for plot labelling
excluded_all_px = set(data_all['participant'].unique()) - set(data['participant'].unique())

print("\nGenerating Plots in CDmem_final_output/...")

# Prepare breakdown data for plotting (Row 2 of plots)
def get_subtype(row):
    if row['item_type'] == 'controlled':
        return 'ctrl_detected' if row['detection_accuracy'] == 1 else 'ctrl_not_detected'
    return 'uncontrolled'

def _make_bd_results(tgt_df, fa_df):
    bd = tgt_df.dropna(subset=['item_type', 'detection_accuracy']).copy()
    bd['item_subtype'] = bd.apply(get_subtype, axis=1)
    res = bd.groupby(['participant', 'control_level', 'item_subtype'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
    res = res.merge(fa_df, on='participant', how='left')
    res['d_prime'] = res.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)
    return res

bd_results = _make_bd_results(targets, fa_rates)
bd_results_all = _make_bd_results(targets_all, fa_rates_all)

# Controlled items for agency-recognition plot (all participants)
df_controlled_all = targets_all[targets_all['item_type'] == 'controlled'].copy()

def draw_bd_bars(ax, bd, y_col):
    colors = {'ctrl_detected': '#2e8b57', 'ctrl_not_detected': '#90ee90', 'uncontrolled': '#fc8d62'}
    labels = {'ctrl_detected': 'Controlled (Detected)', 'ctrl_not_detected': 'Controlled (Not Detected)', 'uncontrolled': 'Uncontrolled'}
    subtypes = ['ctrl_detected', 'ctrl_not_detected', 'uncontrolled']
    bar_w = 0.22
    offsets = [-bar_w - 0.02, 0.0 - 0.02, bar_w + 0.12]
    
    for ci, cond in enumerate(['high', 'low']):
        cx = ci * 1.5
        for si, sub in enumerate(subtypes):
            subset = bd[(bd['control_level'] == cond) & (bd['item_subtype'] == sub)]
            val = subset[y_col].mean() if len(subset) > 0 else 0
            ax.bar(cx + offsets[si], val, width=bar_w, color=colors[sub], edgecolor='white', linewidth=0.5, label=labels[sub] if ci == 0 else None)
    
    ax.set_xticks([0, 1.5])
    ax.set_xticklabels(['High', 'Low'])
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Item Subtype', frameon=True, loc='upper right')

def make_3row_plot(df_2x2, df_bd, y_col, y_label, title, out_path, annotate_stats=False):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Row 1: 2x2 Factorial
    sns.barplot(data=df_2x2, x='control_level', y=y_col, hue='item_type', errorbar='se', palette='Set2', capsize=0.1, ax=axes[0], order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])
    
    if annotate_stats:
        summary = df_2x2.groupby(['item_type', 'control_level'], observed=True)[y_col].agg(['mean', 'std', 'sem']).reindex([
            ('controlled', 'high'), ('controlled', 'low'),
            ('uncontrolled', 'high'), ('uncontrolled', 'low')
        ])
        means = summary['mean'].values
        sds = summary['std'].values
        ses = summary['sem'].values
        
        y_offset = 0.05 if y_col == 'Hit_rate' else 0.1
        
        for i in range(len(means)):
            p = axes[0].patches[i]
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            
            axes[0].text(x, y - (y_offset / 1.5), f"M={means[i]:.2f}", ha='center', va='top', fontsize=10, color='black', fontweight='bold')
            
            err_top = y + ses[i]
            axes[0].text(x, err_top + (y_offset / 1.5), f"SD={sds[i]:.2f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    axes[0].set_title(f"{title}: 2x2 Factorial", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Control Task Level', fontsize=12)
    axes[0].set_ylabel(y_label, fontsize=12)
    if y_col == 'd_prime':
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_ylim(0, 2.5)
    else:
        axes[0].axhline(0.5, color='gray', linestyle='--')
        axes[0].set_ylim(0, 1)
        
    # Row 2: Detection Breakdown
    draw_bd_bars(axes[1], df_bd, y_col)
    axes[1].set_title(f"{title}: Detection Breakdown", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Control Task Level', fontsize=12)
    axes[1].set_ylabel(y_label, fontsize=12)
    if y_col == 'd_prime':
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_ylim(0, 2.5)
    else:
        axes[1].axhline(0.5, color='gray', linestyle='--')
        axes[1].set_ylim(0, 1)
        
    # Row 3: Overall main effect of control
    overall = df_2x2.groupby('control_level', observed=True)[y_col].mean().reindex(['high', 'low'])
    axes[2].bar(['High', 'Low'], overall.values, color=['#1f77b4', '#ff7f0e'], edgecolor='white', width=0.45)
    axes[2].set_title(f"{title}: Overall Main Effect of Control Level", fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Control Task Level', fontsize=12)
    axes[2].set_ylabel(y_label, fontsize=12)
    if y_col == 'd_prime':
        axes[2].axhline(0, color='black', linestyle='--')
        axes[2].set_ylim(0, 2.5)
    else:
        axes[2].axhline(0.5, color='gray', linestyle='--')
        axes[2].set_ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Generate Pooled Plots
make_3row_plot(mem_results_2x2, bd_results, 'd_prime', "d'", "Sensitivity (d')", POOLED_DIR / "dprime_pooled.png", annotate_stats=True)
make_3row_plot(mem_results_2x2, bd_results, 'Hit_rate', "Hit Rate", "Hit Rate", POOLED_DIR / "hitrate_pooled.png")

# Generate Per-Participant Plots (ALL participants, including excluded)
for px in sorted(data_all['participant'].unique()):
    px_2x2 = mem_results_2x2_all[mem_results_2x2_all['participant'] == px]
    px_bd = bd_results_all[bd_results_all['participant'] == px]
    excl_flag = " -- [!] CONSIDER EXCLUSION" if px in excluded_all_px else ""
    if len(px_bd) > 0:
        make_3row_plot(px_2x2, px_bd, 'd_prime', "d'", f"Sensitivity (d') - Participant {px}{excl_flag}", PER_PARTICIPANT_DIR / f"dprime_p{px}.png")
        make_3row_plot(px_2x2, px_bd, 'Hit_rate', "Hit Rate", f"Hit Rate - Participant {px}{excl_flag}", PER_PARTICIPANT_DIR / f"hitrate_p{px}.png")


# ============================================================================
# 5b. SANITY CHECK PLOTS
# ============================================================================
# These plots verify the experimental manipulation worked.
# A 2x2 grid with:
#   [0,0] Psychometric function (calibration data with sigmoid fit + test-phase overlay)
#   [0,1] Detection accuracy by condition (bar chart)
#   [1,0] Agency ratings by condition and detection accuracy
#   [1,1] RT distribution (histograms by condition)
#
# Both are generated pooled (all participants) and per-participant.

def make_sanity_plot(plot_data, out_path, title_suffix=""):
    """Generate a 2x2 sanity check figure from main experiment data."""
    plt.style.use('seaborn-v0_8-whitegrid')
    color_map = {'high': '#1f77b4', 'low': '#ff7f0e'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_data = plot_data.copy()

    # Ensure numeric types
    for col in ['detection_accuracy', 'prop_used', 'agency_rating', 'rt_choice', 'calib_target']:
        if col in plot_data.columns:
            plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')

    # Split calibration vs test
    calib_d = plot_data[plot_data['phase'] == 'calibration'].copy() if 'phase' in plot_data.columns else pd.DataFrame()
    test_d  = plot_data[plot_data['phase'] == 'test'].copy()        if 'phase' in plot_data.columns else plot_data.copy()
    n_px = plot_data['participant'].nunique() if 'participant' in plot_data.columns else 1

    # --- Panel 1: QUEST+ Calibration Convergence ---
    ax = axes[0, 0]
    if not calib_d.empty and 'quest_alpha_sd' in calib_d.columns and 'trial_in_block' in calib_d.columns:
        calib_conv = calib_d.copy()
        calib_conv['trial_in_block'] = pd.to_numeric(calib_conv['trial_in_block'], errors='coerce')
        calib_conv['quest_alpha_sd'] = pd.to_numeric(calib_conv['quest_alpha_sd'], errors='coerce')
        calib_conv['calib_target_label'] = np.where(
            calib_conv['calib_target'] < 0.7, 'Low (~0.55)', 'High (~0.75)')

        target_colors = {'High (~0.75)': color_map['high'], 'Low (~0.55)': color_map['low']}

        if n_px == 1:
            # Per-participant: plot raw trajectories per staircase
            for label, grp in calib_conv.groupby('calib_target_label'):
                grp_sorted = grp.sort_values('trial_in_block')
                ax.plot(grp_sorted['trial_in_block'], grp_sorted['quest_alpha_sd'],
                        marker='o', markersize=3, color=target_colors.get(label, '#666'),
                        label=label, alpha=0.8)
        else:
            # Pooled: average trajectory with error bands
            sns.lineplot(data=calib_conv, x='trial_in_block', y='quest_alpha_sd',
                         hue='calib_target_label', marker='o', errorbar='se',
                         palette=target_colors, ax=ax)
            # Individual trajectories (lightly)
            for px_id in calib_conv['participant'].unique():
                px_d = calib_conv[calib_conv['participant'] == px_id]
                for label, grp in px_d.groupby('calib_target_label'):
                    grp_sorted = grp.sort_values('trial_in_block')
                    ax.plot(grp_sorted['trial_in_block'], grp_sorted['quest_alpha_sd'],
                            color=target_colors.get(label, '#666'), alpha=0.1, linewidth=0.8)

        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Convergence (0.2)')

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Staircase',
                  frameon=True, facecolor='white', framealpha=0.9, fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No calibration data available',
                transform=ax.transAxes, ha='center')

    ax.set_xlabel('Trial in Block')
    ax.set_ylabel('Alpha SD')
    ax.set_title('QUEST+ Calibration Convergence')

    # --- Panel 2: Accuracy by condition ---
    ax = axes[0, 1]
    has_cond = 'control_condition' in test_d.columns
    if has_cond and not test_d.empty:
        acc_data = test_d.groupby('control_condition')['detection_accuracy'].agg(['mean', 'sem']).reindex(['high', 'low']).dropna().reset_index()
        bar_colors = [color_map.get(str(c), '#666') for c in acc_data['control_condition']]
        ax.bar(acc_data['control_condition'].astype(str), acc_data['mean'], yerr=acc_data['sem'], capsize=5, color=bar_colors)
        ax.set_xlabel('Control Condition')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy (Test Phase)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 1])

    # --- Panel 3: Agency ratings by accuracy ---
    ax = axes[1, 0]
    if 'agency_rating' in test_d.columns and test_d['agency_rating'].notna().any():
        if has_cond:
            sns.barplot(data=test_d, x='control_condition', y='agency_rating',
                        hue='detection_accuracy', ax=ax, errorbar='se',
                        palette={0: '#cccccc', 1: '#999999'}, order=['high', 'low'])
            ax.set_xlabel('Control Condition')
        ax.set_ylabel('Agency Rating (1-7)')
        ax.set_title('Agency Ratings by Accuracy (Test Phase)')
        ax.legend(title='Correct (1) / Incorrect (0)')
    else:
        ax.text(0.5, 0.5, 'No agency rating data', transform=ax.transAxes, ha='center')

    # --- Panel 4: RT distribution ---
    ax = axes[1, 1]
    if 'rt_choice' in test_d.columns and test_d['rt_choice'].notna().any():
        df_rt = test_d[test_d['rt_choice'] > 0]
        if has_cond and len(df_rt) > 0:
            for condition in ['high', 'low']:
                rt_sub = df_rt[df_rt['control_condition'] == condition]['rt_choice']
                if len(rt_sub) > 0:
                    ax.hist(rt_sub, bins=30, alpha=0.4, label=condition, density=True, color=color_map.get(condition))
            ax.legend()
        ax.set_xlabel('Reaction Time (s)')
        ax.set_ylabel('Density')
        ax.set_title('RT Distribution (Test Phase)')
    else:
        ax.text(0.5, 0.5, 'No RT data', transform=ax.transAxes, ha='center')

    fig.suptitle(f'Sanity Check{title_suffix}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sanity check plot: {out_path}")


def make_calibration_plot(plot_data, out_path, title_suffix=""):
    """Generate a calibration convergence plot showing QUEST alpha_sd over trials."""
    if 'phase' not in plot_data.columns or 'quest_alpha_sd' not in plot_data.columns:
        return

    calib_d = plot_data[plot_data['phase'] == 'calibration'].copy()
    if len(calib_d) == 0:
        return

    calib_d['trial_in_block'] = pd.to_numeric(calib_d['trial_in_block'], errors='coerce')
    calib_d['quest_alpha_sd'] = pd.to_numeric(calib_d['quest_alpha_sd'], errors='coerce')
    calib_d['calib_target_label'] = np.where(calib_d['calib_target'] < 0.7, 'Low (~0.55)', 'High (~0.75)')

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.figure(figsize=(8, 6))

    # Average trajectory with error bands
    sns.lineplot(data=calib_d, x='trial_in_block', y='quest_alpha_sd',
                 hue='calib_target_label', marker='o', errorbar='se')

    # Individual trajectories (lightly)
    for px_id in calib_d['participant'].unique():
        px_d = calib_d[calib_d['participant'] == px_id]
        sns.lineplot(data=px_d, x='trial_in_block', y='quest_alpha_sd',
                     hue='calib_target_label', alpha=0.15, legend=False)

    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Convergence Threshold (0.2)')
    plt.title(f'Calibration: QUEST Alpha SD Trajectory{title_suffix}')
    plt.xlabel('Trial in Block')
    plt.ylabel('Alpha SD')

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Target Condition')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot: {out_path}")


def make_agency_recognition_plot(plot_data, out_path, title_suffix=""):
    """Generate Sense of Agency vs Recognition Memory plot using z-transformed agency ratings."""
    if plot_data.empty or 'agency_z' not in plot_data.columns or 'said_old_int' not in plot_data.columns:
        return
        
    df = plot_data.dropna(subset=['agency_z', 'said_old_int']).copy()
    if df.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Participant Means (scatter)
    if df['participant'].nunique() > 1:
        px_means = df.groupby(['participant', 'agency_z'])['said_old_int'].mean().reset_index()
        sns.scatterplot(data=px_means, x='agency_z', y='said_old_int', color='#85bfe3', alpha=0.7, 
                        label='Participant Mean', ax=ax, zorder=2)
    else:
        px_means = df.groupby('agency_z')['said_old_int'].mean().reset_index()
        sns.scatterplot(data=px_means, x='agency_z', y='said_old_int', color='#85bfe3', alpha=0.7, 
                        label='Mean per Agency Level', ax=ax, zorder=2)

    # 2. Group Mean ± SE
    if df['participant'].nunique() > 1:
        # Define sensible bins for z-scores (-3 to 3)
        bins = np.arange(-3.5, 4.0, 1.0)
        df['z_bin'] = pd.cut(df['agency_z'], bins=bins)
        df['z_bin_mid'] = df['z_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
        
        bin_stats = df.groupby('z_bin_mid')['said_old_int'].agg(['mean', 'sem', 'count']).dropna()
        bin_stats = bin_stats[bin_stats['count'] >= 5]
        
        if not bin_stats.empty:
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt='o', color='#d62728', capsize=5, capthick=2, markersize=8,
                        label='Group Mean ± SE', zorder=4)

    # 3. Overall Logistic Trend
    def logistic_func(x, b0, b1):
        # use np.clip to prevent overflow in exp
        return 1 / (1 + np.exp(-np.clip(b0 + b1 * x, -100, 100)))
        
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(logistic_func, df['agency_z'], df['said_old_int'], p0=[0, 0], maxfev=5000)
        x_min, x_max = df['agency_z'].min(), df['agency_z'].max()
        x_fit = np.linspace(x_min - 0.2, x_max + 0.2, 100)
        y_fit = logistic_func(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='#2ca02c', linewidth=3, label='Logistic Trend', zorder=3)
    except Exception as e:
        pass

    # 4. Chance line
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='Chance', zorder=1)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Agency Rating at Encoding (z-transformed)')
    ax.set_ylabel('Hit Rate (Proportion Recognised)')
    ax.set_title(f'Sense of Agency vs Recognition Memory{title_suffix}', fontsize=14, fontweight='bold')
    
    # Deduplicate legend just in case
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved agency vs memory plot: {out_path}")

# --- Generate Pooled Sanity Check & Calibration Plots ---
# Pooled plots use post-exclusion data (only valid participants)
make_sanity_plot(data, POOLED_DIR / "sanity_check_pooled.png", title_suffix="  (All Participants)")

if 'df_controlled' in locals():
    make_agency_recognition_plot(df_controlled, POOLED_DIR / "agency_recognition_pooled.png", title_suffix="  (All Participants)")

# --- Generate Per-Participant Sanity Check & Calibration Plots ---
# Iterate over ALL participants (pre-exclusion) so every participant gets a plot.
# Excluded participants are flagged in the title for visual inspection.
for px in sorted(data_all['participant'].unique()):
    px_data = data_all[data_all['participant'] == px]
    excl_flag = " -- [!] CONSIDER EXCLUSION" if px in excluded_all_px else ""
    make_sanity_plot(px_data, PER_PARTICIPANT_DIR / f"sanity_check_p{px}.png", title_suffix=f"  (p. {px}){excl_flag}")

    px_controlled = df_controlled_all[df_controlled_all['participant'] == px]
    if not px_controlled.empty:
        make_agency_recognition_plot(px_controlled, PER_PARTICIPANT_DIR / f"agency_recognition_p{px}.png", title_suffix=f"  (p. {px}){excl_flag}")

print("Analysis Complete! Report and plots generated in:", OUTPUT_DIR)
