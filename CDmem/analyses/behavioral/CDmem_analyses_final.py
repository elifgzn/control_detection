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
from scipy.stats import norm
from pymer4.models import glmer, lmer

# Setup for rpy2/pymer4 on Windows
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION & OUTPUT DIRECTORIES
# ============================================================================
DATA_DIR = Path(r"H:\PHD\control_detection\pilot_data")
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

PARTICIPANT_FILTER = [9, 10, 11, 12, 13, 14, 15, 16, 17]
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

# Synchronize participants between both datasets
valid_px = set(data["participant"].unique()) & set(recog_data["participant"].unique())
data = data[data["participant"].isin(valid_px)].copy()
recog_data = recog_data[recog_data["participant"].isin(valid_px)].copy()

# Recognition RT Outliers (Criterion 4)
recog_data["mem_rt"] = pd.to_numeric(recog_data["mem_rt"], errors="coerce")
px_rt = recog_data.groupby("participant")["mem_rt"].agg(mean_rt="mean", std_rt="std").reset_index()
recog_data = recog_data.merge(px_rt, on="participant")
recog_data["rt_thresh"] = recog_data["mean_rt"] + RT_SD_MULTIPLIER * recog_data["std_rt"]
rt_mask = (recog_data["mem_rt"] > recog_data["rt_thresh"]) | recog_data["mem_rt"].isna()
recog_data = recog_data[~rt_mask].drop(columns=["mean_rt", "std_rt", "rt_thresh"]).copy()

write_report(f"- Excluded for timeout: {excluded_timeout}")
write_report(f"- Excluded for accuracy outliers: {excluded_acc}")
write_report(f"- Excluded for calibration failure: {excluded_calib}")
write_report(f"- Total recognition trials excluded for RT outliers: {rt_mask.sum()}")
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


# Derive recognition variables
recog_data['said_old'] = recog_data['mem_response'].str.lower() == 'yes'
recog_data['said_old_int'] = recog_data['said_old'].astype(int)

targets = recog_data[recog_data['mem_ground_truth'] == 'seen'].copy()
foils = recog_data[recog_data['mem_ground_truth'] == 'unseen'].copy()

# Add encoding-phase variables (agency_rating, detection_accuracy) to targets
# by linking recognition items back to encoding trials via image filenames.
#
# The recognition CSV already contains 'item_type' (controlled/uncontrolled)
# and 'trial_level' (high/low), so we only need to merge agency_rating and
# detection_accuracy from the encoding (test) phase.
#
# Columns in main test data:
#   true_controlled  = 'img_A' or 'img_B'  (which image the participant controlled)
#   img_A_name       = filename of image A
#   img_B_name       = filename of image B
#   control_condition = 'high' or 'low'
#   agency_rating    = subjective agency (1-7)
#   detection_accuracy = 1 (correct) or 0 (incorrect) control detection
#
# We build a lookup table that maps each (participant, mem_filename) pair
# to its agency_rating and detection_accuracy from the encoding phase.

img_lookups = []
for _, row in test_data.iterrows():
    px = row['participant']
    target_img = row['true_controlled']
    _agency = row.get('agency_rating', np.nan)
    _det_acc = row.get('detection_accuracy', np.nan)

    # The controlled image filename
    ctrl_filename = row['img_A_name'] if target_img == 'img_A' else row['img_B_name']
    # The uncontrolled image filename
    unctrl_filename = row['img_B_name'] if target_img == 'img_A' else row['img_A_name']

    img_lookups.append({
        'participant': px,
        'mem_filename': ctrl_filename,
        'agency_rating': _agency,
        'detection_accuracy': _det_acc
    })
    img_lookups.append({
        'participant': px,
        'mem_filename': unctrl_filename,
        'agency_rating': _agency,
        'detection_accuracy': _det_acc
    })
img_lookup = pd.DataFrame(img_lookups).dropna(subset=['mem_filename']).drop_duplicates(subset=['participant', 'mem_filename'])

# The recognition data already has 'item_type' and 'trial_level' columns.
# We check whether they exist and, if so, only merge agency_rating + detection_accuracy.
has_tracking = 'item_type' in targets.columns and 'trial_level' in targets.columns

if has_tracking:
    # Drop agency_rating/detection_accuracy from targets if they already exist (avoid _x/_y suffixes)
    for col in ['agency_rating', 'detection_accuracy']:
        if col in targets.columns:
            targets = targets.drop(columns=[col])
    targets = targets.merge(
        img_lookup[['participant', 'mem_filename', 'agency_rating', 'detection_accuracy']],
        on=['participant', 'mem_filename'], how='left'
    )
    # Rename 'trial_level' to 'control_level' for consistency with the rest of the script
    targets = targets.rename(columns={'trial_level': 'control_level'})
else:
    # Fallback: recognition data lacks condition columns, do a full merge
    # (This path is for older data files that did not log item_type/trial_level)
    img_lookup_full = img_lookup.copy()
    # Re-build with control_level and item_type included
    img_lookups_full = []
    for _, row in test_data.iterrows():
        px = row['participant']
        cond = row['control_condition']
        target_img = row['true_controlled']
        _agency = row.get('agency_rating', np.nan)
        _det_acc = row.get('detection_accuracy', np.nan)
        ctrl_fn = row['img_A_name'] if target_img == 'img_A' else row['img_B_name']
        unctrl_fn = row['img_B_name'] if target_img == 'img_A' else row['img_A_name']
        img_lookups_full.append({'participant': px, 'mem_filename': ctrl_fn, 'control_level': cond, 'item_type': 'controlled', 'agency_rating': _agency, 'detection_accuracy': _det_acc})
        img_lookups_full.append({'participant': px, 'mem_filename': unctrl_fn, 'control_level': cond, 'item_type': 'uncontrolled', 'agency_rating': _agency, 'detection_accuracy': _det_acc})
    img_lookup_full = pd.DataFrame(img_lookups_full).dropna(subset=['mem_filename']).drop_duplicates(subset=['participant', 'mem_filename'])
    targets = targets.merge(img_lookup_full, on=['participant', 'mem_filename'], how='inner')

# Contrast coding for predictors
targets['control_level_c'] = targets['control_level'].map({'high': 0.5, 'low': -0.5})
targets['item_type_c'] = targets['item_type'].map({'controlled': 0.5, 'uncontrolled': -0.5})
targets['detection_accuracy_c'] = targets['detection_accuracy'].map({1: 0.5, 0: -0.5})

# Z-score agency rating within participant
targets['agency_rating'] = pd.to_numeric(targets['agency_rating'], errors='coerce')
targets['agency_z'] = targets.groupby('participant')['agency_rating'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# Log reaction time
targets['log_mem_rt'] = np.log(targets['mem_rt'])


# Create d_prime summary (for ANOVA and Plots)
fa_rates = foils.groupby('participant')['said_old'].mean().reset_index().rename(columns={'said_old': 'FA_rate'})
def calc_dprime(hr, far, clip_val=0.01):
    hr_clipped = np.clip(hr, clip_val, 1.0 - clip_val)
    far_clipped = np.clip(far, clip_val, 1.0 - clip_val)
    return norm.ppf(hr_clipped) - norm.ppf(far_clipped)

mem_results_2x2 = targets.groupby(['participant', 'control_level', 'item_type'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
mem_results_2x2 = mem_results_2x2.merge(fa_rates, on='participant')
mem_results_2x2['d_prime'] = mem_results_2x2.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)


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
            write_report(str(result_fit) + "\n")
            return result_fit
        else:
            write_report("Model fitted but no coefficients returned.\n")
            return None
    except Exception as e:
        write_report(f"Model failed: {e}\n")
        return None

# ============================================================================
# 4. ANALYSES
# ============================================================================
write_report("## ANALYSES\n")

# --- 1) DOES HIGHER MOTOR CONTROL LEAD TO BETTER MEMORY? ---
write_report("### 1) Does higher motor control lead to better memory?\n")

# 1A) 2x2 ANOVA on d'
write_report("**1A) 2 (Control Level) x 2 (Item Type) ANOVA on d'**")
if len(mem_results_2x2['participant'].unique()) >= 2:
    res_1a = pg.rm_anova(data=mem_results_2x2, dv='d_prime', within=['control_level', 'item_type'], subject='participant', detailed=True)
    write_report(res_1a.to_markdown(index=False) + "\n")
    main_ctrl = res_1a[res_1a['Source'] == 'control_level'].iloc[0]
    main_item = res_1a[res_1a['Source'] == 'item_type'].iloc[0]
    interact = res_1a[res_1a['Source'] == 'control_level * item_type'].iloc[0]
    write_report(f"> **APA 7 Reporting Example:**\n> A 2 (Control Level: High vs. Low) x 2 (Item Type: Controlled vs. Uncontrolled) repeated-measures ANOVA on *d'* revealed a main effect of Control Level, *F*({int(main_ctrl['ddof1'])}, {int(main_ctrl['ddof2'])}) = {main_ctrl['F']:.2f}, *p* = {main_ctrl['p-unc']:.3f}, $\eta_p^2$ = {main_ctrl['ng2']:.3f}. The main effect of Item Type was *F*({int(main_item['ddof1'])}, {int(main_item['ddof2'])}) = {main_item['F']:.2f}, *p* = {main_item['p-unc']:.3f}, $\eta_p^2$ = {main_item['ng2']:.3f}. The interaction effect was *F*({int(interact['ddof1'])}, {int(interact['ddof2'])}) = {interact['F']:.2f}, *p* = {interact['p-unc']:.3f}, $\eta_p^2$ = {interact['ng2']:.3f}.\n")

# 1B) Binomial GLMM
write_report("**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**")
df_1b = targets.dropna(subset=['said_old_int', 'control_level_c', 'item_type_c']).copy()
coefs_1b = fit_print_lmm("said_old_int ~ control_level_c * item_type_c + (1 | participant)", df_1b, is_glmer=True)
if coefs_1b is not None:
    row_1b = coefs_1b.filter(pl.col('term') == 'control_level_c')
    if len(row_1b) > 0:
        est = row_1b['estimate'][0]; se = row_1b['std_error'][0]; z = row_1b['z_stat'][0]; p = row_1b['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> A binomial generalized linear mixed model on target trials revealed that Control Level predicted memory hits, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *z* = {z:.2f}, *p* = {p:.3f}.\n")

# 1C) Gaussian LMM
write_report("**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**")
df_1c = targets[targets['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'control_level_c', 'item_type_c']).copy()
coefs_1c = fit_print_lmm("log_mem_rt ~ control_level_c * item_type_c + (1 | participant)", df_1c, is_glmer=False)
if coefs_1c is not None:
    row_1c = coefs_1c.filter(pl.col('term') == 'control_level_c')
    if len(row_1c) > 0:
        est = row_1c['estimate'][0]; se = row_1c['std_error'][0]; t = row_1c['t_stat'][0]; p = row_1c['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> A linear mixed model on log-transformed reaction times for correct old items showed an effect of Control Level, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *t* = {t:.2f}, *p* = {p:.3f}.\n")


# --- 2) DOES THIS MEMORY EFFECT DEPEND ON THE CONSCIOUS DETECTION OF CONTROL? ---
write_report("### 2) Does this memory effect depend on the conscious detection of control?\n")
write_report("*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*\n")
df_controlled = targets[targets['item_type'] == 'controlled'].copy()

# 2D) Binomial GLMM
write_report("**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**")
df_2d = df_controlled.dropna(subset=['said_old_int', 'detection_accuracy_c', 'control_level_c']).copy()
coefs_2d = fit_print_lmm("said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2d, is_glmer=True)
if coefs_2d is not None:
    row_2d = coefs_2d.filter(pl.col('term') == 'detection_accuracy_c')
    if len(row_2d) > 0:
        est = row_2d['estimate'][0]; se = row_2d['std_error'][0]; z = row_2d['z_stat'][0]; p = row_2d['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> Detection accuracy significantly predicted subsequent recognition of controlled items, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *z* = {z:.2f}, *p* = {p:.3f}.\n")

# 2E) Gaussian LMM
write_report("**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**")
df_2e = df_controlled[df_controlled['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'detection_accuracy_c', 'control_level_c']).copy()
coefs_2e = fit_print_lmm("log_mem_rt ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2e, is_glmer=False)
if coefs_2e is not None:
    row_2e = coefs_2e.filter(pl.col('term') == 'detection_accuracy_c')
    if len(row_2e) > 0:
        est = row_2e['estimate'][0]; se = row_2e['std_error'][0]; t = row_2e['t_stat'][0]; p = row_2e['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> Detection accuracy was associated with reaction times for correctly remembered controlled items, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *t* = {t:.2f}, *p* = {p:.3f}.\n")


# --- 3) CAN WE PREDICT THIS EFFECT FROM PARTICIPANTS' SUBJECTIVE AGENCY RATINGS? ---
write_report("### 3) Can we predict this effect from participants' subjective agency ratings?\n")
write_report("*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*\n")

# 3F) Binomial GLMM
write_report("**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**")
df_3f = df_controlled.dropna(subset=['said_old_int', 'agency_z', 'control_level_c']).copy()
coefs_3f = fit_print_lmm("said_old_int ~ agency_z * control_level_c + (1 | participant)", df_3f, is_glmer=True)
if coefs_3f is not None:
    row_3f = coefs_3f.filter(pl.col('term') == 'agency_z')
    if len(row_3f) > 0:
        est = row_3f['estimate'][0]; se = row_3f['std_error'][0]; z = row_3f['z_stat'][0]; p = row_3f['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> Within-participant variation in subjective agency ratings significantly predicted recognition hits for controlled items, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *z* = {z:.2f}, *p* = {p:.3f}.\n")

# 3G) Gaussian LMM
write_report("**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**")
df_3g = df_controlled[df_controlled['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'agency_z', 'control_level_c']).copy()
coefs_3g = fit_print_lmm("log_mem_rt ~ agency_z * control_level_c + (1 | participant)", df_3g, is_glmer=False)
if coefs_3g is not None:
    row_3g = coefs_3g.filter(pl.col('term') == 'agency_z')
    if len(row_3g) > 0:
        est = row_3g['estimate'][0]; se = row_3g['std_error'][0]; t = row_3g['t_stat'][0]; p = row_3g['p_value'][0]
        write_report(f"> **APA 7 Reporting Example:**\n> Subjective agency ratings predicted recognition reaction times for controlled items, $\\beta$ = {est:.2f}, *SE* = {se:.2f}, *t* = {t:.2f}, *p* = {p:.3f}.\n")


# ============================================================================
# 5. PLOTTING
# ============================================================================
print("\nGenerating Plots in CDmem_final_output/...")

# Prepare breakdown data for plotting (Row 2 of plots)
def get_subtype(row):
    if row['item_type'] == 'controlled':
        return 'ctrl_detected' if row['detection_accuracy'] == 1 else 'ctrl_not_detected'
    return 'uncontrolled'

targets_bd = targets.dropna(subset=['item_type', 'detection_accuracy']).copy()
targets_bd['item_subtype'] = targets_bd.apply(get_subtype, axis=1)

bd_results = targets_bd.groupby(['participant', 'control_level', 'item_subtype'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
bd_results = bd_results.merge(fa_rates, on='participant', how='left')
bd_results['d_prime'] = bd_results.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)

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

def make_3row_plot(df_2x2, df_bd, y_col, y_label, title, out_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Row 1: 2x2 Factorial
    sns.barplot(data=df_2x2, x='control_level', y=y_col, hue='item_type', errorbar='se', palette='Set2', capsize=0.1, ax=axes[0], order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])
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
make_3row_plot(mem_results_2x2, bd_results, 'd_prime', "d'", "Sensitivity (d')", POOLED_DIR / "dprime_pooled.png")
make_3row_plot(mem_results_2x2, bd_results, 'Hit_rate', "Hit Rate", "Hit Rate", POOLED_DIR / "hitrate_pooled.png")

# Generate Per-Participant Plots
for px in mem_results_2x2['participant'].unique():
    px_2x2 = mem_results_2x2[mem_results_2x2['participant'] == px]
    px_bd = bd_results[bd_results['participant'] == px]
    if len(px_bd) > 0:
        make_3row_plot(px_2x2, px_bd, 'd_prime', "d'", f"Sensitivity (d') - Participant {px}", PER_PARTICIPANT_DIR / f"dprime_p{px}.png")
        make_3row_plot(px_2x2, px_bd, 'Hit_rate', "Hit Rate", f"Hit Rate - Participant {px}", PER_PARTICIPANT_DIR / f"hitrate_p{px}.png")


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
# Also: Calibration convergence plot (QUEST alpha_sd trajectory over trials)
#
# Both are generated pooled (all participants) and per-participant.

from scipy.optimize import curve_fit

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

    # --- Panel 1: Psychometric function ---
    ax = axes[0, 0]
    if not calib_d.empty:
        def psychometric_func(x, alpha, beta):
            return 0.5 + 0.48 / (1 + np.exp(-np.clip(beta, 0.1, 100) * (x - alpha)))

        # Global sigmoid fit
        try:
            if calib_d['prop_used'].nunique() >= 2:
                popt, _ = curve_fit(psychometric_func, calib_d['prop_used'], calib_d['detection_accuracy'],
                                    p0=[calib_d['prop_used'].mean(), 10],
                                    bounds=([0, 0], [1, 200]), maxfev=5000)
                x_fit = np.linspace(0, 1, 100)
                y_fit = psychometric_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, color='gray', linestyle='-', linewidth=2, alpha=0.6, label='Global Sigmoid Fit')
        except Exception:
            pass

        # Staircase-specific data points
        if 'calib_target' in calib_d.columns:
            calib_d['calib_condition'] = np.where(calib_d['calib_target'] < 0.7, 'low', 'high')
            for condition in ['high', 'low']:
                subset = calib_d[calib_d['calib_condition'] == condition].copy()
                if subset.empty:
                    continue
                if n_px == 1:
                    psych = subset.groupby('prop_used', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                    psych.rename(columns={'prop_used': 'prop_bin'}, inplace=True)
                else:
                    bins = np.linspace(0, 1, 11)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    subset['prop_bin'] = pd.cut(subset['prop_used'], bins=bins, labels=bin_centers)
                    psych = subset.groupby('prop_bin', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                    psych['prop_bin'] = psych['prop_bin'].astype(float)
                psych = psych.dropna(subset=['mean'])
                ax.errorbar(psych['prop_bin'], psych['mean'], yerr=psych['sem'],
                            marker='o', capsize=3, linestyle='None', color=color_map[condition], alpha=0.7,
                            label=f'{condition.capitalize()} {"Samples" if n_px == 1 else "Bins"}')
                target_acc = 0.55 if condition == 'low' else 0.85
                ax.axhline(y=target_acc, color=color_map[condition], linestyle=':', alpha=0.3)

        # Overlay test results
        if not test_d.empty:
            for condition in ['high', 'low']:
                ts = test_d[test_d['control_condition'] == condition]
                if not ts.empty:
                    ax.plot(ts['prop_used'].mean(), ts['detection_accuracy'].mean(), marker='*', markersize=15,
                            color=color_map[condition], markeredgecolor='black', linestyle='None',
                            label=f'{condition.capitalize()} (Test)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='small', loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No calibration data available', transform=ax.transAxes, ha='center')

    ax.set_xlabel('Control Level (prop self-motion)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Psychometric Function ({"Raw" if n_px == 1 else "Binned"} & Global Fit)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])

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
    calib_d['calib_target_label'] = np.where(calib_d['calib_target'] < 0.7, 'Low (~0.55)', 'High (~0.85)')

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


# --- Generate Pooled Sanity Check & Calibration Plots ---
make_sanity_plot(data, POOLED_DIR / "sanity_check_pooled.png", title_suffix="  (All Participants)")
make_calibration_plot(data, POOLED_DIR / "calibration_convergence_pooled.png", title_suffix="  (All Participants)")

# --- Generate Per-Participant Sanity Check & Calibration Plots ---
for px in sorted(data['participant'].unique()):
    px_data = data[data['participant'] == px]
    make_sanity_plot(px_data, PER_PARTICIPANT_DIR / f"sanity_check_p{px}.png", title_suffix=f"  (p. {px})")
    make_calibration_plot(px_data, PER_PARTICIPANT_DIR / f"calibration_convergence_p{px}.png", title_suffix=f"  (p. {px})")


print("Analysis Complete! Report and plots generated in:", OUTPUT_DIR)
