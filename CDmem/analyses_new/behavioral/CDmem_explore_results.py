#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# CDmem Exploratory Analyses
# ==========================
# This script runs three exploratory (post-hoc) analyses on the CDmem dataset.
# It reuses the same data-loading, exclusion, and variable-derivation pipeline
# from CDmem_analyses_final.py.
#
# Explorations:
#   1) Re-run Model 2D excluding participants with 100% detection accuracy in
#      the high-control condition (ceiling effect robustness check).
#   2) Split data by starting condition (starts_with == 'high' vs 'low') and
#      re-run all main models separately for each group.
#   3) Re-run the SoA -> recognition models (3F, 3G) restricted to correctly
#      detected trials only (detection_accuracy == 1).

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
EXPLORE_DIR = OUTPUT_DIR / "exploratory"

for d in [OUTPUT_DIR, EXPLORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_FILE = EXPLORE_DIR / "Exploratory_Report.md"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# CDmem Exploratory Analyses Report\n\n")

def write_report(text):
    print(text)
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# all participants, minus:
# 1 - no triggers
# 5 - reference elctrode appears noisy, renders data unusable
# 28 - no triggers
# other, data-based exclusions will be imposed by code

PARTICIPANT_FILTER = sorted(set(range(1, 51)) - {1, 5, 28})


TIMEOUT_THRESHOLD = 0.50
ACCURACY_SD_THRESHOLD = 2.5
RT_SD_MULTIPLIER = 3.0

# ============================================================================
# 1. DATA LOADING  (identical to CDmem_analyses_final.py)
# ============================================================================
write_report("## Data Loading & Preprocessing\n")

main_files = [f for f in glob.glob(str(DATA_DIR / "CDmem_*.csv")) if re.match(r"^CDmem_1_\d+(?:_\d+)?\.csv$", os.path.basename(f))]
dfs = []
for f in main_files:
    basename = os.path.basename(f)
    px_match = re.match(r"^CDmem_1_(\d+)", basename)
    if px_match and PARTICIPANT_FILTER:
        px_num = int(px_match.group(1))
        if px_num not in PARTICIPANT_FILTER:
            continue
    try:
        df = pd.read_csv(f)
        df["source_file"] = basename
        dfs.append(df)
    except Exception as e:
        write_report(f"[WARNING] Could not load {basename}: {e}")

data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if PARTICIPANT_FILTER and not data.empty:
    data = data[data['participant'].isin(PARTICIPANT_FILTER)].copy()

# Load recognition data
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
# 2. EXCLUSION CRITERIA  (identical to CDmem_analyses_final.py)
# ============================================================================
# Criterion 1: Timeout Rate
test_data = data[data["phase"] == "test"].copy()
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"
timeout_stats = test_data.groupby(["participant", "control_condition"])["is_timeout"].agg(total="count", timeouts="sum").reset_index()
timeout_stats["rate"] = timeout_stats["timeouts"] / timeout_stats["total"]
excluded_timeout = timeout_stats[timeout_stats["rate"] >= TIMEOUT_THRESHOLD]["participant"].unique().tolist()
data = data[~data["participant"].isin(excluded_timeout)].copy()

# Criterion 2: Accuracy Outliers
test_data = data[data["phase"] == "test"].copy()
acc_per_px = test_data.groupby(["participant", "control_condition"])["detection_accuracy"].mean().reset_index()
group_stats = acc_per_px.groupby("control_condition")["detection_accuracy"].agg(group_mean="mean", group_std="std").reset_index()
acc_per_px = acc_per_px.merge(group_stats, on="control_condition")
acc_per_px["z"] = (acc_per_px["detection_accuracy"] - acc_per_px["group_mean"]) / acc_per_px["group_std"]
excluded_acc = acc_per_px[acc_per_px["z"].abs() > ACCURACY_SD_THRESHOLD]["participant"].unique().tolist()
data = data[~data["participant"].isin(excluded_acc)].copy()

# Criterion 3: Calibration failure
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

# Criterion 5: Manipulation failure
test_data_manip = data[data["phase"] == "test"].copy()
test_data_manip["detection_accuracy"] = pd.to_numeric(test_data_manip["detection_accuracy"], errors="coerce")
test_data_manip["agency_rating"] = pd.to_numeric(test_data_manip["agency_rating"], errors="coerce")

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
for px in sorted(data["participant"].unique()):
    px_test = test_data_manip[test_data_manip["participant"] == px]
    high = px_test[px_test["control_condition"] == "high"]
    low = px_test[px_test["control_condition"] == "low"]
    if len(high) == 0 or len(low) == 0:
        continue
    acc_diff = high["detection_accuracy"].mean() - low["detection_accuracy"].mean()
    ag_diff = high["agency_rating"].mean() - low["agency_rating"].mean()
    both_converged = px_convergence.get(px, 0) == 2
    if acc_diff <= 0 and ag_diff <= 0:
        excluded_manip.append(px)
        continue
    if both_converged:
        continue
    t_acc, p_acc = ttest_ind(high["detection_accuracy"].dropna(), low["detection_accuracy"].dropna())
    t_ag, p_ag = ttest_ind(high["agency_rating"].dropna(), low["agency_rating"].dropna())
    acc_ok = acc_diff > 0 and p_acc < 0.05
    ag_ok = ag_diff > 0 and p_ag < 0.05
    if not acc_ok or not ag_ok:
        excluded_manip.append(px)

data = data[~data["participant"].isin(excluded_manip)].copy()

# Synchronize participants between both datasets
valid_px = set(data["participant"].unique()) & set(recog_data["participant"].unique())
data = data[data["participant"].isin(valid_px)].copy()
recog_data = recog_data[recog_data["participant"].isin(valid_px)].copy()

# Recognition RT Outliers
def _apply_rt_outlier_removal(rdf, multiplier):
    rdf = rdf.copy()
    rdf["mem_rt"] = pd.to_numeric(rdf["mem_rt"], errors="coerce")
    px_rt = rdf.groupby("participant")["mem_rt"].agg(mean_rt="mean", std_rt="std").reset_index()
    rdf = rdf.merge(px_rt, on="participant")
    rdf["rt_thresh"] = rdf["mean_rt"] + multiplier * rdf["std_rt"]
    mask = (rdf["mem_rt"] > rdf["rt_thresh"]) | rdf["mem_rt"].isna()
    return rdf[~mask].drop(columns=["mean_rt", "std_rt", "rt_thresh"]).copy(), mask.sum()

recog_data, rt_removed = _apply_rt_outlier_removal(recog_data, RT_SD_MULTIPLIER)

write_report(f"- Excluded for timeout: {excluded_timeout}")
write_report(f"- Excluded for accuracy outliers: {excluded_acc}")
write_report(f"- Excluded for calibration failure: {excluded_calib}")
write_report(f"- Excluded for manipulation failure: {excluded_manip}")
write_report(f"- Total recognition trials excluded for RT outliers: {rt_removed}")
write_report(f"- Final N participants: {len(valid_px)}\n")


# ============================================================================
# 3. VARIABLE DERIVATION  (identical to CDmem_analyses_final.py)
# ============================================================================
test_data = data[data["phase"] == "test"].copy()
test_data["detection_accuracy"] = pd.to_numeric(test_data["detection_accuracy"], errors="coerce")
test_data["agency_rating"] = pd.to_numeric(test_data["agency_rating"], errors="coerce")

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

    tgt['control_level_c'] = tgt['control_level'].map({'high': 0.5, 'low': -0.5})
    tgt['item_type_c'] = tgt['item_type'].map({'controlled': 0.5, 'uncontrolled': -0.5})
    tgt['detection_accuracy_c'] = tgt['detection_accuracy'].map({1: 0.5, 0: -0.5})
    tgt['agency_rating'] = pd.to_numeric(tgt['agency_rating'], errors='coerce')
    tgt['agency_z'] = tgt.groupby('participant')['agency_rating'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    tgt['log_mem_rt'] = np.log(tgt['mem_rt'])

    fa_r = fl.groupby('participant')['said_old'].mean().reset_index().rename(columns={'said_old': 'FA_rate'})
    m2x2 = tgt.groupby(['participant', 'control_level', 'item_type'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
    m2x2 = m2x2.merge(fa_r, on='participant')
    m2x2['d_prime'] = m2x2.apply(lambda r: calc_dprime(r['Hit_rate'], r['FA_rate']), axis=1)

    return tgt, fl, fa_r, m2x2

targets, foils, fa_rates, mem_results_2x2 = derive_recognition_vars(data, recog_data)


# ============================================================================
# Helper: fit and report mixed models  (identical to CDmem_analyses_final.py)
# ============================================================================
def fit_print_lmm(formula, df, family="binomial", is_glmer=True):
    try:
        model = glmer(formula, data=pl.from_pandas(df), family=family) if is_glmer else lmer(formula, data=pl.from_pandas(df))

        import rpy2.rinterface_lib.callbacks as _rpy2_cb
        _orig_print, _orig_warn = _rpy2_cb.consolewrite_print, _rpy2_cb.consolewrite_warnerror
        _rpy2_cb.consolewrite_print = _rpy2_cb.consolewrite_warnerror = lambda x: None
        try:
            model.fit()
        finally:
            _rpy2_cb.consolewrite_print, _rpy2_cb.consolewrite_warnerror = _orig_print, _orig_warn

        result_fit = model.result_fit
        if result_fit is not None and len(result_fit) > 0:
            write_report("**Fixed Effects:**")
            write_report(str(result_fit) + "\n")
        else:
            write_report("Model fitted but no coefficients returned.\n")
            return None

        ranef_var = getattr(model, 'ranef_var', None)
        ranef = getattr(model, 'ranef', None)

        write_report("**Random Effects Variance Components (ranef_var):**")
        if ranef_var is not None:
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
# PLOTTING HELPERS  (following CDmem_analyses_final.py style)
# ============================================================================

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

def draw_bd_bars(ax, bd, y_col='Hit_rate', annotate_stats=False):
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
            sd_val = subset[y_col].std() if len(subset) > 1 else 0
            bar = ax.bar(cx + offsets[si], val, width=bar_w, color=colors[sub], edgecolor='white', linewidth=0.5, label=labels[sub] if ci == 0 else None)
            if annotate_stats:
                bx = cx + offsets[si]
                ax.text(bx, val * 0.5, f"M={val:.2f}", ha='center', va='center', fontsize=8, color='black', fontweight='bold')
                ax.text(bx, val + 0.03, f"SD={sd_val:.2f}", ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

    ax.set_xticks([0, 1.5])
    ax.set_xticklabels(['High', 'Low'])
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Item Subtype', frameon=True, loc='upper right')


def make_hitrate_2x2_plot(tgt_df, fa_df, out_path, title_suffix=""):
    """Generate a 2-panel hit-rate plot:
      Left  – 2x2 factorial (control_level x item_type)
      Right – Detection breakdown (detected / not-detected / uncontrolled)
    """
    m2x2 = tgt_df.groupby(['participant', 'control_level', 'item_type'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
    bd = _make_bd_results(tgt_df, fa_df)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: 2x2 Factorial ---
    sns.barplot(data=m2x2, x='control_level', y='Hit_rate', hue='item_type',
                errorbar='se', palette='Set2', capsize=0.1, ax=axes[0],
                order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])

    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Control Task Level', fontsize=12)
    axes[0].set_ylabel('Hit Rate', fontsize=12)
    axes[0].set_title(f'Hit Rate: 2x2 Factorial{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].legend(title='Item Type', frameon=True)

    # --- Right panel: Detection Breakdown ---
    draw_bd_bars(axes[1], bd, y_col='Hit_rate')
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Control Task Level', fontsize=12)
    axes[1].set_ylabel('Hit Rate', fontsize=12)
    axes[1].set_title(f'Hit Rate: Detection Breakdown{title_suffix}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved hit-rate plot: {out_path}")


def _make_rt_2x2(tgt_df):
    df_rt = tgt_df.dropna(subset=['mem_rt', 'control_level', 'item_type']).copy()
    return df_rt.groupby(['participant', 'control_level', 'item_type'])['mem_rt'].mean().reset_index().rename(columns={'mem_rt': 'mean_rt'})


def _make_rt_bd(tgt_df):
    bd = tgt_df.dropna(subset=['item_type', 'detection_accuracy', 'mem_rt']).copy()
    bd['item_subtype'] = bd.apply(get_subtype, axis=1)
    return bd.groupby(['participant', 'control_level', 'item_subtype'])['mem_rt'].mean().reset_index().rename(columns={'mem_rt': 'mean_rt'})


def make_3row_plot(df_2x2, df_bd, y_col, y_label, title, out_path, annotate_stats=False):
    """Generate 3-row bar plot: Row 1 = 2x2 Factorial, Row 2 = Detection Breakdown, Row 3 = Main Effect."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    is_rt = y_col == 'mean_rt'

    # Row 1: 2x2 Factorial
    sns.barplot(data=df_2x2, x='control_level', y=y_col, hue='item_type',
                errorbar='se', palette='Set2', capsize=0.1, ax=axes[0],
                order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])

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
            if i < len(axes[0].patches) and not np.isnan(means[i]):
                p = axes[0].patches[i]
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                axes[0].text(x, y - (y_offset / 1.5), f"M={means[i]:.2f}", ha='center', va='top', fontsize=10, color='black', fontweight='bold')
                err_top = y + (ses[i] if not np.isnan(ses[i]) else 0)
                axes[0].text(x, err_top + (y_offset / 1.5), f"SD={sds[i]:.2f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    axes[0].set_title(f"{title}: 2x2 Factorial", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Control Task Level', fontsize=12)
    axes[0].set_ylabel(y_label, fontsize=12)
    if is_rt:
        axes[0].set_ylim(0, 2)
    elif y_col == 'd_prime':
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_ylim(0, 2.5)
    else:
        axes[0].axhline(0.5, color='gray', linestyle='--')
        axes[0].set_ylim(0, 1)

    # Row 2: Detection Breakdown
    draw_bd_bars(axes[1], df_bd, y_col, annotate_stats=annotate_stats)
    axes[1].set_title(f"{title}: Detection Breakdown", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Control Task Level', fontsize=12)
    axes[1].set_ylabel(y_label, fontsize=12)
    if is_rt:
        axes[1].set_ylim(0, 2)
    elif y_col == 'd_prime':
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_ylim(0, 2.5)
    else:
        axes[1].axhline(0.5, color='gray', linestyle='--')
        axes[1].set_ylim(0, 1)

    # Row 3: Overall main effect of control
    overall_stats = df_2x2.groupby('control_level', observed=True)[y_col].agg(['mean', 'std']).reindex(['high', 'low'])
    axes[2].bar(['High', 'Low'], overall_stats['mean'].values, color=['#1f77b4', '#ff7f0e'], edgecolor='white', width=0.45)

    if annotate_stats:
        for i, cond in enumerate(['high', 'low']):
            m_val = overall_stats.loc[cond, 'mean']
            sd_val = overall_stats.loc[cond, 'std']
            if i < len(axes[2].patches) and not np.isnan(m_val):
                p = axes[2].patches[i]
                bx = p.get_x() + p.get_width() / 2
                by = p.get_height()
                axes[2].text(bx, by * 0.5, f"M={m_val:.2f}", ha='center', va='center', fontsize=11, color='black', fontweight='bold')
                axes[2].text(bx, by + 0.03, f"SD={sd_val:.2f}", ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

    axes[2].set_title(f"{title}: Overall Main Effect of Control Level", fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Control Task Level', fontsize=12)
    axes[2].set_ylabel(y_label, fontsize=12)
    if is_rt:
        axes[2].set_ylim(0, 2)
    elif y_col == 'd_prime':
        axes[2].axhline(0, color='black', linestyle='--')
        axes[2].set_ylim(0, 2.5)
    else:
        axes[2].axhline(0.5, color='gray', linestyle='--')
        axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved 3-row plot: {out_path}")


def make_agency_recognition_plot(plot_data, out_path, title_suffix="", agency_col='agency_z'):
    """Generate Sense of Agency vs Recognition Memory plot.
    agency_col: 'agency_z' for z-transformed or 'agency_rating' for raw (1-7)."""
    if plot_data.empty or agency_col not in plot_data.columns or 'said_old_int' not in plot_data.columns:
        return

    df = plot_data.dropna(subset=[agency_col, 'said_old_int']).copy()
    if df.empty:
        return

    is_raw = agency_col == 'agency_rating'
    x_label = 'Agency Rating at Encoding (1–7)' if is_raw else 'Agency Rating at Encoding (z-transformed)'
    scale_tag = 'Raw' if is_raw else 'z-scored'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Participant Means (scatter)
    if df['participant'].nunique() > 1:
        px_means = df.groupby(['participant', agency_col])['said_old_int'].mean().reset_index()
        sns.scatterplot(data=px_means, x=agency_col, y='said_old_int', color='#85bfe3', alpha=0.7,
                        label='Participant Mean', ax=ax, zorder=2)
    else:
        px_means = df.groupby(agency_col)['said_old_int'].mean().reset_index()
        sns.scatterplot(data=px_means, x=agency_col, y='said_old_int', color='#85bfe3', alpha=0.7,
                        label='Mean per Agency Level', ax=ax, zorder=2)

    # 2. Group Mean +/- SE
    if df['participant'].nunique() > 1:
        if is_raw:
            bins = np.arange(0.5, 8.5, 1.0)  # bins centred on 1-7
        else:
            bins = np.arange(-3.5, 4.0, 1.0)
        df['a_bin'] = pd.cut(df[agency_col], bins=bins)
        df['a_bin_mid'] = df['a_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)

        bin_stats = df.groupby('a_bin_mid')['said_old_int'].agg(['mean', 'sem', 'count']).dropna()
        bin_stats = bin_stats[bin_stats['count'] >= 5]

        if not bin_stats.empty:
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt='o', color='#d62728', capsize=5, capthick=2, markersize=8,
                        label='Group Mean +/- SE', zorder=4)

    # 3. Overall Logistic Trend
    def logistic_func(x, b0, b1):
        return 1 / (1 + np.exp(-np.clip(b0 + b1 * x, -100, 100)))

    try:
        from scipy.optimize import curve_fit
        if df[agency_col].nunique() >= 2:
            popt, _ = curve_fit(logistic_func, df[agency_col], df['said_old_int'], p0=[0, 0], maxfev=5000)
            x_min, x_max = df[agency_col].min(), df[agency_col].max()
            x_fit = np.linspace(x_min - 0.2, x_max + 0.2, 100)
            y_fit = logistic_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, color='#2ca02c', linewidth=3, label='Logistic Trend', zorder=3)
    except Exception:
        pass

    # 4. Chance line
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='Chance', zorder=1)

    ax.set_ylim(-0.05, 1.05)
    if is_raw:
        ax.set_xlim(0.5, 7.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Hit Rate (Proportion Recognised)')
    ax.set_title(f'Agency ({scale_tag}) vs Recognition Memory{title_suffix}', fontsize=14, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved agency vs memory plot: {out_path}")


def make_agency_rt_plot(plot_data, out_path, title_suffix="", agency_col='agency_z'):
    """Generate Agency Rating vs Recognition RT plot, split by control level.
    Mirrors the agency_recognition plot but with RT on the y-axis.
    agency_col: 'agency_z' for z-transformed or 'agency_rating' for raw (1-7)."""
    if plot_data.empty or agency_col not in plot_data.columns or 'mem_rt' not in plot_data.columns:
        return

    df = plot_data.dropna(subset=[agency_col, 'mem_rt', 'control_level']).copy()
    # Restrict to hits (said_old == 1) since RT is only meaningful for recognised items
    if 'said_old_int' in df.columns:
        df = df[df['said_old_int'] == 1].copy()
    if df.empty:
        return

    is_raw = agency_col == 'agency_rating'
    x_label = 'Agency Rating at Encoding (1–7)' if is_raw else 'Agency Rating at Encoding (z-transformed)'
    scale_tag = 'Raw' if is_raw else 'z-scored'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {'high': '#1f77b4', 'low': '#ff7f0e'}
    label_map = {'high': 'High Control', 'low': 'Low Control'}

    # 1. Participant means as scatter points, split by control level
    for cond in ['high', 'low']:
        cond_df = df[df['control_level'] == cond]
        if cond_df.empty:
            continue
        if cond_df['participant'].nunique() > 1:
            px_means = cond_df.groupby(['participant', agency_col])['mem_rt'].mean().reset_index()
        else:
            px_means = cond_df.groupby(agency_col)['mem_rt'].mean().reset_index()
        sns.scatterplot(data=px_means, x=agency_col, y='mem_rt',
                        color=color_map[cond], alpha=0.4, label=label_map[cond],
                        ax=ax, zorder=2, s=30)

    # 2. Binned group means +/- SE, split by control level
    if df['participant'].nunique() > 1:
        if is_raw:
            bins = np.arange(0.5, 8.5, 1.0)  # bins centred on 1-7
        else:
            bins = np.arange(-3.5, 4.0, 1.0)
        for cond in ['high', 'low']:
            cond_df = df[df['control_level'] == cond].copy()
            if cond_df.empty:
                continue
            cond_df['a_bin'] = pd.cut(cond_df[agency_col], bins=bins)
            cond_df['a_bin_mid'] = cond_df['a_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
            bin_stats = cond_df.groupby('a_bin_mid')['mem_rt'].agg(['mean', 'sem', 'count']).dropna()
            bin_stats = bin_stats[bin_stats['count'] >= 5]
            if not bin_stats.empty:
                offset = 0.05 if cond == 'high' else -0.05
                ax.errorbar(bin_stats.index + offset, bin_stats['mean'], yerr=bin_stats['sem'],
                            fmt='s', color=color_map[cond], capsize=5, capthick=2, markersize=8,
                            zorder=4, markeredgecolor='black', markeredgewidth=0.5)

    # 3. Linear regression lines per condition
    from scipy.stats import linregress
    for cond in ['high', 'low']:
        cond_df = df[df['control_level'] == cond]
        if len(cond_df) < 3 or cond_df[agency_col].nunique() < 2:
            continue
        slope, intercept, r_val, p_val, _ = linregress(cond_df[agency_col], cond_df['mem_rt'])
        x_range = np.linspace(cond_df[agency_col].min() - 0.2, cond_df[agency_col].max() + 0.2, 100)
        ax.plot(x_range, intercept + slope * x_range, color=color_map[cond],
                linewidth=3, linestyle='-', zorder=3)

    ax.set_ylim(0, 2)
    if is_raw:
        ax.set_xlim(0.5, 7.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Recognition RT (seconds)')
    ax.set_title(f'Agency ({scale_tag}) vs Recognition RT{title_suffix}', fontsize=14, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved agency vs RT plot: {out_path}")


# ============================================================================
# EXPLORATION 1: Model 2D without 100% high-control accuracy participants
# ============================================================================
write_report("---")
write_report("## Exploration 1: Model 2D Without 100% High-Control Accuracy Participants\n")
write_report("**Rationale:** Some participants achieve 100% detection accuracy in the high-control "
             "condition (ceiling effect). This may inflate the detection x control interaction. "
             "We re-run Model 2D excluding these participants to check robustness.\n")

# Identify participants with 100% accuracy in the high-control condition
test_data_e1 = data[data["phase"] == "test"].copy()
test_data_e1["detection_accuracy"] = pd.to_numeric(test_data_e1["detection_accuracy"], errors="coerce")
high_acc = test_data_e1[test_data_e1["control_condition"] == "high"].groupby("participant")["detection_accuracy"].mean()
ceiling_px = high_acc[high_acc == 1.0].index.tolist()

write_report(f"Participants with 100% accuracy in high-control condition: {[int(p) for p in ceiling_px]}")
write_report(f"N excluded for ceiling: {len(ceiling_px)}")
write_report(f"N remaining: {len(valid_px) - len(ceiling_px)}\n")

# Filter data
data_e1 = data[~data["participant"].isin(ceiling_px)].copy()
recog_data_e1 = recog_data[~recog_data["participant"].isin(ceiling_px)].copy()

# Re-derive recognition variables
targets_e1, _, fa_rates_e1, _ = derive_recognition_vars(data_e1, recog_data_e1)
df_controlled_e1 = targets_e1[targets_e1['item_type'] == 'controlled'].copy()

# Re-run Model 2D
write_report("**Model 2D (re-run): Binomial GLMM on OLD CONTROLLED ITEMS ONLY:**")
write_report("`said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)`\n")
df_2d_e1 = df_controlled_e1.dropna(subset=['said_old_int', 'detection_accuracy_c', 'control_level_c']).copy()
fit_print_lmm("said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2d_e1, is_glmer=True)

# --- Plots for Exploration 1 ---
make_hitrate_2x2_plot(targets_e1, fa_rates_e1, EXPLORE_DIR / "e1_hitrate_no_ceiling.png",
                      title_suffix="  (Excl. 100% High-Acc Px)")

rt_2x2_e1 = _make_rt_2x2(targets_e1)
rt_bd_e1 = _make_rt_bd(targets_e1)
make_3row_plot(rt_2x2_e1, rt_bd_e1, 'mean_rt', "Mean RT (s)",
               "Recognition RT (Excl. 100% High-Acc Px)",
               EXPLORE_DIR / "e1_rt_no_ceiling.png", annotate_stats=True)

make_agency_recognition_plot(df_controlled_e1, EXPLORE_DIR / "e1_agency_recognition_no_ceiling.png",
                             title_suffix="  (Excl. 100% High-Acc Px)")
make_agency_recognition_plot(df_controlled_e1, EXPLORE_DIR / "e1_agency_recognition_raw_no_ceiling.png",
                             title_suffix="  (Excl. 100% High-Acc Px)", agency_col='agency_rating')

make_agency_rt_plot(df_controlled_e1, EXPLORE_DIR / "e1_agency_rt_no_ceiling.png",
                    title_suffix="  (Excl. 100% High-Acc Px)")
make_agency_rt_plot(df_controlled_e1, EXPLORE_DIR / "e1_agency_rt_raw_no_ceiling.png",
                    title_suffix="  (Excl. 100% High-Acc Px)", agency_col='agency_rating')


# ============================================================================
# EXPLORATION 2: Split by Starting Condition
# ============================================================================
write_report("---")
write_report("## Exploration 2: Split by Starting Condition\n")
write_report("**Rationale:** The experiment counterbalanced whether participants started with the "
             "'high' or 'low' control block. Order effects could influence results. We split by "
             "`starts_with` and re-run all main models separately.\n")
# NOTE: This analysis is probably underpowered because participant number is
# halved (~20 per group), reducing statistical power substantially.
write_report("> **NOTE:** This analysis is probably underpowered because participant number is "
             "halved (~20 per group), reducing statistical power substantially.\n")

# Extract each participant's starting condition
starts_with_lookup = data.groupby("participant")["starts_with"].first().dropna()
starts_high_px = starts_with_lookup[starts_with_lookup == "high"].index.tolist()
starts_low_px = starts_with_lookup[starts_with_lookup == "low"].index.tolist()

write_report(f"Starts with HIGH: N = {len(starts_high_px)}, participants = {[int(p) for p in starts_high_px]}")
write_report(f"Starts with LOW:  N = {len(starts_low_px)}, participants = {[int(p) for p in starts_low_px]}\n")

for group_name, group_px in [("Starts HIGH", starts_high_px), ("Starts LOW", starts_low_px)]:
    write_report(f"### {group_name} (N = {len(group_px)})\n")

    # Filter data for this group
    data_grp = data[data["participant"].isin(group_px)].copy()
    recog_grp = recog_data[recog_data["participant"].isin(group_px)].copy()

    if data_grp.empty or recog_grp.empty:
        write_report(f"  No data available for {group_name}. Skipping.\n")
        continue

    # Derive recognition variables for this group
    targets_grp, _, fa_grp, m2x2_grp = derive_recognition_vars(data_grp, recog_grp)
    df_controlled_grp = targets_grp[targets_grp['item_type'] == 'controlled'].copy()

    # --- Model 1B: said_old ~ control_level * item_type ---
    write_report(f"**Model 1B ({group_name}): Binomial GLMM on OLD ITEMS:**")
    write_report("`said_old_int ~ control_level_c * item_type_c + (1 | participant)`\n")
    df_1b_grp = targets_grp.dropna(subset=['said_old_int', 'control_level_c', 'item_type_c']).copy()
    fit_print_lmm("said_old_int ~ control_level_c * item_type_c + (1 | participant)", df_1b_grp, is_glmer=True)

    # --- Model 1C: log_mem_rt ~ control_level * item_type (hits only) ---
    write_report(f"**Model 1C ({group_name}): Gaussian LMM on OLD ITEMS (hits only):**")
    write_report("`log_mem_rt ~ control_level_c * item_type_c + (1 | participant)`\n")
    df_1c_grp = targets_grp[targets_grp['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'control_level_c', 'item_type_c']).copy()
    fit_print_lmm("log_mem_rt ~ control_level_c * item_type_c + (1 | participant)", df_1c_grp, is_glmer=False)

    # --- Model 2D: said_old ~ detection_accuracy * control_level (controlled items) ---
    write_report(f"**Model 2D ({group_name}): Binomial GLMM on OLD CONTROLLED ITEMS:**")
    write_report("`said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)`\n")
    df_2d_grp = df_controlled_grp.dropna(subset=['said_old_int', 'detection_accuracy_c', 'control_level_c']).copy()
    fit_print_lmm("said_old_int ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2d_grp, is_glmer=True)

    # --- Model 2E: log_mem_rt ~ detection_accuracy * control_level (controlled hits) ---
    write_report(f"**Model 2E ({group_name}): Gaussian LMM on OLD CONTROLLED ITEMS (hits only):**")
    write_report("`log_mem_rt ~ detection_accuracy_c * control_level_c + (1 | participant)`\n")
    df_2e_grp = df_controlled_grp[df_controlled_grp['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'detection_accuracy_c', 'control_level_c']).copy()
    fit_print_lmm("log_mem_rt ~ detection_accuracy_c * control_level_c + (1 | participant)", df_2e_grp, is_glmer=False)

    # --- Model 3F: said_old ~ agency_z * control_level (controlled items) ---
    write_report(f"**Model 3F ({group_name}): Binomial GLMM on OLD CONTROLLED ITEMS:**")
    write_report("`said_old_int ~ agency_z * control_level_c + (1 | participant)`\n")
    df_3f_grp = df_controlled_grp.dropna(subset=['said_old_int', 'agency_z', 'control_level_c']).copy()
    fit_print_lmm("said_old_int ~ agency_z * control_level_c + (1 | participant)", df_3f_grp, is_glmer=True)

    # --- Model 3G: log_mem_rt ~ agency_z * control_level (controlled hits) ---
    write_report(f"**Model 3G ({group_name}): Gaussian LMM on OLD CONTROLLED ITEMS (hits only):**")
    write_report("`log_mem_rt ~ agency_z * control_level_c + (1 | participant)`\n")
    df_3g_grp = df_controlled_grp[df_controlled_grp['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'agency_z', 'control_level_c']).copy()
    fit_print_lmm("log_mem_rt ~ agency_z * control_level_c + (1 | participant)", df_3g_grp, is_glmer=False)

    # --- Plots for this group ---
    safe_name = group_name.lower().replace(" ", "_")

    make_hitrate_2x2_plot(targets_grp, fa_grp, EXPLORE_DIR / f"e2_hitrate_{safe_name}.png",
                          title_suffix=f"  ({group_name}, N={len(group_px)})")

    rt_2x2_grp = _make_rt_2x2(targets_grp)
    rt_bd_grp = _make_rt_bd(targets_grp)
    make_3row_plot(rt_2x2_grp, rt_bd_grp, 'mean_rt', "Mean RT (s)",
                   f"Recognition RT ({group_name})",
                   EXPLORE_DIR / f"e2_rt_{safe_name}.png", annotate_stats=True)

    make_agency_recognition_plot(df_controlled_grp, EXPLORE_DIR / f"e2_agency_recognition_{safe_name}.png",
                                 title_suffix=f"  ({group_name}, N={len(group_px)})")
    make_agency_recognition_plot(df_controlled_grp, EXPLORE_DIR / f"e2_agency_recognition_raw_{safe_name}.png",
                                 title_suffix=f"  ({group_name}, N={len(group_px)})", agency_col='agency_rating')

    make_agency_rt_plot(df_controlled_grp, EXPLORE_DIR / f"e2_agency_rt_{safe_name}.png",
                        title_suffix=f"  ({group_name}, N={len(group_px)})")
    make_agency_rt_plot(df_controlled_grp, EXPLORE_DIR / f"e2_agency_rt_raw_{safe_name}.png",
                        title_suffix=f"  ({group_name}, N={len(group_px)})", agency_col='agency_rating')


# ============================================================================
# EXPLORATION 3: SoA -> Recognition Among Correctly Detected Trials Only
# ============================================================================
write_report("---")
write_report("## Exploration 3: SoA -> Recognition Among Correctly Detected Trials Only\n")
write_report("**Rationale:** Models 3F/3G test whether agency ratings predict recognition across "
             "all controlled trials. But detection accuracy and agency are correlated. By "
             "restricting to correctly detected trials only (detection_accuracy == 1), we test "
             "whether -- among trials where participants correctly detected control -- their "
             "*degree* of agency still predicts recognition.\n")
write_report("Both control levels (high and low) are included to test whether the agency -> "
             "memory link differs by control level even among correctly detected trials.\n")

# Filter to controlled items with detection_accuracy == 1
df_controlled_full = targets[targets['item_type'] == 'controlled'].copy()
df_detected = df_controlled_full[df_controlled_full['detection_accuracy'] == 1].copy()

n_trials_total = len(df_controlled_full)
n_trials_detected = len(df_detected)
n_px_detected = df_detected['participant'].nunique()

write_report(f"Total controlled trials: {n_trials_total}")
write_report(f"Correctly detected trials: {n_trials_detected} ({n_trials_detected/n_trials_total*100:.1f}%)")
write_report(f"N participants with detected trials: {n_px_detected}\n")

# Trial counts by control level
detected_by_level = df_detected.groupby('control_level').size()
write_report("Correctly detected trials by control level:")
for level in ['high', 'low']:
    n = detected_by_level.get(level, 0)
    write_report(f"  - {level}: {n}")
write_report("")

# --- Model 3F-detected: said_old ~ agency_z * control_level (correctly detected only) ---
write_report("**Model 3F-detected: Binomial GLMM on CORRECTLY DETECTED CONTROLLED ITEMS:**")
write_report("`said_old_int ~ agency_z * control_level_c + (1 | participant)`\n")
df_3f_det = df_detected.dropna(subset=['said_old_int', 'agency_z', 'control_level_c']).copy()
fit_print_lmm("said_old_int ~ agency_z * control_level_c + (1 | participant)", df_3f_det, is_glmer=True)

# --- Model 3G-detected: log_mem_rt ~ agency_z * control_level (correctly detected hits only) ---
write_report("**Model 3G-detected: Gaussian LMM on CORRECTLY DETECTED CONTROLLED ITEMS (hits only):**")
write_report("`log_mem_rt ~ agency_z * control_level_c + (1 | participant)`\n")
df_3g_det = df_detected[df_detected['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'agency_z', 'control_level_c']).copy()
fit_print_lmm("log_mem_rt ~ agency_z * control_level_c + (1 | participant)", df_3g_det, is_glmer=False)

# --- Plots for Exploration 3 ---
make_hitrate_2x2_plot(df_detected, fa_rates, EXPLORE_DIR / "e3_hitrate_detected_only.png",
                      title_suffix="  (Correctly Detected Trials Only)")

make_agency_recognition_plot(df_detected, EXPLORE_DIR / "e3_agency_recognition_detected_only.png",
                             title_suffix="  (Correctly Detected Trials Only)")
make_agency_recognition_plot(df_detected, EXPLORE_DIR / "e3_agency_recognition_raw_detected_only.png",
                             title_suffix="  (Correctly Detected Trials Only)", agency_col='agency_rating')

make_agency_rt_plot(df_detected, EXPLORE_DIR / "e3_agency_rt_detected_only.png",
                    title_suffix="  (Correctly Detected Trials Only)")
make_agency_rt_plot(df_detected, EXPLORE_DIR / "e3_agency_rt_raw_detected_only.png",
                    title_suffix="  (Correctly Detected Trials Only)", agency_col='agency_rating')


# ============================================================================
# EXPLORATION 4: Starting Order as an Interacting Factor (Full Sample)
# ============================================================================
write_report("---")
write_report("## Exploration 4: Starting Order as an Interacting Factor (Full Sample)\n")
write_report("**Rationale:** Exploration 2 split the sample by starting condition and re-ran "
             "models separately, but this halves N and cannot formally test whether starting "
             "order *interacts* with other factors. Here we add `starts_with` (contrast-coded: "
             "high = 0.5, low = −0.5) as a between-subjects factor and test its interactions "
             "with all within-subjects predictors using the full sample.\n")

# --- Contrast-code starts_with ---
starts_with_lookup = data.groupby("participant")["starts_with"].first().dropna()
data["starts_with_c"] = data["participant"].map(starts_with_lookup).map({"high": 0.5, "low": -0.5})
targets["starts_with_c"] = targets["participant"].map(starts_with_lookup).map({"high": 0.5, "low": -0.5})

# Also add the raw label for plotting convenience
targets["starts_with"] = targets["participant"].map(starts_with_lookup)

n_starts_high = (starts_with_lookup == "high").sum()
n_starts_low = (starts_with_lookup == "low").sum()
write_report(f"Starts HIGH: N = {n_starts_high}")
write_report(f"Starts LOW:  N = {n_starts_low}\n")

df_controlled_e4 = targets[targets['item_type'] == 'controlled'].copy()

# --- Model 1B-order: said_old ~ control_level * item_type * starts_with ---
write_report("### Model 1B-order: Binomial GLMM on OLD ITEMS\n")
write_report("`said_old_int ~ control_level_c * item_type_c * starts_with_c + (1 | participant)`\n")
df_1b_e4 = targets.dropna(subset=['said_old_int', 'control_level_c', 'item_type_c', 'starts_with_c']).copy()
fit_print_lmm("said_old_int ~ control_level_c * item_type_c * starts_with_c + (1 | participant)", df_1b_e4, is_glmer=True)

# --- Model 1C-order: log_mem_rt ~ control_level * item_type * starts_with (hits only) ---
write_report("### Model 1C-order: Gaussian LMM on OLD ITEMS (hits only)\n")
write_report("`log_mem_rt ~ control_level_c * item_type_c * starts_with_c + (1 | participant)`\n")
df_1c_e4 = targets[targets['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'control_level_c', 'item_type_c', 'starts_with_c']).copy()
fit_print_lmm("log_mem_rt ~ control_level_c * item_type_c * starts_with_c + (1 | participant)", df_1c_e4, is_glmer=False)

# --- Model 2D-order: said_old ~ detection_accuracy * control_level * starts_with (controlled items) ---
write_report("### Model 2D-order: Binomial GLMM on OLD CONTROLLED ITEMS\n")
write_report("`said_old_int ~ detection_accuracy_c * control_level_c * starts_with_c + (1 | participant)`\n")
df_2d_e4 = df_controlled_e4.dropna(subset=['said_old_int', 'detection_accuracy_c', 'control_level_c', 'starts_with_c']).copy()
fit_print_lmm("said_old_int ~ detection_accuracy_c * control_level_c * starts_with_c + (1 | participant)", df_2d_e4, is_glmer=True)

# --- Model 2E-order: log_mem_rt ~ detection_accuracy * control_level * starts_with (controlled hits) ---
write_report("### Model 2E-order: Gaussian LMM on OLD CONTROLLED ITEMS (hits only)\n")
write_report("`log_mem_rt ~ detection_accuracy_c * control_level_c * starts_with_c + (1 | participant)`\n")
df_2e_e4 = df_controlled_e4[df_controlled_e4['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'detection_accuracy_c', 'control_level_c', 'starts_with_c']).copy()
fit_print_lmm("log_mem_rt ~ detection_accuracy_c * control_level_c * starts_with_c + (1 | participant)", df_2e_e4, is_glmer=False)

# --- Model 3F-order: said_old ~ agency_z * control_level * starts_with (controlled items) ---
write_report("### Model 3F-order: Binomial GLMM on OLD CONTROLLED ITEMS\n")
write_report("`said_old_int ~ agency_z * control_level_c * starts_with_c + (1 | participant)`\n")
df_3f_e4 = df_controlled_e4.dropna(subset=['said_old_int', 'agency_z', 'control_level_c', 'starts_with_c']).copy()
fit_print_lmm("said_old_int ~ agency_z * control_level_c * starts_with_c + (1 | participant)", df_3f_e4, is_glmer=True)

# --- Model 3G-order: log_mem_rt ~ agency_z * control_level * starts_with (controlled hits) ---
write_report("### Model 3G-order: Gaussian LMM on OLD CONTROLLED ITEMS (hits only)\n")
write_report("`log_mem_rt ~ agency_z * control_level_c * starts_with_c + (1 | participant)`\n")
df_3g_e4 = df_controlled_e4[df_controlled_e4['said_old_int'] == 1].dropna(subset=['log_mem_rt', 'agency_z', 'control_level_c', 'starts_with_c']).copy()
fit_print_lmm("log_mem_rt ~ agency_z * control_level_c * starts_with_c + (1 | participant)", df_3g_e4, is_glmer=False)


# ============================================================================
# EXPLORATION 4 – PLOTS
# ============================================================================

# ---- A) Interaction Line Plots ----
# These show the classic interaction pattern: x = control_level, y = DV,
# separate lines for starts_high vs starts_low, with SE error bars.

def make_interaction_lineplot(plot_df, y_col, y_label, title, out_path,
                              x_col='control_level', group_col='starts_with',
                              chance_line=None):
    """Classic 2-factor interaction line plot with SE error bars."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    color_map = {'high': '#e74c3c', 'low': '#3498db'}
    label_map = {'high': 'Starts HIGH', 'low': 'Starts LOW'}
    marker_map = {'high': 'o', 'low': 's'}

    x_order = ['high', 'low']
    x_positions = {lv: i for i, lv in enumerate(x_order)}

    for grp in ['high', 'low']:
        grp_df = plot_df[plot_df[group_col] == grp]
        if grp_df.empty:
            continue

        # Aggregate per participant first, then compute group mean ± SE
        px_means = grp_df.groupby(['participant', x_col])[y_col].mean().reset_index()
        summary = px_means.groupby(x_col)[y_col].agg(['mean', 'sem']).reindex(x_order)

        xs = [x_positions[lv] for lv in summary.index]
        offset = -0.03 if grp == 'high' else 0.03  # slight offset to avoid overlap

        ax.errorbar([x + offset for x in xs], summary['mean'], yerr=summary['sem'],
                    fmt=marker_map[grp] + '-', color=color_map[grp], capsize=6,
                    capthick=2, markersize=10, linewidth=2.5, markeredgecolor='white',
                    markeredgewidth=1.5, label=label_map[grp], zorder=3)

    if chance_line is not None:
        ax.axhline(chance_line, color='gray', linestyle='--', alpha=0.5, zorder=1)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(['High', 'Low'], fontsize=12)
    ax.set_xlabel('Control Task Level', fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved interaction line plot: {out_path}")


def make_interaction_lineplot_3way(plot_df, y_col, y_label, title, out_path,
                                   split_col, split_labels,
                                   x_col='control_level',
                                   group_col='starts_with',
                                   chance_line=None):
    """3-way interaction line plot: faceted by split_col (e.g. item_type or detection_accuracy)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    split_vals = sorted(plot_df[split_col].dropna().unique())
    n_panels = len(split_vals)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    color_map = {'high': '#e74c3c', 'low': '#3498db'}
    label_map = {'high': 'Starts HIGH', 'low': 'Starts LOW'}
    marker_map = {'high': 'o', 'low': 's'}
    x_order = ['high', 'low']
    x_positions = {lv: i for i, lv in enumerate(x_order)}

    for ax, sv in zip(axes, split_vals):
        sub_df = plot_df[plot_df[split_col] == sv]
        panel_label = split_labels.get(sv, str(sv))

        for grp in ['high', 'low']:
            grp_df = sub_df[sub_df[group_col] == grp]
            if grp_df.empty:
                continue
            px_means = grp_df.groupby(['participant', x_col])[y_col].mean().reset_index()
            summary = px_means.groupby(x_col)[y_col].agg(['mean', 'sem']).reindex(x_order)
            xs = [x_positions[lv] for lv in summary.index]
            offset = -0.03 if grp == 'high' else 0.03

            ax.errorbar([x + offset for x in xs], summary['mean'], yerr=summary['sem'],
                        fmt=marker_map[grp] + '-', color=color_map[grp], capsize=6,
                        capthick=2, markersize=10, linewidth=2.5, markeredgecolor='white',
                        markeredgewidth=1.5, label=label_map[grp], zorder=3)

        if chance_line is not None:
            ax.axhline(chance_line, color='gray', linestyle='--', alpha=0.5, zorder=1)

        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(['High', 'Low'], fontsize=12)
        ax.set_xlabel('Control Task Level', fontsize=12)
        ax.set_title(panel_label, fontsize=13, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    axes[0].set_ylabel(y_label, fontsize=13)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3-way interaction plot: {out_path}")


# --- A1: Hit Rate by control_level × starts_with (all old items) ---
make_interaction_lineplot(
    targets.dropna(subset=['said_old_int', 'control_level', 'starts_with']),
    y_col='said_old_int', y_label='Hit Rate',
    title='Hit Rate × Starting Order',
    out_path=EXPLORE_DIR / "e4_interaction_hitrate.png",
    chance_line=0.5
)

# --- A2: Hit Rate by control_level × item_type × starts_with (3-way) ---
make_interaction_lineplot_3way(
    targets.dropna(subset=['said_old_int', 'control_level', 'item_type', 'starts_with']),
    y_col='said_old_int', y_label='Hit Rate',
    title='Hit Rate × Item Type × Starting Order',
    out_path=EXPLORE_DIR / "e4_interaction_hitrate_by_itemtype.png",
    split_col='item_type',
    split_labels={'controlled': 'Controlled Items', 'uncontrolled': 'Uncontrolled Items'},
    chance_line=0.5
)

# --- A3: RT by control_level × starts_with (hits only) ---
hits_e4 = targets[targets['said_old_int'] == 1].dropna(subset=['mem_rt', 'control_level', 'starts_with']).copy()
make_interaction_lineplot(
    hits_e4,
    y_col='mem_rt', y_label='Recognition RT (s)',
    title='Recognition RT × Starting Order (Hits Only)',
    out_path=EXPLORE_DIR / "e4_interaction_rt.png"
)

# --- A4: RT by control_level × item_type × starts_with ---
make_interaction_lineplot_3way(
    hits_e4.dropna(subset=['item_type']),
    y_col='mem_rt', y_label='Recognition RT (s)',
    title='Recognition RT × Item Type × Starting Order (Hits Only)',
    out_path=EXPLORE_DIR / "e4_interaction_rt_by_itemtype.png",
    split_col='item_type',
    split_labels={'controlled': 'Controlled Items', 'uncontrolled': 'Uncontrolled Items'}
)

# --- A5: Hit Rate by control_level × detection_accuracy × starts_with (controlled only) ---
df_controlled_e4_plot = df_controlled_e4.dropna(subset=['said_old_int', 'control_level', 'detection_accuracy', 'starts_with']).copy()
make_interaction_lineplot_3way(
    df_controlled_e4_plot,
    y_col='said_old_int', y_label='Hit Rate',
    title='Hit Rate × Detection Accuracy × Starting Order (Controlled Items)',
    out_path=EXPLORE_DIR / "e4_interaction_hitrate_by_detection.png",
    split_col='detection_accuracy',
    split_labels={1: 'Correctly Detected', 0: 'Not Detected'},
    chance_line=0.5
)


# ---- B) Faceted Versions of Existing Plots ----

# --- B1: Faceted Hit-Rate 2x2 Plots ---
def make_faceted_hitrate_plot(tgt_df, fa_df, out_path, facet_col='starts_with',
                               facet_labels=None):
    """Generate side-by-side 2x2 factorial hit-rate plots, one panel per facet level."""
    if facet_labels is None:
        facet_labels = {'high': 'Starts HIGH', 'low': 'Starts LOW'}

    facet_vals = sorted(tgt_df[facet_col].dropna().unique())
    n_panels = len(facet_vals)
    if n_panels == 0:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, fv in zip(axes, facet_vals):
        sub_tgt = tgt_df[tgt_df[facet_col] == fv]
        sub_fa = fa_df[fa_df['participant'].isin(sub_tgt['participant'].unique())]
        m2x2 = sub_tgt.groupby(['participant', 'control_level', 'item_type'])['said_old'].mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
        n_px = sub_tgt['participant'].nunique()

        sns.barplot(data=m2x2, x='control_level', y='Hit_rate', hue='item_type',
                    errorbar='se', palette='Set2', capsize=0.1, ax=ax,
                    order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Control Task Level', fontsize=12)
        ax.set_ylabel('Hit Rate', fontsize=12)
        ax.set_title(f'{facet_labels.get(fv, str(fv))} (N={n_px})', fontsize=13, fontweight='bold')
        ax.legend(title='Item Type', frameon=True)

    fig.suptitle('Hit Rate: 2×2 Factorial by Starting Order', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved faceted hit-rate plot: {out_path}")


make_faceted_hitrate_plot(targets, fa_rates, EXPLORE_DIR / "e4_faceted_hitrate_2x2.png")


# --- B2: Faceted Detection Breakdown Plots ---
def make_faceted_detection_breakdown(tgt_df, fa_df, out_path, facet_col='starts_with',
                                      facet_labels=None, y_col='Hit_rate'):
    if facet_labels is None:
        facet_labels = {'high': 'Starts HIGH', 'low': 'Starts LOW'}

    facet_vals = sorted(tgt_df[facet_col].dropna().unique())
    n_panels = len(facet_vals)
    if n_panels == 0:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, fv in zip(axes, facet_vals):
        sub_tgt = tgt_df[tgt_df[facet_col] == fv]
        sub_fa = fa_df[fa_df['participant'].isin(sub_tgt['participant'].unique())]
        bd = _make_bd_results(sub_tgt, sub_fa)
        n_px = sub_tgt['participant'].nunique()

        draw_bd_bars(ax, bd, y_col=y_col, annotate_stats=True)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Control Task Level', fontsize=12)
        ax.set_ylabel('Hit Rate', fontsize=12)
        ax.set_title(f'{facet_labels.get(fv, str(fv))} (N={n_px})', fontsize=13, fontweight='bold')

    fig.suptitle('Hit Rate: Detection Breakdown by Starting Order', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved faceted detection breakdown: {out_path}")


make_faceted_detection_breakdown(targets, fa_rates, EXPLORE_DIR / "e4_faceted_hitrate_detection_breakdown.png")


# --- B3: Faceted RT 2x2 Plots ---
def make_faceted_rt_plot(tgt_df, out_path, facet_col='starts_with', facet_labels=None):
    if facet_labels is None:
        facet_labels = {'high': 'Starts HIGH', 'low': 'Starts LOW'}

    facet_vals = sorted(tgt_df[facet_col].dropna().unique())
    n_panels = len(facet_vals)
    if n_panels == 0:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, fv in zip(axes, facet_vals):
        sub_tgt = tgt_df[tgt_df[facet_col] == fv]
        rt_2x2 = _make_rt_2x2(sub_tgt)
        n_px = sub_tgt['participant'].nunique()

        sns.barplot(data=rt_2x2, x='control_level', y='mean_rt', hue='item_type',
                    errorbar='se', palette='Set2', capsize=0.1, ax=ax,
                    order=['high', 'low'], hue_order=['controlled', 'uncontrolled'])

        ax.set_ylim(0, 2)
        ax.set_xlabel('Control Task Level', fontsize=12)
        ax.set_ylabel('Mean RT (s)', fontsize=12)
        ax.set_title(f'{facet_labels.get(fv, str(fv))} (N={n_px})', fontsize=13, fontweight='bold')
        ax.legend(title='Item Type', frameon=True)

    fig.suptitle('Recognition RT: 2×2 Factorial by Starting Order', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved faceted RT plot: {out_path}")


make_faceted_rt_plot(targets, EXPLORE_DIR / "e4_faceted_rt_2x2.png")


# --- B4: Faceted Agency vs Recognition Plots ---
for agency_col_name, tag in [('agency_z', 'zscore'), ('agency_rating', 'raw')]:
    for grp_label, grp_val in [('starts_high', 'high'), ('starts_low', 'low')]:
        sub_ctrl = df_controlled_e4[df_controlled_e4['starts_with'] == grp_val].copy()
        n_px = sub_ctrl['participant'].nunique()
        make_agency_recognition_plot(
            sub_ctrl,
            EXPLORE_DIR / f"e4_faceted_agency_recognition_{tag}_{grp_label}.png",
            title_suffix=f"  ({grp_label.replace('_', ' ').title()}, N={n_px})",
            agency_col=agency_col_name
        )

# --- B5: Faceted Agency vs RT Plots ---
for agency_col_name, tag in [('agency_z', 'zscore'), ('agency_rating', 'raw')]:
    for grp_label, grp_val in [('starts_high', 'high'), ('starts_low', 'low')]:
        sub_ctrl = df_controlled_e4[df_controlled_e4['starts_with'] == grp_val].copy()
        n_px = sub_ctrl['participant'].nunique()
        make_agency_rt_plot(
            sub_ctrl,
            EXPLORE_DIR / f"e4_faceted_agency_rt_{tag}_{grp_label}.png",
            title_suffix=f"  ({grp_label.replace('_', ' ').title()}, N={n_px})",
            agency_col=agency_col_name
        )


# ============================================================================
# DONE
# ============================================================================
write_report("\n---")
write_report("*Report generated by CDmem_explore_results.py*")
print(f"\nExploratory analysis complete! Report and plots saved in: {EXPLORE_DIR}")
