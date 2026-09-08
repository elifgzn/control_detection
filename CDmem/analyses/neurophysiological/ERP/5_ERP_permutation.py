import os
import sys
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Which participant(s) do you want to process?
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# all participants, minus:
# 1 - no triggers
# 5 - reference elctrode appears noisy, renders data unusable
# 28 - no triggers
# data-based exclusions: 2, 3, 11, 24, 26, 43, 45, 46, 47
# participants with 100% accuracy: 7,18, 21, 44

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FieldTrip: dfolder = 'D:/MCRL DATA/eeg4_ERPSummaries';
dfolder = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries"
# dfolder = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries_onlycorrect"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config for Pairwise Comparison (Test 4)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Choose any two conditions from the main or detection-split sets to compare:
# 'high_control', 'low_control',
# 'high_control_detected', 'high_control_nondetected',
# 'low_control_detected', 'low_control_nondetected'
pairwise_cond_A = 'high_control_nondetected'
pairwise_cond_B = 'low_control_detected'

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load data for all participants
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# We load both the main ERP fif files and the detection-split ERP fif files.
eeg_set = {}
trial_counts = {}  # per-participant trial counts from summary CSVs

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-ave.fif")
    det_erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-detection-ave.fif")

    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue
    if not os.path.exists(det_erp_file):
        print(f"  Detection ERP file not found, skipping participant {pnum}: {det_erp_file}")
        continue

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    evokeds_main = mne.read_evokeds(erp_file, verbose=False)
    evokeds_det = mne.read_evokeds(det_erp_file, verbose=False)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # CRITICAL FIX: Re-apply montage to restore FCz positions 
    # that were lost in earlier preprocessing runs.
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"
    montage = mne.channels.read_custom_montage(bvef_path)
    montage.rename_channels({'REF': 'FCz'})
    for ev in evokeds_main + evokeds_det:
        ev.set_montage(montage, on_missing='ignore')

    # Store all by comment/condition label
    eeg_set[p_idx] = {ev.comment: ev for ev in evokeds_main + evokeds_det}
    
    # Load trial counts from summary CSVs
    tc = {}
    summary_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-summary.csv")
    if os.path.exists(summary_file):
        sdf = pd.read_csv(summary_file)
        for col in sdf.columns:
            if col.startswith('num_'):
                tc[col.replace('num_', '')] = int(sdf[col].iloc[0])
    det_summary_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-detection-summary.csv")
    if os.path.exists(det_summary_file):
        ddf = pd.read_csv(det_summary_file)
        for col in ddf.columns:
            if col.startswith('num_'):
                tc[col.replace('num_', '')] = int(ddf[col].iloc[0])
    trial_counts[p_idx] = tc

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Electrode and time window selection (matches 4_ERP_graphs.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# elec_include = ['FCz', 'Cz', 'CPz', 'Pz']  # (Wen et al., 2017)
elec_include = ['Fz', 'FCz', 'FC1', 'FC2']  # (following Giersiepen et al., 2024, 2025)
plot_time = [-0.3, 1.2]    # Full time window for the line graphs (matches 4_ERP_graphs.py)
test_time = [0.0, 1.0]     # Time window restricted for the permutation test

# Get shared time axis from first loaded participant
t_axis_full = None
for p_idx in eeg_set:
    if 'high_control' in eeg_set[p_idx]:
        t_axis_full = eeg_set[p_idx]['high_control'].copy().crop(tmin=plot_time[0], tmax=plot_time[1]).times
        break

if t_axis_full is None:
    raise ValueError("No valid ERP data loaded to determine the time axis.")

# Helper to average over picked channels and convert to ÂµV
def extract_p_data(evoked, electrodes, tmin, tmax):
    available_ch = evoked.ch_names
    picked_ch = [ch for ch in electrodes if ch in available_ch]
    ev_crop = evoked.copy().crop(tmin=tmin, tmax=tmax).pick(picked_ch)
    return ev_crop.data.mean(axis=0) * 1e6  # Average over channels, convert to ÂµV

# Extract time-series for all participants
extracted_data = {}
for p_idx in eeg_set:
    p_data = {}
    for cond_name, evoked in eeg_set[p_idx].items():
        p_data[cond_name] = extract_p_data(evoked, elec_include, plot_time[0], plot_time[1])
    extracted_data[p_idx] = p_data

# Identify valid participant subsets for each test
# Test 1 needs: 'high_control', 'low_control'
subs_test1 = [p for p in extracted_data if 'high_control' in extracted_data[p] and 'low_control' in extracted_data[p]]

# Tests 2, 3, and 4 need the four detection-split conditions
det_conds = ['high_control_detected', 'high_control_nondetected', 'low_control_detected', 'low_control_nondetected']
MIN_DET_TRIALS = 15

def _has_enough_det_trials(p_idx):
    """Check if participant has all detection conditions with >= MIN_DET_TRIALS trials each."""
    if not all(c in extracted_data[p_idx] for c in det_conds):
        return False
    tc = trial_counts.get(p_idx, {})
    for c in det_conds:
        if tc.get(c, 0) < MIN_DET_TRIALS:
            return False
    return True

subs_det = [p for p in extracted_data if _has_enough_det_trials(p)]

# Report excluded participants
excluded_det = [plist[p] for p in extracted_data if p not in subs_det]
if excluded_det:
    print(f"\n  âš  {len(excluded_det)} participant(s) excluded from Detection analyses ")
    print(f"    (missing condition or <15 trials per cell): {excluded_det}")
    # Print reason for each
    for p in extracted_data:
        if p not in subs_det:
            tc = trial_counts.get(p, {})
            det_tc = {c: tc.get(c, 'MISSING') for c in det_conds}
            print(f"    Sub {plist[p]:04d}: {det_tc}")
print(f"Found {len(subs_det)} subjects with all 4 detection conditions (>= 15 trials each).")

# Output folder for figures
save_to = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper: extract time indices from an MNE cluster object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# In MNE 1.11+, clusters are returned as tuples of slices,
# e.g. (slice(32, 45),), regardless of out_type. This helper
# converts any cluster format into a plain integer index array.
def _get_cluster_inds(cluster):
    """Return a 1-D integer array of time-point indices for *cluster*."""
    # cluster is a tuple like (slice(start, stop),) or (np.array([...]),)
    if isinstance(cluster, tuple):
        obj = cluster[0]
        if isinstance(obj, slice):
            return np.arange(obj.start, obj.stop)
        else:
            return np.asarray(obj).ravel()
    # boolean mask
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return np.flatnonzero(cluster)
    # fallback
    return np.flatnonzero(np.asarray(cluster))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Topoplot Settings & Helpers (Matching 4_ERP_graphs.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
topo_cmap = plt.cm.RdBu_r
topo_vlim = (-1.5, 1.5)
topo_fig_in = 8 / 2.54

def make_highlight_mask(evoked, highlight_names):
    return np.array([ch in highlight_names for ch in evoked.ch_names])

mask_params = dict(marker='*', markerfacecolor='black', markeredgecolor='black',
                   markersize=10, zorder=10)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper function to run permutation test & plot/save results
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_permutation_test(X_condA, X_condB, label_A, label_B, title, save_filename, colors, linestyles, p_indices, evs_A=None, evs_B=None, trials_A=None, trials_B=None):
    # Find indices for the test window
    test_inds = np.where((t_axis_full >= test_time[0] - 1e-5) & (t_axis_full <= test_time[1] + 1e-5))[0]
    
    # Extract data just for the test
    X_test_A = X_condA[:, test_inds]
    X_test_B = X_condB[:, test_inds]
    
    # Paired difference (A - B) for 1-sample cluster test
    X_diff = X_test_A - X_test_B
    
    n_permutations = 1000
    alpha = 0.05
    
    print(f"\n" + "="*60)
    print(f"  RUNNING TEST: {title}")
    print(f"  N = {len(p_indices)} participants: {[plist[p] for p in p_indices]}")
    print("="*60)
    
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_diff, 
        n_permutations=n_permutations, 
        tail=0,             # two-sided
        n_jobs=-1,
        seed=42
    )
    
    if clusters is None:
        clusters = []
        cluster_p_values = []
        
    good_cluster_inds = np.where(cluster_p_values < alpha)[0]
    
    txt_out = []
    txt_out.append(f"RESULTS FOR: {title}")
    txt_out.append(f"----------------------------------------")
    txt_out.append(f"Total clusters found: {len(clusters)}")
    txt_out.append(f"Significant clusters (p < {alpha}): {len(good_cluster_inds)}")
    txt_out.append(f"----------------------------------------\n")
    
    for i_clu in range(len(clusters)):
        clu_inds = _get_cluster_inds(clusters[i_clu])
        # clu_inds are indices within the cropped X_diff
        real_time_inds = test_inds[clu_inds]
        c_tmin = t_axis_full[real_time_inds[0]]
        c_tmax = t_axis_full[real_time_inds[-1]]
        p_val  = cluster_p_values[i_clu]
        avg_T = np.mean(T_obs[clu_inds])
        sum_T = np.sum(T_obs[clu_inds])
        direction = "Positive" if avg_T > 0 else "Negative"
        sig_marker = " â˜… SIGNIFICANT" if p_val < alpha else ""
        
        # Compute Cohen's d and statistics for every cluster
        cluster_diff_data = X_diff[:, clu_inds]
        participant_mean_diff = np.mean(cluster_diff_data, axis=1)
        mean_diff = np.mean(participant_mean_diff)
        std_diff = np.std(participant_mean_diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
        
        txt_out.append(f"  Cluster {i_clu+1}: {direction} cluster from {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f}){sig_marker}")
        txt_out.append(f"    n_timepoints:    {len(clu_inds)}")
        txt_out.append(f"    Mean T-obs:      {avg_T:8.4f}")
        txt_out.append(f"    Sum T-obs:       {sum_T:8.4f}")
        txt_out.append(f"    Mean difference: {mean_diff:8.4f} ÂµV")
        txt_out.append(f"    SD difference:   {std_diff:8.4f} ÂµV")
        txt_out.append(f"    Cohen's d:       {cohens_d:8.4f}\n")
    
    if len(clusters) == 0:
        txt_out.append(f"  No clusters found at all.\n")
        
    final_text = "\n".join(txt_out)
    print(f"\n{final_text}")
    
    # Save the text to a file
    base, _ = os.path.splitext(save_filename)
    txt_filename = base + '.txt'
    txt_filepath = os.path.join(save_to, txt_filename)
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(final_text)
            
    # Plotting Setup
    # Plotting Setup
    fontsz = 20
    fig_w_in, fig_h_in = 30 / 2.54, 20 / 2.54  # 30x20 cm
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_title(title, fontsize=fontsz)
    
    # # Calculate y-limits dynamically
    # X_all = np.vstack((X_condA, X_condB))
    # max_val = np.max(np.abs(X_all))
    # ylim_val = np.ceil(max_val * 1.1)
    # ylim_val = max(ylim_val, 3.0)  # at least 3 ÂµV for matching 4_ERP_graphs
    # ylimits = [-ylim_val, ylim_val]
    ylimits = [-3,3]

    
    ax.set_ylim(ylimits)
    ax.invert_yaxis()  # Reverse positive & negative poles (standard EEG view)
    ax.set_ylabel('Activity (ÂµV)', fontsize=fontsz)
    
    # X-Axis Setup
    ax.set_xlim([plot_time[0], plot_time[1]])
    ax.set_xlabel('Time (s)', fontsize=fontsz)
    
    # Add horizontal dotted line at y=0 and vertical at t=0
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if plot_time[0] <= 0 <= plot_time[1]:
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        
    # Fill area of significant clusters with light grey
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        clu_inds = _get_cluster_inds(clusters[clu_idx])
        real_time_inds = test_inds[clu_inds]
        c_tmin = t_axis_full[real_time_inds[0]]
        c_tmax = t_axis_full[real_time_inds[-1]]
        
        ax.fill_between([c_tmin, c_tmax], ylimits[0], ylimits[1],
                        color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
        ax.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)
        
    # Plot ERP Lines (No standard error shading, to match 4_ERP_graphs.py exactly)
    grandMean_A = np.mean(X_condA, axis=0)
    grandMean_B = np.mean(X_condB, axis=0)
    
    h_A, = ax.plot(t_axis_full, grandMean_A, color=colors[0], linestyle=linestyles[0], linewidth=2)
    h_B, = ax.plot(t_axis_full, grandMean_B, color=colors[1], linestyle=linestyles[1], linewidth=2)
    
    # Append mean trial counts to legend labels if provided
    leg_A = f"{label_A} (M={np.mean(trials_A):.0f} trials)" if trials_A is not None else label_A
    leg_B = f"{label_B} (M={np.mean(trials_B):.0f} trials)" if trials_B is not None else label_B
    ax.legend([h_A, h_B], [leg_A, leg_B], loc='upper left', fontsize=fontsz * 0.8)
    ax.tick_params(labelsize=fontsz)
    
    # Matching ticks and spines
    ax.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        
    plt.tight_layout()
    
    save_path = os.path.join(save_to, save_filename)
    fig.savefig(save_path, format='png', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    svg_save = os.path.join(save_to, base + '.svg')
    fig.savefig(svg_save, format='svg', bbox_inches='tight', facecolor='white')
    print(f"SVG saved to: {svg_save}")
    print("----------------------------------------")

    # â”€â”€ Topoplots for significant clusters â”€â”€
    if len(good_cluster_inds) > 0 and evs_A is not None and evs_B is not None:
        print("\n  Generating topoplots for significant clusters...")
        GA_A = mne.grand_average(evs_A)
        GA_B = mne.grand_average(evs_B)
        
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            clu_inds = _get_cluster_inds(clusters[clu_idx])
            real_time_inds = test_inds[clu_inds]
            c_tmin = t_axis_full[real_time_inds[0]]
            c_tmax = t_axis_full[real_time_inds[-1]]
            
            for GA, label in zip([GA_A, GA_B], [label_A, label_B]):
                ev_topo = GA.copy().crop(tmin=c_tmin, tmax=c_tmax)
                topo_data = ev_topo.data.mean(axis=1)
                
                fig_t, ax_t = plt.subplots(figsize=(topo_fig_in, topo_fig_in))
                h_mask = make_highlight_mask(ev_topo, elec_include)
                mne.viz.plot_topomap(
                    topo_data, ev_topo.info,
                    axes=ax_t, cmap=topo_cmap,
                    vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),
                    mask=h_mask, mask_params=mask_params,
                    show=False, contours=6
                )
                ax_t.set_title(f"{label}\n({c_tmin*1000:.0f} - {c_tmax*1000:.0f} ms)", fontsize=12)
                fig_t.patch.set_facecolor('white')
                fig_t.tight_layout()
                
                topo_name = f"{base}_topo_cluster{i_clu+1}_{label.replace(' ', '_')}"
                tsave = os.path.join(save_to, f"{topo_name}.png")
                fig_t.savefig(tsave, format='png', dpi=600, bbox_inches='tight', facecolor='white')
                tsave_svg = os.path.join(save_to, f"{topo_name}.svg")
                fig_t.savefig(tsave_svg, format='svg', bbox_inches='tight', facecolor='white')
                print(f"    Saved: {topo_name} (.png + .svg)")

    return test_inds, clusters, cluster_p_values, good_cluster_inds


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# RUN THE TESTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# --- TEST 1: Main Effect of Condition (High vs. Low Control) ---
X1_A = np.array([extracted_data[p]['high_control'] for p in subs_test1])
X1_B = np.array([extracted_data[p]['low_control'] for p in subs_test1])
evs1_A = [eeg_set[p]['high_control'] for p in subs_test1]
evs1_B = [eeg_set[p]['low_control'] for p in subs_test1]

run_permutation_test(
    X_condA=X1_A, 
    X_condB=X1_B, 
    label_A='High Control', 
    label_B='Low Control', 
    title='Main Effect of Condition (High vs. Low Control)', 
    save_filename='01_permut_main_effect_condition.png', 
    colors=[(0.00, 0.44, 0.69), (0.80, 0.47, 0.65)], 
    linestyles=['-', '-'],
    p_indices=subs_test1,
    evs_A=evs1_A,
    evs_B=evs1_B,
    trials_A=[trial_counts.get(p, {}).get('high_control', 0) for p in subs_test1],
    trials_B=[trial_counts.get(p, {}).get('low_control', 0) for p in subs_test1]
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# EXPLORATORY: Split by Starting Condition (High-start vs Low-start)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Participant numbers by which condition they started the experiment with
highstart_pnums = {4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 30, 32, 34, 36, 38, 40, 42, 44, 48, 50}
lowstart_pnums  = {7, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 49}

# Map p_idx -> participant number for filtering
highstart_pidx = [p for p in subs_test1 if plist[p] in highstart_pnums]
lowstart_pidx  = [p for p in subs_test1 if plist[p] in lowstart_pnums]

print(f"\n{'='*60}")
print(f"  EXPLORATORY: Split by Starting Condition")
print(f"  High-start: N={len(highstart_pidx)}, participants={[plist[p] for p in highstart_pidx]}")
print(f"  Low-start:  N={len(lowstart_pidx)}, participants={[plist[p] for p in lowstart_pidx]}")
print(f"{'='*60}")

# --- High-start subgroup: Main Effect of Condition ---
X_hs_A = np.array([extracted_data[p]['high_control'] for p in highstart_pidx])
X_hs_B = np.array([extracted_data[p]['low_control'] for p in highstart_pidx])
evs_hs_A = [eeg_set[p]['high_control'] for p in highstart_pidx]
evs_hs_B = [eeg_set[p]['low_control'] for p in highstart_pidx]

run_permutation_test(
    X_condA=X_hs_A,
    X_condB=X_hs_B,
    label_A='High Control',
    label_B='Low Control',
    title='Main Effect of Condition â€“ High-Start Subgroup',
    save_filename='01_permut_main_effect_condition_highstart.png',
    colors=[(0.00, 0.44, 0.69), (0.80, 0.47, 0.65)],
    linestyles=['-', '-'],
    p_indices=highstart_pidx,
    evs_A=evs_hs_A,
    evs_B=evs_hs_B
)

# --- Low-start subgroup: Main Effect of Condition ---
X_ls_A = np.array([extracted_data[p]['high_control'] for p in lowstart_pidx])
X_ls_B = np.array([extracted_data[p]['low_control'] for p in lowstart_pidx])
evs_ls_A = [eeg_set[p]['high_control'] for p in lowstart_pidx]
evs_ls_B = [eeg_set[p]['low_control'] for p in lowstart_pidx]

run_permutation_test(
    X_condA=X_ls_A,
    X_condB=X_ls_B,
    label_A='High Control',
    label_B='Low Control',
    title='Main Effect of Condition â€“ Low-Start Subgroup',
    save_filename='01_permut_main_effect_condition_lowstart.png',
    colors=[(0.00, 0.44, 0.69), (0.80, 0.47, 0.65)],
    linestyles=['-', '-'],
    p_indices=lowstart_pidx,
    evs_A=evs_ls_A,
    evs_B=evs_ls_B
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEST 5: Interaction of Order Ã— Condition (Between-Subjects)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tests whether the (High âˆ’ Low) control difference wave differs
# between participants who started with High vs Low control.
# This is a between-subjects test â†’ permutation_cluster_test.

diff_hs = np.array([extracted_data[p]['high_control'] - extracted_data[p]['low_control'] for p in highstart_pidx])
diff_ls = np.array([extracted_data[p]['high_control'] - extracted_data[p]['low_control'] for p in lowstart_pidx])

test_inds_5 = np.where((t_axis_full >= test_time[0] - 1e-5) & (t_axis_full <= test_time[1] + 1e-5))[0]
diff_hs_test = diff_hs[:, test_inds_5]
diff_ls_test = diff_ls[:, test_inds_5]

n_perm_5 = 1000
alpha_5 = 0.05

print(f"\n{'='*60}")
print(f"  RUNNING TEST 5: Interaction of Order Ã— Condition (Between-Subjects)")
print(f"  High-start: N={len(highstart_pidx)}, participants={[plist[p] for p in highstart_pidx]}")
print(f"  Low-start:  N={len(lowstart_pidx)}, participants={[plist[p] for p in lowstart_pidx]}")
print(f"{'='*60}")

T_obs_5, clusters_5, cluster_pv_5, H0_5 = permutation_cluster_test(
    [diff_hs_test, diff_ls_test],
    n_permutations=n_perm_5,
    tail=0,
    n_jobs=-1,
    seed=42
)

if clusters_5 is None:
    clusters_5 = []
    cluster_pv_5 = np.array([])

good_5 = np.where(cluster_pv_5 < alpha_5)[0]

txt_5 = []
txt_5.append("RESULTS FOR: Interaction of Order Ã— Condition (Between-Subjects)")
txt_5.append("----------------------------------------")
txt_5.append(f"Total clusters found: {len(clusters_5)}")
txt_5.append(f"Significant clusters (p < {alpha_5}): {len(good_5)}")
txt_5.append("----------------------------------------\n")

for i_clu in range(len(clusters_5)):
    clu_inds = _get_cluster_inds(clusters_5[i_clu])
    real_time_inds = test_inds_5[clu_inds]
    c_tmin = t_axis_full[real_time_inds[0]]
    c_tmax = t_axis_full[real_time_inds[-1]]
    p_val = cluster_pv_5[i_clu]
    avg_stat = np.mean(T_obs_5[clu_inds])
    sum_stat = np.sum(T_obs_5[clu_inds])
    sig_marker = " â˜… SIGNIFICANT" if p_val < alpha_5 else ""

    cluster_mean_hs = np.mean(np.mean(diff_hs[:, real_time_inds], axis=1))
    cluster_mean_ls = np.mean(np.mean(diff_ls[:, real_time_inds], axis=1))

    txt_5.append(f"  Cluster {i_clu+1}: {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f}){sig_marker}")
    txt_5.append(f"    n_timepoints:         {len(clu_inds)}")
    txt_5.append(f"    Mean stat (F):        {avg_stat:8.4f}")
    txt_5.append(f"    Sum stat (F):         {sum_stat:8.4f}")
    txt_5.append(f"    High-start mean(H-L): {cluster_mean_hs:8.4f} ÂµV")
    txt_5.append(f"    Low-start mean(H-L):  {cluster_mean_ls:8.4f} ÂµV\n")

if len(clusters_5) == 0:
    txt_5.append("  No clusters found at all.\n")

final_txt_5 = "\n".join(txt_5)
print(f"\n{final_txt_5}")

txt_path_5 = os.path.join(save_to, '05_permut_interaction_order_x_condition.txt')
with open(txt_path_5, 'w', encoding='utf-8') as f:
    f.write(final_txt_5)

# --- Plot: Difference waves by order group ---
fontsz_5 = 20
fig_w5, fig_h5 = 30 / 2.54, 20 / 2.54
fig5, ax5 = plt.subplots(figsize=(fig_w5, fig_h5))
ax5.set_title('Interaction: Order Ã— Condition\n(Difference Waves: High âˆ’ Low Control)', fontsize=fontsz_5)

ylimits_5 = [-3, 3]
ax5.set_ylim(ylimits_5)
ax5.invert_yaxis()
ax5.set_ylabel('Activity (ÂµV)', fontsize=fontsz_5)
ax5.set_xlim([plot_time[0], plot_time[1]])
ax5.set_xlabel('Time (s)', fontsize=fontsz_5)
ax5.axhline(0, color='black', linestyle='--', linewidth=0.5)
if plot_time[0] <= 0 <= plot_time[1]:
    ax5.axvline(0, color='black', linestyle='--', linewidth=0.5)

for i_clu, clu_idx in enumerate(good_5):
    clu_inds = _get_cluster_inds(clusters_5[clu_idx])
    real_time_inds = test_inds_5[clu_inds]
    c_tmin = t_axis_full[real_time_inds[0]]
    c_tmax = t_axis_full[real_time_inds[-1]]
    ax5.fill_between([c_tmin, c_tmax], ylimits_5[0], ylimits_5[1],
                     color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
    ax5.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
    ax5.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)

gm_diff_hs = np.mean(diff_hs, axis=0)
gm_diff_ls = np.mean(diff_ls, axis=0)
h_hs, = ax5.plot(t_axis_full, gm_diff_hs, color=(0.20, 0.63, 0.17), linestyle='-', linewidth=2)
h_ls, = ax5.plot(t_axis_full, gm_diff_ls, color=(0.89, 0.10, 0.11), linestyle='-', linewidth=2)
ax5.legend([h_hs, h_ls], ['High-Start (High âˆ’ Low)', 'Low-Start (High âˆ’ Low)'],
           loc='upper left', fontsize=fontsz_5)
ax5.tick_params(labelsize=fontsz_5)
ax5.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
fig5.patch.set_facecolor('white')
ax5.set_facecolor('white')
for spine in ax5.spines.values():
    spine.set_linewidth(1)
plt.tight_layout()

save_p5 = os.path.join(save_to, '05_permut_interaction_order_x_condition.png')
fig5.savefig(save_p5, format='png', dpi=600, bbox_inches='tight', facecolor='white')
fig5.savefig(save_p5.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_p5}")

# --- Plot: Four-cell view (all ERPs by order Ã— condition) ---
fig6, ax6 = plt.subplots(figsize=(fig_w5, fig_h5))
ax6.set_title('Order Ã— Condition: Four-Cell View', fontsize=fontsz_5)
ax6.set_ylim(ylimits_5)
ax6.invert_yaxis()
ax6.set_ylabel('Activity (ÂµV)', fontsize=fontsz_5)
ax6.set_xlim([plot_time[0], plot_time[1]])
ax6.set_xlabel('Time (s)', fontsize=fontsz_5)
ax6.axhline(0, color='black', linestyle='--', linewidth=0.5)
if plot_time[0] <= 0 <= plot_time[1]:
    ax6.axvline(0, color='black', linestyle='--', linewidth=0.5)

for i_clu, clu_idx in enumerate(good_5):
    clu_inds = _get_cluster_inds(clusters_5[clu_idx])
    real_time_inds = test_inds_5[clu_inds]
    c_tmin = t_axis_full[real_time_inds[0]]
    c_tmax = t_axis_full[real_time_inds[-1]]
    ax6.fill_between([c_tmin, c_tmax], ylimits_5[0], ylimits_5[1],
                     color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
    ax6.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
    ax6.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)

gm_hs_high = np.mean(X_hs_A, axis=0)
gm_hs_low  = np.mean(X_hs_B, axis=0)
gm_ls_high = np.mean(X_ls_A, axis=0)
gm_ls_low  = np.mean(X_ls_B, axis=0)

h1, = ax6.plot(t_axis_full, gm_hs_high, color=(0.00, 0.44, 0.69), linestyle='-',  linewidth=2)
h2, = ax6.plot(t_axis_full, gm_hs_low,  color=(0.80, 0.47, 0.65), linestyle='-',  linewidth=2)
h3, = ax6.plot(t_axis_full, gm_ls_high, color=(0.00, 0.44, 0.69), linestyle='--', linewidth=2)
h4, = ax6.plot(t_axis_full, gm_ls_low,  color=(0.80, 0.47, 0.65), linestyle='--', linewidth=2)
_hs_h = np.mean([trial_counts.get(p, {}).get('high_control', 0) for p in highstart_pidx])
_hs_l = np.mean([trial_counts.get(p, {}).get('low_control', 0) for p in highstart_pidx])
_ls_h = np.mean([trial_counts.get(p, {}).get('high_control', 0) for p in lowstart_pidx])
_ls_l = np.mean([trial_counts.get(p, {}).get('low_control', 0) for p in lowstart_pidx])
ax6.legend([h1, h2, h3, h4],
           [f'High-Start / High Control (M={_hs_h:.0f})', f'High-Start / Low Control (M={_hs_l:.0f})',
            f'Low-Start / High Control (M={_ls_h:.0f})', f'Low-Start / Low Control (M={_ls_l:.0f})'],
           loc='upper left', fontsize=fontsz_5 * 0.6)
ax6.tick_params(labelsize=fontsz_5)
ax6.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
fig6.patch.set_facecolor('white')
ax6.set_facecolor('white')
for spine in ax6.spines.values():
    spine.set_linewidth(1)
plt.tight_layout()

save_p6 = os.path.join(save_to, '05_permut_interaction_order_x_condition_4cell.png')
fig6.savefig(save_p6, format='png', dpi=600, bbox_inches='tight', facecolor='white')
fig6.savefig(save_p6.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
print(f"Four-cell figure saved to: {save_p6}")

# --- TEST 2: Main Effect of Detection (Detected vs. Non-detected) ---
if len(subs_det) >= 2:
    # Average of high and low control within each detection level
    X2_A = np.array([0.5 * (extracted_data[p]['high_control_detected'] + extracted_data[p]['low_control_detected']) for p in subs_det])
    X2_B = np.array([0.5 * (extracted_data[p]['high_control_nondetected'] + extracted_data[p]['low_control_nondetected']) for p in subs_det])
    run_permutation_test(
        X_condA=X2_A, 
        X_condB=X2_B, 
        label_A='Detected', 
        label_B='Non-detected', 
        title='Main Effect of Detection (Detected vs. Non-detected)', 
        save_filename='02_permut_main_effect_detection.png', 
        colors=[(0.12, 0.53, 0.53), (0.85, 0.37, 0.00)], 
        linestyles=['-', '-'],
        p_indices=subs_det,
        trials_A=[trial_counts.get(p, {}).get('high_control_detected', 0) + trial_counts.get(p, {}).get('low_control_detected', 0) for p in subs_det],
        trials_B=[trial_counts.get(p, {}).get('high_control_nondetected', 0) + trial_counts.get(p, {}).get('low_control_nondetected', 0) for p in subs_det]
    )
else:
    print(f"\n  SKIPPING TEST 2 (Detection): only {len(subs_det)} participant(s) have all detection conditions (need >= 2).")

# --- TEST 3: Interaction (Condition x Detection) ---
if len(subs_det) >= 2:
    # Difference of differences: (High_Det - High_NonDet) vs (Low_Det - Low_NonDet)
    X3_A = np.array([extracted_data[p]['high_control_detected'] - extracted_data[p]['high_control_nondetected'] for p in subs_det])
    X3_B = np.array([extracted_data[p]['low_control_detected'] - extracted_data[p]['low_control_nondetected'] for p in subs_det])
    t3_test_inds, t3_clusters, t3_pvals, t3_good = run_permutation_test(
        X_condA=X3_A, 
        X_condB=X3_B, 
        label_A='High Control (Det - NonDet)', 
        label_B='Low Control (Det - NonDet)', 
        title='Interaction of Condition x Detection (Difference Waves)', 
        save_filename='03_permut_interaction.png', 
        colors=['blue', 'red'], 
        linestyles=['-', '-'],
        p_indices=subs_det,
        trials_A=[trial_counts.get(p, {}).get('high_control_detected', 0) + trial_counts.get(p, {}).get('high_control_nondetected', 0) for p in subs_det],
        trials_B=[trial_counts.get(p, {}).get('low_control_detected', 0) + trial_counts.get(p, {}).get('low_control_nondetected', 0) for p in subs_det]
    )

    # --- Four-cell decomposition plot for the interaction ---
    X_hd = np.array([extracted_data[p]['high_control_detected'] for p in subs_det])
    X_hn = np.array([extracted_data[p]['high_control_nondetected'] for p in subs_det])
    X_ld = np.array([extracted_data[p]['low_control_detected'] for p in subs_det])
    X_ln = np.array([extracted_data[p]['low_control_nondetected'] for p in subs_det])

    fontsz_4c = 20
    fig_w4c, fig_h4c = 30 / 2.54, 20 / 2.54
    fig4c, ax4c = plt.subplots(figsize=(fig_w4c, fig_h4c))
    ax4c.set_title('Condition Ã— Detection: Four-Cell Decomposition', fontsize=fontsz_4c)

    ylimits_4c = [-3, 3]
    ax4c.set_ylim(ylimits_4c)
    ax4c.invert_yaxis()
    ax4c.set_ylabel('Activity (ÂµV)', fontsize=fontsz_4c)
    ax4c.set_xlim([plot_time[0], plot_time[1]])
    ax4c.set_xlabel('Time (s)', fontsize=fontsz_4c)
    ax4c.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if plot_time[0] <= 0 <= plot_time[1]:
        ax4c.axvline(0, color='black', linestyle='--', linewidth=0.5)

    # Shade significant clusters from the interaction test
    for i_clu, clu_idx in enumerate(t3_good):
        clu_inds = _get_cluster_inds(t3_clusters[clu_idx])
        real_time_inds = t3_test_inds[clu_inds]
        c_tmin = t_axis_full[real_time_inds[0]]
        c_tmax = t_axis_full[real_time_inds[-1]]
        ax4c.fill_between([c_tmin, c_tmax], ylimits_4c[0], ylimits_4c[1],
                         color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
        ax4c.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
        ax4c.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)

    gm_hd = np.mean(X_hd, axis=0)
    gm_hn = np.mean(X_hn, axis=0)
    gm_ld = np.mean(X_ld, axis=0)
    gm_ln = np.mean(X_ln, axis=0)

    h_hd, = ax4c.plot(t_axis_full, gm_hd, color=(0.00, 0.44, 0.69), linestyle='-',  linewidth=2)
    h_hn, = ax4c.plot(t_axis_full, gm_hn, color=(0.00, 0.44, 0.69), linestyle='--', linewidth=2)
    h_ld, = ax4c.plot(t_axis_full, gm_ld, color=(0.80, 0.47, 0.65), linestyle='-',  linewidth=2)
    h_ln, = ax4c.plot(t_axis_full, gm_ln, color=(0.80, 0.47, 0.65), linestyle='--', linewidth=2)
    # Build legend labels with mean trial counts
    _tc_hd = np.mean([trial_counts.get(p, {}).get('high_control_detected', 0) for p in subs_det])
    _tc_hn = np.mean([trial_counts.get(p, {}).get('high_control_nondetected', 0) for p in subs_det])
    _tc_ld = np.mean([trial_counts.get(p, {}).get('low_control_detected', 0) for p in subs_det])
    _tc_ln = np.mean([trial_counts.get(p, {}).get('low_control_nondetected', 0) for p in subs_det])
    ax4c.legend([h_hd, h_hn, h_ld, h_ln],
               [f'High Control Detected (M={_tc_hd:.0f})', f'High Control Non-detected (M={_tc_hn:.0f})',
                f'Low Control Detected (M={_tc_ld:.0f})', f'Low Control Non-detected (M={_tc_ln:.0f})'],
               loc='upper left', fontsize=fontsz_4c * 0.6)
    ax4c.tick_params(labelsize=fontsz_4c)
    ax4c.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
    fig4c.patch.set_facecolor('white')
    ax4c.set_facecolor('white')
    for spine in ax4c.spines.values():
        spine.set_linewidth(1)
    plt.tight_layout()

    save_4c = os.path.join(save_to, '03_permut_interaction_4cell.png')
    fig4c.savefig(save_4c, format='png', dpi=600, bbox_inches='tight', facecolor='white')
    fig4c.savefig(save_4c.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
    print(f"Four-cell interaction figure saved to: {save_4c}")

else:
    print(f"\n  SKIPPING TEST 3 (Interaction): only {len(subs_det)} participant(s) have all detection conditions (need >= 2).")

# --- TEST 4: Pairwise Comparison (Configurable) ---
# Identify which participants have both conditions
subs_test4 = [p for p in extracted_data if pairwise_cond_A in extracted_data[p] and pairwise_cond_B in extracted_data[p]]

if len(subs_test4) >= 2:
    X4_A = np.array([extracted_data[p][pairwise_cond_A] for p in subs_test4])
    X4_B = np.array([extracted_data[p][pairwise_cond_B] for p in subs_test4])

    label_A = pairwise_cond_A.replace('_', ' ').title().replace('Nondetected', 'Non-detected')
    label_B = pairwise_cond_B.replace('_', ' ').title().replace('Nondetected', 'Non-detected')

    # Determine line styling dynamically
    style_map = {
        'high_control': ('blue', '-'),
        'low_control': ('red', '-'),
        'high_control_detected': ((0.00, 0.44, 0.69), '-'),
        'high_control_nondetected': ((0.00, 0.44, 0.69), '--'),
        'low_control_detected': ((0.80, 0.47, 0.65), '-'),
        'low_control_nondetected': ((0.80, 0.47, 0.65), '--')
    }

    color_A, style_A = style_map.get(pairwise_cond_A, ('blue', '-'))
    color_B, style_B = style_map.get(pairwise_cond_B, ('red', '-'))

    run_permutation_test(
        X_condA=X4_A, 
        X_condB=X4_B, 
        label_A=label_A, 
        label_B=label_B, 
        title=f"Pairwise Comparison ({label_A} vs. {label_B})", 
        save_filename=f"04_permut_pairwise_{pairwise_cond_A}_vs_{pairwise_cond_B}.png", 
        colors=[color_A, color_B], 
        linestyles=[style_A, style_B],
        p_indices=subs_test4,
        trials_A=[trial_counts.get(p, {}).get(pairwise_cond_A, 0) for p in subs_test4],
        trials_B=[trial_counts.get(p, {}).get(pairwise_cond_B, 0) for p in subs_test4]
    )
else:
    print(f"\n  SKIPPING TEST 4 (Pairwise): only {len(subs_test4)} participant(s) have both '{pairwise_cond_A}' and '{pairwise_cond_B}' (need >= 2).")

print("\nALL PERMUTATION TESTS COMPLETED SUCCESSFULLY!")
plt.show()

