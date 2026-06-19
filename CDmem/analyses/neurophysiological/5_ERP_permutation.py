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

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
# FieldTrip: dfolder = 'D:/MCRL DATA/eeg4_ERPSummaries';
dfolder = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries"

# ──────────────────────────────────────────────────────────────
# Config for Pairwise Comparison (Test 4)
# ──────────────────────────────────────────────────────────────
# Choose any two conditions from the main or detection-split sets to compare:
# 'high_control', 'low_control',
# 'high_control_detected', 'high_control_nondetected',
# 'low_control_detected', 'low_control_nondetected'
pairwise_cond_A = 'high_control_nondetected'
pairwise_cond_B = 'low_control_detected'

# ──────────────────────────────────────────────────────────────
# Load data for all participants
# ──────────────────────────────────────────────────────────────
# We load both the main ERP fif files and the detection-split ERP fif files.
eeg_set = {}

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
    
    # Store all by comment/condition label
    eeg_set[p_idx] = {ev.comment: ev for ev in evokeds_main + evokeds_det}

# ──────────────────────────────────────────────────────────────
# Electrode and time window selection (matches 4_ERP_graphs.py)
# ──────────────────────────────────────────────────────────────
# elec_include = ['FCz', 'Cz', 'CPz', 'Pz']  # (Wen et al., 2017)
elec_include = ['Fz', 'FCz', 'FC1', 'FC2']  # (following Giersiepen et al., 2024, 2025)
# time_include = [0.45, 0.65]  # P500 window (Wen et al., 2017)
time_include = [0.0, 1.0]  # full post-stimulus epoch for exploratory cluster search

# Get shared time axis from first loaded participant
t_axis = None
for p_idx in eeg_set:
    if 'high_control' in eeg_set[p_idx]:
        t_axis = eeg_set[p_idx]['high_control'].copy().crop(tmin=time_include[0], tmax=time_include[1]).times
        break

if t_axis is None:
    raise ValueError("No valid ERP data loaded to determine the time axis.")

# Helper to average over picked channels and convert to µV
def extract_p_data(evoked, electrodes, tmin, tmax):
    available_ch = evoked.ch_names
    picked_ch = [ch for ch in electrodes if ch in available_ch]
    ev_crop = evoked.copy().crop(tmin=tmin, tmax=tmax).pick(picked_ch)
    return ev_crop.data.mean(axis=0) * 1e6  # Average over channels, convert to µV

# Extract time-series for all participants
extracted_data = {}
for p_idx in eeg_set:
    p_data = {}
    for cond_name, evoked in eeg_set[p_idx].items():
        p_data[cond_name] = extract_p_data(evoked, elec_include, time_include[0], time_include[1])
    extracted_data[p_idx] = p_data

# Identify valid participant subsets for each test
# Test 1 needs: 'high_control', 'low_control'
subs_test1 = [p for p in extracted_data if 'high_control' in extracted_data[p] and 'low_control' in extracted_data[p]]

# Tests 2, 3, and 4 need the four detection-split conditions
det_conds = ['high_control_detected', 'high_control_nondetected', 'low_control_detected', 'low_control_nondetected']
subs_det = [p for p in extracted_data if all(c in extracted_data[p] for c in det_conds)]

# Warn about excluded participants
excluded_det = [plist[p] for p in extracted_data if p not in subs_det]
if excluded_det:
    print(f"\n  WARNING: Participant(s) {excluded_det} missing >=1 detection condition. Excluded from detection/interaction/pairwise tests.")

# Output folder for figures
save_to = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Helper function to run permutation test & plot/save results
# ──────────────────────────────────────────────────────────────
def run_permutation_test(X_condA, X_condB, label_A, label_B, title, save_filename, colors, linestyles, p_indices):
    # Paired difference (A - B) for 1-sample cluster test
    X_diff = X_condA - X_condB
    
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
        out_type='mask',    # returns boolean mask for clusters
        n_jobs=-1,
        seed=42
    )
    
    if clusters is None:
        clusters = []
        cluster_p_values = []
        
    good_cluster_inds = np.where(cluster_p_values < alpha)[0]
    
    print(f"\nRESULTS FOR: {title}")
    print(f"----------------------------------------")
    
    if len(good_cluster_inds) == 0:
        print(f"No significant clusters found (alpha = {alpha}).")
    else:
        print(f"Found {len(good_cluster_inds)} significant cluster(s) (alpha = {alpha}):\n")
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            time_inds = np.where(clusters[clu_idx])[0]
            c_tmin = t_axis[time_inds[0]]
            c_tmax = t_axis[time_inds[-1]]
            p_val  = cluster_p_values[clu_idx]
            avg_T = np.mean(T_obs[time_inds])
            direction = "Positive" if avg_T > 0 else "Negative"
            print(f"  Cluster {i_clu+1}: {direction} cluster from {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f})")
            
            # Print Cohen's d and statistics
            cluster_diff_data = X_diff[:, time_inds]
            participant_mean_diff = np.mean(cluster_diff_data, axis=1)
            mean_diff = np.mean(participant_mean_diff)
            std_diff = np.std(participant_mean_diff, ddof=1)
            cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
            print(f"    Mean difference: {mean_diff:8.4f} µV")
            print(f"    SD difference:   {std_diff:8.4f} µV")
            print(f"    Cohen's d:       {cohens_d:8.4f}")
            
    # Plotting Setup
    fontsz = 14
    fig_w_in, fig_h_in = 30 / 2.54, 20 / 2.54  # 30x20 cm
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_title(title, fontsize=fontsz)
    
    # Calculate y-limits dynamically
    X_all = np.vstack((X_condA, X_condB))
    max_val = np.max(np.abs(X_all))
    ylim_val = np.ceil(max_val * 1.1)
    ylim_val = max(ylim_val, 2.0)  # at least 2 µV for readability
    ylimits = [-ylim_val, ylim_val]
    
    ax.set_ylim(ylimits)
    ax.invert_yaxis()  # Reverse positive & negative poles (standard EEG view)
    ax.set_ylabel('Activity (µV)', fontsize=fontsz)
    
    # X-Axis Setup
    ax.set_xlim([time_include[0], time_include[1]])
    ax.set_xlabel('Time (s)', fontsize=fontsz)
    
    # Add horizontal dotted line at y=0 and vertical at t=0
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if time_include[0] <= 0 <= time_include[1]:
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        
    # Fill area of significant clusters with light grey
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        time_inds = np.where(clusters[clu_idx])[0]
        c_tmin = t_axis[time_inds[0]]
        c_tmax = t_axis[time_inds[-1]]
        
        ax.fill_between([c_tmin, c_tmax], ylimits[0], ylimits[1],
                        color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
        ax.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)
        
    # Plot ERP Lines
    grandMean_A = np.mean(X_condA, axis=0)
    grandMean_B = np.mean(X_condB, axis=0)
    
    h_A, = ax.plot(t_axis, grandMean_A, color=colors[0], linestyle=linestyles[0], linewidth=2)
    h_B, = ax.plot(t_axis, grandMean_B, color=colors[1], linestyle=linestyles[1], linewidth=2)
    
    ax.legend([h_A, h_B], [label_A, label_B], loc='upper left', fontsize=fontsz)
    ax.tick_params(labelsize=fontsz)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        
    plt.tight_layout()
    
    save_path = os.path.join(save_to, save_filename)
    fig.savefig(save_path, format='svg', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    print("----------------------------------------")


# ──────────────────────────────────────────────────────────────
# RUN THE TESTS
# ──────────────────────────────────────────────────────────────

# --- TEST 1: Main Effect of Condition (High vs. Low Control) ---
X1_A = np.array([extracted_data[p]['high_control'] for p in subs_test1])
X1_B = np.array([extracted_data[p]['low_control'] for p in subs_test1])
run_permutation_test(
    X_condA=X1_A, 
    X_condB=X1_B, 
    label_A='High Control', 
    label_B='Low Control', 
    title='Main Effect of Condition (High vs. Low Control)', 
    save_filename='01_permut_main_effect_condition.svg', 
    colors=['blue', 'red'], 
    linestyles=['-', '-'],
    p_indices=subs_test1
)

# --- TEST 2: Main Effect of Detection (Detected vs. Non-detected) ---
# Average of high and low control within each detection level
X2_A = np.array([0.5 * (extracted_data[p]['high_control_detected'] + extracted_data[p]['low_control_detected']) for p in subs_det])
X2_B = np.array([0.5 * (extracted_data[p]['high_control_nondetected'] + extracted_data[p]['low_control_nondetected']) for p in subs_det])
run_permutation_test(
    X_condA=X2_A, 
    X_condB=X2_B, 
    label_A='Detected', 
    label_B='Non-detected', 
    title='Main Effect of Detection (Detected vs. Non-detected)', 
    save_filename='02_permut_main_effect_detection.svg', 
    colors=[(0.12, 0.53, 0.53), (0.85, 0.37, 0.00)], 
    linestyles=['-', '-'],
    p_indices=subs_det
)

# --- TEST 3: Interaction (Condition x Detection) ---
# Difference of differences: (High_Det - High_NonDet) vs (Low_Det - Low_NonDet)
X3_A = np.array([extracted_data[p]['high_control_detected'] - extracted_data[p]['high_control_nondetected'] for p in subs_det])
X3_B = np.array([extracted_data[p]['low_control_detected'] - extracted_data[p]['low_control_nondetected'] for p in subs_det])
run_permutation_test(
    X_condA=X3_A, 
    X_condB=X3_B, 
    label_A='High Control (Det - NonDet)', 
    label_B='Low Control (Det - NonDet)', 
    title='Interaction of Condition x Detection (Difference Waves)', 
    save_filename='03_permut_interaction.svg', 
    colors=['blue', 'red'], 
    linestyles=['-', '-'],
    p_indices=subs_det
)

# --- TEST 4: Pairwise Comparison (Configurable) ---
# Identify which participants have both conditions
subs_test4 = [p for p in extracted_data if pairwise_cond_A in extracted_data[p] and pairwise_cond_B in extracted_data[p]]

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
    save_filename=f"04_permut_pairwise_{pairwise_cond_A}_vs_{pairwise_cond_B}.svg", 
    colors=[color_A, color_B], 
    linestyles=[style_A, style_B],
    p_indices=subs_test4
)

print("\nALL PERMUTATION TESTS COMPLETED SUCCESSFULLY!")
plt.show()
