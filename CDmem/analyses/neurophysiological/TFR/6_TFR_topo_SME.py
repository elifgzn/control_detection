"""
6_TFR_topo_SME.py
=================
Whole-Brain Topoplots for Subsequent Memory Effects (SME)
---------------------------------------------------------

PURPOSE:
    Computes and plots grand-average whole-brain EEG topoplots during the full
    3-second stimulus presentation period (0.0 to 3.0 s) for canonical frequency
    bands:
      - Theta (4 - 8 Hz)
      - Alpha (8 - 12 Hz)
      - Beta (13 - 30 Hz)
      - Gamma (30 - 40 Hz)

    Contrasts & Conditions plotted:
      1. Main Effect of Memory (All Items):
         - Recalled, Not Recalled, and Difference (Recalled - Not Recalled)
      2. Main Effect of Memory (Controlled Only):
         - Recalled, Not Recalled, and Difference (Recalled - Not Recalled)
      3. Simple Effects & Interaction (from 4-cell All Items):
         - Low Control Memory Effect (Recalled - Not Recalled)
         - High Control Memory Effect (Recalled - Not Recalled)
         - Interaction: Memory x Control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked_topo_SME"
os.makedirs(figures_path, exist_ok=True)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

# Analysis Window
TOPO_TIME = (0.0, 3.0)  # Full 3-second active period

# Canonical Frequency Bands
BANDS = {
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 12.0),
    'Beta':  (13.0, 30.0),
    'Gamma': (30.0, 40.0)
}

# Colorbar limits (dB)
Z_LIMITS_COND = (-1.0, 1.0)  # For condition power topomaps
Z_LIMITS_DIFF = (-0.8, 0.8)  # For difference topomaps

# ══════════════════════════════════════════════════════════════════
# 1. LOAD MNE INFO FOR CANONICAL SENSOR LAYOUT
# ══════════════════════════════════════════════════════════════════
print("Loading canonical sensor info...")
ref_sub = 4
ref_file = os.path.join(eeg_path, f"CDmem_{ref_sub:04d}-epo.fif")
if not os.path.exists(ref_file):
    for s in plist:
        candidate = os.path.join(eeg_path, f"CDmem_{s:04d}-epo.fif")
        if os.path.exists(candidate):
            ref_file = candidate
            break

info = mne.read_epochs(ref_file, preload=False, verbose=False).info
canonical_ch_names = info.ch_names
print(f"  ✓ Sensor layout loaded ({len(canonical_ch_names)} channels)")

def align_channels(sub_data, sub_ch_names, canonical_ch_names):
    """Align individual subject channels to canonical 65-channel layout."""
    aligned = np.full((len(canonical_ch_names), sub_data.shape[1], sub_data.shape[2]), np.nan, dtype=np.float64)
    for i, ch in enumerate(canonical_ch_names):
        if ch in sub_ch_names:
            orig_idx = sub_ch_names.index(ch)
            aligned[i] = sub_data[orig_idx]
    return aligned

# ══════════════════════════════════════════════════════════════════
# 2. LOAD & ALIGN DATA
# ══════════════════════════════════════════════════════════════════

# --- A. Collapsed Main Effect files (AllItems and ControlledOnly) ---
main_effect_configs = {
    'AllItems': '_TFR_MainEffect_AllItems.npz',
    'ControlledOnly': '_TFR_MainEffect_ControlledOnly.npz'
}

times, freqs = None, None
data_main_effects = {}

for key, suffix in main_effect_configs.items():
    print(f"Loading Main Effect ({key})...")
    sub_rec_list = []
    sub_not_list = []
    
    for sub in plist:
        sub_id = f"{sub:04d}"
        file_path = os.path.join(input_path, f"CDmem_{sub_id}{suffix}")
        if not os.path.exists(file_path):
            continue
        
        saved = np.load(file_path, allow_pickle=True)
        if times is None:
            times = saved['times']
            freqs = saved['freqs']
        
        chs = saved['ch_names'].tolist()
        if 'recalled' in saved and 'not_recalled' in saved:
            rec_aligned = align_channels(saved['recalled'], chs, canonical_ch_names)
            not_aligned = align_channels(saved['not_recalled'], chs, canonical_ch_names)
            sub_rec_list.append(rec_aligned)
            sub_not_list.append(not_aligned)
            
    if sub_rec_list:
        data_main_effects[key] = {
            'recalled': np.array(sub_rec_list),
            'not_recalled': np.array(sub_not_list),
            'n_subs': len(sub_rec_list)
        }
        print(f"  ✓ {key}: loaded {len(sub_rec_list)} subjects")

# --- B. 4-cell AllItems file ---
print("Loading 4-cell condition averages (All Items)...")
sub_4cell = {'low_recalled': [], 'low_not_recalled': [], 'high_recalled': [], 'high_not_recalled': []}

for sub in plist:
    sub_id = f"{sub:04d}"
    file_path = os.path.join(input_path, f"CDmem_{sub_id}_TFR_ConditionAverages_AllItems.npz")
    if not os.path.exists(file_path):
        continue
    saved = np.load(file_path, allow_pickle=True)
    chs = saved['ch_names'].tolist()
    
    if all(c in saved for c in sub_4cell.keys()):
        for c in sub_4cell.keys():
            aligned = align_channels(saved[c], chs, canonical_ch_names)
            sub_4cell[c].append(aligned)

n_4cell = len(sub_4cell['low_recalled'])
print(f"  ✓ 4-cell AllItems: loaded {n_4cell} subjects")
for c in sub_4cell.keys():
    sub_4cell[c] = np.array(sub_4cell[c])

# ══════════════════════════════════════════════════════════════════
# 3. HELPER PLOTTING FUNCTION
# ══════════════════════════════════════════════════════════════════
time_mask = (times >= TOPO_TIME[0]) & (times <= TOPO_TIME[1])

def plot_single_topomap(data_vector, title, filename, vlim, cmap='RdBu_r'):
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(
        data_vector, info, axes=ax, show=False,
        cmap=cmap, sphere='eeglab', vlim=vlim
    )
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
    cbar.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
    ax.set_title(title, fontsize=14, fontname='Times New Roman', pad=12)
    plt.tight_layout()
    out_file = os.path.join(figures_path, filename)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_multi_band_panel(band_data_dict, title_prefix, filename_prefix, vlim, cmap='RdBu_r'):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    im = None
    for idx, (band_name, data_vec) in enumerate(band_data_dict.items()):
        ax = axes[idx]
        im, _ = mne.viz.plot_topomap(
            data_vec, info, axes=ax, show=False,
            cmap=cmap, sphere='eeglab', vlim=vlim
        )
        f_low, f_high = BANDS[band_name]
        ax.set_title(f"{band_name} ({f_low:.0f}–{f_high:.0f} Hz)", fontsize=14, fontname='Times New Roman', pad=10)
    
    fig.subplots_adjust(bottom=0.25)
    cbar_ax = fig.add_axes([0.35, 0.1, 0.3, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Power (dB)', fontname='Times New Roman', fontsize=12)
    fig.suptitle(f"{title_prefix} (0.0–3.0 s)", fontsize=16, fontname='Times New Roman', y=0.98)
    out_file = os.path.join(figures_path, f"{filename_prefix}_all_bands.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved 4-band panel: {filename_prefix}_all_bands.png")

# ══════════════════════════════════════════════════════════════════
# 4. GENERATE TOPOPLOTS FOR EACH CONTRAST AND BAND
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GENERATING TOPOPLOTS (0.0 to 3.0 s)")
print("=" * 70)

# --- A. Main Effect of Memory (All Items) ---
if 'AllItems' in data_main_effects:
    rec_subs = data_main_effects['AllItems']['recalled']
    not_subs = data_main_effects['AllItems']['not_recalled']
    diff_subs = rec_subs - not_subs
    n_s = data_main_effects['AllItems']['n_subs']
    
    ga_rec  = np.nanmean(rec_subs, axis=0)
    ga_not  = np.nanmean(not_subs, axis=0)
    ga_diff = np.nanmean(diff_subs, axis=0)
    
    diff_band_vectors = {}
    
    print("\n[Main Effect Memory - All Items]")
    for band_name, (f_low, f_high) in BANDS.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        
        vec_rec  = np.nanmean(ga_rec[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_not  = np.nanmean(ga_not[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_diff = np.nanmean(ga_diff[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        diff_band_vectors[band_name] = vec_diff
        
        # 1. Condition topoplots
        plot_single_topomap(vec_rec, f"Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[All Items, 0-3s, N={n_s}]",
                            f"00_topo_AllItems_recalled_{band_name}.png", Z_LIMITS_COND)
        plot_single_topomap(vec_not, f"Not Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[All Items, 0-3s, N={n_s}]",
                            f"00_topo_AllItems_not_recalled_{band_name}.png", Z_LIMITS_COND)
        # 2. Difference topoplot
        plot_single_topomap(vec_diff, f"Difference: Recalled - Not Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[Main Effect All Items, 0-3s, N={n_s}]",
                            f"01_topo_diff_MainEffect_AllItems_{band_name}.png", Z_LIMITS_DIFF)
    
    # 4-band combined panel for difference
    plot_multi_band_panel(diff_band_vectors, f"Main Effect Memory (Recalled - Not Recalled) - All Items [N={n_s}]",
                          "01_topo_diff_MainEffect_AllItems", Z_LIMITS_DIFF)

# --- B. Main Effect of Memory (Controlled Only) ---
if 'ControlledOnly' in data_main_effects:
    rec_subs = data_main_effects['ControlledOnly']['recalled']
    not_subs = data_main_effects['ControlledOnly']['not_recalled']
    diff_subs = rec_subs - not_subs
    n_s = data_main_effects['ControlledOnly']['n_subs']
    
    ga_rec  = np.nanmean(rec_subs, axis=0)
    ga_not  = np.nanmean(not_subs, axis=0)
    ga_diff = np.nanmean(diff_subs, axis=0)
    
    diff_band_vectors = {}
    
    print("\n[Main Effect Memory - Controlled Only]")
    for band_name, (f_low, f_high) in BANDS.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        
        vec_rec  = np.nanmean(ga_rec[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_not  = np.nanmean(ga_not[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_diff = np.nanmean(ga_diff[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        diff_band_vectors[band_name] = vec_diff
        
        # Condition topoplots
        plot_single_topomap(vec_rec, f"Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[Controlled Only, 0-3s, N={n_s}]",
                            f"00_topo_ControlledOnly_recalled_{band_name}.png", Z_LIMITS_COND)
        plot_single_topomap(vec_not, f"Not Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[Controlled Only, 0-3s, N={n_s}]",
                            f"00_topo_ControlledOnly_not_recalled_{band_name}.png", Z_LIMITS_COND)
        # Difference topoplot
        plot_single_topomap(vec_diff, f"Difference: Recalled - Not Recalled ({band_name}, {f_low:.0f}-{f_high:.0f} Hz)\n[Main Effect Controlled Only, 0-3s, N={n_s}]",
                            f"01_topo_diff_MainEffect_ControlledOnly_{band_name}.png", Z_LIMITS_DIFF)
    
    plot_multi_band_panel(diff_band_vectors, f"Main Effect Memory (Recalled - Not Recalled) - Controlled Only [N={n_s}]",
                          "01_topo_diff_MainEffect_ControlledOnly", Z_LIMITS_DIFF)

# --- C. Simple Effects & Interaction (4-cell AllItems) ---
if n_4cell >= 2:
    low_diff_subs  = sub_4cell['low_recalled'] - sub_4cell['low_not_recalled']
    high_diff_subs = sub_4cell['high_recalled'] - sub_4cell['high_not_recalled']
    inter_subs     = low_diff_subs - high_diff_subs
    
    ga_low_diff  = np.nanmean(low_diff_subs, axis=0)
    ga_high_diff = np.nanmean(high_diff_subs, axis=0)
    ga_inter     = np.nanmean(inter_subs, axis=0)
    
    low_band_vectors = {}
    high_band_vectors = {}
    inter_band_vectors = {}
    
    print("\n[Simple Effects & Interaction - 4-cell AllItems]")
    for band_name, (f_low, f_high) in BANDS.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        
        vec_low   = np.nanmean(ga_low_diff[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_high  = np.nanmean(ga_high_diff[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        vec_inter = np.nanmean(ga_inter[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
        
        low_band_vectors[band_name] = vec_low
        high_band_vectors[band_name] = vec_high
        inter_band_vectors[band_name] = vec_inter
        
        plot_single_topomap(vec_low, f"Simple Effect Low Control: Recalled - Not Recalled ({band_name})\n[0-3s, N={n_4cell}]",
                            f"01_topo_diff_SimpleEffect_LowControl_{band_name}.png", Z_LIMITS_DIFF)
        plot_single_topomap(vec_high, f"Simple Effect High Control: Recalled - Not Recalled ({band_name})\n[0-3s, N={n_4cell}]",
                            f"01_topo_diff_SimpleEffect_HighControl_{band_name}.png", Z_LIMITS_DIFF)
        plot_single_topomap(vec_inter, f"Interaction: Low Diff - High Diff ({band_name})\n[0-3s, N={n_4cell}]",
                            f"01_topo_diff_Interaction_Memory_x_Control_{band_name}.png", Z_LIMITS_DIFF)
                            
    plot_multi_band_panel(low_band_vectors, f"Simple Effect: Memory within Low Control [N={n_4cell}]",
                          "01_topo_diff_SimpleEffect_LowControl", Z_LIMITS_DIFF)
    plot_multi_band_panel(high_band_vectors, f"Simple Effect: Memory within High Control [N={n_4cell}]",
                          "01_topo_diff_SimpleEffect_HighControl", Z_LIMITS_DIFF)
    plot_multi_band_panel(inter_band_vectors, f"Interaction: Memory x Control [N={n_4cell}]",
                          "01_topo_diff_Interaction_Memory_x_Control", Z_LIMITS_DIFF)

print("\n" + "=" * 70)
print(f"✓ All SME topoplots generated and saved to:\n  {figures_path}")
print("=" * 70)
