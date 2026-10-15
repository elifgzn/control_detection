"""
6_TFR_topo_HL.py
================
Whole-Brain Topoplots for Main Effect of Control (High vs Low)
--------------------------------------------------------------

PURPOSE:
    Computes and plots grand-average whole-brain EEG topoplots during the full
    3-second stimulus presentation period (0.0 to 3.0 s) for canonical frequency
    bands:
      - Theta (4 - 8 Hz)
      - Alpha (8 - 12 Hz)
      - Beta (13 - 30 Hz)
      - Gamma (30 - 60 Hz)
      - (Also includes the significant cluster band: 28 - 40 Hz)

    Contrasts & Conditions plotted:
      - High Control (all trials)
      - Low Control (all trials)
      - Difference: High Control - Low Control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
input_path   = r"H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked_HL_MF"
eeg_path     = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
figures_path = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures_stimlocked_topo_HL"
os.makedirs(figures_path, exist_ok=True)

plist = sorted(set(range(1, 51)) - {1, 5, 28, 2, 3, 11, 24, 26, 43, 45, 46, 47})

# Analysis Window
TOPO_TIME = (0.0, 3.0)  # Full 3-second active tracking period

# Canonical Frequency Bands (+ Cluster band)
BANDS = {
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 12.0),
    'Beta':  (13.0, 30.0),
    'Gamma': (30.0, 60.0),
    'HighBeta_Gamma_Cluster': (28.0, 40.0)
}

# Colorbar limits (dB)
Z_LIMITS_COND = (-1.0, 1.0)  # Condition power limits
Z_LIMITS_DIFF = (-0.8, 0.8)  # Difference limits

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
# 2. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading Main Effect of Control data across participants...")
sub_high_list = []
sub_low_list  = []
times, freqs = None, None

for sub in plist:
    sub_id = f"{sub:04d}"
    file_path = os.path.join(input_path, f"CDmem_{sub_id}_TFR_MainEffect_Control.npz")
    if not os.path.exists(file_path):
        continue
        
    saved = np.load(file_path, allow_pickle=True)
    if times is None:
        times = saved['times']
        freqs = saved['freqs']
        
    chs = saved['ch_names'].tolist()
    if 'high_control' in saved and 'low_control' in saved:
        high_aligned = align_channels(saved['high_control'], chs, canonical_ch_names)
        low_aligned  = align_channels(saved['low_control'], chs, canonical_ch_names)
        sub_high_list.append(high_aligned)
        sub_low_list.append(low_aligned)

n_subs = len(sub_high_list)
print(f"  ✓ Loaded {n_subs} complete subjects")

if n_subs < 2:
    print("Not enough subjects to plot topomaps. Exiting.")
    sys.exit(0)

sub_high_arr = np.array(sub_high_list)
sub_low_arr  = np.array(sub_low_list)
sub_diff_arr = sub_high_arr - sub_low_arr

# Grand averages across subjects
ga_high = np.nanmean(sub_high_arr, axis=0)
ga_low  = np.nanmean(sub_low_arr, axis=0)
ga_diff = np.nanmean(sub_diff_arr, axis=0)

# ══════════════════════════════════════════════════════════════════
# 3. HELPER PLOTTING FUNCTIONS
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
    canonical_keys = ['Theta', 'Alpha', 'Beta', 'Gamma']
    im = None
    for idx, band_name in enumerate(canonical_keys):
        ax = axes[idx]
        data_vec = band_data_dict[band_name]
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
    fig.suptitle(f"{title_prefix} (0.0–3.0 s, N={n_subs})", fontsize=16, fontname='Times New Roman', y=0.98)
    out_file = os.path.join(figures_path, f"{filename_prefix}_all_bands.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved 4-band panel: {filename_prefix}_all_bands.png")

# ══════════════════════════════════════════════════════════════════
# 4. GENERATE TOPOPLOTS FOR EACH BAND
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  GENERATING CONTROL TOPOPLOTS (0.0 to 3.0 s, N={n_subs})")
print("=" * 70)

diff_band_vectors = {}
high_band_vectors = {}
low_band_vectors  = {}

for band_name, (f_low, f_high) in BANDS.items():
    freq_mask = (freqs >= f_low) & (freqs <= f_high)
    
    vec_high = np.nanmean(ga_high[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
    vec_low  = np.nanmean(ga_low[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
    vec_diff = np.nanmean(ga_diff[:, freq_mask, :][:, :, time_mask], axis=(1, 2))
    
    diff_band_vectors[band_name] = vec_diff
    high_band_vectors[band_name] = vec_high
    low_band_vectors[band_name]  = vec_low
    
    # 1. High Control
    plot_single_topomap(vec_high, f"High Control ({band_name}, {f_low:.0f}–{f_high:.0f} Hz)\n[0.0–3.0 s, N={n_subs}]",
                        f"00_topo_High_Control_{band_name}.png", Z_LIMITS_COND)
                        
    # 2. Low Control
    plot_single_topomap(vec_low, f"Low Control ({band_name}, {f_low:.0f}–{f_high:.0f} Hz)\n[0.0–3.0 s, N={n_subs}]",
                        f"00_topo_Low_Control_{band_name}.png", Z_LIMITS_COND)
                        
    # 3. Difference (High - Low)
    plot_single_topomap(vec_diff, f"Difference: High - Low Control ({band_name}, {f_low:.0f}–{f_high:.0f} Hz)\n[0.0–3.0 s, N={n_subs}]",
                        f"01_topo_diff_High_minus_Low_{band_name}.png", Z_LIMITS_DIFF)

# 4. Multi-band combined panels
plot_multi_band_panel(high_band_vectors, "High Control Grand Average Power", "00_topo_High_Control", Z_LIMITS_COND)
plot_multi_band_panel(low_band_vectors,  "Low Control Grand Average Power",  "00_topo_Low_Control",  Z_LIMITS_COND)
plot_multi_band_panel(diff_band_vectors, "Difference: High Control - Low Control", "01_topo_diff_High_minus_Low", Z_LIMITS_DIFF)

print("\n" + "=" * 70)
print(f"✓ All Control topoplots generated and saved to:\n  {figures_path}")
print("=" * 70)
