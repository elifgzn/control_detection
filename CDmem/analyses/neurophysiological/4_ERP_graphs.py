import os
import sys
import numpy as np
import pandas as pd
import mne

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21,22]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
# FieldTrip: dfolder = 'D:/MCRL DATA/eeg4_ERPSummaries';
dfolder = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries"

# Path for saving figures
# FieldTrip: save_to = 'D:/MCRL DATA/eeg5_figures'; dpi = '-r600';
save_to = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Electrode and time window selection for P500 (Wen et al., 2017)
# ──────────────────────────────────────────────────────────────

# electrodes_select = ['FCz', 'Cz', 'CPz', 'Pz']  # (Wen et al., 2017)
electrodes_select = ['Fz', 'FCz', 'FC1', 'FC2']  # (following Giersiepen et al., 2024, 2025)

# Time window for P500 analysis and topoplots (Wen et al., 2017)
time_select = [0.45, 0.65]

# ──────────────────────────────────────────────────────────────
# Condition layout (matches ERP_calculation.py output)
# ──────────────────────────────────────────────────────────────
num_cond = 2
cond_names = ['high_control', 'low_control']

# ──────────────────────────────────────────────────────────────
# Load data for all participants + build grand-average structures
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   alleeg{p,cond} = participant.eegdat{cond};
#   behavSummary   = [behavSummary participant.summary];
#
# In MNE, mne.read_evokeds() loads the FIF saved by ERP_calculation.py.
# Each file contains all 4 Evoked objects (one per condition), identified
# by their .comment attribute (set to the condition label in ERP_calculation.py).

# alleeg[p][cond_label] → Evoked for participant p, condition cond_label
alleeg       = {}   # FieldTrip: alleeg{p, cond}
all_summaries = []   # FieldTrip: behavSummary

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-ave.fif")
    csv_file  = os.path.join(dfolder, f"CDmem_{sub_id}-erp-summary.csv")

    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    # FieldTrip: participant = load([dfolder '/MCRL_' num2str(pnum)]);
    evoked_list = mne.read_evokeds(erp_file, verbose=False)

    # Index by condition label (stored in evoked.comment during ERP_calculation.py)
    alleeg[p_idx] = {ev.comment: ev for ev in evoked_list}

    # Load trial-count summary
    # FieldTrip: behavSummary = [behavSummary participant.summary];
    if os.path.exists(csv_file):
        summary_row = pd.read_csv(csv_file).iloc[0].to_dict()
        all_summaries.append(summary_row)

    # ── (Difference waves removed based on user request) ───────────────────────

# Print behavioral summary table across participants
# FieldTrip: struct2table(behavSummary)
if all_summaries:
    print("\nBehavioral summary (trial counts per condition):")
    print(pd.DataFrame(all_summaries).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# Grand averages across participants
# ──────────────────────────────────────────────────────────────

# ── Non-difference wave grand averages ───────────────────────
# FieldTrip:
#   cfg = [];
#   for cond=1:num_cond
#       GA_dat{cond} = ft_timelockgrandaverage(cfg, alleeg{:,cond});
#   end
#
# MNE: mne.grand_average() averages across a list of Evoked objects.
# It normalizes by number of trials under the hood (equivalent to FieldTrip's
# ft_timelockgrandaverage with default cfg).

print("\nComputing grand averages...")
GA_dat = {}   # FieldTrip: GA_dat{cond}

for cond_label in cond_names:
    # Collect this condition's Evoked across all loaded participants
    evokeds_this_cond = [alleeg[p][cond_label]
                         for p in alleeg if cond_label in alleeg[p]]
    if evokeds_this_cond:
        GA_dat[cond_label] = mne.grand_average(evokeds_this_cond)
        GA_dat[cond_label].comment = cond_label
        print(f"  GA [{cond_label}]: averaged across {len(evokeds_this_cond)} participant(s)")

# ── (Difference wave grand averages removed) ───────────────────────────

print("\nGrand averages ready.")
print(f"  Electrodes selected : {electrodes_select}")
print(f"  Time window         : {time_select[0]:.2f} - {time_select[1]:.2f} s")

# ──────────────────────────────────────────────────────────────
# Average activity across electrodes per participant / condition
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   for p=1:length(plist)
#     for cond=1:num_cond
#       avg_subj = ft_timelockgrandaverage(cfg, alleeg{p,cond});  % select electrodes via cfg.channel
#       pMean    = mean(avg_subj.avg, 1);                         % average over electrode dimension
#       pMeanList(p,cond,:) = squeeze(pMean)';                    % [n_participants, n_conds, n_timepoints]
#     end
#   end
#
# In MNE, evoked.data has shape [n_channels, n_timepoints] in Volts.
# We pick() the selected electrodes, then average across channels to get
# one time series per participant × condition.  Multiply by 1e6 → µV.

import matplotlib
matplotlib.use('TkAgg')   # interactive window; change to 'Agg' for headless/batch saving only
import matplotlib.pyplot as plt

loaded_plist = list(alleeg.keys())           # participant indices that were successfully loaded
n_participants = len(loaded_plist)

# Get shared time axis (same for all participants / conditions)
# FieldTrip: x = avg_subj.time;
times = alleeg[loaded_plist[0]][cond_names[0]].times   # in seconds

# Verify that all requested electrodes actually exist in the data
available_ch = alleeg[loaded_plist[0]][cond_names[0]].ch_names
picked_channels = [ch for ch in electrodes_select if ch in available_ch]
missing = [ch for ch in electrodes_select if ch not in available_ch]
if missing:
    print(f"  WARNING: electrode(s) not found in data and will be skipped: {missing}")
print(f"  Using electrodes: {picked_channels}")

n_timepoints = len(times)

# ── Non-difference wave pMeanList ──────────────────────────────
pMeanList = np.zeros((n_participants, num_cond, n_timepoints))   # [P, C, T]

for p_idx, p_key in enumerate(loaded_plist):
    for cond_idx, cond_label in enumerate(cond_names):
        # Pick selected electrodes and convert to µV
        evoked = alleeg[p_key][cond_label].copy().pick(picked_channels)
        data_uv = evoked.data * 1e6                        # [n_channels, n_timepoints]
        pMeanList[p_idx, cond_idx, :] = np.mean(data_uv, axis=0)   # average over electrodes

# ── Compute grandMean / subjMean for error-bar correction ──────
# FieldTrip:
#   if length(plist) == 1
#     grandMean = squeeze(mean(pMeanList, 2));         % mean over conditions dimension
#   else
#     subjMean = squeeze(mean(pMeanList, 2));           % [P, T] -- mean over conditions
#     grandMean = mean(subjMean, 1);                   % [T]    -- mean over participants
#   end
#
# grandMean is used only in the Cousineau-Morey within-subject error correction
# (see plotting loops below). For single-participant plots there are no error bars.

if n_participants == 1:
    grandMean = np.squeeze(np.mean(pMeanList, axis=1))   # [T]
    subjMean  = grandMean                                 # unused for 1 participant
else:
    subjMean  = np.squeeze(np.mean(pMeanList, axis=1))   # [P, T] -- mean over conditions
    grandMean = np.mean(subjMean, axis=0)                 # [T]    -- mean over participants

# ──────────────────────────────────────────────────────────────
# Plot settings
# ──────────────────────────────────────────────────────────────

ylimits  = [-3, 3]   # µV y-axis range

# Line colors per condition [blue for high, purple for low]
colors      = [[0, 0.44, 0.69], [0.8, 0.47, 0.65]]
linestyles  = ['-', '-']   
linecol_colors = ['blue', 'magenta']

# Figure dimensions matching FieldTrip's PaperPosition [0 0 30 20] (cm)
fig_w_cm, fig_h_cm = 30, 20
fig_w_in = fig_w_cm / 2.54
fig_h_in = fig_h_cm / 2.54
font_size = 20   # FieldTrip: fontsz = 20

plt.close('all')

# ──────────────────────────────────────────────────────────────
# MAIN FIGURE
# ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(fig_w_in, fig_h_in))

H1 = []

# ── Non-difference wave: 2 conditions ───────────────────
labels = ['High Control', 'Low Control']
for cond_idx, cond_label in enumerate(cond_names):
    y = pMeanList[:, cond_idx, :]   # [P, T]

    if n_participants == 1:
        h, = ax1.plot(times, y[0, :],
                      linestyle='-', color=linecol_colors[cond_idx])
    else:
        # Cousineau-Morey correction (factor for 2 conditions is 2/1 = 2.0)
        yCorrect = y - subjMean
        yCorrect = (yCorrect + grandMean) * (2.0)
        errbar   = np.std(yCorrect, axis=0, ddof=0) / np.sqrt(n_participants)

        h, = ax1.plot(times, np.mean(y, axis=0),
                      linestyle=linestyles[cond_idx],
                      color=colors[cond_idx])

    H1.append(h)
save_name = '00_lineplot_main_effects.svg'


# ── Formatting ────────────────────────────────────────────────
# FieldTrip:
#   legend(H, labels, 'Location', 'best');
#   set(gca, 'YDir','reverse');   ylim(ylimits)
#   line([x(1) x(end)], [0 0], 'Color','black','LineStyle','--','LineWidth',0.5);
#   line([0 0], ylimits, 'Color','black','LineStyle','--','LineWidth',0.5);
#   xlabel('Time (s)');  xticks(0:0.1:max(x));  ylabel('microVolt')
#   set(gca,'FontSize',20,'FontName','Times New Roman','LineWidth',1)

ax1.legend(H1, labels, loc='best', fontsize=font_size)
ax1.set_ylim(ylimits)
ax1.invert_yaxis()           # EEG convention: negative up
ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.5)

ax1.set_xlabel('Time (s)', fontsize=font_size)
ax1.set_ylabel('µV', fontsize=font_size)
ax1.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
ax1.tick_params(labelsize=font_size, width=1)
for spine in ax1.spines.values():
    spine.set_linewidth(1)

# FieldTrip: set(gca,'FontName','Times New Roman')
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception:
    pass   # fall back to matplotlib default if font not installed

# FieldTrip: set(gcf,'Color',[1 1 1])   → white background
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Title hidden (FieldTrip: set(title_handle,'Visible','off'))
ax1.set_title('')

plt.tight_layout()

# ── Save figure ───────────────────────────────────────────────
# FieldTrip: print(gcf, '-dsvg', [save_to '00_lineplot_...'], dpi)
# dpi=600 matches FieldTrip's -r600; SVG is vector so DPI mainly affects rasterised elements.
save_path = os.path.join(save_to, save_name)
fig1.savefig(save_path, format='svg', dpi=600, bbox_inches='tight',
             facecolor='white')
print(f"\n  Figure saved: {save_path}")

# ──────────────────────────────────────────────────────────────
# Topographical Plots
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   cfg.highlightchannel = electrodes_select;
#   cfg.layout    = 'layout65.mat';
#   cfg.colorbar  = 'yes';
#   cfg.xlim      = time_select;
#   cfg.colormap  = flipud(brewermap(1000,'RdBu'));
#   cfg.zlim      = [-1.5 1.5];
#   ft_topoplotER(cfg, GA_dat{num_plot});
#
# In MNE, we use the `mask` parameter of plot_topomap() to highlight
# the selected electrodes (equivalent to cfg.highlightchannel).
# MNE handles the 2D projection internally so positions always match.

# Colormap: FieldTrip uses flipud(brewermap(1000,'RdBu')) → reversed RdBu.
# matplotlib's 'RdBu_r' is the reversed version (blue for negative, red for positive).
topo_cmap = plt.cm.RdBu_r

# Color limits matching FieldTrip: cfg.zlim = [-1.5 1.5]   (µV)
# Adjust these if needed for visibility.
topo_vlim = (-1.5, 1.5)   # in µV — will be converted to Volts for MNE
# topo_vlim = (-3, 3)   # in µV — will be converted to Volts for MNE

# Figure size: FieldTrip PaperPosition [0 0 8 8] (cm) → 8×8 cm per topo
topo_fig_in = 8 / 2.54   # ~3.15 inches

# Time window for averaging: FieldTrip cfg.xlim = time_select
t_min, t_max = time_select

# Build a boolean mask for the highlighted electrodes.
# FieldTrip: cfg.highlightchannel = electrodes_select; cfg.highlightsymbol = '*'
# MNE: mask = boolean array [n_channels], True = highlight with mask_params style.
def make_highlight_mask(evoked, highlight_names):
    """Return a boolean array (n_channels,) — True for channels to highlight."""
    mask = np.array([ch in highlight_names for ch in evoked.ch_names])
    return mask

# Style for highlighted channels (FieldTrip: cfg.highlightsymbol = '*', cfg.highlightsize = 10)
mask_params = dict(marker='*', markerfacecolor='black', markeredgecolor='black',
                   markersize=10, zorder=10)

# ── Non-difference wave: one topo per condition ──────────
topo_labels = ['High Control', 'Low Control']
topo_prefix = '00_topo_main_effects'

for cond_idx, cond_label in enumerate(cond_names):
    evoked_topo = GA_dat[cond_label].copy()
    evoked_topo.crop(tmin=t_min, tmax=t_max)
    topo_data = evoked_topo.data.mean(axis=1)   # [n_channels] in Volts

    fig_topo, ax_topo = plt.subplots(figsize=(topo_fig_in, topo_fig_in))

    # FieldTrip: ft_topoplotER(cfg, GA_dat{num_plot})
    highlight_mask = make_highlight_mask(evoked_topo, picked_channels)
    mne.viz.plot_topomap(
        topo_data, evoked_topo.info,
        axes=ax_topo,
        cmap=topo_cmap,
        vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),   # µV → V
        mask=highlight_mask,
        mask_params=mask_params,
        show=False,
        contours=6,
    )

    ax_topo.set_title(topo_labels[cond_idx], fontsize=12)
    fig_topo.patch.set_facecolor('white')
    fig_topo.tight_layout()

    # Save as SVG and TIFF
    # FieldTrip: print(gcf, '-dsvg', [...], dpi); print(gcf, '-dtiff', [...], dpi)
    for fmt in ['svg', 'tiff']:
        topo_save = os.path.join(save_to, f"{topo_prefix}_{cond_label}.{fmt}")
        fig_topo.savefig(topo_save, format=fmt, dpi=600,
                         bbox_inches='tight', facecolor='white')
    print(f"  Topo saved: {topo_prefix}_{cond_label} (.svg + .tiff)")

plt.show()   # display main-effect figures interactively; close windows to continue

# # ══════════════════════════════════════════════════════════════
# # DETECTION-SPLIT ERP PLOTS
# # ══════════════════════════════════════════════════════════════
# # 4 conditions: high_control × detected/nondetected,
# #               low_control  × detected/nondetected
# # Data comes from the separate '-erp-detection-ave.fif' files
# # saved by 3_ERP_calculation.py.
# # ══════════════════════════════════════════════════════════════

# print("\n" + "="*60)
# print("  DETECTION-SPLIT ERP PLOTS")
# print("="*60)

# det_cond_names = [
#     'high_control_detected',
#     'high_control_nondetected',
#     'low_control_detected',
#     'low_control_nondetected'
# ]
# det_num_cond = len(det_cond_names)

# # ── Load detection-split data ─────────────────────────────────
# alleeg_det       = {}
# all_summaries_det = []

# for p_idx, pnum in enumerate(plist):
#     sub_id   = f"{pnum:04d}"
#     erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-detection-ave.fif")
#     csv_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-detection-summary.csv")

#     if not os.path.exists(erp_file):
#         print(f"  Detection ERP not found, skipping participant {pnum}")
#         continue

#     print(f"  Loading detection ERPs for participant {pnum}")
#     evoked_list = mne.read_evokeds(erp_file, verbose=False)
#     alleeg_det[p_idx] = {ev.comment: ev for ev in evoked_list}

#     if os.path.exists(csv_file):
#         all_summaries_det.append(pd.read_csv(csv_file).iloc[0].to_dict())

# if all_summaries_det:
#     print("\n  Detection-split trial counts:")
#     print(pd.DataFrame(all_summaries_det).to_string(index=False))

# # ── Grand averages ────────────────────────────────────────────
# print("\n  Computing detection-split grand averages...")
# GA_det = {}

# for cond_label in det_cond_names:
#     evokeds_this = [alleeg_det[p][cond_label]
#                     for p in alleeg_det if cond_label in alleeg_det[p]]
#     if evokeds_this:
#         GA_det[cond_label] = mne.grand_average(evokeds_this)
#         GA_det[cond_label].comment = cond_label
#         print(f"    GA [{cond_label}]: {len(evokeds_this)} participant(s)")

# # ── pMeanList for detection-split ─────────────────────────────
# det_loaded = list(alleeg_det.keys())
# det_n_part = len(det_loaded)

# if det_n_part == 0:
#     print("  No detection-split data found — skipping plots.")
# else:
#     # Find a reference Evoked to get times and channel names robustly
#     ref_evoked = None
#     for p_key in det_loaded:
#         for c_lab in det_cond_names:
#             if c_lab in alleeg_det[p_key]:
#                 ref_evoked = alleeg_det[p_key][c_lab]
#                 break
#         if ref_evoked is not None:
#             break

#     if ref_evoked is None:
#         print("  No valid detection-split data found (all conditions missing) — skipping plots.")
#     else:
#         det_times = ref_evoked.times
#         det_avail = ref_evoked.ch_names
#         det_picked = [ch for ch in electrodes_select if ch in det_avail]

#         det_pMean = np.full((det_n_part, det_num_cond, len(det_times)), np.nan)

#         for p_i, p_key in enumerate(det_loaded):
#             for c_i, c_lab in enumerate(det_cond_names):
#                 if c_lab in alleeg_det[p_key]:
#                     ev = alleeg_det[p_key][c_lab].copy().pick(det_picked)
#                     det_pMean[p_i, c_i, :] = np.mean(ev.data * 1e6, axis=0)
#                 else:
#                     pnum = plist[p_key]
#                     print(f"    WARNING: Participant {pnum} missing condition {c_lab} (0 trials). Filled with NaN.")

#         if det_n_part == 1:
#             det_grandMean = np.squeeze(np.nanmean(det_pMean, axis=1))
#             det_subjMean  = det_grandMean
#         else:
#             det_subjMean  = np.squeeze(np.nanmean(det_pMean, axis=1))   # [P, T]
#             det_grandMean = np.nanmean(det_subjMean, axis=0)             # [T]

#         # ── Line plot: 4 conditions ───────────────────────────────
#         det_colors = [
#             [0.00, 0.44, 0.69],   # high_detected      — blue solid
#             [0.00, 0.44, 0.69],   # high_nondetected    — blue dashed
#             [0.80, 0.47, 0.65],   # low_detected        — magenta solid
#             [0.80, 0.47, 0.65],   # low_nondetected     — magenta dashed
#         ]
#         det_lstyles = ['-', '--', '-', '--']
#         det_labels  = [
#             'High Control – Detected',
#             'High Control – Non-detected',
#             'Low Control – Detected',
#             'Low Control – Non-detected',
#         ]

#         fig_det, ax_det = plt.subplots(figsize=(fig_w_in, fig_h_in))
#         H_det = []

#         for c_i, c_lab in enumerate(det_cond_names):
#             y = det_pMean[:, c_i, :]

#             if det_n_part == 1:
#                 mean_y = np.nanmean(y, axis=0) if not np.all(np.isnan(y)) else np.full_like(det_times, np.nan)
#                 h, = ax_det.plot(det_times, mean_y,
#                                  linestyle=det_lstyles[c_i], color=det_colors[c_i])
#             else:
#                 cm_factor = det_num_cond / (det_num_cond - 1)
#                 yCorr = (y - det_subjMean + det_grandMean) * cm_factor
#                 n_valid = np.sum(~np.isnan(y[:, 0]))
#                 if n_valid > 0:
#                     errbar = np.nanstd(yCorr, axis=0, ddof=0) / np.sqrt(n_valid)
#                     mean_y = np.nanmean(y, axis=0)
#                 else:
#                     errbar = np.zeros_like(det_times)
#                     mean_y = np.full_like(det_times, np.nan)

#                 h, = ax_det.plot(det_times, mean_y,
#                                  linestyle=det_lstyles[c_i], color=det_colors[c_i])
#             H_det.append(h)

#         ax_det.legend(H_det, det_labels, loc='best', fontsize=font_size - 4)
#         ax_det.set_ylim(ylimits)
#         ax_det.invert_yaxis()
#         ax_det.axhline(0, color='black', linestyle='--', linewidth=0.5)
#         ax_det.axvline(0, color='black', linestyle='--', linewidth=0.5)
#         ax_det.set_xlabel('Time (s)', fontsize=font_size)
#         ax_det.set_ylabel('µV', fontsize=font_size)
#         ax_det.set_xticks(np.arange(np.floor(det_times[0] * 10) / 10, det_times[-1] + 0.01, 0.1))
#         ax_det.tick_params(labelsize=font_size, width=1)
#         for spine in ax_det.spines.values():
#             spine.set_linewidth(1)
#         fig_det.patch.set_facecolor('white')
#         ax_det.set_facecolor('white')
#         ax_det.set_title('')
#         plt.tight_layout()

#         det_line_save = os.path.join(save_to, '00_lineplot_detection_split.svg')
#         fig_det.savefig(det_line_save, format='svg', dpi=600,
#                         bbox_inches='tight', facecolor='white')
#         print(f"\n  Detection line plot saved: {det_line_save}")

#         # ── Topoplots: one per condition ──────────────────────────
#         det_topo_prefix = '00_topo_detection_split'

#         for c_i, c_lab in enumerate(det_cond_names):
#             if c_lab not in GA_det:
#                 print(f"    WARNING: Grand average for {c_lab} not available. Skipping topoplot.")
#                 continue
#             ev_topo = GA_det[c_lab].copy()
#             ev_topo.crop(tmin=t_min, tmax=t_max)
#             topo_d = ev_topo.data.mean(axis=1)

#             fig_t, ax_t = plt.subplots(figsize=(topo_fig_in, topo_fig_in))
#             h_mask = make_highlight_mask(ev_topo, det_picked)
#             mne.viz.plot_topomap(
#                 topo_d, ev_topo.info,
#                 axes=ax_t, cmap=topo_cmap,
#                 vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),
#                 mask=h_mask, mask_params=mask_params,
#                 show=False, contours=6,
#             )
#             ax_t.set_title(det_labels[c_i], fontsize=12)
#             fig_t.patch.set_facecolor('white')
#             fig_t.tight_layout()

#             for fmt in ['svg', 'tiff']:
#                 tsave = os.path.join(save_to, f"{det_topo_prefix}_{c_lab}.{fmt}")
#                 fig_t.savefig(tsave, format=fmt, dpi=600,
#                               bbox_inches='tight', facecolor='white')
#             print(f"  Topo saved: {det_topo_prefix}_{c_lab} (.svg + .tiff)")

#         plt.show()   # display all figures interactively; close windows to end script

