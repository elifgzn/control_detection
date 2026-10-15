import mne
import os
import sys

eeg_path = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked"
info = mne.read_epochs(os.path.join(eeg_path, "CDmem_0004-epo.fif"), preload=False, verbose=False).info
ch_adjacency, ch_adj_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
print("ch_adj_names length:", len(ch_adj_names))
print("ch_adjacency shape:", ch_adjacency.shape)
