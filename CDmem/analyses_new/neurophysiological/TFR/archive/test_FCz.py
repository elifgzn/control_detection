import mne
import numpy as np

bvef_path = r"H:\PHD\control_detection\CDmem\analyses\neurophysiological\CACS-64_REF_new.bvef"
montage = mne.channels.read_custom_montage(bvef_path)
montage.rename_channels({'REF': 'FCz'})

print("FCz pos in montage:", montage.get_positions()['ch_pos']['FCz'])

eeg_path = r'H:\PHD\control_detection\main_data\eeg\eeg3_clean_stimlocked'
info = mne.read_epochs(eeg_path + r'\CDmem_0004-epo.fif', preload=False, verbose=False).info

for ch in info['chs']:
    if ch['ch_name'] == 'FCz':
        print("FCz pos in info:", ch['loc'][:3])
