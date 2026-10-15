import os
import numpy as np

input_path = r'H:\PHD\control_detection\main_data\eeg\eeg4_TFR_stimlocked'
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

for sub in plist:
    file = os.path.join(input_path, f'CDmem_{sub:04d}_TFR_ConditionAverages.npz')
    if os.path.exists(file):
        data = np.load(file)
        print(f"Sub {sub}: {data['low_recalled'].shape}")
