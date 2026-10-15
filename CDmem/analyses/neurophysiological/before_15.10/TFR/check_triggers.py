import mne
import numpy as np

# Load one participant's raw data
raw = mne.io.read_raw_brainvision(r'H:\PHD\control_detection\main_data\eeg\eeg1_raweeg\CDmem_0015.vhdr', preload=False, verbose=False)
events, event_id = mne.events_from_annotations(raw, verbose=False)

# Inverse event dict
inv_event_id = {v: k for k, v in event_id.items()}

# Print first 50 events in test phase
# S 11 or S 13 marks trial start in test phase
test_start_idx = -1
for i, ev in enumerate(events):
    code = inv_event_id[ev[2]]
    if code in ['Stimulus/S 11', 'Stimulus/S 13']:
        test_start_idx = i
        break

if test_start_idx != -1:
    print('First few trials in test phase:')
    for i in range(test_start_idx, test_start_idx + 40):
        if i < len(events):
            time_s = events[i, 0] / raw.info['sfreq']
            code = inv_event_id[events[i, 2]]
            print(f'Time: {time_s:.2f}s - Trigger: {code}')
