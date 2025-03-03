# Functions for loading EEG data from PhysioNet

import os
import mne
import numpy as np
from tqdm import tqdm


# This function loads EEG recordings from the PhysioNet dataset
# Focuses on recordings in which subjects imagined moving their left or right hand
    
def load_physionet_data(base_dir, subjects=range(1, 11), runs=[4, 8, 12]):
    X = [] # Will store EEG recordings
    y = [] # Will store labels (0=left hand, 1=right hand)
    
    for subject in tqdm(subjects, desc="Loading subjects"):
        # PhysioNet dataset uses format like S001, S002, etc.
        subject_id = f"S{subject:03d}"
        subject_dir = os.path.join(base_dir, subject_id)
        
        for run in runs:
            # Runs use format like R04, R08, R12
            run_id = f"R{run:02d}"
            
            # EDF files follow naming convention: S001R04.edf
            edf_file = os.path.join(subject_dir, f"{subject_id}{run_id}.edf")
            
            # Skip if file doesn't exist
            if not os.path.exists(edf_file):
                print(f"File not found: {edf_file}")
                continue
            
            # Load EEG data from EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True)
            
            # Extract events (when subject was cued to imagine movement)
            events, event_id = mne.events_from_annotations(raw)
            
             # Process each event in this recording
            for event_idx in range(len(events)):
                event_type = events[event_idx, 2]
                
                # Skip rest events (T0)
                if event_type == 0:  # Rest
                    continue
                
                # Get start time of the event
                start_time = events[event_idx, 0] / raw.info['sfreq']
                
                # Extract 2s of data after the event
                data, times = raw[:, int(start_time * raw.info['sfreq']):
                               int((start_time + 2) * raw.info['sfreq'])]
                
                # Skip if segment too short (can happen at end of recording)
                if data.shape[1] < int(2 * raw.info['sfreq']):
                    continue
                
                # Store segment and label
                X.append(data)
                
                # Convert event type to label (0: left, 1: right)
                y.append(event_type - 1)
    
    return X, y