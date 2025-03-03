import mne
from tqdm import tqdm


# This function cleans raw EEG brain signals to prepare them for analysis

def preprocess_eeg_data(X, sfreq=160, bandpass=(0.5, 45), notch=60):
    
    X_processed = []

    # Process each recording individually (with a progress bar)
    for x in tqdm(X, desc="Preprocessing"):
    

        x_copy = x.copy()

        # Channel names, metadata
        ch_names = [f'EEG{i:03d}' for i in range(x_copy.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create MNE RawArray
        # (MNE requires a specific structure for EEG data)
        raw = mne.io.RawArray(x_copy, info)
        
        # Apply filters (bandpass & notch)
        raw.filter(bandpass[0], bandpass[1], method='iir', 
                  iir_params=dict(order=4, ftype='butter'), 
                  verbose=False)
        raw.notch_filter(notch, method='iir', 
                        iir_params=dict(order=4, ftype='butter'),
                        verbose=False)
        
        # Get cleaned data back as np array
        data = raw.get_data()
        X_processed.append(data)
    
    return X_processed
